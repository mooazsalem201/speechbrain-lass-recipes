"""Frozen CLAP text encoder: turns a caption into the 512-d query vector.

``laion/larger_clap_general`` is a 1:1 export of the LAION checkpoint that
AudioSep uses (``music_speech_audioset_epoch_15_esc_89.98.pt``; verified
cosine 1.0000 on test captions). Only the text branch is loaded. Embeddings
are L2-normalized (the HF head returns unnormalized projections, unlike
``laion_clap``) and cached per caption, so nothing is ever backpropagated
through the encoder and repeated captions cost nothing.

Authors
 * <your name> 2026
"""

import torch
from torch import nn


class ClapTextEncoder(nn.Module):
    """Frozen HF CLAP text branch with L2-normalized, cached outputs.

    Arguments
    ---------
    source : str
        HuggingFace model id or local path.
    max_length : int
        Token truncation length. CLAP was trained with 77 tokens; longer
        queries are out of distribution anyway.

    Example
    -------
    >>> encoder = ClapTextEncoder()  # doctest: +SKIP
    >>> encoder(["a dog barking"]).shape  # doctest: +SKIP
    torch.Size([1, 512])
    """

    def __init__(self, source="laion/larger_clap_general", max_length=77):
        super().__init__()
        from transformers import AutoTokenizer, ClapTextModelWithProjection

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = ClapTextModelWithProjection.from_pretrained(source)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.max_length = max_length
        self.cache = {}

    @torch.no_grad()
    def forward(self, captions):
        """Encode a list of captions into unit-norm vectors.

        Arguments
        ---------
        captions : list of str

        Returns
        -------
        torch.Tensor
            Shape ``(len(captions), 512)``, on the encoder's device.
        """
        device = next(self.model.parameters()).device
        missing = [c for c in captions if c not in self.cache]
        if missing:
            tokens = self.tokenizer(
                missing,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)
            embeddings = self.model(**tokens).text_embeds
            embeddings = nn.functional.normalize(embeddings, dim=-1)
            for caption, embedding in zip(missing, embeddings):
                self.cache[caption] = embedding
        return torch.stack([self.cache[c] for c in captions]).to(device)
