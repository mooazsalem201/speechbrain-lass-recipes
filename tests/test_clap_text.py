"""Tests for the frozen CLAP text-query encoder (HF transformers, text only).

Downloads ``laion/larger_clap_general`` on first run (verified 2026-08-28 to be
a 1:1 export of AudioSep's checkpoint: cosine 1.0000 on 8 captions).
"""

import os
import sys

import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "recipes", "LASS", "separation"
    ),
)

from clap_text import ClapTextEncoder  # noqa: E402


def test_encodes_captions_to_unit_norm_512d_vectors():
    enc = ClapTextEncoder()
    emb = enc(["a dog barking", "rain falling on a roof"])
    assert emb.shape == (2, 512)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_is_frozen_and_deterministic():
    enc = ClapTextEncoder()
    assert all(not p.requires_grad for p in enc.parameters())
    a = enc(["typing on a keyboard"])
    b = enc(["typing on a keyboard"])
    assert torch.equal(a, b)
    assert not a.requires_grad


def test_repeated_captions_are_served_from_cache():
    enc = ClapTextEncoder()
    enc(["a crowd applauding"])
    calls = {"n": 0}
    original = enc.model.forward

    def counting_forward(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    enc.model.forward = counting_forward
    enc(["a crowd applauding"])  # cached -> no model call
    assert calls["n"] == 0
    enc(["a new caption"])  # not cached -> one model call
    assert calls["n"] == 1
