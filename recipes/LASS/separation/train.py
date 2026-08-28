#!/usr/bin/env python3
"""Train a text-queried sound separator (LASS).

Frozen CLAP text encoder -> FiLM-conditioned ResUNet30 -> T-F mask, trained
with an L1 waveform loss on mixtures built on the fly inside each batch
(AudioSep style). Optional final scoring on the DCASE 2024 Task 9 set.

To run:
    python train.py hparams/resunet_clap_16k.yaml --data_folder /path/to/data

Authors
 * Mooaz Salem 2026
"""

import sys

import torch
import torchaudio
from dcase_eval import evaluate
from hyperpyyaml import load_hyperpyyaml
from prepare_data import prepare_lass

import speechbrain as sb
from speechbrain.dataio.dataio import read_audio, read_audio_info
from speechbrain.utils.distributed import run_on_main


class LASSBrain(sb.Brain):
    """Mix -> encode the caption -> mask -> L1 against the clean target."""

    def compute_forward(self, batch, stage):
        wavs = batch.sig.data.to(self.device)
        mixture, target = self.mix_batch(wavs, stage)
        query = self.hparams.query_encoder(batch.caption).to(self.device)
        return self.modules.model(mixture, query), target

    def mix_batch(self, wavs, stage):
        """Each clip is its own target and the interference of its neighbor.

        The interference is energy-matched to the target (ratio clamped to
        [0.02, 50]), shifted by a random integer gain in +-snr_jitter_db
        during training, and the pair is de-clipped together.
        """
        interference = wavs.roll(1, dims=0)
        energy = wavs.pow(2).mean(dim=1, keepdim=True)
        ratio = energy / interference.pow(2).mean(dim=1, keepdim=True)
        scale = ratio.clamp(0.02, 50).sqrt()
        if stage == sb.Stage.TRAIN:
            jitter = self.hparams.snr_jitter_db
            gain_db = torch.randint(-jitter, jitter + 1, scale.shape)
            scale = scale * 10 ** (gain_db.to(scale.device) / 20)
        mixture = wavs + interference * scale
        peak = mixture.abs().amax(dim=1, keepdim=True)
        declip = torch.where(peak > 1, 0.9 / peak, torch.ones_like(peak))
        return mixture * declip, wavs * declip

    def compute_objectives(self, predictions, batch, stage):
        estimate, target = predictions
        return (estimate - target).abs().mean()

    def on_fit_batch_start(self, batch, should_step):
        """Staircase warm-up: x0.001, x0.01, x0.1, then the full lr."""
        stage = min(self.optimizer_step // self.hparams.warmup_steps, 3)
        for group in self.optimizer.param_groups:
            group["lr"] = self.hparams.lr * 10.0 ** (stage - 3)

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        elif stage == sb.Stage.VALID:
            lr = self.optimizer.param_groups[0]["lr"]
            self.hparams.train_logger.log_stats(
                {"epoch": epoch, "lr": lr},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss},
            )
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_loss}, min_keys=["loss"]
            )

    def separate(self, mixture, caption):
        """numpy mixture + caption -> numpy estimate (for dcase_eval)."""
        wav = torch.from_numpy(mixture)[None].to(self.device)
        query = self.hparams.query_encoder([caption]).to(self.device)
        with torch.no_grad():
            return self.modules.model(wav, query)[0].cpu().numpy()

    def evaluate_dcase(self):
        """Score the best checkpoint on the DCASE 2024 Task 9 set."""
        if self.hparams.dcase_eval_csv is None:
            return
        self.checkpointer.recover_if_possible(min_key="loss")
        self.modules.eval()
        results = evaluate(
            self.separate,
            self.hparams.dcase_eval_csv,
            self.hparams.dcase_audio_dir,
            self.hparams.sample_rate,
        )
        self.hparams.train_logger.log_stats(
            {"DCASE 2024 Task 9": "validation"}, test_stats=results
        )


def dataio_prep(hparams):
    """Datasets yielding a fixed-length mono segment and its caption."""
    sample_rate = hparams["sample_rate"]
    segment = int(hparams["segment_seconds"] * sample_rate)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = read_audio(wav)
        if sig.dim() > 1:
            sig = sig.mean(dim=1)
        sr = read_audio_info(wav).sample_rate
        if sr != sample_rate:
            sig = torchaudio.functional.resample(sig, sr, sample_rate)
        if sig.shape[0] > segment:
            start = torch.randint(0, sig.shape[0] - segment + 1, (1,)).item()
            sig = sig[start : start + segment]
        return torch.nn.functional.pad(sig, (0, segment - sig.shape[0]))

    datasets = {}
    for split in ["train", "valid"]:
        datasets[split] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"{split}_csv"],
            dynamic_items=[audio_pipeline],
            output_keys=["id", "sig", "caption"],
        )
    return datasets


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if not hparams["skip_prep"]:
        run_on_main(
            prepare_lass,
            kwargs={
                "clotho_folder": hparams["clotho_folder"],
                "fsd50k_folder": hparams["fsd50k_folder"],
                "fsd50k_caption_json": hparams["fsd50k_caption_json"],
                "save_folder": hparams["save_folder"],
            },
        )

    datasets = dataio_prep(hparams)
    brain = LASSBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    hparams["query_encoder"].to(brain.device)
    brain.fit(
        brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )
    brain.evaluate_dcase()
