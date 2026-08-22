# speechbrain-lass-recipes

Text-queried (language-queried, **LASS**) sound separation recipes in [SpeechBrain](https://github.com/speechbrain/speechbrain) format. Type *"a dog barking"* — get the dog isolated from the mixture.

**Status: under construction.**

## Planned first recipe (`recipes/LASS/separation/`)

- Frozen CLAP text encoder → 512-d query embedding, FiLM-conditioned ResUNet masker (~26M trainable params), L1 waveform loss — AudioSep-style, at DCASE 2024 Task 9 baseline scale.
- Data: FSD50K + Clotho v2 (direct Zenodo downloads, no YouTube).
- Eval: DCASE 2024 Task 9 validation split — directly comparable to the published leaderboard (baseline: SDR 5.708 at ~1 GPU-day of training).
- Built against the released `speechbrain` pip package so it runs out of the box; layout mirrors SpeechBrain's recipe tree so upstreaming is a copy-paste.

Roadmap after the core recipe: DPRNN-bottleneck variant (AudioSep-DP), caption augmentation + semantic mixing, generative (latent-DiT) variant.
