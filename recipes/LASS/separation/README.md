# Text-queried sound separation (LASS)

Type a description — *"a dog barking"* — and the model isolates that sound
from a mixture. AudioSep-style: a frozen CLAP text encoder conditions a
ResUNet30 masker (FiLM), trained with an L1 waveform loss on mixtures built
on the fly inside each batch. ~26M trainable parameters.

## Data

Download once, place under one `data_folder`:

| What | Where | Put at |
|---|---|---|
| Clotho v2 audio + captions (`development`, `validation`) | [Zenodo 4783391](https://zenodo.org/records/4783391) | `data_folder/clotho/` |
| FSD50K dev audio | [Zenodo 4060432](https://zenodo.org/records/4060432) | `data_folder/fsd50k/FSD50K.dev_audio/` |
| FSD50K GPT-4 auto-captions (DCASE 2024 Task 9) | [Zenodo 10887496](https://zenodo.org/records/10887496) | `data_folder/fsd50k_dev_auto_caption.json` |
| DCASE 2024 Task 9 validation set (for scoring) | [Zenodo 10886481](https://zenodo.org/records/10886481) | anywhere |

`prepare_data.py` writes `train.csv` / `valid.csv` (one row per clip-caption
pair) into `save_folder`; no mixtures are stored on disk.

## Train

```bash
pip install -r extra_requirements.txt
python train.py hparams/resunet_clap_16k.yaml --data_folder /path/to/data \
    --dcase_eval_csv /path/to/lass_synthetic_validation.csv \
    --dcase_audio_dir /path/to/lass_validation
```

The DCASE arguments are optional; when given, the best checkpoint is scored
with the challenge's own metric code (`dcase_eval.py`, ported verbatim) so
numbers are comparable to the [Task 9 leaderboard](https://dcase.community/challenge2024/task-language-queried-audio-source-separation-results).

## Results

| Hparams | Data | SDR | SDRi | SI-SDR | Checkpoint |
|---|---|---|---|---|---|
| resunet_clap_16k.yaml | Clotho + FSD50K | *training pending* | | | |
| DCASE 2024 baseline (reference) | Clotho + FSD50K, 200k steps, 1×A100 | 5.708 | 5.673 | 3.862 | [Zenodo 10887460](https://zenodo.org/records/10887460) |

## Notes

- Build-to-code choices (differ from the AudioSep paper prose): FiLM adds a
  bias only, 6/1/6 blocks with channels 32→384, ±10 dB mixing jitter.
- The CLAP text encoder is `laion/larger_clap_general`, verified to be a 1:1
  export of AudioSep's checkpoint (cosine 1.0000); embeddings are
  L2-normalized and cached.
- Audio is resampled to 16 kHz on the fly; pre-resampling the datasets
  speeds up data loading.
- Tests: `python -m pytest tests` from the repository root.
