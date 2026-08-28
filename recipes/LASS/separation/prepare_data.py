"""Data preparation for the LASS recipe: builds CSV manifests.

Training audio = Clotho v2 (development split, 5 human captions per clip)
+ FSD50K dev clips with the DCASE 2024 Task 9 GPT-4 auto-captions.
Validation = Clotho v2 validation split. One manifest row per
(clip, caption) pair; the training script samples rows and mixes them
on the fly, so no mixtures are stored on disk.

Download the audio yourself (see README): Clotho v2 from Zenodo record
4783391, FSD50K from Zenodo record 4060432, the FSD50K auto-captions
from Zenodo record 10887496.

Authors
 * Mooaz Salem 2026
"""

import csv
import json
import os

from speechbrain.dataio.dataio import read_audio_info

CLOTHO_CAPTION_COLUMNS = [f"caption_{i}" for i in range(1, 6)]


def prepare_lass(
    clotho_folder, fsd50k_folder, fsd50k_caption_json, save_folder
):
    """Write ``train.csv`` and ``valid.csv`` into ``save_folder``.

    Arguments
    ---------
    clotho_folder : str
        Contains ``clotho_audio_<split>/`` and ``clotho_captions_<split>.csv``
        for ``development`` (and optionally ``validation``).
    fsd50k_folder : str
        Contains ``FSD50K.dev_audio/``.
    fsd50k_caption_json : str
        ``fsd50k_dev_auto_caption.json`` from the DCASE 2024 Task 9 baseline.
    save_folder : str
        Where the manifests are written.
    """
    os.makedirs(save_folder, exist_ok=True)
    train = _clotho_rows(clotho_folder, "development") + _fsd50k_rows(
        fsd50k_folder, fsd50k_caption_json
    )
    _write_csv(os.path.join(save_folder, "train.csv"), train)
    if os.path.isfile(
        os.path.join(clotho_folder, "clotho_captions_validation.csv")
    ):
        valid = _clotho_rows(clotho_folder, "validation")
        _write_csv(os.path.join(save_folder, "valid.csv"), valid)


def _clotho_rows(folder, split):
    audio_dir = os.path.join(folder, f"clotho_audio_{split}")
    rows = []
    with open(os.path.join(folder, f"clotho_captions_{split}.csv")) as f:
        for entry in csv.DictReader(f):
            wav = os.path.abspath(os.path.join(audio_dir, entry["file_name"]))
            stem = os.path.splitext(entry["file_name"])[0]
            for i, column in enumerate(CLOTHO_CAPTION_COLUMNS, start=1):
                rows.append(_row(f"clotho_{stem}_{i}", wav, entry[column]))
    return rows


def _fsd50k_rows(folder, caption_json):
    audio_dir = os.path.join(folder, "FSD50K.dev_audio")
    with open(caption_json) as f:
        entries = json.load(f)["data"]
    rows = []
    for entry in entries:
        wav = os.path.abspath(os.path.join(audio_dir, entry["wav"]))
        stem = os.path.splitext(entry["wav"])[0]
        rows.append(_row(f"fsd50k_{stem}", wav, entry["caption"]))
    return rows


def _row(row_id, wav, caption):
    if not os.path.isfile(wav):
        raise FileNotFoundError(f"missing audio file: {wav}")
    info = read_audio_info(wav)
    duration = info.num_frames / info.sample_rate
    return {
        "ID": row_id,
        "duration": duration,
        "wav": wav,
        "caption": caption.strip(),
    }


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ID", "duration", "wav", "caption"]
        )
        writer.writeheader()
        writer.writerows(rows)
