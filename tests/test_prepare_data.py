"""Tests for the LASS data preparation (manifest creation)."""

import csv
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "recipes", "LASS", "separation"
    ),
)

from prepare_data import prepare_lass  # noqa: E402


def _wav(path, seconds, sr=16000):
    sf.write(str(path), np.zeros(int(seconds * sr), dtype=np.float32), sr)


def _fake_clotho(root, split, files):
    (root / f"clotho_audio_{split}").mkdir(parents=True)
    with open(root / f"clotho_captions_{split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name"] + [f"caption_{i}" for i in range(1, 6)])
        for name, seconds in files:
            _wav(root / f"clotho_audio_{split}" / name, seconds)
            w.writerow([name] + [f"{name} cap {i}" for i in range(1, 6)])


def _fake_fsd50k(root, caption_json):
    (root / "FSD50K.dev_audio").mkdir(parents=True)
    _wav(root / "FSD50K.dev_audio" / "10000.wav", 0.5)
    _wav(root / "FSD50K.dev_audio" / "10001.wav", 0.25)
    data = [
        {"wav": "10000.wav", "caption": "Breathing, then a cough."},
        {"wav": "10001.wav", "caption": "A rattle shakes."},
    ]
    json.dump({"data": data}, open(caption_json, "w"))


def test_train_manifest_has_one_row_per_caption(tmp_path):
    clotho, fsd = tmp_path / "clotho", tmp_path / "fsd50k"
    _fake_clotho(clotho, "development", [("a.wav", 2.0)])
    caps = tmp_path / "fsd50k_dev_auto_caption.json"
    _fake_fsd50k(fsd, caps)
    save = tmp_path / "manifests"

    prepare_lass(str(clotho), str(fsd), str(caps), str(save))

    rows = list(csv.DictReader(open(save / "train.csv")))
    assert [r["caption"] for r in rows] == [
        "a.wav cap 1",
        "a.wav cap 2",
        "a.wav cap 3",
        "a.wav cap 4",
        "a.wav cap 5",
        "Breathing, then a cough.",
        "A rattle shakes.",
    ]
    assert rows[0]["ID"] == "clotho_a_1"
    assert rows[5]["ID"] == "fsd50k_10000"
    assert float(rows[0]["duration"]) == 2.0
    assert float(rows[6]["duration"]) == 0.25
    assert os.path.isabs(rows[0]["wav"]) and rows[0]["wav"].endswith("a.wav")


def test_valid_manifest_comes_from_clotho_validation_split(tmp_path):
    clotho, fsd = tmp_path / "clotho", tmp_path / "fsd50k"
    _fake_clotho(clotho, "development", [("a.wav", 1.0)])
    _fake_clotho(clotho, "validation", [("v.wav", 1.5)])
    caps = tmp_path / "caps.json"
    _fake_fsd50k(fsd, caps)
    save = tmp_path / "manifests"

    prepare_lass(str(clotho), str(fsd), str(caps), str(save))

    rows = list(csv.DictReader(open(save / "valid.csv")))
    assert len(rows) == 5
    assert all(r["ID"].startswith("clotho_v_") for r in rows)
    assert all(float(r["duration"]) == 1.5 for r in rows)


def test_missing_audio_file_is_reported_not_silently_skipped(tmp_path):
    clotho, fsd = tmp_path / "clotho", tmp_path / "fsd50k"
    _fake_clotho(clotho, "development", [("a.wav", 1.0)])
    caps = tmp_path / "caps.json"
    _fake_fsd50k(fsd, caps)
    os.remove(fsd / "FSD50K.dev_audio" / "10001.wav")
    save = tmp_path / "manifests"

    try:
        prepare_lass(str(clotho), str(fsd), str(caps), str(save))
    except FileNotFoundError as e:
        assert "10001.wav" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError for missing wav")
