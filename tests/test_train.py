"""End-to-end smoke test: the recipe trains for one debug epoch on tiny data.

Mirrors SpeechBrain's recipe tests: run ``train.py`` with the real hparams
file, override paths/sizes, and check it completes and logs the epoch.
"""

import csv
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

RECIPE = os.path.join(
    os.path.dirname(__file__), "..", "recipes", "LASS", "separation"
)


def _manifest(path, folder, names, seconds=1.0, sr=16000):
    rng = np.random.default_rng(0)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ID", "duration", "wav", "caption"]
        )
        writer.writeheader()
        for name in names:
            wav = os.path.join(folder, name + ".wav")
            sf.write(wav, 0.1 * rng.standard_normal(int(seconds * sr)), sr)
            writer.writerow(
                {"ID": name, "duration": seconds, "wav": wav, "caption": name}
            )


def test_one_debug_epoch_runs_and_logs(tmp_path):
    train_csv, valid_csv = tmp_path / "train.csv", tmp_path / "valid.csv"
    _manifest(train_csv, tmp_path, ["a dog barking", "rain", "a car", "birds"])
    _manifest(valid_csv, tmp_path, ["typing", "a crowd applauding"])
    output = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            os.path.join(RECIPE, "train.py"),
            os.path.join(RECIPE, "hparams", "resunet_clap_16k.yaml"),
            "--debug",
            f"--data_folder={tmp_path}",
            f"--output_folder={output}",
            f"--train_csv={train_csv}",
            f"--valid_csv={valid_csv}",
            "--skip_prep=True",
            "--number_of_epochs=1",
            "--batch_size=2",
            "--segment_seconds=1.0",
            "--num_workers=0",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    log = open(output / "train_log.txt").read()
    assert "epoch: 1" in log and "train loss" in log
