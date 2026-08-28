"""Tests for the DCASE 2024 Task 9 metric port.

Expected values were produced by the challenge baseline's own
``utils.calculate_sdr`` / ``utils.calculate_sisdr`` (Audio-AGI/
dcase2024_task9_baseline) on the same deterministic inputs. Our port must
match them exactly, otherwise our numbers stop being comparable to the
published leaderboard.
"""

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "recipes", "LASS", "separation"
    ),
)

from dcase_eval import calculate_sdr, calculate_sisdr, mix_at_snr  # noqa: E402


def _signals():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(16000).astype(np.float32)
    est = ref + 0.1 * rng.standard_normal(16000).astype(np.float32)
    return ref, est


def test_sdr_matches_reference_implementation():
    ref, est = _signals()
    assert calculate_sdr(ref=ref, est=est) == 20.00750088901096


def test_sisdr_matches_reference_implementation():
    ref, est = _signals()
    assert calculate_sisdr(ref=ref, est=est) == 20.00380277633667


def test_sisdr_is_scale_invariant_but_sdr_is_not():
    ref, est = _signals()
    assert calculate_sisdr(ref=ref, est=2 * est) == 20.00380277633667
    assert calculate_sdr(ref=ref, est=2 * est) == -0.16276967850978868


def test_sdr_of_identical_signals_hits_eps_ceiling():
    ref, _ = _signals()
    assert calculate_sdr(ref=ref, est=ref) == 99.97672970303586


def test_mix_at_snr_hits_requested_snr():
    rng = np.random.default_rng(1)
    source = 0.02 * rng.standard_normal(16000).astype(np.float32)
    noise = 0.02 * rng.standard_normal(16000).astype(np.float32)
    mixture, source_out = mix_at_snr(source, noise, snr_db=-6)
    achieved = 10 * np.log10(
        np.mean(source_out**2) / np.mean((mixture - source_out) ** 2)
    )
    assert abs(achieved - (-6)) < 1e-4
    assert np.array_equal(source_out, source)  # no clipping -> untouched


def test_mix_at_snr_declips_mixture_and_source_together():
    rng = np.random.default_rng(2)
    source = 0.9 * rng.standard_normal(16000).astype(np.float32)
    noise = 0.9 * rng.standard_normal(16000).astype(np.float32)
    mixture, source_out = mix_at_snr(source, noise, snr_db=0)
    assert np.max(np.abs(mixture)) <= 0.9 + 1e-6
    # both signals scaled by the same factor, so the ratio is preserved
    ratio = source_out[np.abs(source) > 1e-3] / source[np.abs(source) > 1e-3]
    assert np.allclose(ratio, ratio[0], atol=1e-6)
