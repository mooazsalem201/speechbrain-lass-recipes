"""DCASE 2024 Task 9 evaluation for language-queried separation (LASS).

Metrics and mixture synthesis are ported verbatim from the challenge
baseline (github.com/Audio-AGI/dcase2024_task9_baseline, MIT license) so
that scores stay directly comparable to the published leaderboard.
Reference point: the released baseline checkpoint scores
SDR 5.708 / SDRi 5.673 / SI-SDR 3.862 on the synthetic validation set.

Do not "improve" the metric functions: any change breaks comparability.

Authors
 * <your name> 2026
"""

import csv
import os

import numpy as np
import torch
import torchaudio

from speechbrain.dataio.dataio import read_audio, read_audio_info


def calculate_sdr(ref, est, eps=1e-10):
    """Signal-to-distortion ratio in dB (challenge definition).

    Arguments
    ---------
    ref : np.ndarray
        Reference (clean target) signal.
    est : np.ndarray
        Estimated signal.
    eps : float
        Floor applied to both signal and noise power.

    Returns
    -------
    float
        SDR in dB.

    Example
    -------
    >>> ref = np.ones(4, dtype=np.float32)
    >>> round(float(calculate_sdr(ref, ref * 1.1)), 3)
    20.0
    """
    # Computed in float64 so results do not drift with NumPy's float32
    # reduction order; agrees with the challenge code to ~1e-6 dB.
    reference = np.asarray(ref, dtype=np.float64)
    noise = np.asarray(est, dtype=np.float64) - reference
    numerator = np.clip(a=np.mean(reference**2), a_min=eps, a_max=None)
    denominator = np.clip(a=np.mean(noise**2), a_min=eps, a_max=None)
    sdr = 10.0 * np.log10(numerator / denominator)
    return sdr


def calculate_sisdr(ref, est):
    """Scale-invariant SDR in dB (challenge definition).

    Arguments
    ---------
    ref : np.ndarray
        Reference (clean target) signal.
    est : np.ndarray
        Estimated signal.

    Returns
    -------
    float
        SI-SDR in dB.

    Example
    -------
    >>> ref = np.ones(4, dtype=np.float32)
    >>> bool(calculate_sisdr(ref, 2 * ref) > 60)  # scale does not matter
    True
    """
    eps = np.finfo(ref.dtype).eps  # the challenge's float32 eps
    reference = np.asarray(ref, dtype=np.float64)
    estimate = np.asarray(est, dtype=np.float64)
    reference = reference.reshape(reference.size, 1)
    estimate = estimate.reshape(estimate.size, 1)
    Rss = np.dot(reference.T, reference)
    # get the scaling factor for clean sources
    a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)
    e_true = a * reference
    e_res = estimate - e_true
    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()
    sisdr = 10 * np.log10((eps + Sss) / (eps + Snn))
    return sisdr


def mix_at_snr(source, noise, snr_db):
    """Scale ``noise`` to the requested SNR, add it to ``source``, de-clip.

    If the mixture peaks above 1, both the mixture and the source are scaled
    by ``0.9 / peak`` so the target stays consistent with the mixture.

    Arguments
    ---------
    source : np.ndarray
        Clean target signal.
    noise : np.ndarray
        Interference signal (same length as ``source``).
    snr_db : int or float
        Target signal-to-noise ratio in dB.

    Returns
    -------
    mixture : np.ndarray
    source : np.ndarray
        The (possibly rescaled) target to score against.

    Example
    -------
    >>> ones = np.ones(4)
    >>> mixture, source = mix_at_snr(ones, ones, snr_db=0)
    >>> np.round(mixture, 2).tolist(), np.round(source, 2).tolist()
    ([0.9, 0.9, 0.9, 0.9], [0.45, 0.45, 0.45, 0.45])
    """
    source_power = np.mean(source**2)
    noise_power = np.mean(noise**2)
    desired_noise_power = source_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(desired_noise_power / noise_power)
    mixture = source + noise
    max_value = np.max(np.abs(mixture))
    if max_value > 1:
        source = source * (0.9 / max_value)
        mixture = mixture * (0.9 / max_value)
    return mixture, source


def load_mono(path, sample_rate):
    """Load a wav as a mono float32 numpy array at ``sample_rate``."""
    wav = read_audio(path)
    if wav.dim() > 1:
        wav = wav.mean(dim=1)
    sr = read_audio_info(path).sample_rate
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.numpy()


def evaluate(separate, eval_csv, audio_dir, sample_rate=16000):
    """Score a separation function on the DCASE synthetic validation set.

    Arguments
    ---------
    separate : callable
        ``separate(mixture: np.ndarray, caption: str) -> np.ndarray``.
    eval_csv : str
        ``lass_synthetic_validation.csv`` (source, noise, snr, caption).
    audio_dir : str
        Folder with the validation wavs.
    sample_rate : int
        Evaluation sample rate (the challenge uses 16 kHz).

    Returns
    -------
    dict
        Mean ``SDR``, ``SDRi`` and ``SI-SDR`` in dB.
    """
    with open(eval_csv) as f:
        rows = list(csv.reader(f))[1:]
    sdrs, sdris, sisdrs = [], [], []
    with torch.no_grad():
        for source_id, noise_id, snr, caption in rows:
            source = load_mono(
                os.path.join(audio_dir, source_id + ".wav"), sample_rate
            )
            noise = load_mono(
                os.path.join(audio_dir, noise_id + ".wav"), sample_rate
            )
            mixture, source = mix_at_snr(source, noise, int(snr))
            sdr_no_sep = calculate_sdr(ref=source, est=mixture)
            estimate = separate(mixture, caption)
            sdr = calculate_sdr(ref=source, est=estimate)
            sdrs.append(sdr)
            sdris.append(sdr - sdr_no_sep)
            sisdrs.append(calculate_sisdr(ref=source, est=estimate))
    return {
        "SDR": float(np.mean(sdrs)),
        "SDRi": float(np.mean(sdris)),
        "SI-SDR": float(np.mean(sisdrs)),
    }
