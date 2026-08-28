"""Tests for the LASS separator: FiLM-conditioned ResUNet30 + T-F masking.

Topology follows the released AudioSep code (not the paper prose):
6 encoder blocks (32,64,128,256,384,384) + 1 bottleneck + 6 decoder blocks,
one residual conv block each, FiLM = one Linear(cond_dim -> C) per BatchNorm
site whose output is added after the BN (no multiplicative term), and a
3-map head (sigmoid magnitude mask + tanh phase rotation).
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

from models import LASSMasker, ResUNet30  # noqa: E402

N_BINS = 513  # n_fft=1024 -> 513 bins (the net crops to 512 and pads back)


def test_resunet_maps_spectrogram_to_three_maps_of_same_shape():
    net = ResUNet30(cond_dim=512)
    spec = torch.rand(1, 1, 40, N_BINS)  # (batch, channel, time, freq)
    out = net(spec, torch.randn(1, 512))
    assert out.shape == (1, 3, 40, N_BINS)  # time 40 is not a multiple of 32


def test_query_changes_the_output():
    net = ResUNet30(cond_dim=512).eval()
    spec = torch.rand(1, 1, 32, N_BINS)
    a = net(spec, torch.randn(1, 512))
    b = net(spec, torch.randn(1, 512))
    assert not torch.allclose(a, b)


def test_parameter_count_matches_audiosep_resunet30():
    n = sum(p.numel() for p in ResUNet30(cond_dim=512).parameters())
    assert 24_000_000 < n < 29_000_000  # AudioSep: ~21.4M net + ~5.1M FiLM


def test_masker_returns_waveform_of_same_length():
    masker = LASSMasker(n_fft=1024, hop_length=160, cond_dim=512)
    wav = torch.randn(2, 16000)
    out = masker(wav, torch.randn(2, 512))
    assert out.shape == wav.shape


def test_masker_never_amplifies_a_bin_magnitude_only_attenuates():
    masker = LASSMasker(n_fft=1024, hop_length=160, cond_dim=512).eval()
    wav = torch.randn(1, 16000)
    mag_mask = masker.mask(wav, torch.randn(1, 512))
    assert mag_mask.min() >= 0 and mag_mask.max() <= 1
