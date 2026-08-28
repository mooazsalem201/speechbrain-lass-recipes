"""FiLM-conditioned ResUNet30 masker for text-queried sound separation.

Topology follows the *released* AudioSep code (github.com/Audio-AGI/AudioSep,
MIT license), which differs from the paper prose: 6 encoder blocks with
channels 32/64/128/256/384/384, one bottleneck block, 6 mirrored decoder
blocks, one residual conv block each. Text conditioning is FiLM with a bias
term only: one ``Linear(cond_dim, C)`` per BatchNorm, added after the BN and
before the activation. The head predicts 3 maps: a sigmoid magnitude mask and
a tanh (cos, sin) pair that rotates the mixture phase.

Spectrograms are laid out as ``(batch, channel, time, freq)``.

Authors
 * <your name> 2026
"""

import torch
from torch import nn
from torch.nn import functional as F

LEAKY_SLOPE = 0.01
BN_MOMENTUM = 0.01
CHANNELS = (32, 64, 128, 256, 384, 384)


class FiLM(nn.Module):
    """Adds a per-channel bias computed from the query vector."""

    def __init__(self, cond_dim, channels):
        super().__init__()
        self.linear = nn.Linear(cond_dim, channels)

    def forward(self, x, cond):
        return x + self.linear(cond)[:, :, None, None]


class ConvBlockRes(nn.Module):
    """[BN -> +beta -> LeakyReLU -> 3x3 conv] x2, plus a shortcut."""

    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM)
        self.film1 = FiLM(cond_dim, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.film2 = FiLM(cond_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        h = self.conv1(F.leaky_relu(self.film1(self.bn1(x), cond), LEAKY_SLOPE))
        h = self.conv2(F.leaky_relu(self.film2(self.bn2(h), cond), LEAKY_SLOPE))
        return h + self.shortcut(x)


class EncoderBlock(nn.Module):
    """Residual block followed by average pooling; keeps the skip tensor."""

    def __init__(self, in_channels, out_channels, cond_dim, downsample):
        super().__init__()
        self.block = ConvBlockRes(in_channels, out_channels, cond_dim)
        self.downsample = downsample

    def forward(self, x, cond):
        skip = self.block(x, cond)
        return F.avg_pool2d(skip, self.downsample), skip


class DecoderBlock(nn.Module):
    """Transposed-conv upsampling, skip concatenation, residual block."""

    def __init__(self, in_channels, out_channels, cond_dim, upsample):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM)
        self.film = FiLM(cond_dim, in_channels)
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, upsample, stride=upsample
        )
        self.block = ConvBlockRes(out_channels * 2, out_channels, cond_dim)

    def forward(self, x, skip, cond):
        x = self.up(F.leaky_relu(self.film(self.bn(x), cond), LEAKY_SLOPE))
        return self.block(torch.cat([x, skip], dim=1), cond)


class ResUNet30(nn.Module):
    """Text-conditioned ResUNet predicting 3 maps per T-F bin.

    Arguments
    ---------
    cond_dim : int
        Size of the query vector (512 for CLAP).
    n_bins : int
        Frequency bins of the input spectrogram (``n_fft // 2 + 1``). The
        last bin is cropped inside the net and padded back at the output.
    channels : tuple
        Encoder channel widths (mirrored by the decoder).

    Example
    -------
    >>> net = ResUNet30(cond_dim=512)
    >>> net(torch.rand(1, 1, 40, 513), torch.randn(1, 512)).shape
    torch.Size([1, 3, 40, 513])
    """

    def __init__(self, cond_dim=512, n_bins=513, channels=CHANNELS, n_maps=3):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(n_bins, momentum=BN_MOMENTUM)
        self.pre_conv = nn.Conv2d(1, channels[0], 1)
        widths = (channels[0],) + tuple(channels)
        self.encoders = nn.ModuleList()
        for i in range(len(channels)):
            pool = (1, 2) if i == len(channels) - 1 else (2, 2)
            self.encoders.append(
                EncoderBlock(widths[i], widths[i + 1], cond_dim, pool)
            )
        self.bottleneck = ConvBlockRes(channels[-1], channels[-1], cond_dim)
        # Decoder e undoes encoder e: it upsamples what the previous stage
        # produced and outputs the width of encoder e's skip tensor.
        self.decoders = nn.ModuleList()
        in_channels = channels[-1]
        for e in reversed(range(len(channels))):
            up = (1, 2) if e == len(channels) - 1 else (2, 2)
            self.decoders.append(
                DecoderBlock(in_channels, widths[e + 1], cond_dim, up)
            )
            in_channels = widths[e + 1]
        self.after_block = ConvBlockRes(widths[1], channels[0], cond_dim)
        self.after_conv = nn.Conv2d(channels[0], n_maps, 1)
        self.time_multiple = 2 ** (len(channels) - 1)

    def forward(self, spec, cond):
        """(batch, 1, time, n_bins) magnitudes -> (batch, 3, time, n_bins)."""
        x = self.bn0(spec.transpose(1, 3)).transpose(1, 3)
        n_frames = x.shape[2]
        pad = (-n_frames) % self.time_multiple
        x = F.pad(x, (0, 0, 0, pad))[..., :-1]  # pad time, crop last bin
        x = self.pre_conv(x)
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x, cond)
            skips.append(skip)
        x = self.bottleneck(x, cond)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip, cond)
        x = self.after_conv(self.after_block(x, cond))
        return F.pad(x[:, :, :n_frames], (0, 1))  # restore the cropped bin


class LASSMasker(nn.Module):
    """Waveform in, separated waveform out: STFT -> ResUNet30 -> mask -> iSTFT.

    Arguments
    ---------
    n_fft : int
    hop_length : int
    cond_dim : int
        Size of the query vector.
    channels : tuple
        Passed to :class:`ResUNet30`.

    Example
    -------
    >>> masker = LASSMasker(n_fft=1024, hop_length=160)
    >>> masker(torch.randn(2, 16000), torch.randn(2, 512)).shape
    torch.Size([2, 16000])
    """

    def __init__(
        self, n_fft=1024, hop_length=160, cond_dim=512, channels=CHANNELS
    ):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
        self.net = ResUNet30(cond_dim, n_fft // 2 + 1, channels)

    def _stft(self, wav):
        spec = torch.stft(
            wav,
            self.n_fft,
            self.hop_length,
            window=self.window,
            return_complex=True,
        )
        return spec.transpose(1, 2)  # (batch, time, freq)

    def _maps(self, spec, cond):
        out = self.net(spec.abs().unsqueeze(1), cond)
        mag_mask = torch.sigmoid(out[:, 0])
        cos, sin = torch.tanh(out[:, 1]), torch.tanh(out[:, 2])
        norm = torch.sqrt(cos**2 + sin**2).clamp_min(1e-8)
        return mag_mask, cos / norm, sin / norm

    def mask(self, wav, cond):
        """Magnitude mask in [0, 1], shape (batch, time, freq)."""
        return self._maps(self._stft(wav), cond)[0]

    def forward(self, wav, cond):
        spec = self._stft(wav)
        mag_mask, cos_rot, sin_rot = self._maps(spec, cond)
        unit = spec / spec.abs().clamp_min(1e-8)  # e^{j angle(X)}
        rotation = torch.complex(cos_rot, sin_rot)  # e^{j delta}
        estimate = F.relu(spec.abs() * mag_mask) * unit * rotation
        return torch.istft(
            estimate.transpose(1, 2),
            self.n_fft,
            self.hop_length,
            window=self.window,
            length=wav.shape[-1],
        )
