# encoding: utf-8
"""
@author: eric
@contact: master2017@163.com
@software: PyCharm
@file: cal_demucs.py
@time: 2024/6/23 下午1:42
"""
import math
import typing as tp
import torch as th
from torch.nn import functional as F
import numpy as np


def pad1d(x: th.Tensor, paddings: tp.Tuple[int, int], mode: str = 'constant', value: float = 0.):
    """
    :param x:
    :param paddings:
    :param mode:
    :param value:
    :return:
    """
    x0 = x
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            extra_pad_right = min(padding_right, extra_pad)
            extra_pad_left = extra_pad - extra_pad_right
            paddings = (padding_left - extra_pad_left, padding_right - extra_pad_right)
            x = F.pad(x, (extra_pad_left, extra_pad_right))
    out = F.pad(x, paddings, mode, value)
    assert out.shape[-1] == length + padding_left + padding_right
    assert (out[..., padding_left: padding_left + length] == x0).all()
    return out


def spectro(x, n_fft=512, hop_length=None, pad=0):
    """
    :param x:
    :param n_fft:
    :param hop_length:
    :param pad:
    :return:
    """
    *other, length = x.shape
    x = x.reshape(-1, length)

    device_type = x.device.type
    is_other_gpu = not device_type in ["cuda", "cpu"]

    if is_other_gpu:
        x = x.cpu()
    z = th.stft(x,
                n_fft * (1 + pad),
                hop_length or n_fft // 4,
                window=th.hann_window(n_fft).to(x),
                win_length=n_fft,
                normalized=True,
                center=True,
                return_complex=True,
                pad_mode='reflect')
    _, freqs, frame = z.shape
    # print(f"z.shape = {z.shape}")
    return z.view(*other, freqs, frame)


def demucs_spec(x, nfft=4096):
    """
    :param x:
    :param nfft:
    :return:
    """
    hop_length = nfft // 4
    hl = hop_length
    nfft = nfft
    x0 = x  # noqa
    assert hl == nfft // 4
    le = int(math.ceil(x.shape[-1] / hl))
    pad = hl // 2 * 3
    x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")

    z = spectro(x, nfft, hl)[..., :-1, :]
    assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
    z = z[..., 2: 2 + le]
    return z


def demucs_magnitude(z, cac=True):
    """
    :param z:
    :param cac:
    :return:
    """
    if cac:
        B, C, Fr, T = z.shape
        m = th.view_as_real(z).permute(0, 1, 4, 2, 3)
        m = m.reshape(B, C * 2, Fr, T)
    else:
        m = z.abs()
    return m


def demucs_mask(m):
    """
    :param m:
    :return:
    """
    B, S, C, Fr, T = m.shape
    out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
    out = th.view_as_complex(out.contiguous())
    return out


def demucs_ispectro(z, hop_length=None, length=None, pad=0):
    """
    :param z:
    :param hop_length:
    :param length:
    :param pad:
    :return:
    """
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)

    device_type = z.device.type
    is_other_gpu = not device_type in ["cuda", "cpu"]

    if is_other_gpu:
        z = z.cpu()
    x = th.istft(z,
                 n_fft,
                 hop_length,
                 window=th.hann_window(win_length).to(z.real),
                 win_length=win_length,
                 normalized=True,
                 length=length,
                 center=True)
    _, length = x.shape
    x = x.view(*other, length)
    return x


def demucs_ispec(z, length=None, nfft=4096, scale=0):
    """
    :param z:
    :param length:
    :param nfft:
    :param scale:
    :return:
    """
    hop_length = nfft // 4
    hl = hop_length // (4 ** scale)
    z = F.pad(z, (0, 0, 0, 1))
    z = F.pad(z, (2, 2))
    pad = hl // 2 * 3
    le = hl * int(math.ceil(length / hl)) + 2 * pad
    x = demucs_ispectro(z, hl, length=le)
    x = x[..., pad: pad + length]
    return x


def demucs_post_process(m, xt, padded_m, segment, samplerate, B, S):
    """
    :param m:
    :param xt:
    :param padded_m:
    :param segment:
    :param samplerate:
    :param B:
    :param S:
    :return:
    """
    zout = demucs_mask(m)
    training_length = int(segment * samplerate)
    x = demucs_ispec(zout, length=training_length)

    meant = padded_m.mean(dim=(1, 2), keepdim=True)
    stdt = padded_m.std(dim=(1, 2), keepdim=True)

    xt = xt.view(B, S, -1, training_length)
    xt = xt * stdt[:, None] + meant[:, None]
    out = xt + x

    return out
