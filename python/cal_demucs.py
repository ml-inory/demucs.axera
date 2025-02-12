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

import torch

def create_sliding_windows(tensor, kernel_size):
    """
    创建滚动窗口视图，用于后续计算中值。

    参数:
    - tensor: 输入的一维张量 (length,)
    - kernel_size: 中值滤波器的核大小，必须是奇数

    返回:
    - 滚动窗口视图 (new_length, kernel_size)
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    pad_width = kernel_size // 2
    # 使用边缘填充以处理边界
    padded_tensor = torch.nn.functional.pad(tensor.unsqueeze(0).unsqueeze(0), 
                                            (pad_width, pad_width), mode='replicate').squeeze()

    length = padded_tensor.shape[0]
    new_length = length - kernel_size + 1
    strides = padded_tensor.stride()

    # 创建滚动窗口视图
    windows = padded_tensor.as_strided(
        size=(new_length, kernel_size),
        stride=(strides[0], strides[0])
    )

    return windows

def median_filter_1d(tensor, kernel_size=3):
    """
    对一维张量应用中值滤波。

    参数:
    - tensor: 输入的一维张量 (length,)
    - kernel_size: 中值滤波器的核大小，必须是奇数

    返回:
    - 处理后的一维张量 (length,)
    """
    windows = create_sliding_windows(tensor, kernel_size)
    # 计算每个窗口的中值
    filtered_tensor, _ = torch.median(windows, dim=-1)
    return filtered_tensor

def apply_median_filter_centered_N(tensor, kernel_size=3, stride=1024, N=512):
    """
    对一个shape为(B, length)的torch.Tensor，每隔stride长度取中心点左右各N个数据做中值滤波

    参数:
    - tensor: 输入的torch.Tensor，形状为(B, length)
    - kernel_size: 中值滤波器的核大小，默认为3
    - stride: 每次处理的步长，默认为1024
    - N: 每个切片的半径，默认为512

    返回:
    - 处理后的torch.Tensor
    """
    origin_shape = tensor.shape
    tensor = tensor.view(-1, origin_shape[-1])

    batch_size, length = tensor.shape
    filtered_tensor = tensor.clone()  # 克隆原始张量以避免修改原始数据

    for i in range(batch_size):
        start = 0
        while start < length:
            center = min(start + stride // 2, length - 1)  # 确定中心点
            segment_start = max(center - N, 0)
            segment_end = min(center + N + 1, length)  # +1 是因为切片是左闭右开
            
            segment = tensor[i, segment_start:segment_end]
            
            if segment.shape[0] >= kernel_size:  # 确保有足够的数据来应用中值滤波
                # 应用中值滤波
                filtered_segment = median_filter_1d(segment, kernel_size=kernel_size)
                
                # 将过滤后的段放回原位置
                filtered_tensor[i, segment_start:segment_end] = filtered_segment
            
            start += stride

    return filtered_tensor.view(origin_shape)

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

    # 滤波
    x = apply_median_filter_centered_N(x, kernel_size=11, stride=1024, N=32)

    meant = padded_m.mean(dim=(1, 2), keepdim=True)
    stdt = padded_m.std(dim=(1, 2), keepdim=True)

    xt = xt.view(B, S, -1, training_length)
    xt = xt * stdt[:, None] + meant[:, None]
    out = xt + x

    return out
