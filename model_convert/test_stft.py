from demucs.stft_process import STFT_Process
import torch
import torch.nn.functional as F
from demucs.pretrained import *
from demucs.hdemucs import pad1d
import math
import numpy as np

NFFT = 4096
HOP_LENGTH = 1024
WINDOW = "hann"
PAD_MODE = "reflect"

# demucs
htdemucs = get_model("htdemucs").models[0]
# fake input
fake_stft_input = torch.randn((1, 2, int(5 * 44100)), dtype=torch.float32)


# =========== test htdemucs ===========
# htdemucs(fake_stft_input)

# =========== stft ============
# z = htdemucs._spec(fake_stft_input)
# demucs_mag = htdemucs._magnitude(z).to(fake_stft_input.device)
# print(f"demucs_mag.shape = {demucs_mag.size()}")

# # mix = torch.randn((1, 2, 44100), dtype=torch.float32)
# stft_module = STFT_Process(model_type="stft_B",
#                            n_fft=NFFT,
#                            hop_len=HOP_LENGTH,
#                            window_type=WINDOW).eval()

# le = int(math.ceil(fake_stft_input.shape[-1] / HOP_LENGTH))
# pad = HOP_LENGTH // 2 * 3
# mix_padded = pad1d(fake_stft_input, (pad, pad + le * HOP_LENGTH - fake_stft_input.shape[-1]), mode="reflect")

# real_part, imag_part = stft_module.forward(mix_padded.permute(1, 0, 2), PAD_MODE)
# real_part = real_part.unsqueeze(1)
# imag_part = imag_part.unsqueeze(1)
# z = torch.concat((real_part, imag_part), dim=1)
# C, _, Fr, T = z.shape
# z = z.reshape(-1, Fr, T).unsqueeze(0)
# z = z[..., :-1, 2 : 2 + le]
# z = z / np.sqrt(NFFT)
# print(f"stft_module output.size: {z.size()}")

# np.testing.assert_allclose(z.numpy(), demucs_mag.numpy(), atol=1e-4, rtol=1e-3)


# =========== istft ===============
fake_istft_input = torch.randn((1, 4, 4, 2048, 336), dtype=torch.float32)

zout = htdemucs._mask(None, fake_istft_input)
# print(f"zout.size() = {zout.size()}")
demucs_istft_out = htdemucs._ispec(zout, length=fake_stft_input.shape[-1])
print(f"demucs_istft_out.shape = {demucs_istft_out.size()}")

# mask
B, S, C, Fr, T = fake_istft_input.shape
zout = fake_istft_input.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
zout = torch.view_as_complex(zout.contiguous())
# print(f"zout.size() = {zout.size()}")

# ispectro
hl = HOP_LENGTH
length = fake_stft_input.shape[-1]
z = F.pad(zout, (0, 0, 0, 1))
z = F.pad(z, (2, 2))
pad = hl // 2 * 3
le = hl * int(math.ceil(length / hl)) + 2 * pad

*other, freqs, frames = z.shape
n_fft = 2 * freqs - 2
z = z.view(-1, freqs, frames)
print(f"z.size = {z.size()}")

istft_module = STFT_Process(model_type="istft_A",
                            n_fft=n_fft,
                            hop_len=HOP_LENGTH,
                            window_type=WINDOW).eval()
x = istft_module.forward(z.abs(), z.angle()).squeeze(1)
print(f"x.size() = {x.size()}")
_, le = x.shape
x = x.view(*other, le)
print(f"pad = {pad} length = {length}")
x = x[..., pad: pad + length]
print(f"istft output.shape = {x.size()}")

x = x * math.sqrt(NFFT)

np.testing.assert_allclose(x.numpy(), demucs_istft_out.numpy(), atol=1e-5, rtol=1e-4)