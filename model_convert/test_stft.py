from demucs.stft_process import STFT_Process
import torch
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

mix = torch.randn((1, 2, 44100), dtype=torch.float32)

z = htdemucs._spec(mix)
print(z.size())
demucs_mag = htdemucs._magnitude(z).to(mix.device)
print(f"demucs_mag.shape = {demucs_mag.size()}")

# mix = torch.randn((1, 2, 44100), dtype=torch.float32)
stft_module = STFT_Process(model_type="stft_B",
                           n_fft=NFFT,
                           hop_len=HOP_LENGTH,
                           window_type=WINDOW).eval()

le = int(math.ceil(mix.shape[-1] / HOP_LENGTH))
pad = HOP_LENGTH // 2 * 3
mix_padded = pad1d(mix, (pad, pad + le * HOP_LENGTH - mix.shape[-1]), mode="reflect")

real_part, imag_part = stft_module.forward(mix_padded.permute(1, 0, 2), PAD_MODE)
real_part = real_part.unsqueeze(1)
imag_part = imag_part.unsqueeze(1)
z = torch.concat((real_part, imag_part), dim=1)
C, _, Fr, T = z.shape
z = z.reshape(-1, Fr, T).unsqueeze(0)
z = z[..., :-1, 2 : 2 + le]
z = z / np.sqrt(NFFT)
print(f"stft_module output.size: {z.size()}")

np.testing.assert_allclose(z.numpy(), demucs_mag.numpy(), atol=1e-5, rtol=1e-3)