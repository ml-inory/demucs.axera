from demucs.pretrained import *
import torch
from torch import nn
from torch.nn import functional as F
from fractions import Fraction
from einops import rearrange
from onnxsim import simplify
import onnx
import onnxruntime as ort
import soundfile as sf
import numpy as np
import librosa
import tqdm
from cal_demucs import *
import os
import tarfile
import argparse
from pathlib import Path


class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        if isinstance(tensor, TensorChunk):
            self.tensor = tensor.tensor
            self.offset = offset + tensor.offset
        else:
            self.tensor = tensor
            self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
        assert out.shape[-1] == target_length
        return out
    

def generate_data(mix,
                overlap: float = 0.25,
                device: tp.Union[str, torch.device] = 'cpu',
                len_model_sources=4,
                segment=5,
                samplerate=44100,
                save_path="calibration_dataset",
                max_num=-1
                ) -> torch.Tensor:
    """
    :param mix:
    :param overlap:
    :param device:
    :param transition_power:
    :param len_model_sources:
    :param segment:
    :param samplerate:
    :param model:
    :return:
    """
    model_weights = [1.]*len_model_sources
    totals = [0.] * len_model_sources
    batch, channels, length = mix.shape

    segment_length: int = int(samplerate * segment)
    stride = int((1 - overlap) * segment_length)

    os.makedirs(save_path, exist_ok=True)
    mix_path = os.path.join(save_path, "mix")
    mag_path = os.path.join(save_path, "mag")

    os.makedirs(mix_path, exist_ok=True)
    os.makedirs(mag_path, exist_ok=True)

    mix_files = tarfile.open(f"{save_path}/mix.tar.gz", "w:gz")
    mag_files = tarfile.open(f"{save_path}/mag.tar.gz", "w:gz")

    chunk_index = 0
    for offset in tqdm.tqdm(range(0, length, stride)):
        chunk = TensorChunk(mix, offset, segment_length)

        chunk = TensorChunk(chunk)
        padded_mix = chunk.padded(segment_length).to(device)

        input1 = padded_mix.numpy()
        z = demucs_spec(padded_mix)
        mag = demucs_magnitude(z).to(padded_mix.device)
        input2 = mag.numpy()

        input1 = (input1 - input1.mean()) / (input1.std() + 1e-5)
        mean_mag = input2.mean()
        std_mag = input2.std()
        input2 = (input2 - mean_mag) / (std_mag + 1e-5)

        mix_filename = os.path.join(mix_path, f"{chunk_index}.npy")
        mag_filename = os.path.join(mag_path, f"{chunk_index}.npy")
  
        np.save(mix_filename, input1)
        np.save(mag_filename, input2)

        mix_files.add(mix_filename)
        mag_files.add(mag_filename)

        offset += segment_length
        chunk_index += 1
        if max_num > 0 and chunk_index >= max_num:
            print(f"Exceed max_num {max_num}, break")
            break

    mix_files.close()
    mag_files.close()

    print(f"Saved dataset to {save_path}")

    return input2.shape

def main():
    model_name = "htdemucs"
    model = get_model(model_name).models[0]

    target_sr = 44100
    overlap = 0.25
    seconds_split = 5 # Fraction(39, 5) # 输入长度，单位秒
    max_num = 10

    input_audio = "../test.wav"
    wav, origin_sr = sf.read(input_audio, always_2d=True, dtype="float32")
    if origin_sr != target_sr:
        print(f"Origin sample rate is {origin_sr}, resampling to {target_sr}...")
        wav = librosa.resample(wav, orig_sr=origin_sr, target_sr=target_sr)

    if wav.shape[0] != 2:
        wav = wav.transpose()

    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std() + 1e-8
    wav = torch.from_numpy(wav)

    input2_shape = generate_data(
        wav[None],
        overlap=overlap,
        save_path="calibration_dataset",
        segment=seconds_split,
        max_num=max_num
    )

    # Export ONNX
    model.forward = model.forward_for_export

    input_names = ("mix", "mag")
    output_names = ("x", "xt")

    segment_length = int(target_sr * seconds_split)
    inputs = (
        torch.zeros(1,2,segment_length, dtype=torch.float32),
        torch.zeros(*input2_shape, dtype=torch.float32),
    )
    onnx_name = model_name + ".onnx"
    torch.onnx.export(model,               # model being run
        inputs,                    # model input (or a tuple for multiple inputs)
        onnx_name,              # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=16,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = input_names, # the model's input names
        output_names = output_names, # the model's output names
    )
    sim_model,_ = simplify(onnx_name)
    onnx.save(sim_model, onnx_name)
    print(f"Exported model to {onnx_name}")

if __name__ == "__main__":
    main()
