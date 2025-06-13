from demucs.pretrained import *
import torch
from torch.nn import functional as F
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
import glob
import argparse


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
    

def generate_data(input_audios,
                overlap: float = 0.25,
                device: tp.Union[str, torch.device] = 'cpu',
                len_model_sources=4,
                segment=5,
                samplerate=44100,
                save_path="calibration_dataset",
                max_num=-1,
                target_sr=44100,
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
    os.makedirs(save_path, exist_ok=True)
    mix_path = os.path.join(save_path, "mix")
    mag_path = os.path.join(save_path, "mag")

    os.makedirs(mix_path, exist_ok=True)
    os.makedirs(mag_path, exist_ok=True)

    mix_files = tarfile.open(f"{save_path}/mix.tar.gz", "w:gz")
    mag_files = tarfile.open(f"{save_path}/mag.tar.gz", "w:gz")

    for input_audio in input_audios:
        audio_name = os.path.splitext(os.path.basename(input_audio))[0]
        print(f"Saving {audio_name}......")
        
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

        mix = wav[None]

        batch, channels, length = mix.shape

        segment_length: int = int(samplerate * segment)
        stride = int((1 - overlap) * segment_length)

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

            mix_filename = os.path.join(mix_path, f"{audio_name}_{chunk_index}.npy")
            mag_filename = os.path.join(mag_path, f"{audio_name}_{chunk_index}.npy")
    
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=str, required=False, default="./music", help="Path of music")
    parser.add_argument("--output_path", "-o", type=str, required=False, default="./calibration_dataset", help="Seperated wav path")
    parser.add_argument("--overlap", type=float, required=False, default=0.25)
    parser.add_argument("--segment", type=float, required=False, default=5, help="Split in seconds")
    parser.add_argument("--max_num", type=int, default=-1, required=False, help="max data num")
    parser.add_argument("--only_data", action="store_true", required=False, help="Ignore model, only generate data")
    return parser.parse_args()


def main():
    args = get_args()

    target_sr = 44100
    overlap = args.overlap
    seconds_split = args.segment # Fraction(39, 5) # 输入长度，单位秒
    max_num = args.max_num

    input2_shape = generate_data(
        glob.glob(args.input_path + "/*.wav"),
        overlap=overlap,
        save_path=args.output_path,
        segment=seconds_split,
        max_num=max_num
    )


    if not args.only_data:
        # Export ONNX
        model_name = "htdemucs"
        model = get_model(model_name).models[0]

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
