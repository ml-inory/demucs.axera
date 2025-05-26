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
                models,
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
    # x_path = os.path.join(save_path, "x")
    # xt_path = os.path.join(save_path, "xt")
    # saved_0_path = os.path.join(save_path, "saved_0")
    # saved_1_path = os.path.join(save_path, "saved_1")
    # saved_2_path = os.path.join(save_path, "saved_2")
    # saved_3_path = os.path.join(save_path, "saved_3")
    # saved_t_0_path = os.path.join(save_path, "saved_t_0")
    # saved_t_1_path = os.path.join(save_path, "saved_t_1")
    # saved_t_2_path = os.path.join(save_path, "saved_t_2")
    # saved_t_3_path = os.path.join(save_path, "saved_t_3")
    os.makedirs(mix_path, exist_ok=True)
    os.makedirs(mag_path, exist_ok=True)
    # os.makedirs(x_path, exist_ok=True)
    # os.makedirs(xt_path, exist_ok=True)
    # os.makedirs(saved_0_path, exist_ok=True)
    # os.makedirs(saved_1_path, exist_ok=True)
    # os.makedirs(saved_2_path, exist_ok=True)
    # os.makedirs(saved_3_path, exist_ok=True)
    # os.makedirs(saved_t_0_path, exist_ok=True)
    # os.makedirs(saved_t_1_path, exist_ok=True)
    # os.makedirs(saved_t_2_path, exist_ok=True)
    # os.makedirs(saved_t_3_path, exist_ok=True)

    mix_files = tarfile.open(f"{save_path}/mix.tar.gz", "w:gz")
    mag_files = tarfile.open(f"{save_path}/mag.tar.gz", "w:gz")
    # x_files = tarfile.open(f"{save_path}/x.tar.gz", "w:gz")
    # xt_files = tarfile.open(f"{save_path}/xt.tar.gz", "w:gz")
    # saved_0_files = tarfile.open(f"{save_path}/saved_0.tar.gz", "w:gz")
    # saved_1_files = tarfile.open(f"{save_path}/saved_1.tar.gz", "w:gz")
    # saved_2_files = tarfile.open(f"{save_path}/saved_2.tar.gz", "w:gz")
    # saved_3_files = tarfile.open(f"{save_path}/saved_3.tar.gz", "w:gz")
    # saved_t_0_files = tarfile.open(f"{save_path}/saved_t_0.tar.gz", "w:gz")
    # saved_t_1_files = tarfile.open(f"{save_path}/saved_t_1.tar.gz", "w:gz")
    # saved_t_2_files = tarfile.open(f"{save_path}/saved_t_2.tar.gz", "w:gz")
    # saved_t_3_files = tarfile.open(f"{save_path}/saved_t_3.tar.gz", "w:gz")

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

        # x, xt, saved_0, saved_1, saved_2, saved_3, saved_t_0, saved_t_1, saved_t_2, saved_t_3 = models[0].run(None, {"mix": input1, "mag": input2})
        # x_decoder, xt = models[1].run(None, {"mix": input1, "in_x": saved_3})

        mix_filename = os.path.join(mix_path, f"{chunk_index}.npy")
        mag_filename = os.path.join(mag_path, f"{chunk_index}.npy")
        # x_filename = os.path.join(x_path, f"{chunk_index}.npy")
        # xt_filename = os.path.join(xt_path, f"{chunk_index}.npy")
        # saved_0_filename = os.path.join(saved_0_path, f"{chunk_index}.npy")
        # saved_1_filename = os.path.join(saved_1_path, f"{chunk_index}.npy")
        # saved_2_filename = os.path.join(saved_2_path, f"{chunk_index}.npy")
        # saved_3_filename = os.path.join(saved_3_path, f"{chunk_index}.npy")
        # saved_t_0_filename = os.path.join(saved_t_0_path, f"{chunk_index}.npy")
        # saved_t_1_filename = os.path.join(saved_t_1_path, f"{chunk_index}.npy")
        # saved_t_2_filename = os.path.join(saved_t_2_path, f"{chunk_index}.npy")
        # saved_t_3_filename = os.path.join(saved_t_3_path, f"{chunk_index}.npy")

        np.save(mix_filename, input1)
        np.save(mag_filename, input2)
        # np.save(x_filename, x)
        # np.save(xt_filename, xt)
        # np.save(saved_0_filename, saved_0)
        # np.save(saved_1_filename, saved_1)
        # np.save(saved_2_filename, saved_2)
        # np.save(saved_3_filename, saved_3)
        # np.save(saved_t_0_filename, saved_t_0)
        # np.save(saved_t_1_filename, saved_t_1)
        # np.save(saved_t_2_filename, saved_t_2)
        # np.save(saved_t_3_filename, saved_t_3)

        mix_files.add(mix_filename)
        mag_files.add(mag_filename)
        # x_files.add(x_filename)
        # xt_files.add(xt_filename)
        # saved_0_files.add(saved_0_filename)
        # saved_1_files.add(saved_1_filename)
        # saved_2_files.add(saved_2_filename)
        # saved_3_files.add(saved_3_filename)
        # saved_t_0_files.add(saved_t_0_filename)
        # saved_t_1_files.add(saved_t_1_filename)
        # saved_t_2_files.add(saved_t_2_filename)
        # saved_t_3_files.add(saved_t_3_filename)

        offset += segment_length
        chunk_index += 1
        if max_num > 0 and chunk_index >= max_num:
            print(f"Exceed max_num {max_num}, break")
            break

    mix_files.close()
    mag_files.close()
    # x_files.close()
    # xt_files.close()
    # saved_0_files.close()
    # saved_1_files.close()
    # saved_2_files.close()
    # saved_3_files.close()
    # saved_t_0_files.close()
    # saved_t_1_files.close()
    # saved_t_2_files.close()
    # saved_t_3_files.close()

    print(f"Saved dataset to {save_path}")

    return input2.shape

def main():
    model_name = "htdemucs"
    # model = get_model("3ee1a65f", Path("./release_models"))
    model = get_model(model_name).models[0]

    target_sr = 44100
    overlap = 0.25
    seconds_split = 5 # Fraction(39, 5) # 输入长度，单位秒
    # max_num = int(60 / seconds_split)
    max_num = 10

    # input_audio = "../test.wav"
    # wav, origin_sr = sf.read(input_audio, always_2d=True, dtype="float32")
    # if origin_sr != target_sr:
    #     print(f"Origin sample rate is {origin_sr}, resampling to {target_sr}...")
    #     wav = librosa.resample(wav, orig_sr=origin_sr, target_sr=target_sr)

    # if wav.shape[0] != 2:
    #     wav = wav.transpose()

    # ref = wav.mean(0)
    # wav -= ref.mean()
    # wav /= ref.std() + 1e-8
    # wav = torch.from_numpy(wav)

<<<<<<< HEAD
    # input2_shape = generate_data(
    #     wav[None],
    #     overlap=overlap,
    #     save_path="calibration_dataset",
    #     segment=seconds_split,
    #     max_num=max_num
    # )

    # Export ONNX
    # model.forward = model.forward_for_export
    model.use_train_segment = False

    input_names = ("mix", )
    # output_names = ("drums_x","bass_x","other_x","vocals_x", "drums_xt","bass_xt","other_xt","vocals_xt")
    output_names = ("x", )
=======
    input2_shape = generate_data(
        wav[None],
        models=[None],
        overlap=overlap,
        save_path="calib_apollo",
        segment=seconds_split,
        max_num=max_num
    )

    # Export ONNX
    # model.forward = model.forward_for_export

    input_names = ("mix",)
    # output_names = ("drums_x","bass_x","other_x","vocals_x", "drums_xt","bass_xt","other_xt","vocals_xt")
    output_names = ("x",)
>>>>>>> 60c6754 (merge stft and istft to model)

    segment_length = int(target_sr * seconds_split)
    inputs = (
        torch.zeros(1,2,segment_length, dtype=torch.float32),
        # torch.zeros(*input2_shape, dtype=torch.float32),
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

    # Export ONNX
    # model.forward = model.forward_for_export

    # input_names = ("mix", "mag")
    # # output_names = ("drums_x","bass_x","other_x","vocals_x", "drums_xt","bass_xt","other_xt","vocals_xt")
    # output_names = ("x", "xt")

    # segment_length = int(target_sr * seconds_split)
    # inputs = (
    #     torch.zeros(1,2,segment_length, dtype=torch.float32),
    #     torch.zeros(*input2_shape, dtype=torch.float32),
    # )
    # onnx_name = model_name + ".onnx"
    # torch.onnx.export(model,               # model being run
    #     inputs,                    # model input (or a tuple for multiple inputs)
    #     onnx_name,              # where to save the model (can be a file or file-like object)
    #     export_params=True,        # store the trained parameter weights inside the model file
    #     opset_version=16,          # the ONNX version to export the model to
    #     do_constant_folding=True,  # whether to execute constant folding for optimization
    #     input_names = input_names, # the model's input names
    #     output_names = output_names, # the model's output names
    # )
    # sim_model,_ = simplify(onnx_name)
    # onnx.save(sim_model, onnx_name)
    # print(f"Exported model to {onnx_name}")

    # # Export ONNX
    # model.forward = model.forward_encoder

    # input_names = ("mix", "mag")
    # output_names = ("x", "xt", "saved_0", "saved_1", "saved_2", "saved_3", "saved_t_0", "saved_t_1", "saved_t_2", "saved_t_3")
    # segment_length = int(target_sr * seconds_split)
    # inputs = (
    #     torch.zeros(1,2,segment_length, dtype=torch.float32),
    #     torch.zeros(1,4,2048,216, dtype=torch.float32),
    # )
    # onnx_name = model_name + "_encoder.onnx"
    # torch.onnx.export(model,               # model being run
    #     inputs,                    # model input (or a tuple for multiple inputs)
    #     onnx_name,              # where to save the model (can be a file or file-like object)
    #     export_params=True,        # store the trained parameter weights inside the model file
    #     opset_version=16,          # the ONNX version to export the model to
    #     do_constant_folding=True,  # whether to execute constant folding for optimization
    #     input_names = input_names, # the model's input names
    #     output_names = output_names # the model's output names
    # )
    # sim_model,_ = simplify(onnx_name)
    # onnx.save(sim_model, onnx_name)

    # print(f"Exported model to {onnx_name}")

    # encoder_model = ort.InferenceSession(onnx_name, providers=['CPUExecutionProvider'])

    # input2_shape = generate_data(
    #     wav[None],
    #     models=[encoder_model],
    #     overlap=overlap,
    #     save_path="calibration_dataset",
    #     segment=seconds_split,
    #     max_num=max_num
    # )

    # # Export ONNX
    # model.forward = model.forward_decoder

    # input_names = ("in_x", "in_xt", "saved_0", "saved_1", "saved_2", "saved_3", "saved_t_0", "saved_t_1", "saved_t_2", "saved_t_3")
    # output_names = ("x", "xt")
    # segment_length = int(target_sr * seconds_split)
    # inputs = (
    #     torch.zeros(1,384,8,216, dtype=torch.float32), # in_x
    #     torch.zeros(1,384,862, dtype=torch.float32),
        
    #     torch.zeros(1,48,512,216, dtype=torch.float32),
    #     torch.zeros(1,96,128,216, dtype=torch.float32),
    #     torch.zeros(1,192,32,216, dtype=torch.float32),
    #     torch.zeros(1,384,8,216, dtype=torch.float32),

    #     torch.zeros(1,48,55125, dtype=torch.float32),
    #     torch.zeros(1,96,13782, dtype=torch.float32),
    #     torch.zeros(1,192,3446, dtype=torch.float32),
    #     torch.zeros(1,384,862, dtype=torch.float32),
    # )
    # onnx_name = model_name + "_decoder.onnx"
    # torch.onnx.export(model,               # model being run
    #     inputs,                    # model input (or a tuple for multiple inputs)
    #     onnx_name,              # where to save the model (can be a file or file-like object)
    #     export_params=True,        # store the trained parameter weights inside the model file
    #     opset_version=16,          # the ONNX version to export the model to
    #     do_constant_folding=True,  # whether to execute constant folding for optimization
    #     input_names = input_names, # the model's input names
    #     output_names = output_names # the model's output names
    # )
    # sim_model,_ = simplify(onnx_name)
    # onnx.save(sim_model, onnx_name)
    # print(f"Exported model to {onnx_name}")


if __name__ == "__main__":
    main()
