import typing as tp
import torch as th
from torch.nn import functional as F
import tqdm
import torch
from fractions import Fraction
from cal_demucs import demucs_spec, demucs_magnitude, demucs_post_process
import onnxruntime as ort
import numpy as np
import time
import sys

# from mnn_wrapper import MNNWrapper


def center_trim(tensor: torch.Tensor, reference: tp.Union[torch.Tensor, int]):
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


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


def apply_model(mix,
                overlap: float = 0.25,
                device: tp.Union[str, th.device] = 'cpu',
                transition_power: float = 1.,
                len_model_sources=4,
                segment=Fraction(39, 5),
                samplerate=44100,
                model=None
                ) -> th.Tensor:
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
    futures = []

    for offset in tqdm.tqdm(range(0, length, stride)):
        chunk = TensorChunk(mix, offset, segment_length)
        future = run_model(model, chunk, device, samplerate, segment)
        futures.append((future, offset))
        offset += segment_length

    out = th.zeros(batch, len_model_sources, channels, length, device=mix.device)
    sum_weight = th.zeros(length, device=mix.device)
    weight = th.cat([th.arange(1, segment_length // 2 + 1, device=device),
                        th.arange(segment_length - segment_length // 2, 0, -1, device=device)])
    weight = (weight / weight.max())**transition_power
    for future, offset in futures:
        chunk_out = future
        chunk_length = chunk_out.shape[-1]
        out[..., offset:offset + segment_length] += (weight[:chunk_length] * chunk_out).to(mix.device)
        sum_weight[offset:offset + segment_length] += weight[:chunk_length].to(mix.device)
    out /= sum_weight

    for k, inst_weight in enumerate(model_weights):
        out[:, k, :, :] *= inst_weight
        totals[k] += inst_weight
    for k in range(out.shape[1]):
        out[:, k, :, :] /= totals[k]
    return out


index = 0
def run_model(model, mix, device, samplerate, segment):
    """
    :param model:
    :param mix:
    :param device:
    :param samplerate:
    :param segment:
    :return:
    """
    global index

    length = mix.shape[-1]
    valid_length = int(segment * samplerate)
    mix = TensorChunk(mix)
    padded_mix = mix.padded(valid_length).to(device)

    # import time
    start = time.time()
    input1 = padded_mix.numpy()
    z = demucs_spec(padded_mix)
    # print(f"z.shape = {z.shape}")
    mag = demucs_magnitude(z).to(padded_mix.device)
    # print(f"mag.shape = {mag.shape}")
    input2 = mag.numpy()
    print(f"preprocess take {time.time() - start}s")

    input1 = (input1 - input1.mean()) / (input1.std() + 1e-5)

    mean_mag = input2.mean()
    std_mag = input2.std()
    
    input2 = (input2 - mean_mag) / (std_mag + 1e-5)

    # input1.tofile(f"sim/mix/{index}.bin")
    # input2.tofile(f"sim/mag/{index}.bin")
    
    if isinstance(model, list):
        pass
        # if len(model) == 4:
        #     model_encoder_freq, model_time, model_decoder_freq, model_decoder_time = model
        # elif len(model) == 3:
        #     model_encoder_freq, model_time, model_decoder_freq = model

        # start = time.time()
        # if isinstance(model_encoder_freq, ort.InferenceSession):
        #     saved_0, saved_1, saved_2, saved_3 = model_encoder_freq.run(output_names=None, input_feed={"mag": input2})
        #     x = saved_3
        # # elif isinstance(model_encoder_freq, NCNNWrapper):
        # #     saved_0, saved_1, saved_2, saved_3 = model_encoder_freq.run(["saved_0", "saved_1", "saved_2", "saved_3"], {"mag": input2})
        # #     x = saved_3
        # elif isinstance(model_encoder_freq, MNNWrapper):
        #     outputs = model_encoder_freq.run({"mag": input2})
        #     saved_0 = outputs["saved_0"]
        #     saved_1 = outputs["saved_1"] 
        #     saved_2 = outputs["saved_2"] 
        #     saved_3 = outputs["saved_3"]
        #     x = saved_3
        # else:
        #     saved_0, saved_1, saved_2, saved_3 = model_encoder_freq.run(input_feed={"mag": input2})
        #     x = saved_3
        # print(f"model_encoder_freq take {time.time() - start}s")

        # start = time.time()
        # if isinstance(model_time, ort.InferenceSession):
        #     outputs_quant = model_time.run(output_names=None, input_feed={"mix": input1, "in_x": x, "saved_1": saved_1, "saved_2": saved_2, "saved_3": saved_3})
        # else:
        #     outputs_quant = model_time.run(input_feed={"mix": input1, "in_x": x, })
        # print(f"model_time take {time.time() - start}s")

        # if isinstance(model_time, ort.InferenceSession):
        #     x, xt = outputs_quant
        # else:
        #     x = outputs_quant["x"]
        #     xt = outputs_quant["xt"]
        #     saved_t_0 = outputs_quant["saved_t_0"]
        #     saved_t_1 = outputs_quant["saved_t_1"]
        #     saved_t_2 = outputs_quant["saved_t_2"]
        #     saved_t_3 = outputs_quant["saved_t_3"]

        # start = time.time()
        # if isinstance(model_decoder_freq, ort.InferenceSession):
        #     # print(f"x.shape = {x.shape}")
        #     # print(f"saved_0.shape = {saved_0.shape}")
        #     # print(f"saved_1.shape = {saved_1.shape}")
        #     # print(f"saved_2.shape = {saved_2.shape}")
        #     # print(f"saved_3.shape = {saved_3.shape}")
        #     outputs = model_decoder_freq.run(output_names=None, input_feed={"in_x": x, "saved_0": saved_0, "saved_1": saved_1, "saved_2": saved_2, "saved_3": saved_3})
        #     x = outputs[0]
        # # elif isinstance(model_decoder_freq, NCNNWrapper):
        # #     outputs = model_decoder_freq.run(["x"], {"in_x": x, "saved_0": saved_0, "saved_1": saved_1, "saved_2": saved_2, "saved_3": saved_3})
        # #     x = outputs[0]
        # elif isinstance(model_decoder_freq, MNNWrapper):
        #     outputs = model_decoder_freq.run({"saved_0": saved_0, "saved_1": saved_1, "saved_2": saved_2, "saved_3": saved_3})
        #     x = outputs["x"]
        # else:
        #     outputs = model_decoder_freq.run(input_feed={"in_x": x, "saved_0": saved_0,})
        #     x = outputs["x"]
        # print(f"model_decoder_freq take {time.time() - start}s")

        # start = time.time()
        # if isinstance(model_decoder_time, ort.InferenceSession):
        #     outputs = model_decoder_time.run(output_names=None, input_feed={"in_xt": xt, 
        #                                                                     "saved_t_0": saved_t_0, 
        #                                                                     "saved_t_1": saved_t_1, 
        #                                                                     "saved_t_2": saved_t_2, 
        #                                                                     "saved_t_3": saved_t_3, 
        #                                                 })
        #     xt = outputs[0]
        # # elif isinstance(model_decoder_time, NCNNWrapper):
        # #     outputs = model_decoder_time.run(["xt"], {"in_xt": xt, 
        # #                                                                     "saved_t_0": saved_t_0, 
        # #                                                                     "saved_t_1": saved_t_1, 
        # #                                                                     "saved_t_2": saved_t_2, 
        # #                                                                     "saved_t_3": saved_t_3, 
        # #                                                 })
        # #     xt = outputs[0]
        # elif isinstance(model_decoder_time, MNNWrapper):
        #     outputs = model_decoder_time.run({
        #                                     "saved_t_0": saved_t_0, 
        #                                     "saved_t_1": saved_t_1, 
        #                                     "saved_t_2": saved_t_2, 
        #                                     "saved_t_3": saved_t_3
        #                                     })
        #     xt = outputs["xt"]

        # print(f"model_decoder_time take {time.time() - start}s")
    else:
        if isinstance(model, ort.InferenceSession):
            outputs = model.run(None, {"mix": input1, "mag": input2})
            x, xt = outputs
        else:
            outputs = model.run({"mix": input1, "mag": input2})
            x = outputs["x"]
            xt = outputs["xt"]
        
    x = th.from_numpy(x)
    xt = th.from_numpy(xt)

    # x = th.from_numpy(np.load(f"x_{index}.npy")) # x有梳子噪声
    # xt = th.from_numpy(np.load(f"xt_{index}.npy"))

    # np.save(f"test_data/input/mix/{index}.npy", input1)
    # np.save(f"test_data/input/mag/{index}.npy", input2)
    # np.save(f"test_data/output/x/{index}.npy", x.numpy())
    # np.save(f"xt_gt/{index}.npy", xt.numpy())
    # if index == 0:
    #     input1.tofile(f"../model_convert/test_data/input/mix.bin")
    #     input2.tofile(f"../model_convert/test_data/input/mag.bin")
    #     x.numpy().tofile(f"../model_convert/test_data/output/x.bin")
    #     xt.numpy().tofile(f"../model_convert/test_data/output/xt.bin")

    # np.save(f"mix_quant_{index}.npy", input1)
    # np.save(f"mag_quant_{index}.npy", input2)
    # np.save(f"x_quant_{index}.npy", x.numpy())
    # np.save(f"xt_quant_{index}.npy", xt.numpy())
        
    S = 4  # len(self.source)
    B, C, Fq, T = input2.shape

    x = x.view(B, S, -1, Fq, T)
    x = x * std_mag + mean_mag

    index += 1

    start = time.time()
    out = demucs_post_process(x, xt, padded_mix, segment, samplerate, B, S)
    print(f"post_process take {time.time() - start}s")

    return center_trim(out, length)
