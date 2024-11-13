import typing as tp
import torch as th
from torch.nn import functional as F
import tqdm
import torch
from fractions import Fraction
from cal_demucs import demucs_spec, demucs_magnitude, demucs_post_process


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

    chunk_index = 0
    for offset in tqdm.tqdm(range(0, length, stride)):
        chunk = TensorChunk(mix, offset, segment_length)
        future = run_model(model, chunk, device, samplerate, segment, chunk_index)
        futures.append((future, offset))
        offset += segment_length
        chunk_index += 1

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


def run_model(model, mix, device, samplerate, segment, chunk_index):
    """
    :param model:
    :param mix:
    :param device:
    :param samplerate:
    :param segment:
    :return:
    """
    length = mix.shape[-1]
    valid_length = int(segment * samplerate)
    mix = TensorChunk(mix)
    padded_mix = mix.padded(valid_length).to(device)

    input1 = padded_mix.numpy()
    z = demucs_spec(padded_mix)
    mag = demucs_magnitude(z).to(padded_mix.device)
    input2 = mag.numpy()

    input_names = model.get_inputs()
    output_names = model.get_outputs()
    inputs = {input_names[0]: input1, input_names[1]: input2}
    # outputs = model.run(output_names, inputs)
    outputs = model.run(inputs)

    # import os
    # os.makedirs("input", exist_ok=True)
    # os.makedirs("gt", exist_ok=True)
    # for i, name in enumerate(input_names):
    #     inputs[name].tofile(os.path.join("input", f"input{i}_{chunk_index}.bin"))
    # outputs[0].tofile(os.path.join("gt", f"output0_{chunk_index}.bin"))
    # outputs[1].tofile(os.path.join("gt", f"output1_{chunk_index}.bin"))

    x = th.from_numpy(outputs[output_names[0]])
    xt = th.from_numpy(outputs[output_names[1]])
    S = 4  # len(self.source)
    B, C, Fq, T = input2.shape
    out = demucs_post_process(x, xt, padded_mix, segment, samplerate, B, S)

    return center_trim(out, length)
