from demucs.pretrained import *
import torch
from torch import nn
from torch.nn import functional as F
from fractions import Fraction
from einops import rearrange
from onnxsim import simplify
import onnx


def forward_for_export(self, mix, mag):
    length = mix.shape[-1]
    length_pre_pad = None
    # if self.use_train_segment:
    #     if self.training:
    #         self.segment = Fraction(mix.shape[-1], self.samplerate)
    #     else:
    #         training_length = int(self.segment * self.samplerate)
    #         if mix.shape[-1] < training_length:
    #             length_pre_pad = mix.shape[-1]
    #             mix = F.pad(mix, (0, training_length - length_pre_pad))
    # z = self._spec(mix)
    # mag = self._magnitude(z).to(mix.device)
    x = mag

    B, C, Fq, T = x.shape

    # unlike previous Demucs, we always normalize because it is easier.
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) / (1e-5 + std)
    # x will be the freq. branch input.

    # Prepare the time branch input.
    xt = mix
    meant = xt.mean(dim=(1, 2), keepdim=True)
    stdt = xt.std(dim=(1, 2), keepdim=True)
    xt = (xt - meant) / (1e-5 + stdt)

    # okay, this is a giant mess I know...
    saved = []  # skip connections, freq.
    saved_t = []  # skip connections, time.
    lengths = []  # saved lengths to properly remove padding, freq branch.
    lengths_t = []  # saved lengths for time branch.
    for idx, encode in enumerate(self.encoder):
        lengths.append(x.shape[-1])
        inject = None
        if idx < len(self.tencoder):
            # we have not yet merged branches.
            lengths_t.append(xt.shape[-1])
            tenc = self.tencoder[idx]
            xt = tenc(xt)
            if not tenc.empty:
                # save for skip connection
                saved_t.append(xt)
            else:
                # tenc contains just the first conv., so that now time and freq.
                # branches have the same shape and can be merged.
                inject = xt
        x = encode(x, inject)
        if idx == 0 and self.freq_emb is not None:
            # add frequency embedding to allow for non equivariant convolutions
            # over the frequency axis.
            frs = torch.arange(x.shape[-2], device=x.device)
            emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
            x = x + self.freq_emb_scale * emb

        saved.append(x)
    if self.crosstransformer:
        if self.bottom_channels:
            b, c, f, t = x.shape
            x = rearrange(x, "b c f t-> b c (f t)")
            x = self.channel_upsampler(x)
            x = rearrange(x, "b c (f t)-> b c f t", f=f)
            xt = self.channel_upsampler_t(xt)

        x, xt = self.crosstransformer(x, xt)

        if self.bottom_channels:
            x = rearrange(x, "b c f t-> b c (f t)")
            x = self.channel_downsampler(x)
            x = rearrange(x, "b c (f t)-> b c f t", f=f)
            xt = self.channel_downsampler_t(xt)

    for idx, decode in enumerate(self.decoder):
        skip = saved.pop(-1)
        x, pre = decode(x, skip, lengths.pop(-1))
        # `pre` contains the output just before final transposed convolution,
        # which is used when the freq. and time branch separate.

        offset = self.depth - len(self.tdecoder)
        if idx >= offset:
            tdec = self.tdecoder[idx - offset]
            length_t = lengths_t.pop(-1)
            if tdec.empty:
                assert pre.shape[2] == 1, pre.shape
                pre = pre[:, :, 0]
                xt, _ = tdec(pre, None, length_t)
            else:
                skip = saved_t.pop(-1)
                xt, _ = tdec(xt, skip, length_t)

    # Let's make sure we used all stored skip connections.
    assert len(saved) == 0
    assert len(lengths_t) == 0
    assert len(saved_t) == 0

    S = len(self.sources)
    x = x.view(B, S, -1, Fq, T)
    x = x * std[:, None] + mean[:, None]

    return x, xt

    # to cpu as mps doesnt support complex numbers
    # demucs issue #435 ##432
    # NOTE: in this case z already is on cpu
    # TODO: remove this when mps supports complex numbers
    # x_is_mps = x.device.type == "mps"
    # if x_is_mps:
    #     x = x.cpu()

    # zout = self._mask(z, x)
    # x = self._ispec(zout, training_length)

    # # back to mps device
    # if x_is_mps:
    #     x = x.to("mps")

    # xt = xt.view(B, S, -1, training_length)

    # xt = xt * stdt[:, None] + meant[:, None]
    # x = xt + x
    # if length_pre_pad:
    #     x = x[..., :length_pre_pad]
    # return x


model = get_model(DEFAULT_MODEL).models[0]
model.forward = model.forward_for_export
model.eval()

input_names = ("mix", "mag")
output_names = ("x", "xt")
inputs = (
    torch.zeros(1,2,343980, dtype=torch.float32),
    torch.zeros(1,4,2048,336, dtype=torch.float32),
)
onnx_name = DEFAULT_MODEL + ".onnx"
torch.onnx.export(model,               # model being run
    inputs,                    # model input (or a tuple for multiple inputs)
    onnx_name,              # where to save the model (can be a file or file-like object)
    export_params=True,        # store the trained parameter weights inside the model file
    opset_version=16,          # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names = input_names, # the model's input names
    output_names = output_names # the model's output names
)
sim_model,_ = simplify(onnx_name)
onnx.save(sim_model, onnx_name)