import threading
import queue

import onnxruntime as ort
import numpy as np
import argparse
import os
import soundfile as sf
import librosa
from apply import run_model, TensorChunk, center_trim
import torch
from fractions import Fraction
from cal_demucs import *
import sounddevice as sd
import time


def wrapped_targetFunc(f, input_queue: queue.Queue, output_queue: queue.Queue, **kwargs):
    while True:
        try:
            work = input_queue.get(timeout=10)  # or whatever
        except queue.Empty:
            return
        out = f(work, **kwargs)
        input_queue.task_done()
        if output_queue is not None:
            output_queue.put_nowait(out)


def preprocess_target(work, segment, samplerate):
    mix, offset = work
    length = mix.shape[-1]
    mix = TensorChunk(mix)
    segment_length = int(segment * samplerate)
    padded_mix = mix.padded(segment_length)

    input1 = padded_mix.numpy()
    z = demucs_spec(padded_mix)
    mag = demucs_magnitude(z).to(padded_mix.device)
    input2 = mag.numpy()
    return (input1, input2, length, offset)


def run_target(work, model):
    input1, input2, length, offset = work
    padded_mix = th.from_numpy(input1)
    outputs = model.run(None, {"mix": input1, "mag": input2})
    x = th.from_numpy(outputs[0])
    xt = th.from_numpy(outputs[1])
    return (x, xt, padded_mix, length, offset)


def postprocess_target(work, segment, samplerate, ref_mean, ref_std):
    B = 1
    S = 4
    x, xt, padded_mix, length, offset = work
    out = demucs_post_process(x, xt, padded_mix, segment, samplerate, B, S)
    out = center_trim(out, length)
    out = out * ref_std + ref_mean
    return (out.numpy(), offset)


def play_target(work, samplerate, futures, seg_num, stride):
    # print(work.shape)    
    chunk_out, offset = work
    futures.append((chunk_out[0], offset))
    print(f"{len(futures)} / {seg_num}")
    source = chunk_out[0, -1] # vocal
    source = source / max(1.01 * np.abs(source).max(), 1)
    source = source[..., :stride]
    source = source.transpose()
    sd.play(source, samplerate=samplerate, blocking=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio", "-i", type=str, required=True, help="Input audio file(.wav)")
    parser.add_argument("--output_path", "-o", type=str, required=False, default="./output", help="Seperated wav path")
    parser.add_argument("--model", "-m", type=str, required=False, default="../models/htdemucs_ft.onnx", help="demucs onnx model")
    parser.add_argument("--overlap", type=float, required=False, default=0.25)
    return parser.parse_args()


def main():
    args = get_args()
    assert os.path.exists(args.input_audio), f"Input audio {args.input_audio} not exist"
    assert os.path.exists(args.model), f"Model {args.model} not exist"
    os.makedirs(args.output_path, exist_ok=True)

    input_audio = args.input_audio
    output_path = args.output_path
    model_path = args.model
    overlap = args.overlap

    target_sr = 44100
    segment = Fraction(39, 5)

    input_queue = queue.Queue()
    pre_queue = queue.Queue()
    run_queue = queue.Queue()
    post_queue = queue.Queue()

    print(f"Input audio: {input_audio}")
    print(f"Output path: {output_path}")
    print(f"Model: {model_path}")
    print(f"Overlap: {overlap}")

    print("Loading audio...")
    wav, origin_sr = sf.read(input_audio, always_2d=True, dtype="float32")
    if origin_sr != target_sr:
        print(f"Origin sample rate is {origin_sr}, resampling to {target_sr}...")
        wav = librosa.resample(wav, orig_sr=origin_sr, target_sr=target_sr)

    if wav.shape[0] != 2:
        wav = wav.transpose()
    # print(wav.shape)

    print("Loading model...")
    model = ort.InferenceSession(model_path)

    print("Preprocessing audio...")
    ref = wav.mean(0)
    ref_mean = ref.mean()
    ref_std = ref.std() + 1e-8
    wav -= ref_mean
    wav /= ref_std
    wav = torch.from_numpy(wav)[None, ...]
    
    batch, channels, length = wav.shape
    segment_length: int = int(target_sr * segment)
    stride = int((1 - overlap) * segment_length)
    futures = []
    seg_num = int(np.ceil(length / stride))

    print("Create threads")
    pre_thread = threading.Thread(target=wrapped_targetFunc, 
                                  args=(preprocess_target, input_queue, pre_queue),
                                  kwargs={"segment": segment, "samplerate": target_sr})
    run_thread = threading.Thread(target=wrapped_targetFunc,
                                  args=(run_target, pre_queue, run_queue),
                                  kwargs={"model": model})
    post_thread = threading.Thread(target=wrapped_targetFunc,
                                   args=(postprocess_target, run_queue, post_queue),
                                   kwargs={"segment": segment, "samplerate": target_sr, 
                                           "ref_mean": ref_mean, "ref_std": ref_std})
    play_thread = threading.Thread(target=wrapped_targetFunc,
                                   args=(play_target, post_queue, None),
                                   kwargs={"samplerate": target_sr, "futures": futures, 
                                           "seg_num": seg_num, "stride": stride})    

    for offset in range(0, length, stride):
        chunk = TensorChunk(wav, offset, segment_length)
        input_queue.put_nowait((chunk, offset))
        
    print("Run threads")
    pre_thread.start()
    run_thread.start()
    post_thread.start()
    play_thread.start()

    pre_thread.join()
    run_thread.join()
    post_thread.join()
    play_thread.join()

    out = th.zeros(batch, 4, channels, length)
    sum_weight = th.zeros(length)
    weight = th.cat([th.arange(1, segment_length // 2 + 1),
                        th.arange(segment_length - segment_length // 2, 0, -1)])
    weight = weight / weight.max()
    for future, offset in futures:
        chunk_out = future
        chunk_length = chunk_out.shape[-1]
        out[..., offset:offset + segment_length] += (weight[:chunk_length] * chunk_out)
        sum_weight[offset:offset + segment_length] += weight[:chunk_length]
    out /= sum_weight

    out = out.numpy()

    sources = ['drums', 'bass', 'other', 'vocals']
    res = dict(zip(sources, out[0]))
    print("Saving audio...")
    for name, source in res.items():
        source = source / max(1.01 * np.abs(source).max(), 1)
        if source.shape[1] != 2:
            source = source.transpose()
        audio_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(input_audio))[0]}_{name}.wav")
        sf.write(audio_path, source, samplerate=target_sr)
        print(f"Save {name} to {audio_path}")


if __name__ == "__main__":
    main()