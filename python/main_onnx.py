import onnxruntime as ort
import numpy as np
import argparse
import os
import soundfile as sf
import librosa
from apply import apply_model
import torch


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

    target_sr = 44100

    print(f"Input audio: {input_audio}")
    print(f"Output path: {output_path}")
    print(f"Model: {model_path}")
    print(f"Overlap: {args.overlap}")

    print("Loading audio...")
    wav, origin_sr = sf.read(input_audio, always_2d=True, dtype="float32")
    if origin_sr != target_sr:
        print(f"Origin sample rate is {origin_sr}, resampling to {target_sr}...")
        wav = librosa.resample(wav, orig_sr=origin_sr, target_sr=target_sr)

    if wav.shape[0] != 2:
        wav = wav.transpose()
    # print(wav.shape)

    print("Loading model...")
    sess = ort.InferenceSession(model_path)

    print("Preprocessing audio...")
    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std() + 1e-8
    wav = torch.from_numpy(wav)

    print("Running model...")
    out = apply_model(
        wav[None],
        overlap=args.overlap,
        model=sess,
    )

    print("Postprocessing...")
    out *= ref.std() + 1e-8
    out += ref.mean()
    wav *= ref.std() + 1e-8
    wav += ref.mean()

    out = out.numpy()
    print(out.shape)

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