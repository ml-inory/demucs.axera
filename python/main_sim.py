import sys
sys.path.append("/data/yangrongzhao/Codes/npu-codebase/")
sys.path.append("/data/yangrongzhao/Codes/npu-codebase/simulator")
from sim import InferenceSession
import onnxruntime as ort
import numpy as np
import argparse
import os
import soundfile as sf
import librosa
from apply import apply_model
import torch
import time
from os import environ
from fractions import Fraction
from audio import AudioFile, save_audio
import glob

environ["OMP_NUM_THREADS"] = "8"
environ["OMP_WAIT_POLICY"] = 'ACTIVE'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio", "-i", type=str, required=True, help="Input audio file(.wav)")
    parser.add_argument("--output_path", "-o", type=str, required=False, default="./output", help="Seperated wav path")
    parser.add_argument("--model", "-m", type=str, required=False, default="../models/htdemucs.axmodel", help="demucs onnx model")
    parser.add_argument("--overlap", type=float, required=False, default=0.25)
    parser.add_argument("--segment", type=float, required=False, default=5, help="Split in seconds")
    return parser.parse_args()


def main():
    args = get_args()
    assert os.path.exists(args.input_audio), f"Input audio {args.input_audio} not exist"
    # assert os.path.exists(args.model), f"Model {args.model} not exist"
    os.makedirs(args.output_path, exist_ok=True)

    input_audio = args.input_audio
    output_path = args.output_path
    model_path = args.model
    segment = args.segment
    # segment = Fraction(39, 5)

    target_sr = 44100

    print(f"Input audio: {input_audio}")
    print(f"Output path: {output_path}")
    # print(f"Model: {model_path}")
    print(f"Overlap: {args.overlap}")

    assert os.path.exists(input_audio)
    if os.path.isfile(input_audio):
        input_audios = [input_audio]
    else:
        input_audios = glob.glob(input_audio + "/*")

    for input_audio in input_audios:
        print("Loading audio...")
        wav = AudioFile(input_audio).read(streams=0, samplerate=target_sr, channels=2)
        # wav, origin_sr = sf.read(input_audio, always_2d=True, dtype="float32")
        # if origin_sr != target_sr:
        #     print(f"Origin sample rate is {origin_sr}, resampling to {target_sr}...")
        #     wav = librosa.resample(wav, orig_sr=origin_sr, target_sr=target_sr)
        # if wav.shape[0] != 2:
        #     wav = wav.transpose()
        # print(wav.shape)

        print("Loading model...")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 设置线程数
        sess_options.intra_op_num_threads = 8  # 设置计算线程数
        sess_options.inter_op_num_threads = 8  # 设置并行任务线程数


        # sess = [
        #     ort.InferenceSession("./models/htdemucs_encoder_freq.ort", sess_options, providers=['CPUExecutionProvider']), 
        #     InferenceSession.load_from_model("./models/htdemucs_time_v6.axmodel"), 
        #     MNNWrapper("./models/htdemucs_decoder_freq.mnn"),
        #     MNNWrapper("./models/htdemucs_decoder_time.mnn"),
        # ]
        sess = InferenceSession("../model_convert/htdemucs/quant/quant_axmodel.onnx")
        # sess = ort.InferenceSession("../models/apollo_sim.onnx", sess_options, providers=['CPUExecutionProvider'])


        print("Preprocessing audio...")
        start = time.time()
        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std() + 1e-8
        # wav = torch.from_numpy(wav)
        print(f"preprocess audio take {1000 * (time.time() - start)}ms")

        print("Running model...")
        out = apply_model(
            wav[None],
            overlap=args.overlap,
            model=sess,
            segment=segment
        )

        print("Postprocessing...")
        out *= ref.std() + 1e-8
        out += ref.mean()
        # wav *= ref.std() + 1e-8
        # wav += ref.mean()

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