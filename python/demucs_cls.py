from axengine import InferenceSession
import numpy as np
import argparse
import os
import soundfile as sf
import librosa
from apply import apply_model
import torch


class Demucs:
    def __init__(self, model_path, overlap=0.25, segment=5):
        self.model = InferenceSession.load_from_model(model_path)
        self.overlap = overlap
        self.segment = segment

    def run(self, input_audio, target_sr=44100, output_path='output'):
        os.makedirs(output_path, exist_ok=True)

        wav, origin_sr = sf.read(input_audio, always_2d=True, dtype="float32")
        if origin_sr != target_sr:
            print(f"Origin sample rate is {origin_sr}, resampling to {target_sr}...")
            wav = librosa.resample(wav, orig_sr=origin_sr, target_sr=target_sr)
        if wav.shape[0] != 2:
            wav = wav.transpose()
        # print(wav.shape)

        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std() + 1e-8
        wav = torch.from_numpy(wav)

        out = apply_model(
            wav[None],
            overlap=self.overlap,
            model=self.model,
            segment=self.segment
        )

        out *= ref.std() + 1e-8
        out += ref.mean()
        out = out.numpy()

        sources = ['drums', 'bass', 'other', 'vocals']
        res = dict(zip(sources, out[0]))
        audio_files = []
        for name, source in res.items():
            source = source / max(1.01 * np.abs(source).max(), 1)
            
            if source.shape[1] != 2:
                source = source.transpose()

            audio_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(input_audio))[0]}_{name}.wav")
            sf.write(audio_path, source, samplerate=target_sr)
            audio_files.append(audio_path)
        return audio_files
