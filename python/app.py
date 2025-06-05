import gradio as gr
import numpy as np
import soundfile as sf
import io
import os
from demucs_cls import Demucs

demucs = Demucs('../models/htdemucs.axmodel')
print("Load model finish")

def cleanup_temp_files(files):
    for file in files:
        if os.path.exists(file):
            if os.path.isdir(file):
                os.system(f"rm -rf {file}")
            else:
                os.remove(file)


def process_audio(input_file, pr=gr.Progress(track_tqdm=True)):
    global demucs

    output_path = 'output'
    cleanup_temp_files([output_path])

    print("Running demucs")
    output_files = demucs.run(input_file, output_path=output_path)
    print(output_files)

    return [gr.Audio(output_files[0], type='filepath', sources=None),
            gr.Audio(output_files[1], type='filepath', sources=None),
            gr.Audio(output_files[2], type='filepath', sources=None),
            gr.Audio(output_files[3], type='filepath', sources=None)]



with gr.Blocks() as demo:
    gr.Markdown("## Demucs音轨分离demo")
    gr.Markdown("上传一个 WAV 文件，模型将其分为drums、bass、other、vocal四轨，对应四种乐器")
    
    audio_input = gr.Audio(type="filepath", label="上传 WAV 文件")
    
    with gr.Tab("Drums"):
        drums_audio = gr.Audio(type="filepath", label="drums")

    with gr.Tab("Bass"):
        bass_audio = gr.Audio(type="filepath", label="bass")

    with gr.Tab("Other"):
        other_audio = gr.Audio(type="filepath", label="other")

    with gr.Tab("Vocals"):
        vocals_audio = gr.Audio(type="filepath", label="vocals")

    submit_btn = gr.Button("处理音频")
    
    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[drums_audio, bass_audio, other_audio, vocals_audio]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")