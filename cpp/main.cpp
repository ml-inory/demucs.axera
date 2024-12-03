#include <cstdio>
#include <ax_sys_api.h>
#include <ax_engine_api.h>
#include <vector>
#include <cmath>
#include <filesystem>

#include "cmdline.h"
#include "EngineWrapper.hpp"
#include "utilities/file.hpp"
#include "utilities/timer.hpp"
#include "AudioFile.h"
#include "librosa.h"
#include "TensorChunk.hpp"
#include "cal_demucs.hpp"


std::vector<std::vector<float>> preprocess_audio(const std::vector<std::vector<float>>& samples, float& ref_mean, float& ref_std) {
    int channels = samples.size();
    int num_samples = samples[0].size();
    std::vector<float> ref(num_samples, 0);
    std::vector<std::vector<float>> res(channels, std::vector<float>(num_samples));

    // wav -= wav.mean(0)
    if (channels == 1) {
        // mono
        ref = samples[0];
    } else {
        for (int i = 0; i < num_samples; i++) {
            ref[i] = (samples[0][i] + samples[1][i]) / 2.0f;
        }
    }

    // ref_mean
    float ref_sum = 0;
    for (int i = 0; i < num_samples; i++) {
        ref_sum += ref[i];
    }
    ref_mean = ref_sum / num_samples;

    // ref_std
    ref_std = 0;
    for (int i = 0; i < num_samples; i++) {
        ref_std += (ref[i] - ref_mean) * (ref[i] - ref_mean);
    }
    ref_std = sqrtf(ref_std / num_samples);

    for (int i = 0; i < num_samples; i++) {
        res[0][i] = (samples[0][i] - ref_mean) / ref_std + 1e-8;
        res[1][i] = (samples[1][i] - ref_mean) / ref_std + 1e-8;
    }
    return res;
}

void normalize_audio(AUDIO_DATA& audio_data) {
    float max_num = 1.0f;
    int num_channels = audio_data.size();
    int num_samples = audio_data[0].size();
    for (int i = 0; i < num_channels; i++) {
        for (int n = 0; n < num_samples; n++) {
            float val = std::fabs(audio_data[i][n]);
            if (val > max_num) {
                max_num = val;
            }
        }
    }
    for (int i = 0; i < num_channels; i++) {
        for (int n = 0; n < num_samples; n++) {
            audio_data[i][n] /= 1.01 * max_num;
        }
    }
}

int main(int argc, char** argv) {
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "demucs axmodel", false, "../models/htdemucs_ft.axmodel");
    cmd.add<std::string>("input", 'i', "input audio file", true, "");
    cmd.add<std::string>("output", 'o', "output path", false, "output");
    cmd.add<float>("overlap", 0, "segment overlap ratio", false, 0.25f);
    cmd.parse_check(argc, argv);

    // 0. get app args, can be removed from user's app
    auto model_path = cmd.get<std::string>("model");
    auto input_audio_file = cmd.get<std::string>("input");
    auto output_path = cmd.get<std::string>("output");
    auto overlap = cmd.get<float>("overlap");

    printf("model_path: %s\n", model_path.c_str());
    printf("input_audio: %s\n", input_audio_file.c_str());
    printf("output_path: %s\n", output_path.c_str());
    printf("overlap: %.2f\n", overlap);

    // check file existence
    if (!utilities::exists(model_path)) {
        printf("model %s not exist!\n", model_path.c_str());
        return -1;
    }
    if (!utilities::exists(input_audio_file)) {
        printf("audio %s not exist!\n", input_audio_file.c_str());
    }
    if (!utilities::exists(output_path)) {
        if (!utilities::create_directory(output_path)) {
            printf("Create output path %s failed!\n", output_path.c_str());
            return -1;
        }
    }

    // init system
    int ret = AX_SYS_Init();
    if (0 != ret) {
        fprintf(stderr, "AX_SYS_Init failed! ret = 0x%x\n", ret);
        return -1;
    }

    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = static_cast<AX_ENGINE_NPU_MODE_T>(0);
    ret = AX_ENGINE_Init(&npu_attr);
    if (0 != ret) {
        fprintf(stderr, "Init ax-engine failed{0x%8x}.\n", ret);
        return -1;
    }

    // timer
    utilities::timer timer;

    // load model
    timer.start();
    EngineWrapper model;
    if (0 != model.Init(model_path.c_str())) {
        printf("Init model %s failed!\n", model_path.c_str());
        return -1;
    }
    timer.stop();
    printf("Load model take %.2fms\n", timer.elapsed<utilities::timer::milliseconds>());

    // load audio
    timer.start();
    AudioFile<float> audio_file;
    if (!audio_file.load(input_audio_file)) {
        printf("Load audio %s failed!\n", input_audio_file.c_str());
        return -1;
    }
    timer.stop();
    printf("Load audio take %.2fms\n", timer.elapsed<utilities::timer::milliseconds>());

    auto& t_samples = audio_file.samples;
    float ref_mean, ref_std;
    timer.start();
    auto samples = preprocess_audio(t_samples, ref_mean, ref_std);
    timer.stop();
    printf("Preprocess audio take %.2fms\n", timer.elapsed<utilities::timer::milliseconds>());

    const int samplerate = 44100;
    int channels = samples.size();
    int length = samples[0].size();
    int segment_length = samplerate * 39 / 5;
    int stride = (1 - overlap) * segment_length;

    int x_size = model.GetOutputSize(0) / sizeof(float);
    int xt_size = model.GetOutputSize(1) / sizeof(float);
    std::vector<float> x(x_size);
    std::vector<float> xt(xt_size);
    std::vector<std::pair<AUDIO_DATA, int>> futures;
    int chunk_index = 0;
    for (int offset = 0; offset < length; offset += stride) {
        TensorChunk chunk(samples, offset, segment_length);

        // preprocess
        timer.start();
        auto padded_mix = chunk.padded(segment_length);
        auto z = demucs_spec(padded_mix);
        auto mag = demucs_magnitude(z);
        timer.stop();
        printf("Preprocess chunk take %.2fms\n", timer.elapsed<utilities::timer::milliseconds>());

        // run
        timer.start();
        model.SetInput(padded_mix.data(), 0);
        model.SetInput(mag.data(), 1);
        if (0 != model.RunSync()) {
            printf("model run failed!\n");
            return -1;
        }
        model.GetOutput(x.data(), 0);
        model.GetOutput(xt.data(), 1);
        timer.stop();
        printf("Run take %.2fms\n", timer.elapsed<utilities::timer::milliseconds>());

        // postprocess
        auto out = demucs_post_process(x, xt, padded_mix, segment_length, 1, 4, chunk.length);

        futures.push_back(std::make_pair(out, offset));

        printf("%d/%d\n", ++chunk_index, int(ceilf(length * 1.0f / stride)));
    }

    // apply weight

    // pack data
    AUDIO_DATA drums(2), bass(2), other(2), vocals(2);
    for (auto& f : futures) {
        // (S * 2, length)
        AUDIO_DATA& x = f.first;
        // drums
        drums[0].insert(drums[0].end(), x[0].begin(), x[0].begin() + stride);
        drums[1].insert(drums[1].end(), x[1].begin(), x[1].begin() + stride);
        // bass
        bass[0].insert(bass[0].end(), x[2].begin(), x[2].begin() + stride);
        bass[1].insert(bass[1].end(), x[3].begin(), x[3].begin() + stride);
        // other
        other[0].insert(other[0].end(), x[4].begin(), x[4].begin() + stride);
        other[1].insert(other[1].end(), x[5].begin(), x[5].begin() + stride);
        // vocals
        vocals[0].insert(vocals[0].end(), x[6].begin(), x[6].begin() + stride);
        vocals[1].insert(vocals[1].end(), x[7].begin(), x[7].begin() + stride);
    }

    // * std + mean
    int wav_len = drums[0].size();
    for (int i = 0; i < channels; i++) {
        for (int n = 0; n < wav_len; n++) {
            drums[i][n] = (drums[i][n] * ref_std + 1e-8) + ref_mean;
            bass[i][n] = (bass[i][n] * ref_std + 1e-8) + ref_mean;
            other[i][n] = (other[i][n] * ref_std + 1e-8) + ref_mean;
            vocals[i][n] = (vocals[i][n] * ref_std + 1e-8) + ref_mean;
        }
    }

    // normalize
    normalize_audio(drums);
    normalize_audio(bass);
    normalize_audio(other);
    normalize_audio(vocals);

    // save audio
    std::string basename = std::filesystem::path(input_audio_file).stem().string();
    std::string drum_path = output_path + "/" + basename + "_drums.wav";
    std::string bass_path = output_path + "/" + basename + "_bass.wav";
    std::string other_path = output_path + "/" + basename + "_other.wav";
    std::string vocals_path = output_path + "/" + basename + "_vocals.wav";

    AudioFile<float> output_audio_file;
    output_audio_file.setBitDepth(32);
    output_audio_file.setSampleRate(44100);

    output_audio_file.setAudioBuffer(drums);
    output_audio_file.save(drum_path);

    output_audio_file.setAudioBuffer(bass);
    output_audio_file.save(bass_path);

    output_audio_file.setAudioBuffer(other);
    output_audio_file.save(other_path);

    output_audio_file.setAudioBuffer(vocals);
    output_audio_file.save(vocals_path);


    return 0;
}