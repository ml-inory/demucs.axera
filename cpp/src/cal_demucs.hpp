#pragma once

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <vector>
#include "fft.h"
#include "common.h"
#include "utilities/timer.hpp"
#include "librosa.h"


typedef std::vector<std::complex<float>>    VectorComplex;
typedef std::vector<std::vector<std::complex<float>>>   MatrixComplex;


static AUDIO_DATA pad1d(const AUDIO_DATA& x, int padding_left, int padding_right) {
    int max_pad = std::max(padding_left, padding_right);
    AUDIO_DATA src = x;
    int channels = src.size();
    int length = x[0].size();
    if (length <= max_pad) {
        int extra_pad = max_pad - length + 1;
        int extra_pad_right = std::min(padding_right, extra_pad);
        int extra_pad_left = extra_pad - extra_pad_right;
        // pad zero
        int padded_len = extra_pad_left + extra_pad_right + length;
        for (int i = 0; i < channels; i++) {
            src[i].resize(extra_pad_left + extra_pad_right);
            std::fill(src[i].begin(), src[i].end(), 0);
            src[i].insert(src[i].begin() + extra_pad_left, x[i].begin(), x[i].end());
        }
    }
    int out_len = padding_left + padding_right + length;
    AUDIO_DATA out;
    out.resize(channels);

    // reflect
    for (int i = 0; i < channels; i++) {
        out[i].resize(out_len);
        // left
        for (int n = 0; n < padding_left; n++) {
            out[i][n] = src[i][padding_left - n];
        }
        // origin
        memcpy(out[i].data() + padding_left, src[i].data(), sizeof(float) * src[i].size());
        // right
        for (int n = 0; n < padding_right; n++) {
            out[i][padding_left + length + n] = src[i][length - n - 2];
        }        
    }

    return out;
}


static std::vector<Fft::mat_complex> spectro(AUDIO_DATA& x, int n_fft, int hop_length) {
    std::vector<Fft::mat_complex> z;
    int channels = x.size();
    for (int i = 0; i < channels; i++) {
        z.push_back(librosa::Feature::stft(x[i], n_fft, hop_length, "hann", true, "reflect", true));
    }
    return z;
}


static std::vector<Fft::mat_complex> demucs_spec(AUDIO_DATA& x, int nfft=4096) {
    int hop_length = nfft / 4;
    int length = x[0].size();
    int le = int(std::ceil(length * 1.0 / hop_length));
    int pad = hop_length / 2 * 3;
    auto pad_x = pad1d(x, pad, pad + le * hop_length - length);

    // z = spectro(x, nfft, hl)[..., :-1, :]
    // z = z[..., 2: 2 + le]
    auto z = spectro(pad_x, nfft, hop_length);
    int channels = z.size();
    for (int i = 0; i < channels; i++) {
        z[i].erase(z[i].end() - 1);
        for (int k = 0; k < z[i].size(); k++) {
            Fft::vec_complex new_z(z[i][k].begin() + 2, z[i][k].begin() + 2 + le);
            z[i][k] = new_z;
        }
    }
    return z;
}


static std::vector<float> demucs_magnitude(const std::vector<Fft::mat_complex>& z) {
    // B, C, Fr, T = z.shape
    // m = th.view_as_real(z).permute(0, 1, 4, 2, 3)
    // m = m.reshape(B, C * 2, Fr, T)
    int channels = z.size();
    int Fr = z[0].size();
    int T = z[0][0].size();

    std::vector<float> m(channels * 2 * Fr * T);
    for (int c = 0; c < channels; c++) {
        for (int f = 0; f < Fr; f++) {
            for (int t = 0; t < T; t++) {
                m[c * 2 * Fr * T + f * T + t] = z[c][f][t].real();
                m[(c * 2 + 1) * Fr * T + f * T + t] = z[c][f][t].imag();
            }
        }
    }
    return m;
}

static std::vector<MatrixComplex> demucs_mask(const std::vector<float>& m) {
    const int B = 1;
    const int S = 4;
    const int Fr = 2048;
    const int T = 336;
    std::vector<MatrixComplex> result(B * S * 2, MatrixComplex(Fr, VectorComplex(T)));
    // view as (B, S, 2, 2, Fr, T)
    // Permute to (B, S, 2, Fr, T, 2)
    for (int s = 0; s < S * 2; s++) {
        for (int f = 0; f < Fr; f++) {
            for (int t = 0; t < T; t++) {
                float real_part = m[s * Fr * T + f * T + t];;
                float imag_part = m[s * Fr * T + f * T + t + Fr * T];
                result[s][f][t] = std::complex<float>(real_part, imag_part);
            }
        }
    }
    return result;
}

static std::vector<std::vector<float>> demucs_ispectro(std::vector<MatrixComplex>& z, int hop_length, int length) {
    const int freqs = 2049;
    int n_fft = 2 * freqs - 2;
    int win_length = n_fft;
    int batch_size = z.size();
    std::vector<std::vector<float>> x;
    for (int i = 0; i < batch_size; i++) {
        x.push_back(librosa::Feature::istft(z[i], n_fft, hop_length, "hann", true, "reflect", true));
    }
        
    return x;
}

static std::vector<std::vector<float>> demucs_ispec(const std::vector<MatrixComplex>& z, int length, int nfft=4096) {
    // hop_length = nfft // 4
    // hl = hop_length // (4 ** scale)
    // z = F.pad(z, (0, 0, 0, 1))
    // z = F.pad(z, (2, 2))
    // pad = hl // 2 * 3
    // le = hl * int(math.ceil(length / hl)) + 2 * pad
    // x = demucs_ispectro(z, hl, length=le)
    // x = x[..., pad: pad + length]
    // return x
    int hop_length = nfft / 4;
    const int S = 4;
    const int Fr = 2048;
    const int T = 336;
    std::vector<MatrixComplex> batch_z(S * 2, MatrixComplex(Fr + 1, VectorComplex(T + 4)));
    for (int n = 0; n < S * 2; n++) {
        for (int i = 0; i < Fr; i++) {
            std::copy(z[n][i].begin(), z[n][i].end(), batch_z[n][i].begin() + 2);
        }
    }

    int pad = hop_length / 2 * 3;
    int le = hop_length * int(ceilf(length * 1.0f / hop_length)) + 2 * pad;
    auto x = demucs_ispectro(batch_z, hop_length, le);
    return x;
}

static std::vector<std::vector<float>> center_trim(std::vector<std::vector<float>>& x, int length) {
    int delta = x[0].size() - length;
    int start = delta / 2;
    int end = x[0].size() - (delta - delta / 2);
    std::vector<std::vector<float>> res;
    for (int i = 0; i < x.size(); i++) {
        res.push_back(std::vector<float>(x[i].begin() + start, x[i].begin() + end));
    }
    return res;
}
    

static std::vector<std::vector<float>> demucs_post_process(const std::vector<float>& m, 
    const std::vector<float>& xt, 
    const std::vector<std::vector<float>>& padded_mix, 
    int segment_length, int B, int S, int trim_len) {

    utilities::timer timer;
    timer.start();
    auto zout = demucs_mask(m);
    timer.stop();
    printf("demucs_mask take %.2fms\n", timer.elapsed<utilities::timer::milliseconds>());

    timer.start();
    auto x = demucs_ispec(zout, segment_length);
    timer.stop();
    printf("demucs_ispec take %.2fms\n", timer.elapsed<utilities::timer::milliseconds>());
    return center_trim(x, trim_len);
}