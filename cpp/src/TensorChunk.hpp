#pragma once

#include <vector>
#include <algorithm>

#include "common.h"


class TensorChunk {
public:
    TensorChunk(AUDIO_DATA& tensor_, int offset_, int length_):
        tensor(tensor_),
        offset(offset_) {
            int total_length = tensor_[0].size();
            length = std::min(total_length - offset, length_);
    }

    AUDIO_DATA padded(int target_length) {
        int delta = target_length - length;
        int total_length = tensor[0].size();

        int start = offset - delta / 2;
        int end = start + target_length;

        int correct_start = std::max(0, start);
        int correct_end = std::min(total_length, end);

        int pad_left = correct_start - start;
        int pad_right = end - correct_end;

        AUDIO_DATA out(tensor.size());
        for (size_t i = 0; i < tensor.size(); i++) {
            out[i].resize(pad_left + pad_right, 0);
            out[i].insert(out[i].begin() + pad_left, tensor[i].begin() + correct_start, tensor[i].begin() + correct_end);
        }

        return out;
    }

public:
    int offset, length;
    AUDIO_DATA& tensor;
};