#include <cstdio>
#include <ax_sys_api.h>
#include <ax_engine_api.h>
#include <vector>
#include <cmath>
#include <filesystem>
#include <algorithm>

#include "cmdline.h"
#include "EngineWrapper.hpp"
#include "utilities/file.hpp"
#include "utilities/timer.hpp"
#include "AudioFile.h"
#include "librosa.h"
#include "TensorChunk.hpp"
#include "cal_demucs.hpp"
#include "data.hpp"
#include <iostream>
#include <fstream>

int main(int argc, char** argv) {
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "demucs axmodel", false, "../models/apollo.axmodel");
    cmd.add<std::string>("model_output", 'o', "output file", false, "model_output.txt");
    cmd.add<float>("segment", 0, "segment length in seconds", false, 2.08f);
    cmd.parse_check(argc, argv);

    // 0. get app args, can be removed from user's app
    auto model_path = cmd.get<std::string>("model");
    auto output_path = cmd.get<std::string>("model_output");
    auto segment = cmd.get<float>("segment");

    printf("model_path: %s\n", model_path.c_str());
    printf("segment: %.2f\n", segment);

    // check file existence
    if (!utilities::exists(model_path)) {
        printf("model %s not exist!\n", model_path.c_str());
        return -1;
    }

    // 初始化系统
    int ret = AX_SYS_Init();
    if (0 != ret) {
        fprintf(stderr, "AX_SYS_Init failed! ret = 0x%x\n", ret);
        return -1;
    }

    // 初始化axengine
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

    // 加载模型
    timer.start();
    EngineWrapper model;
    if (0 != model.Init(model_path.c_str())) {
        printf("Init model %s failed!\n", model_path.c_str());
        return -1;
    }
    timer.stop();
    printf("Load model take %.2fms\n", timer.elapsed<utilities::timer::milliseconds>());

    // 获取输入信息
    for (uint32_t i = 0; i < model.GetInputNum(); ++i) {
        // 获取输入名称
        std::string name;
        if (0 > model.GetInputName(name, i)) {
            printf("GetInputName of index %d failed!\n", i);
            return -1;
        }

        // 获取输入形状
        std::vector<int> shape;
        if (0 > model.GetInputShape(shape, i)) {
            printf("GetInputShape of index %d failed!\n", i);
            return -1;
        }

        // 获取输入数据类型
        std::string dtype;
        if (0 > model.GetInputDType(dtype, i)) {
            printf("GetInputDType of index %d failed!\n", i);
            return -1;
        }

        printf("Input[%d]: %s\n", i, name.c_str());
        printf("    Shape [");
        for (uint32_t j = 0; j < shape.size(); ++j) {
            printf("%d", (int)shape[j]);
            if (j + 1 < shape.size()) printf(", ");
        }
        printf("] %s\n", dtype.c_str());
        printf("    Size: %d\n", model.GetInputSize(i));
    }

    // 获取输出信息
    for (uint32_t i = 0; i < model.GetOutputNum(); ++i) {
        // 获取输出名称
        std::string name;
        if (0 > model.GetOutputName(name, i)) {
            printf("GetOutputName of index %d failed!\n", i);
            return -1;
        }

        // 获取输出形状
        std::vector<int> shape;
        if (0 > model.GetOutputShape(shape, i)) {
            printf("GetOutputShape of index %d failed!\n", i);
            return -1;
        }

        // 获取输出数据类型
        std::string dtype;
        if (0 > model.GetOutputDType(dtype, i)) {
            printf("GetOutputDType of index %d failed!\n", i);
            return -1;
        }

        printf("Output[%d]: %s\n", i, name.c_str());
        printf("    Shape [");
        for (uint32_t j = 0; j < shape.size(); ++j) {
            printf("%d", (int)shape[j]);
            if (j + 1 < shape.size()) printf(", ");
        }
        printf("] %s\n", dtype.c_str());
    }

    float model_in_freq_tmp[MODEL_IN_FREQ_LEN / 2] = {0.0};
    float model_in_time_tmp[MODEL_IN_TIME_LEN / 2] = {0.0};
    std::ofstream outFile(output_path);
    if (!outFile) {
        std::cerr << "无法打开文件: " << output_path << std::endl;
        return -1;
    }
    for (int round = 0; round < 10; round++) {
        printf("round: %d\n", round);
        outFile << "round = " << round << std::endl;

        // for (int i = 0; i < MODEL_IN_FREQ_LEN; i++) {
        //     model_in_freq_tmp[i] = model_in_freq[i] * (round/10.0 + 1);
        // }
        // for (int i = 0; i < MODEL_IN_TIME_LEN; i++) {
        //     model_in_time_tmp[i] = model_in_time[i] * (round/10.0 + 1);
        // }
        
        // 设置输入数据
        timer.start();
        if (0 > model.SetInput(model_in_time_tmp, 0) || 0 > model.SetInput(model_in_freq_tmp, 1)) {
            printf("SetInput failed!\n");
            return -1;
        }
        timer.stop();
        printf("SetInput take %.2fms\n", timer.elapsed<utilities::timer::milliseconds>());

        // 运行模型
        timer.start();
        if (0 > model.RunSync()) {
            printf("RunSync failed!\n");
            return -1;
        }
        timer.stop();
        printf("RunSync take %.2fms\n", timer.elapsed<utilities::timer::milliseconds>());

        // 获取输出数据
        timer.start();
        for (int i = 0; i < model.GetOutputNum(); i++) {
            int output_size = model.GetOutputSize(i) / sizeof(float);
            std::vector<float> output(output_size, 0);
            if (0 > model.GetOutput(output.data(), i)) {
                printf("GetOutput of index %d failed!\n", i);
                return -1;
            }
            for (int j = 0; j < 100; ++j) {
                outFile << output[j] << ", ";
            }
            outFile << std::endl;
        }
        timer.stop();
        printf("GetOutput take %.2fms\n", timer.elapsed<utilities::timer::milliseconds>());

    }

    outFile.close();
    std::cout << "数组已保存到: " << output_path << std::endl;

    // 释放模型（可以不显式调用，析构时会被调用）
    model.Release();

    return 0;
}