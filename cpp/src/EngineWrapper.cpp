/**************************************************************************************************
 *
 * Copyright (c) 2019-2023 Axera Semiconductor (Ningbo) Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor (Ningbo) Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Ningbo) Co., Ltd.
 *
 **************************************************************************************************/
#include "EngineWrapper.hpp"
#include "utils/io.hpp"

#include <cstdlib>

static const char *strAlgoModelType[AX_ENGINE_MODEL_TYPE_BUTT] = {"3.6T", "7.2T", "10.8T"};

/// @brief npu type
typedef enum axNPU_TYPE_E {
    AX_NPU_DEFAULT = 0,        /* running under default NPU according to system */
    AX_STD_VNPU_1 = (1 << 0),  /* running under STD VNPU1 */
    AX_STD_VNPU_2 = (1 << 1),  /* running under STD VNPU2 */
    AX_STD_VNPU_3 = (1 << 2),  /* running under STD VNPU3 */
    AX_BL_VNPU_1 = (1 << 3),   /* running under BIG-LITTLE VNPU1 */
    AX_BL_VNPU_2 = (1 << 4)    /* running under BIG-LITTLE VNPU2 */
} AX_NPU_TYPE_E;

static AX_S32 CheckModelVNpu(const std::string &strModel, const AX_ENGINE_MODEL_TYPE_T &eModelType, const AX_S32 &nNpuType, AX_U32 &nNpuSet) {
    AX_ENGINE_NPU_ATTR_T stNpuAttr;
    memset(&stNpuAttr, 0x00, sizeof(stNpuAttr));

    auto ret = AX_ENGINE_GetVNPUAttr(&stNpuAttr);
    if (ret == 0) {
        // VNPU DISABLE
        if (stNpuAttr.eHardMode == AX_ENGINE_VIRTUAL_NPU_DISABLE) {
            nNpuSet = 0x01; // NON-VNPU (0b111)
        }
        // STD VNPU
        else if (stNpuAttr.eHardMode == AX_ENGINE_VIRTUAL_NPU_STD) {
            // 7.2T & 10.8T no allow
            if (eModelType == AX_ENGINE_MODEL_TYPE1
                || eModelType == AX_ENGINE_MODEL_TYPE2) {
                return -1;
            }

            // default STD VNPU2
            if (nNpuType == 0) {
                nNpuSet = 0x02; // VNPU2 (0b010)
            }
            else {
                if (nNpuType & AX_STD_VNPU_1) {
                    nNpuSet |= 0x01; // VNPU1 (0b001)
                }
                if (nNpuType & AX_STD_VNPU_2) {
                    nNpuSet |= 0x02; // VNPU2 (0b010)
                }
                if (nNpuType & AX_STD_VNPU_3) {
                    nNpuSet |= 0x04; // VNPU3 (0b100)
                }
            }
        }
        // BL VNPU
        else if (stNpuAttr.eHardMode == AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE) {
            // 10.8T no allow
            if (eModelType == AX_ENGINE_MODEL_TYPE2) {
                return -1;
            }

            // default BL VNPU
            if (nNpuType == 0) {
                // 7.2T default BL VNPU1
                if (eModelType == AX_ENGINE_MODEL_TYPE1) {
                    nNpuSet = 0x01; // VNPU1 (0b001)
                }
                // 3.6T default BL VNPU2
                else {
                    nNpuSet = 0x02; // VNPU2 (0b010)
                }
            }
            else {
                // 7.2T
                if (eModelType == AX_ENGINE_MODEL_TYPE1) {
                    // no allow set to BL VNPU2
                    if (nNpuType & AX_BL_VNPU_2) {
                        return -1;
                    }
                    if (nNpuType & AX_BL_VNPU_1) {
                        nNpuSet |= 0x01; // VNPU1 (0b001)
                    }
                }
                // 3.6T
                else {
                    if (nNpuType & AX_BL_VNPU_1) {
                        nNpuSet |= 0x01; // VNPU1 (0b001)
                    }
                    if (nNpuType & AX_BL_VNPU_2) {
                        nNpuSet |= 0x02; // VNPU2 (0b010)
                    }
                }
            }
        }
    }
    else {
        printf("AX_ENGINE_GetVNPUAttr fail ret = %x\n", ret);
    }

    return ret;
}


int EngineWrapper::Init(const char* strModelPath, uint32_t nNpuType)
{
    AX_S32 ret = 0;

    // 1. load model
    AX_BOOL bLoadModelUseCmm = AX_FALSE;
    AX_CHAR *pModelBufferVirAddr = nullptr;
    AX_U64 u64ModelBufferPhyAddr = 0;
    AX_U32 nModelBufferSize = 0;

    std::vector<char> model_buffer;

    if (bLoadModelUseCmm) {
        if (!utils::read_file(strModelPath, (AX_VOID **)&pModelBufferVirAddr, u64ModelBufferPhyAddr, nModelBufferSize)) {
            printf("ALGO read model(%s) fail\n", strModelPath);
            return -1;
        }
    }
    else {
        if (!utils::read_file(strModelPath, model_buffer)) {
            printf("ALGO read model(%s) fail\n", strModelPath);
            return -1;
        }

        pModelBufferVirAddr = model_buffer.data();
        nModelBufferSize = model_buffer.size();
    }

    auto freeModelBuffer = [&]() {
        if (bLoadModelUseCmm) {
            if (u64ModelBufferPhyAddr != 0) {
                AX_SYS_MemFree(u64ModelBufferPhyAddr, &pModelBufferVirAddr);
            }
        }
        else {
            std::vector<char>().swap(model_buffer);
        }
        return;
    };

    // 1.1 Get Model Type
     AX_ENGINE_MODEL_TYPE_T eModelType = AX_ENGINE_MODEL_TYPE0;
     ret = AX_ENGINE_GetModelType(pModelBufferVirAddr, nModelBufferSize, &eModelType);
     if (0 != ret || eModelType >= AX_ENGINE_MODEL_TYPE_BUTT) {
         printf("%s AX_ENGINE_GetModelType fail ret=%x, eModelType=%d\n", strModelPath, eModelType);
         freeModelBuffer();
         return -1;
     }

    // 1.2 Check VNPU
     AX_ENGINE_NPU_SET_T nNpuSet = 0;
     ret = CheckModelVNpu(strModelPath, eModelType, nNpuType, nNpuSet);
     if (0 != ret) {
         printf("ALGO CheckModelVNpu fail\n");
         freeModelBuffer();
         return -1;
     }

    // 2. create handle
    AX_ENGINE_HANDLE handle = nullptr;
    ret = AX_ENGINE_CreateHandle(&handle, pModelBufferVirAddr, nModelBufferSize);
    auto deinit_handle = [&handle]() {
        if (handle) {
            AX_ENGINE_DestroyHandle(handle);
        }
        return -1;
    };

    freeModelBuffer();

    if (0 != ret || !handle) {
        printf("ALGO Create model(%s) handle fail\n", strModelPath);

        return deinit_handle();
    }

    // 3. create context
    ret = AX_ENGINE_CreateContext(handle);
    if (0 != ret) {
        return deinit_handle();
    }

    // 4. set io
    m_io_info = nullptr;
    ret = AX_ENGINE_GetIOInfo(handle, &m_io_info);
    if (0 != ret) {
        return deinit_handle();
    }
    m_input_num = m_io_info->nInputSize;
    m_output_num = m_io_info->nOutputSize;

    // 4.1 query io
//    AX_IMG_FORMAT_E eDtype;
//    ret = utils::query_model_input_size(m_io_info, m_input_size, eDtype);//FIXME.
//    if (0 != ret) {
//        printf("model(%s) query model input size fail\n", strModelPath.c_str());
//        return deinit_handle();
//    }

//    if (!(eDtype == AX_FORMAT_YUV420_SEMIPLANAR ||  eDtype == AX_FORMAT_YUV420_SEMIPLANAR_VU ||
//        eDtype == AX_FORMAT_RGB888 || eDtype == AX_FORMAT_BGR888)) {
//        printf("model(%s) data type is: 0x%02X, unsupport\n", strModelPath, eDtype);
//        return deinit_handle();
//    }

    // 4.2 brief io
#ifdef __DEBUG__
    printf("brief_io_info\n");
    utils::brief_io_info(strModelPath, m_io_info);
#endif

    //5. Config VNPU
    // printf("model(%s) nNpuSet: 0x%08X\n", strModelPath.c_str(), nNpuSet);
    // will do nothing for using create handle v2 api

    // 6. prepare io
    // AX_U32 nIoDepth = (stCtx.vecOutputBufferFlag.size() == 0) ? 1 : stCtx.vecOutputBufferFlag.size();
    // ret = utils::prepare_io(strModelPath, m_io_info, m_io, utils::IO_BUFFER_STRATEGY_DEFAULT);
    // printf("use IO_BUFFER_STRATEGY_DEFAULT!\n");
    ret = utils::prepare_io(strModelPath, m_io_info, m_io, utils::IO_BUFFER_STRATEGY_CACHED);
    if (0 != ret) {
        printf("prepare io failed!\n");
        utils::free_io(m_io);
        return deinit_handle();
    }

    m_handle = handle;
    m_hasInit = true;

    return 0;
}

int EngineWrapper::GetInputShape(std::vector<int>& shape, int index) {
    if (index >= m_input_num) {
        return -1;
    }

    auto& input = m_io_info->pInputs[index];
    shape.resize(input.nShapeSize);
    for (uint32_t j = 0; j < input.nShapeSize; ++j) {
        shape[j] = (int)input.pShape[j];
    }

    return 0;
}

int EngineWrapper::GetOutputShape(std::vector<int>& shape, int index) {
    if (index >= m_output_num) {
        return -1;
    }

    auto& output = m_io_info->pOutputs[index];
    shape.resize(output.nShapeSize);
    for (uint32_t j = 0; j < output.nShapeSize; ++j) {
        shape[j] = (int)output.pShape[j];
    }

    return 0;
}

int EngineWrapper::GetInputName(std::string& name, int index) {
    if (index >= m_input_num) {
        return -1;
    }

    auto& input = m_io_info->pInputs[index];
    name = std::string(input.pName);

    return 0;
}

int EngineWrapper::GetOutputName(std::string& name, int index) {
    if (index >= m_output_num) {
        return -1;
    }

    auto& output = m_io_info->pOutputs[index];
    name = std::string(output.pName);

    return 0;
}

int EngineWrapper::GetInputDType(std::string& dtype, int index) {
    auto describe_data_type = [](AX_ENGINE_DATA_TYPE_T type) -> const char* {
        switch (type) {
            case AX_ENGINE_DT_UINT8:
                return "uint8";
            case AX_ENGINE_DT_UINT16:
                return "uint16";
            case AX_ENGINE_DT_FLOAT32:
                return "float32";
            case AX_ENGINE_DT_SINT16:
                return "sint16";
            case AX_ENGINE_DT_SINT8:
                return "sint8";
            case AX_ENGINE_DT_SINT32:
                return "sint32";
            case AX_ENGINE_DT_UINT32:
                return "uint32";
            case AX_ENGINE_DT_FLOAT64:
                return "float64";
            case AX_ENGINE_DT_UINT10_PACKED:
                return "uint10_packed";
            case AX_ENGINE_DT_UINT12_PACKED:
                return "uint12_packed";
            case AX_ENGINE_DT_UINT14_PACKED:
                return "uint14_packed";
            case AX_ENGINE_DT_UINT16_PACKED:
                return "uint16_packed";
            default:
                return "unknown";
        }
    };

    if (index >= m_input_num) {
        return -1;
    }

    auto& input = m_io_info->pInputs[index];
    dtype = std::string(describe_data_type(input.eDataType));
    return 0;
}

int EngineWrapper::GetOutputDType(std::string& dtype, int index) {
    auto describe_data_type = [](AX_ENGINE_DATA_TYPE_T type) -> const char* {
        switch (type) {
            case AX_ENGINE_DT_UINT8:
                return "uint8";
            case AX_ENGINE_DT_UINT16:
                return "uint16";
            case AX_ENGINE_DT_FLOAT32:
                return "float32";
            case AX_ENGINE_DT_SINT16:
                return "sint16";
            case AX_ENGINE_DT_SINT8:
                return "sint8";
            case AX_ENGINE_DT_SINT32:
                return "sint32";
            case AX_ENGINE_DT_UINT32:
                return "uint32";
            case AX_ENGINE_DT_FLOAT64:
                return "float64";
            case AX_ENGINE_DT_UINT10_PACKED:
                return "uint10_packed";
            case AX_ENGINE_DT_UINT12_PACKED:
                return "uint12_packed";
            case AX_ENGINE_DT_UINT14_PACKED:
                return "uint14_packed";
            case AX_ENGINE_DT_UINT16_PACKED:
                return "uint16_packed";
            default:
                return "unknown";
        }
    };

    if (index >= m_output_num) {
        return -1;
    }

    auto& output = m_io_info->pOutputs[index];
    dtype = std::string(describe_data_type(output.eDataType));
    return 0;
}

int EngineWrapper::SetInput(void* pInput, int index) {
    return utils::push_io_input(pInput, index, m_io);
}

int EngineWrapper::RunSync()
{
    if (!m_hasInit)
        return -1;

    // 7.3 run & benchmark
    auto ret = AX_ENGINE_RunSync(m_handle, &m_io);
    if (0 != ret) {
        printf("AX_ENGINE_RunSync failed. ret=0x%x\n", ret);
        return ret;
    }

    return 0;
}

int EngineWrapper::GetOutput(void* pOutput, int index) {
    return utils::push_io_output(pOutput, index, m_io);
}

int EngineWrapper::GetInputSize(int index) {
    return m_io.pInputs[index].nSize;
}

int EngineWrapper::GetOutputSize(int index) {
    return m_io.pOutputs[index].nSize;
}

void* EngineWrapper::GetInputPtr(int index) {
    return m_io.pInputs[index].pVirAddr;
}

void* EngineWrapper::GetOutputPtr(int index) {
    return m_io.pOutputs[index].pVirAddr;
}

int EngineWrapper::Release()
{
    if (m_handle) {
        utils::free_io(m_io);
        AX_ENGINE_DestroyHandle(m_handle);
        m_handle = nullptr;
    }
    return 0;
}
