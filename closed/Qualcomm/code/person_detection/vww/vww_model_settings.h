/*======================= COPYRIGHT NOTICE ==================================*]
[* Copyright (c) 2021-2022 Qualcomm Technologies, Inc.                       *]
[* All Rights Reserved.                                                      *]
[* Confidential and Proprietary - Qualcomm Technologies, Inc.                *]
[*===========================================================================*/

#ifndef V0_1_KWS_MODEL_SETTINGS_H_
#define V0_1_KWS_MODEL_SETTINGS_H_
#ifndef INPUT_DATATYPE
#define INPUT_DATATYPE "uint8" //"uint8", "int8", "float"
#endif //INPUT_DATATYPE
const char *modelName = "vww_model.eai";
const char *inputPath = "000000000724.bin";
constexpr int kNumCols = 96;
constexpr int kNumRows = 96;
constexpr int kNumChannels = 3;

constexpr int kKwsInputSize = kNumCols * kNumRows * kNumChannels;
//quantization info
constexpr float qOutput_Sacle = 0.017484;
constexpr int qOutput_Zero = -7;

constexpr int kCategoryCount = 2;

const bool SAVE_OUTPUT = false;
/*=============================== For UART ==================================*/
char const *PATH_TO_SERIAL = "/dev/ttyMSM0"; 
#endif //V0_1_KWS_MODEL_SETTINGS_H_
