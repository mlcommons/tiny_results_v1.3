/*======================= COPYRIGHT NOTICE ==================================*]
[* Copyright (c) 2021-2022 Qualcomm Technologies, Inc.                       *]
[* All Rights Reserved.                                                      *]
[* Confidential and Proprietary - Qualcomm Technologies, Inc.                *]
[*===========================================================================*/

#ifndef V0_1_KWS_MODEL_SETTINGS_H_
#define V0_1_KWS_MODEL_SETTINGS_H_
#ifndef INPUT_DATATYPE
#define INPUT_DATATYPE "int8" //"uint8", "int8", "float"
#endif //INPUT_DATATYPE
const char *modelName = "kws_model.eai";
const char *inputPath = "tst_000003_Up_8.bin";
constexpr int kNumCols = 10;
constexpr int kNumRows = 49;
constexpr int kNumChannels = 1;

constexpr int kKwsInputSize = kNumCols * kNumRows * kNumChannels;
//quantization info
constexpr float qOutput_Sacle = 0.187649;
constexpr int qOutput_Zero = 0;

constexpr int kCategoryCount = 12;
const char* kCategoryLabels[kCategoryCount] = {
  "down", "go", "left", "no", "off", "on",
  "right", "stop", "up", "yes", "silence", "unknown"
};
const bool SAVE_OUTPUT = false;
/*=============================== For UART ==================================*/
char const *PATH_TO_SERIAL = "/dev/ttyMSM0"; 
#endif //V0_1_KWS_MODEL_SETTINGS_H_