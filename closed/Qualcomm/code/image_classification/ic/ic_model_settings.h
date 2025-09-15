/*======================= COPYRIGHT NOTICE ==================================*]
[* Copyright (c) 2021-2022 Qualcomm Technologies, Inc.                       *]
[* All Rights Reserved.                                                      *]
[* Confidential and Proprietary - Qualcomm Technologies, Inc.                *]
[*===========================================================================*/

#ifndef V0_1_IC_MODEL_SETTINGS_H_
#define V0_1_IC_MODEL_SETTINGS_H_
#ifndef INPUT_DATATYPE
#define INPUT_DATATYPE "uint8" //"uint8", "int8", "float"
#endif //INPUT_DATATYPE
// #ifndef _MODEL_LOAD_FROM_FILE
// #define _MODEL_LOAD_FROM_FILE
// #endif //_MODEL_LOAD_FROM_FILE
const char *modelName = "ic_model.eai";
const char *inputPath = "toy_spaniel_s_000285.bin";
const char *perf_mode = "turbo"; //{"lowsvs_d1", "minsvs", "lowsvs", "svs", "svs_l1", "nom", "nom_l1", "turbo", "turbo_l1"};
constexpr int kNumCols = 32;
constexpr int kNumRows = 32;
constexpr int kNumChannels = 3;

constexpr int kIcInputSize = kNumCols * kNumRows * kNumChannels;

constexpr float qOutput_Sacle = 0.204412;
constexpr int qOutput_Zero = 0;

constexpr int kCategoryCount = 10;
constexpr int kAirplaneIndex = 0;
constexpr int kAutomobileIndex = 1;
constexpr int kBirdIndex = 2;
constexpr int kCatIndex = 3;
constexpr int kDeerIndex = 4;
constexpr int kDogIndex = 5;
constexpr int kFrogIndex = 6;
constexpr int kHorseIndex = 7;
constexpr int kShipIndex = 8;
constexpr int kTruckIndex = 9;
const char* kCategoryLabels[kCategoryCount] = {
	"airplane",
	"automobile",
	"bird",
	"cat",
	"deer",
	"dog",
	"frog",
	"horse",
	"ship"
	"truck",
};
const bool SAVE_OUTPUT = true;
/*=============================== For UART ==================================*/
char const *PATH_TO_SERIAL = "/dev/ttyMSM0"; 
// char const *PATH_TO_SERIAL = "/dev/tty"; 

#endif //V0_1_IC_MODEL_SETTINGS_H_
