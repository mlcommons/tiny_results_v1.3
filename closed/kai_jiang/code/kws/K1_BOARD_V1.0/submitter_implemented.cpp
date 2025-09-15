/*
Copyright 2020 EEMBC and The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file reflects a modified version of th_lib from EEMBC. The reporting logic
in th_results is copied from the original in EEMBC.
==============================================================================*/
/// \file
/// \brief C++ implementations of submitter_implemented.h

#include "submitter_implemented.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include <assert.h>
#include <onnxruntime_cxx_api.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <thread>

#include "internally_implemented.h"

//#include "imagenet_labels.h"
#include "runtime_ort_env.h"
#include "uart.h"
#include "utils.h"
#include "model_settings.h"

// #include "mbed.h"
std::vector<Ort::Value> input_values;
std::vector<Ort::Value> output_values;
std::vector<int64_t> input_shape = {1, numRows * numCols * numChannels};
NetSession* g_session_ptr;
extern int uart_fd;

// Implement this method to prepare for inference and preprocess inputs.
void th_load_tensor() {
  int8_t input_values_int8[inputSize];
  size_t bytes = ee_get_buffer(reinterpret_cast<uint8_t *>(input_values_int8), 
                               inputSize * sizeof(int8_t));
  if (bytes / sizeof(int8_t) != inputSize) {
    th_printf("Input db has %d elemented, expected %d\n", bytes / sizeof(int8_t), inputSize);
    return;
  }
  
  static std::vector<float> input_tensor_values(inputSize);  //需要用static，因为CreateTensor不会复制数据
  for(int i = 0; i < inputSize; ++i)
  {
    input_tensor_values[i] = ((float)input_values_int8[i] - 83) * 0.5847029089927673;
  }
  
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  input_values[0] = Ort::Value::CreateTensor<float>( memory_info, 
                                              input_tensor_values.data(), 
                                              input_tensor_values.size(), 
                                              input_shape.data(), 
                                              input_shape.size() );
}

// Add to this method to return real inference results.
void th_results() {
  /**
   * The results need to be printed back in exactly this format
   */
  const float* p_result = output_values[0].GetTensorData<float>();

  th_printf("m-results-[");
  for (size_t i = 0; i < categoryCount; i++) {
    th_printf("%.3f", p_result[i]);
    if (i < (categoryCount - 1)) {
      th_printf(",");
    }
  }
  th_printf("]\r\n");
}

// Implement this method with the logic to perform one inference cycle.
void th_infer() 
{
	output_values = g_session_ptr->Run(input_values.data());
}

/// \brief optional API.
void th_final_initialize(void) {
	const char* net_name = "kws_ref_model.onnx";
	const char* net_param_path = "./models/kws_ref_model.onnx";
	char* profile_prefix = nullptr;
	int num_threads = 4;
	
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort_test");
	Ort::AllocatorWithDefaultOptions allocator;
	Ort::SessionOptions session_options;
	std::unordered_map<std::string, std::string> provider_options;
	OrtStatus* status = Ort::SessionOptionsRuntimeEnvInit(session_options, provider_options);
	session_options.SetIntraOpNumThreads(num_threads);
	session_options.SetInterOpNumThreads(num_threads);
	
	if (profile_prefix != nullptr && strcmp(profile_prefix, "None") != 0) {
		std::string profile_path = net_name;
		profile_path = profile_prefix + profile_path;
		session_options.EnableProfiling(profile_path.c_str());

		std::string opt_net_path = net_name;
		opt_net_path = profile_prefix + opt_net_path + "_opt.onnx";
		session_options.SetOptimizedModelFilePath(opt_net_path.c_str());
	}
	
	static NetSession session(env, net_param_path, session_options);
	g_session_ptr = &session;
	
	auto input_count = session.GetInputCount();
	auto output_count = session.GetOutputCount();
	
	input_values.reserve(input_count);
	for (size_t i = 0; i < input_count; i++) {
		session.SetInputShape(i, input_shape);
		input_values.push_back(session.CreatorInputValue(i));  //set initial values for input_values
	}
}

void th_pre() {}
void th_post() {}

void th_command_ready(char volatile *p_command) {
  p_command = p_command;
  ee_serial_command_parser_callback((char *)p_command);
}

// th_libc implementations.
int th_strncmp(const char *str1, const char *str2, size_t n) {
  return strncmp(str1, str2, n);
}

char *th_strncpy(char *dest, const char *src, size_t n) {
  return strncpy(dest, src, n);
}

size_t th_strnlen(const char *str, size_t maxlen) {
  return strnlen(str, maxlen);
}

char *th_strcat(char *dest, const char *src) { return strcat(dest, src); }

char *th_strtok(char *str1, const char *sep) { return strtok(str1, sep); }

int th_atoi(const char *str) { return atoi(str); }

void *th_memset(void *b, int c, size_t len) { return memset(b, c, len); }

void *th_memcpy(void *dst, const void *src, size_t n) {
  return memcpy(dst, src, n);
}

/* N.B.: Many embedded *printf SDKs do not support all format specifiers. */
//int th_vprintf(const char *format, va_list ap) { return vprintf(format, ap); }
int th_vprintf(const char *format, va_list ap) { 
  char buffer[1024];
  int len = vsnprintf(buffer, sizeof(buffer), format, ap);
  if (len >= 0 && len < sizeof(buffer)) {
    // if (uart_fd == -1) {
    //     uart_fd = open_serial("/dev/ttyS7", B9600);
    // }
    write(uart_fd, buffer, strlen(buffer));
  }
  return len;
}

void th_printf(const char *p_fmt, ...) {
  va_list args;
  va_start(args, p_fmt);
  (void)th_vprintf(p_fmt, args); /* ignore return */
  va_end(args);
}

char th_getchar() { return getchar(); }

// UnbufferedSerial pc(USBTX, USBRX);

// void th_serialport_initialize(void) { pc.baud(115200); }
void th_timestamp(void) {
  unsigned long microSeconds = 0ul;
  /* USER CODE 2 BEGIN */
  auto tp = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> duration = tp.time_since_epoch();
  microSeconds = static_cast<unsigned long>(duration.count());
  /* USER CODE 2 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP, microSeconds);
}

void th_timestamp_initialize(void) {
  /* USER CODE 1 BEGIN */
  // Setting up BOTH perf and energy here
  /* USER CODE 1 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP_MODE);
  /* Always call the timestamp on initialize so that the open-drain output
     is set to "1" (so that we catch a falling edge) */
  th_timestamp();
}
