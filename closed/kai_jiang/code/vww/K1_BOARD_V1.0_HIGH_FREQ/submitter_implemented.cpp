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
#include "model_params.h"

// #include "mbed.h"
float g_result[OUTPUT_ARRAY_SIZE + 1] = {0};
std::vector<int64_t> input_shape = {1, INPUT_ARRAY_CHANNELS, INPUT_ARRAY_WIDTH, INPUT_ARRAY_HEIGHT};
NetSession* g_session_ptr;
std::vector<Ort::Value> input_values;
extern int uart_fd;

// Implement this method to prepare for inference and preprocess inputs.
void th_load_tensor() {
  uint8_t input_values_uint8[INPUT_ARRAY_SIZE];
  size_t bytes = ee_get_buffer(input_values_uint8, INPUT_ARRAY_SIZE);
  if (bytes != INPUT_ARRAY_SIZE) {
    th_printf("Input db has %d elemented, expected %d\n", bytes, INPUT_ARRAY_SIZE);
    return;
  }

  static std::vector<float> input_tensor_values(INPUT_ARRAY_SIZE);
  for(int i = 0; i < INPUT_ARRAY_SIZE; ++i)
  {
    input_tensor_values[i] = (float)input_values_uint8[i] / 255.0;  //mobilenet模型的每个输入数据都需要除以255.0，否则精度很低
  }
  
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  input_values[0] = Ort::Value::CreateTensor<float>( memory_info, 
                                              input_tensor_values.data(), 
                                              INPUT_ARRAY_SIZE, 
                                              input_shape.data(), 
                                              input_shape.size() );
}

// Add to this method to return real inference results.
void th_results() {
  const int nresults = OUTPUT_ARRAY_SIZE;
  //int results[nresults];
 // float results[nresults];
  /**
   * The results need to be printed back in exactly this format; if easier
   * to just modify this loop than copy to results[] above, do that.
   */

  th_printf("m-results-[");
  for (size_t i = 0; i < nresults; i++) {
    /* N.B. Be sure %f is enabled in SDK */
	//results[i] = g_result[i];
    th_printf("%.3f", g_result[i]);
    if (i < (nresults - 1)) {
      th_printf(",");
    }
  }
  th_printf("]\r\n");
}

// Implement this method with the logic to perform one inference cycle.
void th_infer() 
{
	auto output_tensors = g_session_ptr->Run(input_values.data());
	const float* p_result = output_tensors[0].GetTensorData<float>();
  for(int i = 0; i < OUTPUT_ARRAY_SIZE; i++)
  {
    g_result[i] = p_result[i];
  }
}

/// \brief optional API.
void th_final_initialize(void) {
	const char* net_name = "mobilenet_v1.onnx";
	const char* net_param_path = "./models/mobilenet_v1.onnx";
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
	
//	std::vector<float> mean_value = {123.675f, 116.28f, 103.53f};
//	std::vector<float> scale_value = {58.395f, 57.12f, 57.375f};
	
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
