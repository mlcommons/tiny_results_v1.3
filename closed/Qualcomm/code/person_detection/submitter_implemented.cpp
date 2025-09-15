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

This file reflects a modified version of th_lib from EEMBC. All wrapped libc
methods from th_libc.h and all testharness methods from th_lib.h are here.
==============================================================================*/
/// \file
/// \brief Submitter-implemented methods required to perform inference.

#include "api/submitter_implemented.h"
#include "api/internally_implemented.h"
#include "util/quantization_helpers.h"
#include "vww/vww_model_settings.h"
#include "eai/eai_tiny_api.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>

struct eai_run_context context;
int serial_fd = 0;
int fd_initialized = 0;
char output_buffer[30000];
float calculate_result(struct eai_run_context *context)
{
    th_printf("m-results-[");
    for (int i = 0; i < context->tensor_count[1]; i++)
    {
        eai_tensor_info_t *tensor_o = &context->tensors[1][i];
        // Output data type is int8
        int8_t *p_o = reinterpret_cast<int8_t *>(tensor_o->address);
        if (kCategoryCount != tensor_o->tensor_size / sizeof(int8_t))
        {
            th_printf("output data type is int32, output data dimension: %lu != %d, tensor size is %lu", tensor_o->tensor_size / sizeof(int8_t), kCategoryCount, tensor_o->tensor_size);
        }
        for (uint8_t j = 0; j < tensor_o->tensor_size / sizeof(int8_t); j++)
        {
            // int32 to float
            float converted = (*p_o - qOutput_Zero) * qOutput_Sacle;
            th_printf("%0.3f", converted);
            if (j < (kCategoryCount - 1))
            {
                th_printf(",");
            }
            p_o++;
        }
        th_printf("]\r\n");
    }
    return 0;
}

// uint8 -> int8
int Uint8ToInt8(uint8_t *address, uint32_t tensor_size)
{
    uint8_t *p = address;
    for (uint32_t i = 0; i < tensor_size; i++)
    {
        int8_t tmp = *p / 2;
        memcpy(p, &tmp, sizeof(int8_t));
        p++;
    }
    return 0;
}

void th_load_tensor()
{
    int ret = 0;
    if (context.tensor_count[0] > 1)
    {
        th_printf("Input count is %d > 1, not supported", context.tensor_count[0]);
    }
    for (int i = 0; i < context.tensor_count[0]; i++)
    {
        eai_tensor_info_t *tensor = &context.tensors[0][i];
#ifdef _MODEL_LOAD_FROM_FILE
        initialize_i(&context);
        size_t read_size = fread(tensor->address, 1, tensor->tensor_size, context.io_file[0][i]);
#else
        size_t read_size = ee_get_buffer(reinterpret_cast<uint8_t *>(tensor->address),
                                         tensor->tensor_size * sizeof(int8_t));
#endif
        th_printf("Input db has %zu elemented, expected %d\n", read_size,
                  tensor->tensor_size);
        if (strncmp("uint8", INPUT_DATATYPE, 10) == 0)
        {
            Uint8ToInt8(tensor->address, tensor->tensor_size);
        }
        if (read_size / sizeof(int8_t) != tensor->tensor_size)
        {
            th_printf("Input db has %d elemented, expected %d\n", read_size / sizeof(int8_t),
                      tensor->tensor_size);
            return;
        }
        if (read_size == 0)
        {
            ret = -2; // indicating end of file
            break;
        }
        if (read_size != tensor->tensor_size)
        {
            ret = -1;
            th_printf("fail to load inputs %d\n", ret);
            break;
        }
    }
    ret = fill_io_batch(&context);
    if (ret)
    {
        th_printf("fail to fill i/o tensor buffer\n");
    }
}

// Add to this method to return real inference results.
void th_results()
{
    bool save_output = false;
    if (save_output)
    {
        initialize_o(&context);
        int ret = 0;
        ret = save_outputs(&context);
        if (ret)
        {
            printf("fail to save outputs %d", ret);
        }
        fclose(context.io_file[1][0]);
    }
    calculate_result(&context);
    /**
     * The results need to be printed back in exactly this format; if easier
     * to just modify this loop than copy to results[] above, do that.
     */
}

// Implement this method with the logic to perform one inference cycle.
void th_infer()
{
    EAI_RESULT eai_ret = EAI_SUCCESS;
    eai_ret = eai_execute(context.eai_handle, &context.eai_batch, 1);
    if (eai_ret == EAI_NEED_MORE)
    {
        th_printf("eAI_execute need more\n");
    }
    if (eai_ret != EAI_SUCCESS)
    {
        th_printf("fail to eai_execute ret = %d\n", eai_ret);
    }
}

void logging(char volatile* p_command){}

/// \brief optional API.

// clode fd 
void die(int fd, char *msg) {
    perror(msg);
    close(fd);
    fd_initialized = 0;
    exit(1);
}
// initiallize fd
void serial_initialize(void)
{
    serial_fd = open(PATH_TO_SERIAL, O_RDWR | O_NONBLOCK | O_NOCTTY);
    if(serial_fd == -1) { 
        char log[] ={"open serial failed"};
        die(serial_fd, log); }
    fd_initialized = 1;
}

// / \brief optional API.
void th_final_initialize(void)
{
    int ret = 0;
    memset(&context, 0, sizeof(struct eai_run_context));
    context.model_name = modelName;
    context.output_path = ".";
    context.use_enpu = 1;
    context.hw_cfg_heap_type = 0;
    context.buffer_heap_type = 0;
    context.input_file[0] = inputPath;
    context.input_file_count = 1;
    context.allocate_io_buf = 0;
    context.affinity = 6;
    context.priority = 0;
    

    // load model
    ret = load_model(&context);
    if (ret)
    {
        th_printf("fail to load eai model %s\n", context.model_name);
    }
    ret = init_eai(&context);
    if (ret)
    {
        th_printf("fail to initialize eai context\n");
    }
    ret = get_model_io(&context);
    if (ret)
    {
        th_printf("fail to get model input/output information\n");
    }
    print_model_io(&context);
    if (context.input_file_count != context.tensor_count[0])
    {
        th_printf("number of input files not match with input tensor in the model\n");
        ret = -1;
    }
}

void th_pre() {}
void th_post() {}

void th_command_ready(char volatile *p_command)
{
    // del \r and \n in command
    int move = 0;
    int i = 0;
    do {
        if (*(p_command + i + move) == '\n' || *(p_command + i + move) == '\r'){
            move++;
        }
        else{
            *(p_command + i) = *(p_command + i + move);
            i++;
        }

    }while (*(p_command + i) != (char)0);
    ee_serial_command_parser_callback((char *)p_command);
}

// th_libc implementations.
int th_strncmp(const char *str1, const char *str2, size_t n)
{
    return strncmp(str1, str2, n);
}

char *th_strncpy(char *dest, const char *src, size_t n)
{
    return strncpy(dest, src, n);
}

size_t th_strnlen(const char *str, size_t maxlen)
{
    // return strnlen(str, maxlen);
    return strlen(str);
}

char *th_strcat(char *dest, const char *src) { return strcat(dest, src); }

char *th_strtok(char *str1, const char *sep) { return strtok(str1, sep); }

int th_atoi(const char *str) { return atoi(str); }

void *th_memset(void *b, int c, size_t len) { return memset(b, c, len); }

void *th_memcpy(void *dst, const void *src, size_t n)
{
    return memcpy(dst, src, n);
}

/* N.B.: Many embedded *th_printf SDKs do not support all format specifiers. */
int th_vprintf(const char *format, va_list ap) 
{
    int result;
    int out_len;
    int L_WRITE = 1;
    if (!fd_initialized){
        serial_initialize();
    }
    result = vsprintf(output_buffer, format, ap);
    out_len = strlen(output_buffer);
    logging(output_buffer);

    for (int i = 0; i < out_len; i++) {
        if(write(serial_fd, output_buffer+i, L_WRITE) != L_WRITE) {
            char log[] ={"write()"};
            die(serial_fd, log);
        }
    }
  return result;
}
void th_printf(const char *p_fmt, ...)
{
    va_list args;
    va_start(args, p_fmt);
    (void)th_vprintf(p_fmt, args); /* ignore return */
    va_end(args);
}

char th_getchar() 
{
    if (!fd_initialized){
        serial_initialize();
    }
    int bytesread = 0;
    char buffer[1];
    while (bytesread != 1) { 
        bytesread = read(serial_fd, buffer, 1);
        }
    return buffer[0];
    // return getchar();

}

void th_serialport_initialize(void) 
{
    serial_initialize();
}

void th_timestamp(void)
{
    
    unsigned long microSeconds = 0ul;
    // /* USER CODE 2 BEGIN */
    struct timeval time;
    gettimeofday(&time, NULL);
    int64_t tv_sec = (int64_t) time.tv_sec;
    int64_t tv_usec = (int64_t) time.tv_usec;
    microSeconds = (unsigned long)((int64_t)(tv_sec * 1000000 + tv_usec));
    // microSeconds = (unsigned long)((uint64_t)clock() * 1000000 / CLOCKS_PER_SEC);
    // /* USER CODE 2 END */
    // /* This message must NOT be changed. */
    th_printf(EE_MSG_TIMESTAMP, microSeconds);
}

void th_timestamp_initialize(void)
{
    /* USER CODE 1 BEGIN */
    // Setting up BOTH perf and energy here
    /* USER CODE 1 END */
    /* This message must NOT be changed. */
    th_printf(EE_MSG_TIMESTAMP_MODE);
    /* Always call the timestamp on initialize so that the open-drain output
       is set to "1" (so that we catch a falling edge) */
    th_timestamp();
}