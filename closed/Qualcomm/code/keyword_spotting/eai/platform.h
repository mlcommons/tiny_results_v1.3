/*========================================================================*/
/**
\file platform.h

    Copyright (c) Qualcomm Technologies, Inc.
    All rights reserved.
    Confidential and Proprietary - Qualcomm Technologies, Inc.
*/
/*========================================================================*/

#ifndef __PLATFORM_H__
#define __PLATFORM_H__

#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>


int64_t get_sys_time();

void * malloc_align(size_t alignment, size_t size);
int free_align(void *ptr);

// int preprocess_argc_argv(int *pArgc, char*** pArgV);
#define IS_ALIGNED(addr, alignment)     ((((size_t) addr) & (alignment - 1)) == 0)
#define PAD_TO_VALUE(addr, alignment)   ((((size_t) addr) + (alignment - 1)) & ~(alignment - 1))
#endif
