/*========================================================================*/
/**
\file platform.c

    Copyright (c) Qualcomm Technologies, Inc.
    All rights reserved.
    Confidential and Proprietary - Qualcomm Technologies, Inc.
*/
/*========================================================================*/

#include <stdint.h>
#if defined __linux__ || defined EAI_ADSP_ENABLE
#include <sys/time.h>
#endif
#include "platform.h"

int64_t get_sys_time(){
#if defined __linux__ || defined EAI_ADSP_ENABLE
    struct timeval time;
    gettimeofday(&time, NULL);
    int64_t tv_sec = (int64_t) time.tv_sec;
    int64_t tv_usec = (int64_t) time.tv_usec;
    return (int64_t)(tv_sec * 1000000 + tv_usec);
#endif
    return 0;
}

// Allocate size bytes, aligned to alignment bytes
void *malloc_align(size_t alignment, size_t size)
{

    // `alignment` must be less than or equal to 256 since pAllocOffset is one byte
    if (alignment > 256) {
        return NULL;
    }

    size_t padded_size = size + alignment;
    size_t unaligned = (size_t) malloc(padded_size);

    // Malloc failure - return NULL ptr
    if ((void *) unaligned == NULL) {
        return (void *) unaligned;
    }

    size_t aligned = 0;         // Address of aligned memory
    uint8_t *pAllocOffset = 0;  // Number of bytes between unaligned and aligned addresses

    // Check if original address is already aligned
    if (IS_ALIGNED(unaligned, alignment) == 0) {
        // Original address is unaligned - move up to next alignment byte boundary
        aligned = PAD_TO_VALUE(unaligned, alignment);
        // Store (aligned - unaligned) in uint8_t before aligned memory
        pAllocOffset = (uint8_t *) aligned - 1;
        *pAllocOffset = aligned - unaligned;
    }
    else {
        // Already aligned - move up alignment bytes so we can store the offset
        aligned = unaligned + alignment;

        // Store (alignment) in uint8_t before aligned memory
        pAllocOffset = (uint8_t *) aligned - 1;
        *pAllocOffset = (uint8_t) alignment; // If alignment is 256, 0 is automatically stored since uint8_t's max value is 255
    }
    return (void *) aligned;
}

// Free original memory associated with aligned ptr
int free_align(void *ptr) {
    if (ptr == NULL) {
        return -1;
    }
    // Set pAllocOffset to uint8_t containing byte offset between unaligned and aligned memory
    uint8_t *pAllocOffset = (uint8_t *) ptr - 1;

    // Get unaligned memory address
    size_t aligned = (size_t) ptr;
    size_t unaligned = 0;
    // Check if offset is 0
    if (*pAllocOffset == 0) {
        // Offset is actually 256 bytes
        unaligned = aligned - 256;
    } else {
        unaligned = aligned - *pAllocOffset;
    }

    // Free unaligned memory
    free((void *) unaligned);
    return 0;
}
