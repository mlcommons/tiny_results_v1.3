/*========================================================================*/
/**
\file eai_log.h

    Copyright (c) Qualcomm Technologies, Inc.
    All rights reserved.
    Confidential and Proprietary - Qualcomm Technologies, Inc.
*/
/*========================================================================*/

#ifndef __EAI_LOG__
#define __EAI_LOG__

#include <stdio.h>
#include <inttypes.h>

#ifdef EAI_ADSP_ENABLE
#define FARF_HIGH 1
#include "HAP_farf.h"
#else
#define FARF(...)
#endif

#define EAI_LOG(fmt, ...)                                                                                   \
{                                                                                                       \
    FARF(HIGH, fmt, ##__VA_ARGS__);                                                                     \
    printf(fmt, ##__VA_ARGS__);                                                                         \
}
#endif // #ifdef __EAI_LOG__


