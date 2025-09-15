cmake_minimum_required(VERSION 2.8.0 FATAL_ERROR)

project(eai)

include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

if (NOT DEFINED EAI_BUILD_STATIC_RUNTIME)
    set (EAI_BUILD_STATIC_RUNTIME 1)
endif ()
if (NOT DEFINED FORCE_32BIT_BUILD)
    set (FORCE_32BIT_BUILD 0)
endif ()

#set(CMAKE_VERBOSE_MAKEFILE ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif ()


add_compile_options(-Wall)
add_compile_options(-D_FILE_OFFSET_BITS=64)
#add_compile_options(-Werror)

#enable performance logging
add_definitions(-DPERF_LOGGING)
add_definitions(-DUSE_NPU_HW=6)

if(DSP_BUILD)
    add_compile_options(-mv79)
endif()

if (${CMAKE_BUILD_TYPE} STREQUAL "RELEASE")
    add_compile_options(-O2)
else()
    add_compile_options(-O0)
endif()

IF(${PLATFORM} STREQUAL "hexagon_sim")
  add_definitions(-DCOSIM=1)
ENDIF()

if(${PLATFORM} STREQUAL "adsp")
    add_definitions(-DENABLE_POWER_TESTING=1)
    add_definitions(-DEAI_ADSP_ENABLE=1)
endif()

include(config.cmake)
config_compiler_and_linker()

add_subdirectory(example/eai_sample_app)
add_subdirectory(example/image_classification)
add_subdirectory(example/keyword_spotting)
add_subdirectory(example/person_detection)
add_subdirectory(example/anomaly_detection)



