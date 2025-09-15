cmake_minimum_required(VERSION 3.17)

project(eai_sdk_test_app C)

if (NOT DEFINED EAI_BUILD_STATIC_RUNTIME)
    set (EAI_BUILD_STATIC_RUNTIME 1)
endif ()

set(EAI_TINY_IC_SOURCE
    ./main.cpp
    ./submitter_implemented.cpp
    ./api/internally_implemented.cpp
    ./eai/eai_tiny_api.c
    ./eai/platform.c
)

link_directories(${EAI_SDK_LIB_PATH})
set (eai_link_libs PUBLIC eai)

if (${PLATFORM} STREQUAL "adsp")
    add_library(eai_tiny_ic SHARED ${EAI_TINY_IC_SOURCE})
else()
    add_executable(eai_tiny_ic ${EAI_TINY_IC_SOURCE})
endif()

target_include_directories (eai_tiny_ic
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/api
        ${CMAKE_CURRENT_SOURCE_DIR}/eai
        ${CMAKE_CURRENT_SOURCE_DIR}/ic
        ${CMAKE_CURRENT_SOURCE_DIR}/util
        ../../eai_runtime/api
)

if (${PLATFORM} STREQUAL "linux")
    list(APPEND eai_link_libs m pthread)
elseif (${PLATFORM} STREQUAL "adsp")
    list(APPEND eai_link_libs m)
elseif (${PLATFORM} STREQUAL "hexagon_sim")
    add_definitions(-DEXEC_HEXAGON_SIM)
    target_link_options(eai_tiny_ic  PRIVATE -z muldefs)
endif()

#enpuv3 sim uses some c++ features
if (${PLATFORM} STREQUAL "linux" AND USE_NPU_HW EQUAL v4)
    list(APPEND eai_link_libs  stdc++)
endif()

target_link_libraries(eai_tiny_ic PRIVATE ${eai_link_libs})


install (TARGETS eai_tiny_ic DESTINATION bin)


