cmake_minimum_required(VERSION 3.18)
project(mnist LANGUAGES CXX CUDA)

# Define executable name
set(TARGET_NAME main)

# Define source file
set(SRC main.cu)

# Default to Release mode if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA flags
set(CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")

# Debug mode
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
endif()

# PTXAS mode (inherits from Debug mode)
if (CMAKE_BUILD_TYPE STREQUAL "PTXAS")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G --ptxas-options=-v")
endif()

# Add executable
add_executable(${TARGET_NAME} ${SRC})

# Set CUDA compilation flags
target_compile_options(${TARGET_NAME} PRIVATE ${CUDA_NVCC_FLAGS})

# Link CUDA
set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

