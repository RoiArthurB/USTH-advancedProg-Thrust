cmake_minimum_required(VERSION 3.0)
set(EXECUTABLE_NAME LW3)
set(PROJECT_NAME LW3)

project(${EXECUTABLE_NAME} LANGUAGES CXX)

find_package(CUDA REQUIRED)

include_directories("." include "/usr/local/cuda/include")

set(CUDA_VERBOSE_BUILD ON)

set(CUDA_HOST_COMPILATION_CPP ON)
set(CUDA_NVCC_FLAGS -std=c++11 -O3 -Wno-deprecated-gpu-targets --gpu-architecture=sm_52 --disable-warnings)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 11)

set(CUDA_SEPARABLE_COMPILATION ON)

set(HEADERS student.hpp utils/chronoGPU.hpp utils/chronoCPU.hpp utils/utils.cuh )
set(SOURCES main.cu student.cu utils/chronoCPU.cpp utils/chronoGPU.cu )
cuda_add_executable(${EXECUTABLE_NAME} ${SOURCES} ${HEADERS})
