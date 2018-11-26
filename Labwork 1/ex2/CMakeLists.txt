cmake_minimum_required(VERSION 3.12.3)
set(EXECUTABLE_NAME Exercise1)
set(PROJECT_NAME Exercise1)

# set CUDA
set(CUDA_STANDARD 14)
set(CUDA_STANDARD_REQUIRED ON)
set(CUDA_64_BIT_DEVICE_CODE ON)
set(CUDA_VERBOSE_BUILD  ON)
project(${EXECUTABLE_NAME} LANGUAGES CXX CUDA)


# Tell CMake to run moc when necessary
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
	
include_directories(include "D:/CUDA Samples/v10.0/common/inc")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")  
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=sm_52")

set(HEADERS Exercise.hpp include/ChronoGPU.hpp)
set(SOURCES main.cpp Exercise.cu include/ChronoGPU.cu)

#Remember dependencies!  
add_executable(${EXECUTABLE_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${EXECUTABLE_NAME})

# Enable C++14 standard compliance
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 14)
