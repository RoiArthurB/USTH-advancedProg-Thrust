cmake_minimum_required(VERSION 3.0)
set(EXECUTABLE_NAME ProjectMedianFilter)
set(PROJECT_NAME ProjectMedianFilter)

project(${EXECUTABLE_NAME} LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
	
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")  
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)
if (NOT CUDA_FOUND)
    message("CUDA location not found")
endif()

set(CUDA_VERBOSE_BUILD ON)

if (WIN32)
    include_directories(utils students "D:/CUDA Samples/v10.0/common/inc")
    list(APPEND CUDA_NVCC_FLAGS -O3 -Wno-deprecated-gpu-targets --gpu-architecture=sm_52 --expt-extended-lambda)
else()
    include_directories(utils students "${CUDA_SDK_ROOT_DIR}/samples/common/inc")
    list(APPEND CUDA_NVCC_FLAGS -std=c++11 -O3 -Wno-deprecated-gpu-targets --gpu-architecture=sm_30 --expt-extended-lambda)
endif(WIN32)

set(HEADERS students/student1.hpp students/student2.hpp students/student3.hpp students/student4.hpp utils/chronoGPU.hpp utils/chronoCPU.hpp utils/utils.cuh utils/ppm.hpp )
set(SOURCES students/student1.cu students/student2.cu students/student3.cu students/student4.cu main.cu utils/chronoCPU.cpp utils/chronoGPU.cu utils/ppm.cpp )

cuda_add_executable(${EXECUTABLE_NAME} ${SOURCES} ${HEADERS})