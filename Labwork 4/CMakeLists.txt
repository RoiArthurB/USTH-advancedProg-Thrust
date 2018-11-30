cmake_minimum_required(VERSION 3.0)
set(EXECUTABLE_NAME LW4)
set(PROJECT_NAME LW4)

project(${EXECUTABLE_NAME} LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
	
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")  
set(CMAKE_CXX_STANDARD 11)

find_package(OpenGL REQUIRED)

set(CMAKE_PREFIX_PATH "d:/glew-2.1.0") 
set(CMAKE_LIBRARY_PATH "d:/glew-2.1.0/lib/Release/x64/") 
find_package(GLEW REQUIRED)

find_package(CUDA REQUIRED)
if (NOT CUDA_FOUND)
    message("CUDA location not found")
endif()

#set(CUDA_VERBOSE_BUILD ON)

if (WIN32)
    set(CUDA_SAMPLES "D:/CUDA Samples/v10.0/common/")
    set(GLEW_LIBS_PATH "${CUDA_SAMPLES}lib/x64/")
    list(APPEND CUDA_NVCC_FLAGS -O3 -Wno-deprecated-gpu-targets --gpu-architecture=sm_52 --expt-extended-lambda)
    link_libraries("${GLEW_LIBS_PATH}freeglut.lib" "${GLEW_LIBS_PATH}glew64.lib") 
endif(WIN32)
if (LINUX)
    set(CUDA_SAMPLES "/usr/local/cuda/samples/common/")
    list(APPEND CUDA_NVCC_FLAGS -std=c++11 -O3 -Wno-deprecated-gpu-targets --gpu-architecture=sm_30 --expt-extended-lambda --disable-warnings)
    link_libraries(-lfreeglut -lglew64) 
endif(LINUX)

include_directories(utils ${GLEW_INCLUDE_DIR} "${CUDA_SAMPLES}inc")
set(HEADERS student.hpp app.hpp utils/chronoGPU.hpp utils/chronoCPU.hpp utils/utils.cuh utils/common.hpp utils/ppm.hpp )
set(SOURCES main.cu student.cu app.cu utils/chronoCPU.cpp utils/chronoGPU.cu utils/ppm.cpp )

cuda_add_executable(${EXECUTABLE_NAME} ${SOURCES} ${HEADERS})

add_custom_command(TARGET ${EXECUTABLE_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_SAMPLES}../bin/win64/Release/glew64.dll" ${PROJECT_BINARY_DIR}/.
                   VERBATIM )
add_custom_command(TARGET ${EXECUTABLE_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_SAMPLES}../bin/win64/Release/freeglut.dll" ${PROJECT_BINARY_DIR}/.
                   VERBATIM )