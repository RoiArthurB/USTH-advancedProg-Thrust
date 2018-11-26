#ifndef __COMMON_HPP
#define __COMMON_HPP

#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define HANDLE_ERROR(_exp) do {											\
    const cudaError_t err = (_exp);										\
    if ( err != cudaSuccess ) {											\
        std::cerr	<< cudaGetErrorString( err ) << " in " << __FILE__	\
					<< " at line " << __LINE__ << std::endl;			\
        exit( EXIT_FAILURE );											\
    }																	\
} while (0)

#endif

