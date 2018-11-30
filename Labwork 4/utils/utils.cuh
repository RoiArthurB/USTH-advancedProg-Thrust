#ifndef __UTILS_CUH
#define __UTILS_CUH

#include <iostream>

#include <float.h>

#include <stdarg.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  include <intrin.h>
#endif

#include "common.hpp"

__host__ __device__
static float clampf( const float val, const float min , const float max ) {
#ifdef __CUDACC__
	return fminf( max, fmaxf( min, val ) );
#else
	return std::min<float>( max, std::max<float>( min, val ) );
#endif
}

static std::string getNameCPU() {
	std::string name;
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	// Get extended ids.
	int CPUInfo[4] = {-1};
	__cpuid(CPUInfo, 0x80000000);
	unsigned int nExIds = CPUInfo[0];
 
	// Get the information associated with each extended ID.
	char CPUBrandString[0x40] = { 0 };
	for( unsigned int i=0x80000000; i<=nExIds; ++i)
	{
		__cpuid(CPUInfo, i);
 
		// Interpret CPU brand string and cache information.
		if  (i == 0x80000002)
			memcpy( CPUBrandString, CPUInfo, sizeof(CPUInfo) );
		else if( i == 0x80000003 )
			memcpy( CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
		else if( i == 0x80000004 )
			memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
	}
	name = CPUBrandString;
#else
	name = "??? On Linux or MacOS, check system information ! :-p";
#endif
	return name;
}

#endif
