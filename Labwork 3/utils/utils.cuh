#ifndef __UTILS_CUH
#define __UTILS_CUH

#include <iostream>

#include <stdarg.h>
#include <cmath>

#include "common.hpp"

__host__ __device__
static uchar3 floatToRGB( const float &l ) {
	const float H = 360.f * l;

	const float Hp = H / 60.f;
	
	uchar3 res;

	if ( Hp < 1.f ) {
		const float X = ( 1.f - fabsf( Hp - 1.f ) );
		res.x = 0;
		res.y = static_cast<unsigned char>(255.f * X);
		res.z = 255;
	}
	else if ( Hp < 2.f ) {
		const float X = ( 1.f - fabsf( Hp - 2.f ) );
		res.x = static_cast<unsigned char>(255.f * X);
		res.y = 0;
		res.z = 255;
	}
	else if ( Hp < 3.f ) {
		const float X = ( 1.f - fabsf( Hp - 3.f ) );
		res.x = 255;
		res.y = 0;
		res.z = static_cast<unsigned char>(255.f * X);
	}
	else if ( Hp < 4.f ) {
		const float X = ( 1.f - fabsf( Hp - 4.f ) );
		res.x = 255;
		res.y = static_cast<unsigned char>(255.f * X);
		res.z = 0;
	}
	else if ( Hp < 5.f ) {
		const float X = ( 1.f - fabsf( Hp - 5.f ) );
		res.x = static_cast<unsigned char>(255.f * X);
		res.y = 255;
		res.z = 0;
	}
	else if ( Hp < 6.f ) {
		const float X = ( 1.f - fabsf( Hp - 6.f ) );
		res.x = 0;
		res.y = 255;
		res.z = static_cast<unsigned char>(255.f * X);
	}
	else {
		res.x = 0;
		res.y = 0;
		res.z = 0;
	}
	return res;
}

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
