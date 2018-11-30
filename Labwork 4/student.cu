/*
* LW 5 - Atomics
* --------------------------
* Histogram equalization
*
* File: student.cu
*/

#include "student.hpp"
#include "utils/common.hpp"
#include "utils/chronoGPU.hpp"
#include "utils/utils.cuh"

// converts a RGB color to a HSV one ...
__device__
float3 RGB2HSV( const uchar4 inRGB ) {
	const float R = (float)( inRGB.x ) / 256.f;
	const float G = (float)( inRGB.y ) / 256.f;
	const float B = (float)( inRGB.z ) / 256.f;

	const float min		= fminf( R, fminf( G, B ) );
	const float max		= fmaxf( R, fmaxf( G, B ) );
	const float delta	= max - min;

	// H
	float H;
	if( delta < FLT_EPSILON )
		H = 0.f;
	else if	( max == R )
		H = 60.f * ( G - B ) / ( delta + FLT_EPSILON )+ 360.f;
	else if ( max == G )
		H = 60.f * ( B - R ) / ( delta + FLT_EPSILON ) + 120.f;
	else
		H = 60.f * ( R - G ) / ( delta + FLT_EPSILON ) + 240.f;
	while	( H >= 360.f )
		H -= 360.f ;

	// S
	float S = max < FLT_EPSILON ? 0.f : 1.f - min / max;

	// V
	float V = max;

	return make_float3(H, S, V);
}


// converts a HSV color to a RGB one ...
__device__
uchar4 HSV2RGB( const float H, const float S, const float V )
{
	const float	d	= H / 60.f;
	const int	hi	= (int)d % 6;
	const float f	= d - (float)hi;

	const float l   = V * ( 1.f - S );
	const float m	= V * ( 1.f - f * S );
	const float n	= V * ( 1.f - ( 1.f - f ) * S );

	float R, G, B;

	if		( hi == 0 )
		{ R = V; G = n;	B = l; }
	else if ( hi == 1 )
		{ R = m; G = V;	B = l; }
	else if ( hi == 2 )
		{ R = l; G = V;	B = n; }
	else if ( hi == 3 )
		{ R = l; G = m;	B = V; }
	else if ( hi == 4 )
		{ R = n; G = l;	B = V; }
	else
		{ R = V; G = l;	B = m; }

	return make_uchar4( R*256.f, G*256.f, B*256.f, 255 );
}


// ============================================ Exercise 1
// Conversion from RGB (inRGB) to HSV (outH, outS, outV)
// Launched with 2D grid
__global__
void rgb2hsv(	const uchar4 *const inRGB, const int width, const int height,
				float *const outH, float *const outS, float *const outV ) {
	/// TODO
}

// Conversion from HSV (inH, inS, inV) to RGB (outRGB)
// Launched with 2D grid
__global__
void hsv2rgb(	const float *const inH, const float *const inS, const float *const inV, 
				const int width, const int height, uchar4 *const outRGB ) {	
	/// TODO
}

// ============================================ Exercise 2
// Compute histogram
// Launched in 1D with dimBlock = SIZE_HISTO
__global__
void histo( const float *const inV, const int sizeV, unsigned int *const outHisto ) {
	/// TODO 
}

// ============================================ Exercise 3
// Compute repart function
// Launched on 1 block of SIZE_HISTO threads
__global__
void repart( const unsigned int *const inHisto, unsigned int *const outRepart ) {
	/// TODO 
}

// Equalized final image
// Launched with 2D grid
__global__
void equalization( const unsigned int *const inRepart, const int sizeRepart, 
					float *const outV, const int sizeV ) {
	__shared__ float s_repart[SIZE_HISTO];
	if(threadIdx.x < SIZE_HISTO)
		s_repart[threadIdx.x] = inRepart[threadIdx.x]/static_cast<float>(sizeV);
	__syncthreads();
	for( int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < sizeV;
			tid += blockDim.x * gridDim.x ) {
		unsigned val = static_cast<unsigned>( outV[tid]*256.f );
		outV[tid] = (float)(sizeRepart - 1) / (float)(sizeRepart) * s_repart[val];
	}
}
