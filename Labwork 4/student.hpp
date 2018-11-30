/*
* LW 5 - Atomics
* --------------------------
* Histogram equalization
*
* File: student.hpp
*/
#ifndef __STUDENT_HPP
#define __STUDENT_HPP

// Comment to enable GPU computing

#define SIZE_HISTO 256

// ============================================ Exercice 1
#define CONVERSIONS_NOT_IMPLEMENTED
// Conversion from RGB (inRGB) to HSV (outH, outS, outV)
// Launch with 2D grid
__global__
void rgb2hsv(	const uchar4 *const inRGB, const int width, const int height,
				float *const outH, float *const outS, float *const outV );

// Conversion from HSV (inH, inS, inV) to RGB (outRGB)
// Launch with 2D grid
__global__
void hsv2rgb(	const float *const inH, const float *const inS, const float *const inV,
				const int width, const int height, uchar4 *const outRGB );

// ============================================ Exercice 2
#define HISTO_NOT_IMPLEMENTED
// Compute histogram
// Launched with dimBlock = SIZE_HISTO
__global__
void histo( const float *const inV, const int sizeV, unsigned int *const outHisto );

// ============================================ Exercice 3
#define REPART_NOT_IMPLEMENTED
// Compute repart function
// Launched on 1 block of SIZE_HISTO threads
__global__
void repart( const unsigned int *const inHisto, unsigned int *const outRepart );

// Equalized final image
// Launch with 2D grid
__global__
void equalization( const unsigned int *const inRepart, const int sizeRepart, 
					float *const outV, const int sizeV );

#endif
