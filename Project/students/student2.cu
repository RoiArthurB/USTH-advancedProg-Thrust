#include "student2.hpp"
#include "../utils/common.hpp"
#include "../utils/chronoGPU.hpp"
#include "../utils/utils.cuh"

#include <iostream>
#include <string>

/*
 *	+=======================+
 *	|						|
 *	|	DEVICE FUNCTIONS	|
 *	|						|
 *	+=======================+	
 */
 
// converts a RGB color to a HSV one ...
__device__
float3 RGB2HSV( const uchar3 inRGB ) {
	const float R = (float)( inRGB.x ) / 256.f;
	const float G = (float)( inRGB.y ) / 256.f;
	const float B = (float)( inRGB.z ) / 256.f;

	const float min		= fminf( R, fminf( G, B ) );
	const float max		= fmaxf( R, fmaxf( G, B ) );
	const float delta	= max - min;

	// H
	float H;
	if( delta <= 0 )
		H = 0.f;
	else if	( max == R )
		H = 60.f * ( G - B ) / delta + 360.f;
	else if ( max == G )
		H = 60.f * ( B - R ) / delta + 120.f;
	else
		H = 60.f * ( R - G ) / delta + 240.f;
	while	( H >= 360.f )
		H -= 360.f ;

	// S
	float S = max <= 0 ? 0.f : 1.f - min / max;

	// V
	float V = max;

	return make_float3(H, S, V);
}


// converts a HSV color to a RGB one ...
__device__
uchar3 HSV2RGB( const float H, const float S, const float V )
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

	return make_uchar3( R*256.f, G*256.f, B*256.f);//, 255 );
}

/*
 *	+=======================+
 *	|						|
 *	|	GLOBAL  FUNCTIONS	|
 *	|						|
 *	+=======================+	
 */
 
// Conversion from RGB (inRGB) to HSV (outHSV)
// Launched with 2D grid
__global__ 
void rgb2hsv( uchar3 *inRGB, float3 *outHSV, const int width, const int height ) {
	// Calculate tid
    unsigned int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx >= width || tidy >= height) return;
    
	int tid = tidx + (tidy * width);
	
	// Process
	float3 hsv = RGB2HSV( inRGB[tid] );

	// Output
	outHSV[tid] = hsv;
}

// Conversion from HSV (inH, inS, inV) to RGB (outRGB)
// Launched with 2D grid
__global__
void hsv2rgb(	float3 *inHSV, uchar3 *outRGB, const int width, const int height ) {
	// Calculate tid
    unsigned int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx >= width || tidy >= height) return;
    
	int tid = tidx + (tidy * width);

	// Process
	uchar3 rgb = HSV2RGB( inHSV[tid].x, inHSV[tid].y, inHSV[tid].z );

	// Output
	outRGB[tid] = rgb;
}


/* Exercice 2.
* Here, you have to apply the Median Filter on the input image.
* Calculations have to be done using CUDA. 
* You have to measure the computation times, and to return the number of ms 
* your calculation took (do not include the memcpy).
*
* @param in: input image
* @param out: output (filtered) image
* @param size: width of the kernel 
*/
float student2(const PPMBitmap &in, PPMBitmap &out, const int size) {
	ChronoGPU chrUP, chrDOWN, chrGPU;

    // Preparing var
    //======================
    chrUP.start();
    
    int width = in.getWidth(); int height = in.getHeight();
    
    //Calculate number of pixels
    int pixelCount = width * height;
    
    uchar3 *devInput;
    float3 *devHSV;

    //Allocate CUDA memory    
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devHSV, pixelCount * sizeof(float3));
    
    // Get usable input image
    // PPMBitmap => uchar3
    uchar3 hostImage[pixelCount];
    int i = 0;
    for (int w = 0; w < width; w ++) {
    	for (int h = 0; h < height; h ++) {
    		PPMBitmap::RGBcol pixel = in.getPixel( w, h );
    		hostImage[i++] = make_uchar3(pixel.r, pixel.g, pixel.b);
    	}
    }
    
    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, hostImage, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	chrUP.stop();
    
    // Processing
    //======================
    chrGPU.start();
    // Start GPU processing (KERNEL)
    //Create 32x32 Blocks
    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((width  + (blockSize.x-1))/blockSize.x, 
    					 (height + (blockSize.y-1))/blockSize.y );
        
    // Convertion from RGB to HSV
    rgb2hsv<<<gridSize, blockSize>>>(devInput, devHSV, width, height);

	// Median Filter
/*    grayscale2D<<<gridSize, blockSize>>>(devInput, devGray, inputImage->width, inputImage->height);
*/
	// Convertion from HSV to RGB
    hsv2rgb<<<gridSize, blockSize>>>(devHSV, devInput, width, height);

	chrGPU.stop();
	
    // Cleaning
    //======================
    chrDOWN.start();
    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(hostImage, devInput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // uchar3 => PPMBitmap
    i = 0;
    for (int w = 0; w < width; w ++) {
    	for (int h = 0; h < height; h ++) {
    		out.setPixel( w, h, PPMBitmap::RGBcol(hostImage[i].x, hostImage[i].y, hostImage[i].z) );
    		i++;
    	}
    }
    
    // Free CUDA Memory
    cudaFree(&devInput);
	cudaFree(&devHSV);

	chrDOWN.stop();

    // Return
    //======================
    return chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime(); //0.f;
}


































