#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>

#include <ppm.hpp>

#include "student1.hpp"
#include "student2.hpp"
#include "student3.hpp"
#include "student4.hpp"

// ==========================================================================================
void usage( const char*prg ) {
	#ifdef WIN32
	const char*last_slash = strrchr(prg, '\\');
	#else
	const char*last_slash = strrchr(prg, '/');
	#endif
	std::cout << "Usage: " << (last_slash==nullptr ? prg : last_slash+1) << " -i <image.ppm> [-f <image_output_filtered.ppm>] [-r <image_output_region.ppm>] [-d <size>] [-t <threshold>] [-cuda] "<< std::endl;
	std::cout << "\twhere <image_input.ppm> is the input image, <image_output_filtered.ppm> the output filtered image,"<<std::endl;
	std::cout << "\t<image_output_region.ppm> the image of region, <size> is the width of the median filter,"<<std::endl;
	std::cout << "\t<threshold> is the threshold value for image segmentation, and -cuda for using Cuda (it uses Thrust otherwise)." << std::endl;
}

// ==========================================================================================
void usageAndExit( const char*prg, const int code ) {
	usage(prg);
	exit( code );
}

// ==========================================================================================
void exercise1( const PPMBitmap& input, PPMBitmap& output, const int d, const char*const outputFileName ) 
{
	std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;
	std::cout << "Exercise 1 ... " << std::endl;
	// call student part
	float elapsedTime = student1(input, output, d);
	std::cout << "your implementation runs in " << elapsedTime << " ms."<<std::endl;
	if( outputFileName )
		output.saveTo(outputFileName);
}

// ==========================================================================================
void exercise2( const PPMBitmap& input, PPMBitmap& output, const int d, const char*const outputFileName ) 
{
	std::cout << std::endl;
	std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;
	std::cout << "Exercise 2 ... " << std::endl;
	// call student part
	float elapsedTime = student2(input, output, d);
	std::cout << "your implementation runs in " << elapsedTime << " ms."<<std::endl;
	if( outputFileName )
		output.saveTo(outputFileName);	
}

void exercise3( const PPMBitmap& input, PPMBitmap& output, const int t, const char*const outputFileName ) 
{
	std::cout << std::endl;
	std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;
	std::cout << "Exercise 3 ... " << std::endl;
	// call student part
	float elapsedTime = student3(input, output, t);
	std::cout << "your implementation runs in " << elapsedTime << " ms."<<std::endl;
	if( outputFileName )
		output.saveTo(outputFileName);	
}

void exercise4( const PPMBitmap& input, PPMBitmap& output, const int t, const char*const outputFileName ) 
{
	std::cout << std::endl;
	std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;
	std::cout << "Exercise 4 ... " << std::endl;
	// call student part
	float elapsedTime = student4(input, output, t);
	std::cout << "your implementation runs in " << elapsedTime << " ms."<<std::endl;
	if( outputFileName )
		output.saveTo(outputFileName);		
}

// ==========================================================================================
int main( int ac, char**av)
{
	// parse the command line
	char *input = nullptr;
	char *output_filter = nullptr;
	char *output_region = nullptr;
	bool useThrust = true;
	unsigned d = 3;
	unsigned t = 128;
	for(int i=1; i<ac; ++i) {
		if( !strcmp(av[i], "-cuda") ) 
			useThrust = false;
		else if( !strcmp(av[i], "-i" ))
		{
			if( i+1 == ac ) usageAndExit( av[0], EXIT_FAILURE );
			input = av[++i];
		}
		else if( !strcmp(av[i], "-f" ))
		{
			if( i+1 == ac ) usageAndExit( av[0], EXIT_FAILURE );
			output_filter = av[++i];
		}
		else if( !strcmp(av[i], "-r" ))
		{
			if( i+1 == ac ) usageAndExit( av[0], EXIT_FAILURE );
			output_region = av[++i];
		}
		else if( !strcmp(av[i], "-d") ) 
		{
			if( i+1 == ac ) usageAndExit( av[0], EXIT_FAILURE );
			sscanf(av[++i], "%d", &d);
		}
		else if( !strcmp(av[i], "-t") ) 
		{
			if( i+1 == ac ) usageAndExit( av[0], EXIT_FAILURE );
			sscanf(av[++i], "%d", &t);
		}
		else
			usageAndExit( av[0], EXIT_FAILURE );
	}

	if( input == nullptr )
		usageAndExit( av[0], EXIT_SUCCESS );

	if( output_filter == nullptr ) {
		std::string name(input);
		name.erase(name.size() - 4, 4).append("_filtered.ppm");
		output_filter = strdup( name.c_str() );
	}
	
	if( output_region == nullptr ) {
		std::string name(input);
		name.erase(name.size() - 4, 4).append("_segmented.ppm");
		output_region = strdup( name.c_str() );
	}

	if( d<1 ) d = 1;
	else if ( d>15 ) d = 15;
	if( t>255 ) t = 255;

	std::cout << "Original  image: " << input << std::endl;
	std::cout << "Filtered  image: " << output_filter << std::endl;
	std::cout << "Segmented image: " << output_region << std::endl;
	std::cout << "Filter width: " << d << std::endl;
	std::cout << "Segmentation threshold: " << t << std::endl;
	std::cout << "Computation done using " << (useThrust ? "Thrust" : "Cuda") << std::endl;

	// read the input image
	PPMBitmap in( input );
	PPMBitmap filtered( in.getWidth(), in.getHeight() );
	PPMBitmap segmented( in.getWidth(), in.getHeight() );
	
	if( useThrust ) {
		// call first exercise
		exercise1(in, filtered, d, output_filter);

		// call third exercise
		exercise3(filtered, segmented, t, output_region);
	}
	else {
		// call second exercise
		exercise2(in, filtered, d, output_filter);

		// call last exercise
		exercise4(filtered, segmented, t, output_region);
	}
	return EXIT_SUCCESS;
}
