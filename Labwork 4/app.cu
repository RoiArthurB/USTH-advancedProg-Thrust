/*
* LW 5 - Atomics
* --------------------------
* Histogram equalization
*
* File: app.cu
*/

#include "app.hpp"
// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "helper_cuda.h"
#include "helper_gl.h"
#include <iostream>
#include <iomanip>     
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include "utils/chronoCPU.hpp"
#include "utils/chronoGPU.hpp"
#include "utils/utils.cuh"
#include "utils/ppm.hpp"
#include "student.hpp"

App *App::m_instance = nullptr; // Singleton

App *App::createInstance() {
	if ( m_instance == nullptr )
		m_instance = new App();
	else
		std::cerr << "Instance already created" << std::endl;
	
	return m_instance;
}

void App::initData( int argc, char **argv ) {
	App *app = getInstance();

	std::cout << "Initializing data..." << std::endl;
	
	app->m_appName	= "Histogram equalization";
	app->m_imgFile	= "Undefined";
	
	// get source image
	if ( argc == 1 ) {
		std::cerr << "Please give a file..." << std::endl;
		printUsage();
		exit(EXIT_FAILURE);
	}
	char inputName[2048];
	for ( int i = 1; i < argc; ++i ) {
		if ( !strcmp( argv[i], "-f" ) ) {
			if ( sscanf( argv[++i], "%s", inputName ) != 1 ) {
				printUsage();
				exit(EXIT_FAILURE);
			}
			else
				app->m_imgFile = inputName;
		}
		else {
			printUsage();
			exit(EXIT_FAILURE);
		}
	}


	app->m_equalized = false;
	app->m_haveToWork = true;
	app->m_pboGL	= (GLuint)NULL;
	app->m_rcCUDA	= NULL;

	app->loadImgSrc( app->m_imgFile );
	app->m_hostImgEqualized	= new uchar4[app->m_width * app->m_height];

	app->m_devImgOut	= NULL;
	app->m_devH			= NULL;
	app->m_devS			= NULL;
	app->m_devV			= NULL;
	app->m_devHisto		= NULL;
	
	app->m_propGPU	= cudaDeviceProp();
	app->m_nameCPU	= getNameCPU(); 
	

	std::cout << "-> Done." << std::endl;
}

void App::initGL() {
	App *app = getInstance();
	std::cout << "Initializing GLUT..." << std::endl;

	int argc = 1;
    char *argv = (char *)"";
	glutInit( &argc, &argv );
	glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowSize( app->m_width, app->m_height );
    glutCreateWindow( app->m_appName.c_str() );

	glutDisplayFunc(App::display);
	glutIdleFunc(App::idle);
    glutKeyboardFunc(App::handleKeys);
	
#if defined(__APPLE__) || defined(__MACOSX)
	atexit( App::cleanAndExit );
#else
	glutCloseFunc( App::cleanAndExit );
#endif

    std::cout << "-> Done." << std::endl;

	std::cout << "Initializing glew..." << std::endl;

	if ( glewInit() != GLEW_OK ) {
		std::cerr << "Error: " << glewGetErrorString( glewInit() );
		app->cleanAndExit();
	}

    if (!glewIsSupported("GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        std::cerr << "Error: failed to get minimal extensions for demo" << std::endl
			<< "This application requires:"			<< std::endl
			<< "  - OpenGL version 1.5"				<< std::endl
			<< "  - GL_ARB_vertex_buffer_object"	<< std::endl
			<< "  - GL_ARB_pixel_buffer_object"		<< std::endl << std::endl;
        app->cleanAndExit();
    }

	glGenBuffers( 1, &app->m_pboGL);
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, app->m_pboGL);
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, app->m_width * app->m_height * sizeof(uchar4), app->m_hostImgEqualized, GL_STREAM_COPY );
	
	HANDLE_ERROR( cudaGraphicsGLRegisterBuffer( &app->m_rcCUDA, app->m_pboGL, cudaGraphicsMapFlagsWriteDiscard ) );

    std::cout << "-> Done." << std::endl;
}

void App::initGPU() {
	App *app = getInstance();	
	std::cout << "Initializing GPU..." << std::endl;

	// Get number of CUDA capable devices
	int nbDev;
	HANDLE_ERROR( cudaGetDeviceCount( &nbDev ) );

	if ( nbDev == 0 ) {
		std::cerr << "Error: no CUDA capable device" << std::endl;
		app->cleanAndExit();
	}

	int bestDev = app->chooseBestDevice();

	// Set app->m_propGPU
	HANDLE_ERROR( cudaGetDeviceProperties( &app->m_propGPU, bestDev ) );

	// Requires compute capability >= 1.3
	if ( bestDev == -1 || ( app->m_propGPU.major < 2 ) ) {
		std::cerr << "Error: no CUDA capable device is detected with compute capability >= 2.x" << std::endl;
		app->cleanAndExit();
	}
    HANDLE_ERROR( cudaSetDevice( bestDev ) );
	
	// Configure thread distribution (only for image sized kernels)
	app->m_dimGrid	= dim3( ( app->m_width + 15 ) / 16, ( app->m_height + 15 ) / 16 );
	app->m_dimBlock	= dim3( 16, 16 );
	unsigned int sizeImg = app->m_width * app->m_height;
	unsigned int dimBlockHisto = ( sizeImg + SIZE_HISTO - 1 ) / SIZE_HISTO;

	// small number of block to emphasize the need of local memory ...
	app->m_dimBlockHisto	= dim3( std::min<unsigned int>( dimBlockHisto, 8*app->m_propGPU.multiProcessorCount ) );

	// Set GL
    //HANDLE_ERROR( cudaGLSetGLDevice( bestDev ) );
	cudaDeviceSynchronize();
	std::cout << "-> Done." << std::endl;
}

void App::initDataGPU() {
	App *app = getInstance();	

	// Image source
	const unsigned int nbPixels =  app->m_width * app->m_height;
	HANDLE_ERROR( cudaMalloc( (void**)&app->m_devImgSrc, nbPixels * sizeof(uchar4) ) );
	HANDLE_ERROR( cudaMemcpy( app->m_devImgSrc, app->m_hostImgSource, nbPixels * sizeof(uchar4), cudaMemcpyHostToDevice) );
	
	// HSV
	HANDLE_ERROR( cudaMalloc( (void**)&app->m_devH, nbPixels * sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&app->m_devS, nbPixels * sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&app->m_devV, nbPixels * sizeof(float) ) );
	
	// Histo and repart arrays
	HANDLE_ERROR( cudaMalloc( (void**)&app->m_devHisto,  sizeof(unsigned int) * SIZE_HISTO ) );
	HANDLE_ERROR( cudaMemset( app->m_devHisto, 0, sizeof(unsigned int) * SIZE_HISTO ) );
	HANDLE_ERROR( cudaMalloc( (void**)&app->m_devRepart, sizeof(unsigned int) * SIZE_HISTO ) );
	HANDLE_ERROR( cudaMemset( app->m_devRepart, 0, sizeof(unsigned int) * SIZE_HISTO ) );
}

void App::loadImgSrc( const std::string &imgName ) {
	App *app = getInstance();

	PPMBitmap in( imgName.c_str() );
	
	app->m_width = in.getWidth(); app->m_height = in.getHeight();
	app->m_hostImgSource = new uchar4[app->m_width * app->m_height];
	
	for ( int y = 0; y < app->m_height; ++ y ) {
		for ( int x = 0; x < app->m_width; ++ x ) {
			const int idPixel = x + y * app->m_width;
			const PPMBitmap::RGBcol col = in.getPixel( x, y );
			app->m_hostImgSource[idPixel].x = col.r;
			app->m_hostImgSource[idPixel].y = col.g;
			app->m_hostImgSource[idPixel].z = col.b;
			app->m_hostImgSource[idPixel].w = 255;
		}
	}
}

void App::launch( int &argc, char **argv ) {
	App *app = getInstance();
	std::cout	<< "==============================================="	<< std::endl
				<< "            Initializing application           "	<< std::endl
				<< "==============================================="	<< std::endl << std::endl;
	app->initData( argc, argv );
	app->initGPU();
	app->initGL();
	app->initDataGPU();
	
	std::cout	<< "==============================================="	<< std::endl
				<< "        Computing histogram equalization       "	<< std::endl
				<< "==============================================="	<< std::endl << std::endl;
	
	std::cout << "Image has " << app->m_width << " x " << app->m_height << std::endl << std::endl;
		
	glutMainLoop();

	cudaDeviceReset();
}

void App::idle() {
	App *app = getInstance();
	
	float time = 0.f;

	if ( !app->m_rcCUDA )
		return;
    	
	HANDLE_ERROR( cudaGraphicsMapResources( 1, &app->m_rcCUDA, NULL ) );
	size_t size;
	HANDLE_ERROR( cudaGraphicsResourceGetMappedPointer( (void **)&app->m_devImgOut, &size, app->m_rcCUDA ) );
	if ( app->m_equalized ) {
		if ( app->m_haveToWork ) {
			std::cout << "==============================" << std::endl;
			ChronoGPU chr;
			chr.start();

			// RGB to HSV
			rgb2hsv<<<app->m_dimGrid, app->m_dimBlock>>>( app->m_devImgSrc, app->m_width, app->m_height, 
															app->m_devH, app->m_devS, app->m_devV ); 		
			chr.stop();
			time = chr.elapsedTime();
			std::cout << "RGB to HSV: " << time << " ms." << std::endl;
			std::cout << "==============================" << std::endl;

#ifndef HISTO_NOT_IMPLEMENTED
			chr.start();

			// Histogram
			histo<<<app->m_dimBlockHisto, SIZE_HISTO>>>( app->m_devV, app->m_width * app->m_height, app->m_devHisto );
			
			chr.stop();
			time = chr.elapsedTime();
			std::cout << "Histogram: " << time << " ms." << std::endl;
			app->checkHisto();
			std::cout << "==============================" << std::endl;
#endif

#ifndef REPART_NOT_IMPLEMENTED
			chr.start();

			// Repartition
			repart<<<1, SIZE_HISTO>>>( app->m_devHisto, app->m_devRepart ); 

			chr.stop();
			time = chr.elapsedTime();
			std::cout << "Repart: " << time << " ms." << std::endl;
			app->checkRepart();
			std::cout << "==============================" << std::endl;

			chr.start();

			// Equalization
			equalization<<<app->m_dimBlockHisto,SIZE_HISTO>>>( app->m_devRepart, SIZE_HISTO, app->m_devV, app->m_width * app->m_height );

			chr.stop();
			time = chr.elapsedTime();
			std::cout << "Equalization: " << time << " ms." << std::endl;
			std::cout << "==============================" << std::endl;

#endif
			chr.start();
			// HSV to RGB
			hsv2rgb<<<app->m_dimGrid, app->m_dimBlock>>>( app->m_devH, app->m_devS, app->m_devV, 
															app->m_width, app->m_height, app->m_devImgOut ); 

			chr.stop();
			time = chr.elapsedTime();
			std::cout << "HSV to RGB: " << time << " ms." << std::endl;
			std::cout << "==============================" << std::endl;

			std::cout << "Image should be equalized" << std::endl;

			HANDLE_ERROR( cudaMemcpy( app->m_hostImgEqualized,app->m_devImgOut, 
								app->m_width * app->m_height * sizeof(uchar4), 
								cudaMemcpyDeviceToHost ) ); // Save result

			app->m_haveToWork = false;
		}
		// Copy display
		HANDLE_ERROR( cudaMemcpy( app->m_devImgOut, app->m_hostImgEqualized, 
								app->m_width * app->m_height * sizeof(uchar4), 
								cudaMemcpyHostToDevice ) ); // Display source

	}
	else {	
		// Copy display
		HANDLE_ERROR( cudaMemcpy( app->m_devImgOut, app->m_hostImgSource, 
								app->m_width * app->m_height * sizeof(uchar4), 
								cudaMemcpyHostToDevice ) ); // Display source
	}
			
	HANDLE_ERROR( cudaGraphicsUnmapResources( 1, &app->m_rcCUDA, NULL ) );
	glutPostRedisplay();
}

void App::display() {
	App *app = getInstance();
	
	glClearColor( 0.f, 0.f, 0.f, 1.f );
    glClear( GL_COLOR_BUFFER_BIT );
	
    GLint iViewport[4];
    glGetIntegerv(GL_VIEWPORT, iViewport);
    glPixelZoom(static_cast<float>(iViewport[2])/static_cast<float>(app->m_width),static_cast<float>(iViewport[3])/static_cast<float>(app->m_height));
    glDrawPixels( static_cast<GLsizei>( app->m_width ), 
				static_cast<GLsizei>( app->m_height ), GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
	
	app->setWindowTitle("%s - %s (source size: %d x %d)", 
						app->m_appName.c_str(), 
#ifndef EQUALIZATION_NOT_IMPLEMENTED
						app->m_equalized ? "Equalized" : 
#endif	
						"Source",
						app->m_width, app->m_height );	
}

void App::handleKeys( unsigned char key, int, int ) {
	App *app = getInstance();

	switch( key ) {
	case 27: //ESC
		app->cleanAndExit();
		break;
	case ' ': // SPACE
#ifndef CONVERSIONS_NOT_IMPLEMENTED
		app->m_equalized = !app->m_equalized;
#else
		std::cerr << "Please implement something! (CONVERSIONS_NOT_IMPLEMENTED is defined)" << std::endl;
#endif
		break;
	case 'p' : case 'P' :
		app->saveToPPM();
		break;
	default:
		break;
	}
}

void App::cleanAndExit() {
	App *app = getInstance();
	if( app == NULL ) return;
	App::m_instance = NULL;

	std::cout << "Clean up before exit" << std::endl;
	if ( app->m_hostImgSource ) {
		delete app->m_hostImgSource;
		app->m_hostImgSource = NULL;
	}
	if ( app->m_hostImgEqualized ) {
		delete app->m_hostImgEqualized;
		app->m_hostImgEqualized = NULL;
	}
	
	cudaFree( app->m_devImgSrc );
	cudaFree( app->m_devImgOut );
	cudaFree( app->m_devHisto );
	cudaFree( app->m_devRepart );
	cudaFree( app->m_devH );
	cudaFree( app->m_devS );
	cudaFree( app->m_devV );

	cudaGraphicsUnregisterResource( app->m_rcCUDA );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, (GLuint)NULL );
	glDeleteBuffers( 1, &app->m_pboGL );
	
	cudaDeviceReset();

	std::cout << std::endl << "Goodbye!" << std::endl;
	exit( EXIT_SUCCESS );
}

void App::setWindowTitle( const char *s, ... ) {
	va_list ap;
	va_start( ap, s );

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	unsigned int length = static_cast<unsigned int>( _vscprintf( s, ap ) );
#else
	unsigned int length = 4096;
#endif

	char *buffer = new char[length +1];

	vsprintf( buffer, s, ap );

	va_end( ap );

	glutSetWindowTitle( buffer );

	delete[] buffer;
}

void App::saveToPPM() {
	App *app = getInstance();

	if ( app->m_equalized ) {
#ifndef EQUALIZATION_NOT_IMPLEMENTED
		std::string ppmName = app->m_imgFile;
		ppmName.erase( ppmName.end() - 4, ppmName.end() ); // erase .ppm
		ppmName += "_";
		ppmName += "Equalized";
		ppmName += ".ppm";

		std::cout << "Saving image as " << ppmName << std::endl;

		PPMBitmap out( app->m_width, app->m_height );
		for ( int i = 0; i < app->m_width; ++i ) {
			for ( int j = 0; j < app->m_height; ++j ) {
				const uchar4 &col = app->m_hostImgEqualized[i + j * app->m_width];
				out.setPixel( i, j, PPMBitmap::RGBcol( col.x, col.y, col.z ) );
			}
		}
		out.saveTo( ppmName.c_str() );													

		std::cout << "-> Done." << std::endl;
#else
		std::cout << "Finish to code if you want me to save the result!" << std::endl;
#endif
	}
	else {
		std::cout << "I won't save the source image!" << std::endl;
	}
}

void App::printUsage() {
	std::cout	<< "Usage: " << std::endl
				<< " -f <F>: F = image path (PPM P6 file needed)" 
				<< std::endl << std::endl;
}

void App::printControls() {
	std::cout	<< "Controls: "														<< std::endl
				<< " - Press <space> to switch between source and equlized image"	<< std::endl
				<< " - Press <p> to save the current image in a PPM file"			<< std::endl
				<< std::endl;
}

int App::chooseBestDevice() {
	// Get number of CUDA capable devices
	int nbDev;
	HANDLE_ERROR( cudaGetDeviceCount( &nbDev ) );

	if ( nbDev == 0 ) {
		std::cerr << "Error: no CUDA capable device" << std::endl;
		exit( EXIT_FAILURE );
	}

	// Choose best device
	int currentDev	= 0;
	int bestDev		= -1;
	int bestMajor	= 0;
	int bestMinor   = 0;
	cudaDeviceProp propDev;
	while ( currentDev < nbDev ) {
		HANDLE_ERROR( cudaGetDeviceProperties( &propDev, currentDev ) );
		if ( propDev.major > bestMajor ) {
			bestDev		= currentDev;
			bestMajor	= propDev.major;
			bestMinor   = propDev.minor;
		}
		else if (propDev.minor >= bestMinor ) {
			bestDev = currentDev;
			bestMinor = propDev.minor;
		}
		++currentDev;
	}
	printf("Best device: %d (%d.%d)\n", bestDev, bestMajor, bestMinor);
	return bestDev;
}

void App::checkHisto() {
	App *app = getInstance();
	unsigned int *histo = new unsigned int[SIZE_HISTO];
	cudaMemcpy( histo, app->m_devHisto, SIZE_HISTO * sizeof(unsigned int), cudaMemcpyDeviceToHost ); 
	float *hostV = new float[app->m_width * app->m_height];
	cudaMemcpy( hostV, app->m_devV, app->m_width * app->m_height * sizeof(float), cudaMemcpyDeviceToHost ); 
	for( long i = 0; i < app->m_width * app->m_height; ++i ) 
		--histo[(unsigned int)( hostV[i] * 256.f )];
	for( int i = 0; i < SIZE_HISTO; ++i ) {
		if( histo[i] != 0 ) {
			std::cerr<< "Histogram error at index " << i << std::endl;
			delete hostV;
			delete histo;
			app->cleanAndExit();
		}
	}
	std::cout << "No error" << std::endl;
	delete hostV;
	delete histo;
}

void App::checkRepart() {
	App *app = getInstance();
	unsigned int *repartCmp	= new unsigned int[SIZE_HISTO];
	unsigned int *repart	= new unsigned int[SIZE_HISTO];
	cudaMemcpy( repart, app->m_devRepart, SIZE_HISTO * sizeof(unsigned int), cudaMemcpyDeviceToHost ); 
	unsigned int *histo		= new unsigned int[SIZE_HISTO];
	cudaMemcpy( histo, app->m_devHisto, SIZE_HISTO * sizeof(unsigned int), cudaMemcpyDeviceToHost ); 
	repartCmp[0] = histo[0];
	for( unsigned i = 1; i < SIZE_HISTO; ++i )
		repartCmp[i] = histo[i] + repartCmp[i - 1]; 
	for( int i = 0; i < SIZE_HISTO; ++i ) {
		if( repart[i] != repartCmp[i] ) {
			std::cerr<<"Repart error ar index " << i << std::endl;
			delete repartCmp;
			delete repart;
			delete histo;
			app->cleanAndExit();
		}
	}
	std::cout << "No error" << std::endl;
	delete repartCmp;
	delete repart;
	delete histo;
}
