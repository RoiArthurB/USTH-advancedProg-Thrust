/*
* LW 5 - Atomics
* --------------------------
* Histogram equalization
*
* File: app.hpp
*/

#ifndef __APP_HPP
#define __APP_HPP

// GL includes
//#include "GL/glew.h"
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	#include "GL/wglew.h"
	#if defined(WIN64) || defined(_WIN64)
		#pragma comment(lib, "glew64.lib")
	#else
		#pragma comment(lib, "glew32.lib")
	#endif
#endif
#if defined(__APPLE__) || defined(__MACOSX)
	#include <GLUT/glut.h>
	#ifndef glutCloseFunc
		#define glutCloseFunc glutWMCloseFunc
	#endif
	#include <OpenGL/gl.h>
	#include <OpenGL/glu.h>
#else
	#include "GL/freeglut.h"
	#include "GL/gl.h"
	#include "GL/glu.h"
#endif

#include <string>
#include "utils/common.hpp"

#define WIDTH_IMG_HISTO		512
#define HEIGHT_IMG_HISTO	512

class App { // Singleton
private:
	App() {}; // undefined
public:
	~App() { cleanAndExit(); delete m_instance; }

	static App *createInstance();
	static App *getInstance() { return App::m_instance; }

	void launch( int &argc, char **argv );

protected: 
	static App *m_instance;
	
	bool		m_equalized;
	bool		m_haveToWork;
	
	// GL buffers
	GLuint					m_pboGL;
	cudaGraphicsResource	*m_rcCUDA;
	
	// Pixels buffers
	int m_width;
	int m_height;
	
	// images on host
	uchar4	*m_hostImgSource;
	uchar4	*m_hostImgEqualized;

	// Data on device
	uchar4			*m_devImgSrc;
	uchar4			*m_devImgOut; // display
	unsigned int	*m_devHisto;
	unsigned int	*m_devRepart;

	// HSV on device
	float *m_devH;
	float *m_devS;
	float *m_devV;

	// Histos on device

	std::string m_appName;
	std::string m_imgFile;
	
	std::string		m_nameCPU;
	cudaDeviceProp	m_propGPU;
	
	// GPU configuration
	dim3 m_dimGrid;		// ie. number of blocks / grid
	dim3 m_dimBlock;	// ie. number of threads / block
	dim3 m_dimBlockHisto;	// ie. number of threads / block

	// Initialization
	void initData( int argc, char **argv );
	void initGL();
	void initGPU();
	void initDataGPU();
	void loadImgSrc( const std::string &imgName );
	
	// GL functions
	static void idle();
	static void display();
	static void handleKeys( unsigned char key, int, int );
	static void cleanAndExit();
	void setWindowTitle( const char *s, ... ); 
	
	// Others...
	void saveToPPM();
	void printUsage();
	void printControls();
	int	 chooseBestDevice();

	void checkHisto();
	void checkRepart();
};

#endif





