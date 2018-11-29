#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>

#include "student.hpp"

// ==========================================================================================
void usage( const char*prg ) {
	std::cout << "Usage: " << prg << " [-n N] [-h]"<< std::endl;
	std::cout << "\twhere N is the size of the used array in each exercise,"<<std::endl;
	std::cout << "\tand option -h display usage." << std::endl;
}

// ==========================================================================================
void usageAndExit( const char*prg, const int code ) {
	usage(prg);
	exit( code );
}

// ==========================================================================================
void exercise1( const int size ) {
	std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;
	std::cout << "Exercise 1 ... " << std::endl;
	// creates the input for
	thrust::host_vector<ColoredObject> cObjects;
	cObjects.reserve(size);
	auto coin_rand = std::bind(std::uniform_int_distribution<int>(0,1), std::mt19937_64());
	std::cout << "Fill the input array with "<<size<<" colored objects."<<std::endl;
	int nbBlue = 0;
	for(int i=0; i<size; ++i) {		
		if( coin_rand() == 1 ) {
			cObjects.push_back(ColoredObject(ColoredObject::BLUE));
			++nbBlue;
		}
		else { // obtain 0
			cObjects.push_back(ColoredObject(ColoredObject::RED));
		}
	}
	std::cout << "Input array contains "<<nbBlue<<" blue objets."<<std::endl;
	// call the student version ...
	thrust::host_vector<ColoredObject> student = compactBlue(cObjects);
	if( nbBlue != student.size() )
		std::cout << "Error: the result does not contain the good number of ColoredObject objects ..." <<std::endl;
	else {
		for(thrust::host_vector<ColoredObject>::iterator iter=student.begin(); iter<student.end(); ++iter)
			if((*iter).color != ColoredObject::BLUE) {
				std::cout << "Error: the result does not contain only BLUE ColoredObject objects ..." <<std::endl;
				break;
			}
	}
}

// ==========================================================================================
void exercise2( const int size ) {
	std::cout << std::endl << std::endl;
	std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;
	std::cout << "Exercise 2 ... " << std::endl;
	std::cout << "Fill the input array with "<<size<<" integers."<<std::endl;
	thrust::host_vector<int> iArray, v;
	iArray.reserve(size);
	v.reserve(size);
	for(int i=0; i<size; ++i) v.push_back(i); // 0, 1, 2, ... size-1
	for (int i = 0; i < size; i++) { // shuffle the sequence
		const int pos = rand() % v.size();
		iArray.push_back(v[pos]);
		v[pos] = v.back();
		v.pop_back();
	}
	// call the student work
	thrust::host_vector<int> student = radixSort(iArray);
	if( size != student.size() )
		std::cout << "Error: the result does not contain the good number of integers ..." <<std::endl;
	else {
		for(int i=1; i<size; ++i)
			if(student[i-1] > student[i]) {
				std::cout << "Error: the result is not sorted (position "<<i<<") ..." <<std::endl;
				break;
			}
	}
}

void exercise3( const int size ) {
	std::cout << std::endl << std::endl;
	std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;
	std::cout << "Exercise 3 ... " << std::endl;
	std::cout << "Fill the input array with "<<size<<" integers."<<std::endl;
	thrust::host_vector<int> iArray, v;
	iArray.reserve(size);
	v.reserve(size);
	for(int i=0; i<size; ++i) v.push_back(i); // 0, 1, 2, ... size-1
	for (int i = 0; i < size; i++) { // shuffle the sequence
		const int pos = rand() % v.size();
		iArray.push_back(v[pos]);
		v[pos] = v.back();
		v.pop_back();
	}
	// call the student work
	thrust::host_vector<int> student = quickSort(iArray);
	if( size != student.size() )
		std::cout << "Error: the result does not contain the good number of integers ..." <<std::endl;
	else {
		for(int i=1; i<size; ++i)
			if(student[i-1] > student[i]) {
				std::cout << "Error: the result is not sorted (position "<<i<<") ..." <<std::endl;
				break;
			}
	}
}

// ==========================================================================================
int main( int ac, char**av)
{
	// parse the command line
	int size = 1<<20;
	for(int i=1; i<ac; ++i) {
		if( !strcmp(av[i], "-n") ) {
			if( i+1 == ac ) usageAndExit( av[0], EXIT_FAILURE );
			int n = size;
			if( sscanf( av[++i], "%d", &n ) != 1 ) usageAndExit(av[0], EXIT_FAILURE);
			if ((n > 0) && (n < (1<<30)))
				size = n;
		}
		else if ( !strcmp(av[i], "-h") )
			usage(av[0]);
	}

	// call first exercise
	exercise1(size);

	// call second exercise
	exercise2(size);

	// call last exercise
	exercise3(size);

	return EXIT_SUCCESS;
}
