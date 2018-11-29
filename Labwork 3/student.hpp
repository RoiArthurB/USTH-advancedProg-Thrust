/*
 * student.hpp
 *
 *  Created on: 17 janv. 2016
 *      Author: aveneau
 */

#pragma once

#include <thrust/host_vector.h>

// ==========================================================================================
// exercise 1
struct ColoredObject {
	// definition of a color
	enum Color { BLUE, RED };
	// attribute color
	Color color;
	// some data ...
	int pad[3];
	// constructors
	__device__ __host__
	ColoredObject(ColoredObject::Color color) : color(color) {}
	__device__ __host__
	ColoredObject() {}
};

// Given a vector of ColoredObject, this function returns a new vector that contains only blue objects
thrust::host_vector<ColoredObject> compactBlue(const thrust::host_vector<ColoredObject>& input);

// ==========================================================================================
// exercise 2
thrust::host_vector<int> radixSort( const thrust::host_vector<int>& input );

// ==========================================================================================
// exercise 3
thrust::host_vector<int> quickSort( const thrust::host_vector<int>& input );

