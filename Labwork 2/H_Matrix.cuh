#pragma once

#include <iostream>
#include <cstdlib>
#include <ctime>

#include <thrust/host_vector.h>
#include "D_Matrix.cuh"


struct H_Matrix
{
	// data members
	int m_n; // matrix size

	thrust::host_vector<int> h_val; // matrix, by lines

	// returns a new random matrix of size nxn
	static H_Matrix random(const int n);

	// destructor
	~H_Matrix() {}

	// constructor
	explicit
	H_Matrix(const int n) : m_n(n), h_val(n*n)
	{
	}

	// constructor by copy
	H_Matrix(const H_Matrix& that) : m_n(that.m_n), h_val(that.h_val)
	{
	}

	// creates a D_Matrix from a H_Matrix
	D_Matrix export2Device() const;

	// check equality (with an epsilon)
	bool operator==(const H_Matrix& that) const;
	
	// check inequality (with an epsilon)
	bool operator!=(const H_Matrix& that) const {
		return !(*this == that);
	}

	// returns this times that ...
	H_Matrix operator*(const H_Matrix& that) const;

	// returns this plus that ...
	H_Matrix operator+(const H_Matrix& that) const;

	// roll this matrix for a given number of lines ...
	void diffusion(const int line, H_Matrix& that) const;

	// returns transposition of this 
	H_Matrix transpose() const;
};