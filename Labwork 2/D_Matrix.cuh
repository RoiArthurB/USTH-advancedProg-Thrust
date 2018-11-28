#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/copy.h>

struct D_Matrix 
{
	// data members
	int m_n; // matrix size

	thrust::device_ptr<int> d_val; // matrix, by lines

	// factory of empty matrix
	static D_Matrix createEmpty(const int n);

	// destructor
	~D_Matrix() {
		thrust::device_free(d_val);
	}

	// constructor
	explicit
	D_Matrix(const int n) : m_n(n), d_val(thrust::device_malloc<int>(sizeof(int)*(n*n)))
	{
	}

	// constructor by copy
	D_Matrix( const D_Matrix& that) : m_n(that.m_n), d_val(thrust::device_malloc<int>(sizeof(int)*that.m_n*that.m_n))
	{
		thrust::copy_n(that.d_val, m_n*m_n, d_val);
	}

	// copy operator
	D_Matrix& operator=(const D_Matrix& that);

	// returns the matrix data as a host_vector
	void data(thrust::host_vector<int>& h_val) const;

	// Exercise 1
	static bool Exo1IsDone();
	// returns this plus that ...
	D_Matrix operator+(const D_Matrix& that) const;

	// Exercise 2
	static bool Exo2IsDone();
	// returns a new matrix, as the transposition of this
	D_Matrix transpose() const;

	// Exercise 3
	static bool Exo3IsDone();
	// creates a Matrix using one line ...
	void diffusion(const int line, D_Matrix& dest) const;

	// Exercise 4
	static bool Exo4IsDone();
	// returns this times that ...
	D_Matrix product1(const D_Matrix& that) const;

	// Exercise 5
	static bool Exo5IsDone();
	// returns this times that ...
	D_Matrix product2(const D_Matrix& that) const;

};