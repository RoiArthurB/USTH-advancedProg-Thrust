
#include <thrust/fill.h>

#include "D_Matrix.cuh"
#include "H_Matrix.cuh"

// factory of empty matrix
D_Matrix D_Matrix::createEmpty(const int n) {
	D_Matrix result(n);
	thrust::fill_n(result.d_val, n*n, 0);
	return result;
}


// copy operator
D_Matrix& D_Matrix::operator=(const D_Matrix& that) {
	if (this != &that) {
		thrust::device_free(d_val);
		m_n = that.m_n;
		d_val = thrust::device_malloc<int>(sizeof(int)*m_n*m_n);
		thrust::copy_n(that.d_val, m_n*m_n, d_val);
	}
	return *this;
}

// returns the matrix data as a host_vector
void D_Matrix::data(thrust::host_vector<int>& h_val) const {
	thrust::copy_n(d_val, m_n*m_n, h_val.begin());
}

