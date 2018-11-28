#include "H_Matrix.cuh"
#include "D_Matrix.cuh"

#include <algorithm>

// returns a new random matrix of size nxn
H_Matrix H_Matrix::random(const int n) {
	H_Matrix result(n);
	const unsigned int dec = std::max(static_cast<unsigned int>(ceilf(log2f(RAND_MAX)))-4u,4u);
	for (int i = 0; i < n*n; ++i)
		result.h_val[i] = std::rand() >> dec;
	return result;
}


// creates a D_Matrix from a H_Matrix
D_Matrix H_Matrix::export2Device() const {
	D_Matrix result(m_n);
	thrust::copy(h_val.begin(), h_val.end(), result.d_val);
	return result;
}


// check equality (with an epsilon)
bool H_Matrix::operator==(const H_Matrix& that) const {
	if (m_n != that.m_n) return false;
	for (int i = m_n*m_n; i-- > 0;)
		if (h_val[i] != that.h_val[i])
			return false;
	return true;
}

// returns this times that ...
H_Matrix H_Matrix::operator*(const H_Matrix& that) const {
	if (m_n != that.m_n)
		throw "bad product (incompatible matrices)";
	H_Matrix result(m_n);

	for (int row = m_n; row--; ) {
		for (int col = m_n; col--; ) {
			int sum = 0;
			for (int k = m_n; k--; ) {
				// sum += A[row,k] * B[k,col]
				sum += h_val[row*m_n + k] * that.h_val[k*m_n + col];
			}
			result.h_val[row*m_n + col] = sum;
		}
	}
	return result;
}

// returns this plus that ...
H_Matrix H_Matrix::operator+(const H_Matrix& that) const {
	if (m_n != that.m_n)
		throw "bad product (incompatible matrices)";
	H_Matrix result(m_n);

	for (int idx = m_n*m_n; idx-- > 0; ) {
		result.h_val[idx] = h_val[idx] + that.h_val[idx];
	}
	return result;
}

// diffuse this matrix for a given line number ...
void H_Matrix::diffusion(const int line, H_Matrix& that) const {
	for(int l=0; l<m_n; ++l) {
		// copies one line from position "line" to position l*m_n
		memcpy(&that.h_val[l*m_n], &h_val[line*m_n], m_n * sizeof(int));
	}
}


// returns transposition of this 
H_Matrix H_Matrix::transpose() const
{
	H_Matrix result(m_n);
	for (int row = 0; row < m_n; row++)
		for (int col = 0; col < m_n; col++)
			result.h_val[col*m_n + row] = h_val[row*m_n + col];
	return result;
}