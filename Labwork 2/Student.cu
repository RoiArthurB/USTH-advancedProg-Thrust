#include "D_Matrix.cuh"
#include "H_Matrix.cuh"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>

//////////////////////////////////////////////////////////////////////////////////
// Exercice 1
bool D_Matrix::Exo1IsDone() {
	return true;
}

// returns this times that ...
D_Matrix D_Matrix::operator+(const D_Matrix& that) const
{
	// do "d_val + that.d_val" 
	/*	GOOD TO KNOW :
		m_n		// size of row / col
		d_val	// pointer to data / iterator
			-> d_val + m_n * m_n	// end pointer
	*/
	D_Matrix result(m_n);
	
	const int size = m_n * m_n;
	thrust::transform(d_val, d_val + size, that.d_val, result.d_val, thrust::plus<int>());

	return result;
}

//////////////////////////////////////////////////////////////////////////////////
// Exercice 2
bool D_Matrix::Exo2IsDone() {
	return true;
}
// define the Matrix::transpose function
D_Matrix D_Matrix::transpose() const
{
	D_Matrix result(m_n);

	// col = idx % m_n
	// row = idx / m_n

	thrust::scatter(
		d_val, d_val + (m_n * m_n),
		thrust::make_transform_iterator(
			thrust::make_counting_iterator(0),
			//swap row and col 
			(thrust::placeholders::_1 % m_n) * m_n + (thrust::placeholders::_1 / m_n) 
		),
		result.d_val
	);

	return result;
}

//////////////////////////////////////////////////////////////////////////////////
// Exercice 3
bool D_Matrix::Exo3IsDone() {
	return true;
}
struct DiffusionFunctor : public thrust::unary_function<int, int> {
	const thrust::device_ptr<int> d_val;
	const int m_n;
	DiffusionFunctor(const thrust::device_ptr<int> val, const int n) : d_val(val), m_n(n) {}
	__device__ int operator()(const int i){
		return d_val[i % m_n];
	}
};
void D_Matrix::diffusion(const int line, D_Matrix& result) const 
{
	//thrust::copy_n(d_val + m_n * line , m_n, result.d_val);
	thrust::copy_n(
		thrust::make_transform_iterator(
			thrust::make_counting_iterator(0),
			DiffusionFunctor(d_val + m_n * line, m_n)),
		m_n*m_n,
		result.d_val
	);
}

//////////////////////////////////////////////////////////////////////////////////
// Exercice 4
bool D_Matrix::Exo4IsDone() {
	return true;
}
// returns this times that ...
D_Matrix D_Matrix::product1(const D_Matrix& that) const
{	
	const int size = m_n * m_n;

	// 3. Transpose
	D_Matrix d_TB = that.transpose();

	// 4. Allocate C matrix
	D_Matrix d_C(m_n);

	// 5. For each row in C
	for(int line = 0; line < m_n; line++){
			// 6.1 Diffusion
			D_Matrix d_TBi(m_n);	
			d_TB.diffusion(line, d_TBi);
		
			// 6.2 Map
			D_Matrix d_D(m_n);
			thrust::transform(d_val, d_val + size, d_TBi.d_val, d_D.d_val, thrust::multiplies<int>());

			// 8.
			auto fkeys = (thrust::placeholders::_1 / m_n) * m_n + line;
			thrust::device_vector<int> column(m_n);
			thrust::device_vector<int> keys(m_n);

			thrust::reduce_by_key(
				thrust::make_transform_iterator(thrust::make_counting_iterator(0), fkeys),
				thrust::make_transform_iterator(thrust::make_counting_iterator(size), fkeys),
				d_D.d_val,
				keys.begin(),
				column.begin()
			);

			// 9.
			thrust::scatter(column.begin(), column.end(), keys.begin(), d_C.d_val);
	}

	// 11. / 12.
	return d_C;
}


//////////////////////////////////////////////////////////////////////////////////
// Exercice 5
bool D_Matrix::Exo5IsDone() {
	return false;
}
// returns this times that ...
D_Matrix D_Matrix::product2(const D_Matrix& that) const {
	return D_Matrix(m_n);
}
