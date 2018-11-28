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
	return false;
}

// returns this times that ...
D_Matrix D_Matrix::operator+(const D_Matrix& that) const
{
  // do "d_val + that.d_val" 
  D_Matrix result(m_n);
  return result;
}



//////////////////////////////////////////////////////////////////////////////////
// Exercice 2
bool D_Matrix::Exo2IsDone() {
	return false;
}
// define the Matrix::transpose function
D_Matrix D_Matrix::transpose() const
{
	D_Matrix result(m_n);
	return result;
}



//////////////////////////////////////////////////////////////////////////////////
// Exercice 3
bool D_Matrix::Exo3IsDone() {
	return false;
}
void D_Matrix::diffusion(const int line, D_Matrix& result) const 
{
}



//////////////////////////////////////////////////////////////////////////////////
// Exercice 4
bool D_Matrix::Exo4IsDone() {
	return false;
}
// returns this times that ...
D_Matrix D_Matrix::product1(const D_Matrix& that) const
{	
	D_Matrix result(m_n);
	return result;
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
