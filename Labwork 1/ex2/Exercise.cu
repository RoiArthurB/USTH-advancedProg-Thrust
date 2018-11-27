#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include "Exercise.hpp"
#include "include/chronoGPU.hpp"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>

class EvenOddFunctor1 : public thrust::unary_function<const long long, long long> {
	const long long m_half_size;
public:
	__host__ __device__ EvenOddFunctor1() = delete;
	__host__ __device__ EvenOddFunctor1(const long long size) : m_half_size(size/2) {}
	EvenOddFunctor1(const EvenOddFunctor1&) = default;
	__host__ __device__ long long operator()(const long long &idx) {
		return ( idx < m_half_size ) ? idx * 2 : 1 + (idx - m_half_size) * 2;
	}	
};
void Exercise::Question1(const thrust::host_vector<int>& A, thrust::host_vector<int>& OE ) const {
  	// TODO: extract values at even and odd indices from A and put them into OE.
	// TODO: using GATHER
	ChronoGPU chrUP, chrDOWN, chrGPU;

	chrUP.start();
	thrust::device_vector<int> d_A(A);
	thrust::device_vector<int> d_OE(A.size());
	chrUP.stop();

	const long long size = static_cast<const long long>(A.size());
	for (int i=3; i--;) 
	{
		chrUP.start();
		thrust::device_vector<int> d_A(A);
		thrust::device_vector<int> d_OE(size);
		chrUP.stop();

		chrGPU.start();
		EvenOddFunctor1 oef(size);
		
		auto begin = thrust::make_transform_iterator(
			thrust::make_counting_iterator(static_cast<long long>(0)), oef);
		auto end = thrust::make_transform_iterator(
			thrust::make_counting_iterator(size), oef);

		thrust::gather(	begin, end, d_A.begin(), d_OE.begin());
		chrGPU.stop();

		chrDOWN.start();
		OE = d_OE;
		chrDOWN.stop();
	}

	float elapsed = chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime();
	std::cout <<"Question 1 done in " << elapsed << "ms" << std::endl;
	std::cout <<" - UP time 	:" << chrUP.elapsedTime() << "ms" << std::endl;
	std::cout <<" - GPU time 	:" << chrGPU.elapsedTime() << "ms" << std::endl;
	std::cout <<" - DOWN time 	:" << chrDOWN.elapsedTime() << "ms" << std::endl;
}



void Exercise::Question2(const thrust::host_vector<int>&A, 
						thrust::host_vector<int>&OE) const 
{
  // TODO: idem q1 using SCATTER
}




template <typename T>
void Exercise::Question3(const thrust::host_vector<T>& A,
						thrust::host_vector<T>&OE) const 
{
  // TODO: idem for big objects
}


struct MyDataType {
	MyDataType(int i) : m_i(i) {}
	MyDataType() = default;
	~MyDataType() = default;
	int m_i;
	operator int() const { return m_i; }

	// TODO: add what you want ...
};

// Warning: do not modify the following function ...
void Exercise::checkQuestion3() const {
	const size_t size = sizeof(MyDataType)*m_size;
	std::cout<<"Check exercice 3 with arrays of size "<<(size>>20)<<" Mb"<<std::endl;
	checkQuestion3withDataType(MyDataType(0));
}
