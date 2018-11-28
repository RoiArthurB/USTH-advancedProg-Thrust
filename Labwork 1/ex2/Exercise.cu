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

		thrust::gather(
			thrust::make_transform_iterator( thrust::make_counting_iterator(static_cast<long long>(0)), oef), 
			thrust::make_transform_iterator(thrust::make_counting_iterator(size), oef), 
			d_A.begin(), d_OE.begin()
		);
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


class EvenOddFunctor2 : public thrust::unary_function<const long long, long long> {
	const long long m_half_size;
public:
	__host__ __device__ EvenOddFunctor2() = delete;
	__host__ __device__ EvenOddFunctor2(const long long size) : m_half_size(size/2) {}
	EvenOddFunctor2(const EvenOddFunctor2&) = default;
	__host__ __device__ long long operator()(const long long &idx) {
		const long long oe = idx&0x1;
		const long long idx2 = idx >> 1;
		return idx2 + oe * m_half_size;
	}
};
void Exercise::Question2(const thrust::host_vector<int>&A, thrust::host_vector<int>&OE) const {
	// TODO: idem q1 using SCATTER
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

		thrust::scatter(
			d_A.begin(), d_A.end(),
			thrust::make_transform_iterator( 
				thrust::make_counting_iterator(static_cast<long long>(0)), 
				EvenOddFunctor2(size)),
			d_OE.begin()
		);
		chrGPU.stop();

		chrDOWN.start();
		OE = d_OE;
		chrDOWN.stop();
	}

	float elapsed = chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime();
	std::cout <<"Question 2 done in " << elapsed << "ms" << std::endl;
	std::cout <<" - UP time 	:" << chrUP.elapsedTime() << "ms" << std::endl;
	std::cout <<" - GPU time 	:" << chrGPU.elapsedTime() << "ms" << std::endl;
	std::cout <<" - DOWN time 	:" << chrDOWN.elapsedTime() << "ms" << std::endl;
}

template <typename T>
void Exercise::Question3(const thrust::host_vector<T>& A, thrust::host_vector<T>&OE) const {
	// TODO: idem for big objects
	ChronoGPU chrUP, chrDOWN, chrGPU;
	for (int i=3; i--; ){
		chrUP.start();
  		thrust::device_vector<T> gpuA(A);
  		thrust::device_vector<T> gpuOE(OE.size());
		thrust::counting_iterator<int>X(0);
		chrUP.stop();

		chrGPU.start();
		thrust::scatter(//thrust::device, 
			gpuA.begin(), gpuA.end(),
			thrust::make_transform_iterator(X, evenOddScatter(gpuA.size())),
			//thrust::make_transform_iterator(X + gpuA.size(), evenOddFunction(gpuA.size())),
			gpuOE.begin()
		);
		chrGPU.stop();

		chrDOWN.start();
		OE = gpuOE;
		chrDOWN.stop();
	}
	float elapsed = chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime();
	std::cout << "Question3 done in " << elapsed << std::endl;
	std::cout << "	- UP time : " << chrUP.elapsedTime() << std::endl;
	std::cout << "	- GPU time : "<< chrGPU.elapsedTime() << std::endl;
	std::cout << "	- DOWN time : " << chrDOWN.elapsedTime() << std::endl;
}


struct MyDataType {
	MyDataType(int i) : m_i(i) {}
	MyDataType() = default;
	~MyDataType() = default;
	int m_i;
	operator int() const { return m_i; }

	// TODO: add what you want ...
	int x[10];
};
