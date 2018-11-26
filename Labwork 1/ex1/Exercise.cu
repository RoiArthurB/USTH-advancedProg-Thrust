#include "Exercise.hpp"
#include "./include/chronoGPU.hpp"
#include <thrust/device_vector.h>

void Exercise::Question1(const thrust::host_vector<int>& A, const thrust::host_vector<int>& B, thrust::host_vector<int>&C) const {
	// TODO: addition of two vectors using thrust

	ChronoGPU chrUP, chrDOWN, chrGPU;

	for(int i = 3; --i; ){
		chrUP.start();

		thrust::device_vector<int> d_A(A);
		thrust::device_vector<int> d_B(B);
		thrust::device_vector<int> d_C(A.size());
	
		chrUP.stop();

		chrGPU.start();

		thrust::transform( d_A.begin(), d_A.end(), d_B.begin(), d_C.begin(), 
			thrust::placeholders::_1 + thrust::placeholders::_2 );

		chrGPU.stop();

		chrDOWN.start();

		C = d_C;

		chrDOWN.stop();


	}

	float elapsed = chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime();

	std::cout << "Question 1 done in " << elapsed << std::endl;
	std::cout << "	- UP time " << chrUP.elapsedTime() << std::endl;
	std::cout << "	- GPU time " << chrGPU.elapsedTime() << std::endl;
	std::cout << "	- DOWN time " << chrDOWN.elapsedTime() << std::endl;

//	thrust::transform( A.begin(), A.end(), B.begin(), C.begin(), thrust::plus<float>() );
}


void Exercise::Question2(thrust::host_vector<int>&A) const 
{
  // TODO: addition using ad hoc iterators
}



void Exercise::Question3(const thrust::host_vector<int>& A,
												const thrust::host_vector<int>& B, 
												const thrust::host_vector<int>& C, 
												thrust::host_vector<int>&D) const 
{
  // TODO
}
