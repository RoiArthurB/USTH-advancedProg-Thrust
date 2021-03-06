#pragma once

#include <iostream>
#include <thrust/host_vector.h>

class Exercise 
{
  const unsigned m_size;

public:

  Exercise(const unsigned size=(1<<16))
    : m_size(size) 
  {}

  Exercise(const Exercise& ex) 
    : m_size(ex.m_size)
  {}
  
  void run() {
    checkQuestion1();
    checkQuestion2();
    checkQuestion3();
  }

  void checkQuestion1() const {
    thrust::host_vector<int> A(m_size);
    thrust::host_vector<int> B(m_size);
    thrust::host_vector<int> C(m_size);

    for(int i=static_cast<const int>(m_size); i--; ) {
      A[i] = i;
      B[i] = m_size - i;
      C[i] = -1;
    }

    Question1(A, B, C);
    
    for(int i=static_cast<const int>(m_size); i--; ) {
      if( C[i] != m_size ) {
        std::cerr<<"Error in "<<__FUNCTION__
          <<": bad result at position "<<i
          <<" (receiving "<<C[i]<<")"
          <<std::endl;
        break;
      }
    }
  }

  void checkQuestion2() const {
    thrust::host_vector<int> A(m_size);

    for(int i=static_cast<const int>(m_size); i--; ) {
      A[i] = -1;
    }

    Question2(A);
    
    for(int i=static_cast<const int>(m_size); i--; ) {
      if( A[i] != (1+i+4) ) {
        std::cerr<<"Error in "<<__FUNCTION__
          <<": bad result at position "<<i
          <<" (receiving "<<A[i]<<")"
          <<std::endl;
        break;
      }
    }
  }
  // function that adds two elements and return a new one
  class AdditionFunctor : public thrust::binary_function<float, float, float> {
    public :
      __host__ __device__ float operator()( const float &x, const float &y ) {
        return x + y ;
      }
  };

  void checkQuestion3() const {
    thrust::host_vector<int> A(m_size);
    thrust::host_vector<int> B(m_size);
    thrust::host_vector<int> C(m_size);
    thrust::host_vector<int> D(m_size);

    for(int i=static_cast<const int>(m_size); i--; ) {
      A[i] = i;
      B[i] = 2*(m_size - i);
      C[i] = i;
      D[i] = -1;
    }

    Question3(A, B, C, D);
    
    for(int i=static_cast<const int>(m_size); i--; ) {
      if( D[i] != 2*m_size ) {
	      std::cerr<<"Error in "<<__FUNCTION__
		      <<": bad result at position "<<i
          <<" (receiving "<<D[i]<<")"
		      <<std::endl;
	      break;
      }
    }
  }

  //function that adds three elements and returns a new one
  typedef thrust::tuple<float, float, float>myFloat3;
  class AdditionFunctor3 : public thrust::unary_function<myFloat3, float> {
    public :
      __device__ float operator()( const myFloat3& tuple ) {
        return thrust::get<0>(tuple) + thrust::get<1>(tuple) + thrust::get<2>(tuple) ;
      }
  };

  // students have to implement the following in "Exercise.cpp":
  void Question1(const thrust::host_vector<int>& A,
                const thrust::host_vector<int>& B, 
                thrust::host_vector<int>&C) const;
  void Question2(thrust::host_vector<int>&A) const;
  void Question3(const thrust::host_vector<int>& A,
                const thrust::host_vector<int>& B, 
                const thrust::host_vector<int>& C, 
                thrust::host_vector<int>&D) const;

};
