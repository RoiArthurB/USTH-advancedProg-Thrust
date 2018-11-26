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

  void checkQuestion1() const 
  {
  	const size_t size = sizeof(int)*m_size;
	  std::cout<<"Check exercice 1 with arrays of size "<<(size>>20)<<" Mb"<<std::endl;
    thrust::host_vector<int> A(m_size);
    thrust::host_vector<int> OE(m_size);

    for(int i=static_cast<const int>(m_size); i--; ) {
      A[i] = i;
    }

    Question1(A, OE);
    
    const int half_size = static_cast<const int>(m_size/2);
    for(int i=static_cast<const int>(m_size); i--; ) 
    {
      const bool test = (OE[i] & 0x1) == (i<half_size ? /* even case */ 0 : /*  odd case */ 1);
      if( ! test ) {
        std::cerr<<"Error in "<<__FUNCTION__
          <<": bad result at position "<<i
          <<" (receiving "<<OE[i]<<")"
          <<std::endl;
        break;
      }
    }
  }

  void checkQuestion2() const {
  	const size_t size = sizeof(int)*m_size;
	  std::cout<<"Check exercice 2 with arrays of size "<<(size>>20)<<" Mb"<<std::endl;
    thrust::host_vector<int> A(m_size);
    thrust::host_vector<int> OE(m_size);

    for(int i=static_cast<const int>(m_size); i--; ) {
      A[i] = i;
    }

    Question2(A, OE);
    
    const int half_size = static_cast<const int>(m_size/2);
    for(int i=static_cast<const int>(m_size); i--; ) {
      const bool test = (OE[i] & 0x1) == (i<half_size ? /* even case */ 0 : /*  odd case */ 1);
      if( ! test ) {
        std::cerr<<"Error in "<<__FUNCTION__
          <<": bad result at position "<<i
          <<" (receiving "<<OE[i]<<")"
          <<std::endl;
        break;
      }
    }
  }

  void checkQuestion3() const;

  template<typename T>
  void checkQuestion3withDataType(const T&) const {
    thrust::host_vector<T> A(m_size);
    thrust::host_vector<T> OE(m_size);

    for(int i=static_cast<const int>(m_size); i--; ) {
      A[i] = T(i); // you should provide a cast from int to T
    }

    Question3(A, OE);
    
    const int half_size = static_cast<const int>(m_size/2);
    for(int i=static_cast<const int>(m_size); i--; ) {
      const bool test = (int(OE[i]) & 0x1) == (i<half_size ? /* even case */ 0 : /*  odd case */ 1);
      if( ! test ) {
	      std::cerr<<"Error in "<<__FUNCTION__
		      <<": bad result at position "<<i
          <<" (receiving "<<int(OE[i])<<")"
		      <<std::endl;
	      break;
      }
    }
  }

  // students have to implement the 3 functions following in "Exercise.cpp":
  void Question1(const thrust::host_vector<int>& A,
                thrust::host_vector<int>&OE) const;
  void Question2(const thrust::host_vector<int>& A,
                thrust::host_vector<int>&OE) const;

  template<typename T>
  void Question3(const thrust::host_vector<T>& A,
                thrust::host_vector<T>&OE) const;
  //
};
