#include <iostream>
#include "student.hpp"

// do not forget to add the needed included files
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>



// ==========================================================================================
// Exercise 1
// Feel free to add any function you need, it is your file ;-)
class IsBlue : public thrust::unary_function<ColoredObject,int> {
        public:
                __device__ int operator()(ColoredObject &color) {
                        return int( color.color == ColoredObject::BLUE );

                }
};
// mandatory function: should returns the blue objects contained in the input parameter
thrust::host_vector<ColoredObject> compactBlue( const thrust::host_vector<ColoredObject>& input ) {

	thrust::device_vector<ColoredObject> d_input(input);
	thrust::device_vector<ColoredObject> d_answer(input.size());

	thrust::device_vector<int> d_predicat(input.size());
	thrust::device_vector<int> d_results(input.size());

	thrust::transform(d_input.begin(),d_input.end(),d_predicat.begin(),IsBlue());

	// Exclusive_SCAN
	//============================
	
	// Sum the result of transform with the isBlue Functor
	thrust::exclusive_scan(
		d_predicat.begin(),
		d_predicat.end(),
		d_results.begin()
	);

	// SCATTER
	//============================
	
	// Sum the result of transform with the isBlue Functor
	thrust::scatter_if(
		d_input.begin(),
		d_input.end(),
		d_results.begin(),
		d_predicat.begin(),
		d_answer.begin()
	);

	// COPY_N  : Copying the reusults on the host
	//============================
	int last_value = d_results[ d_results.size() - 1 ];
	thrust::host_vector<ColoredObject> answer( last_value );                // allocating host vector
	thrust::copy_n( d_answer.begin(), last_value, answer.begin() );         // copying device to host

	return answer;
}

// ==========================================================================================
// Exercise 2
// Feel free to add any function you need, it is your file ;-)
thrust::host_vector<int> radixSort( const thrust::host_vector<int>& h_input ) {
	thrust::host_vector<int> answer;
	return answer;
}

// ==========================================================================================
// Exercise 3
// Feel free to add any function you need, it is your file ;-)
thrust::host_vector<int> quickSort( const thrust::host_vector<int>& h_input ) {
	thrust::host_vector<int> answer;
	return answer;
}
