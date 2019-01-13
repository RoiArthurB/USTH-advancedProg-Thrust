#include <iostream>
#include "student.hpp"

// do not forget to add the needed included files
#include "./utils/chronoGPU.hpp"
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

	ChronoGPU chr;

	chr.start();

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

	chr.stop();

	std::cout << "Exercise 1 done in " << chr.elapsedTime() << " ms" << std::endl;

	return answer;
}

// ==========================================================================================
// Exercise 2
// Feel free to add any function you need, it is your file ;-)
struct WhereToGoFunctor : public thrust::unary_function<thrust::tuple<unsigned,unsigned,unsigned>, unsigned>{
	const unsigned size;
	WhereToGoFunctor(const unsigned s) : size(s) {}
	unsigned operator()(const thrust::tuple<unsigned,unsigned,unsigned>& t){
		return thrust::get<0>(t) ? size - thrust::get<2>(t) : thrust::get<1>(t); 
	}
};
void display(const thrust::host_vector<unsigned>& v, const std::string& msg = "array"){
	std::cout << msg << ";";
	for(const unsigned& u : v)
		std::cout << u << " ";
	std::cout << std::endl;
}
thrust::host_vector<int> radixSort( const thrust::host_vector<int>& h_input ) {
	const unsigned size = unsigned(h_input.size());
	//
	//thrust::device_vector<unsigned> d_output(size);
	thrust::device_vector<unsigned> d_io[2] = {h_input, thrust::device_vector<unsigned>(size) }; //d_output};
	thrust::device_vector<unsigned> d_flags(size);
	thrust::device_vector<unsigned> d_bits0(size);
	thrust::device_vector<unsigned> d_bits1(size);

	ChronoGPU chr;
	chr.start();

	for(int i = 0; i<32; i++){
		//Get the bits, scan in one sense then the other, scatter!
		const int ioIn = i & 0x1;
		const int ioOut = i - ioIn;

		thrust::transform(
			d_io[ioIn].begin(),
			d_io[ioIn].end(),
			d_flags.begin(),
			(thrust::placeholders::_1 >> i) & 0x1
		);
		thrust::exclusive_scan(
			thrust::make_transform_iterator(
				d_flags.begin(),
				1 - thrust::placeholders::_1 //needs to inverts the flag
			),
			thrust::make_transform_iterator(
				d_flags.end(),
				1 - thrust::placeholders::_1 //needs to inverts the flag
			),
			d_bits0.begin()
		);
		// scan for bits 1: we will use SIZE-#, so the last one will be at pos size-1, previous at siz... 
		thrust::inclusive_scan(
			d_flags.rbegin(),
			d_flags.rend(),
			d_bits1.rbegin()
		);
		thrust::scatter(
			d_io[ioIn].begin(),
			d_io[ioIn].end(),
			thrust::make_transform_iterator(
				thrust::make_zip_iterator(
					thrust::make_tuple(d_flags.begin(), d_bits0.begin(), d_bits1.begin())
				),
				WhereToGoFunctor(size)
			),
			d_io[ioOut].begin()
		);
	}

	chr.stop();

	std::cout << "Exercise 2 done in " << chr.elapsedTime() << " ms" << std::endl;
	thrust::host_vector<int> answer;
	return answer;//d_io;//answer;
}

// ==========================================================================================
// Exercise 3
// Feel free to add any function you need, it is your file ;-)
thrust::host_vector<int> quickSort( const thrust::host_vector<int>& h_input ) {
	thrust::host_vector<int> answer;
	return answer;
}
