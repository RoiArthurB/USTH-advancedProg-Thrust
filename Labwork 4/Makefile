CC=/usr/local/cuda/bin/nvcc
CXX=/usr/local/cuda/bin/nvcc
CXXFLAGS=-std=c++11 -g -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50
CPPFLAGS=-Iinclude

run: main.o app.o utils/chronoCPU.o utils/chronoGPU.o utils/ppm.o student.o
	$(CXX) $(CXXFLAGS) $+ -lGL -lGLU -lX11 -lglut -o $@

%.o: %.cu
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

.PHONY: clean

clean:
	find . -iname '*.o' -delete
	rm -f run
