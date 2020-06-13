# Default build target
all: printinfo arr_mult arr_mult_add arr_mult_reduce

# Normal targets
clean:
	rm -rf ./printinfo ./arr_mult ./arr_mult_add ./arr_mult_reduce

printinfo:
	g++  ./printinfo.cpp /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 -lm -fopenmp -o printinfo

arr_mult:
	g++  ./arr_mult.cpp /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 -lm -fopenmp -o arr_mult

arr_mult_add:
	g++  ./arr_mult_add.cpp /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 -lm -fopenmp -o arr_mult_add

arr_mult_reduce:
	g++  ./arr_mult_reduce.cpp /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 -lm -fopenmp -o arr_mult_reduce


# Virtual targets
.PHONY: all printinfo arr_mult arr_mult_add arr_mult_reduce clean 
