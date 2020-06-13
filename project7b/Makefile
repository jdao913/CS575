# Default build target
all: sum sum_mp sum_simd sum_cl

# Normal targets
clean:
	rm -f ./sum ./sum_mp ./sum_simd

sum: 
	g++ -o sum ./sum.c -lm -fopenmp

sum_mp: 
	g++ -o sum_mp ./sum_openMP.c -lm -fopenmp

sum_simd: 
	g++ -o sum_simd ./sum_SIMD.c -lm -fopenmp

sum_cl: 
	g++ -o sum_cl ./sum_cl.c /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 -lm -fopenmp


# Virtual targets
.PHONY: all sum sum_mp sum_cl clean
