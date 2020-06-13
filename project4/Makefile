# Default build target
all: simd_test simd_multi

# Normal targets
clean:
	rm ./simd_test ./simd_multi

simd_test: 
	g++ -o simd_test ./simd_test.c -lm -fopenmp

simd_multi:
	g++ -o simd_multi ./simd_multithread.c -lm -fopenmp

renew:
	make clean
	make simd_test

# Virtual targets
.PHONY: all simd_test simd_multi clean renew
