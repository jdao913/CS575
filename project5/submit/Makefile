# Default build target
all: mc_sim

# Normal targets
clean:
	rm ./mc_sim

mc_sim: 
	nvcc -o mc_sim montecarlo.cu

# Virtual targets
.PHONY: all mc_sim clean
