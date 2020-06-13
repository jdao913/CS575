# Default build target
all: mc_sim

# Normal targets
clean:
	rm ./mc_sim

mc_sim: 
	g++ -o mc_sim mc_sim.c -lm -fopenmp

# Virtual targets
.PHONY: all mc_sim clean
