# Default build target
all: grain_sim

# Normal targets
clean:
	rm ./grain_sim

grain_sim: 
	g++ -o grain_sim ./grain_sim.c -lm -fopenmp

renew:
	make clean
	make grain_sim

# Virtual targets
.PHONY: all grain_sim clean renew
