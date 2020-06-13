# Default build target
all: run

# Normal targets
clean:
	rm ./proj

run: 
	g++ -o proj project0.c -lm -fopenmp

# Virtual targets
.PHONY: all run clean
