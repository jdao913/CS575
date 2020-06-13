# Default build target
all: integrate

# Normal targets
clean:
	rm ./integrate

integrate: 
	g++ -o integrate integrate.c -lm -fopenmp

renew:
	make clean
	make integrate

# Virtual targets
.PHONY: all integrate clean
