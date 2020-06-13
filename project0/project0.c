
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

// #define NUMT 4
#define SIZE 16384
// #define NUMTRIES 10000

float A[SIZE];
float B[SIZE];
float C[SIZE];

int main(int argc, char** argv) {

    #ifndef _OPENMP
        fprintf( stderr, "OpenMP is not supported here -- sorry.\n" );
        return 1;
    #endif

    int NUMT = atoi(argv[1]);
    int NUMTRIES = atoi(argv[2]);

	// inialize the arrays:
	for( int i = 0; i < SIZE; i++ )
	{
		A[ i ] = 1.;
		B[ i ] = 2.;
	}

        omp_set_num_threads( NUMT );
        fprintf( stderr, "Using %d threads\n", NUMT );
        fprintf( stderr, "Running %d trials\n", NUMTRIES );

        double maxMegaMults = 0.;
        double avgMegaMults = 0.;

        for( int t = 0; t < NUMTRIES; t++ )
        {
            double time0 = omp_get_wtime( );

            #pragma omp parallel for
            for( int i = 0; i < SIZE; i++ )
            {
                    C[i] = A[i] * B[i];
            }

            double time1 = omp_get_wtime( );
            double megaMults = (double)SIZE/(time1-time0)/1000000.;
            avgMegaMults = avgMegaMults + megaMults;
            if( megaMults > maxMegaMults ) {
                maxMegaMults = megaMults;
            }
        }
        avgMegaMults = avgMegaMults / NUMTRIES;

        printf( "Peak Performance = %8.2lf MegaMults/Sec\n", maxMegaMults );
        printf( "Average Performance = %8.2lf MegaMults/Sec\n", avgMegaMults );

	// note: %lf stands for "long float", which is how printf prints a "double"
	//        %d stands for "decimal integer", not "double"

        return 0;
}