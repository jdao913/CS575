#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


int     Size;
float * A;
float * Sums;

int main(int argc, const char** argv) {
    #ifndef _OPENMP
        fprintf(stderr, "No OpenMP support!\n");
        return 1;
    #endif

    if (argc != 2) {
        fprintf(stderr, "Error: Requires argument of how many threads to use.\n");
        return 0;
    }
    int NUMT = atoi(argv[1]);

    omp_set_num_threads( NUMT );	// set the number of threads to use in the for-loop
	FILE *fp = fopen( "./signal.txt", "r" );
    FILE *out_f = fopen("./auto_corr_mp.csv", "w+");
	if ( fp == NULL ) {
		fprintf( stderr, "Cannot open file 'signal.txt'\n" );
		exit( 1 );
	}
	fscanf( fp, "%d", &Size );
	A =     (float *)malloc( 2 * Size * sizeof(float) );
	Sums  = (float *)malloc( 1 * Size * sizeof(float) );
	for(int i = 0; i < Size; i++ ) {
		fscanf( fp, "%f", &A[i] );
		A[i+Size] = A[i];		// duplicate the array
	}
	fclose( fp );

    double time0 = omp_get_wtime( );
    #pragma omp parallel for default(none), shared(Size, A, Sums)
	for( int shift = 0; shift < Size; shift++ ) {
		float sum = 0.;
		for( int i = 0; i < Size; i++ )
		{
			sum += A[i] * A[i + shift];
		}
		Sums[shift] = sum;	// note the "fix #2" from false sharing if you are using OpenMP
	}
    double time1 = omp_get_wtime( );
    double perf = (double)(Size) * (double)(Size)/ ( time1 - time0 ) / 1000000.;
    printf("Threads: %i\tPerformance: %f MegaMultsPerSecond\n", NUMT, perf);

    for (int i = 0; i < Size; i++) {
		fprintf(out_f, "%f\n", Sums[i]);
	}

    return 1;
}