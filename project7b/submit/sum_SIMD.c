#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/sysinfo.h>
#include <xmmintrin.h>
#include <math.h>

int     Size;
float * A;
float * Sums;

#define SSE_WIDTH       4

float SimdMulSum( float *a, float *b, int len ) {
    float sum[4] = { 0., 0., 0., 0. };
    int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
    register float *pa = a;
    register float *pb = b;

    __m128 ss = _mm_loadu_ps( &sum[0] );
    for( int i = 0; i < limit; i += SSE_WIDTH )
    {
        ss = _mm_add_ps( ss, _mm_mul_ps( _mm_loadu_ps( pa ), _mm_loadu_ps( pb ) ) );
        pa += SSE_WIDTH;
        pb += SSE_WIDTH;
    }
    _mm_storeu_ps( &sum[0], ss );

    for( int i = limit; i < len; i++ )
    {
        sum[0] += a[i] * b[i];
    }

    return sum[0] + sum[1] + sum[2] + sum[3];
}

int main(int argc, const char** argv) {
    #ifndef _OPENMP
        fprintf(stderr, "No OpenMP support!\n");
        return 1;
    #endif

	FILE *fp = fopen( "./signal.txt", "r" );
    FILE *out_f = fopen("./auto_corr_SIMD.csv", "w+");
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
	for( int shift = 0; shift < Size; shift++ ) {
		Sums[shift] = SimdMulSum(&A[0], &A[0+shift], Size);	// note the "fix #2" from false sharing if you are using OpenMP
	}
    double time1 = omp_get_wtime( );
    double perf = (double)(Size) * (double)(Size)/ ( time1 - time0 ) / 1000000.;
    printf("Performance: %f MegaMultsPerSecond\n", perf);

    for (int i = 0; i < Size; i++) {
		fprintf(out_f, "%f\n", Sums[i]);
	}

    return 1;
}