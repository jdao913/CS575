#include <stdio.h>
#include <stdlib.h>

int     Size;
float * A;
float * Sums;

int main(int argc, const char** argv) {
	FILE *fp = fopen( "./signal.txt", "r" );
	FILE *out_f = fopen("./auto_corr.csv", "w+");
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

	for( int shift = 0; shift < Size; shift++ ) {
		float sum = 0.;
		for( int i = 0; i < Size; i++ )
		{
			sum += A[i] * A[i + shift];
		}
		Sums[shift] = sum;	// note the "fix #2" from false sharing if you are using OpenMP
	}

	for (int i = 0; i < Size; i++) {
		fprintf(out_f, "%f\n", Sums[i]);
	}
	
	return 1;
}