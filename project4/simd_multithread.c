#include <xmmintrin.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <sys/sysinfo.h>

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

int main (int argc, char* argv[]) {
    #ifndef _OPENMP
        fprintf(stderr, "No OpenMP support!\n");
        return 1;
    #endif
    if (argc != 2) {
        fprintf(stderr, "Error: Requires single argument of how many threads to use.\n");
        return 0;
    }
    int NUMT = atoi(argv[1]);
    if (NUMT > get_nprocs_conf()) {
        fprintf(stderr, "Error: Specifying more threads than there are processors. Input less threads (argument 1)\n");
        return 0;
    }
    omp_set_num_threads( NUMT );    // set the number of threads to use in the for-loop
    printf("Using %i threads\n", NUMT);

    int arr_sizes[22] = {1024, 2*1024, 4*1024, 8*1024, 16*1024, 24*1024, 35*1024, 48*1024, 64*1024, 100*1024, 200*1024, 300*1024, 400*1024,
                        500*1024, 600*1024, 700*1024, 800*1024, 1024*1024, 
                    2*1024*1024, 4*1024*1024, 6*1024*1024, 8*1024*1024};
    int num_trials = 50;

    char save_file[50];
    snprintf(save_file, 50, "./simd_multi%i_data.csv", NUMT);
    FILE *fp = fopen(save_file, "w+");
    fprintf(fp, "Array Size, SSE Max Performance, SSE Avg Performance, Loop Max Performance, Loop Avg Performance\n");
    for (int s = 0; s < 22; s++) {
        // Initialize arrays
        float *array_a;
        float *array_b;
        int curr_size = arr_sizes[s];
        array_a = (float *) malloc(sizeof(float)*curr_size);
        array_b = (float *) malloc(sizeof(float)*curr_size);
        for (int i = 0; i < curr_size; i++) {
            array_a[i] = sqrtf((float)i);
            array_b[i] = sqrtf((float)i);
        }
        float *thread_sums = (float *) malloc(sizeof(float)*NUMT);
        // Run trials
        printf("Testing array size of %i\n", curr_size);
        double SSE_avgPerformance = 0.;
        double loop_avgPerformance = 0.;
        double SSE_maxPerformance = 0.;
        double loop_maxPerformance = 0.;
        for (int i = 0; i < num_trials; i++) {
            // Test SSE code
            double time0 = omp_get_wtime( );
            #pragma omp parallel 
            {
                int threadnum = omp_get_thread_num();
                int low = curr_size/NUMT * threadnum;
                int high = curr_size/NUMT * (threadnum + 1);
                int curr_len = curr_size/NUMT;
                if (threadnum == NUMT -1 && high < curr_size) {
                    curr_len = curr_size - low;
                }
                thread_sums[threadnum] = SimdMulSum(array_a+low, array_b+low, curr_len);
            }
            float SSE_sum = 0;
            for (int j = 0; j < NUMT; j ++) {
                SSE_sum += thread_sums[j];
            }
            double time1 = omp_get_wtime( );
            // Test regular for loop
            float loop_sum = 0;
            #pragma omp parallel for default(none), shared(array_a, array_b, curr_size), reduction(+:loop_sum)
            for (int j = 0; j < curr_size; j++) {
                loop_sum += array_a[j] * array_b[j];
            }
            double time2 = omp_get_wtime( );
            // printf("SSE sum: %f\tloop sum: %f\n", SSE_sum, loop_sum);
            // Calculate performances
            double SSE_megaMulsPerSecond = (double)(curr_size) / ( time1 - time0 ) / 1000000.;
            double loop_megaMulsPerSecond = (double)(curr_size) / ( time2 - time1 ) / 1000000.;
            SSE_avgPerformance += (SSE_megaMulsPerSecond - SSE_avgPerformance) / (i+1);
            loop_avgPerformance += (loop_megaMulsPerSecond - loop_avgPerformance) / (i+1);
            if(SSE_megaMulsPerSecond > SSE_maxPerformance) {
                SSE_maxPerformance = SSE_megaMulsPerSecond;
            }
            if(loop_megaMulsPerSecond > loop_maxPerformance) {
                loop_maxPerformance = loop_megaMulsPerSecond;
            }
        }
        printf("SSE avg perf: %f\t SSE max perf: %f\n", SSE_avgPerformance, SSE_maxPerformance);
        printf("loop avg perf: %f\t loop max perf: %f\n", loop_avgPerformance, loop_maxPerformance);
        printf("\n");
        fprintf(fp, "%i, %f, %f, %f, %f\n", curr_size, SSE_maxPerformance, SSE_avgPerformance, loop_maxPerformance, loop_avgPerformance);
    }
}