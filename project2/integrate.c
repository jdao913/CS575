#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <sys/sysinfo.h>

#define XMIN     -1.
#define XMAX      1.
#define YMIN     -1.
#define YMAX      1.
#define N         4

int NUMNODES;

double Height( int iu, int iv )	// iu,iv = 0 .. NUMNODES-1
{
    double x = -1.  +  2.*(double)iu /(double)(NUMNODES-1);	// -1. to +1.
    double y = -1.  +  2.*(double)iv /(double)(NUMNODES-1);	// -1. to +1.

    double xn = pow( fabs(x), (double)N );
    double yn = pow( fabs(y), (double)N );
    double r = 1. - xn - yn;
    if( r < 0. )
            return 0.;
    double height = pow( 1. - xn - yn, 1./(double)N );
    return height;
}

int main( int argc, char *argv[ ] )
{
    #ifndef _OPENMP
        fprintf(stderr, "No OpenMP support!\n");
        return 1;
    #endif

    if (argc != 4) {
        fprintf(stderr, "Error: Requires 3 arguments of how many threads to use (1), how many nodes to use (2), and how many trials to run (3).\n");
        return 0;
    }
    int NUMT = atoi(argv[1]);
    NUMNODES = atoi(argv[2]);
    int NUMTRIALS = atoi(argv[3]);
    if (NUMT > get_nprocs_conf()) {
        fprintf(stderr, "Error: Specifying more threads than there are processors. Input less threads (argument 1)\n");
        return 0;
    }
    omp_set_num_threads( NUMT );	// set the number of threads to use in the for-loop
    printf("Using %i threads to compute %i nodes\n", NUMT, NUMNODES);

    // the area of a single full-sized tile:

    double fullTileArea = (  ( ( XMAX - XMIN )/(double)(NUMNODES-1) )  *
                            ( ( YMAX - YMIN )/(double)(NUMNODES-1) )  );
    double dx = (XMAX - XMIN) / (double)(NUMNODES-1);
    double dy = (YMAX - YMIN) / (double)(NUMNODES-1);

    double maxPerformance = 0.;
    double avgPerformance = 0.;
    double volume;
    for (int t = 0; t < NUMTRIALS; t++) {

        // sum up the weighted heights into the variable "volume"
        // using an OpenMP for loop and a reduction:

        volume = 0.0;
        double time0 = omp_get_wtime( );
        #pragma omp parallel for default(none), shared(fullTileArea, NUMNODES, dx, dy), reduction(+:volume)
        for (int i = 0; i < NUMNODES*NUMNODES; i++) {
            double node_x = i % NUMNODES;
            double node_y = i / NUMNODES;
            
            double tileArea = fullTileArea;
            // If x position of node is at edge, in order to get center of tile, need to shift inwards (-) or outwards (+)
            // by quarter of tile width (dx)
            if (node_x == 0) {
                tileArea /= 2;
                node_x += dx / 4;
            } else if (node_x == NUMNODES - 1) {
                tileArea /= 2;
                node_x -= dx / 4;
            }
            // If y position of node is at edge, in order to get center of tile, need to shift downwards (-) or upwards (+)
            // by quarter of tile width (dy)
            if (node_y == 0) {
                tileArea /= 2;
                node_y += dy / 4;
            } else if (node_y == NUMNODES - 1) {
                tileArea /= 2;
                node_y -= dy / 4;
            }
            double z = Height(node_x, node_y);
            double partVolume = tileArea * z * 2.0;
            volume += partVolume;
        }
        double time1 = omp_get_wtime( );
        double megaHeightsPerSecond = (double)(NUMNODES*NUMNODES) / ( time1 - time0 ) / 1000000.;
        avgPerformance += (megaHeightsPerSecond - avgPerformance) / (t+1);
        if(megaHeightsPerSecond > maxPerformance) {
            maxPerformance = megaHeightsPerSecond;
        }
    }
    printf("Computed Volume: %.6f\tMax Performance: %.3f\tAvg Performace: %.3f\n", volume, maxPerformance, avgPerformance);

}