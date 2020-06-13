#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <sys/sysinfo.h>

unsigned int seed = 0;
int	NowYear;		// 2020 - 2025
int	NowMonth;		// 0 - 11

float   NowPrecip;		// inches of rain per month
float   NowTemp;		// temperature this month
float   NowHeight;		// grain height in inches
int     NowNumDeer;		// number of deer in the current population
int     NowNumMice;     // number of mice in the current population

const float GRAIN_GROWS_PER_MONTH =		12.0;
const float ONE_DEER_EATS_PER_MONTH =		0.8;
const float ONE_MOUSE_EATS_PER_MONTH =      0.05;

const float AVG_PRECIP_PER_MONTH =		7.0;	// average
const float AMP_PRECIP_PER_MONTH =		6.0;	// plus or minus
const float RANDOM_PRECIP =			2.0;	// plus or minus noise

const float AVG_TEMP =				60.0;	// average
const float AMP_TEMP =				20.0;	// plus or minus
const float RANDOM_TEMP =			10.0;	// plus or minus noise

const float MIDTEMP =				40.0;
const float MIDPRECIP =				10.0;

omp_lock_t	Lock;
int		NumInThreadTeam;
int		NumAtBarrier;
int		NumGone;
void	InitBarrier( int );
void	WaitBarrier( );

FILE *fp;

// specify how many threads will be in the barrier:
//	(also init's the Lock)
void InitBarrier( int n ) {
    NumInThreadTeam = n;
    NumAtBarrier = 0;
    omp_init_lock( &Lock );
}

// have the calling thread wait here until all the other threads catch up:
void WaitBarrier() {
    omp_set_lock( &Lock );
    {
        NumAtBarrier++;
        if( NumAtBarrier == NumInThreadTeam ) {
            NumGone = 0;
            NumAtBarrier = 0;
            // let all other threads get back to what they were doing
            // before this one unlocks, knowing that they might immediately
            // call WaitBarrier( ) again:
            while( NumGone != NumInThreadTeam-1 );
            omp_unset_lock( &Lock );
            return;
        }
    }
    omp_unset_lock( &Lock );
    while( NumAtBarrier != 0 );	// this waits for the nth thread to arrive

    #pragma omp atomic
    NumGone++;			// this flags how many threads have returned
}

float SQR( float x ) {
    return x*x;
}

float RandF( unsigned int *seedp,  float low, float high ) {
    float r = (float) rand_r( seedp );              // 0 - RAND_MAX
    return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}


int RandInt( unsigned int *seedp, int ilow, int ihigh ) {
    float low = (float)ilow;
    float high = (float)ihigh + 0.9999f;
    return (int)(  RandF(seedp, low,high) );
}

void printState() {
    char line[] = "--------------------------------------------------\n";
    char timeStr[50];
    char dataStr[150];
    snprintf(timeStr, 50, "\tMonth: %i\t Year: %i\n", NowMonth, NowYear);
    snprintf(dataStr, 150, "Temp:\t\t\t%f\nPrecip:\t\t\t%f\nGrain Height:\t\t%f\nNumber of Deer:\t\t%i\nNumber of Mice:\t\t%i\n", NowTemp, NowPrecip, NowHeight, NowNumDeer, NowNumMice);
    printf("%s", line);
    printf("%s", timeStr);
    printf("%s", line);
    printf("%s", dataStr);
    printf("%s", line);
}

void GrainDeer() {
    while (NowYear < 2026) {
        int nextNumDeer = NowNumDeer;
        if (NowNumDeer > NowHeight) {
            nextNumDeer -= 1;
        } else if (NowNumDeer < NowHeight) {
            nextNumDeer += 1;
        }
        // DoneComputing barrier:
        WaitBarrier();

        NowNumDeer = nextNumDeer;
        // DoneAssigning barrier:
        WaitBarrier();

        // DonePrinting barrier:
        WaitBarrier();

    }
}

void Grain() {
    while (NowYear < 2026) {
        float tempFactor = exp(   -SQR(  ( NowTemp - MIDTEMP ) / 10.  )   );
        float precipFactor = exp(   -SQR(  ( NowPrecip - MIDPRECIP ) / 10.  )   );
        float nextHeight = NowHeight;
        nextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
        nextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
        nextHeight -= (float)NowNumMice * ONE_MOUSE_EATS_PER_MONTH;
        if (nextHeight < 0) {
            nextHeight = 0.0;
        }
        // DoneComputing barrier:
        WaitBarrier();

        NowHeight = nextHeight;
        // DoneAssigning barrier:
        WaitBarrier();

        // DonePrinting barrier:
        WaitBarrier();

    }
}

void Mice() {
    while (NowYear < 2026) {
        int nextNumMice = NowNumMice;
        if (NowNumMice > 10.0*NowHeight) {
            nextNumMice -= 2;
        } else if (NowNumMice < 10.0*NowHeight) {
            nextNumMice += 2;
        }
        // DoneComputing barrier:
        WaitBarrier();

        NowNumMice = nextNumMice;
        // DoneAssigning barrier:
        WaitBarrier();

        // DonePrinting barrier:
        WaitBarrier();
    }
}

void Watcher() {
    while (NowYear < 2026) {
        // DoneComputing barrier:
        WaitBarrier();

        // DoneAssigning barrier:
        WaitBarrier();

        printState();
        // Save data to file
        fprintf(fp, "%i, %i, %f, %f, %f, %i, %i\n", NowMonth, NowYear, NowTemp, NowPrecip, NowHeight, NowNumDeer, NowNumMice);
        // Increment time
        NowMonth += 1;
        if (NowMonth > 11) {
            NowMonth = 0;
            NowYear += 1;
        }
        // Update env params
        float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );
        float temp = AVG_TEMP - AMP_TEMP * cos( ang );
        NowTemp = temp + RandF( &seed, -RANDOM_TEMP, RANDOM_TEMP );

        float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
        NowPrecip = precip + RandF( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
        if( NowPrecip < 0. )
            NowPrecip = 0.;
        // DonePrinting barrier:
        WaitBarrier();
    }
}

int main (int argc, char* argv[]) {
    fp = fopen("./sim_data.csv", "w+");
    fprintf(fp, "Month, Year, Temp, Precip, Grain Height, Number of Deer, Number of Mice\n");

    // starting date and time:
    NowMonth =    0;
    NowYear  = 2020;

    // starting state (feel free to change this if you want):
    NowNumDeer = 2;
    NowHeight =  15.;
    NowNumMice = 6;

    float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );
    float temp = AVG_TEMP - AMP_TEMP * cos( ang );
    NowTemp = temp + RandF( &seed, -RANDOM_TEMP, RANDOM_TEMP );

    float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
    NowPrecip = precip + RandF( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
    if( NowPrecip < 0. )
        NowPrecip = 0.;

    omp_set_num_threads(4);     // same as # of sections
    InitBarrier(4);
    #pragma omp parallel sections 
    {
        #pragma omp section 
        {
            GrainDeer( );
        }

        #pragma omp section 
        {
            Grain();
        }

        #pragma omp section 
        {
            Watcher( );
        }

        #pragma omp section
        {
            Mice( );
        }
    }
    fclose(fp);
    return 0;
}