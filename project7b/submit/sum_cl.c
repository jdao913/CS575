#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/sysinfo.h>
#include <xmmintrin.h>
#include <math.h>
#include "CL/cl.h"
#include "CL/cl_platform.h"

int     Size;
const char * CL_FILE_NAME = { "sum.cl" };

void Wait( cl_command_queue );

int main(int argc, const char** argv) {

	if (argc != 2) {
		fprintf(stderr, "Error: Requires exactly one argument of local size.\n");
		return 1;
	}
	int LOCAL_SIZE = atoi(argv[1]);

    #ifndef _OPENMP
        fprintf(stderr, "No OpenMP support!\n");
        return 1;
    #endif

    // Read in signal file allocate the host memory buffers:
	FILE *fp = fopen( "./signal.txt", "r" );
    FILE *out_f = fopen("./auto_corr_SIMD.csv", "w+");
	if ( fp == NULL ) {
		fprintf( stderr, "Cannot open file 'signal.txt'\n" );
		exit( 1 );
	}
	fscanf( fp, "%d", &Size );
	float *hA =     (float *)malloc( 2 * Size * sizeof(float) );
	float *hSums  = (float *)malloc( 1 * Size * sizeof(float) );
	for(int i = 0; i < Size; i++ ) {
		fscanf( fp, "%f", &hA[i] );
		hA[i+Size] = hA[i];		// duplicate the array
	}
	fclose( fp );

    // Open cl source code file
    fp = fopen( CL_FILE_NAME, "r" );
	if( fp == NULL )
	{
		fprintf( stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME );
		return 1;
	}

	cl_int status;		// returned status from opencl calls
				// test against CL_SUCCESS

	// get the platform id:
	cl_platform_id platform;
	status = clGetPlatformIDs( 1, &platform, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clGetPlatformIDs failed (2)\n" );
	
	// get the device id:
	cl_device_id device;
	status = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clGetDeviceIDs failed (2)\n" );

	// 3. create an opencl context:
	cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateContext failed\n" );

	// 4. create an opencl command queue:
	cl_command_queue cmdQueue = clCreateCommandQueue( context, device, 0, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateCommandQueue failed\n" );

	// 5. allocate the device memory buffers:
	cl_mem dA =     clCreateBuffer( context, CL_MEM_READ_ONLY,  2*Size*sizeof(cl_float), NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateBuffer failed (1)\n" );
	cl_mem dSums  = clCreateBuffer( context, CL_MEM_WRITE_ONLY, 1*Size*sizeof(cl_float), NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateBuffer failed (1)\n" );

	// 6. enqueue the 2 commands to write the data from the host buffers to the device buffers:
	status = clEnqueueWriteBuffer( cmdQueue, dA, CL_FALSE, 0, Size, hA, 0, NULL, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clEnqueueWriteBuffer failed (0)\n" );

	// status = clEnqueueWriteBuffer( cmdQueue, dSums, CL_FALSE, 0, Size, hSums, 0, NULL, NULL );
	// if( status != CL_SUCCESS )
	// 	fprintf( stderr, "clEnqueueWriteBuffer failed (1)\n" );

	Wait( cmdQueue );

	// 7. read the kernel code from a file:
	fseek( fp, 0, SEEK_END );
	size_t fileSize = ftell( fp );
	fseek( fp, 0, SEEK_SET );
	// char *clProgramText = new char[ fileSize+1 ];		// leave room for '\0'
	char* clProgramText = (char *)malloc((fileSize+1)*sizeof(char));
	size_t n = fread( clProgramText, 1, fileSize, fp );
	clProgramText[fileSize] = '\0';
	fclose( fp );
	if( n != fileSize )
		fprintf( stderr, "Expected to read %d bytes read from '%s' -- actually read %d.\n", fileSize, CL_FILE_NAME, n );

	// create the text for the kernel program:
	char *strings[1];
	strings[0] = clProgramText;
	cl_program program = clCreateProgramWithSource( context, 1, (const char **)strings, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateProgramWithSource failed\n" );
	// delete [ ] clProgramText;
	free(clProgramText);

	// 8. compile and link the kernel code:

	char *options = { "" };
	status = clBuildProgram( program, 1, &device, options, NULL, NULL );
	if( status != CL_SUCCESS )
	{
		size_t size;
		clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size );
		// cl_char *log = new cl_char[ size ];
		cl_char *log = (cl_char *)malloc(size * sizeof(cl_char));
		clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_LOG, size, log, NULL );
		fprintf( stderr, "clBuildProgram failed:\n%s\n", log );
		// delete [ ] log;
		free(log);
	}

	// 9. create the kernel object:
	cl_kernel kernel = clCreateKernel( program, "AutoCorrelate", &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateKernel failed\n" );

	// 10. setup the arguments to the kernel object:
	status = clSetKernelArg( kernel, 0, sizeof(cl_mem), &dA );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clSetKernelArg failed (0)\n" );
	status = clSetKernelArg( kernel, 1, sizeof(cl_mem), &dSums  );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clSetKernelArg failed (1)\n" );

	// 11. enqueue the kernel object for execution:
	size_t globalWorkSize[3] = { Size, 1, 1 };
	size_t localWorkSize[3]  = { LOCAL_SIZE,   1, 1 };

	Wait( cmdQueue );
	double time0 = omp_get_wtime( );

	status = clEnqueueNDRangeKernel( cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clEnqueueNDRangeKernel failed: %d\n", status );

	Wait( cmdQueue );
	double time1 = omp_get_wtime( );

	// 12. read the results buffer back from the device to the host:
	status = clEnqueueReadBuffer( cmdQueue, dSums, CL_TRUE, 0, Size, hSums, 0, NULL, NULL );
	if( status != CL_SUCCESS )
			fprintf( stderr, "clEnqueueReadBuffer failed\n" );


	double perf = (double)(Size) * (double)(Size)/ ( time1 - time0 ) / 1000000.;
    printf("Local Size: %i\tPerformance: %f MegaMultsPerSecond\n", LOCAL_SIZE, perf);

    for (int i = 0; i < Size; i++) {
		fprintf(out_f, "%f\n", hSums[i]);
	}

    return 1;
}

// wait until all queued tasks have taken place:
void
Wait( cl_command_queue queue )
{
      cl_event wait;
      cl_int      status;

      status = clEnqueueMarker( queue, &wait );
      if( status != CL_SUCCESS )
	      fprintf( stderr, "Wait: clEnqueueMarker failed\n" );

      status = clWaitForEvents( 1, &wait );
      if( status != CL_SUCCESS )
	      fprintf( stderr, "Wait: clWaitForEvents failed\n" );
}
