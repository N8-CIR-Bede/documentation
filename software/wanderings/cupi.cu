// Modified from global reduction code in 
// https://sodocumentation.net/cuda/topic/6566/parallel-reduction--e-g--how-to-sum-an-array-

// System includes
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <math.h>
// CUDA runtime
#include <cuda_runtime.h>

#define BILLION 1000000000L
 
// Note this cannot be executed on the CPU - just on the GPU
__device__ double f( double a )
{
    return (4.0 / (1.0 + a*a));
}


__global__ void PiEstSingleBlock(long N, double *piest) {


    int idx = threadIdx.x;
    int blockSize = blockDim.x; // We are exploiting the fact that there is just one thread group
    double h;
    double sum = 0.0;
    h = 1.0/(double)N;
    // Do the parallel partial sums for pi
    for (long i = idx+1; i <= N; i += blockSize)
        sum += f(h * ((double)i - 0.5));
    __shared__ double p[1024]; // The maximum number of threads is 1024
			       // We can make this storage dynamic in size to paramterise
			       // over the number of threads used. 
    // Now add the partial sums together 
    p[idx] = h*sum;
    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { //uniform
        if (idx<size)
            p[idx] += p[idx+size];
        __syncthreads();
    }
    if (idx == 0)
        *piest = p[0];
}


int main(void) {
	struct timespec start , stop ; // variables for timing
	double accum ; // elapsed time variable
	const unsigned int blockSize=1024;
	dim3 numThreads;
	double pi25DT=3.141592653589793238462643;
	double x;
	double *mypi;
	long N = 1000000;
	
	double sum, h;
	h = 1.0/(double)N; // For CPU version of loop
	
	numThreads.x = blockSize;
	
	cudaMallocManaged(&mypi, sizeof(double));
	clock_gettime ( CLOCK_REALTIME ,&start );
	PiEstSingleBlock<<<1,numThreads>>>(N, mypi);
	cudaDeviceSynchronize(); // Cannot get sensible timing without synchronising host and device
	clock_gettime ( CLOCK_REALTIME ,&stop );
	accum =( stop.tv_sec - start.tv_sec )+ 
	       ( stop.tv_nsec - start.tv_nsec )/(double)BILLION ;
	
	printf("Pi estimate %.16f error is %.16f", mypi[0],pi25DT-mypi[0]);
	printf("\n");
	printf("Time to compute mypi is %lf sec.\n",accum);
	clock_gettime ( CLOCK_REALTIME ,&start );
	sum  = 0.0;
	for (long i=1; i <= N; i++){
	   x = h * ((double)i - 0.5);
	   sum += 4.0/(1.0+x*x);
	}
	clock_gettime ( CLOCK_REALTIME ,&stop );
	accum =( stop.tv_sec - start.tv_sec )+ 
	       ( stop.tv_nsec - start.tv_nsec )/(double)BILLION ;
	
	printf("CPU pi is %.16f  error is %.16f\n",h*sum,pi25DT-h*sum);
	printf("Time to compute CPU pi is %lf sec.\n",accum);
}
