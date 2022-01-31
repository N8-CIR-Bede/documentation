// System includes
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <math.h>
// CUDA runtime
#include <cuda_runtime.h>

#define BILLION 1000000000L
 
template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
 if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
 if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
 if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
 if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
 if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
 if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce6(float *g_idata, float *g_odata, unsigned int n) {
extern __shared__ float sdata[];
//__shared__ float sdata[1024];
unsigned int tid = threadIdx.x;
unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int i = blockIdx.x*(blockSize*2) + tid;
unsigned int gridSize = blockSize*2*gridDim.x;
 sdata[tid] = 0;
// This loop needs to be generalised to deal with case where n is not a power of 2
 while (i < n){sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
 __syncthreads();
 if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
 if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
 if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
 if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
 if (tid < 32)warpReduce<blockSize>(sdata, tid);
 if (tid == 0) g_odata[blockIdx.x] = sdata[0];
 if (index == 0) { for (int j=1; j < blockDim.x; j++){g_odata[0] +=g_odata[j];}}
}

template <unsigned int blockSize>
__global__ void sumCommSingleBlock(float *a, int arraySize, float *out) {
    int idx = threadIdx.x;
    float sum = 0.0;
    for (int i = idx; i < arraySize; i += blockSize)
        sum += a[i];
    extern __shared__ float r[];
    r[idx] = sum;
    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { //uniform
        if (idx<size)
            r[idx] += r[idx+size];
        __syncthreads();
    }
    if (idx == 0)
        *out = r[0];
}


__device__ double f( double a )
{
    return (4.0 / (1.0 + a*a));
}

template <unsigned int blockSize>
__global__ void PiEstSingleBlock(long N, double *piest) {


    int idx = threadIdx.x;
    double h;
    double sum = 0.0;
    h = 1.0/(double)N;
    // Do the parallel partial sums for pi
    for (long i = idx+1; i <= N; i += blockSize)
        sum += f(h * ((double)i - 0.5));
     __shared__ double p[1024];
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

__global__ void init(float *a, unsigned int n){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
	  a[i] = (float)i+1.0;
	}
}



int main(void) {
	struct timespec start , stop ; // variables for timing
	double accum ; // elapsed time variable
	const unsigned int blockSize=1024;
	dim3 numThreads;
        dim3 numBlocks;
	unsigned int m=32768;
	float *a;
	float *partsums;
	double *mypi;
	long N = 1000000;
	
	double sum, h;
	h = 1.0/(double)N; // For CPU version of loop
	cudaMallocManaged(&a, m * sizeof(float));
	numThreads.x = blockSize;
	numBlocks.x = (m + numThreads.x - 1) / numThreads.x;
	cudaMallocManaged(&partsums, numBlocks.x * sizeof(float));
	printf("Numblks x %d Blksize x %d\n",numBlocks.x, numThreads.x);
	init<<<numBlocks,numThreads>>>(a,m);
	cudaDeviceSynchronize(); // This is critical if going to look at output on CPU!
	for (int i = 0; i < 4; i++) {
	    printf(" %10.1f", a[i]);
	}
	printf("\n");
	clock_gettime ( CLOCK_REALTIME ,&start );
	reduce6<blockSize><<<numBlocks,numThreads,blockSize*sizeof(float)>>>(a,partsums,m);
	cudaDeviceSynchronize();  // This is critical if going to look at output on CPU!
	clock_gettime ( CLOCK_REALTIME ,&stop );
 	for (int i = 0; i < 4; i++) {
 	    printf(" %10.1f", partsums[i]);
 	}
//        printf(" %10.1f", partsums[0]);
	printf("\n");
	accum =( stop.tv_sec - start.tv_sec )+ 
	       ( stop.tv_nsec - start.tv_nsec )/(double)BILLION ;
	printf (" Multiblock reduce : %lf sec %lf MBytes/s.\n",accum, 1.e-6*m*sizeof(float)/accum);
	clock_gettime ( CLOCK_REALTIME ,&start );
	sumCommSingleBlock<blockSize><<<1,numThreads>>>(a,m,partsums);
	clock_gettime ( CLOCK_REALTIME ,&stop );
	accum =( stop.tv_sec - start.tv_sec )+ 
	       ( stop.tv_nsec - start.tv_nsec )/(double)BILLION ;
	
	printf("1 block sum %10.1f", partsums[0]);
	printf("\n");
	printf (" Single block reduce : %lf sec %lf MBytes/s.\n",accum , 1.e-6*m*sizeof(float)/accum);
	printf("True answer %10.1f \n",0.5*(float)m*((float)m+1.0));
	cudaMallocManaged(&mypi, sizeof(double));
	clock_gettime ( CLOCK_REALTIME ,&start );
	PiEstSingleBlock<blockSize><<<1,numThreads>>>(N, mypi);
	cudaDeviceSynchronize(); 
	clock_gettime ( CLOCK_REALTIME ,&stop );
	accum =( stop.tv_sec - start.tv_sec )+ 
	       ( stop.tv_nsec - start.tv_nsec )/(double)BILLION ;
	
	printf("Pi estimate %.16f", mypi[0]);
	printf("\n");
	printf("Time to compute mypi is %.16f sec.\n",accum);
	sum  = 0.0;
	for (long i=1; i <= N; i++){
	   sum += 4.0/(1.0+h * ((double)i - 0.5)*h * ((double)i - 0.5));
	}
	printf("CPU pi is %.16f \n",h*sum);
}


