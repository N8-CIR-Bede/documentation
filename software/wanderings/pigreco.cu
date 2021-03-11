// Approximation of Pi using a simple, and not optimized, CUDA program
// Copyleft Alessandro Re
// From https://gist.github.com/akiross/17e722c5bea92bd2c310324eac643df6
//
// GCC 6.x not supported by CUDA 8, I used compat version
//
// nvcc -std=c++11 -ccbin=gcc5 pigreco.cu -c
// g++5 pigreco.o -lcudart -L/usr/local/cuda/lib64 -o pigreco
//
// This code is basically equivalent to the following Python code:
//
// def pigreco(NUM):
//     from random import random as rand
//     def sqrad():
//         x, y = rand(), rand()
//         return x*x + y*y
//     return 4 * sum(1 - int(test()) for _ in range(NUM)) / NUM
//
// Python version takes, on this machine, 3.5 seconds to compute 10M tests
// CUDA version takes, on this machine, 1.6 seconds to compute 20.48G tests
//
#include <stdio.h>
#include <iostream>
#include <limits>
#include <cuda.h>
#include <curand_kernel.h>


using std::cout;
using std::endl;

typedef unsigned long long Count;
typedef std::numeric_limits<double> DblLim;

const Count WARP_SIZE = 32; // Warp size
const Count NBLOCKS = 1792; // Number of total cuda cores on my GPU 
			    // 5120 for v100; 1792 for Quadro P4000
const Count ITERATIONS = 1000000; // Number of points to generate (each thread)

// This kernel is 
__global__ void picount(Count *totals) {
	// Define some shared memory: all threads in this block
	__shared__ Count counter[WARP_SIZE];

	// Unique ID of the thread
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// Initialize RNG
	curandState_t rng;
	curand_init(clock64(), tid, 0, &rng);

	// Initialize the counter
	counter[threadIdx.x] = 0;

	// Computation loop
	for (int i = 0; i < ITERATIONS; i++) {
		float x = curand_uniform(&rng); // Random x position in [0,1]
		float y = curand_uniform(&rng); // Random y position in [0,1]
		counter[threadIdx.x] += 1 - int(x * x + y * y); // Hit test - I think this is clever- CA
	}

	// The first thread in *every block* should sum the results
	if (threadIdx.x == 0) {
		// Reset count for this block
		totals[blockIdx.x] = 0;
		// Accumulate results
		for (int i = 0; i < WARP_SIZE; i++) {
			totals[blockIdx.x] += counter[i];
		}
	}
}

int main(int argc, char **argv) {
	struct timespec start , stop ; // variables for timing
	double accum ; // elapsed time variable
	double pi25DT=3.141592653589793238462643;
	double estimate; 
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		cout << "CUDA device missing! Do you need to use optirun?\n";
		return 1;
	}
	cout << "Starting simulation with " << NBLOCKS << " blocks, " << WARP_SIZE << " threads, and " << ITERATIONS << " iterations\n";

	// Allocate host and device memory to store the counters
	Count *hOut, *dOut;
	hOut = new Count[NBLOCKS]; // Host memory
	cudaMalloc(&dOut, sizeof(Count) * NBLOCKS); // Device memory
	clock_gettime ( CLOCK_REALTIME ,&start ); // timer start
	// Launch kernel
	picount<<<NBLOCKS, WARP_SIZE>>>(dOut);
        cudaDeviceSynchronize(); // Need matrices to be defined before continue
	clock_gettime ( CLOCK_REALTIME ,&stop ); // timer stop
	// Copy back memory used on device and free
	cudaMemcpy(hOut, dOut, sizeof(Count) * NBLOCKS, cudaMemcpyDeviceToHost);
	cudaFree(dOut);

	// Compute total hits
	Count total = 0;
	for (int i = 0; i < NBLOCKS; i++) {
		total += hOut[i];
	}
	Count tests = NBLOCKS * ITERATIONS * WARP_SIZE;
	cout << "Approximated PI using " << tests << " random tests\n";

	// Set maximum precision for decimal printing
	cout.precision(DblLim::digits10); // Original code failed with max_digits10
	estimate = 4.0 * (double)total/(double)tests;
	cout << "PI ~= " << estimate << endl;
	printf("Pi  error is %.16f \n", pi25DT-estimate);
 	accum =( stop.tv_sec - start.tv_sec )+ // elapsed time create A
	       ( stop.tv_nsec - start.tv_nsec )*1.e-9 ;	
 	printf ("Monte pi took : %lf sec .\n",accum ); // print el. time

	return 0;
}
