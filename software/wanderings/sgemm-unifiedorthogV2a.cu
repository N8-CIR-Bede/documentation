// nvcc 036 sgemm.cu -lcublas
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"


#define printlim 10
#define BILLION 1000000000L
#define IDX2C(i,j,ld) (((j)*( ld ))+( i ))

__global__  void initmatrix(int m, int n, float *x) {
	float rk,rkplus1,pi;
	float one=1.0f;
	float two=2.0f;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
        long index = i + j*m;
	pi = two*asin(one);
        rkplus1 = one/(float(m) + one);  
	rk = sqrt(two*rkplus1);
        if ( i < m && j < n) {
	  x[index] = rk*__sinf((float)(i+1)*(float)(j+1)*pi*rkplus1);
	   
        }
     }

__global__  void initident(int m, int n, float *x) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
        long index = i + j*m;
        if ( i < m && j < n) {
	  x[index] = 0.e0;
	  if ( i == j ) x[index] = 1.e0;
        }
     }


int main(void) {
	struct timespec start , stop ; // variables for timing
	double accum ; // elapsed time variable
	cublasHandle_t handle; // CUBLAS context	
	int m=20000; // #defines in cuda call get the literal swapped in - invalid code
	int k=20000;
	int n=20000;
	int i,j; // i-row index, j- column index
	float *a; // mxk matrix
	float *b; // kxn matrix
	float *c; // mxn matrix
        dim3 blockSize;
        dim3 numBlocks;
  	
	// unified memory for a,b,c
	cudaMallocManaged(&a, m*k * sizeof(cuComplex));
	cudaMallocManaged(&b, k*n * sizeof(cuComplex));
	cudaMallocManaged(&c, m*n * sizeof(cuComplex));
	
	clock_gettime ( CLOCK_REALTIME ,&start ); // timer start

        blockSize.x = 32; // Have a 32 by 32 block - 1024 threads - max allowed
	blockSize.y = 32;

        numBlocks.x = (m + blockSize.x - 1) / blockSize.x;
        numBlocks.y = (k + blockSize.y - 1) / blockSize.y;	
	printf("Numblks x %d Blksize x %d\n",numBlocks.x, blockSize.x);
        // a - orthogonal symmetric matix
        initmatrix<<<numBlocks,blockSize>>>(m,k,a);

        numBlocks.x = (k + blockSize.x - 1) / blockSize.x;
        numBlocks.y = (n + blockSize.y - 1) / blockSize.y;
        // b - orthogonal symmetric matix
        initmatrix<<<numBlocks,blockSize>>>(k,n,b);

        numBlocks.x = (m + blockSize.x - 1) / blockSize.x;
        numBlocks.y = (n + blockSize.y - 1) / blockSize.y;
	// c - ones along diagonal; zero elsewhere
        initident<<<numBlocks,blockSize>>>(m,n,c);
        cudaDeviceSynchronize(); // Need matrices to be defined before continue
	clock_gettime ( CLOCK_REALTIME ,&stop ); // timer stop
 	accum =( stop.tv_sec - start.tv_sec )+ // elapsed time create A
	       ( stop.tv_nsec - start.tv_nsec )/(double)BILLION ;	
 	printf (" Initialise : %lf sec .\n",accum ); // print el. time
	// print a row by row
	printf("a:\n");
	
	cublasCreate(&handle); // initialize CUBLAS context
	float al = 1.0f; // al =1
	float bet = 1.0f; // bet =1
	// matrix - matrix multiplication : c = al*a*b + bet *c
	// a -mxk matrix , b -kxn matrix , c -mxn matrix ;
	// al ,bet -scalars
	clock_gettime ( CLOCK_REALTIME ,&start ); // timer start
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, a, m, b, k,
		&bet, c, m);
	cudaDeviceSynchronize();
	clock_gettime ( CLOCK_REALTIME ,&stop );  // timer stop
 	accum =( stop.tv_sec - start.tv_sec )+    // elapsed time for MN(2K+3) flops
	       ( stop.tv_nsec - start.tv_nsec )/(double)BILLION ;
        printf (" sgemm : %lf sec .\n",accum ); // print el. time
        printf (" Gflops : %lf \n",(double)m*(double)n*(2.e0*k+3.0)*1.e-9/accum); // print Gflops
	printf("c after Sgemm :\n");
	for (i = 0;i < printlim;i++) {
		for (j = 0;j < printlim;j++) {
			printf(" %7.4f", c[IDX2C(i, j, m)]); // print c after Sgemm
		}
		printf("\n");
	}

	cudaFree(a); // free memory
	cudaFree(b); // free memory
	cudaFree(c); // free memory
	cublasDestroy(handle); // destroy CUBLAS context
	return EXIT_SUCCESS;
}
