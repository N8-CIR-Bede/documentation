// nvcc 036 sgemm.cu -lcublas
#include <stdio.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*( ld ))+( i ))
#define m 10000 // a - mxk matrix
#define n 10000 // b - kxn matrix
#define k 10000 // c - mxn matrix
#define printlim 8
#define BILLION 1000000000L;
int main(void) {
	struct timespec start , stop ; // variables for timing
	double accum ; // elapsed time variable
	cublasHandle_t handle; // CUBLAS context
	int i, j; // i-row index ,j- column index
	float* a; // mxk matrix
	float* b; // kxn matrix
	float* c; // mxn matrix
	float two=2.0f, one=1.0f, zero=0.0f;
	float pi,rkplus1,rf; // Generate square orthonormal matrices
	pi = two * asin(one); 
  	rkplus1 = one/(float(m) + one); //assumes m=n=k for now
	rf = sqrt(two*rkplus1);
	// unified memory for a,b,c
	cudaMallocManaged(&a, m*k * sizeof(cuComplex));
	cudaMallocManaged(&b, k*n * sizeof(cuComplex));
	cudaMallocManaged(&c, m*n * sizeof(cuComplex));
	// define an mxk matrix a column by column
	// The matrices generated are orthonormal when square, so
	// A*B = I
	// Generate a and compute how long it takes on CPU
	clock_gettime ( CLOCK_REALTIME ,&start ); // timer start
	for (j = 0;j < k;j++) {
	     for (i = 0;i < m;i++) {  
		a[IDX2C(i, j, m)] = rf*sin((i+1)*(j+1)*pi*rkplus1);
		}  
	}  
	clock_gettime ( CLOCK_REALTIME ,&stop ); // timer stop
 	accum =( stop.tv_sec - start.tv_sec )+ // elapsed time create A
	       ( stop.tv_nsec - start.tv_nsec )/(double)BILLION ;	
 	printf (" Create a : %lf sec .\n",accum ); // print el. time
	// print part of a row by row
	printf("a:\n");
	for (i = 0;i < printlim;i++) {
		for (j = 0;j < printlim;j++) {
			printf(" %.7f", a[IDX2C(i, j, m)]);
		}
		printf("\n");
	}
	// define a kxn matrix b column by column
	// b:
	for (j = 0;j < n;j++) {  
		for (i = 0;i < k;i++) {  
			b[IDX2C(i, j, k)] = rf*sin((i+1)*(j+1)*pi*rkplus1);
		}  
	}  
	// print part of b row by row
	printf("b:\n");
	for (i = 0;i < printlim;i++) {
		for (j = 0;j < printlim;j++) {
			printf(" %.7f", b[IDX2C(i, j, k)]);
		}
		printf("\n");
	}
	// define an mxn matrix c column by column
	// c:
	for (j = 0;j < n;j++) {  
		for (i = 0;i < m;i++) {  
			c[IDX2C(i, j, m)] = rf*sin((i+1)*(j+1)*pi*rkplus1);
		}  
	}  
	//  
	// print part of c row by row
	printf("c:\n");
	for (i = 0;i < printlim;i++) {
		for (j = 0;j < printlim;j++) {
			printf(" %.7f", c[IDX2C(i, j, m)]);
		}
		printf("\n");
	}
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
	clock_gettime ( CLOCK_REALTIME ,&stop ); // timer stop
 	accum =( stop.tv_sec - start.tv_sec )+ // elapsed time MN(2K+3)
	       ( stop.tv_nsec - start.tv_nsec )/(double)BILLION ;
        printf (" sgemm : %lf sec .\n",accum ); // print el. time
        printf (" Gflops : %lf \n",(double)m*(double)n*(2.e0*k+3.0)*1.e-9/accum); // print Gflops
	printf("c after Sgemm :\n");
	for (i = 0;i < printlim;i++) {
		for (j = 0;j < printlim;j++) {
			printf(" %.7f", c[IDX2C(i, j, m)]); // print c after Sgemm
		}
		printf("\n");
	}

	cudaFree(a); // free memory
	cudaFree(b); // free memory
	cudaFree(c); // free memory
	cublasDestroy(handle); // destroy CUBLAS context
	return EXIT_SUCCESS;
}
