// nvcc 036sgemm.c -lcublas
// From "Matrix computations on the GPU CUBLAS, CUSOLVER and MAGMA by example. Version 2017" 
// by Andrzej Chrzeszczyk and Jacob Anders
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*( ld ))+( i ))
#define m 6 // a - mxk matrix
#define n 4 // b - kxn matrix
#define k 5 // c - mxn matrix
int main(void) {
	cudaError_t cudaStat; // cudaMalloc status
	cublasStatus_t stat; // CUBLAS functions status
	cublasHandle_t handle; // CUBLAS context
	int i, j; // i-row index ,j- column index
	float* a; // mxk matrix a on the host
	float* b; // kxn matrix b on the host
	float* c; // mxn matrix c on the host
	a = (float*)malloc(m*k * sizeof(float)); // host memory for a
	b = (float*)malloc(k*n * sizeof(float)); // host memory for b
	c = (float*)malloc(m*n * sizeof(float)); // host memory for c
	// define an mxk matrix a column by column
	int ind = 11; // a:
	for (j = 0;j < k;j++) {                           // 11 ,17 ,23 ,29 ,35
		for (i = 0;i < m;i++) {                   // 12 ,18 ,24 ,30 ,36
			a[IDX2C(i, j, m)] = (float)ind++; // 13 ,19 ,25 ,31 ,37
		}                                         // 14 ,20 ,26 ,32 ,38
	}                                                 // 15 ,21 ,27 ,33 ,39
	                                                  // 16 ,22 ,28 ,34 ,40
	// print a row by row
	printf("a:\n");
	for (i = 0;i < m;i++) {
		for (j = 0;j < k;j++) {
			printf(" %5.0f", a[IDX2C(i, j, m)]);
		}
		printf("\n");
	}
	// define a kxn matrix b column by column
	ind = 11; // b:
	for (j = 0;j < n;j++) {                           // 11 ,16 ,21 ,26
		for (i = 0;i < k;i++) { 		  // 12 ,17 ,22 ,27
			b[IDX2C(i, j, k)] = (float)ind++; // 13 ,18 ,23 ,28
		} 					  // 14 ,19 ,24 ,29
	}  						  // 15 ,20 ,25 ,30
	// print b row by row
	printf("b:\n");
	for (i = 0;i < k;i++) {
		for (j = 0;j < n;j++) {
			printf(" %5.0f", b[IDX2C(i, j, k)]);
		}
		printf("\n");
	}
	// define an mxn matrix c column by column
	ind = 11; // c:
	for (j = 0;j < n;j++) { 			  // 11 ,17 ,23 ,29
		for (i = 0;i < m;i++) { 		  // 12 ,18 ,24 ,30
			c[IDX2C(i, j, m)] = (float)ind++; // 13 ,19 ,25 ,31
		} 					  // 14 ,20 ,26 ,32
	} 					          // 15 ,21 ,27 ,33
	 						  // 16 ,22 ,28 ,34
	// print c row by row
	printf("c:\n");
	for (i = 0;i < m;i++) {
		for (j = 0;j < n;j++) {
			printf(" %5.0f", c[IDX2C(i, j, m)]);
		}
		printf("\n");
	}
	// on the device
	float* d_a; // d_a - a on the device
	float* d_b; // d_b - b on the device
	float* d_c; // d_c - c on the device
	cudaStat = cudaMalloc((void **)& d_a, m*k * sizeof(*a)); // device
	// memory alloc for a
	cudaStat = cudaMalloc((void **)& d_b, k*n * sizeof(*b)); // device
	// memory alloc for b
	cudaStat = cudaMalloc((void **)& d_c, m*n * sizeof(*c)); // device
	// memory alloc for c
	stat = cublasCreate(&handle); // initialize CUBLAS context
	// copy matrices from the host to the device
	stat = cublasSetMatrix(m, k, sizeof(*a), a, m, d_a, m); //a -> d_a
	stat = cublasSetMatrix(k, n, sizeof(*b), b, k, d_b, k); //b -> d_b
	stat = cublasSetMatrix(m, n, sizeof(*c), c, m, d_c, m); //c -> d_c
	float al = 1.0f; // al =1
	float bet = 1.0f; // bet =1
	// matrix - matrix multiplication : d_c = al*d_a *d_b + bet *d_c
	// d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix ;
	// al ,bet -scalars
	stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, d_a,
		m, d_b, k, &bet, d_c, m);
	stat = cublasGetMatrix(m, n, sizeof(*c), d_c, m, c, m); // cp d_c ->c
	printf("c after Sgemm :\n");
	for (i = 0;i < m;i++) {
		for (j = 0;j < n;j++) {
			printf(" %7.0f", c[IDX2C(i, j, m)]); // print c after Sgemm
		}
		printf("\n");
	}
	cudaFree(d_a); // free device memory
	cudaFree(d_b); // free device memory
	cudaFree(d_c); // free device memory
	cublasDestroy(handle); // destroy CUBLAS context
	free(a); // free host memory
	free(b); // free host memory
	free(c); // free host memory
	return EXIT_SUCCESS;
}
