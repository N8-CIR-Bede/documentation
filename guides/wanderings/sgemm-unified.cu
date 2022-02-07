// nvcc 036 sgemm.cu -lcublas
#include <stdio.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*( ld ))+( i ))
#define m 6 // a - mxk matrix
#define n 4 // b - kxn matrix
#define k 5 // c - mxn matrix
int main(void) {
	cublasHandle_t handle; // CUBLAS context
	int i, j; // i-row index ,j- column index
	float* a; // mxk matrix
	float* b; // kxn matrix
	float* c; // mxn matrix
	// unified memory for a,b,c
	cudaMallocManaged(&a, m*k * sizeof(float)); 
	cudaMallocManaged(&b, k*n * sizeof(float));
	cudaMallocManaged(&c, m*n * sizeof(float));
	// define an mxk matrix a column by column
	int ind = 11; // a:
	for (j = 0;j < k;j++) { 		  // 11 ,17 ,23 ,29 ,35
		for (i = 0;i < m;i++) { 	  // 12 ,18 ,24 ,30 ,36
		a[IDX2C(i, j, m)] = (float)ind++; // 13 ,19 ,25 ,31 ,37
		} 				  // 14 ,20 ,26 ,32 ,38
	} 					  // 15 ,21 ,27 ,33 ,39
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
	for (j = 0;j < n;j++) {  			  // 11 ,16 ,21 ,26
		for (i = 0;i < k;i++) { 		  // 12 ,17 ,22 ,27
			b[IDX2C(i, j, k)] = (float)ind++; // 13 ,18 ,23 ,28
		} 				   	  // 14 ,19 ,24 ,29
	} 						  // 15 ,20 ,25 ,30
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
	for (j = 0;j < n;j++) {  			  // 11 ,17 ,23 ,29
		for (i = 0;i < m;i++) { 		  // 12 ,18 ,24 ,30
			c[IDX2C(i, j, m)] = (float)ind++; // 13 ,19 ,25 ,31
		}  					  // 14 ,20 ,26 ,32
	} 						  // 15 ,21 ,27 ,33
							  // 16 ,22 ,28 ,34
	// print c row by row
	printf("c:\n");
	for (i = 0;i < m;i++) {
		for (j = 0;j < n;j++) {
			printf(" %5.0f", c[IDX2C(i, j, m)]);
		}
		printf("\n");
	}
	cublasCreate(&handle); // initialize CUBLAS context
	float al = 1.0f; // al =1
	float bet = 1.0f; // bet =1
	// matrix - matrix multiplication : c = al*a*b + bet *c
	// a -mxk matrix , b -kxn matrix , c -mxn matrix ;
	// al ,bet -scalars
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, a, m, b, k,
		&bet, c, m);
	cudaDeviceSynchronize();
	printf("c after Sgemm :\n");
	for (i = 0;i < m;i++) {
		for (j = 0;j < n;j++) {
			printf(" %7.0f", c[IDX2C(i, j, m)]); // print c after Sgemm
		}
		printf("\n");
	}
	cudaFree(a); // free memory
	cudaFree(b); // free memory
	cudaFree(c); // free memory
	cublasDestroy(handle); // destroy CUBLAS context
	return EXIT_SUCCESS;
}
