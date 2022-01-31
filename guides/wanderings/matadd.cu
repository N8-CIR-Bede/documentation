// Based heavily on https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/
#include <stdio.h> 
const int N = 1024; const int blocksize = 16;

__global__ void add_matrix( float *a, float *b, float *c, int N) {
int i = blockIdx.x * blockDim.x + threadIdx.x; // blockIdx, blockDim and threadIdx are predefined
int j = blockIdx.y * blockDim.y + threadIdx.y; // variables - initialised from meta-arguments
int index = i + j*N;
if ( i < N && j < N ) // Keep indices in range 
  c[index] = a[index] + b[index]; 
}

int main(void){
const int size = N*N*sizeof(float);
float *a ; float *b; float *c ;
float maxError = 0.0f;
cudaMallocManaged( (void**)&a, size );
cudaMallocManaged( (void**)&b, size ); 
cudaMallocManaged( (void**)&c, size );
for ( int i = 0; i < N*N; ++i ) {
  a[i] = 1.0f; b[i] = 3.5f; }

dim3 dimBlock( blocksize, blocksize );     // dim3 structure to deal with 1D, 2D or 3D thread collections.
dim3 dimGrid( N/dimBlock.x, N/dimBlock.y); // dimBlock.x - first dimension, dimBlock.y - second dimension
					   // dimBlock.z for third dimension (not used)
add_matrix<<<dimGrid, dimBlock>>>( a, b, c, N);    // Note meta arguments that pass information on 
						   // Number of thread groups (Grid) and number of
						   // threads in each group (Block).

// Wait for GPU to finish before accessing on host - major source of errors
      cudaDeviceSynchronize();
					 	   

for (int j = 0; j < N; j++){		
 for (int i = 0; i < N;i++) {
     maxError = fmax(maxError, fabs(c[i+j*N]-4.5f));
 }
}
printf("Max error: %.16f\n", maxError );

cudaFree( a ); cudaFree( b ); cudaFree( c ); // CLEAN UP, RETURN
return 0;
}
