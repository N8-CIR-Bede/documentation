#include <stdio.h> // Reference???
const int N = 1024; const int blocksize = 16;
__global__ void add_matrix( float *a, float *b, float *c, int N, float rf, float pirkplus1) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
int index = i + j*N;
if ( i < N && j < N )
  c[index] = rf*__sinf((float)(i+1)*(float)(j+1)*pirkplus1); 
}

int main(void){
float *a = new float[N*N]; float *b = new float[N*N]; float *c = new float[N*N];
float two=2.0f, one=1.0f;
float pi,rkplus1,rf; // Generate square orthonormal matrices
pi = two * asin(one); 
rkplus1 = one/(float(N) + one);  
rf = sqrt(two*rkplus1);
for ( int i = 0; i < N*N; ++i ) {
  a[i] = 1.0f; b[i] = 3.5f; }

float *ad, *bd, *cd;

const int size = N*N*sizeof(float); 
cudaMalloc( (void**)&ad, size );
cudaMalloc( (void**)&bd, size ); 
cudaMalloc( (void**)&cd, size );
cudaMemcpy( ad, a, size, cudaMemcpyHostToDevice ); // COPY DATA TO GPU
cudaMemcpy( bd, b, size, cudaMemcpyHostToDevice );

dim3 dimBlock( blocksize, blocksize );
dim3 dimGrid( N/dimBlock.x, N/dimBlock.y );

add_matrix<<<dimGrid, dimBlock>>>( ad, bd, cd, N, rf, pi*rkplus1 );

cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost );
for (int i = 0; i < 10;i++) {
     printf(" %7.5f", c[i]);
}
printf("\n");
cudaFree( ad ); cudaFree( bd ); cudaFree( cd ); // CLEAN UP, RETURN
return 0;
}
