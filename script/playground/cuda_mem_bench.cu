#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  size_t N = 2147483648 * 2;
  float *y, *d_y;
  // float *x, *d_x;
  y = (float*)malloc(N*sizeof(float));
  // x = (float*)malloc(N*sizeof(float));
  // cudaMallocHost((void**)&y, N*sizeof(float));
  // cudaMallocHost((void**)&x, N*sizeof(float));
  

  // cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (size_t i = 0; i < N; i++) {
    // x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  // saxpy<<<(N+511)/512, 512>>>(N, 2.0f, d_x, d_y);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // float maxError = 0.0f;
  // for (int i = 0; i < N; i++) {
  //   maxError = max(maxError, abs(y[i]-4.0f));
  // }

  // printf("Max error: %f\n", maxError);
  printf("Effective Bandwidth (GB/s): %f\n", N*4/milliseconds/1e6);
}
