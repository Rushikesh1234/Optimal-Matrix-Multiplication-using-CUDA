/*
 * _MATRIXMUL_GPU_CU_
 *
 * 2022 Mert SIDE
 *
 * CS5375 Computer Systems Organization and Architecture 
 * Guest Lecture: GPU Programming
 *
 * Multiplying two matrices on the GPU
 *
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

// ------------------------------------------------------------------ GPUmatmul
__global__ void GPUmatmul(int N, float *x, float *y, float *ans)
{
  // each thread computes one element of the block sub-matrix
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < N && j < N)
  {
   for (int k = 0; k < N; k++) 
   {
     float a = x[i*N+k];
     float b = y[k*N+j];
     ans[i*N+j] += a*b;
   }
  }
}

// ---------------------------------------------------------------------- check
bool check(int N, float *ans)
{
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      if(ans[i*N+j] != 20.0) return false;
    }
  }
  return true;
}

// ----------------------------------------------------------------------- MAIN
int main(void)
{
  // size of matrix
  int N = 1<<9; // binary left-shift: 1 * 2^9 = 512
  printf("Size of matrix (N) is %d by %d.n", N, N);
  int iter = 3;
  clock_t t;
  
  // Martices
  N = 1<<3;
  float *x, *y, *ans;

  // TODO: Allocate Unified Memory - accessible from both CPU and GPU
  
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  cudaMallocManaged(&ans, N*sizeof(float));  

  // ..........................................................................
  // initialize x,y and ans arrays on the host
  for (int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      x[i*N+j] = 5;
      y[i*N+j] = (i==j?1:0);
      ans[i*N+j] = (float)0.000000000000;
    }
  }

  // ..........................................................................
  float avg=0;
  std::cout<<"Starting Optimized GPU computation"<<std::endl;
  // Run kernel on GPU
  for(int i = 0; i <= iter; i++) {
    t = clock();


  // Thread size for CUDA excution
  int thread = 4;

  // Block size
  int block = N / thread;
  
  // Use dim3 structs for block  and grid dimensions
  dim3 threads(thread, thread);
  dim3 blocks(block, block);
 

  GPUmatmul<<<blocks,threads>>>(N, x, y,ans);
    
  cudaDeviceSynchronize();

  t = clock() - t;
  if(i) avg += t; //we will ignore the first run
  // printf ("It took GPU-%d %f ms.n",i,(((float)t)/CLOCKS_PER_SEC)*1000);
  }

  avg /= iter;
  avg /= CLOCKS_PER_SEC;
  avg *= 1000;
 
 /*
  for(int i=0; i<N; i++)
  {
    for(int j=0; j<N; j++)
    {
      printf("%f ", ans[i*N+j]); 
    }
  }
*/
  printf("n");
  printf("It took %f ms on avg.n", avg);
  if(check(N,ans)) std::cout<<"RUN OK."<<std::endl;
  else std::cout<<"RUN NOT OK."<<std::endl;
  
  // ..........................................................................
  
  // TODO: Free memory
  
  cudaFree(x);
  cudaFree(y);
  cudaFree(ans);

  return 0;
}
/* EOF */
}
