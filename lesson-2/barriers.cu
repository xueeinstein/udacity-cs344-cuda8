/*
 * Shift array forward and sync with barriers
 *
 * compile:
 * nvcc -o barriers barriers.cu
 */

#include <stdio.h>

#define ARRAY_SIZE 16

__global__ void shiftArray() {
  int idx = threadIdx.x;
  __shared__ int array[ARRAY_SIZE];

  array[idx] = threadIdx.x;
  __syncthreads();

  if (idx < ARRAY_SIZE - 1) {
    int tmp = array[idx + 1];
    __syncthreads();

    array[idx] = tmp;
    __syncthreads();
  }

  printf("thread ID: %d, array value: %d\n", idx, array[idx]);
}

int main(int argc, char ** argv) {

  // launch the kernel
  shiftArray<<<1, ARRAY_SIZE>>>();

  // force the printf()s to flush
  cudaDeviceSynchronize();

  printf("That's all!\n");
  return 0;
}