/*
 * Demonstrate atomic memory operations
 *
 * compile:
 * nvcc -o atomics atomics.cu -I ../inc
 */

#include <stdio.h>
#include "gputimer.h"

#define NUM_THREADS 1000000
#define ARRAY_SIZE 100

#define BLOCK_WIDTH 1000

void print_array(int *array, int size) {
  printf("{ ");
  for (int i = 0; i < size; i++) {
    printf("%d ", array[i]);
  }
  printf("}\n");
}

__global__ void increment_naive(int *g) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  idx = idx % ARRAY_SIZE;
  g[idx] = g[idx] + 1;
}

__global__ void increment_atomic(int *g) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  idx = idx % ARRAY_SIZE;
  atomicAdd(& g[idx], 1);
}

int main(int argc, char **argv) {
  int is_atomic = 0;
  if (argc == 2) {
    if (strcmp(argv[1], "-t") == 0) is_atomic = 1;
  }
  GpuTimer timer;
  int h_array[ARRAY_SIZE];
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

  int * d_array;
  cudaMalloc((void **) &d_array, ARRAY_BYTES);
  cudaMemset((void *) d_array, 0, ARRAY_BYTES);

  timer.Start();
  if (is_atomic == 1) {
    printf("atomic!\n");
    increment_atomic<<<NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
  } else {
    increment_naive<<<NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
  }
  timer.Stop();

  cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
  print_array(h_array, ARRAY_SIZE);
  printf("Time elapsed = %g ms\n", timer.Elapsed());

  cudaFree(d_array);
  return 0;
}