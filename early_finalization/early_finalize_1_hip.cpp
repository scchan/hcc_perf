#include <stdio.h>
#include "hip/hip_runtime.h"
#include <vector>

extern __global__ void sum(int* input, const size_t N, int* output);
extern __global__ void neg_sum(int* input, const size_t N, int* output);

#ifdef _OBJ_1
__global__ void sum(int* input, const size_t N, int* output) {
  *output = 0;
  for (int i = 0; i < N; ++i) {
    *output+=input[i];
  }
}
#endif

#ifdef _OBJ_2
__global__ void neg_sum(int* input, const size_t N, int* output) {
  *output = 0;
  for (int i = 0; i < N; ++i) {
    *output+=-input[i];
  }
}
#endif

#ifdef _OBJ_3
int main() {
  constexpr size_t N = 64;
  std::vector<int> host_input(N);
  for (int i = 0; i < N; ++i)
    host_input[i] = i;

  int* input;
  hipMalloc(&input, N * sizeof(int));
  hipMemcpy(input, host_input.data(), host_input.size()*sizeof(int), hipMemcpyHostToDevice);

  int* output;
  hipMalloc(&output, sizeof(int));

  constexpr unsigned int blocks = 1;
  constexpr unsigned int threads_per_block = 1;

  int s = 0;
  hipLaunchKernelGGL(sum, dim3(blocks), dim3(threads_per_block), 0, 0, input, N, output);
  hipMemcpy(&s, output, sizeof(int), hipMemcpyDeviceToHost);

  int ns = 0;
  hipLaunchKernelGGL(neg_sum, dim3(blocks), dim3(threads_per_block), 0, 0, input, N, output);
  hipMemcpy(&ns, output, sizeof(int), hipMemcpyDeviceToHost);

  int host_s = 0;
  int host_ns = 0;
  for (int i = 0; i < N; ++i) {
    host_s+=host_input[i];
    host_ns+=-host_input[i];
  }

  printf("sum:  expected=%d, actual=%d\n", host_s, s);
  printf("neg_sum:  expected=%d, actual=%d\n", host_ns, ns);

  return (s==host_s && ns==host_ns)?0:1;
}
#endif

