
#include <stdio.h>
#include "hip/hip_runtime.h"
#include <vector>

template<typename T>
__global__ void add(T* input1, T* input2, int N) {
  for(int i = 0; i < N; i++) {
    input1[i] += input2[i];
  }
}


template<typename T>
bool run_add() {
  
  constexpr size_t N = 64;
  std::vector<T> host_input(N);
  std::vector<T> host_expected(N);
  for (int i = 0; i < N; ++i) {
    host_input[i] = (T)i;
    host_expected[i] = host_input[i] + host_input[i];
  }

  T* input1;
  hipMalloc(&input1, N * sizeof(T));
  hipMemcpy(input1, host_input.data(), host_input.size()*sizeof(T), hipMemcpyHostToDevice);


  T* input2;
  hipMalloc(&input2, N * sizeof(T));
  hipMemcpy(input2, host_input.data(), host_input.size()*sizeof(T), hipMemcpyHostToDevice);


  constexpr unsigned int blocks = 1;
  constexpr unsigned int threads_per_block = 1;
  hipLaunchKernelGGL(add<T>, dim3(blocks), dim3(threads_per_block), 0, 0, input1, input2, N);

  hipMemcpy(host_input.data(), input1, host_input.size()*sizeof(T), hipMemcpyDeviceToHost);

  bool equal = true;
  for (int i = 0; i < N; i++) {
    equal &= (host_input[i] == host_expected[i]);
  }
  return equal;
}


