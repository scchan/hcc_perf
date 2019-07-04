#include <stdio.h>
#include "hip/hip_runtime.h"
#include <vector>
#include <thread>

extern __global__ void sum(hipLaunchParm lp, int* input, const size_t N, int* output);
extern __global__ void neg_sum(hipLaunchParm lp, int* input, const size_t N, int* output);

__global__ void sum(int* input, const size_t N, int* output) {
  *output = 0;
  for (int i = 0; i < N; ++i) {
    *output+=input[i];
  }
}

__global__ void neg_sum(int* input, const size_t N, int* output) {
  *output = 0;
  for (int i = 0; i < N; ++i) {
    *output+=-input[i];
  }
}

int run_hip_sums() {
  constexpr size_t N = 2048;
  std::vector<int> host_input(N);
  for (int i = 0; i < N; ++i)
    host_input[i] = i;

  hipStream_t stream = nullptr;
  hipStreamCreate(&stream);

  int* input;
  hipMalloc(&input, N * sizeof(int));
  hipMemcpy(input, host_input.data(), host_input.size()*sizeof(int), hipMemcpyHostToDevice);

  int* output;
  hipMalloc(&output, sizeof(int));

  constexpr unsigned int blocks = 1;
  constexpr unsigned int threads_per_block = 1;

  int s = 0;
  hipLaunchKernelGGL(sum, dim3(blocks), dim3(threads_per_block), 0, stream, input, N, output);
  hipMemcpy(&s, output, sizeof(int), hipMemcpyDeviceToHost);

  int ns = 0;
  hipLaunchKernelGGL(neg_sum, dim3(blocks), dim3(threads_per_block), 0, stream, input, N, output);
  hipMemcpy(&ns, output, sizeof(int), hipMemcpyDeviceToHost);

  int host_s = 0;
  int host_ns = 0;
  for (int i = 0; i < N; ++i) {
    host_s+=host_input[i];
    host_ns+=-host_input[i];
  }

  hipStreamDestroy(stream);

  printf("sum:  expected=%d, actual=%d\n", host_s, s);
  printf("neg_sum:  expected=%d, actual=%d\n", host_ns, ns);

  return (s==host_s && ns==host_ns)?0:1;
}

int main() {
  int num_threads_per_device = 64;
  std::vector<std::thread> threads;
  int num_iter_per_thread = 100;

  int gpuCount;
  hipGetDeviceCount(&gpuCount);
  printf("gpuCount: %d\n", gpuCount);

  for (int d = 0; d < gpuCount; ++d) {
    hipSetDevice(d);
    for (int i = 0; i < num_threads_per_device; ++i) {
      threads.push_back(std::move(
        std::thread(
          [&]() {
            for (int i = 0; i < num_iter_per_thread; ++i) {
              run_hip_sums();
            }
          }
        )
      ));
    }
  }

  for (auto& t : threads) {
    t.join();
  }
  return 0;
}

