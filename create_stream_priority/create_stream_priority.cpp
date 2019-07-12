
#include <cstdio>
#include <vector>
#include "hip/hip_runtime.h"


__global__ void dummy() {
    #if 0
  *output = 0;
  for (int i = 0; i < N; ++i) {
    *output+=input[i];
  }
  #endif
}


int main() {

  constexpr int num_stream_per_priority = 30;
  std::vector<hipStream_t> v_stream;

  int lp, gp;
  hipDeviceGetStreamPriorityRange(&lp, &gp);
  printf("Stream priority range: %d to %d\n", lp, gp);
 
  int num_streams = 0;
  for (int p = gp; p <= lp; ++p) {
      for (int np = 0; np < num_stream_per_priority; ++np, ++num_streams) {
          printf("hcc-queue - test - Creating stream %d with priority %d\n", num_streams, p);
          hipStream_t s;
          hipStreamCreateWithPriority(&s , hipStreamDefault, p);
          hipLaunchKernelGGL(dummy, dim3(1), dim3(1), 0, s);
          v_stream.push_back(s);
      }
  } 

  for (auto& s : v_stream) {
      hipStreamDestroy(s);
  }

  return 0;
}