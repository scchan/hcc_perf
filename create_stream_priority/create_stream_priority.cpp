
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#include <getopt.h>
#endif

#include "hip/hip_runtime.h"

#define CHECK_HIP_ERROR(a) { if ((a) != HIP_SUCCESS) { abort(); } }

__global__ void dummy() {
    #if 0
  *output = 0;
  for (int i = 0; i < N; ++i) {
    *output+=input[i];
  }
  #endif
}


void run(const int thread_id, const bool descending, const int num_streams_per_priority) {

  int lp, gp;
  CHECK_HIP_ERROR(hipDeviceGetStreamPriorityRange(&lp, &gp));
  printf("Stream priority range: %d to %d\n", lp, gp);

  int start_priority, end_priority, bump;
  if (descending) {
    start_priority = gp;
    end_priority = lp;
    bump = 1;
  }
  else {
    start_priority = lp;
    end_priority = gp;
    bump = -1; 
  }
  
  int num_streams = 0;
  std::vector<hipStream_t> v_stream;
  for (int p = start_priority; ; p+=bump) {
      for (int np = 0; np < num_streams_per_priority; ++np, ++num_streams) {
          printf("Thread %d: creating stream %d with priority %d\n", thread_id, num_streams, p);
          hipStream_t s;
          CHECK_HIP_ERROR(hipStreamCreateWithPriority(&s , hipStreamDefault, p));
          hipLaunchKernelGGL(dummy, dim3(1), dim3(1), 0, s);
          v_stream.push_back(s);
      }
      if (p == end_priority)
        break;
  } 

  for (auto& s : v_stream) {
      CHECK_HIP_ERROR(hipStreamSynchronize(s));
      CHECK_HIP_ERROR(hipStreamDestroy(s));
  }

}

int main(int argc, char* argv[]) {

  int num_threads = 16;
  int num_streams_per_priority = 30;

#ifdef _GNU_SOURCE
  while (1) {
    static struct option opts[] = {
      {"streams", required_argument, 0, 's'},
      {"threads", required_argument, 0, 't'},
      {0, 0, 0, 0}
    };
    
    int parse_idx = 0;
    int s = getopt_long(argc, argv, "s:t:", opts, &parse_idx);
    if (s == -1) break;

    switch (s) {
      case 's':
        num_streams_per_priority = std::stoi(std::string(optarg));
        break;
      case 't':
        num_threads = std::stoi(std::string(optarg));
        break;
      default:
        abort();
    }
  }
#endif // _GNU_SOURCE

  std::vector<std::thread> threads;
  bool descending = true;
  for (auto i = 0; i < num_threads; ++i) {
    threads.push_back(std::thread(run, i, descending, num_streams_per_priority));
    descending = !descending;
  }
  for (auto& t : threads) {
    t.join();
  }
  return 0;
}