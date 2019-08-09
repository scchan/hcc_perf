
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

__global__ void dummy(int* output, int init_value, int multiplier, int num_iters) {
  *output = init_value;
  for (int i = 0; i < num_iters; ++i) {
    *output+=*output*multiplier;
  }
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

  struct gpu_resources {
    hipStream_t s = 0;
    int stream_priority;
    int* output = nullptr;
    gpu_resources(int stream_priority) : stream_priority(stream_priority) {
      CHECK_HIP_ERROR(hipStreamCreateWithPriority(&s , hipStreamDefault, stream_priority));
      CHECK_HIP_ERROR(hipMalloc(&output, sizeof(output))); 
    }
    gpu_resources(gpu_resources&& t) : 
       s(t.s), stream_priority(t.stream_priority),
       output(t.output) {
      t.s = 0;
      t.output = nullptr;
    }
    ~gpu_resources() {
      if (s)      CHECK_HIP_ERROR(hipStreamDestroy(s));
      if (output) CHECK_HIP_ERROR(hipFree(output));
    }
  };

  std::vector<gpu_resources> v_gpu_resources;
  int num_streams = 0;
  for (int p = start_priority; ; p+=bump) {
      for (int np = 0; np < num_streams_per_priority; ++np, ++num_streams) {
          printf("Thread %d: creating stream %d with priority %d\n", thread_id, num_streams, p);
          gpu_resources r(p);
          hipLaunchKernelGGL(dummy, dim3(1), dim3(1), 0, r.s, r.output, 0, 1, 1000);
          v_gpu_resources.push_back(std::move(r));
      }
      if (p == end_priority)
        break;
  }
  for (auto& r : v_gpu_resources) {
    CHECK_HIP_ERROR(hipStreamSynchronize(r.s)); 
  }
}

int main(int argc, char* argv[]) {

  int num_threads = 8;
  int num_streams_per_priority = 24;

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