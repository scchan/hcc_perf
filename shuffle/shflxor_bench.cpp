#include <vector>
#include <random>
#include <future>
#include <iostream>
#include <algorithm>
#include <typeinfo>
#include "hip_runtime.h"

#define CUDA_SAFE_CALL(X) (X)

void initializeEvents(hipEvent_t *start, hipEvent_t *stop){
  CUDA_SAFE_CALL( hipEventCreate(start) );
	CUDA_SAFE_CALL( hipEventCreate(stop) );
	CUDA_SAFE_CALL( hipEventRecord(*start, 0) );
}

float finalizeEvents(hipEvent_t start, hipEvent_t stop){
	CUDA_SAFE_CALL( hipGetLastError() );
	CUDA_SAFE_CALL( hipEventRecord(stop, 0) );
	CUDA_SAFE_CALL( hipEventSynchronize(stop) );
	float kernel_time;
	CUDA_SAFE_CALL( hipEventElapsedTime(&kernel_time, start, stop) );
	CUDA_SAFE_CALL( hipEventDestroy(start) );
	CUDA_SAFE_CALL( hipEventDestroy(stop) );
	return kernel_time;
}



template<int WIDTH>
__global__ 
void run_shfl_xor_const_width
                 (hipLaunchParm lp , int* input, int* output, int iter) {
  int id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int data = input[id];

  int i = 0;

#define UNROLL16
#ifdef UNROLL16
#define UNROLL_16   16
  for (; (i+UNROLL_16) < iter; i+=UNROLL_16) {
    data = __shfl_xor(data, data, WIDTH);
    data = __shfl_xor(data, data, WIDTH);
    data = __shfl_xor(data, data, WIDTH);
    data = __shfl_xor(data, data, WIDTH);
    data = __shfl_xor(data, data, WIDTH);
    data = __shfl_xor(data, data, WIDTH);
    data = __shfl_xor(data, data, WIDTH);
    data = __shfl_xor(data, data, WIDTH);
    data = __shfl_xor(data, data, WIDTH);
    data = __shfl_xor(data, data, WIDTH);
    data = __shfl_xor(data, data, WIDTH);
    data = __shfl_xor(data, data, WIDTH);
    data = __shfl_xor(data, data, WIDTH);
    data = __shfl_xor(data, data, WIDTH);
    data = __shfl_xor(data, data, WIDTH);
    data = __shfl_xor(data, data, WIDTH);
  }
#endif

  for(; i < iter; i++) {
    data = __shfl_xor(data, data, WIDTH);
  }

  output[id] = data;
}

template<int WIDTH>
int test_shfl_xor_const_width(const int n, const int blockSize, const int launch_iter=1, const int shfl_iter=1, const bool verify=true) {


  std::vector<int> input(n);
  std::future<void> inputFuture = std::async([&]() {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> input_dist(0, WIDTH-1);
    std::generate(std::begin(input), std::end(input),[&]() { return input_dist(generator); }); 
  });
  inputFuture.wait();

  int* gpuInput;
  hipMalloc(&gpuInput, n * sizeof(int));
  hipMemcpy(gpuInput, input.data(), n * sizeof(int), hipMemcpyHostToDevice);

  int* gpuOutput;
  hipMalloc(&gpuOutput, n * sizeof(int));

  
  // warm up
  {
	  hipEvent_t start, stop;

    initializeEvents(&start, &stop);

    hipLaunchKernel(HIP_KERNEL_NAME(run_shfl_xor_const_width<WIDTH>)
                    , dim3(n/blockSize), dim3(blockSize), 0, 0
                    , gpuInput, gpuOutput, shfl_iter); 

    float time_ms = finalizeEvents(start, stop);
  }


  // measure the performance
  hipEvent_t start, stop;
  initializeEvents(&start, &stop);

  for (int i = 0; i < launch_iter; i++) {
    hipLaunchKernel(HIP_KERNEL_NAME(run_shfl_xor_const_width<WIDTH>)
                    , dim3(n/blockSize), dim3(blockSize), 0, 0
                    , gpuInput, gpuOutput, shfl_iter); 
  }
  float time_ms = finalizeEvents(start, stop);
  std::vector<int> output(n);
  hipMemcpy(output.data(), gpuOutput, n * sizeof(int), hipMemcpyDeviceToHost);
  

#define WAVE_SIZE  64

  // verification
  int errors = 0;
  if (verify) {
    for (int i = 0; i < n; i+=WAVE_SIZE) {

      int local_output[WAVE_SIZE];
      for (int j = 0; j < shfl_iter; j++) {

        for (int k = 0; k < WAVE_SIZE; k++) {
          unsigned int block = i;
          unsigned int current_lane = k;
          unsigned int mask = input[i+k];
          unsigned int target = current_lane ^ mask;
          if (target >= ((current_lane + WIDTH) & ~(WIDTH-1)))
            target = current_lane;
          local_output[k] = input[i+target];
        }

        for (int k = 0; k < WIDTH; k++) {
          input[i+k] = local_output[k];
        }
      }
      for (int k = 0; k < WIDTH; k++) {
        if (input[i+k] != output[i+k]) {
          errors++;
        }
      }
    }
  }

  std::cout << __FUNCTION__ << "<" << WIDTH 
            << "> total(" << launch_iter << " launches, " << shfl_iter << " shfl_xor/lane/kernel): " 
            << time_ms << "ms, "
            << time_ms/(double)launch_iter << " ms/kernel, "
            << errors << " errors"
            << std::endl;

  hipFree(gpuInput);
  hipFree(gpuOutput);

  return errors;
}


template<int WIDTH>
void run_test_shfl_xor_const_width(const int num, const int blockSize, const int launch_iter, const int shfl_iter) {
  test_shfl_xor_const_width<WIDTH>(num, blockSize, launch_iter, shfl_iter);
  test_shfl_xor_const_width<WIDTH>(num, blockSize, launch_iter, shfl_iter);
  test_shfl_xor_const_width<WIDTH>(num, blockSize, launch_iter, shfl_iter);
  test_shfl_xor_const_width<WIDTH>(num, blockSize, launch_iter, shfl_iter);
  test_shfl_xor_const_width<WIDTH>(num, blockSize, launch_iter, shfl_iter);
  test_shfl_xor_const_width<WIDTH>(num, blockSize, launch_iter, shfl_iter);
  test_shfl_xor_const_width<WIDTH>(num, blockSize, launch_iter, shfl_iter);
}


int main() {
#define LAUNCH_ITER 10


  for (int i = 1; i <= 1000000; i *= 10) {
    run_test_shfl_xor_const_width<16>(64,64,LAUNCH_ITER, i);
  }

  for (int i = 1; i <= 1000000; i *= 10) {
    run_test_shfl_xor_const_width<32>(64,64,LAUNCH_ITER, i);
  }

  for (int i = 1; i <= 1000000; i *= 10) {
    run_test_shfl_xor_const_width<64>(64,64,LAUNCH_ITER, i);
  }

  return 0;
}






