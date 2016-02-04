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
void run_shfl_const_width
                 (hipLaunchParm lp , int* input, int* output, int iter) {
  int id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int data = input[id];
  for(int i = 0; i < iter; i++) {
    data = __shfl(data, data, WIDTH);
  }
  output[id] = data;
}

template<int WIDTH>
int test_shfl_const_width(const int n, const int blockSize, const int launch_iter=1, const int shfl_iter=1, const bool verify=true) {


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

    hipLaunchKernel(HIP_KERNEL_NAME(run_shfl_const_width<WIDTH>)
                    , dim3(n/blockSize), dim3(blockSize), 0, 0
                    , gpuInput, gpuOutput, shfl_iter); 

    float time_ms = finalizeEvents(start, stop);
  }


  // measure the performance
  hipEvent_t start, stop;
  initializeEvents(&start, &stop);

  for (int i = 0; i < launch_iter; i++) {
    hipLaunchKernel(HIP_KERNEL_NAME(run_shfl_const_width<WIDTH>)
                    , dim3(n/blockSize), dim3(blockSize), 0, 0
                    , gpuInput, gpuOutput, shfl_iter); 
  }
  float time_ms = finalizeEvents(start, stop);
  std::vector<int> output(n);
  hipMemcpy(output.data(), gpuOutput, n * sizeof(int), hipMemcpyDeviceToHost);
  

  // verification
  int errors = 0;
  if (verify) {
    for (int i = 0; i < n; i+=WIDTH) {
      int local_output[WIDTH];
      for (int j = 0; j < shfl_iter; j++) {
        for (int k = 0; k < WIDTH; k++) {
          local_output[k] = input[i+input[i+k]];
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
            << "> total(" << launch_iter << " launches, " << shfl_iter << " shfl/lane/kernel): " 
            << time_ms << "ms, "
            << time_ms/(double)launch_iter << " ms/kernel, "
            << errors << " errors"
            << std::endl;

  hipFree(gpuInput);
  hipFree(gpuOutput);

  return errors;
}

void run_test_shfl_const_width(const int num, const int blockSize, const int launch_iter, const int shfl_iter) {
  test_shfl_const_width<2>(num, blockSize, launch_iter, shfl_iter);
  test_shfl_const_width<4>(num, blockSize, launch_iter, shfl_iter);
  test_shfl_const_width<8>(num, blockSize, launch_iter, shfl_iter);
  test_shfl_const_width<16>(num, blockSize, launch_iter, shfl_iter);
  test_shfl_const_width<32>(num, blockSize, launch_iter, shfl_iter);
  test_shfl_const_width<64>(num, blockSize, launch_iter, shfl_iter);
}








#ifdef __HCC__

__global__ 
void run_activelaneperm_random
                 (hipLaunchParm lp , int* input, int* output, int iter) {
  int id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int data = input[id];
  for(int i = 0; i < iter; i++) {
    data = hc::__hsail_activelanepermute_b32(data, data, 0, 0);
  }
  output[id] = data;
}

int test_activelaneperm_random(const int n, const int blockSize, const int launch_iter=1, const int shfl_iter=1, const bool verify=true) {


  std::vector<int> input(n);
  std::future<void> inputFuture = std::async([&]() {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> input_dist(0, 64-1);
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

    hipLaunchKernel(HIP_KERNEL_NAME(run_activelaneperm_random)
                    , dim3(n/blockSize), dim3(blockSize), 0, 0
                    , gpuInput, gpuOutput, shfl_iter); 

    float time_ms = finalizeEvents(start, stop);
  }


  // measure the performance
  hipEvent_t start, stop;
  initializeEvents(&start, &stop);

  for (int i = 0; i < launch_iter; i++) {
    hipLaunchKernel(HIP_KERNEL_NAME(run_activelaneperm_random)
                    , dim3(n/blockSize), dim3(blockSize), 0, 0
                    , gpuInput, gpuOutput, shfl_iter); 
  }
  float time_ms = finalizeEvents(start, stop);
  std::vector<int> output(n);
  hipMemcpy(output.data(), gpuOutput, n * sizeof(int), hipMemcpyDeviceToHost);
  

  // verification
  int errors = 0;
  if (verify) {
    for (int i = 0; i < n; i+=64) {
      int local_output[64];
      for (int j = 0; j < shfl_iter; j++) {
        for (int k = 0; k < 64; k++) {
          local_output[k] = input[i+input[i+k]];
        }
        for (int k = 0; k < 64; k++) {
          input[i+k] = local_output[k];
        }
      }
      for (int k = 0; k < 64; k++) {
        if (input[i+k] != output[i+k]) {
          errors++;
        }
      }
    }
  }

  std::cout << __FUNCTION__ 
            << "total(" << launch_iter << " launches, " << shfl_iter << " activelanepermute/lane/kernel): " 
            << time_ms << "ms, "
            << time_ms/(double)launch_iter << " ms/kernel, "
            << errors << " errors"
            << std::endl;

  hipFree(gpuInput);
  hipFree(gpuOutput);

  return errors;
}

void run_test_activelaneperm_random(const int num, const int blockSize, const int launch_iter, const int shfl_iter) {
  test_activelaneperm_random(num, blockSize, launch_iter, shfl_iter);
}

#endif // #ifdef __HCC__


int main() {
#define LAUNCH_ITER 10

  for (int i = 1; i <= 1000000; i *= 10) {
    run_test_shfl_const_width(64,64,LAUNCH_ITER, i);
  }

#ifdef __HCC__
  for (int i = 1; i <= 1000000; i *= 10) {
    run_test_activelaneperm_random(64,64,LAUNCH_ITER, i);
  }
#endif 

  return 0;
}






