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

extern "C" int amdgcn_ds_bpermute(int index, int src) __HC__;

__global__ 
void run_ds_bperm
                 (hipLaunchParm lp , int* input, int* output, int iter) {
  int id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int data = input[id];

  int i = 0;

#define UNROLL16
#ifdef UNROLL16
#define UNROLL_16   16
  for (; (i+UNROLL_16) < iter; i+=UNROLL_16) {
    data = amdgcn_ds_bpermute(data*4,data);
    data = amdgcn_ds_bpermute(data*4,data);
    data = amdgcn_ds_bpermute(data*4,data);
    data = amdgcn_ds_bpermute(data*4,data);
    data = amdgcn_ds_bpermute(data*4,data);
    data = amdgcn_ds_bpermute(data*4,data);
    data = amdgcn_ds_bpermute(data*4,data);
    data = amdgcn_ds_bpermute(data*4,data);
    data = amdgcn_ds_bpermute(data*4,data);
    data = amdgcn_ds_bpermute(data*4,data);
    data = amdgcn_ds_bpermute(data*4,data);
    data = amdgcn_ds_bpermute(data*4,data);
    data = amdgcn_ds_bpermute(data*4,data);
    data = amdgcn_ds_bpermute(data*4,data);
    data = amdgcn_ds_bpermute(data*4,data);
    data = amdgcn_ds_bpermute(data*4,data);
  }
#endif

  for(; i < iter; i++) {
    data = amdgcn_ds_bpermute(data*4,data);
  }


  output[id] = data;
}

int test_ds_bperm(const int n, const int blockSize, const int launch_iter=1, const int shfl_iter=1, const bool verify=true) {

#define WIDTH 64  

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

    hipLaunchKernel(HIP_KERNEL_NAME(run_ds_bperm)
                    , dim3(n/blockSize), dim3(blockSize), 0, 0
                    , gpuInput, gpuOutput, shfl_iter); 

    float time_ms = finalizeEvents(start, stop);
  }


  // measure the performance
  hipEvent_t start, stop;
  initializeEvents(&start, &stop);

  for (int i = 0; i < launch_iter; i++) {
    hipLaunchKernel(HIP_KERNEL_NAME(run_ds_bperm)
                    , dim3(n/blockSize), dim3(blockSize), 0, 0
                    , gpuInput, gpuOutput, shfl_iter); 
  }
  float time_ms = finalizeEvents(start, stop);
  std::vector<int> output(n);
  hipMemcpy(output.data(), gpuOutput, n * sizeof(int), hipMemcpyDeviceToHost);
  

  // verification
  int errors = 0;
  if (verify) {

//#define VERBOSE 1

#ifdef VERBOSE
    for (int i = 0; i < n; i++) {
      std::cout << "input[" << i << "]=" << input[i] << std::endl;
    }
#endif

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

#ifdef VERBOSE
          std::cout << "output[" << (i+k) << "]: expected=" <<  input[i+k] << " actual=" << output[i+k] << std::endl;
#endif

          errors++;
        }
      }
    }
  }

  std::cout << __FUNCTION__ << "<" << WIDTH 
            << "> total(" << launch_iter << " launches, " << shfl_iter << " ds_bperm/lane/kernel): " 
            << time_ms << "ms, "
            << time_ms/(double)launch_iter << " ms/kernel, "
            << errors << " errors"
            << std::endl;

  hipFree(gpuInput);
  hipFree(gpuOutput);

  return errors;
}

int main() {
#define LAUNCH_ITER 10

  for (int i = 1; i <= 1000000; i *= 10) {
    test_ds_bperm(64,64,LAUNCH_ITER, i);
  }
  

  return 0;
}






