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

extern "C" int amdgcn_wave_rshift_1(int) __HC__;

__global__ 
void run_amdgcn_wave_rshift_1
                 (hipLaunchParm lp , int* input, int* output, int iter) {
  int id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int data = input[id];

#if 0
  #pragma unroll 32
  for(int i = 0; i < iter; i++) {
    data = amdgcn_wave_rshift_1(data);
  }


#else

  int i = 0;


#define UNROLL16
#ifdef UNROLL16
#define UNROLL_16   16
  for (; (i+UNROLL_16) < iter; i+=UNROLL_16) {
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
  }
#endif


#ifdef UNROLL8
#define UNROLL_8   8
  for (; (i+UNROLL_8) < iter; i+=UNROLL_8) {
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
  }
#endif


//#define UNROLL2 1
#ifdef UNROLL2
#define UNROLL_2   2
  for (; (i+UNROLL_2) < iter; i+=UNROLL_2) {
    data = amdgcn_wave_rshift_1(data);
    data = amdgcn_wave_rshift_1(data);
  }
#endif


  for(; i < iter; i++) {
    data = amdgcn_wave_rshift_1(data);
  }

#endif

  output[id] = data;
}

int test_amdgcn_wave_rshift_1
              (const int n, const int blockSize, const int launch_iter=1, const int shfl_iter=1, const bool verify=true) {

  const int WIDTH = 64;
  const int DELTA = 1;

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

    hipLaunchKernel(HIP_KERNEL_NAME(run_amdgcn_wave_rshift_1)
                    , dim3(n/blockSize), dim3(blockSize), 0, 0
                    , gpuInput, gpuOutput, shfl_iter); 

    float time_ms = finalizeEvents(start, stop);
  }


  // measure the performance
  hipEvent_t start, stop;
  initializeEvents(&start, &stop);

  for (int i = 0; i < launch_iter; i++) {
    hipLaunchKernel(HIP_KERNEL_NAME(run_amdgcn_wave_rshift_1)
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
          unsigned int lane = ((k-(int)DELTA)<0)?k:(k-DELTA);
          local_output[k] = input[i+lane];
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

  std::cout << __FUNCTION__ << "<" << DELTA << "," << WIDTH 
            << "> total(" << launch_iter << " launches, " << shfl_iter << " wavefront_shift_right/lane/kernel): " 
            << time_ms << "ms, "
            << time_ms/(double)launch_iter << " ms/kernel, "
            << errors << " errors"
            << std::endl;

  hipFree(gpuInput);
  hipFree(gpuOutput);

  return errors;
}




extern "C" int amdgcn_wave_lshift_1(int) __HC__;

__global__ 
void run_amdgcn_wave_lshift_1
                 (hipLaunchParm lp , int* input, int* output, int iter) {
  int id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int data = input[id];

#if 0
  #pragma unroll 32
  for(int i = 0; i < iter; i++) {
    data = amdgcn_wave_lshift_1(data);
  }


#else

  int i = 0;


#define UNROLL16
#ifdef UNROLL16
#define UNROLL_16   16
  for (; (i+UNROLL_16) < iter; i+=UNROLL_16) {
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
  }
#endif


#ifdef UNROLL8
#define UNROLL_8   8
  for (; (i+UNROLL_8) < iter; i+=UNROLL_8) {
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
  }
#endif


//#define UNROLL2 1
#ifdef UNROLL2
#define UNROLL_2   2
  for (; (i+UNROLL_2) < iter; i+=UNROLL_2) {
    data = amdgcn_wave_lshift_1(data);
    data = amdgcn_wave_lshift_1(data);
  }
#endif


  for(; i < iter; i++) {
    data = amdgcn_wave_lshift_1(data);
  }

#endif

  output[id] = data;
}

int test_amdgcn_wave_lshift_1
              (const int n, const int blockSize, const int launch_iter=1, const int shfl_iter=1, const bool verify=true) {

  const int WIDTH = 64;
  const int DELTA = 1;

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

    hipLaunchKernel(HIP_KERNEL_NAME(run_amdgcn_wave_lshift_1)
                    , dim3(n/blockSize), dim3(blockSize), 0, 0
                    , gpuInput, gpuOutput, shfl_iter); 

    float time_ms = finalizeEvents(start, stop);
  }


  // measure the performance
  hipEvent_t start, stop;
  initializeEvents(&start, &stop);

  for (int i = 0; i < launch_iter; i++) {
    hipLaunchKernel(HIP_KERNEL_NAME(run_amdgcn_wave_lshift_1)
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
          unsigned int lane = ((k+(int)DELTA)<WIDTH)?(k+DELTA):k;
          local_output[k] = input[i+lane];
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

  std::cout << __FUNCTION__ << "<" << DELTA << "," << WIDTH 
            << "> total(" << launch_iter << " launches, " << shfl_iter << " wavefront_shift_left/lane/kernel): " 
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
    test_amdgcn_wave_rshift_1(64,64,LAUNCH_ITER, i);
  }


  for (int i = 1; i <= 1000000; i *= 10) {
    test_amdgcn_wave_lshift_1(64,64,LAUNCH_ITER, i);
  }

  return 0;
}






