#include <vector>
#include <random>
#include <future>
#include <iostream>
#include <algorithm>
#include <typeinfo>
#include "hip_runtime.h"


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


  {
	  hipEvent_t start, stop;

    initializeEvents(&start, &stop);

    hipLaunchKernel(HIP_KERNEL_NAME(run_shfl_const_width<WIDTH>)
                    , dim3(n/blockSize), dim3(blockSize), 0, 0
                    , gpuInput, gpuOutput, shfl_iter); 

    float time_ms = finalizeEvents(start, stop);
    std::cout << __FUNCTION__ << "<"  << WIDTH << "> warm up: " << time_ms << "ms" << std::endl;
  }



  {
	  hipEvent_t start, stop;

    initializeEvents(&start, &stop);

    for (int i = 0; i < launch_iter; i++) {
      hipLaunchKernel(HIP_KERNEL_NAME(run_shfl_const_width<WIDTH>)
                      , dim3(n/blockSize), dim3(blockSize), 0, 0
                      , gpuInput, gpuOutput, shfl_iter); 
    }
    float time_ms = finalizeEvents(start, stop);
    std::cout << __FUNCTION__ << "<" << WIDTH 
              << "> total(" << launch_iter << " launches, " << shfl_iter << " shfl/lane): " 
              << time_ms << "ms, "
              << time_ms/(double)launch_iter << "ms/iteration"
              << std::endl;
  }

  std::vector<int> output(n);
  hipMemcpy(output.data(), gpuOutput, n * sizeof(int), hipMemcpyDeviceToHost);
  
  int errors = 0;
  if (verify) {
    for (int j = 0; j < shfl_iter; j++) {
      for (int i = 0; i < n; i+=WIDTH) {
        int local_output[WIDTH];
        for (int k = 0; k < WIDTH; k++) {
          local_output[k] = input[i + input[i+k]];
        }
        for (int k = 0; k < WIDTH; k++) {
          input[i+k] = local_output[k];
        }
      }
    }

    for (int i = 0; i < n; i++) {
      if (input[i] != output[i]) {
        errors++;
      }
    }


#if 0  
    int blockOrigin = 0;
    int logicalCounter = 0;
    for (int i = 0; i < n; i++) {
      int expected = input[blockOrigin + input[i]];
      if (expected != output[i]) {
        errors++;
      }
      logicalCounter++;
      if (logicalCounter>=WIDTH) {
        logicalCounter = 0;
        blockOrigin+=WIDTH;
      }
    }
#endif


  }

  hipFree(gpuInput);
  hipFree(gpuOutput);

  return errors;
}

void run_test_shfl_const_width(const int num, const int blockSize, const int launch_iter, const int shfl_iter) {

  {int errors =  test_shfl_const_width<2>(num, blockSize, launch_iter, shfl_iter);
   std::cout << "test_shfl_const_width: width=2, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}

  {int errors =  test_shfl_const_width<4>(num, blockSize, launch_iter, shfl_iter);
   std::cout << "test_shfl_const_width: width=4, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}

  {int errors =  test_shfl_const_width<8>(num, blockSize, launch_iter, shfl_iter);
   std::cout << "test_shfl_const_width: width=8, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}

  {int errors =  test_shfl_const_width<16>(num, blockSize, launch_iter, shfl_iter);
   std::cout << "test_shfl_const_width: width=16, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}

  {int errors =  test_shfl_const_width<32>(num, blockSize, launch_iter, shfl_iter);
   std::cout << "test_shfl_const_width: width=32, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}

  {int errors =  test_shfl_const_width<64>(num, blockSize, launch_iter, shfl_iter);
   std::cout << "test_shfl_const_width: width=64, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}
}

int main() {

#define LAUNCH_ITER 10


  for (int i = 1; i <= 1000000; i *= 10) {
    run_test_shfl_const_width(64,64,LAUNCH_ITER, i);
  }


#if 0
  run_test_shfl_const_width<int>(128,64,ITER);
  run_test_shfl_const_width<int>(128,128,ITER);

  run_test_shfl_const_width<int>(1024*1024,64,ITER);
  run_test_shfl_const_width<int>(1024*1024,128,ITER);
  run_test_shfl_const_width<int>(1024*1024,256,ITER);

  run_test_shfl_const_width<int>(1024*1024*50,64,ITER);
  run_test_shfl_const_width<int>(1024*1024*50,128,ITER);
  run_test_shfl_const_width<int>(1024*1024*50,256,ITER);


  run_test_shfl_const_width<float>(64,64,ITER);

  run_test_shfl_const_width<float>(128,64,ITER);
  run_test_shfl_const_width<float>(128,128,ITER);

  run_test_shfl_const_width<float>(1024*1024,64,ITER);
  run_test_shfl_const_width<float>(1024*1024,128,ITER);
  run_test_shfl_const_width<float>(1024*1024,256,ITER);

  run_test_shfl_const_width<float>(1024*1024*50,64,ITER);
  run_test_shfl_const_width<float>(1024*1024*50,128,ITER);
  run_test_shfl_const_width<float>(1024*1024*50,256,ITER);
#endif

  return 0;
}






