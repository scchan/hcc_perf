#include <vector>
#include <random>
#include <future>
#include <iostream>
#include <algorithm>
#include <typeinfo>
#include "hip_runtime.h"


template<typename T, int WIDTH>
__global__ 
void run_shfl_const_width
                 (hipLaunchParm lp , T* input, int* srcLane, T* output) {

  int id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  T src = input[id];
  int lane = srcLane[id];

  T out = __shfl(src, lane, WIDTH);

  output[id] = out;
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


template<typename T, int WIDTH>
int test_shfl_const_width(const int n, const int blockSize, const int iter=1, const bool verify=true) {


  std::vector<int> srcLane(n);
  std::future<void> srcLaneFuture = std::async([&]() {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> lane_dist(0, WIDTH-1);
    std::generate(std::begin(srcLane), std::end(srcLane),[&]() { return lane_dist(generator); }); 
  });

  std::vector<T> input(n);
  std::future<void> inputFuture = std::async([&]() {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> input_dist(0, 1024);
    std::generate(std::begin(input), std::end(input),[&]() { return (T)input_dist(generator); }); 
  });

  srcLaneFuture.wait();
  inputFuture.wait();

  T* gpuInput;
  hipMalloc(&gpuInput, n * sizeof(T));
  hipMemcpy(gpuInput, input.data(), n * sizeof(T), hipMemcpyHostToDevice);

  int* gpuSrcLane;
  hipMalloc(&gpuSrcLane, n * sizeof(int));
  hipMemcpy(gpuSrcLane, srcLane.data(), n * sizeof(int), hipMemcpyHostToDevice);

  T* gpuOutput;
  hipMalloc(&gpuOutput, n * sizeof(T));


  {
	  hipEvent_t start, stop;

    initializeEvents(&start, &stop);

    hipLaunchKernel(HIP_KERNEL_NAME(run_shfl_const_width<T, WIDTH>)
                    , dim3(n/blockSize), dim3(blockSize), 0, 0
                    , gpuInput, gpuSrcLane, gpuOutput); 

    float time_ms = finalizeEvents(start, stop);
    std::cout << __FUNCTION__ << "<" << typeid(T).name() << ", " << WIDTH << "> warm up: " << time_ms << "ms" << std::endl;
  }



  {
	  hipEvent_t start, stop;

    initializeEvents(&start, &stop);

    for (int i = 0; i < iter; i++) {
      hipLaunchKernel(HIP_KERNEL_NAME(run_shfl_const_width<T, WIDTH>)
                      , dim3(n/blockSize), dim3(blockSize), 0, 0
                      , gpuInput, gpuSrcLane, gpuOutput); 
    }
    float time_ms = finalizeEvents(start, stop);
    std::cout << __FUNCTION__ << "<" << typeid(T).name() << ", " << WIDTH 
              << "> total(" << iter << " iterations): " 
              << time_ms << "ms, "
              << time_ms/(double)iter << "ms/iteration"
              << std::endl;
  }

  std::vector<T> output(n);
  hipMemcpy(output.data(), gpuOutput, n * sizeof(T), hipMemcpyDeviceToHost);
  
  int errors = 0;
  if (verify) {
    int blockOrigin = 0;
    int logicalCounter = 0;
    for (int i = 0; i < n; i++) {
      T expected = input[blockOrigin + srcLane[i]];
      if (expected != output[i]) {
        errors++;
      }
      logicalCounter++;
      if (logicalCounter>=WIDTH) {
        logicalCounter = 0;
        blockOrigin+=WIDTH;
      }
    }
  }

  hipFree(gpuInput);
  hipFree(gpuSrcLane);
  hipFree(gpuOutput);

  return errors;
}


template<typename T>
void run_test_shfl_const_width(const int num, const int blockSize, const int iter) {

  {int errors =  test_shfl_const_width<T, 2>(num, blockSize, iter);
   std::cout << "test_shfl_const_width: width=2, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}

  {int errors =  test_shfl_const_width<T, 4>(num, blockSize, iter);
   std::cout << "test_shfl_const_width: width=4, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}

  {int errors =  test_shfl_const_width<T, 8>(num, blockSize, iter);
   std::cout << "test_shfl_const_width: width=8, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}

  {int errors =  test_shfl_const_width<T, 16>(num, blockSize, iter);
   std::cout << "test_shfl_const_width: width=16, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}

  {int errors =  test_shfl_const_width<T, 32>(num, blockSize, iter);
   std::cout << "test_shfl_const_width: width=32, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}

  {int errors =  test_shfl_const_width<T, 64>(num, blockSize, iter);
   std::cout << "test_shfl_const_width: width=64, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}
}

int main() {

#define ITER 10

  run_test_shfl_const_width<int>(64,64,ITER);

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

  return 0;
}






