#include <vector>
#include <random>
#include <future>
#include <iostream>
#include "hip_runtime.h"


template<typename T, int WIDTH>
__global__ 
void run_shfl_const_width
                 (hipLaunchParm lp , T* input, int* srcLane, T* output) {

  int id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  T src = input[id];
  int lane = srcLane[id];

  T out = hc::__shfl(src, lane, WIDTH);

  output[id] = out;
}


template<typename T, int WIDTH>
int test_shfl_const_width(const int n, const int blockSize) {


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

  hipLaunchKernel(HIP_KERNEL_NAME(run_shfl_const_width<T, WIDTH>)
                  , dim3(n/blockSize), dim3(blockSize), 0, 0
                  , gpuInput, gpuSrcLane, gpuOutput); 


  std::vector<T> output(n);
  hipMemcpy(output.data(), gpuOutput, n * sizeof(T), hipMemcpyDeviceToHost);


  int errors = 0;
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

  hipFree(gpuInput);
  hipFree(gpuSrcLane);
  hipFree(gpuOutput);

  return errors;
}


template<typename T>
void run_test_shfl_const_width(const int num, const int blockSize) {

  {int errors =  test_shfl_const_width<T, 2>(num, blockSize);
   std::cout << "test_shfl_const_width: width=2, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}

  {int errors =  test_shfl_const_width<T, 4>(num, blockSize);
   std::cout << "test_shfl_const_width: width=4, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}

  {int errors =  test_shfl_const_width<T, 8>(num, blockSize);
   std::cout << "test_shfl_const_width: width=8, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}

  {int errors =  test_shfl_const_width<T, 16>(num, blockSize);
   std::cout << "test_shfl_const_width: width=16, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}

  {int errors =  test_shfl_const_width<T, 32>(num, blockSize);
   std::cout << "test_shfl_const_width: width=32, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}

  {int errors =  test_shfl_const_width<T, 64>(num, blockSize);
   std::cout << "test_shfl_const_width: width=64, num=" << num << ", blockSize=" << blockSize << ": " << errors << " errors" << std::endl;}
}

int main() {

  int errors;
  int num;
  int blockSize;

  run_test_shfl_const_width<int>(64,64);

  run_test_shfl_const_width<int>(128,64);
  run_test_shfl_const_width<int>(128,128);

  run_test_shfl_const_width<int>(1024*1024,64);
  run_test_shfl_const_width<int>(1024*1024,128);
  run_test_shfl_const_width<int>(1024*1024,256);

  run_test_shfl_const_width<int>(1024*1024*50,64);
  run_test_shfl_const_width<int>(1024*1024*50,128);
  run_test_shfl_const_width<int>(1024*1024*50,256);


  run_test_shfl_const_width<float>(64,64);

  run_test_shfl_const_width<float>(128,64);
  run_test_shfl_const_width<float>(128,128);

  run_test_shfl_const_width<float>(1024*1024,64);
  run_test_shfl_const_width<float>(1024*1024,128);
  run_test_shfl_const_width<float>(1024*1024,256);

  run_test_shfl_const_width<float>(1024*1024*50,64);
  run_test_shfl_const_width<float>(1024*1024*50,128);
  run_test_shfl_const_width<float>(1024*1024*50,256);

  return 0;
}






