#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <typeinfo>
#include <utility>
#include <vector>

#include "hip/hip_runtime.h"

template<typename T, size_t num>
struct kernargs {
    T data[num];
};

template<typename T, size_t num>
__global__ void accumulate(T* a, kernargs<T, num> kargs) {
    *a = static_cast<T>(0);
    for (uint32_t i = 0; i < num; ++i) {
        *a+=kargs.data[i];
    }
}

template<typename T, size_t num>
class kernargs_test {
public:
    using kernargs_s = kernargs<T, num>;
    kernargs_test(hipStream_t stream, T init_value) 
        : stream(stream), init_value(init_value) {
        //std::cout << "sizeof " << typeid(kernargs_s).name() << ": " << sizeof(kernargs_s) << std::endl;
        start();
    }
    bool check() {
        std::call_once(result_flag, [this]() {
            hipMemcpyAsync(&actual_r, gpu_r_ptr, sizeof(T), hipMemcpyDeviceToHost, stream);
            hipStreamSynchronize(stream);
            hipFree(gpu_r_ptr);
        });
        return (expected_r == actual_r);
    }
    ~kernargs_test() { check(); }
private:

    void start() {
        kernargs_s kernargs;
        expected_r = actual_r = static_cast<T>(0);
        T v = init_value;
        for (uint32_t i = 0; i < num; ++i) {
            expected_r+=v;
            kernargs.data[i] = v;
            v = v + static_cast<T>(1);
        }
        hipMalloc(&gpu_r_ptr, sizeof(T));
        hipLaunchKernelGGL(accumulate<T, num>, dim3(1), dim3(1), 0, stream, gpu_r_ptr, kernargs);
    }

    hipStream_t stream;
    T init_value;
    T expected_r;
    T actual_r;
    T* gpu_r_ptr = nullptr;
    std::once_flag result_flag;
};

template<typename T, size_t kernarg_buffer_size>
bool test_kernarg_buffer_recycling() {
    // This test the recycling of kernarg buffers in the runtime
    constexpr size_t pool_size_bytes = (2 * 1024 * 1024);
    constexpr uint32_t num_of_tests_per_batch = static_cast<uint32_t>(pool_size_bytes / kernarg_buffer_size * 0.2);
    constexpr size_t num_elements_in_kernargs = kernarg_buffer_size / sizeof(T);

    using k_test = kernargs_test<T, num_elements_in_kernargs>;
   
    std::vector<std::unique_ptr<k_test>> tests;
    tests.reserve(num_of_tests_per_batch);

    hipStream_t stream;
    hipStreamCreate(&stream);

    constexpr uint32_t num_batches = 16;
    bool all_passed = true;
    for (uint32_t i = 0; i < num_batches; ++i) {
        for (uint32_t j = 0; j < num_of_tests_per_batch; ++j) {
            tests.push_back(std::make_unique<k_test>(stream, i*j));
        }
        for (auto& t : tests) {
            all_passed = all_passed && t->check();
        }
        tests.clear();
    }

    hipStreamDestroy(stream);
    return all_passed;
}

int main() {

  constexpr size_t output_ptr_size = 8;
  constexpr size_t implicit_kernarg_size = 56;
  constexpr size_t other_args = output_ptr_size + implicit_kernarg_size;

  int device_count = 0;
  hipGetDeviceCount(&device_count);
  std::vector<std::pair<std::thread,bool>> t;
  for (int i = 0; i < device_count; ++i) {
      t.push_back(std::make_pair(std::thread([i, &t]() {
          hipSetDevice(i);
          bool error = false;
          error = error || !test_kernarg_buffer_recycling<uint32_t, 512 - other_args>();
          error = error || !test_kernarg_buffer_recycling<uint32_t, 1024 - other_args>();
          error = error || !test_kernarg_buffer_recycling<uint32_t, 2048 - other_args>();
          error = error || !test_kernarg_buffer_recycling<uint32_t, 4096 - other_args>();
          t[i].second = error;
      }), true));
  }

  bool error = false;
  for (auto& i : t) {
      i.first.join();
      error = error || i.second;
  }

  if (error) {
      std::cout << "Error found." << std::endl;
  }
  return error;
}