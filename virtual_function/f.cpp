#include <cstdlib>
#include "f.h"
#include "hip/hip_runtime.h"
#include <iostream>

constexpr uint32_t num_objects = 2;

__global__ void k(void** p) {
    if (threadIdx.x == 0) {
        new(*p) b(threadIdx.x + 10);
    }
    else if (threadIdx.x == 1) {
        new(*(p + 1)) bd(threadIdx.x + 100);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (uint32_t i = 0; i < num_objects; ++i) {
          auto b_ptr = reinterpret_cast<b*>(*(p + i));
          b_ptr->base_print();
          b_ptr->virtual_print();
      }
    }
}

#define HIP_CHECK_ERROR(x) \
    {hipError_t e = x;\
    if (e != HIP_SUCCESS) {\
        std::cerr << __FILE__ << ":" << __LINE__ << " HIP error " << e << std::endl;\
        std::exit(1);\
    }}\

int main() {
    void** buffers{nullptr};
    HIP_CHECK_ERROR(hipHostMalloc(&buffers, num_objects * sizeof(void*)));
    const size_t obj_size = get_max_object_size() + 1024;
    for(uint32_t i = 0; i < num_objects; ++i) {
        HIP_CHECK_ERROR(hipMalloc(buffers + i, obj_size));
    }
    k<<<1, 2>>>(buffers);

    for(uint32_t i = 0; i < num_objects; ++i) {
        HIP_CHECK_ERROR(hipFree(*(buffers + i)));
    }
    HIP_CHECK_ERROR(hipHostFree(buffers));
    return 0;
}