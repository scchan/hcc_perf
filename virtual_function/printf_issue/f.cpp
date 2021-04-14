#include <cstdlib>
#include "f.h"
#include "hip/hip_runtime.h"
#include "hip_helper.h"
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

      // print the member variables of the derived class directly
      auto bd_ptr = reinterpret_cast<bd*>(*(p + 1));
      printf("Expected from bd::virtual_print(): v=%d, bd_v=%d\n", bd_ptr->v, bd_ptr->bd_v);
    }
}

int main() {
    void** buffers{nullptr};
    HIP_CHECK_ERROR(hipHostMalloc(&buffers, num_objects * sizeof(void*)));
    constexpr size_t obj_size = get_max_object_size<b, bd>();
    for(uint32_t i = 0; i < num_objects; ++i) {
        HIP_CHECK_ERROR(hipMalloc(buffers + i, obj_size));
    }
    k<<<1, num_objects>>>(buffers);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    for(uint32_t i = 0; i < num_objects; ++i) {
        HIP_CHECK_ERROR(hipFree(*(buffers + i)));
    }
    HIP_CHECK_ERROR(hipHostFree(buffers));
    return 0;
}