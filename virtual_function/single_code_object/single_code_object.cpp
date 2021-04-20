#include <cstdlib>
#include "f.h"
#include "hip/hip_runtime.h"
#include "hip_helper.h"
#include <iostream>

constexpr uint32_t num_objects = 2;

__global__ void k(void** p) {

    if (threadIdx.x == 0) {
        new(*p) b(threadIdx.x + 100);
    }
    else if (threadIdx.x == 1) {
        new(*(p + 1)) bd(threadIdx.x + 200);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t expected_base[] = {100, 201};
        uint32_t expected_virtual[] = {100, BD_MAGIC + 1 + 200};

        for(auto i = 0; i < num_objects; ++i) {
            auto ptr = reinterpret_cast<const b*>(*(p + i));
            printf("object %u: get_base()=%u, expected=%u  get_virtual()=%u, expected=%u\n", i
              , ptr->get_base(), expected_base[i]
              , ptr->get_virtual(), expected_virtual[i]);
        }
        
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