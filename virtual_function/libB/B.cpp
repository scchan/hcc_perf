#include "B.h"
#include "f.h"
#include "hip/hip_runtime.h"
#include "hip_helper.h"

__global__ void B_construct_object(b** p, const uint32_t n) {
    const uint32_t idx = blockIdx.x * 64 + threadIdx.x;
    if (idx < n) {
        if (idx % 2 == 0) {
            new(*(p+idx)) b(threadIdx.x + 1000000);
        }
        else {
            new(*(p+idx)) bd(threadIdx.x + 2000000);
        }
    }
}


void run_B_construct_object(b** p, const uint32_t n) {
    uint32_t num_blocks = (n + 64 - 1)/64;
    B_construct_object<<<num_blocks, 64>>>(p, n);
}