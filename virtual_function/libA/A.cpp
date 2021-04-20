#include "f.h"
#include "hip/hip_runtime.h"
#include "hip_helper.h"
#include "A.h"

__global__ void A_invoke_virtual(b** p, const uint32_t n) {
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < n; ++i) {
            auto base = p[i]->get_base();
            auto virt = p[i]->get_virtual();
            printf("%s  object %d:\t get_base()=%u\t get_virtual()=%u\n",
                __PRETTY_FUNCTION__, i, base, virt);
        }
    }
}

void run_A_invoke_virtual(b** p, const uint32_t n) {
    A_invoke_virtual<<<1, 1>>>(p, n);
}