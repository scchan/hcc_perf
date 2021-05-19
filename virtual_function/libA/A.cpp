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

__global__ void A_print_vaddr(b** p, const uint32_t n, void** fptrb) {
    if (threadIdx.x == 0 && n > 0) {
        union {
            b::fp f;
            void* cp;
        } u;
        u.f = p[threadIdx.x]->get_virtual_print_addr();
        printf("A_print_vaddr: %p\n", u.f);
        fptrb[0] = u.cp;
    }
}

void run_A_invoke_virtual(b** p, const uint32_t n) {
    A_invoke_virtual<<<1, 1>>>(p, n);
}

void run_A_print_vaddr(b** p, const uint32_t n) {
    void** f{nullptr};
    HIP_CHECK_ERROR(hipHostMalloc(&f, sizeof(void*)));
    A_print_vaddr<<<1, 1>>>(p, n, f);
    printf("%s: virtual function addr: %p\n", __PRETTY_FUNCTION__, f[0]);
    HIP_CHECK_ERROR(hipHostFree(f));
}