#include <cstdio>
#include "hip/hip_runtime.h"

class b {
public:
    __host__ __device__
    b(uint32_t i) : v(i) {
    }
    uint32_t v;
};

class bd: public b {
public:
    __host__ __device__
    bd(uint32_t i) : b(i), bd_v(i + 1000) {
        printf("%s, %u, %u\n", 
        __PRETTY_FUNCTION__, v, bd_v);
    }
    uint32_t bd_v;
};

extern "C"
__global__ void k(void* buffer) {
    new(buffer) bd(1000);
}

int main() {

#if 0
    printf("CPU:\n");
    bd cpu(1000);
#endif

    void* buffer{nullptr};
    hipMalloc(&buffer, sizeof(bd));
    printf("GPU:\n");
    k<<<1,1>>>(buffer);
    hipFree(buffer);
    return 0;
}

