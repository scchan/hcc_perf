#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "hip/hip_runtime.h"

#define HIP_CHECK_ERROR(x) \
    {hipError_t e = x;\
    if (e != HIP_SUCCESS) {\
        std::cerr << __FILE__ << ":" << __LINE__ << " HIP error " << e << std::endl;\
        std::exit(1);\
    }}


constexpr int flag_ready = 234;

__launch_bounds__(1024)
__global__ void kernel_A(int* flag, int* work, const int iter) {

    if (blockIdx.x == 0
        && threadIdx.x == 0) {
        while (atomicCAS(flag, flag_ready, flag_ready + 1) != flag_ready);
    }

    for (auto i = 0; i < iter; ++i) {
        for (auto x = 0; x < gridDim.x; ++x) {
            atomicAdd(work + x, threadIdx.x);
        }
    }
}

__global__ void kernel_B(int* flag) {
    if (blockIdx.x == 0
        && threadIdx.x == 0) {
        atomicAdd(flag, flag_ready);
    }
}

__global__ void dummy() {}



int main() {

    int* flag;
    HIP_CHECK_ERROR(hipMalloc(&flag, sizeof(int)));
    int host_flag = 0;
    HIP_CHECK_ERROR(hipMemcpy(flag, &host_flag, sizeof(int), hipMemcpyHostToDevice));

    std::vector<hipStream_t> vs(2);
    for (auto& s : vs) {
        HIP_CHECK_ERROR(hipStreamCreate(&s));
    }

    constexpr int block_size = 256;
    constexpr int num_blocks = 1024;
    constexpr int iter = 64;
    int* work;
    HIP_CHECK_ERROR(hipMalloc(&work, num_blocks * sizeof(int)));
    kernel_A<<<num_blocks, block_size, 0, vs[0]>>>(flag, work, iter);

    std::this_thread::sleep_for(std::chrono::seconds(5));
    kernel_B<<<1, 1, 0, vs[1]>>>(flag);

#define NULL_STREAM_IMPLICIT_SYNC 1
    auto sync = [&] {
#ifdef NULL_STREAM_IMPLICIT_SYNC
        // launch a dummy kernel with the null stream to do an implicit sync with all the streams
        dummy<<<1,1,0,0>>>();
#else
        HIP_CHECK_ERROR(hipDeviceSynchronize());
#endif
    };

    sync();

    hipMemcpyAsync(&host_flag, flag, sizeof(int), hipMemcpyDeviceToHost, vs[0]);
 
    sync();

    std::cout << "flag: GPU=" << host_flag << " expected=" << flag_ready+1 << std::endl;

    for (auto& s : vs) {
        HIP_CHECK_ERROR(hipStreamDestroy(s));
    }
    HIP_CHECK_ERROR(hipFree(flag));
    return 0;
}