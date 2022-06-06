
#include <algorithm>
#include <iostream>
#include <vector>
#include "hip/hip_runtime.h"

__global__ void kernel_shfl_up(int* a, int* b, int n) {
    auto i = hipThreadIdx_x;
    auto d = a[i];
    d = __shfl_up(d, n);
    b[i] = d;
}

__global__ void kernel_shfl_down(int* a, int* b, int n) {
    auto i = hipThreadIdx_x;
    auto d = a[i];
    d = __shfl_down(d,n);
    b[i] = d;
}
__global__ void kernel_shfl_xor(int* a, int mask) {
    auto i = hipThreadIdx_x;
    auto d = a[i];
    d = __shfl_xor(d,mask);
    a[i] = d;
}
__global__ void kernel_shfl_up_width(int* a, int n, const int w) {
    auto i = hipThreadIdx_x;
    auto d = a[i];
    d = __shfl_up(d, n, w);
    a[i] = d;
}

__global__ void kernel_shfl_down_width(int* a, int n, const int w) {
    auto i = hipThreadIdx_x;
    auto d = a[i];
    d = __shfl_down(d, n, w);
    a[i] = d;
}
__global__ void kernel_shfl_xor_width(int* a, int mask, const int w) {
    auto i = hipThreadIdx_x;
    auto d = a[i];
    d = __shfl_xor(d,mask,w);
    a[i] = d;
}
__global__ void kernel_shfl(int* in, int* out, int* index) {
    int i = in[hipThreadIdx_x];
    int x = index[hipThreadIdx_x];
    int o = __shfl(i,x);
    out[hipThreadIdx_x] = o;
}


int main() {
    constexpr int num_threads = 32;

    std::vector<int> d_cpu(num_threads);
    std::generate(d_cpu.begin(), d_cpu.end(), []{ static int c = 1000; return c++;});
    int* d_gpu = nullptr;
    hipMalloc(&d_gpu, d_cpu.size() * sizeof(int));
    hipMemcpy(d_gpu, d_cpu.data(), d_cpu.size() * sizeof(int), hipMemcpyHostToDevice);
 
    std::vector<int> index_cpu(num_threads);
    std::generate(index_cpu.begin(), index_cpu.end(), []{ static int c = 16; return (c++)%32;});
    int* index_gpu = nullptr;
    hipMalloc(&index_gpu, index_cpu.size() * sizeof(int));
    hipMemcpy(index_gpu, index_cpu.data(), index_cpu.size() * sizeof(int), hipMemcpyHostToDevice);
    
    std::vector<int> out_cpu(num_threads);
    int* out_gpu = nullptr;
    hipMalloc(&out_gpu, out_cpu.size() * sizeof(int));

    kernel_shfl<<<1, num_threads>>>(d_gpu, out_gpu, index_gpu);

    hipMemcpy(out_cpu.data(), out_gpu, out_cpu.size() * sizeof(int), hipMemcpyDeviceToHost);

    std::cout << "kernel_shfl output: ";
    for(auto d : out_cpu) {
        std::cout << d << ", ";
    }
    std::cout << std::endl;

    for(int delta = 0; delta < num_threads; ++delta) {
        kernel_shfl_up<<<1, num_threads>>>(d_gpu, out_gpu, delta);
        hipMemcpy(out_cpu.data(), out_gpu, out_cpu.size() * sizeof(int), hipMemcpyDeviceToHost);
        std::cout << "kernel_shfl_up(" << delta << "(: ";
        for(auto d : out_cpu) {
            std::cout << d << ", ";
        }
        std::cout << std::endl;
    }

    hipFree(d_gpu);
    hipFree(index_gpu);
    hipFree(out_gpu);
    return 0;
}