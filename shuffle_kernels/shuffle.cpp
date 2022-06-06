
#include "hip/hip_runtime.h"

__global__ void kernel_shfl_up(int* a, int n) {
    auto i = hipThreadIdx_x;
    auto d = a[i];
    d = __shfl_up(d, n);
    a[i] = d;
}

__global__ void kernel_shfl_down(int* a, int n) {
    auto i = hipThreadIdx_x;
    auto d = a[i];
    d = __shfl_up(d,n);
    a[i] = d;
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
    d = __shfl_up(d, n, w);
    a[i] = d;
}
__global__ void kernel_shfl_xor_width(int* a, int mask, const int w) {
    auto i = hipThreadIdx_x;
    auto d = a[i];
    d = __shfl_xor(d,mask,w);
    a[i] = d;
}


int main() {

}