
#if defined(__clang__) && defined(__HIP__)
#include "hip/hip_runtime.h"
#endif

static constexpr int num_elements = 16;

template<typename T, int N>
struct A {
    T a[N];
};


#if defined(__HIP__) || defined(__NVCC__)
template<typename T, int N>
__global__ void k_const_aggregate(const A<T, N> a, T* b, const int iter) {
    b[0] = static_cast<T>(0);
    int idx = blockIdx.x % N;
    for (int i = 0; i < iter; ++i) {
        b[0] += a.a[idx];
        idx = (idx == N-1) ? 0 : idx + 1;
    }
}
template __global__ void k_const_aggregate(const A<int, num_elements>, int*, const int);

template<typename T, int N>
__global__ void k(A<T, N> a, T* b, const int iter) {
    b[0] = static_cast<T>(0);
    int idx = blockIdx.x % N;
    for (int i = 0; i < iter; ++i) {
        b[0] += a.a[idx];
        idx = (idx == N-1) ? 0 : idx + 1;
    }
}
template __global__ void k(A<int, num_elements>, int*, const int);


template<typename T, int N>
__attribute((noinline))
__device__ void k_const_aggregate_callee(const A<T, N> a, T* b, const int iter) {
    b[0] = static_cast<T>(0);
    int idx = blockIdx.x % N;
    for (int i = 0; i < iter; ++i) {
        b[0] += a.a[idx];
        idx = (idx == N-1) ? 0 : idx + 1;
    }
}
template<typename T, int N>
__global__ void k_const_aggregate_caller(const A<T, N> a, T* b, const int iter) {
    k_const_aggregate_callee(a, b, iter);
}
template __global__ void k_const_aggregate_caller(const A<int, num_elements>, int*, const int);


#endif

template<typename T, int N>
void host_k_const_aggregate(const A<T, N> a, T* b, const int blockidx, const int iter) {
    b[0] = static_cast<T>(0);
    int idx = blockidx;
    for (int i = 0; i < iter; ++i) {
        b[0] += a.a[idx];
        idx = (idx == N-1) ? 0 : idx + 1;
    }
}
template void host_k_const_aggregate(const A<int, num_elements>, int*, const int, const int);


template<typename T, int N>
void host_k(A<T, N> a, T* b, const int blockidx, const int iter) {
    b[0] = static_cast<T>(0);
    int idx = blockidx;
    for (int i = 0; i < iter; ++i) {
        b[0] += a.a[idx];
        idx = (idx == N-1) ? 0 : idx + 1;
    }
}
template void host_k(A<int, num_elements>, int*, const int, const int);

