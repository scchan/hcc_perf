#include <vector>
#include <cstdlib>
#include "hip/hip_runtime.h"

static constexpr int num_elements = 16;

template<typename T, int N>
class A {

   public:
  __host__ __device__ T& operator[](int32_t index) {
    return a[index];
  }

  __host__ __device__ __forceinline__ const T& operator[](int32_t index) const {
    return a[index];
  }

    T a[N];
};


template<typename T, int N>
__global__ void k_const_aggregate(const A<T, N> a, T* b, const int iter) {
    b[0] = static_cast<T>(0);
    int idx = blockIdx.x % N;
    for (int i = 0; i < iter; ++i) {
        //b[0] += a.a[idx];
        b[0] += a[idx];
        idx = (idx == N-1) ? 0 : idx + 1;
    }
}
template __global__ void k_const_aggregate(const A<int, num_elements>, int*, const int);

#if 1
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
#endif


template <typename T, int32_t capacity = 8>
struct TArray {
  TArray() : size_(0), data_() {
  }

  TArray(int32_t size) : size_(size), data_() {

  #if 0
    ORT_ENFORCE(
        0 <= size && size <= capacity,
        "TArray size must be within range [0, ", capacity, "]. Actual: ", size);
  #endif
  }

  TArray(const std::vector<T>& vec) : TArray(static_cast<int32_t>(vec.size())) {
// std::is_trivially_copyable is not implemented in older versions of GCC
#if !defined(__GNUC__) || __GNUC__ >= 5
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
#endif
    memcpy(data_, vec.data(), vec.size() * sizeof(T));
  }

  void SetSize(int32_t size) {
#if 0
    ORT_ENFORCE(
        0 <= size && size <= capacity,
        "TArray size must be within range [0, ", capacity, "]. Actual: ", size);
#endif
    size_ = size;
  }

  __host__ __device__ int32_t Size() const {
    return size_;
  }

  __host__ __device__ T& operator[](int32_t index) {
    return data_[index];
  }

  __host__ __device__ __forceinline__ const T& operator[](int32_t index) const {
    return data_[index];
  }

  __host__ __device__ T* Data() {
    return data_;
  }

  __host__ __device__ const T* Data() const {
    return data_;
  }

  static constexpr int32_t Capacity() { return capacity; };

 private:
  int32_t size_;
  T data_[capacity];
};


struct fast_divmod {
  fast_divmod(int d = 1) {
    d_ = d == 0 ? 1 : d;
    //ORT_ENFORCE(d_ >= 1 && d_ <= static_cast<uint32_t>(std::numeric_limits<int>::max()));

    for (l_ = 0; l_ < 32; l_++) if ((1U << l_) >= d_) break;

    uint64_t one = 1;
    uint64_t m = ((one << 32) * ((one << l_) - d_)) / d_ + 1;
    M_ = static_cast<uint32_t>(m);
    // according to paper, the value of m' should fit in a unsigned integer.
    //ORT_ENFORCE(M_ > 0 && M_ == m);
  }

  __host__ __device__ inline int div(int n) const {
#ifdef __HIP_DEVICE_COMPILE__
    uint32_t t = __umulhi(M_, n);
    return (t + n) >> l_;
#else
    // Using uint64_t for t, then t + n won't overflow.
    uint64_t t = ((uint64_t) M_ * n) >> 32;
    return static_cast<int>((t + n) >> l_);
#endif
  }

  __host__ __device__ inline int mod(int n) const {
    return n - div(n) * d_;
  }

  __host__ __device__ inline void divmod(int n, int& q, int& r) const {
    q = div(n);
    r = n - q * d_;
  }

  uint32_t d_;  // divisor
  uint32_t M_;  // m' in the paper.
  uint32_t l_;  // l_ = ceil(log2(d_))
};


template<typename T, int N>
__global__ void k_const_TArray(const TArray<T, N> a, T* b, const int iter) {
    b[0] = static_cast<T>(0);
    int idx = blockIdx.x % N;
    for (int i = 0; i < iter; ++i) {
        b[0] += a[idx];
        idx = (idx == N-1) ? 0 : idx + 1;
    }
}
template __global__ void k_const_TArray(const TArray<int, num_elements>, int*, const int);

#ifndef HIP_LONG
#define HIP_LONG int32_t
#endif

#define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(N)          \
  HIP_LONG id = blockDim.x * blockIdx.x + threadIdx.x;     \
  if (id >= N)                                              \
    return;


template <typename T>
#if 1
__global__ void _TransposeKernel(int32_t shape_rank, const TArray<T> input_strides,
                                  const T* __restrict__ input_data, const TArray<fast_divmod> output_strides, T* __restrict__ output_data, HIP_LONG N) {
#else
__global__ void _TransposeKernel(int32_t shape_rank, const int64_t* input_strides,
  const T* __restrict__ input_data, const fast_divmod* output_strides, T* __restrict__ output_data, HIP_LONG N) {
#endif
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(N);
  HIP_LONG input_index = 0;
  HIP_LONG output_index = id;
  // #pragma unroll
  // for (auto dim = 0; dim < input_strides.GetCapacity(); ++dim) {
  //   if (dim >= shape_rank) {
  //     break;
  //   }
  for (auto dim = 0; dim < shape_rank; ++dim) {
    int out_coord, r;
    output_strides[dim].divmod(output_index, out_coord, r);
    output_index = r;
    input_index += input_strides[dim] * out_coord;
  }
  output_data[id] = input_data[input_index];
}
template __global__ void _TransposeKernel(int32_t, const TArray<int64_t>, const int64_t*, const TArray<fast_divmod>, int64_t*, HIP_LONG);





#if 0
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



#if 0
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
#endif
