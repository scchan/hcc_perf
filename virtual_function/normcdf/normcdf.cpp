#include "hip/hip_runtime.h"
#include "hip_helper.h"
#include <iostream>
#include <cmath>

struct compute_base {
    __host__ 
    __device__ 
    virtual float compute(const float x) const = 0;
};
struct compute_normcdf : compute_base {
    __device__
    __host__
    virtual float compute(const float x) const override {
        return (1.0f + erff(x / sqrtf(2.0f))) / 2.0f;
    }
};

__global__ void construct_normcdf_obj(compute_normcdf* b) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new(b) compute_normcdf();
    }
}


__global__ void compute(const compute_base* b, const float* input, float* output, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = b[i].compute(input[i]);
    }
}



int main() {

}