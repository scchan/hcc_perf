#include "hip/hip_runtime.h"
#include "hip_helper.h"
#include <iostream>
#include <cmath>

extern "C" {
    __device__ void __cxa_pure_virtual(void) {
        abort();
    }
}

class compute_base {
public:
    __host__ 
    __device__ 
    virtual float compute(float x) = 0;
    //{ return x; }
};

class compute_normcdf : compute_base {
public:
#if 0
    __device__
    virtual float compute(float x) override {
        return normcdf(x);
    }
#endif
    __device__
    __host__
    virtual float compute(float x) override {
        return (1.0f + erff(x / sqrtf(2.0f))) / 2.0f;
    }
};


#if 0
class compute_deleted_vir : compute_base {
public:
    __device__
    __host__
    virtual float compute(float x) override = delete;
};
#endif


__global__ void construct_normcdf_obj(compute_normcdf* b) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new(b) compute_normcdf();
    }
}


int main() {

}