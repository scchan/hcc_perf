
#include "hip/hip_runtime.h"

struct base {
  __host__
  __device__
  virtual void pv() = 0;
  __host__ 
  __device__ 
  virtual void dv() = delete;
};

struct derived:base {
  __host__
  __device__
  virtual void pv() override {};
};

__device__ void foo(derived* b) {
    new(b) derived();
}



#if 0
__global__ void bar(derived* b) {
    foo(b);
}

int main() {
   bar<<<1,1>>>(nullptr);
}
#endif