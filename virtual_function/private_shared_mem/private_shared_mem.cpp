#include <cstdio>
#include "hip/hip_runtime.h"

struct b {
  __host__
  __device__
  virtual void my_print() {
    printf("Hello\n");
  }
};

struct d : b {
  __host__ 
  __device__
  virtual void my_print() override {
    printf("World\n");
  }
};

__global__ void abcd() {
    d x;
    x.my_print();
}

__global__ void xyz() {
  __shared__ char b[128];
  new ((d*)b) d();
  ((d*)b)->my_print();
}

int main() {
    abcd<<<1,1>>>();
    xyz<<<1,1>>>();
    hipDeviceSynchronize();
    return 0;
}
