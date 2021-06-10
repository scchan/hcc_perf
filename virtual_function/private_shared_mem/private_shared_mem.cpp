#include <cstdio>
#include "hip/hip_runtime.h"

struct b {
  __host__
  __device__
  virtual void my_print() {
    printf("Hello\n");
  }
  __host__ 
  __device__
  virtual int get() {
    return 100;
  }
};

struct d : b {
  __host__ 
  __device__
  virtual void my_print() override {
    printf("World\n");
  }
  __host__ 
  __device__
  virtual int get() override {
    return 200;
  }
};

__global__ void abcd() {
    d x;
    x.my_print();
}

__global__ void xyz_mod() {
  __shared__ char b[1024];
  new ((d*)b) d();
  auto v = ((d*)b)->get();
  printf("%s: get(): %d\n", __PRETTY_FUNCTION__, v);
}

__global__ void xyz() {
  __shared__ char b[1024];
  new ((d*)b) d();
  ((d*)b)->my_print();
}

int main() {
    abcd<<<1,1>>>();
    hipDeviceSynchronize();
    xyz_mod<<<1,1>>>();
    hipDeviceSynchronize();
    xyz<<<1,1>>>();
    hipDeviceSynchronize();
    return 0;
}
