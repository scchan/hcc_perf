#include "hip/hip_runtime.h"

__device__ void A(const float a){
    printf("A:%f\n", a);
}

__device__ void B(const float b){
    printf("B:%f\n", b);
}

__device__ void C(const float c){
    printf("C:%f\n", c);
}

__device__ void (*fn_array[]) (const float) = {&A, &B, &B};

__global__ void load_fn_ptr(float* v)
{
    for(int i = 0; i < 3; ++i)
        v[i] = i;
    fn_array[2] = &C;
}

__global__ void call_fn(const float *v, const int N){
#if 0
  for(int i = 0; i < 3; ++i) {
      printf("call_fn: v[%d]: %f\n", i, v[i]);
      fn_array[i](v[i]);
  }
#endif
  fn_array[N](v[N]);
}

int main(int argc, char *argv[])
{
    float* v = nullptr;
    hipMalloc(&v,sizeof(float) * 3u);

    load_fn_ptr<<<1,1>>>(v);

    for (int i = 0; i < 3; ++i) {
      call_fn<<<1,1>>>(v, i);
      hipDeviceSynchronize();
    }
    hipFree(v);
    return 0;
}