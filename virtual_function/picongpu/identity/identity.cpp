#include "hip/hip_runtime.h"
#include "hip_helper.h"
#include <tuple>

__device__ float identity(const float a){
    return a;
}

__device__ float relu(const float a){
    return (a>0.0f?a:0.0f);
}

__device__ float (*array[]) (const float) = {&identity, &relu};

__global__ void vadd_hip(const float *a, float *c, int N){
  float cc = array[N](*a);
  //printf("vadd_hip: *a=%f, cc=%f\n", *a, cc);
  *c = cc;
}

int main(int argc, char *argv[])
{
  float* input;
  HIP_CHECK_ERROR(hipMalloc(&input, sizeof(float)));
  float* output;
  HIP_CHECK_ERROR(hipMalloc(&output, sizeof(float)));

  float t = -1234.0f;
  HIP_CHECK_ERROR(hipMemcpy(input, &t, sizeof(float), hipMemcpyHostToDevice));

  t = 999.0f; 
  HIP_CHECK_ERROR(hipMemcpy(output, &t, sizeof(float), hipMemcpyHostToDevice));

  vadd_hip<<<1,1>>>(input, output, 0);
  HIP_CHECK_ERROR(hipMemcpy(&t, output, sizeof(float), hipMemcpyDeviceToHost));
  printf("Expected: %f, Actual: %f\n", -1234.0f, t);

  t = 999.0f; 
  HIP_CHECK_ERROR(hipMemcpy(output, &t, sizeof(float), hipMemcpyHostToDevice));
  vadd_hip<<<1,1>>>(input, output, 1);
  HIP_CHECK_ERROR(hipMemcpy(&t, output, sizeof(float), hipMemcpyDeviceToHost));
  printf("Expected: %f, Actual: %f\n", 0.0f, t);

  HIP_CHECK_ERROR(hipFree(input));
  HIP_CHECK_ERROR(hipFree(output));

  return 0;
}