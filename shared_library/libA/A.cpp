#include "hip/hip_runtime.h"
#include "A.h"

__global__ void hello() {
    printf("Hello\n");
}

void run_hello() {
    hello<<<1,1>>>();
    hipDeviceSynchronize();
}
