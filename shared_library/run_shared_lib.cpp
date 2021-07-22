#include "hip/hip_runtime.h"
#include "A.h"

__global__ void world() {
   printf("World\n");
}

int main() {
   run_hello();
   world<<<1,1>>>();
   hipDeviceSynchronize();
   return 0;
}