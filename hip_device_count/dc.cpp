#include <cstdio>
#include "hip/hip_runtime.h"

int main() {
  int count = 0;
  hipGetDeviceCount(&count);
  printf("device count: %d\n", count);
  return 0;
}

