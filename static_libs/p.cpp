
#include "hip/hip_runtime.h"
#include "a.hpp"

int main() {
    kernel_a_0<<<1,1,1>>>();
    return 0;
}