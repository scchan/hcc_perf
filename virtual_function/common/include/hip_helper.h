
#pragma once

#include <cstdlib>
#include <iostream>
#include "hip/hip_runtime.h"

#define HIP_CHECK_ERROR(x) \
    {hipError_t e = x;\
    if (e != HIP_SUCCESS) {\
        std::cerr << __FILE__ << ":" << __LINE__ << " HIP error " << e << std::endl;\
        std::exit(1);\
    }}\

#define HIP_HOST_DEVICE __host__ __device__

template<size_t n>
HIP_HOST_DEVICE
constexpr size_t __get_max_object_size_impl() {
    return 0;
}
template<size_t n, typename T, typename... Targs>
HIP_HOST_DEVICE
constexpr size_t __get_max_object_size_impl() {
    size_t this_size = sizeof(T);
    size_t rest_max = __get_max_object_size_impl<n+1, Targs...>();
    if (this_size > rest_max)
        return this_size;
    else
        return rest_max;
}
template<typename... Targs>
HIP_HOST_DEVICE
constexpr size_t get_max_object_size() {
    return __get_max_object_size_impl<0, Targs...>();
}
