#include "f.h"
#include "hip/hip_runtime.h"
#include "hip_helper.h"

template<typename T>
__global__ void copy_construct_from_host_object(T* s , uint32_t* data) {
#if 0
    T d(*s);
#endif
    //T d(1234);
    new (s) T(1234);
    T d(1234);
    //T d(s->v);
    //T e(d);
    //*data = d.v;
    //*data = d.get_virtual();
   // *data = d.v;
#if 0
    printf("%s get_base()=%u, get_virtual()=%u\n", __PRETTY_FUNCTION__,
            d.get_base(), d.get_virtual());
#endif

}

template<typename T>
void __run_copy_construct_from_host_object() {
    T* buffer{nullptr};
    HIP_CHECK_ERROR(hipHostMalloc(&buffer, sizeof(T)));
    new (buffer) T(1234);

    uint32_t* data{nullptr};
    HIP_CHECK_ERROR(hipHostMalloc(&data, sizeof(uint32_t)));
    *data = 0xDEADBEEF;

    copy_construct_from_host_object<<<1, 1>>>(buffer, data);
    buffer->~T();
    printf("%s: data=%u\n", __PRETTY_FUNCTION__, *data);

    HIP_CHECK_ERROR(hipHostFree(buffer));
    HIP_CHECK_ERROR(hipHostFree(data));
}

void run_copy_construct_from_host_objects() {
    __run_copy_construct_from_host_object<b>();
    //__run_copy_construct_from_host_object<bd>();
}

int main() {
    run_copy_construct_from_host_objects();
    return 0;
}