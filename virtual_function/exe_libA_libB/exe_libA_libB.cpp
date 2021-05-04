#include "A.h"
#include "B.h"
#include "hip/hip_runtime.h"
#include "hip_helper.h"

constexpr uint32_t num_objects = 2;

int main() {
    void** buffers{nullptr};
    HIP_CHECK_ERROR(hipHostMalloc(&buffers, num_objects * sizeof(void*)));
    constexpr size_t obj_size = get_max_object_size<b, bd>();
    for(uint32_t i = 0; i < num_objects; ++i) {
        HIP_CHECK_ERROR(hipMalloc(buffers + i, obj_size));
    }

    run_B_construct_object(reinterpret_cast<b**>(buffers), num_objects);
    run_A_invoke_virtual(reinterpret_cast<b**>(buffers), num_objects);

    for(uint32_t i = 0; i < num_objects; ++i) {
        HIP_CHECK_ERROR(hipFree(*(buffers + i)));
    }
    HIP_CHECK_ERROR(hipHostFree(buffers));
    return 0;
}