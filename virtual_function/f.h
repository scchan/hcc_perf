#include <algorithm>
#include <cstdio>
#include "hip/hip_runtime.h"

class b {
public:
    __host__
    __device__
    b() : v(0) {}

    __host__
    __device__
    b(uint32_t i) : v(i) {}

    __host__
    __device__
    void base_print() {
        printf("%s: %d\n", __PRETTY_FUNCTION__, this->v);
    }

    __host__
    __device__
    virtual void virtual_print() {
        printf("%s: %d\n", __PRETTY_FUNCTION__, this->v);
    }
protected:
    int v;
};

class bd : public b {
public:
    __host__
    __device__
    bd() : bd(0) {}

    __host__
    __device__
    bd(uint32_t i) : b(i), bd_v(i + 1234) {}
  
    __host__
    __device__
    void virtual_print() override {
        printf("%s: v=%d, bd_v=%d\n", __PRETTY_FUNCTION__, this->v, this->bd_v);
    }
protected:
    int bd_v;
};


__host__
__device__
inline size_t get_max_object_size() {
    return std::max(sizeof(b), sizeof(bd));
}