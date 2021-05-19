#pragma once

#include <algorithm>
#include <cstdio>
#include "hip/hip_runtime.h"
#include "hip_helper.h"

#include <tuple>

//#define __DEBUG__ 1


class b {
public:
    __host__
    __device__
    b() : b(0) {}

    __host__
    __device__
    b(uint32_t i) : v(i) {
#ifdef __DEBUG__
        printf("%s: v=%u\n", __PRETTY_FUNCTION__, this->v);
#endif
    }

    __host__
    __device__
    void base_print() {
        printf("%s: %u\n", __PRETTY_FUNCTION__, this->v);
    }

    __host__
    __device__
    virtual void virtual_print() {
        printf("%s: %u\n", __PRETTY_FUNCTION__, this->v);
    }

    typedef decltype(&b::virtual_print) fp;

    __host__
    __device__
    virtual fp get_virtual_print_addr() {
        return &b::virtual_print;
    }

    __host__
    __device__
    uint32_t get_base() const { return v; }

    HIP_HOST_DEVICE
    virtual uint32_t get_virtual() const { return get_base(); }

public:
    uint32_t v;
};


#define BD_MAGIC 900000
class bd : public b {
public:
    __host__
    __device__
    bd() : bd(0) {}

    __host__
    __device__
    bd(uint32_t i) : b(i), bd_v(i + BD_MAGIC) {
#ifdef __DEBUG__
        printf("%s: v=%u, bd_v=%u\n", __PRETTY_FUNCTION__, this->v, this->bd_v);
#endif
    }
  
    __host__
    __device__
    void virtual_print() override {
        printf("%s: v=%u, bd_v=%u\n", __PRETTY_FUNCTION__, this->v, this->bd_v);
    }

    HIP_HOST_DEVICE
    virtual uint32_t get_virtual() const override { return bd_v; }

public:
    uint32_t bd_v;
};
