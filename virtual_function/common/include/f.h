#pragma once

#include <algorithm>
#include <cstdio>
#include "hip/hip_runtime.h"
#include "hip_helper.h"

#include <tuple>

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

    __host__
    __device__
    int get_value_base() { return v; }

    HIP_HOST_DEVICE
    virtual int get_virtual_value() { return get_value_base(); }

public:
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

    HIP_HOST_DEVICE
    virtual int get_virtual_value() override { return bd_v; }

public:
    int bd_v;
};
