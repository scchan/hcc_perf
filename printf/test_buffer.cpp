
#include <cstdlib>
#include <cstdio>
#include <hc.hpp>

#include <hsa.h>
#include "amd_hsa_kernel_code.h"
#include <hc_printf_mod.hpp>

#ifndef ADDR_SPACE
#define ADDR_SPACE(X) __attribute((address_space(X))) 
#endif

extern "C" ADDR_SPACE(1) void* get_dummy_ptr() [[hc]];
extern "C" ADDR_SPACE(2) hsa_kernel_dispatch_packet_t* __dispatch_ptr() [[hc]];


void foo(const char* src, ADDR_SPACE(1) char* dst) [[hc]] {

/*
    unsigned char* p_src = (unsigned char*)&src;
    for(int i = 0; i < 8; i++) {
      dst[i] = p_src[i];
    }
*/
}

void bar(const char* src, ADDR_SPACE(1) char* dst) [[hc]] {

  


}


int main() {

  const char* print_hello = "hello";
  const char* print_hi = "hi from %d\n";

#if 0
  hc::parallel_for_each(hc::extent<1>(64),[=](hc::index<1> i) [[hc]] {

    ADDR_SPACE(1) void** buffer = (ADDR_SPACE(1) void**) get_dummy_ptr();
    buffer[0] = (ADDR_SPACE(1)void*)print_hello;

  }).wait();
#endif 

  hc::parallel_for_each(hc::extent<1>(64),[=](hc::index<1> i) [[hc]] {

    ADDR_SPACE(1) hsa_kernel_dispatch_packet_t* dispatch_packet = (ADDR_SPACE(1) hsa_kernel_dispatch_packet_t* )__dispatch_ptr();

    ADDR_SPACE(1) amd_kernel_code_t* code_struct = (ADDR_SPACE(1) amd_kernel_code_t*) dispatch_packet->kernel_object;
    uint64_t kernel_arg_size = code_struct->kernarg_segment_byte_size;

    ADDR_SPACE(1) char* kernarg = (ADDR_SPACE(1) char*) dispatch_packet->kernarg_address;

    // copy the printf buffer pointer
    ADDR_SPACE(1) unsigned int* printf_buffer_addr_location = (ADDR_SPACE(1) unsigned int*) (kernarg + kernel_arg_size);
    ADDR_SPACE(1) hc::PrintfPacket* printf_buffer;


#if 0
    union {
      unsigned int i[2];
      ADDR_SPACE(1) hc::PrintfPacket* ppb;
    } u;
    u.i[0] = printf_buffer_addr_location[0];
    u.i[1] = printf_buffer_addr_location[1];
    printf_buffer = u.ppb;
#endif

    #if 1


    ADDR_SPACE(1) unsigned int* printf_buffer_ptr = (ADDR_SPACE(1) unsigned int*)&printf_buffer;
    for (int i = 0; i < 2; i++) {
      printf_buffer_ptr[i] = printf_buffer_addr_location[i];
    }
    #endif
    

    hc::printf(printf_buffer, print_hello);
    hc::printf(printf_buffer, print_hi, i[0]);

  }).wait();


}


