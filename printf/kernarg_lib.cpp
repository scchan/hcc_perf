
#include <hsa.h>
#include "amd_hsa_kernel_code.h"
#include "kernarg_lib.hpp"

extern "C" ADDR_SPACE(2) hsa_kernel_dispatch_packet_t* __dispatch_ptr() [[hc]];

uint64_t get_kernarg_pointer(void** kernarg_ptr) [[hc]] {
  ADDR_SPACE(1) hsa_kernel_dispatch_packet_t* dispatch_packet = (ADDR_SPACE(1) hsa_kernel_dispatch_packet_t* )__dispatch_ptr();
  ADDR_SPACE(1) amd_kernel_code_t* code_struct = (ADDR_SPACE(1) amd_kernel_code_t*) dispatch_packet->kernel_object;
  uint64_t kernel_arg_size = code_struct->kernarg_segment_byte_size;
  *kernarg_ptr = (void*) dispatch_packet->kernarg_address;
  return kernel_arg_size;
}

void* get_printf_buffer_pointer() [[hc]] {
  unsigned char* kernarg_buf;
  uint64_t kernarg_size;
  kernarg_size = get_kernarg_pointer((void**)&kernarg_buf);
  ADDR_SPACE(1) uint64_t* buf = (ADDR_SPACE(1) uint64_t*)(kernarg_buf + kernarg_size);
  return (void*) *buf;
}

