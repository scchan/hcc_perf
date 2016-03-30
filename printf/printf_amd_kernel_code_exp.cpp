
#include "amd_hsa_kernel_code.h"

#include<hc.hpp>
#include<hc_printf.hpp>

#define PRINTF_BUFFER_SIZE (512)

extern "C" hsa_kernel_dispatch_packet_t* __dispatch_ptr() [[hc]];

using namespace hc;

int main() {

/*
  hc::accelerator acc;
  PrintfPacket* printf_buffer = createPrintfBuffer(acc, PRINTF_BUFFER_SIZE);
  */

  const char* print_hello = "hello world!\n";
  const char* print_kernarg_size = "kernel arg size: %d\n";
  const char* print_grid_size = "x: %d, y: %d, z: %d\n";
  const char* print_hex = "0x%02x,";

  parallel_for_each(extent<1>(64),[=](index<1> i) [[hc]] {
    if (i[0]==0) {
      hsa_kernel_dispatch_packet_t* packet = (hsa_kernel_dispatch_packet_t*) __dispatch_ptr();
      amd_kernel_code_t* code_struct = (amd_kernel_code_t*) packet->kernel_object;
      uint64_t kernel_arg_size = code_struct->kernarg_segment_byte_size;
      unsigned char* kernarg_addr = (unsigned char*)packet->kernarg_address;


union {
  void* p;
  __attribute__((address_space(1)))void* as1p; 
}u;

      u.p = (kernarg_addr + kernel_arg_size - 8);

 //     PrintfPacket* printf_buffer = *((PrintfPacket**)(kernarg_addr + kernel_arg_size - 8));

      __attribute__((address_space(1)))  PrintfPacket* printf_buffer = ( __attribute__((address_space(1))) PrintfPacket*)u.as1p;


      //printf(printf_buffer, print_grid_size, packet->grid_size_x, packet->grid_size_y, packet->grid_size_z);

      //printf(printf_buffer, print_kernarg_size, (unsigned int)kernel_arg_size);
      

      printf(printf_buffer, print_hello);

      /*
      for (int i = 0; i < kernel_arg_size+8; i++) {
        printf(printf_buffer, print_hex, kernarg_addr[i]);
      }
      */

    }
  }).wait();

/*
  processPrintfBuffer(printf_buffer);
  std::printf("\n");

  deletePrintfBuffer(printf_buffer);
*/

  return 0;
}
