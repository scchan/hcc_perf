
#include <cstdlib>
#include <cstdio>
#include <hc.hpp>

#include <hsa.h>
#include "amd_hsa_kernel_code.h"


#include <hc_printf.hpp>

#include <hc_printf_device.hpp>

#ifndef ADDR_SPACE
#define ADDR_SPACE(X) __attribute((address_space(X))) 
#endif

extern "C" ADDR_SPACE(2) hsa_kernel_dispatch_packet_t* __dispatch_ptr() [[hc]];


union PointerConverter {
  unsigned int* ui_ptr;
  unsigned char uca[8];
  unsigned int  uia[2];
};

void pointer_experiment() {
  
  unsigned int* ptr;
  
  PointerConverter pc;
  pc.ui_ptr = ptr;
  
  printf("dumping in hex as chars: \n");
  for (int i = 0; i < 8; i++) {
    printf("0x%02x,",pc.uca[i]);
  }
  printf("\n");


  printf("dumping in hex as unsigned int: \n");
  for (int i = 0; i < 2; i++) {
    printf("0x%08x,",pc.uia[i]);
  }
  printf("\n");


  PointerConverter new_pc;
  new_pc.uia[0] =  *(unsigned int*)&ptr;
  new_pc.uia[1] =  *(((unsigned int*)&ptr)+1);

  printf("dumping in hex as chars: \n");
  for (int i = 0; i < 8; i++) {
    printf("0x%02x,",new_pc.uca[i]);
  }
  printf("\n");


  printf("dumping in hex as unsigned int: \n");
  for (int i = 0; i < 2; i++) {
    printf("0x%08x,",new_pc.uia[i]);
  }
  printf("\n");


}

int main() {

 // pointer_experiment();

  const char* print_hello = "hello\n";
  const char* print_hi = "hi from %d\n";

  const char* print_kernel_arg_size = "kernel_arg_size = %d\n";
  const char* print_kernel_arg = "dumping kernel arg:\n";
  const char* print_hex = "0x%02x,";

  const char* print_printf_buffer_pointer = "dumping printf buffer pointer:\n";

  const char* print_pointer = "%p";
  

  const char* print_newline = "\n";
  
  hc::accelerator acc;
  hc_debug::PrintfPacket* pb = hc_debug::createPrintfBuffer(acc,512);

  hc::parallel_for_each(hc::extent<1>(64),[=](hc::index<1> i) [[hc]] {

    if (i[0] == 0) {

      ADDR_SPACE(1) hsa_kernel_dispatch_packet_t* dispatch_packet = (ADDR_SPACE(1) hsa_kernel_dispatch_packet_t* )__dispatch_ptr();
      ADDR_SPACE(1) amd_kernel_code_t* code_struct = (ADDR_SPACE(1) amd_kernel_code_t*) dispatch_packet->kernel_object;
      uint64_t kernel_arg_size = code_struct->kernarg_segment_byte_size;
      ADDR_SPACE(1) char* kernarg = (ADDR_SPACE(1) char*) dispatch_packet->kernarg_address;

      // copy the printf buffer pointer
      ADDR_SPACE(1) unsigned char* printf_buffer_addr_location = (ADDR_SPACE(1) unsigned char*) (kernarg + kernel_arg_size);
      ADDR_SPACE(1) hc_debug::PrintfPacket* printf_buffer;
      volatile ADDR_SPACE(0) unsigned char* printf_buffer_ptr = (ADDR_SPACE(0) unsigned char*)&printf_buffer;
      for (int i = 0; i < 8; i++) {
        printf_buffer_ptr[i] = printf_buffer_addr_location[i];
      }

      // print kernel arg size
      hc_debug::printf(pb, print_kernel_arg_size, (int)kernel_arg_size);

      // print kernel arg contain
      hc_debug::printf(pb, print_kernel_arg);
      for (int i = 0; i < kernel_arg_size+8; i++) {
        hc_debug::printf(pb, print_hex, (unsigned char)kernarg[i]);
      }
      hc_debug::printf(pb, print_newline);


      // print printf buffer pointer
      hc_debug::printf(pb, print_printf_buffer_pointer);
      for (int i = 0; i < 8; i++) {
        hc_debug::printf(pb, print_hex, (unsigned char)printf_buffer_ptr[i]);
      }
      hc_debug::printf(pb, print_newline);

      


#if 0
      union {
       volatile hc_debug::PrintfPacket* pp;
       volatile ADDR_SPACE(1) hc_debug::PrintfPacket* vpp;
      } u;

      u.vpp = printf_buffer;
      hc_debug::printf((hc_debug::PrintfPacket*) u.pp, print_hello);
#endif


      hc::printf(print_hello);

    }
  }).wait();

  hc_debug::processPrintfBuffer(pb);
  hc_debug::deletePrintfBuffer(pb);

  return 0;
}


