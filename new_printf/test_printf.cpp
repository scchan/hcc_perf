#include <cstdio>
#include <cstdint>

#include <hc.hpp>
#include <hc_printf.hpp>

int main() {
  hc::accelerator acc;
  hc::PrintfPacket* pb = hc::createPrintfBuffer(acc, 1024);

  const char* print_uint64 = "uint64: %lu\n";
  uint64_t u64_data = 1234;

  const char* print_int64 = "int64: %ld\n";
  uint64_t s64_data = 5678;

  const char* print_sizeof_printfpacket = "sizeof(PrintfPacket): %d\n";
  const char* print_hidden_buffer_addr = "hidden addr: %p\n";

  hc::parallel_for_each(hc::extent<1>(64), [=](hc::index<1> idx) [[hc]] {
    if (idx[0]==0) {
      hc::printf(pb, print_uint64, u64_data);
      hc::printf(pb, print_int64, s64_data);
      hc::printf(pb, print_sizeof_printfpacket,sizeof(hc::PrintfPacket));
    }
  }).wait();
  hc::processPrintfBuffer(pb);


  
  const char* print_hello = "hello";
  hc::parallel_for_each(hc::extent<1>(64), [=](hc::index<1> idx) [[hc]] {
    if (idx[0]==0) {

      ADDR_SPACE(1) unsigned char* cp = (ADDR_SPACE(1) unsigned char*) hc::get_printf_buffer_pointer();
 
      int counter = 1;
      ADDR_SPACE(1) hc::PrintfPacket* pp = (ADDR_SPACE(1) hc::PrintfPacket*) (cp + counter * sizeof(hc::PrintfPacket));
      pp[0].data.array_ui[0] = 4;
      counter++;

      pp = (ADDR_SPACE(1) hc::PrintfPacket*) (cp + counter * sizeof(hc::PrintfPacket));
      pp[0].type=hc::PRINTF_UNSIGNED_INT;
      pp[0].data.ui = 1;
      counter++;

      pp = (ADDR_SPACE(1) hc::PrintfPacket*) (cp + counter * sizeof(hc::PrintfPacket));
      pp[0].type=hc::PRINTF_CONST_VOID_PTR;
      pp[0].data.addr_space_1_cptr = (ADDR_SPACE(1) const char*) print_hello;
      counter++;
 
    }
  }).wait();
  //hc::processPrintfBuffer(pb);

  hc::deletePrintfBuffer(pb);
  return 0;
}

