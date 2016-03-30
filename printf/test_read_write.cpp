
#include <cstdlib>
#include <cstdio>
#include <hc.hpp>

#include <hc_am.hpp>

#include "kernarg_lib.hpp"

#include <hc_printf.hpp>
//#include <hc_printf_device.hpp>


void pointer_capture() {

  hc::accelerator acc;
  int* buf = hc::am_alloc(sizeof(int), acc, 0);

  hc::parallel_for_each(hc::extent<1>(1), [=](hc::index<1> i) [[hc]] {
    if (i[0]==0) {
      buf[0] = 1234;
    }
  }).wait();

  int host_buf;
  hc::am_copy(&host_buf, buf, sizeof(int));
  hc::am_free(buf);
  
  printf("host: %d\n",host_buf);
}



void pointer_from_kernarg(int d) {

  hc::accelerator acc;

  printf("sizeof(Printfpacket) = %lu\n",sizeof(hc_debug::PrintfPacket));

#define NUM_PACKETS 10
  hc_debug::PrintfPacket cpuPackets[NUM_PACKETS];
  size_t packets_size = sizeof(hc_debug::PrintfPacket)*NUM_PACKETS;
  memset(cpuPackets, 0, packets_size);

  hc_debug::PrintfPacket* packets = (hc_debug::PrintfPacket*) hc::am_alloc(packets_size,acc,0);
  hc::am_copy(packets, cpuPackets, packets_size);

  uint64_t* hidden_pointer = (uint64_t*)hc::am_alloc(sizeof(uint64_t),acc,0);
  


  hc_debug::PrintfPacket* pb = hc_debug::createPrintfBuffer(acc, 1024);
  const char* print_gpu_printfpacket_size = "GPU sizeof(PrintfPacket) = %u\n";
  const char* print_packet_type_address = "packet[%d].type address = %lx\n";
  const char* print_packet_data_address = "packet[%d].data address = %lx\n";
  const char* print_hello = "hello\n";


  printf("print_hello address = %p\n",print_hello);

  hc::parallel_for_each(hc::extent<1>(1), [=](hc::index<1> i) [[hc]] {


    ADDR_SPACE(1) unsigned char* cp = (ADDR_SPACE(1) unsigned char*) get_printf_buffer_pointer();
    *hidden_pointer = (uint64_t)(cp);

    if (i[0]==0) {

      int counter = 1;
      ADDR_SPACE(1) hc_debug::PrintfPacket* pp = (ADDR_SPACE(1) hc_debug::PrintfPacket*) (cp + counter * sizeof(hc_debug::PrintfPacket));
      pp[0].data.aui[0] = 4;
      counter++;

      pp = (ADDR_SPACE(1) hc_debug::PrintfPacket*) (cp + counter * sizeof(hc_debug::PrintfPacket));
      pp[0].type=PRINTF_UNSIGNED_INT;
      pp[0].data.ui = 1;
      counter++;

      pp = (ADDR_SPACE(1) hc_debug::PrintfPacket*) (cp + counter * sizeof(hc_debug::PrintfPacket));
      pp[0].type=PRINTF_CONST_VOID_PTR;
      pp[0].data.addr_space_1_cptr = (ADDR_SPACE(1) const char*) print_hello;
      counter++;

#if 1
      for (int i = 0; i < NUM_PACKETS; i++) {
        // hacks to get around the address calculation bugs
        ADDR_SPACE(1) hc_debug::PrintfPacket* pp = (ADDR_SPACE(1) hc_debug::PrintfPacket*) (cp + i * sizeof(hc_debug::PrintfPacket));
        packets[i].type = pp[0].type;
        packets[i].data.aui[0] = pp[0].data.aui[0];
        packets[i].data.aui[1] = pp[0].data.aui[1];
      }
#endif


#if 0
      // this shows that the address/offset for type and data are wrong!!!
      hc_debug::PrintfPacket* p =  (hc_debug::PrintfPacket*)cp;
      for (int i = 0; i < NUM_PACKETS; i++) {
        hc_debug::printf(pb, print_packet_type_address, i, (uint64_t)&(p[i].type));
        hc_debug::printf(pb, print_packet_data_address, i, (uint64_t)&(p[i].data));
      }

#endif
    }

  }).wait();

  

  uint64_t host_hidden_pointer;
  hc::am_copy(&host_hidden_pointer, hidden_pointer, sizeof(uint64_t));
  printf("hidden pointer: %lx\n", host_hidden_pointer);

  hc::am_copy(cpuPackets, packets, packets_size);
  for (int i = 0; i < NUM_PACKETS; i++) {
    printf("packet[%d]   type=%s   data=%u\n",i, hc_debug::getPrintfPacketDataTypeString(cpuPackets[i].type),cpuPackets[i].data.ui);
  }

  hc_debug::processPrintfBuffer(pb);
  hc_debug::deletePrintfBuffer(pb);

  hc::am_free(hidden_pointer);
  hc::am_free(packets);
}

int main() {

#if 0
  hc::accelerator acc;
  hc_debug::PrintfPacket* pb = hc_debug::createPrintfBuffer(acc,512);

  hc::parallel_for_each(hc::extent<1>(64),[=](hc::index<1> i) [[hc]] {


    


  }).wait();

  hc_debug::processPrintfBuffer(pb);
  hc_debug::deletePrintfBuffer(pb);
#endif


  //pointer_capture();

  int d = 1234;
  pointer_from_kernarg(d);


  return 0;
}


