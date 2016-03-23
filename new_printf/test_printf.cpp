#include <cstdio>
#include <cstdint>

#include <hc.hpp>
#include <hc_printf.hpp>

int main() {
  hc::accelerator acc;
  hc::PrintfPacket* pb = hc::createPrintfBuffer(acc, 1024);

  const char* print_uint64 = "uint64: %lu\n";
  uint64_t u64_data = 1234;


  hc::parallel_for_each(hc::extent<1>(64), [=](hc::index<1> idx) [[hc]] {
    if (idx[0]==0) {
      hc::printf(pb, print_uint64, u64_data);
    }
  }).wait();

  hc::processPrintfBuffer(pb);


  hc::deletePrintfBuffer(pb);
  return 0;
}

