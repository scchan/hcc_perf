#include<hc.hpp>
#include<hc_printf.hpp>

#define PRINTF_BUFFER_SIZE (512)


using namespace hc;

int main() {

  
  hc::accelerator acc;
  PrintfPacket* printf_buffer = createPrintfBuffer(acc, PRINTF_BUFFER_SIZE);
  
  const char* hello = "Hello World from %d!\n";
  parallel_for_each(extent<1>(64),[=](index<1> i) [[hc]] {
    printf(printf_buffer, hello, i[0]);
  }).wait();


  processPrintfBuffer(printf_buffer);
  deletePrintfBuffer(printf_buffer);

  return 0;
}
