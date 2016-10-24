#include <cstdio>
#include <hc.hpp>

//#define GLOBAL_VAR_BUG 1

#ifdef GLOBAL_VAR_BUG
__attribute__((hc)) 
#else
__attribute__((address_space(1))) 
#endif
char buffer[1024*1024*32];


int main() {
  parallel_for_each(hc::extent<1>(1), [=](hc::index<1> i) [[hc]] {
    if (i[0]==0) {
      const char* helloWorld = "Hello World!";
      for (int j = 0; j <= 12; j++)
        buffer[j] = helloWorld[j];
    }
  }).wait();

}
