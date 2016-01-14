#include "hc.hpp"
#include<iostream>

#define VECTOR_SIZE 1024 * 1024 * 128

int main() {
  // simple vector addition example
  std::vector<int> data0(VECTOR_SIZE, 1);
  std::vector<int> data1(VECTOR_SIZE, 2);
  std::vector<int> data_out(data0.size(), 0);

  hc::array_view<int, 1> av0(data0.size(), data0);
  hc::array_view<int, 1> av1(data1.size(), data1);
  hc::array_view<int, 1> av2(data_out.size(), data_out);

  av2.discard_data();

  hc::parallel_for_each(av0.get_extent(), [=] (hc::index<1> idx) __attribute__((hc, cpu))
  {
    av2[idx] = av0[idx] + av1[idx];
  });

  for(int i = 0; i < VECTOR_SIZE; i++ )
  {
    if(av2[i] != 3) {
      std::cout<< "Test fails" <<std::endl;
      return 1;
    }
  }
}
