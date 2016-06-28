#include <cstdio>
#include <hc.hpp>

int main() {

  float f = 1234.0f;
  hc::array_view<float,1> af(1,&f);

  hc::parallel_for_each(af.get_extent(), [=](hc::index<1> i) [[hc]] {
    af[i]+=i[0];
  });

  printf("af[0]: %f\n",af[0]);
}
