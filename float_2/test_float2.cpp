#include <hc.hpp>
#include <hc_short_vector.hpp>

int main() {

  hc::array_view<hc::short_vector::float2,1> data(64);
  hc::parallel_for_each(data.get_extent(), [=](hc::index<1> i) [[hc]] {
    hc::short_vector::float2 v = { (float)i[0], 100.0f + i[0] };
    data[i] = v;
  });

  for (int i = 0; i < data.get_extent()[0]; i++) {
    printf("data[%d]: (%f,%f)\n",i,data[i].x,data[i].y);
  }

  return 0;

}
