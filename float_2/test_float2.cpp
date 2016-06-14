#include <hc.hpp>
#include <hc_short_vector.hpp>

int main(int argc, char* argv[]) {

  hc::array_view<hc::short_vector::float2,1> data(64);
  hc::short_vector::float2 hf2 = { 1000.0f, 2000.0f };
  
  if (argc >=3) {
    hf2.x = atof(argv[1]);
    hf2.y = atof(argv[2]);
  }

  hc::parallel_for_each(data.get_extent(), [=](hc::index<1> i) [[hc]] {
    hc::short_vector::float2 v = { (float)i[0], 100.0f + i[0] };
    v+=hf2;
    data[i] = v;
  });

  for (int i = 0; i < data.get_extent()[0]; i++) {
    printf("data[%d]: (%f,%f)\n",i,data[i].x,data[i].y);
  }

  return 0;

}
