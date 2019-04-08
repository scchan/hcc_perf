#include <hc.hpp>
#include "is_gfx.hpp"

int main() {
  
  hc::array_view<int, 1> data(1);
  data[0] = -1;
  hc::parallel_for_each(data.get_extent(), [=](hc::index<1> i) [[hc]] {
    if (is_gfx900())
      data[i] = 1;
    else
      data[i] = 0;
  });
  printf("testing is_gfx900() data[0]: %d\n", data[0]);

  data[0] = -1;
  hc::parallel_for_each(data.get_extent(), [=](hc::index<1> i) [[hc]] {
    data[i] = get_isa_version();
  });
  printf("testing get_isa_version(): %d\n", data[0]);

  return 0;
}
