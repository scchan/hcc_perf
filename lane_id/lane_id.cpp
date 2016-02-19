
#include <iostream>
#include <hc.hpp>

#define NUM 1024


#ifdef USE_LC

extern "C" int amdgcn_mbcnt_hi(int,int) [[hc]];
extern "C" int amdgcn_mbcnt_lo(int,int) [[hc]];
int get_lane_id() [[hc]] {
  int lane_id = amdgcn_mbcnt_hi(-1,0);
  return amdgcn_mbcnt_lo(-1,lane_id);
}

#else

int get_lane_id() [[hc]] {
  return  hc::__hsail_get_lane_id();
}

#endif


int main() {

  hc::array_view<int, 1> av(NUM);

  hc::parallel_for_each(av.get_extent(), [=](hc::index<1> i) [[hc]] {
    av[i] = get_lane_id();
  });

  for (int i = 0; i < NUM; i++) {
    std::cout << "av[" << i << "] = " << av[i] << std::endl;
  }

  return 0;
}
