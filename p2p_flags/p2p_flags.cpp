
#include <cstdio>
#include <vector>
#include <atomic>
#include <cassert>

#include <hc.hpp>
#include <hc_am.hpp>


int main() {

  // pick the default accelerator as the first player
  hc::accelerator currentAccelerator;
  std::vector<hc::accelerator> gpus = currentAccelerator.get_peers();
  if (gpus.size() == 0) {
    printf("no peers!\n");
    exit(1);
  }

  std::vector<hc::accelerator> devices;
  devices.push_back(currentAccelerator);
  devices.push_back(gpus[0]);


#ifdef USE_DEVICE_MEM

  std::atomic<unsigned int>* p2pflag = (std::atomic<unsigned int>*) hc::am_alloc(sizeof(std::atomic<unsigned int>)
                                                                               , devices[0]
                                                                               , 0
                                                                               );
  assert(p2pflag != nullptr);
  
  assert(hc::am_map_to_peers(p2pflag, 1, devices.data()) == AM_SUCCESS);
#else


  std::atomic<unsigned int>* p2pflag = (std::atomic<unsigned int>*) hc::am_alloc(sizeof(std::atomic<unsigned int>)
                                                                               , devices[0]
                                                                               , amHostCoherent
                                                                               );

  

#endif

  p2pflag->store(0);

  std::vector<hc::completion_future> futures;
  constexpr unsigned int BASE = 100;
  for(int i = 0; i < 2; i++) {
    futures.push_back(hc::parallel_for_each(devices[i].get_default_view(),hc::extent<1>(1), [=](hc::index<1> idx) [[hc]] {
      if (idx[0] == 0) {
        if (i == 0) {
          p2pflag->store(BASE+i, std::memory_order_release);
          while(p2pflag->load(std::memory_order_acquire) != BASE+i+1) ;
        }
        else if (i == 1) {
          while(p2pflag->load(std::memory_order_acquire) != BASE+i-1) ;
          p2pflag->store(BASE+i, std::memory_order_release);
        }
      }
    }));
  }

  for (int i = 1; i >= 0; --i) {
    printf("Waiting device #%d\n",i);
    futures[i].wait();
  }

  hc::am_free(p2pflag);

  return 0;
}
