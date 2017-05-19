
#include <cstdio>
#include <cassert>
#include <atomic>
#include <algorithm>
#include <iostream>

#include <hc.hpp>
#include <hc_am.hpp>

int main() {

  am_status_t amStatus;

  // how man GPUs
  constexpr unsigned int maxPlayers = 2;

  // how many times/gpu
  constexpr unsigned int hits = 0;

  // pick the default accelerator as the first player
  hc::accelerator currentAccelerator;
  std::vector<hc::accelerator> gpus = currentAccelerator.get_peers();
  if (gpus.size() == 0) {
    printf("No peers found, quiting...\n");
    exit(1);
  }
  else {
    printf("Default accelerator has %zu peers\n", gpus.size());
  }

  char* hostPinned = nullptr;
  hostPinned = hc::am_alloc(sizeof(std::atomic<unsigned int>), currentAccelerator, amHostPinned);
  printf("shared memory address: 0x%p\n",hostPinned);

  constexpr unsigned int initValue = 1234;
  std::atomic<unsigned int>* shared_counter = new(hostPinned) std::atomic<unsigned int>(initValue);


#if 1
  if (maxPlayers > 1 && gpus.size() != 0) {
    amStatus =hc:: am_map_to_peers(hostPinned, std::min((unsigned int)gpus.size(),maxPlayers-1), gpus.data());
    assert(amStatus == AM_SUCCESS);
  }
#endif

  gpus.insert(gpus.begin(), currentAccelerator);
  unsigned int numGPUs = std::min((unsigned int)gpus.size(), maxPlayers);

  std::vector<hc::completion_future> futures;
  std::vector<hc::array_view<unsigned int,1>> finalValues;

  for (int i = 0; i < numGPUs; i++) {

    hc::array_view<unsigned int,1> finalValue(1);
    finalValues.push_back(finalValue);

    futures.push_back(
      hc::parallel_for_each(gpus[i].get_default_view()
                            , hc::extent<1>(1)
                            , [=](hc::index<1> idx) [[hc]] {

        unsigned int gpuID = i;
        unsigned int count = 0;
        unsigned int next = initValue + gpuID;
        unsigned int last = shared_counter->load(std::memory_order_relaxed);

        #pragma nounroll
        while (count < hits) {
          unsigned int expected = next;
          if (std::atomic_compare_exchange_weak_explicit(shared_counter
                                                        , &expected
                                                        , expected + 1
                                                        , std::memory_order_seq_cst
                                                        , std::memory_order_relaxed)) {

            last = expected;
            next+=numGPUs;
            count++;
          }
        }

        // Make sure that all GPUs are running the kernels in parallel
        // since kernel completion affects memory visibility
        // Let the kernel spin for a while before quitting
        #pragma nounroll
        for (int i = 0; i < (1024 * 1024 * 16); ++i) {
          if (shared_counter->load(std::memory_order_relaxed) == 0xFFFFFFFF)
            break;
        }

        finalValue[0] = last;
      })
    );

    std::cout << "GPU %" << i << "(" ;
    std::wcout<< gpus[i].get_description();
    std::cout << ") started\n" << std::endl;
  }
  printf("All GPUs have started\n");

  for (int i = 0; i < futures.size(); ++i) {
    printf("Waiting for GPU #%d to finish\n", i);
    futures[i].wait();
  }

  for (int i = 0; i < finalValues.size(); ++i) {
    printf("GPU #%d final value: %u\n", i, finalValues[i][0]);
  }

  if (hostPinned)
    hc::am_free(hostPinned);

  return 0;
}
