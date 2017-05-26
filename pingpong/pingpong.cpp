
#include <cstdio>
#include <cassert>
#include <atomic>
#include <algorithm>
#include <iostream>

#include <hc.hpp>
#include <hc_am.hpp>


#include "hsa/hsa_ext_amd.h"

int main() {

  am_status_t amStatus;

  // how man GPUs
  constexpr unsigned int maxPlayers = 2;

  // how many times/gpu
  constexpr unsigned int hits = 1;

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

  gpus.insert(gpus.begin(), currentAccelerator);
  unsigned int numGPUs = std::min((unsigned int)gpus.size(), maxPlayers);


#if USE_HC_AM
  hostPinned = hc::am_alloc(sizeof(std::atomic<unsigned int>), currentAccelerator
                           //, amHostPinned
                           , amHostCoherent
                           );
  printf("shared memory address: 0x%p\n",hostPinned);

  if (maxPlayers > 1 && gpus.size() != 0) {
    amStatus =hc:: am_map_to_peers(hostPinned, std::min((unsigned int)gpus.size(),maxPlayers-1), gpus.data());
    assert(amStatus == AM_SUCCESS);
  }
#else

  hsa_amd_memory_pool_t* alloc_region = static_cast<hsa_amd_memory_pool_t*>(currentAccelerator.get_hsa_am_system_region());
  assert(alloc_region->handle != -1);

  hsa_status_t hs;
  hs = hsa_amd_memory_pool_allocate(*alloc_region, sizeof(std::atomic<unsigned int>), 0, (void**)&hostPinned);
  assert(hs == HSA_STATUS_SUCCESS);


  hsa_agent_t agents[numGPUs];
  for (int i = 0; i < numGPUs; i++) {
    agents[i] = *(static_cast<hsa_agent_t*> (gpus[i].get_default_view().get_hsa_agent()));
  }
  hs = hsa_amd_agents_allow_access(numGPUs, agents, nullptr, hostPinned);
  assert(hs == HSA_STATUS_SUCCESS);
#endif

  constexpr unsigned int initValue = 1234;
  std::atomic<unsigned int>* shared_counter = new(hostPinned) std::atomic<unsigned int>(initValue);
  std::vector<hc::completion_future> futures;
  std::vector<hc::array_view<unsigned int,1>> finalValues;

  for (int i = 0; i < numGPUs; i++) {

    hc::array_view<unsigned int,1> finalValue(1);
    finalValues.push_back(finalValue);

    futures.push_back(
      hc::parallel_for_each(gpus[i].get_default_view()
                            , hc::extent<1>(1)
                            , [=](hc::index<1> idx) [[hc]] {

        // spin for a while here to ensure that all GPUs have started
        // and that each of them have loaded the inital value of 
        // "shared_counter" into their cache
        #pragma nounroll
        for (int i = 0; i < (1024 * 1024 * 16); ++i) {
          if (shared_counter->load(std::memory_order_relaxed) == 0xFFFFFFFF)
            break;
        }

        // counts how many times this GPU has updated the shared_counter
        unsigned int count = 0;

        unsigned int gpuID = i;
        unsigned int next = initValue + gpuID;

        // last known value of shared_counter observed by this GPU
        unsigned int last = shared_counter->load(std::memory_order_relaxed);


        // each GPU waits for its turn (according to the gpuID) to increment the shared_counter
        #pragma nounroll
        while (count < hits) {
          unsigned int expected = next;
          if (std::atomic_compare_exchange_weak_explicit(shared_counter
                                                        , &expected
                                                        , expected + 1
                                                        , std::memory_order_seq_cst
                                                        , std::memory_order_relaxed  
                                                        //, std::memory_order_acquire
                                                        )) {
            last = expected;
            next+=numGPUs;
            count++;
          }
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
    printf("GPU #%d final value: %u\n", i, finalValues[i][0]);
  }

  if (hostPinned) {
#if USE_HC_AM
    hc::am_free(hostPinned);
#else
    hs = hsa_amd_memory_pool_free(hostPinned);
    assert(hs == HSA_STATUS_SUCCESS);
#endif
  }

  return 0;
}
