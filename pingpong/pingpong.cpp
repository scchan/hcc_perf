
#include <cstdio>
#include <cassert>
#include <atomic>
#include <algorithm>
#include <iostream>
#include <unistd.h>

#include <hc.hpp>
#include <hc_am.hpp>


#include "hsa/hsa_ext_amd.h"

int main(int argc, char* argv[]) {
  // how man GPUs
  unsigned int maxPlayers = 4;

  // how many times/gpu
  unsigned int hits = 10;

  // initial value of the counter
  unsigned int initValue = 1234;


  // process the command line arguments
  {
    const char* options = "h:i:p:";
    int opt;
    while ((opt = getopt(argc, argv, options))!=-1) {
      switch(opt) {
        case 'h':
          hits = atoi(optarg);
          break;
        case 'i':
          initValue = atoi(optarg);
          break;
        case 'p':
          maxPlayers = atoi(optarg);
          break;
        default:
          abort();
      }
    }

    printf("Max players: %d\n", maxPlayers);
    printf("# of hits: %d\n", hits);
    printf("Counter initial value: %d\n", initValue);
  }

  am_status_t amStatus;

  // pick the default accelerator as the first player
  hc::accelerator currentAccelerator;
  std::vector<hc::accelerator> gpus = currentAccelerator.get_peers();
  if (gpus.size() == 0) {
    printf("No peers found, quiting...\n");
    exit(1);
  }
  else {
    std::cout << "Default accelerator: ";
    std::wcout<< currentAccelerator.get_description();
    std::cout << std::endl;

    for (auto&& p : gpus) {
      std::cout << "\t peer: ";
      std::wcout<< p.get_description();
      std::cout << std::endl;
    }
  }


  gpus.insert(gpus.begin(), currentAccelerator);
  unsigned int numGPUs = std::min((unsigned int)gpus.size(), maxPlayers);

  char* hostPinned = nullptr;


#ifdef USE_LAMBDA

#ifndef USE_ROCR_POOL_API
  auto allocate_mem = [&](void** ptr, size_t size) {
    *ptr = hc::am_alloc(size, currentAccelerator, amHostCoherent);
     printf("shared memory address: %p\n",hostPinned);
     assert(hostPinned != nullptr);
  };
#else
  auto allocate_mem = [&](void** ptr, size_t size) {
    hsa_amd_memory_pool_t* alloc_region = static_cast<hsa_amd_memory_pool_t*>(currentAccelerator.get_hsa_am_finegrained_system_region());
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
  };
#endif

  allocate_mem((void**)&hostPinned, sizeof(std::atomic<unsigned int>));

#else

  hostPinned = hc::am_alloc(sizeof(std::atomic<unsigned int>), currentAccelerator
                           , amHostCoherent
                           );
  printf("shared memory address: %p\n",hostPinned);
  assert(hostPinned != nullptr);

#endif

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
        for (int j = 0; j < (1024 * 1024 * 16); ++j) {
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
    printf("GPU #%d actual final value: %u, expected final value: %u\n\n"
            , i, finalValues[i][0], initValue + (hits-1) * numGPUs + i);
  }

  if (hostPinned) {
#if USE_HC_AM
    hc::am_free(hostPinned);
#else
    hsa_status_t hs = hsa_amd_memory_pool_free(hostPinned);
    assert(hs == HSA_STATUS_SUCCESS);
#endif
  }

  return 0;
}
