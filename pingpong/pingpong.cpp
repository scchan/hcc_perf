
#include <cstdio>
#include <cassert>
#include <atomic>
#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <thread>

#include <hc.hpp>
#include <hc_am.hpp>


#include "hsa/hsa_ext_amd.h"


static inline void* allocate_shared_mem(size_t size, hc::accelerator accelerator, int allocationMode) {

  void* hostPinned = hc::am_alloc(sizeof(std::atomic<unsigned int>), accelerator
                           , allocationMode
                           );
  printf("shared memory address: %p\n",hostPinned);
  assert(hostPinned != nullptr);
  return hostPinned;
}

static inline void __buffer_flush() [[hc]] {
#if 0
  asm volatile (
         "s_waitcnt vmcnt(0);"
         "buffer_wbinvl1_vol;"
         );
#endif
}


int main(int argc, char* argv[]) {
  // how man GPUs
  unsigned int maxPlayers = 4;

  // how many times/gpu
  unsigned int hits = 10;

  // initial value of the counter
  unsigned int initValue = 1234;

  int allocationMode = amHostCoherent;

  bool threadEmptyKernel = false;

  enum P2PTest {
    LOCKFREE_ATOMIC_COUNTER = 0
    ,LOCK_ATOMIC_COUNTER_SAME_CACHELINE = 1
    ,LOCK_ATOMIC_COUNTER_DIFFERENT_CACHELINE = 2
    ,INVALID_P2P_TEST // last entry
  };
  P2PTest test = LOCKFREE_ATOMIC_COUNTER;

  // process the command line arguments
  {
    const char* options = "eh:i:p:t:a:";
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
        case 't':
          test = (P2PTest) atoi(optarg);
          assert(test < INVALID_P2P_TEST);
          break;
        case 'a':
          {
            allocationMode = atoi(optarg);
            break;
          }
        case 'e':
          threadEmptyKernel = true;
          break;
        default:
          abort();
      }
    }

    printf("Max players: %d\n", maxPlayers);
    printf("# of hits: %d\n", hits);
    printf("Counter initial value: %d\n", initValue);
    printf("test: %d\n", test);
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
  
  std::atomic<unsigned int>* shared_counter = nullptr;
  std::atomic<unsigned int>* lock = nullptr;

  switch(test) {

    case LOCKFREE_ATOMIC_COUNTER:
      hostPinned = (char*) allocate_shared_mem(sizeof(std::atomic<unsigned int>), currentAccelerator, allocationMode);
      shared_counter = new(hostPinned) std::atomic<unsigned int>(initValue);
      break;

    case LOCK_ATOMIC_COUNTER_SAME_CACHELINE:
      // create the counter and the lock on the same cacheline
      hostPinned = (char*) allocate_shared_mem(sizeof(std::atomic<unsigned int>)*2, currentAccelerator, allocationMode);
      shared_counter = new(hostPinned) std::atomic<unsigned int>(initValue);
      lock = new(hostPinned + sizeof(std::atomic<unsigned int>)) std::atomic<unsigned int>(0);
      break;

    case LOCK_ATOMIC_COUNTER_DIFFERENT_CACHELINE:
      // create the counter and the lock on the same cacheline
      hostPinned = (char*) allocate_shared_mem(sizeof(std::atomic<unsigned int>), currentAccelerator, allocationMode);
      shared_counter = new(hostPinned) std::atomic<unsigned int>(initValue);
      hostPinned = (char*) allocate_shared_mem(sizeof(std::atomic<unsigned int>), currentAccelerator, allocationMode);
      lock = new(hostPinned) std::atomic<unsigned int>(0);
      break;


    default:
      abort();
  }

  std::vector<hc::completion_future> futures;
  std::vector<hc::array_view<unsigned int,1>> finalValues;

  for (int i = 0; i < numGPUs; i++) {

    hc::array_view<unsigned int,1> finalValue(1);
    finalValues.push_back(finalValue);


    switch (test) {
      case LOCKFREE_ATOMIC_COUNTER:
       

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

              __buffer_flush();

              if (std::atomic_compare_exchange_weak_explicit(shared_counter
                                                           , &expected
                                                           , expected + 1
                                                           , std::memory_order_seq_cst
                                                           , std::memory_order_relaxed  
                                                           )) {
                last = expected;
                next+=numGPUs;
                count++;


                __buffer_flush();

              }
            } // while(count < hits)
            finalValue[0] = last;
          })
        );
        break;


      case LOCK_ATOMIC_COUNTER_SAME_CACHELINE:
      case LOCK_ATOMIC_COUNTER_DIFFERENT_CACHELINE:
 

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
               unsigned int unlocked = 0;

               __buffer_flush();

               if (std::atomic_compare_exchange_weak_explicit(lock
                                                             , &unlocked
                                                             , (unsigned int)1
                                                             , std::memory_order_seq_cst
                                                             , std::memory_order_relaxed  
                                                             )) {

                 if (shared_counter->load(std::memory_order_relaxed) == expected) {
                   last = expected;
                   next+=numGPUs;
                   count++;
                   shared_counter->store(expected + 1, std::memory_order_relaxed);
                 }
                 lock->store(0, std::memory_order_release);


                 __buffer_flush();
              }
            }
            finalValue[0] = last;
          })
        );
        break;
      default:
        abort();
    }

    std::cout << "GPU %" << i << "(" ;
    std::wcout<< gpus[i].get_description();
    std::cout << ") started\n" << std::endl;
  }
  printf("All GPUs have started\n");

 
  std::atomic<bool> allDone(false);
  std::thread* emptyKernels = nullptr;
  if (threadEmptyKernel) {

    // we create a host thread that keeps on 
    // launching empty kernels to differnt devices
    // we hope that the barrier after kernel would
    // flush and invalidate the cache
    emptyKernels = new std::thread([&]() {
    
      std::vector<hc::accelerator_view> acc_views;
      for(auto&& a : gpus) {
        acc_views.push_back(a.create_view());
      }

      while(!allDone.load()) {
        for(auto&& v : acc_views) {
          // launch an empty kenrel
          hc::parallel_for_each(v, hc::extent<1>(1), [](hc::index<1> i) [[hc]] {

          });
        }
      }
    });
  }


  for (int i = 0; i < futures.size(); ++i) {
    printf("Waiting for GPU #%d to finish\n", i);
    futures[i].wait();
    printf("GPU #%d actual final value: %u, expected final value: %u\n\n"
            , i, finalValues[i][0], initValue + (hits-1) * numGPUs + i);
  }

  if (emptyKernels != nullptr) {
    allDone.store(true);
    emptyKernels->join();
    delete emptyKernels;
  }

  switch(test) {
    case LOCKFREE_ATOMIC_COUNTER:
    case LOCK_ATOMIC_COUNTER_SAME_CACHELINE:
      hc::am_free(shared_counter);
      break;
    case LOCK_ATOMIC_COUNTER_DIFFERENT_CACHELINE:
      hc::am_free(shared_counter);
      hc::am_free(lock);
      break;
    default: ;
  }

  return 0;
}
