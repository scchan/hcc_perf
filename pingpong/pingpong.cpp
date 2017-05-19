
#include <cstdio>
#include <atomic>
#include <hc.hpp>
#include <hc_am.hpp>

int main() {

  std::vector<hc::accelerator> accelerators = hc::accelerator::get_all();
  std::vector<hc::accelerator> gpus;
  for (auto&& a : accelerators) {
    if (a.is_hsa_accelerator())
      gpus.push_back(a);
  }

  printf("Found %zu HSA accelerators\n", gpus.size());
  if (gpus.size() < 2) {
    printf("Not enough players, quiting...\n");
    exit(0);
  }

  std::atomic<unsigned int>* shared_counter = new std::atomic<unsigned int>(0);

  hc::accelerator currentAccelerator = gpus[0];
  hc::am_memory_host_lock(currentAccelerator
                          , shared_counter
                          , sizeof(std::atomic<unsigned int>)
                          , gpus.data()+1
                          , gpus.size()-1);

  // how many times/gpu
  constexpr unsigned int hits = 1024;
  unsigned int numGPUs = gpus.size();
  std::vector<hc::completion_future> futures;
  for (int i = 0; i < numGPUs; i++) {
    futures.push_back(
      hc::parallel_for_each(gpus[i].get_default_view()
                            , hc::extent<1>(1)
                            , [=](hc::index<1> idx) [[hc]] {

        unsigned int gpuID = i;
        unsigned int count = 0;
        unsigned int next = gpuID;

        #pragma nounroll
        while (count < hits) {
          unsigned int expected = next;
          if (std::atomic_compare_exchange_weak_explicit(shared_counter
                                                        , &expected
                                                        , expected + 1
                                                        , std::memory_order_seq_cst
                                                        , std::memory_order_relaxed)) {
            next+=numGPUs;
            count++;
          }
        }
      })
    );
  }

  for (auto&& f : futures)
    f.wait();

  hc::am_memory_host_unlock(currentAccelerator, shared_counter);
  delete shared_counter;
  return 0;
}
