
#include <cassert>
#include <atomic>
#include "hc.hpp"
#include "hc_am_internal.hpp"

int main() {

  constexpr unsigned long long init_value = 0;
  constexpr unsigned long long new_value = 1234;

  char* mem = hc::internal::am_alloc_host_coherent(sizeof(std::atomic_ullong));
  std::atomic_ullong* a = new(mem) std::atomic_ullong(init_value);

  hc::array_view<unsigned long long,1> av_old(1);
  av_old[0] = 0xFFFFFFFFFFFFFFFF;

  hc::array_view<bool> av_success(1);
  av_success[0] = false;

  hc::array_view<unsigned long long,1> av_load_after_cas(1);
  av_load_after_cas[0] = 0xFFFFFFFFFFFFFFFF;

  hc::parallel_for_each(hc::extent<1>(1), [=](hc::index<1> i) [[hc]] {
    auto old = a->load();
    av_old[0] = old;
    auto success = a->compare_exchange_weak(old, new_value);
    av_success[0] = success;
    av_load_after_cas[0] = a->load();
  }).wait();

  assert(init_value == av_old[0]);
  assert(av_success[0]);
  assert(new_value == av_load_after_cas[0]);

  hc::am_free(mem);
  return 0;
}
