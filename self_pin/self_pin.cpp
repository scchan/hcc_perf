
#include <vector>
#include <hc.hpp>
#include <hc_am.hpp>

template <typename T>
class PrePinned {
public:
  PrePinned() [[cpu]] {
    hc::accelerator acc;
    auto peers = acc.get_peers();
    peers.push_back(acc);
    am_status_t status;
    status = hc::am_memory_host_lock(acc, this, sizeof(PrePinned<T>)
                                , peers.data(), peers.size());
    assert(status == AM_SUCCESS);
  }
  ~PrePinned() [[cpu]] {
    hc::accelerator acc;
    am_status_t status;
    status = hc::am_memory_host_unlock(acc, this);
    assert(status == AM_SUCCESS);
  }
  T data;
};


#if 0
#ifdef __HCC_CPU__
PrePinned<int> bar;
#else
[[hc]] PrePinned<int> bar;
#endif
#else
PrePinned<int> bar;
#endif

int main() {

//  PrePinned<int> bar;

  bar.data = 1234;
  hc::array_view<int,1> av(1);

  printf("before bar.data: %d\n", bar.data);

  PrePinned<int>* barp = &bar;

  hc::parallel_for_each(hc::extent<1>(1), [=] (hc::index<1> i) [[hc]] {
    av[0] = barp->data;
  }).wait();

  printf("after bar.data: %d\n", bar.data);
  printf("av[0]:%d\n",av[0]);

  return 0;
}


