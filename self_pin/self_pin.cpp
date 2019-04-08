
#include <vector>
#include <hc.hpp>
#include <hc_am.hpp>

#define DEBUG 1

void pin_memory(void* p, size_t s) {
    hc::accelerator acc;
    auto peers = acc.get_peers();
    peers.push_back(acc);
    am_status_t status;
#ifdef DEBUG
    printf("trying to pin %p\n",p);
#endif
    status = hc::am_memory_host_lock(acc, p, s
                                , peers.data(), peers.size());
    assert(status == AM_SUCCESS);
}

void unpin_memory(void* p) {
  hc::accelerator acc;
  am_status_t status;
  status = am_memory_host_unlock(acc, p);
  assert(status == AM_SUCCESS);
}

template <typename T>
class Register {
public:
  void add(T* p) { _r.push_back(p); }
  ~Register() {
    for(auto p : _r) {
      unpin_memory(p);
      delete p;
    }
  }
private:
  std::vector<T*> _r;
};

template <typename T>
class PrePinned {

public:

  ~PrePinned() [[cpu]] {
    hc::accelerator acc;
    am_status_t status;
    status = hc::am_memory_host_unlock(acc, this);
    assert(status == AM_SUCCESS);
  }

  static PrePinned<T>* get_new() {
    auto n = new PrePinned<T>();
    hc::AmPointerInfo info;
    auto status = hc::am_memtracker_getinfo(&info, n);
    assert(status == AM_SUCCESS);
    n->pinned = reinterpret_cast<PrePinned<T>*>(info._devicePointer);
    r.add(n);
    return n->pinned;
  };

  T data;

private:
  PrePinned() [[cpu]] {
#if 0
    hc::accelerator acc;
    auto peers = acc.get_peers();
    peers.push_back(acc);
    am_status_t status;
    status = hc::am_memory_host_lock(acc, this, sizeof(PrePinned<T>)
                                , peers.data(), peers.size());
    assert(status == AM_SUCCESS);
#else
    pin_memory(this, sizeof(*this));
#endif

    
    r.add(this);
  }
  PrePinned<T>* pinned;

  static Register<PrePinned<T>> r;
};

template <typename T>
Register<PrePinned<T>> PrePinned<T>::r;

#if 0
#ifdef __HCC_CPU__
PrePinned<int> bar;
#else
[[hc]] PrePinned<int> bar;
#endif
#else
//PrePinned<int> bar;
#endif


PrePinned<int>* const bar = PrePinned<int>::get_new();
PrePinned<int>* const foobar = PrePinned<int>::get_new();

int main() {


  bar->data = 1234;
  hc::array_view<int,1> av(1);

  printf("before bar.data: %d\n", bar->data);

  PrePinned<int>* const barp = bar;

  int foo = 1234;
  pin_memory(&foo, sizeof(foo));
  int* foop = &foo;

  hc::parallel_for_each(hc::extent<1>(1), [=] (hc::index<1> i) [[hc]] {
   // av[0] = barp->data;
   av[0] = *foop;
   *foop = *foop+1;
  }).wait();

  printf("after bar.data: %d\n", bar->data);
  printf("av[0]:%d\n",av[0]);
  printf("*foop: %d\n",*foop);

  return 0;
}


