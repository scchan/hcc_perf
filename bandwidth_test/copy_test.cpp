#include <cstdlib>
#include <string>
#include <iostream>
#include <iomanip>
#include <memory>
#include <sstream>
#include<utility>

#include <hc.hpp>
#include <hc_am.hpp>

#include "StackTimer.hpp"


class mem_allocator {
  public:
    mem_allocator(const std::string& mem_type_name) : mem_type_name(mem_type_name) {}
    virtual void* allocate(size_t size){ return nullptr; };
    virtual void free(void* ptr) {};
    virtual ~mem_allocator() {};
    const std::string& get_mem_type_name() { return mem_type_name; }
  protected:
    const std::string& mem_type_name;
};

class host_mem_allocator : public mem_allocator {
  public:
    host_mem_allocator() : mem_allocator(mem_type_name) {}
    void* allocate(size_t size) { return std::malloc(size); }
    void free(void* ptr) { std::free(ptr); }
  private:
    static const std::string mem_type_name; 
};
const std::string host_mem_allocator::mem_type_name = std::string("host_mem");


class am_device_mem_allocator : public mem_allocator {
  public:
    am_device_mem_allocator(hc::accelerator acc) : mem_allocator(mem_type_name), acc(acc) {} 
    void* allocate(size_t size) { return hc::am_alloc(size, acc, 0); }
    void free(void* ptr) { hc::am_free(ptr); }
  private:
    hc::accelerator acc;
    static const std::string mem_type_name;
};
const std::string am_device_mem_allocator::mem_type_name = std::string("device_mem");

class am_pinned_mem_allocator : public mem_allocator {
  public:
    am_pinned_mem_allocator(hc::accelerator acc) : mem_allocator(mem_type_name), acc(acc) {} 
    void* allocate(size_t size) { return hc::am_alloc(size, acc, amHostPinned); }
    void free(void* ptr) { hc::am_free(ptr); }
  private:
    hc::accelerator acc;
    static const std::string mem_type_name;
};
const std::string am_pinned_mem_allocator::mem_type_name = std::string("pinned_mem");

template<class T>
class buffer {
  public:
    buffer() = delete;
    buffer(T allocator, size_t size) : allocator(allocator) {
      data = allocator.allocate(size);
    }
    ~buffer() { allocator.free(data); }
    void* get_data() { return data; }
  private:
    void* data;
    T allocator;
};

class host_buffer : public buffer<host_mem_allocator> { 
  public:
    host_buffer(size_t size) : buffer(host_mem_allocator(), size) {}
};

class device_buffer : public buffer<am_device_mem_allocator> { 
  public:
    device_buffer(hc::accelerator& acc, size_t size) : buffer(am_device_mem_allocator(acc), size) {}
};

class pinned_buffer: public buffer<am_pinned_mem_allocator> {
  public:
    pinned_buffer(hc::accelerator& acc, size_t size) : buffer(am_pinned_mem_allocator(acc), size) {}
};

TimerEventQueue event_queue;

template <class src_allocator_t, class dest_allocator_t>
void run_copy_test(src_allocator_t& src_allocator
                   , dest_allocator_t& dest_allocator
                   , const unsigned int iter) {

  hc::accelerator acc;
  auto acc_view = acc.get_default_view();

  size_t sizes_mb[] = { 128, 512, 1024 }; 

  

  for (auto size_mb : sizes_mb) {
    const auto buffer_size = size_mb * (1024*1024);

    std::stringstream ss;
    ss << src_allocator.get_mem_type_name() <<  " to " 
       << dest_allocator.get_mem_type_name()
       << "(" << size_mb << " MB)";

    auto src_buffer = buffer<src_allocator_t>(src_allocator, buffer_size); 
    auto dest_buffer = buffer<dest_allocator_t>(dest_allocator, buffer_size); 
    for (unsigned int i = 0; i < iter; ++i) {

      SimpleTimer timer(event_queue, ss.str().c_str());

      acc_view.copy(src_buffer.get_data()
                    , dest_buffer.get_data()
                    , buffer_size);
    }

    const double average = event_queue.getAverageTime(ss.str().c_str());
    const double bandwidth = size_mb/(average/1000.0);

    std::cout << src_allocator.get_mem_type_name()
              << " to " << dest_allocator.get_mem_type_name() << ","
              << "\t size(MB)=" << size_mb
              << "\t average_time(ms)="
              << std::fixed << std::setprecision(4)
              << average
              << "\t bandwidth(MB/s)="
              << bandwidth
              << std::endl;
  }
}


int main() {

  hc::accelerator acc;
  auto acc_view = acc.get_default_view();

#if 0
  std::vector<std::unique_ptr<mem_allocator>> allocators ;
  allocators.push_back(std::unique_ptr<mem_allocator>(new host_mem_allocator()));
  allocators.push_back(std::unique_ptr<mem_allocator>(new am_device_mem_allocator(acc)));

  for (auto& src_allocator : allocators) {
    for (auto& dest_allocator : allocators) {
      if (src_allocator == dest_allocator)
        continue;
      run_copy_test(*src_allocator, *dest_allocator, 10);
    }
  }
#endif

  host_mem_allocator cpu_mem_allocator;
  am_device_mem_allocator device_mem_allocator(acc);
  am_pinned_mem_allocator pinned_mem_allocator(acc);
 
  run_copy_test(cpu_mem_allocator, device_mem_allocator, 10);
  run_copy_test(device_mem_allocator, cpu_mem_allocator, 10);

  run_copy_test(pinned_mem_allocator, device_mem_allocator, 10);
  run_copy_test(device_mem_allocator, pinned_mem_allocator, 10);


  return 0;
}
