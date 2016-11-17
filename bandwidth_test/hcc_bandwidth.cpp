#include <iostream>
#include <iomanip>
#include <hc.hpp>
#include <StackTimer.hpp>

constexpr int size_KB = 1024;
constexpr int size_MB = size_KB * 1024;
constexpr int size_GB = size_MB * 1024;

void array_write_bandwidth(hc::accelerator& a, const unsigned int size, const unsigned int iter, TimerEventQueue& eventQueue) {
  hc::array<char,1> buffer(size, a.get_default_view());
  char* host_buffer = (char*)malloc(size);
  for (int i = 0; i < iter; i++) {
    SimpleTimer timer(eventQueue, __FUNCTION__);
    hc::completion_future future = hc::copy_async(host_buffer, host_buffer+size, buffer);
    future.wait();
  }
  free(host_buffer);
}

void run_benchmark(hc::accelerator& acc) {
  static int count = 0;
  std::cout << std::endl;
  std::cout << "accelerator #" << count << std::endl;
  count++;


  constexpr int column_width = 30;
  std::cout << std::setw(column_width);
  std::cout << "(Size)";
  std::cout << std::setw(column_width);
  std::cout << "(Average Time (ms))";
  std::cout << std::setw(column_width);
  std::cout << "(Average Bandwidth (GB/s))";
  std::cout << std::endl;

  std::cout << "===============================================================================================" << std::endl;
  
  std::vector<int> sizes = { 
                                 1 * size_MB
                             ,  16 * size_MB
                             ,  32 * size_MB
                             , 128 * size_MB
                             , 256 * size_MB
                             , 512 * size_MB
                             ,   1 * size_GB
                           };


  for (auto s = sizes.begin(); s!=sizes.end(); s++) {
    TimerEventQueue eventQueue;
    array_write_bandwidth(acc, *s, 10, eventQueue);

    std::cout << std::setw(column_width-4);
    std::cout << std::setprecision(0);
    if (*s >= size_MB) {
      std::cout << *s/(float)size_MB << " MB";
    } else if (*s >= size_KB) {
      std::cout << *s/(float)size_KB << " KB";
    } else {
      std::cout << *s << " B";
    }

    std::cout << std::setw(column_width-4);
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    std::cout << eventQueue.getAverageTime();

    std::cout << std::setw(column_width-4);
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    std::cout << ((double)*s/double(1024*1024*1024))/(eventQueue.getAverageTime() / 1000.0);
    
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  std::vector<hc::accelerator> all_accelerators = hc::accelerator::get_all();
  for (auto acc = all_accelerators.begin(); acc != all_accelerators.end(); acc++) {
    if (acc->is_hsa_accelerator()) {
      run_benchmark(*acc);
    }
  }
  return 0;
}


