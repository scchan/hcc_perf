#include <iostream>
#include <iomanip>
#include <string>

#include <hc.hpp>
#include <hc_am.hpp>
#include <StackTimer.hpp>

constexpr int size_KB = 1024;
constexpr int size_MB = size_KB * 1024;
constexpr int size_GB = size_MB * 1024;

enum BenchmarkKind {
   ARRAY_READ_BENCHMARK = 0
  ,ARRAY_WRITE_BENCHMARK
  ,BUFFER_READ_BENCHMARK
  ,BUFFER_WRITE_BENCHMARK
  ,BENCHMARKKIND_LAST
};

const char* BenchmarkName[] = {
  "Array Read"
  ,"Array Write"
  ,"Buffer Read"
  ,"Buffer Write"
  ,nullptr
};


void buffer_bandwidth(hc::accelerator& a, const unsigned int size, const unsigned int iter, TimerEventQueue& eventQueue, bool pinnedHostBuffer, BenchmarkKind kind) {

  if (!pinnedHostBuffer) {
    // skip for now
    return;
  }

  char* host_buffer;
  if (pinnedHostBuffer) {
    host_buffer = hc::am_alloc(size, a, amHostPinned);
  }
  else {
    host_buffer = (char*)malloc(size);
  }

  char* device_buffer = hc::am_alloc(size, a, 0);

  char* source_buffer = nullptr;
  char* dest_buffer = nullptr;

  switch(kind) {
    case BUFFER_READ_BENCHMARK:
     source_buffer = device_buffer;
     dest_buffer = host_buffer;
     break;
    case BUFFER_WRITE_BENCHMARK:  
     source_buffer = host_buffer;
     dest_buffer = device_buffer;
     break;
    default:
     exit(1);
  };
  
  hc::accelerator_view acc_view = a.get_default_view();
  for (int i = 0; i < iter; i++) {
    SimpleTimer timer(eventQueue, __FUNCTION__);
    hc::completion_future future = acc_view.copy_async(source_buffer, dest_buffer, size);
    future.wait();
  }

  if (pinnedHostBuffer) {
    hc::am_free(host_buffer);
  }
  else {
    free(host_buffer);
  }
  hc::am_free(device_buffer);
}


void array_write_bandwidth(hc::accelerator& a, const unsigned int size, const unsigned int iter, TimerEventQueue& eventQueue, bool pinnedHostBuffer) {
  hc::array<char,1> buffer(size, a.get_default_view());
  char* host_buffer = nullptr;
  if (pinnedHostBuffer) {
    host_buffer = hc::am_alloc(size, a, amHostPinned);
  }
  else {
    host_buffer = (char*)malloc(size);
  }

  for (int i = 0; i < iter; i++) {
    SimpleTimer timer(eventQueue, __FUNCTION__);
    hc::completion_future future = hc::copy_async(host_buffer, host_buffer+size, buffer);
    future.wait();
  }

  if (pinnedHostBuffer) {
    hc::am_free(host_buffer);
  }
  else {
    free(host_buffer);
  }
}



void array_read_bandwidth(hc::accelerator& a, const unsigned int size, const unsigned int iter, TimerEventQueue& eventQueue, bool pinnedHostBuffer) {
  hc::array<char,1> buffer(size, a.get_default_view());
  char* host_buffer = nullptr;
  if (pinnedHostBuffer) {
    host_buffer = hc::am_alloc(size, a, amHostPinned);
  }
  else {
    host_buffer = (char*)malloc(size);
  }

  for (int i = 0; i < iter; i++) {
    SimpleTimer timer(eventQueue, __FUNCTION__);
    hc::completion_future future = hc::copy_async(buffer, host_buffer);
    future.wait();
  }

  if (pinnedHostBuffer) {
    hc::am_free(host_buffer);
  }
  else {
    free(host_buffer);
  }
}



void run_benchmark(hc::accelerator& acc, const int acc_id, const BenchmarkKind benchKind, const bool pinnedHostBuffer) {
  std::cout << std::endl;
  std::cout << "accelerator #" << acc_id << std::endl;
  std::cout << BenchmarkName[benchKind] << ",";
  std::cout << static_cast<const char*>(pinnedHostBuffer?"Pinned":"Unpinned") << " Host Memory" << std::endl;;
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

    switch(benchKind) {
      case ARRAY_READ_BENCHMARK:
        array_read_bandwidth(acc, *s, 10, eventQueue, pinnedHostBuffer);
        break;
      case ARRAY_WRITE_BENCHMARK:
        array_write_bandwidth(acc, *s, 10, eventQueue, pinnedHostBuffer);
        break;
      case BUFFER_READ_BENCHMARK:
      case BUFFER_WRITE_BENCHMARK:
        buffer_bandwidth(acc, *s, 10, eventQueue, pinnedHostBuffer, benchKind);
        break;
      default:
        exit(1);
    };

    if (eventQueue.getNumEvents() == 0) {
      std::cout << "Skipped, unsupported test" << std::endl;
      continue;
    }


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

  amdtScopedMarker fmarker =   HC_SCOPE_MARKER ;

  std::vector<hc::accelerator> all_accelerators = hc::accelerator::get_all();

  
  for (int k = 0; k < BENCHMARKKIND_LAST; k++) {
    int acc_id = 0;
    for (auto acc = all_accelerators.begin(); acc != all_accelerators.end(); acc++, acc_id++) {
      if (acc->is_hsa_accelerator()) {
        run_benchmark(*acc, acc_id, static_cast<BenchmarkKind>(k), false);
      }
    }
    
    acc_id = 0;
    for (auto acc = all_accelerators.begin(); acc != all_accelerators.end(); acc++, acc_id++) {
      if (acc->is_hsa_accelerator()) {
        run_benchmark(*acc, acc_id, static_cast<BenchmarkKind>(k), true);
      }
    }
  }

#if 0


  for (auto acc = all_accelerators.begin(); acc != all_accelerators.end(); acc++) {
    if (acc->is_hsa_accelerator()) {
      run_benchmark(*acc, ARRAY_READ_BENCHMARK, false);
    }
  }

  for (auto acc = all_accelerators.begin(); acc != all_accelerators.end(); acc++) {
    if (acc->is_hsa_accelerator()) {
      run_benchmark(*acc, ARRAY_READ_BENCHMARK, true);
    }
  }

  for (auto acc = all_accelerators.begin(); acc != all_accelerators.end(); acc++) {
    if (acc->is_hsa_accelerator()) {
      run_benchmark(*acc, ARRAY_WRITE_BENCHMARK, false);
    }
  }

  for (auto acc = all_accelerators.begin(); acc != all_accelerators.end(); acc++) {
    if (acc->is_hsa_accelerator()) {
      run_benchmark(*acc, ARRAY_WRITE_BENCHMARK, true);
    }
  }





  for (auto acc = all_accelerators.begin(); acc != all_accelerators.end(); acc++) {
    if (acc->is_hsa_accelerator()) {
      run_benchmark(*acc, BUFFER_READ_BENCHMARK, false);
    }
  }

  for (auto acc = all_accelerators.begin(); acc != all_accelerators.end(); acc++) {
    if (acc->is_hsa_accelerator()) {
      run_benchmark(*acc, BUFFER_READ_BENCHMARK, true);
    }
  }

  for (auto acc = all_accelerators.begin(); acc != all_accelerators.end(); acc++) {
    if (acc->is_hsa_accelerator()) {
      run_benchmark(*acc, BUFFER_WRITE_BENCHMARK, false);
    }
  }

  for (auto acc = all_accelerators.begin(); acc != all_accelerators.end(); acc++) {
    if (acc->is_hsa_accelerator()) {
      run_benchmark(*acc, BUFFER_WRITE_BENCHMARK, true);
    }
  }
#endif

  return 0;
}

