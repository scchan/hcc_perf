
#include <iostream>
#include <cstdint>
#include <hc.hpp>

#define NUM_ARRAY 512

//#define ARRAY_BYTE_SIZE (512 * 1048576)
#define ARRAY_BYTE_SIZE  (1024)

uint64_t allocate_buffer(uint64_t size) {
  try {
    hc::array<int, 1>* array = new hc::array<int, 1>(size/sizeof(int));
    delete array;
  } catch(...) {
    exit(1);
  }
  return size;
}

uint64_t find_max_buffer_size() {
  uint64_t increment = 128 * 1048576;
  uint64_t size = 0;
  while (true) {
    size+=increment;
    uint64_t returned_size = allocate_buffer(size);
    if (size == returned_size) {
      std::cout << size << " bytes buffer allocation sucessful" << std::endl;
    }
    else {
      return size;
    }
#if 0
    try {
      hc::array<int, 1>* array = new hc::array<int, 1>(size/sizeof(int));
      std::cout << size << " bytes buffer allocation sucessful" << std::endl;
      size<<=1;
      delete array;
    } catch(...) {
      break;
    }
#endif

  }
  return size;
}


int main(int argc, char* argv[]) {

#if 0
  int num_allocations = 1;
  try {
    hc::array<int, 1>* arrays[NUM_ARRAY];
    for (int i = 0; i < NUM_ARRAY; i++) {
      arrays[i] = new hc::array<int, 1>(ARRAY_BYTE_SIZE/sizeof(int));
      num_allocations++;
    }
  } catch(std::exception e) {
    std::cout << num_allocations << " successful allocations" << std::endl;
    std::cout << e.what() << std::endl;
    exit(1);
  }
  std::cout << num_allocations << " successful allocations" << std::endl;
#endif 

  uint64_t max_size = find_max_buffer_size();
  std::cout << "max buffer size: " << max_size << std::endl;
  
  return 0;
}
