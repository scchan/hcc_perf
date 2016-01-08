
#include <iostream>
#include <hc.hpp>

#define NUM_ARRAY 512
#define ARRAY_BYTE_SIZE (512 * 1048576)

int main(int argc, char* argv[]) {
  int i;
  try {

    hc::array<int, 1>* arrays[NUM_ARRAY];
    for (i = 0; i < NUM_ARRAY; i++) {
      arrays[i] = new hc::array<int, 1>(ARRAY_BYTE_SIZE/sizeof(int));
    }

  } catch(std::exception e) {
    std::cout << e.what() << std::endl;
  }

}
