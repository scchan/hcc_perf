#include <vector>
#include <iostream>
#include <string>
#include <cmath>

//__attribute__((visibility("default")))
std::string global_string;

void print(int i) {
  std::cout << "i: " << i << std::endl;
}

static void print_string(std::string s) {
  std::cout << "s: " << s << std::endl;
}

#if 1
static std::string convert_to_string(int i) {
  //std::string s = std::to_string(i);
  std::string s;
  return s;
}
#endif

__attribute__((visibility("default")))
std::string convert_to_string2(int i) {
  std::string s = std::to_string(i);
  return s;
}


__attribute__((visibility("default")))
float foo(int i) {
  //std::cout << "i: " << i << std::endl;
  print(std::abs(i));
  //print_string("Hello World\n");
  return sinf((float)i);
}




