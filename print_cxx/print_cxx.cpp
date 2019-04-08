#include <string>
#include <iostream>

int main() {
  std::string s("Hello");
  std::string w(" World!\n");
  s.insert(s.end(), w.cbegin(), w.cend());
  
  std::cout << s;
  std::cout << "__cplusplus: " << __cplusplus << std::endl;
  std::cout << "_GLIBCXX_USE_CXX11_ABI: " << _GLIBCXX_USE_CXX11_ABI << std::endl;

  return 0;
}
