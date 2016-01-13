#include<iostream>
#include<typeinfo>
#include<hc.hpp>

template <typename T>
void print_type_size() {
  std::cout << "sizeof(" << typeid(T).name() << "): " << sizeof(T) << std::endl;

  hc::array_view<int,1> s(1);
  hc::parallel_for_each(hc::extent<1>(1), [=] (hc::index<1> i) [[hc]] {
    s[0] = sizeof(T);
  });
  std::cout << "sizeof(" << typeid(T).name() << ") on GPU: " << s[0] << std::endl;

}

int main() {

  print_type_size<int>();
  print_type_size<long>();
  print_type_size<long long>();

  return 0;
}
