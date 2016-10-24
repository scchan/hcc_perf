#include <cstdio>
#include <hc.hpp>

//#define GLOBAL_VAR_BUG 1

#ifdef GLOBAL_VAR_BUG
__attribute__((hc)) int foo[1] = { 1234 };
#else
__attribute__((address_space(1))) int foo[1] = { 1234 };
#endif

#define N 1

template<typename T>
int test_global_var() {
  hc::array_view<T,1> a0(N);
  hc::parallel_for_each(a0.get_extent(),[=] (hc::index<1> i) [[hc]] {
    if (i[0] == 0) {
      foo[i[0]]++;
      a0[i] = (T)foo[i[0]];
    }
  }).wait();
  return (int)a0[0];
}


int main() {
  int n0 = test_global_var<float>();
  int n1 = test_global_var<double>();

  int errors = 0;
  if (n0!=1235)
    errors++;
  if (n1!=1236)
    errors++;

  return errors;
}
