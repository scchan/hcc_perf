#include <hc.hpp>


#define N 1024

template<typename T>
void test_multiply_add() {

  hc::array_view<T,1> a0(N);
  hc::array_view<T,1> a1(N);
  hc::array_view<T,1> a2(N);
  hc::array_view<T,1> a3(N);

  hc::parallel_for_each(a0.get_extent(),[=] (hc::index<1> i) [[hc]] {
    a0[i] += a1[i]*a2[i];
  });


  hc::parallel_for_each(a0.get_extent(),[=] (hc::index<1> i) [[hc]] {
    a3[i] = a1[i]*a2[i]+a0[i];
  });


}


int main() {

  test_multiply_add<float>();
  test_multiply_add<double>();

  return 0;

}
