
#include <hc.hpp>


template<typename T>
hc::array_view<T,1>* create_av(const int n) {
  hc::array<T,1> a = hc::array<T,1>(hc::extent<1>(n), hc::accelerator().get_auto_selection_view());
  return new hc::array_view<T,1>(a);
}

int main() {
   float sf = 1234.0f;

  printf("address of sf: 0x%016llX\n",(unsigned long long)&sf);

#define NUM_ARRAY  100
  hc::array_view<float,1>* a_av[NUM_ARRAY];
  for (int i = 0; i < NUM_ARRAY; i++) {
    a_av[i] = create_av<float>(1);
  }
  for (int i = 0; i < NUM_ARRAY; i++) {
//    hc::copy(*a_av[i],&sf);
    hc::copy(&sf,*a_av[i]);
  }

  for (int i = 0; i < NUM_ARRAY; i++) {
    delete a_av[i];
  }
 
  return 0;
}
