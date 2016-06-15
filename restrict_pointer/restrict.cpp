#include <hc.hpp>


void foo(int* a, int* b, int* c, const int n) {
  for (int i = 0; i < n; i++) {
    a[i] = b[i] + c[i];
  }
}


#if 0

void bar(int* restrict a, int* restrict b, int* restrict c, const int n) {
  for (int i = 0; i < n; i++) {
    a[i] = b[i] + c[i];
  }
}

#endif



void  bar(int* __restrict__ a, int* __restrict__ b, int*  __restrict__ c, const int n) [[hc]] {
  for (int i = 0; i < n; i++) {
    a[i] = b[i] + c[i];
  }
}


#define THREADS          1024
#define N_PER_THREADS      64

int main() {

  hc::array_view<int, 1> a(THREADS*N_PER_THREADS);
  hc::array_view<int, 1> b(THREADS*N_PER_THREADS);
  hc::array_view<int, 1> c(THREADS*N_PER_THREADS);

  hc::array_view<int, 1> n(1);
  n[0] = N_PER_THREADS;

  const int n_per_threads = N_PER_THREADS;
 
  hc::parallel_for_each(hc::extent<1>(THREADS),[=](hc::index<1> i) [[hc]] {
    bar( &a[i[0] * N_PER_THREADS], &b[i[0] * N_PER_THREADS], &c[i[0] * N_PER_THREADS], n[0]);
  }).wait();


}



