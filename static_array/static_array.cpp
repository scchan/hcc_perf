
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>

// header file for the hc API
#include <hc.hpp>

#define N  (1024 * 500)

const double dd[]  = { 100.0, 101.0, 102.0, 103.0 };

int main() {

  double host_y[N];

  double host_result_y[N];
  for (int i = 0; i < N; i++) {
    static double d[] = { 200.0, 201.0, 202.0, 203.0 };
    host_result_y[i] = d[i%4] + dd[i%4];
  }

  hc::array_view<double, 1> y(N, host_y);

  hc::parallel_for_each(hc::extent<1>(N)
                      , [=](hc::index<1> i) [[hc]] {
    static double d[] = { 200.0, 201.0, 202.0, 203.0 };
    y[i] = d[i[0]%4] + dd[i[0]%4];
  });
   
  // verify the results
  int errors = 0;
  for (int i = 0; i < N; i++) {
    if (host_result_y[i] != y[i]) {
      errors++;
    }
    std::cout << "y[" << i << "] = " << y[i] << std::endl;
  }
  std::cout << errors << " errors" << std::endl;

  return errors;
}
