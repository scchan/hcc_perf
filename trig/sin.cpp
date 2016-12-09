#include <hc.hpp>
#include <hc_math.hpp>

#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

#define N 1024

void test_sin() {

  std::vector<float> input(N);
  
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0f, (float) (2.0f * M_PI));
  auto gen = std::bind(distribution, generator);
  std::generate(input.begin(), input.end(), gen);

  hc::array_view<float,1> a0(N, input);
  hc::array_view<float,1> output_sin(N);

  hc::parallel_for_each(a0.get_extent(),[=] (hc::index<1> i) [[hc]] {
    output_sin[i] = sinf(a0[i]);
  }).wait();


  for (int i = 0; i < N; i++) {
    auto v = input[i];
    auto s = sinf(v);
    if (std::abs(s - output_sin[i]) >= std::abs(s * 0.01f)) {
      printf("sin(%f)   expected=%f, actual=%f\n", v, s, output_sin[i]);
    }
  }
}

int main() {
  test_sin();
  return 0;
}
