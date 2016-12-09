#include <hc.hpp>
#include <hc_math.hpp>

#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

#define N 1024

void test_cos() {

  std::vector<float> input(N);
  
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0f, (float) (2.0f * M_PI));
  auto gen = std::bind(distribution, generator);
  std::generate(input.begin(), input.end(), gen);

  hc::array_view<float,1> a0(N, input);
  hc::array_view<float,1> output_cos(N);

  hc::parallel_for_each(a0.get_extent(),[=] (hc::index<1> i) [[hc]] {
    output_cos[i] = cosf(a0[i]);
  }).wait();


  for (int i = 0; i < N; i++) {
    auto v = input[i];
    auto c = cosf(v);
    if (std::abs(c - output_cos[i]) >= std::abs(c * 0.01f)) {
      printf("cos(%f)   expected=%f, actual=%f\n", v, c, output_cos[i]);
    }
  }
}

int main() {
  test_cos();
  return 0;
}
