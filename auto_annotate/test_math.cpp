#include <iostream>
#include <hc.hpp>
#include <hc_math.hpp>

float compute2(float f)  {
  return cosf(f);
}

float compute(float f) [[hc,cpu]] {
  return cosf(f);
}

int main(int argc, char* argv[]) {

  float f = 1.0f;
  if (argc == 2) {
    f = (float)atof(argv[1]);
  }

  float cpu = compute(f);
  float cpu2 = cosf(f);

  hc::array_view<float,1> gpu(1);
  gpu[0] = f;

  hc::array_view<float,1> gpu2(1);
  gpu2[0] = f;

  hc::parallel_for_each(gpu.get_extent(), [=] (hc::index<1> i) [[hc]] {
    gpu[i] = compute(gpu[i]);
    gpu2[i] = compute2(gpu2[i]);
  });

  std::cout << "cpu: " << cpu << std::endl;
  std::cout << "cpu2: " << cpu2 << std::endl;
  std::cout << "gpu: " << gpu[0] << std::endl;
  std::cout << "gpu2: " << gpu2[0] << std::endl;

  return 0;
}
