#include <cstdio>
#include <hc.hpp>

extern "C" int sum(hc::array_view<int,1>& input);

int main() {

  hc::array_view<int,1> av(64);
  for (int i = 0;i < 64; i++)
    av[i] = i;

  int s = sum(av);

  printf("sum: %d\n",s);

  return 0;
}
