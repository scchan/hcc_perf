#include <hc.hpp>

int main() {

  hc::array_view<int,1> s(1);
  hc::array_view<int,1> ab(1);

  int d = 100;

  hc::parallel_for_each(s.get_extent(),[&,s,ab,d](hc::index<1> i) [[hc]]  {
    int b = i[0] + *(&d);
    ab[i] = b;
    s[i] = sizeof(&b);
  });
  
  printf("sizeof private: %d\n",s[0]);
  printf("ab[0]: %d\n",ab[0]);

  return 0;
}
