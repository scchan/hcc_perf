
#include <cstdio>
#include <hc.hpp>

extern "C" int amdgcn_ds_bpermute(int index, int src) [[hc]];


#define LANE  16

void pattern_test() {

  hc::array_view<int,1>  in(64);
  hc::array_view<int,1> out(64);

#define TAG  255

  for (int i = 0; i < 64; i++) {
    for (int j = 0; j < 64; j++) {
      if (j == i)
        in[j] = TAG;
      else
        in[j] = 0;
    }

    hc::parallel_for_each(out.get_extent(), [=](hc::index<1> idx) [[hc]] {
      int data = in[idx];
      data = amdgcn_ds_bpermute(LANE<<2, data);
      out[idx] = data;
    }).wait();

    printf("\n\n");
    printf("Pattern test, tag in %d:", i);
    for (int j = 0; j < 64; j++) {
      if (out[i] != in[LANE]) {
        printf("out[%d]: expected=%d, actual=%d\n", i, in[LANE], out[i]);
      }
    }
  }
}



int main() {

  hc::array_view<int,1> out(64);
  int ind = 0;

#define LANE 16
  hc::parallel_for_each(out.get_extent(), [=](hc::index<1> i) [[hc]] {
    int data = i[0];
    data = amdgcn_ds_bpermute(LANE<<2, data);
    out[i] = data;
  }).wait();

  printf("Broadcast test: \n");
  for (int i = 0; i < 64; i++) {
    if (out[i]!=LANE) {
      printf("out[%d]: expected=%d, actual=%d\n",i,LANE,out[i]);
    }
  }


  hc::parallel_for_each(out.get_extent(), [=](hc::index<1> i) [[hc]] {
    int data = i[0];
    data = amdgcn_ds_bpermute(data*4, data);
    out[i] = data;
  }).wait();

  printf("\n\n");
  printf("Identity test: \n");
  for (int i = 0; i < 64; i++) {
    if (out[i]!=i) {
      printf("out[%d]: expected=%d, actual=%d\n",i,i,out[i]);
    }
  }

  pattern_test();
  return 0;
}
