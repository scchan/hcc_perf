
#include <stdio.h>
#include "StackTimer.h"


int foo(int a) {
  int t;
  STimer timer;
  timer = timer_start(__FUNCTION__);
  t = a+10;
  timer_stop(timer);
  return t;
}

int main(int argc, char* argv[]) {
  int b;
  STimer timer;
  timer = timer_start(__FUNCTION__);
  b = foo(argc);
  printf("b: %d\n",b);
  timer_stop(timer);
  return 0;
}
