#include <cstdio>
#include <cstdlib>

extern "C" bool run_add_unsigned();
extern "C" bool run_add_int();

#ifdef LINK_DEP
int main() {
  bool i = run_add_int();
  printf("int test %s\n", i?"passed":"failed");

  bool u = run_add_unsigned();
  printf("unsigned test %s\n", u?"passed":"failed");

  return 0;
}
#endif

#ifdef DLOPEN

#include <dlfcn.h>


void dlopen_test(const char* lib_name, const char* func_name) {
  void* handle = dlopen(lib_name, RTLD_LAZY|RTLD_GLOBAL);
  if (handle==NULL) {
    printf("error when opening %s\n", lib_name);
    exit(1);
  }
  typedef decltype(run_add_int) f_type;
  f_type *func = reinterpret_cast<f_type*>(dlsym(handle, func_name));
  bool r = func();
  printf("%s test %s\n", func_name, r?"passed":"failed");
  dlclose(handle);
}


int main() {
  dlopen_test("./libsl0.so", "run_add_int");
  dlopen_test("./libsl1.so", "run_add_unsigned");
  return 0;
}
#endif



