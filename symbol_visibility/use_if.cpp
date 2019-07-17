#include <cstdio>
#include "if.hpp"

int main() {
    foo* f = create_foo_int(1234);
    public_add_foo(f, 1);
    printf("foo: %d\n", f->get());
    delete f;
    return 0;
}