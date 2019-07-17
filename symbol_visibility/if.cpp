#include "if.hpp"

foo::foo(int n) : _n(n) { 
}

foo::foo(float f) : _n((int)f) { 
}


foo* create_foo_int(int n) {
    return new foo(n);
}

foo* create_foo_float(float f) {
    return new foo(f);
}

void internal_work_on_foo(foo* f, int n) {
    f->add(n);
}