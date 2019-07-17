#include <cstdio>

#define __PUBLIC_API__     __attribute__((visibility("default")))
#define __INTERNAL_API__   __attribute__((visibility("hidden")))

class foo {
    public:
        foo() : _n(0) { }
        __PUBLIC_API__ foo(int);
        foo(float);
        void add(int n) { _n+= n; }
        int get() { return _n; }
    private:
        int _n;
};

__INTERNAL_API__
void internal_work_on_foo(foo*, int);


#ifdef USE_VISIBILITY_PRAGMA
#pragma GCC visibility push (default)

foo* create_foo_int(int);
foo* create_foo_float(float);
void public_add_foo(int);

#pragma GCC visibility pop
#else

__PUBLIC_API__ 
foo* create_foo_int(int);

__PUBLIC_API__ 
foo* create_foo_float(float);

__PUBLIC_API__ 
void public_add_foo(foo*, int);

#endif