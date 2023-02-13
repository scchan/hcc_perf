
extern "C"
__attribute__((noinline))
double half_to_double(_Float16 h) {
    double d = static_cast<double>(h);
    return d;
}

extern "C"
__attribute__((noinline))
_Float16 double_to_half(double d) {
    _Float16 h = static_cast<_Float16>(d);
    return h;
}

