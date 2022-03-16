
#include <cstdio>
#include <cstdlib>
#include <iostream>

int main() {
 
    std::cout << "__STDCPP_DEFAULT_NEW_ALIGNMENT__ = " << __STDCPP_DEFAULT_NEW_ALIGNMENT__ << std::endl;

    auto i = new int{};
    printf("alignof(i) = %lu\n", alignof(i));
    delete i;
    return 0;
}