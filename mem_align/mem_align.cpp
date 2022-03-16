
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
    std::cout << "__STDCPP_DEFAULT_NEW_ALIGNMENT__ = " << __STDCPP_DEFAULT_NEW_ALIGNMENT__ << std::endl;

    std::vector<int*> vi;
    for (int i = 0; i < 16; ++i) {
        vi.push_back(new int{});
    }
    
    for(auto i : vi) {
        auto j = reinterpret_cast<std::intptr_t>(i);
        printf("j %% 8 = %lu\n", j%8);
        printf("j %% 16 = %lu\n", j%16);
        printf("j %% 32 = %lu\n", j%32);
        delete i;
    }
    return 0;
}