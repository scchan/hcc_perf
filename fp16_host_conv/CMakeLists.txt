cmake_minimum_required(VERSION 3.16.0)

project(fp16_host_conv)

add_executable(fp16_host_conv c.cpp test.cpp)

SET(HOST_ARCH_FLAGS -march=x86-64 -msse2 -mfpmath=sse -m64)
SET(CLANG_RT_FLAGS  -rtlib=compiler-rt -unwindlib=libgcc)
SET(SAVE_TEMPS_FLAG -save-temps)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(fp16_host_conv PRIVATE ${CLANG_RT_FLAGS})
    target_link_options(fp16_host_conv PRIVATE ${CLANG_RT_FLAGS})
endif()

target_compile_options(fp16_host_conv PRIVATE "-g" ${SAVE_TEMPS_FLAG} ${HOST_ARCH_FLAGS})
