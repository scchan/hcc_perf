cmake_minimum_required(VERSION 3.16.0)

project(static_libs)
set(HIP_CXX_COMPILER "/opt/rocm/llvm/bin/clang++")
set(CMAKE_CXX_COMPILER ${HIP_CXX_COMPILER} CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_STANDARD "17" CACHE STRING "" FORCE)

message(STATUS "HIP_COMPILER: ${HIP_COMPILER}")
message(STATUS "HIP_CXX_COMPILER: ${HIP_CXX_COMPILER}")

set(GPU_ARCH_OPTS "--offload-arch=gfx900")
set(HIP_COMPILE_OPTS "-xhip")
set(HIP_LINK_RT_OPTS "--hip-link")
set(HIP_RDC "-fgpu-rdc")

add_library(a STATIC a.cpp)
target_compile_options(a PRIVATE ${HIP_COMPILE_OPTS} ${GPU_ARCH_OPTS} ${HIP_RDC})

add_executable(p p.cpp)
target_compile_options(p PRIVATE ${HIP_COMPILE_OPTS} ${GPU_ARCH_OPTS} ${HIP_RDC})
target_link_libraries(p a)
target_link_options(p PRIVATE ${HIP_LINK_RT_OPTS} ${GPU_ARCH_OPTS} ${HIP_RDC})