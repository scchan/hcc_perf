
cmake_minimum_required(VERSION 3.16.0)

project(openMP_virtual_function_examples)

set(CMAKE_CXX_COMPILER "/opt/rocm/llvm/bin/clang++" CACHE FILEPATH "" FORCE)
set(CMAKE_C_COMPILER "/opt/rocm/llvm/bin/clang" CACHE FILEPATH "" FORCE)

set(CMAKE_CXX_STANDARD "17" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# HIP headers and libs
set(HIP_PATH "/opt/rocm/hip")
include_directories("${HIP_PATH}/include")
link_directories("${HIP_PATH}/lib")
link_libraries(amdhip64)
add_definitions(-D__HIP_PLATFORM_AMD__=1)

set(OMP_OFFLOAD_TRIPLE "amdgcn-amd-amdhsa")
set(CLANG_OMP_TARGET_OFFLOAD_FLAGS "-target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=${OMP_OFFLOAD_TRIPLE} -Xopenmp-target=${OMP_OFFLOAD_TRIPLE}")
set(OMP_GPU "gfx900")
set(CLANG_OMP_TARGET_OFFLOAD_FLAGS "${CLANG_OMP_TARGET_OFFLOAD_FLAGS} -march=${OMP_GPU}")

set(CLANG_OMP_TARGET_OFFLOAD_FLAGS "${CLANG_OMP_TARGET_OFFLOAD_FLAGS} -v -g -save-temps")



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CLANG_OMP_TARGET_OFFLOAD_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CLANG_OMP_TARGET_OFFLOAD_FLAGS}")

add_subdirectory(simple)
