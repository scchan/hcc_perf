
cmake_minimum_required(VERSION 3.16.0)

project(openMP_virtual_function_examples)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/rocm/lib/cmake")

set(AOMP OFF CACHE BOOL "Is a compiler from AOMP build?")
if (AOMP)
  set(AOMP_DIR "$ENV{HOME}/rocm/aomp")
  set(CMAKE_CXX_COMPILER "${AOMP_DIR}/bin/clang++" CACHE FILEPATH "" FORCE)
  set(CMAKE_C_COMPILER "${AOMP_DIR}/bin/clang" CACHE PATH "" FORCE)
else()
  set(CMAKE_CXX_COMPILER "/opt/rocm/llvm/bin/clang++" CACHE FILEPATH "" FORCE)
  set(CMAKE_C_COMPILER "/opt/rocm/llvm/bin/clang" CACHE FILEPATH "" FORCE)
endif()

set(CMAKE_CXX_STANDARD "17" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Force to code object v3
add_compile_options(-mcode-object-version=3)
add_link_options(-mcode-object-version=3)

# HIP headers and libs
set(HIP_PATH "/opt/rocm/hip")
include_directories("${HIP_PATH}/include")
link_directories("${HIP_PATH}/lib")
link_libraries(amdhip64)
add_definitions(-D__HIP_PLATFORM_AMD__=1)
link_directories("/opt/rocm/amdgcn/bitcode")


if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND AOMP)
  set(CMAKE_CXX_LINKER "env LD_LIBRARY_PATH=${AOMP_DIR}/lib-debug ${CMAKE_CXX_COMPILER}")
  set(CMAKE_C_LINKER "env LD_LIBRARY_PATH=${AOMP_DIR}/lib-debug ${CMAKE_C_COMPILER}")
endif()

set(OMP_OFFLOAD_TRIPLE "amdgcn-amd-amdhsa")
set(CLANG_OMP_TARGET_OFFLOAD_FLAGS "-target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=${OMP_OFFLOAD_TRIPLE} -Xopenmp-target=${OMP_OFFLOAD_TRIPLE}")
set(OMP_GPU "gfx900")
set(CLANG_OMP_TARGET_OFFLOAD_FLAGS "${CLANG_OMP_TARGET_OFFLOAD_FLAGS} -march=${OMP_GPU}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CLANG_OMP_TARGET_OFFLOAD_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CLANG_OMP_TARGET_OFFLOAD_FLAGS}")

add_subdirectory(simple)
