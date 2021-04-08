cmake_minimum_required(VERSION 3.16.0)

project(gpu_virtual_function)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/rocm/lib/cmake")

set(HIP_CXX_COMPILER "/opt/rocm/llvm/bin/clang++")
set(CMAKE_CXX_COMPILER ${HIP_CXX_COMPILER})
set(CMAKE_CXX_STANDARD 17)

find_package(hip REQUIRED CONFIG PATHS /opt/rocm)
include_directories(${HIP_INCLUDE_DIR})

if (FALSE)
find_path(
    HIP_CXX_COMPILER
    NAMES amdclang++ clang++ 
    HINTS /opt/rocm/bin /opt/rocm/llvm/bin
    REQUIRED)
set(CMAKE_CXX_COMPILER ${HIP_CXX_COMPILER})
find_path(
    HIP_C_COMPILER
    NAMES amdclang clang 
    HINTS /opt/rocm/bin /opt/rocm/llvm/bin
    REQUIRED)
set(CMAKE_C_COMPILER ${HIP_C_COMPILER})
endif()

message(STATUS "HIP_COMPILER: ${HIP_COMPILER}")
message(STATUS "HIP_CXX_COMPILER: ${HIP_CXX_COMPILER}")

add_executable(f f.cpp)

target_link_libraries(
    f
    pthread
    ${HIP_LIBRARIES}
)

