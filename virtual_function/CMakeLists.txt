cmake_minimum_required(VERSION 3.16.0)

project(gpu_virtual_function)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/rocm/lib/cmake")

set(HIP_CXX_COMPILER "/opt/rocm/llvm/bin/clang++")
set(CMAKE_CXX_COMPILER ${HIP_CXX_COMPILER} CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_STANDARD "17" CACHE STRING "" FORCE)

find_package(hip REQUIRED CONFIG PATHS /opt/rocm)
include_directories(${HIP_INCLUDE_DIR})

message(STATUS "HIP_COMPILER: ${HIP_COMPILER}")
message(STATUS "HIP_CXX_COMPILER: ${HIP_CXX_COMPILER}")

include_directories("common/include")

add_subdirectory(printf_issue)
add_subdirectory(single_code_object)
add_subdirectory(libA)
add_subdirectory(libB)
