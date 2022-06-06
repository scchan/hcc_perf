cmake_minimum_required(VERSION 3.16.0)

project(shuffle)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/rocm/lib/cmake")

set(HIP_CXX_COMPILER "/opt/rocm/llvm/bin/clang++")
set(CMAKE_CXX_COMPILER ${HIP_CXX_COMPILER} CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_STANDARD "17" CACHE STRING "" FORCE)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -offload_arch=gfx90a")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g1 -save-temps -O3")


find_package(hip REQUIRED CONFIG PATHS /opt/rocm)
include_directories(${HIP_INCLUDE_DIR})

message(STATUS "HIP_COMPILER: ${HIP_COMPILER}")
message(STATUS "HIP_CXX_COMPILER: ${HIP_CXX_COMPILER}")

add_executable(shuffle shuffle.cpp)
target_link_libraries(shuffle m ${HIP_LIBRARIES})

