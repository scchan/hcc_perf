cmake_minimum_required(VERSION 3.16.0)

project(shared_library)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/rocm/lib/cmake")

set(HIP_CXX_COMPILER "/opt/rocm/llvm/bin/clang++")
set(CMAKE_CXX_COMPILER ${HIP_CXX_COMPILER} CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_STANDARD "17" CACHE STRING "" FORCE)

find_package(hip REQUIRED CONFIG PATHS /opt/rocm)
include_directories(${HIP_INCLUDE_DIR})

message(STATUS "HIP_COMPILER: ${HIP_COMPILER}")
message(STATUS "HIP_CXX_COMPILER: ${HIP_CXX_COMPILER}")

include_directories("libA")

add_subdirectory(libA)

add_executable(run_shared_lib run_shared_lib.cpp)
add_dependencies(run_shared_lib A)
target_link_libraries(
    run_shared_lib
    pthread
    ${HIP_LIBRARIES}
    A
)

add_executable(run_shared_lib.rdc run_shared_lib.cpp)
target_compile_options(run_shared_lib.rdc PRIVATE -fgpu-rdc)
target_link_options(run_shared_lib.rdc PRIVATE -fgpu-rdc)
add_dependencies(run_shared_lib.rdc A_rdc)
target_link_libraries(
    run_shared_lib.rdc
    pthread
    ${HIP_LIBRARIES}
    A_rdc
)
