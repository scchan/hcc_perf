cmake_minimum_required(VERSION 3.16.0)

project(libA)

add_library(A SHARED A.cpp)
target_link_libraries(
    A
    pthread
    ${HIP_LIBRARIES}
)


add_library(A_rdc SHARED A.cpp)
target_link_libraries(
    A_rdc
    pthread
    ${HIP_LIBRARIES}
)
target_compile_options(A_rdc PRIVATE -fgpu-rdc)
target_link_options(A_rdc PRIVATE -fgpu-rdc)