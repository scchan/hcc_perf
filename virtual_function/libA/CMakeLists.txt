cmake_minimum_required(VERSION 3.16.0)

project(libA)

add_library(A SHARED A.cpp)
target_link_libraries(
    A
    pthread
    ${HIP_LIBRARIES}
)

