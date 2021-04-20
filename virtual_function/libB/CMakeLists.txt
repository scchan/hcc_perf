cmake_minimum_required(VERSION 3.16.0)

project(libB)

add_library(B SHARED B.cpp)
target_link_libraries(
    B
    pthread
    ${HIP_LIBRARIES}
)

