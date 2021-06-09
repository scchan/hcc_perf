cmake_minimum_required(VERSION 3.16.0)

project(simple)

add_executable(simple simple.cpp)
target_link_libraries(
    simple
    pthread
    ${HIP_LIBRARIES}
)

