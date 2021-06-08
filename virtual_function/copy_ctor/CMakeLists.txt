cmake_minimum_required(VERSION 3.16.0)

project(copy_ctor)

add_executable(copy_ctor copy_ctor.cpp)
target_link_libraries(
    copy_ctor
    pthread
    ${HIP_LIBRARIES}
)

