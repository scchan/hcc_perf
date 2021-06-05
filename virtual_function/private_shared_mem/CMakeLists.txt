cmake_minimum_required(VERSION 3.16.0)

project(private_shared_mem)

add_executable(private_shared_mem private_shared_mem.cpp)
target_link_libraries(
    private_shared_mem
    pthread
    ${HIP_LIBRARIES}
)

