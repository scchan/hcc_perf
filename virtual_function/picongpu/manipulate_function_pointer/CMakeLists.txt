cmake_minimum_required(VERSION 3.16.0)

project(manipulate_function_pointer)

add_executable(manipulate_function_pointer manipulate_function_pointer.cpp)
target_link_libraries(
    manipulate_function_pointer
    pthread
    ${HIP_LIBRARIES}
)

