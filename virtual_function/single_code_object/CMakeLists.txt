cmake_minimum_required(VERSION 3.16.0)

project(single_code_object)

add_executable(single_code_object single_code_object.cpp)
target_link_libraries(
    single_code_object
    pthread
    ${HIP_LIBRARIES}
)

