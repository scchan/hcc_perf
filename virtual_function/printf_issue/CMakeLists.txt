cmake_minimum_required(VERSION 3.16.0)

project(printf_issue)

add_executable(f f.cpp)
target_link_libraries(
    f
    pthread
    ${HIP_LIBRARIES}
)

