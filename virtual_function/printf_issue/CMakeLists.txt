cmake_minimum_required(VERSION 3.16.0)

project(printf_issue)

add_executable(printf_issue printf_issue.cpp)
target_link_libraries(
    printf_issue
    pthread
    ${HIP_LIBRARIES}
)

