cmake_minimum_required(VERSION 3.16.0)

project(identity)

add_executable(identity identity.cpp)
target_link_libraries(
    identity
    pthread
    ${HIP_LIBRARIES}
)

