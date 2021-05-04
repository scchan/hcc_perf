cmake_minimum_required(VERSION 3.16.0)

project(exe_libA_liB)

include_directories("../libA" "../libB")

add_executable(exe_libA_libB exe_libA_libB.cpp)
add_dependencies(exe_libA_libB A B)
target_link_libraries(
    exe_libA_libB
    pthread
    ${HIP_LIBRARIES}
    A
    B
)

