cmake_minimum_required(VERSION 3.16.0)

project(normcdf)

add_executable(normcdf normcdf.cpp)
target_link_libraries(
    normcdf
    pthread
    m
    ${HIP_LIBRARIES}
)

