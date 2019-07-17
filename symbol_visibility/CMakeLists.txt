
cmake_minimum_required(VERSION 3.00)

add_library(if SHARED
  if.cpp
  if_assist.cpp
)
set_target_properties(if PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(if PROPERTIES VISIBILITY_INLINES_HIDDEN 1)

add_executable(use_if
  use_if.cpp
)
target_link_libraries(use_if if)