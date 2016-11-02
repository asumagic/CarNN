project(CarNN)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto -fomit-frame-pointer -fstrict-aliasing")

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -std=c++14")

link_libraries(Box2D sfml-system sfml-window sfml-graphics)

set(CMAKE_CXX_COMPILER "clang++-4.0")

cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
aux_source_directory(./entities SRC_LIST)
aux_source_directory(./neural SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})
