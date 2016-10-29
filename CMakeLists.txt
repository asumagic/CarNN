project(CarNN)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -O2 -std=c++14")

link_libraries(Box2D sfml-system sfml-window sfml-graphics)

cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})
