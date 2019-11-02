cmake_minimum_required(VERSION 2.8)
project(CarNN)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -std=c++17 -fopenmp")

link_libraries(Box2D sfml-system sfml-window sfml-graphics jsoncpp)

aux_source_directory(. SRC_LIST)
aux_source_directory(./entities SRC_LIST)
aux_source_directory(./neural SRC_LIST)

add_executable(${PROJECT_NAME} ${SRC_LIST})
