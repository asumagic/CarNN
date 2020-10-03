cmake_minimum_required(VERSION 2.8)
project(CarNN)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -std=c++17 -fopenmp")

link_libraries(box2d sfml-system sfml-window sfml-graphics jsoncpp fmt)

add_executable(${PROJECT_NAME}
	entities/car.cpp
	entities/checkpoint.cpp
	entities/wheel.cpp
	neural/mutator.cpp
	neural/network.cpp
	neural/neuron.cpp
	body.cpp
	main.cpp
	randomutil.cpp
	world.cpp
)
