#pragma once

#include "entities/car.hpp"
#include "entities/checkpoint.hpp"
#include "world.hpp"

#include <thread>

class SimulationUnit
{
	public:
	World world;

	std::vector<Car*>        cars;
	std::vector<Checkpoint*> checkpoints;
	Body*                    wall;
	CarCheckpointListener    contact_listener;

	std::size_t ticks_elapsed   = 0;
	float       seconds_elapsed = 0.0f;
};

class Simulation
{
	public:
	Simulation();

	void load_map(const char* fname);
	void load_checkpoints(const char* fname);
	void init_cars();

	SimulationUnit& optimal_unit();

	std::vector<SimulationUnit> units;

	std::vector<sf::Vertex> wall_vertices;
	b2Vec2                  car_origin;

	sf::VertexArray checkpoint_vertices;

	std::vector<Car*> cars;
};
