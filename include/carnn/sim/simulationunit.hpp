#pragma once

#include <carnn/sim/entities/car.hpp>
#include <carnn/sim/fwd.hpp>
#include <carnn/sim/world.hpp>
#include <vector>

namespace sim
{
class SimulationUnit
{
	public:
	World world;

	std::vector<entities::Car*>        cars;
	std::vector<entities::Checkpoint*> checkpoints;
	entities::Body*                    wall;
	entities::CarCheckpointListener    contact_listener;

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

	std::vector<entities::Car*> cars;
};
} // namespace sim
