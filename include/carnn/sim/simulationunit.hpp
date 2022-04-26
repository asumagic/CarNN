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

struct MapSettings
{
	std::string map_path;
	std::string checkpoint_path;
	bool flip = false;
};

class Simulation
{
	public:
	Simulation(MapSettings settings);

	void load_map();
	void load_checkpoints();
	void init_cars();

	SimulationUnit& optimal_unit();

	MapSettings settings;

	std::vector<SimulationUnit> units;

	std::vector<sf::Vertex> wall_vertices;
	b2Vec2                  car_origin;

	sf::VertexArray checkpoint_vertices;

	std::vector<entities::Car*> cars;
};
} // namespace sim
