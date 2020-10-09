#pragma once

#include "entities/checkpoint.hpp"
#include "world.hpp"

#include <thread>

class SimulationUnit
{
	public:
	World world;

	std::vector<Checkpoint*> checkpoints;
};

class Simulation
{
	public:
	Simulation() : units(std::thread::hardware_concurrency()) {}

	SimulationUnit& optimal_unit()
	{
		const auto it = std::min_element(units.begin(), units.end(), [](const auto& a, const auto& b) {
			return a.world.body_count() < b.world.body_count();
		});

		return *it;
	}

	std::vector<SimulationUnit> units;
};
