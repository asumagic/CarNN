#pragma once

#include <carnn/sim/entities/body.hpp>

class Car;

class Checkpoint : public Body
{
	public:
	Checkpoint(World& world, const b2BodyDef bdef, const bool do_render = true);

	std::size_t id;

	sf::Vector2f origin;
	sf::Vector2f p1, p2;
};
