#pragma once

#include "../body.hpp"

class Car;

class Checkpoint : public Body
{
	public:
	Checkpoint(World& world, const b2BodyDef bdef, const bool do_render = true);

	std::size_t id;

	sf::Vector2f origin;
};
