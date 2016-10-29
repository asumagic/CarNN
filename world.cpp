#include "world.hpp"

World::World(const b2Vec2 gravity) : _gravity{gravity}, _world{gravity} {}

World& World::step(const float dt, const int vel_it, const int pos_it)
{
	_world.Step(dt, vel_it, pos_it);
	return *this;
}

World& World::update()
{
	for (Body& b : _bodies)
		b.update();

	return *this;
}

World& World::render(sf::RenderTarget& target)
{
	for (Body& b : _bodies)
		b.render(target);

	return *this;
}

Body& World::add_body(const b2BodyDef bdef)
{
	_bodies.emplace_back(*this, bdef);
	return _bodies.back();
}

b2World& World::get()
{
	return _world;
}
