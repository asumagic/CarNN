#include <carnn/sim/world.hpp>

#include <carnn/util/line.hpp>
#include <carnn/util/maths.hpp>

namespace sim
{
World::World(const b2Vec2 gravity) : _gravity{gravity}, _world{gravity} {}

World& World::step(const float speed, const int vel_it, const int pos_it)
{
	_world.Step(speed, vel_it, pos_it);
	return *this;
}

World& World::update()
{
	// Update bodies
	for (auto& b : _bodies)
		b->update();

	return *this;
}

World& World::render(sf::RenderTarget& target)
{
	// Render bodies
	for (auto& b : _bodies)
		b->render(target);

	return *this;
}

void World::update_view(sf::RenderTarget& target, sf::Vector2f origin, float czoom)
{
	sf::View new_view{
		util::lerp(target.getView().getCenter(), origin, 0.2f * 1.0f / 30.0f * 60.f),
		sf::Vector2f{target.getSize()} * czoom};
	target.setView(new_view);
}

b2World& World::get() { return _world; }
} // namespace sim
