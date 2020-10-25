#ifndef WORLD_HPP
#define WORLD_HPP

#include <SFML/Graphics.hpp>
#include <box2d/box2d.h>
#include <vector>

#include <carnn/sim/entities/body.hpp>

class World
{
	public:
	static constexpr float scale = 5.0f;

	World(const b2Vec2 gravity = b2Vec2{0, 0});

	World& step(const float speed, const int vel_it, const int pos_it);
	World& update();
	World& render(sf::RenderTarget& target);

	void update_view(sf::RenderTarget& target, sf::Vector2f origin, float czoom);

	template<typename T = Body>
	T& add_body(const b2BodyDef bdef)
	{
		// Emplace a body and return
		_bodies.emplace_back(std::make_unique<T>(*this, bdef));
		return *static_cast<T*>(_bodies.back().get());
	}

	b2World& get();

	std::size_t body_count() const { return _bodies.size(); }

	private:
	std::vector<std::unique_ptr<Body>> _bodies;

	b2Vec2  _gravity;
	b2World _world;
};

#endif // WORLD_HPP
