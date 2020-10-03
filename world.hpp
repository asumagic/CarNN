#ifndef WORLD_HPP
#define WORLD_HPP

#include <SFML/Graphics.hpp>
#include <box2d/box2d.h>
#include <vector>

#include "body.hpp"

class Body;

class World
{
	public:
	static constexpr float world_scale = 5.0f;

	World(const b2Vec2 gravity = b2Vec2{0, 0});

	World& step(const float speed, const int vel_it, const int pos_it);
	World& update();
	World& render(sf::RenderTarget& target);

	void  set_dt(const float dt);
	float dt() const;

	void update_view(sf::RenderTarget& target, sf::Vector2f origin, float czoom);

	// changes bodies
	std::vector<sf::Vertex> import_map(const std::string fname, Body*& wall, b2Vec2& car_origin);

	template<typename T = Body>
	T& add_body(const b2BodyDef bdef)
	{
		// Emplace a body and return
		_bodies.emplace_back(std::make_unique<T>(*this, bdef));
		return *static_cast<T*>(_bodies.back().get());
	}

	b2World& get();

	private:
	std::vector<std::unique_ptr<Body>> _bodies;

	float _dt = 0.f;

	b2Vec2  _gravity;
	b2World _world;
};

#endif // WORLD_HPP
