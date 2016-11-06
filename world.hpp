#ifndef WORLD_HPP
#define WORLD_HPP

#include <Box2D/Box2D.h>
#include <SFML/Graphics.hpp>
#include <vector>
#include "body.hpp"

class Body;

// @TODO move elsewhere
struct Line
{
	Line(unsigned x1, unsigned y1, unsigned x2, unsigned y2) : p1{static_cast<float>(x1), static_cast<float>(y1)},
															   p2{static_cast<float>(x2), static_cast<float>(y2)} {}

	sf::Vector2f p1, p2;

	bool operator==(const Line& other) const
	{
		return (p1 == other.p1 && p2 == other.p2) ||
			   (p2 == other.p1 && p1 == other.p2);
	}

};

class World
{
public:
	World(const b2Vec2 gravity = b2Vec2{0, 0});

	World& step(const float speed, const int vel_it, const int pos_it);
	World& update();
	World& render(sf::RenderTarget& target);

	void set_dt(const float dt);
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

	b2Vec2 _gravity;
	b2World _world;
};

#endif // WORLD_HPP
