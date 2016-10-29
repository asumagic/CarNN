#ifndef WORLD_HPP
#define WORLD_HPP

#include <Box2D/Box2D.h>
#include <SFML/Graphics.hpp>
#include <vector>
#include "body.hpp"

class Body;

class World
{
public:
	World(const b2Vec2 gravity = b2Vec2{0, 0});

	World& step(const float dt, const int vel_it, const int pos_it);
	World& update();
	World& render(sf::RenderTarget& target);

	Body& add_body(const b2BodyDef bdef);

	b2World& get();

private:
	std::vector<Body> _bodies;

	b2Vec2 _gravity;
	b2World _world;
};

#endif // WORLD_HPP
