#ifndef BODY_HPP
#define BODY_HPP

#include <Box2D/Box2D.h>
#include "world.hpp"

class World;

class Body
{
public:
	Body(World& world, const b2BodyDef bdef, const bool do_render = true);
	virtual ~Body() {}

	virtual void update();
	virtual void render(sf::RenderTarget& target);

	b2Vec2 front_normal();
	b2Vec2 lateral_normal();

	b2Vec2 forward_velocity();
	b2Vec2 lateral_velocity();

	b2Fixture& add_fixture(const b2FixtureDef fdef);

	Body& with_color(const sf::Color c);

	b2Body& get();
	b2BodyDef& definition();

protected:
	bool _do_render;

	std::vector<std::unique_ptr<sf::Shape>> _shapes;
	sf::Color _next_color;

	b2BodyDef _bdef;
	b2Body* _body;
	World& _world;
};

#endif // BODY_HPP
