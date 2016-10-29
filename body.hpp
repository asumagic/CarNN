#ifndef BODY_HPP
#define BODY_HPP

#include <Box2D/Box2D.h>
#include "world.hpp"

class World;

class Body
{
public:
	Body(World& world, const b2BodyDef bdef);

	void update();
	void render(sf::RenderTarget& target);

	b2Fixture& add_fixture(const b2FixtureDef fdef);

	b2Body& get();
	b2BodyDef& definition();

private:
	std::vector<std::unique_ptr<sf::Shape>> _shapes;

	b2BodyDef _bdef;
	b2Body* _body;
	World& _world;
};

#endif // BODY_HPP
