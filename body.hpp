#pragma once

#include <SFML/Graphics.hpp>
#include <box2d/box2d.h>
#include <vector>

class World;
class Body;

enum class BodyType
{
	BodyAny = 0,
	BodyCar,
	BodyWheel,
	BodyCheckpoint,
	BodyWall
};

struct BodyUserData
{
	Body*    body;
	BodyType type = BodyType::BodyAny;
};

class Body
{
	public:
	Body(World& world, const b2BodyDef bdef, const bool do_render = true);
	virtual ~Body() {}

	// Delete move and copy constructors
	Body(const Body&)  = delete;
	Body(const Body&&) = delete;

	virtual void update();
	virtual void render(sf::RenderTarget& target);

	void set_type(const BodyType type);

	b2Vec2 front_normal() const;
	b2Vec2 lateral_normal() const;

	b2Vec2 forward_velocity();
	b2Vec2 lateral_velocity();

	b2Fixture& add_fixture(const b2FixtureDef fdef);

	Body& with_color(const sf::Color c);

	World& world();

	b2Body&    get();
	b2BodyDef& definition();

	protected:
	bool _do_render;

	std::vector<std::unique_ptr<sf::Shape>> _shapes;
	sf::Color                               _next_color;

	BodyUserData _bud;

	b2BodyDef _bdef;
	b2Body*   _body;
	World&    _world;
};
