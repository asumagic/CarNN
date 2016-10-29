#include "body.hpp"
#include <iostream>

Body::Body(World& world, const b2BodyDef bdef) : _bdef(bdef), _world(world)
{
	_body = _world.get().CreateBody(&bdef);
}

void Body::update()
{
	// @TODO change shapes if any was modified
	for (auto& shape : _shapes)
	{
		b2Vec2 pos = _body->GetPosition();
		shape->setPosition(pos.x, pos.y);
	}
}

void Body::render(sf::RenderTarget& target)
{
	for (auto& shape : _shapes)
		target.draw(*shape);
}

b2Fixture& Body::add_fixture(const b2FixtureDef fdef)
{
	const b2Shape& shape = *fdef.shape;
	switch (shape.GetType())
	{
	case b2Shape::e_circle: {
		sf::CircleShape cshape{shape.m_radius};
		cshape.setFillColor(sf::Color::Transparent);
		cshape.setOutlineColor(sf::Color::Red);
		cshape.setOutlineThickness(0.12f);
		_shapes.push_back(std::make_unique<sf::CircleShape>(cshape));
	} break;

	default:
		std::cerr << "WARNING: Adding a fixture with a shape that cannot be drawn" << std::endl;
	}

	return *(_body->CreateFixture(&fdef));
}

b2Body& Body::get()
{
	return *_body;
}

b2BodyDef& Body::definition()
{
	return _bdef;
}


