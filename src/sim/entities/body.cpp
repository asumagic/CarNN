#include <carnn/sim/entities/body.hpp>

#include <carnn/sim/world.hpp>

namespace sim::entities
{
Body::Body(World& world, const b2BodyDef bdef, const bool do_render) :
	_do_render(do_render), _next_color{210, 120, 0}, _bdef(bdef), _world(world)
{
	_body     = _world.get().CreateBody(&bdef);
	_bud.body = this;
	_body->GetUserData().pointer = reinterpret_cast<std::uintptr_t>(&_bud);
}

void Body::update()
{
	if (_do_render)
	{
		// @TODO change shapes if any was modified
		for (auto& shape : _shapes)
		{
			// Update the shape position
			b2Vec2 pos = _body->GetPosition();

			shape->setPosition(pos.x, pos.y);
			shape->setRotation(_body->GetAngle() * 57.295779513f);
		}
	}
}

void Body::render(sf::RenderTarget& target)
{
	// Render updated shapes
	if (_do_render)
	{
		for (auto& shape : _shapes)
			target.draw(*shape);
	}
}

void Body::set_type(const BodyType type) { _bud.type = type; }

b2Vec2 Body::front_normal() const { return _body->GetWorldVector(b2Vec2{0.f, -1.f}); }

b2Vec2 Body::lateral_normal() const { return _body->GetWorldVector(b2Vec2{1.f, 0.f}); }

b2Vec2 Body::forward_velocity()
{
	b2Vec2 normal = front_normal();
	return b2Dot(normal, _body->GetLinearVelocity()) * normal;
}

b2Vec2 Body::lateral_velocity()
{
	b2Vec2 normal = lateral_normal();
	return b2Dot(normal, _body->GetLinearVelocity()) * normal;
}

b2Fixture& Body::add_fixture(const b2FixtureDef fdef)
{
	const b2Shape& shape = *fdef.shape;
	if (_do_render)
	{
		switch (shape.GetType())
		{
		case b2Shape::e_polygon:
		{
			const b2PolygonShape& polyshape = *dynamic_cast<const b2PolygonShape*>(&shape);
			sf::ConvexShape       cshape{static_cast<size_t>(polyshape.m_count)};
			for (size_t i = 0; i < cshape.getPointCount(); ++i)
			{
				b2Vec2 b2dvec = polyshape.m_vertices[static_cast<int>(i)];
				cshape.setPoint(i, sf::Vector2f{b2dvec.x, b2dvec.y});
			}
			_shapes.push_back(std::make_unique<sf::ConvexShape>(cshape)); // @TODO modify the pushed shape directly
		}
		break;

		case b2Shape::e_circle:
		case b2Shape::e_edge:
		case b2Shape::e_chain:
		default: break;
		}
	}

	with_color(_next_color);

	return *(_body->CreateFixture(&fdef));
}

Body& Body::with_color(const sf::Color c, std::uint8_t outline_alpha)
{
	_next_color = c;

	for (auto& shape : _shapes)
	{
		shape->setFillColor(_next_color);
		shape->setOutlineColor(
			{static_cast<uint8_t>((255.f * 0.2f) + _next_color.r),
			 static_cast<uint8_t>((255.f * 0.2f) + _next_color.g),
			 static_cast<uint8_t>((255.f * 0.2f) + _next_color.b),
			 outline_alpha});

		shape->setOutlineThickness(0.08f);
	}

	return *this;
}

World& Body::world() { return _world; }

b2Body& Body::get() { return *_body; }

b2BodyDef& Body::definition() { return _bdef; }
} // namespace sim::entities
