#include "car.hpp"

Car::Car(World& world, const b2BodyDef bdef, const bool do_render) : Body(world, bdef, do_render)
{
	b2RevoluteJointDef rjdef;
	rjdef.bodyA = _body;
	rjdef.enableLimit = true;
	rjdef.lowerAngle = 0.f;
	rjdef.upperAngle = 0.f;
	rjdef.collideConnected = false;
	rjdef.localAnchorB.SetZero();

	for (size_t i = 0; i < 4; ++i)
	{
		b2BodyDef bdef;
		bdef.position = {0.f, 0.f};
		bdef.type = b2_dynamicBody;

		std::array<b2Vec2, 8> vertices = {{
			{-0.20f, -0.25f},
			{-0.12f, -0.30f},
			{ 0.12f, -0.30f},
			{ 0.20f, -0.25f},
			{ 0.20f,  0.25f},
			{ 0.12f,  0.30f},
			{-0.12f,  0.30f},
			{-0.20f,  0.25f}
		}};

		b2PolygonShape shape;
		shape.Set(vertices.data(), vertices.size());

		b2FixtureDef fixdef;
		fixdef.shape = &shape;
		fixdef.density = 100.f;

		_wheels.push_back(&_world.add_body<Wheel>(bdef));
		Wheel* w = _wheels.back();
		w->with_color(sf::Color{20, 20, 20}).add_fixture(fixdef);

		rjdef.bodyB = &w->get();

		if (i < 2)
		{
			rjdef.localAnchorA.Set(-1.8f + (i * 3.6f), -1);
			_front_joints[i] = dynamic_cast<b2RevoluteJoint*>(_world.get().CreateJoint(&rjdef));
		}
		else
		{
			rjdef.localAnchorA.Set(-1.8f + ((i - 2) * 3.6f), 2);
			_world.get().CreateJoint(&rjdef);
		}
	}
}

void Car::update()
{
	Body::update();
	for (Wheel* wheel : _wheels)
	{
		wheel->cancel_lateral_force(_drift ? 0.3f : 1.f);
		wheel->drag();
		wheel->update();
	}
}

void Car::render(sf::RenderTarget& target)
{
	for (Wheel* wheel : _wheels)
		wheel->render(target);
	Body::render(target);
}

void Car::accelerate(VDirection direction)
{
	for (size_t i = 0; i < 2; ++i)
		_wheels[i]->accelerate(direction, 1.f);
	for (size_t i = 2; i < 4; ++i)
		_wheels[i]->accelerate(direction, 0.6f);
}

void Car::apply_torque(Direction direction, float by)
{
	by *= 0.01f * _angle_lock;
	for (size_t i = 0; i < 2; ++i)
	{
		float desired_angle;

		if (by == 0.f)
			desired_angle = (direction == Direction::Left) ? -_angle_lock : (direction == Direction::Right ? _angle_lock : 0.f);
		else
			desired_angle = std::max(-_angle_lock, std::min(_angle_lock, by));

		float fspeed = _turn_speed * _world.dt(),
			  cangle = _front_joints[0]->GetJointAngle(),
			  to_turn = b2Clamp(desired_angle - cangle, -fspeed, fspeed),
			  final_angle = cangle + to_turn;

		for (b2RevoluteJoint* joint : _front_joints)
			joint->SetLimits(final_angle, final_angle);

	}
}

void Car::set_drift(const bool is_drifting)
{
	_drift = is_drifting;
}
