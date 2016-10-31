#include "wheel.hpp"
#include <iostream>

Wheel::Wheel(World& world, const b2BodyDef bdef, const bool do_render) : Body(world, bdef, do_render) {}

void Wheel::cancel_lateral_force(const float multiplier)
{
	float mul = multiplier * _world.dt() * 60.f;
	b2Vec2 impulse = _body->GetMass() * _world.dt() * 60.f * -lateral_velocity();
	if (impulse.Length() > (_max_lateral_impulse * mul))
		impulse *= (_max_lateral_impulse * mul) / impulse.Length();

	_body->ApplyLinearImpulse(impulse, _body->GetWorldCenter(), false);
	_body->ApplyAngularImpulse(_impulse_magnitude * _body->GetInertia() * -_body->GetAngularVelocity(), false);
}

void Wheel::drag()
{
	b2Vec2 fvel = forward_velocity();
	float fspeed = fvel.Normalize();
	float drag = _drag * fspeed;
	_body->ApplyForce(drag * fvel, _body->GetWorldCenter(), false);
}

void Wheel::accelerate(VDirection direction, float power)
{
	float desired_speed = (direction == VDirection::Forward ? _forward_speed : _backwards_speed);

	b2Vec2 current_fnormal = front_normal();
	//float current_speed = b2Dot(forward_velocity(), current_fnormal);

	float final_force;
	if (desired_speed > 0)
		final_force = _max_accel_force * _backwards_mul * power;
	else if (desired_speed < 0)
		final_force = -_max_accel_force * _forward_mul * power;
	else
		return;

	_body->ApplyForce(final_force * current_fnormal, _body->GetWorldCenter(), true);
}

void Wheel::apply_torque(Direction direction, float max)
{
	if (direction == Direction::Left)
		_body->ApplyTorque(-max, true);
	else
		_body->ApplyTorque(max, true);
}
