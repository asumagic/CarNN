#include "wheel.hpp"

#include "../world.hpp"

Wheel::Wheel(World& world, const b2BodyDef bdef, const bool do_render) : Body(world, bdef, do_render)
{
	set_type(BodyType::BodyWheel);
}

void Wheel::cancel_lateral_force(const float multiplier)
{
	float  mul     = multiplier * _world.dt() * 60.f;
	b2Vec2 impulse = _body->GetMass() * _world.dt() * 60.f * -lateral_velocity();
	if (impulse.Length() > (_max_lateral_impulse * mul))
		impulse *= (_max_lateral_impulse * mul) / impulse.Length();

	_body->ApplyLinearImpulse(impulse, _body->GetWorldCenter(), false);
	_body->ApplyAngularImpulse(_impulse_magnitude * _body->GetInertia() * -_body->GetAngularVelocity(), false);
}

void Wheel::drag()
{
	b2Vec2 fvel   = forward_velocity();
	float  fspeed = fvel.Normalize();
	float  drag   = _drag * fspeed;
	_body->ApplyForce(drag * fvel, _body->GetWorldCenter(), false);
}

void Wheel::accelerate(float by)
{
	float desired_speed = (by > 0.f ? _forward_speed : _backwards_speed);

	b2Vec2 current_fnormal = front_normal();

	float final_force;
	if (desired_speed > 0)
		final_force = _max_accel_force * _backwards_mul * std::abs(by);
	else if (desired_speed < 0)
		final_force = -_max_accel_force * _forward_mul * std::abs(by);
	else
		return;

	_body->ApplyForce(final_force * current_fnormal, _body->GetWorldCenter(), true);
}
