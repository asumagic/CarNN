#include <carnn/sim/entities/wheel.hpp>

#include <carnn/sim/world.hpp>
#include <carnn/util/maths.hpp>

Wheel::Wheel(World& world, const b2BodyDef bdef, const bool do_render) : Body(world, bdef, do_render)
{
	set_type(BodyType::BodyWheel);
}

void Wheel::cancel_lateral_force(const float multiplier)
{
	float  mul     = multiplier * 60.f / 30.0f;
	b2Vec2 impulse = _body->GetMass() * -lateral_velocity();
	if (impulse.Length() > (_max_lateral_impulse * mul))
		impulse *= (_max_lateral_impulse * mul) / impulse.Length();

	_body->ApplyLinearImpulse(impulse, _body->GetWorldCenter(), false);
	_body->ApplyAngularImpulse(_impulse_magnitude * _body->GetInertia() * -_body->GetAngularVelocity(), false);
}

void Wheel::drag(float brake_intensity)
{
	b2Vec2 fvel   = forward_velocity();
	float  fspeed = fvel.Normalize();
	float  drag   = lerp(_drag, _brake_drag, brake_intensity) * fspeed;
	_body->ApplyForce(drag * fvel, _body->GetWorldCenter(), false);
}

void Wheel::accelerate(float throttle)
{
	b2Vec2 current_fnormal = front_normal();

	float final_force;
	if (throttle > 0)
		final_force = _max_accel_force * _forward_mul * std::abs(throttle);
	else if (throttle < 0)
		final_force = -_max_accel_force * _backwards_mul * std::abs(throttle);
	else
		return;

	_body->ApplyForce(final_force * current_fnormal, _body->GetWorldCenter(), true);
}
