#pragma once

#include "../body.hpp"

enum class Direction
{
	Left,
	Right,
	None
};

class Wheel : public Body
{
	public:
	Wheel(World& world, const b2BodyDef bdef, const bool do_render = true);

	void cancel_lateral_force(const float multiplier);
	void drag();
	void accelerate(float by); // power 0..1

	private:
	static constexpr float _drag = -11.f, _impulse_magnitude = .01f, _forward_speed = -5000.f, _backwards_speed = 700.f,
						   _forward_mul = 0.16f, _backwards_mul = 0.05f, _max_accel_force = 600.f,
						   _max_lateral_impulse = 15.f;
};
