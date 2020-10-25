#pragma once

#include <carnn/sim/entities/body.hpp>

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
	void drag(float brake_intensity);
	void accelerate(float throttle); // power -1..1

	private:
	static constexpr float _drag = -10.0f, _brake_drag = -60.0f, _impulse_magnitude = .1f, _forward_mul = 0.15f,
						   _backwards_mul = 0.05f, _max_accel_force = 600.f, _max_lateral_impulse = 15.f;
};
