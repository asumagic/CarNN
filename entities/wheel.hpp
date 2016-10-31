#ifndef WHEEL_HPP
#define WHEEL_HPP

#include "../body.hpp"

enum class Direction
{
	Left,
	Right,
	None
};

enum class VDirection : bool
{
	Forward,
	Backwards
};

class Wheel : public Body
{
public:
	Wheel(World& world, const b2BodyDef bdef, const bool do_render = true);

	void cancel_lateral_force(const float multiplier);
	void drag();
	void accelerate(VDirection direction, float power); // power 0..1
	void apply_torque(Direction direction, float max);

private:
	const float _drag = -11.f,
				_impulse_magnitude = .01f,
				_forward_speed = -5000.f,
				_backwards_speed = 700.f,
				_forward_mul = 0.13f,
				_backwards_mul = 0.05f,
				_max_accel_force = 600.f,
				_max_lateral_impulse = 15.f;
};

#endif // WHEEL_HPP
