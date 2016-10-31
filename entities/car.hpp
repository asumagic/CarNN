#ifndef CAR_HPP
#define CAR_HPP

#include <vector>
#include "../body.hpp"
#include "../world.hpp"
#include "./wheel.hpp"

class Car : public Body
{
public:
	Car(World& world, const b2BodyDef bdef, const bool do_render = true);

	void update() override;
	void render(sf::RenderTarget& target) override;

	void accelerate(VDirection direction);
	void apply_torque(Direction direction, float by);
	void set_drift(const bool is_drifting);

private:
	std::vector<Wheel*> _wheels;
	std::array<b2RevoluteJoint*, 2> _front_joints;

	bool _drift = false;

	const float _angle_lock = 0.9f,
				_turn_speed = 4.f;
};

#endif // CAR_HPP
