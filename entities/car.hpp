#ifndef CAR_HPP
#define CAR_HPP

#include <vector>
#include "../neural/network.hpp"
#include "../body.hpp"
#include "../world.hpp"
#include "checkpoint.hpp"
#include "wheel.hpp"

constexpr size_t total_rays = 15;

enum AxonControl : size_t
{
	Axon_Forward,
	Axon_Backwards,
	Axon_Steer_Left,
	Axon_Steer_Right,
	Axon_Drift
};

class CarCheckpointListener : public b2ContactListener
{
	void BeginContact(b2Contact* contact) override;
	void EndContact(b2Contact* contact) override;
};

class Car : public Body
{
public:
	Car(World& world, const b2BodyDef bdef, const bool do_render = true);

	void update() override;
	void render(sf::RenderTarget& target) override;

	void contact_checkpoint(Checkpoint& cp);

	void accelerate(float by);
	void apply_torque(float by);
	void set_drift(const float drift_amount);

	void transform(const b2Vec2 pos, const float angle);

	void compute_raycasts(Body& wall_body);

	void add_synapses(Network& n);

private:
	std::array<sf::Vertex, total_rays * 2> _rays;
	std::vector<Wheel*> _wheels;
	std::array<b2RevoluteJoint*, 2> _front_joints;

	std::array<double, total_rays> _net_inputs;
	double _net_front_speed = 0., _net_lateral_speed = 0.;

	float _drift_amount = 0.0;

	size_t _reached_checkpoints = 0;

	const float _angle_lock = 0.9f,
				_turn_speed = 6.f;
};

#endif // CAR_HPP
