#ifndef CAR_HPP
#define CAR_HPP

#include <vector>
#include "../neural/network.hpp"
#include "../body.hpp"
#include "../world.hpp"
#include "checkpoint.hpp"
#include "wheel.hpp"

constexpr size_t total_rays = 7;

enum AxonControl : size_t
{
	Axon_Forward,
	Axon_Backwards,
	Axon_Steer_Left,
	Axon_Steer_Right,
	Axon_Drift,
	//Axon_Feedback
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

	void reset(); // TODO: get rid of it

	void update() override;
	void render(sf::RenderTarget& target) override;

	void contact_checkpoint(Checkpoint& cp);
	void set_target_checkpoint(Checkpoint* cp);
	std::size_t reached_checkpoints() const;

	b2Vec2 direction_to_objective() const;

	float fitness() const;
	void fitness_penalty(float value);

	void accelerate(float by);
	void apply_torque(float by);
	void set_drift(const float drift_amount);
	//void feedback(float v);

	void transform(const b2Vec2 pos, const float angle);

	void compute_raycasts(Body& wall_body);

	void update_inputs(proper::Network& n);

private:
	std::array<sf::Vertex, total_rays * 2> _rays{};
	std::vector<Wheel*> _wheels;
	std::array<b2RevoluteJoint*, 2> _front_joints{};

	std::array<double, total_rays> _net_inputs{};

	b2Vec2 _net_direction{0.0f, 0.0f};

	//double _net_feedback = 0.;

	float _drift_amount = 0.0;

	mutable float _fitness = 0.0f;
	float _fitness_bias = 0.0f;

	Checkpoint* _latest_checkpoint = nullptr;
	Checkpoint* _target_checkpoint = nullptr;

	const float _angle_lock = 0.9f,
				_turn_speed = 6.f;
};

#endif // CAR_HPP
