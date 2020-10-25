#pragma once

#include <carnn/neural/fwd.hpp>
#include <carnn/sim/entities/body.hpp>
#include <carnn/sim/fwd.hpp>
#include <vector>

constexpr size_t total_rays = 12;

namespace sim::entities
{
enum AxonControl : size_t
{
	Axon_Forward,
	Axon_Backwards,
	Axon_Brake,
	Axon_Steer_Left,
	Axon_Steer_Right,
	Axon_Drift,
	// Axon_Feedback
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

	void        contact_checkpoint(Checkpoint& cp);
	void        set_target_checkpoint(Checkpoint* cp);
	std::size_t reached_checkpoints() const;

	b2Vec2 direction_to_objective() const;

	float fitness() const;
	void  fitness_penalty(float value);

	void wall_collision();

	void accelerate(float by);
	void steer(float towards);
	void set_drift(const float drift_amount);
	void brake(float by);

	void transform(const b2Vec2 pos, const float angle);

	void compute_raycasts();

	void update_inputs(neural::Network& n);

	bool dead = false;

	// TODO: make this less garbage
	SimulationUnit* unit       = nullptr;
	Individual*     individual = nullptr;

	private:
	std::array<sf::Vertex, total_rays * 2> _rays{};
	std::vector<Wheel*>                    _wheels;
	std::array<b2RevoluteJoint*, 2>        _front_joints{};

	std::array<double, total_rays> _ray_distances{};
	std::size_t                    _ray_update_frequency = 0;

	std::size_t _reached_checkpoints = 0;

	float _drift_amount = 0.0;

	float _brake_amount = 0.0f;

	mutable float _fitness      = 0.0f;
	float         _fitness_bias = 0.0f;

	float _acceleration_factor = 1.0f;

	Checkpoint* _latest_checkpoint = nullptr;
	Checkpoint* _target_checkpoint = nullptr;

	const float _angle_lock = 0.8f, _turn_speed = 1.0f;
};
} // namespace sim::entities
