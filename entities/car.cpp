#include "car.hpp"
#include "../maths.hpp"
#include "../neural/network.hpp"
#include "../world.hpp"
#include "checkpoint.hpp"
#include "wheel.hpp"
#include <iostream>

void CarCheckpointListener::BeginContact(b2Contact* contact)
{
	b2Fixture *   fixA = contact->GetFixtureA(), *fixB = contact->GetFixtureB();
	BodyUserData &bodyA_BUD = *static_cast<BodyUserData*>(fixA->GetBody()->GetUserData()),
				 &bodyB_BUD = *static_cast<BodyUserData*>(fixB->GetBody()->GetUserData());

	if (bodyA_BUD.type == BodyType::BodyCar && bodyB_BUD.type == BodyType::BodyCheckpoint)
		static_cast<Car*>(bodyA_BUD.body)->contact_checkpoint(*static_cast<Checkpoint*>(bodyB_BUD.body));
	else if (bodyB_BUD.type == BodyType::BodyCar && bodyA_BUD.type == BodyType::BodyCheckpoint)
		static_cast<Car*>(bodyB_BUD.body)->contact_checkpoint(*static_cast<Checkpoint*>(bodyA_BUD.body));

	if (bodyA_BUD.type == BodyType::BodyCar && bodyB_BUD.type == BodyType::BodyWall)
		static_cast<Car*>(bodyA_BUD.body)->wall_collision();
	else if (bodyB_BUD.type == BodyType::BodyCar && bodyA_BUD.type == BodyType::BodyWall)
		static_cast<Car*>(bodyB_BUD.body)->wall_collision();
}

void CarCheckpointListener::EndContact(b2Contact*) {}

Car::Car(World& world, const b2BodyDef bdef, const bool do_render) : Body(world, bdef, do_render)
{
	set_type(BodyType::BodyCar);

	b2RevoluteJointDef rjdef;
	rjdef.bodyA            = _body;
	rjdef.enableLimit      = true;
	rjdef.enableMotor      = true;
	rjdef.lowerAngle       = 0.f;
	rjdef.upperAngle       = 0.f;
	rjdef.collideConnected = false;
	rjdef.localAnchorB.SetZero();

	for (size_t i = 0; i < 4; ++i)
	{
		b2BodyDef bdef;
		bdef.position = {0.f, 0.f};
		bdef.type     = b2_dynamicBody;

		std::array<b2Vec2, 8> vertices
			= {{{-0.20f, -0.25f},
				{-0.12f, -0.30f},
				{0.12f, -0.30f},
				{0.20f, -0.25f},
				{0.20f, 0.25f},
				{0.12f, 0.30f},
				{-0.12f, 0.30f},
				{-0.20f, 0.25f}}};

		b2PolygonShape shape;
		shape.Set(vertices.data(), vertices.size());

		b2FixtureDef fixdef;
		fixdef.shape             = &shape;
		fixdef.density           = 100.0f;
		fixdef.filter.groupIndex = -1;
		fixdef.friction          = 0.95f;

		_wheels.push_back(&_world.add_body<Wheel>(bdef));
		Wheel* w = _wheels.back();
		w->with_color(sf::Color{20, 20, 20, 40}).add_fixture(fixdef);

		rjdef.bodyB = &w->get();

		if (i < 2)
		{
			rjdef.localAnchorA.Set(-1.8f + (i * 3.6f), -1);
			_front_joints[i] = dynamic_cast<b2RevoluteJoint*>(_world.get().CreateJoint(&rjdef));
		}
		else
		{
			rjdef.localAnchorA.Set(-1.8f + ((i - 2) * 3.6f), 2);
			_world.get().CreateJoint(&rjdef);
		}
	}
}

void Car::reset()
{
	dead                 = false;
	_latest_checkpoint   = nullptr;
	_reached_checkpoints = 0;
	_fitness_bias        = 0.0f;
	//_net_feedback = 0.0f;
	_fitness              = 0.0f;
	_acceleration_factor  = 1.0f;
	_ray_update_frequency = 0;

	for (Wheel* wheel : _wheels)
	{
		wheel->get().SetLinearVelocity({0.0f, 0.0f});
		wheel->get().SetAngularVelocity(0.0f);
	}

	get().SetLinearVelocity({0.0f, 0.0f});
	get().SetAngularVelocity(0.0f);
}

void Car::update()
{
	_body->SetSleepingAllowed(true);
	_body->SetAwake(!dead);

	if (dead)
	{
		return;
	}

	for (Wheel* wheel : _wheels)
	{
		wheel->cancel_lateral_force(lerp(1.f, 0.1f, _drift_amount));
		wheel->drag(_brake_amount);
		wheel->update();
	}

	Body::update();

	/*if (std::abs(_body->GetAngularVelocity()) > 0.5)
	{
		printf("%f\n", _body->GetAngularVelocity());
		fitness_penalty(10.0f);
	}*/
}

void Car::render(sf::RenderTarget& target)
{
	if (!dead)
	{
		target.draw(_rays.data(), _rays.size(), sf::Lines);
	}

	Body::render(target);
}

void Car::contact_checkpoint(Checkpoint& cp)
{
	if (_target_checkpoint == &cp)
	{
		_latest_checkpoint = &cp;
		++_reached_checkpoints;
	}
}

void Car::set_target_checkpoint(Checkpoint* cp) { _target_checkpoint = cp; }

std::size_t Car::reached_checkpoints() const { return _reached_checkpoints; }

b2Vec2 Car::direction_to_objective() const
{
	if (_target_checkpoint == nullptr)
	{
		return {0.0f, 0.0f};
	}

	b2Vec2 worldspace_dir = b2Vec2{_target_checkpoint->origin.x, _target_checkpoint->origin.y} - _body->GetPosition();
	worldspace_dir.Normalize();

	b2Vec2 worldspace_car_dir = front_normal();

	const float worldspace_angle = atan2(worldspace_dir.y, worldspace_dir.x);
	const float body_angle       = atan2(worldspace_car_dir.y, worldspace_car_dir.x);
	const float real_angle       = body_angle - worldspace_angle;

	return b2Vec2(cos(real_angle), sin(real_angle));
}

float Car::fitness() const
{
	const sf::Vector2f body_origin{_body->GetPosition().x, _body->GetPosition().y};

	if (reached_checkpoints() == 0)
	{
		return 0.0f;
	}

	if (_target_checkpoint == nullptr)
	{
		return _fitness + _fitness_bias;
	}

	const auto pow2 = [](auto a) { return a * a; };

	const float body_distance = std::sqrt(
		pow2(_target_checkpoint->origin.x - body_origin.x) + pow2(_target_checkpoint->origin.y - body_origin.y));

	const float checkpoint_distance = std::sqrt(
		pow2(_target_checkpoint->origin.x - _latest_checkpoint->origin.x)
		+ pow2(_target_checkpoint->origin.y - _latest_checkpoint->origin.y));

	const float normalized_distance = 1.0f - std::clamp(body_distance / checkpoint_distance, 0.0f, 1.0f);

	const float scale = 1000.0f;

	_fitness = ((reached_checkpoints() + 1) * scale + normalized_distance * scale * 0.8f);

	return _fitness + _fitness_bias;
}

void Car::fitness_penalty(float value) { _fitness_bias -= value; }

void Car::wall_collision()
{
	_acceleration_factor = 0.0f;
	fitness_penalty(300);
	dead = true;
}

void Car::accelerate(float by)
{
	by *= _acceleration_factor;

	for (size_t i = 0; i < 2; ++i)
		_wheels[i]->accelerate(by);
}

void Car::steer(float towards)
{
	for (size_t i = 0; i < 2; ++i)
	{
		float desired_angle = lerp(-_angle_lock, _angle_lock, towards * 0.5 + 0.5);

		float fspeed      = _turn_speed * _world.dt();
		float cangle      = _front_joints[i]->GetJointAngle();
		float to_turn     = b2Clamp(desired_angle - cangle, -fspeed, fspeed);
		float final_angle = cangle + to_turn;

		_front_joints[i]->SetLimits(final_angle, final_angle);
	}
}

void Car::set_drift(const float drift_amount) { _drift_amount = drift_amount; }

void Car::brake(float by) { _brake_amount = by; }

void Car::transform(const b2Vec2 pos, const float angle)
{
	_body->SetTransform(pos, angle);
	for (Wheel* wheel : _wheels)
		wheel->get().SetTransform(pos, angle);
}

class RayCastCallback : public b2RayCastCallback
{
	public:
	float ReportFixture(
		b2Fixture* fixture, [[maybe_unused]] const b2Vec2& point, [[maybe_unused]] const b2Vec2& normal, float fraction)
	{
		auto* data = static_cast<BodyUserData*>(fixture->GetBody()->GetUserData());
		if (data->type == BodyType::BodyWall)
		{
			closest_fraction = fraction;
			return fraction;
		}

		return -1;
	}

	float closest_fraction = 1.0f;
};

void Car::compute_raycasts()
{
	++_ray_update_frequency;

	const size_t ray_count = _rays.size() / 2;
	const float  radius    = 64.f;
	for (size_t i = 0; i < ray_count; ++i)
	{
		if ((i + _ray_update_frequency) % 6 != 0)
		{
			continue;
		}

		float rad_angle = _body->GetAngle() - (float(i) / float(ray_count - 1)) * float(M_PI);

		const b2Vec2 p1 = _body->GetPosition();
		const b2Vec2 p2 = b2Vec2(p1.x + (cos(rad_angle) * radius), p1.y + (sin(rad_angle) * radius));

		RayCastCallback raycast;
		_world.get().RayCast(&raycast, p1, p2);

		// closest_frac = clamp(random_gauss_double(closest_frac, 0.01), 0.0001, 0.999);

		const sf::Color col{
			static_cast<uint8_t>(lerp(200, 0, raycast.closest_fraction)),
			static_cast<uint8_t>(lerp(0, 200, raycast.closest_fraction)),
			0,
			static_cast<uint8_t>(lerp(150, 0, raycast.closest_fraction))};

		sf::Vertex v1{sf::Vector2f{p1.x, p1.y}};
		v1.color     = col;
		_rays[i * 2] = v1;

		b2Vec2     hpoint = p1 + raycast.closest_fraction * (p2 - p1);
		sf::Vertex v2{sf::Vector2f{hpoint.x, hpoint.y}};
		v2.color           = col;
		_rays[(i * 2) + 1] = v2;

		_ray_distances[i] = static_cast<double>(1.f - raycast.closest_fraction);
	}
}

void Car::update_inputs(Network& n)
{
	auto& inputs = n.inputs();

	assert(inputs.neurons.size() == _ray_distances.size() + 4);

	std::size_t i = 0;

	b2Vec2 objective_dir = 0.5f * direction_to_objective() + b2Vec2(0.5f, 0.5f);

	// inputs.neurons[i++].value = _net_feedback;
	inputs.neurons[i++].partial_activation = objective_dir.x;
	inputs.neurons[i++].partial_activation = objective_dir.y;
	inputs.neurons[i++].partial_activation = lerp(0.0, 1.0, forward_velocity().Length() / 6.0f);
	inputs.neurons[i++].partial_activation = lerp(0.0, 1.0, lateral_velocity().Length() / 1.0f);

	for (std::size_t j = 0; j < _ray_distances.size(); ++j, ++i)
	{
		inputs.neurons[i].partial_activation = _ray_distances[j];
	}
}
