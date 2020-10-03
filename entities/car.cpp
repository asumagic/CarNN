#include "car.hpp"
#include "../maths.hpp"
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
		static_cast<Car*>(bodyA_BUD.body)->cut_engines();
	else if (bodyB_BUD.type == BodyType::BodyCar && bodyA_BUD.type == BodyType::BodyWall)
		static_cast<Car*>(bodyB_BUD.body)->cut_engines();
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
		fixdef.density           = 50.f;
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
	_latest_checkpoint = nullptr;
	_fitness_bias      = 0.0f;
	//_net_feedback = 0.0f;
	_fitness             = 0.0f;
	_acceleration_factor = 1.0f;

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
	for (Wheel* wheel : _wheels)
	{
		wheel->cancel_lateral_force(lerp(1.f, 0.5f, _drift_amount));
		wheel->drag();
		wheel->update();
	}

	Body::update();

	/*if (std::abs(_body->GetAngularVelocity()) > 0.5)
	{
		printf("%f\n", _body->GetAngularVelocity());
		fitness_penalty(10.0f);
	}*/

	float angle    = _body->GetAngle();
	_net_direction = 0.5f * b2Vec2{std::cos(angle), std::sin(angle)} + b2Vec2{0.5f, 0.5f};
}

void Car::render(sf::RenderTarget& target)
{
	target.draw(_rays.data(), _rays.size(), sf::Lines);
	Body::render(target);
}

void Car::contact_checkpoint(Checkpoint& cp)
{
	if (reached_checkpoints() == cp.id)
	{
		_latest_checkpoint = &cp;
	}
}

void Car::set_target_checkpoint(Checkpoint* cp) { _target_checkpoint = cp; }

std::size_t Car::reached_checkpoints() const { return _latest_checkpoint != nullptr ? _latest_checkpoint->id + 1 : 0; }

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

	_fitness = std::max(_fitness, ((reached_checkpoints() + 1) * scale + normalized_distance * scale * 0.8f));

	return _fitness + _fitness_bias;
}

void Car::fitness_penalty(float value) { _fitness_bias -= value; }

void Car::cut_engines() { _acceleration_factor = 0.0f; }

void Car::accelerate(float by)
{
	by *= _acceleration_factor * 2.0f;

	for (size_t i = 0; i < 2; ++i)
		_wheels[i]->accelerate(1.1f * by);
	for (size_t i = 2; i < 4; ++i)
		_wheels[i]->accelerate(0.7f * by);
}

void Car::apply_torque(float by)
{
	for (size_t i = 0; i < 2; ++i)
	{
		float desired_angle = lerp(-_angle_lock, _angle_lock, by * 0.5 + 0.5);

		float fspeed      = _turn_speed * _world.dt();
		float cangle      = _front_joints[i]->GetJointAngle();
		float to_turn     = b2Clamp(desired_angle - cangle, -fspeed, fspeed);
		float final_angle = cangle + to_turn;

		_front_joints[i]->SetLimits(final_angle, final_angle);
	}
}

void Car::set_drift(const float drift_amount) { _drift_amount = drift_amount; }
/*
void Car::feedback(float v)
{
	_net_feedback = v;
}
*/
void Car::transform(const b2Vec2 pos, const float angle)
{
	_body->SetTransform(pos, angle);
	for (Wheel* wheel : _wheels)
		wheel->get().SetTransform(pos, angle);
}

void Car::compute_raycasts(Body& wall_body)
{
	++_raycast_updates;

	const size_t ray_count = _rays.size() / 2;
	const float  radius    = 96.f;
	for (size_t i = 0; i < ray_count; ++i)
	{
		if ((i + _raycast_updates) % 4 != 0)
		{
			continue;
		}

		float rad_angle = _body->GetAngle()
			- (static_cast<float>(i) / static_cast<float>(ray_count - 1)) * static_cast<float>(M_PI);

		b2RayCastInput rin;
		rin.p1          = _body->GetPosition();
		rin.p2          = b2Vec2(rin.p1.x + (cos(rad_angle) * radius), rin.p1.y + (sin(rad_angle) * radius));
		rin.maxFraction = 1.f;

		float closest_frac = 1.f;
		for (b2Fixture* f = wall_body.get().GetFixtureList(); f; f = f->GetNext())
		{
			b2RayCastOutput rout;
			if (f->RayCast(&rout, rin, 0) && rout.fraction < closest_frac)
				closest_frac = rout.fraction;
		}

		// closest_frac = clamp(random_gauss_double(closest_frac, 0.01), 0.0001, 0.999);

		const sf::Color col{
			static_cast<uint8_t>(lerp(200, 0, closest_frac)),
			static_cast<uint8_t>(lerp(0, 200, closest_frac)),
			0,
			static_cast<uint8_t>(lerp(150, 0, closest_frac))};

		sf::Vertex v1{sf::Vector2f{rin.p1.x, rin.p1.y}};
		v1.color     = col;
		_rays[i * 2] = v1;

		b2Vec2     hpoint = rin.p1 + closest_frac * (rin.p2 - rin.p1);
		sf::Vertex v2{sf::Vector2f{hpoint.x, hpoint.y}};
		v2.color           = col;
		_rays[(i * 2) + 1] = v2;

		_net_inputs[i] = static_cast<double>(1.f - closest_frac);
	}
}

void Car::update_inputs(Network& n)
{
	auto& inputs = n.inputs();

	assert(inputs.neurons.size() == _net_inputs.size() + 4);

	std::size_t i = 0;

	b2Vec2 objective_dir = 0.5f * direction_to_objective() + b2Vec2(0.5f, 0.5f);

	// inputs.neurons[i++].value = _net_feedback;
	inputs.neurons[i++].value = objective_dir.x;
	inputs.neurons[i++].value = objective_dir.y;
	inputs.neurons[i++].value = lerp(0, 1, forward_velocity().Length() / 4.0f);
	inputs.neurons[i++].value = lerp(0, 1, lateral_velocity().Length() / 4.0f);

	for (std::size_t j = 0; j < _net_inputs.size(); ++j, ++i)
	{
		inputs.neurons[i].value = _net_inputs[j];
	}
}
