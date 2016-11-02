#include "car.hpp"
#include "../maths.hpp"

Car::Car(World& world, const b2BodyDef bdef, const bool do_render) : Body(world, bdef, do_render)
{
	b2RevoluteJointDef rjdef;
	rjdef.bodyA = _body;
	rjdef.enableLimit = true;
	rjdef.lowerAngle = 0.f;
	rjdef.upperAngle = 0.f;
	rjdef.collideConnected = false;
	rjdef.localAnchorB.SetZero();

	for (size_t i = 0; i < 4; ++i)
	{
		b2BodyDef bdef;
		bdef.position = {0.f, 0.f};
		bdef.type = b2_dynamicBody;

		std::array<b2Vec2, 8> vertices = {{
			{-0.20f, -0.25f},
			{-0.12f, -0.30f},
			{ 0.12f, -0.30f},
			{ 0.20f, -0.25f},
			{ 0.20f,  0.25f},
			{ 0.12f,  0.30f},
			{-0.12f,  0.30f},
			{-0.20f,  0.25f}
		}};

		b2PolygonShape shape;
		shape.Set(vertices.data(), vertices.size());

		b2FixtureDef fixdef;
		fixdef.shape = &shape;
		fixdef.density = 100.f;

		_wheels.push_back(&_world.add_body<Wheel>(bdef));
		Wheel* w = _wheels.back();
		w->with_color(sf::Color{20, 20, 20}).add_fixture(fixdef);

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
#include <iostream>
void Car::update()
{
	for (Wheel* wheel : _wheels)
	{
		wheel->cancel_lateral_force(lerp(1.f, 0.3f, _drift_amount));
		wheel->drag();
		wheel->update();
	}

	Body::update();

	_net_front_speed = static_cast<double>(forward_velocity().Length()) / 8.;
	_net_lateral_speed = static_cast<double>(lateral_velocity().Length()) / 8.;
}

void Car::render(sf::RenderTarget& target)
{
	for (Wheel* wheel : _wheels)
		wheel->render(target);
	target.draw(_rays.data(), _rays.size(), sf::Lines);
	Body::render(target);
}

void Car::accelerate(float by)
{
	for (size_t i = 0; i < 2; ++i)
		_wheels[i]->accelerate(1.1f * by);
	for (size_t i = 2; i < 4; ++i)
		_wheels[i]->accelerate(0.7f * by);
}

void Car::apply_torque(float by)
{
	by *= 0.01f * _angle_lock;
	for (size_t i = 0; i < 2; ++i)
	{
		float desired_angle = std::max(-_angle_lock, std::min(_angle_lock, by));

		float fspeed = _turn_speed * _world.dt(),
			  cangle = _front_joints[0]->GetJointAngle(),
			  to_turn = b2Clamp(desired_angle - cangle, -fspeed, fspeed),
			  final_angle = cangle + to_turn;

		for (b2RevoluteJoint* joint : _front_joints)
			joint->SetLimits(final_angle, final_angle);
	}
}

void Car::set_drift(const float drift_amount)
{
	_drift_amount = drift_amount;
}

void Car::transform(const b2Vec2 pos, const float angle)
{
	_body->SetTransform(pos, angle);
	for (Wheel* wheel: _wheels)
		wheel->get().SetTransform(pos, angle);
}

void Car::compute_raycasts(std::vector<Body*>& obstacles)
{
	const size_t ray_count = _rays.size() / 2;
	const float radius = 96.f;
	for (size_t i = 0; i < ray_count; ++i)
	{
		float rad_angle = _body->GetAngle() - (static_cast<float>(i) / static_cast<float>(ray_count - 1)) * static_cast<float>(M_PI);

		b2RayCastInput rin;
		rin.p1 = _body->GetPosition();
		rin.p2 = b2Vec2{rin.p1.x + (cos(rad_angle) * radius), rin.p1.y + (sin(rad_angle) * radius)};
		rin.maxFraction = 1.f;

		float closest_frac = 1.f;
		for (Body* b : obstacles)
		{
			b2Fixture* f = b->get().GetFixtureList();
			b2RayCastOutput rout;
			if (f->RayCast(&rout, rin, 0) && rout.fraction < closest_frac)
				closest_frac = rout.fraction;
		}

		const sf::Color col{static_cast<uint8_t>(lerp(200, 0, closest_frac)),
							static_cast<uint8_t>(lerp(0, 200, closest_frac)),
							0,
							static_cast<uint8_t>(lerp(150, 0, closest_frac))};

		sf::Vertex v1{sf::Vector2f{rin.p1.x, rin.p1.y}};
		v1.color = col;
		_rays[i * 2] = v1;

		b2Vec2 hpoint = rin.p1 + closest_frac * (rin.p2 - rin.p1);
		sf::Vertex v2{sf::Vector2f{hpoint.x, hpoint.y}};
		v2.color = col;
		_rays[(i * 2) + 1] = v2;

		_net_inputs[i] = static_cast<double>(1.f - closest_frac);
	}
}

void Car::add_synapses(Network &n)
{
	for (double& in : _net_inputs)
		n.add_synapse(in);

	n.add_synapse(_net_front_speed);
	n.add_synapse(_net_lateral_speed);
}
