#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <fmt/core.h>
#include <iostream>
#include <fstream>
#include <json/value.h>
#include <json/reader.h>
#include "world.hpp"
#include "entities/checkpoint.hpp"
#include "entities/wheel.hpp"
#include "entities/car.hpp"
#include "neural/network.hpp"
#include "randomutil.hpp"

int app(sf::RenderWindow& win)
{
	World w{b2Vec2{0.f, 0.f}};

	std::ifstream race_config{"race.json", std::ios::binary};
	Json::Value root;
	Json::CharReaderBuilder reader;
	Json::parseFromStream(reader, race_config, &root, nullptr);

	Body* wall_body;
	b2Vec2 car_pos;
	std::vector<sf::Vertex> wall_vertices = w.import_map(root["filepath"].asString(), wall_body, car_pos);

	std::vector<Checkpoint*> checkpoints;

	b2BodyDef cp_bdef;
	cp_bdef.type = b2_staticBody;
	sf::VertexArray checkpoint_vertices{sf::Lines};
	{
		size_t i = 0;
		for (const Json::Value& cp : root["checkpoints"])
		{
			sf::Vector2f p1{cp["p1"].get(Json::ArrayIndex{0}, 0).asFloat() * 5.f, cp["p1"].get(Json::ArrayIndex{1}, 0).asFloat() * 5.f},
						 p2{cp["p2"].get(Json::ArrayIndex{0}, 0).asFloat() * 5.f, cp["p2"].get(Json::ArrayIndex{1}, 0).asFloat() * 5.f};

			sf::Vector2f center{p1 + (p2 - p1) / 2.f};

			// Extend the line by 5 pixels each side
			p1.x += (p1.x > center.x) ? 5.f : -5.f;  // @todo compact if possible
			p2.x += (p2.x > center.x) ? 5.f : -5.f;  // for (float& x : {p1.x, p2.x} doesn't work since you can't get a reference to both
			p1.y += (p1.y > center.y) ? 5.f : -5.f;
			p2.y += (p2.y > center.y) ? 5.f : -5.f;

			const static sf::Color cp_col{0, 127, 0, 100};

			checkpoint_vertices.append(sf::Vertex{p1, cp_col});
			checkpoint_vertices.append(sf::Vertex{p2, cp_col});

			b2EdgeShape cp_shape;
			cp_shape.Set(b2Vec2{p1.x, p1.y}, b2Vec2{p2.x, p2.y});

			b2FixtureDef cp_fdef;
			cp_fdef.shape = &cp_shape;
			cp_fdef.isSensor = true;

			Checkpoint& cpb = w.add_body<Checkpoint>(cp_bdef);
			cpb.origin = center;
			cpb.id = i++;
			cpb.add_fixture(cp_fdef);

			checkpoints.push_back(&cpb);
		}
	}

	b2BodyDef bdef;
	bdef.type = b2_dynamicBody;

	std::array<b2Vec2, 8> vertices = {{
		{-1.50f, -0.30f},
		{-1.00f, -1.90f},
		{-0.50f, -2.10f},
		{ 0.50f, -2.10f},
		{ 1.00f, -1.90f},
		{ 1.50f, -0.30f},
		{ 1.50f,  2.00f},
		{-1.50f,  2.00f}
	}};

	b2PolygonShape shape;
	shape.Set(vertices.data(), vertices.size()); // shape.SetAsBox(1.5f, 2.1f);

	b2FixtureDef fixdef;
	fixdef.shape = &shape;
	fixdef.density = 50.f;
	fixdef.friction = 0.7f;
	fixdef.restitution = 0.2f;
	fixdef.filter.groupIndex = -1;

	Mutator mutator;

	std::vector<Car*> cars;
	std::vector<Network> networks;

	float total_time = 0.0f;

	const auto reset_car = [&](Car& car) {
		car.transform(car_pos, static_cast<float>(0.5 * M_PI));
		car.reset();
	};

	const auto reset = [&] {
		total_time = 0.0f;

		for (auto& car : cars)
		{
			reset_car(*car);
		}
	};

	const auto mutate = [&] {
		std::vector<NetworkResult> results;

		for (std::size_t i = 0; i < cars.size(); ++i)
		{
			results.push_back({
				&networks[i],
				cars[i]
			});
		}

		mutator.darwin(results);

		reset();
	};

	for (std::size_t i = 0; i < 150; ++i)
	{
		Car& car = w.add_body<Car>(bdef);
		cars.push_back(&car);

		car.with_color(sf::Color{200, 50, 0, 50}).add_fixture(fixdef);
		car.transform(car_pos, static_cast<float>(0.5 * M_PI));

		networks.emplace_back(total_rays + 4, 5, 20);
	}

	// Add the listener AFTER moving the car
	CarCheckpointListener listener;
	w.get().SetContactListener(&listener);

	// Define the time data (delta time + simulation speed)
	sf::Clock dtclock;
	float speed = 10.0f;

	// Define the camera settings
	float czoom = 0.1f;

	sf::Font infofnt;
	if (!infofnt.loadFromFile("Cantarell-Bold.ttf"))
		std::cerr << "WARNING: Failed to load font." << std::endl;

	bool fast_simulation = false;
	std::size_t ticks = 0;

	sf::Text car_info;
	car_info.setFont(infofnt);
	car_info.setFillColor(sf::Color::White);
	car_info.setCharacterSize(30);
	car_info.setScale(0.4f, 0.4f);
	car_info.setPosition(16.f, 16.f);
	car_info.setStyle(sf::Text::Bold);

	sf::Text fps_info;
	fps_info.setFont(infofnt);
	fps_info.setFillColor(sf::Color{80, 80, 80});
	fps_info.setCharacterSize(30);
	fps_info.setScale(0.4f, 0.4f);
	fps_info.setStyle(sf::Text::Bold);

	while (win.isOpen())
	{
		auto top_car_it = std::max_element(cars.begin(), cars.end(), [&](const Car* a, const Car* b) {
			return a->fitness() <
				   b->fitness();
		});

		Car& top_car = **top_car_it;

		// @TODO move this mess
		#pragma omp parallel for
		for (std::size_t i = 0; i < cars.size(); ++i)
		{
			Car& c = *cars[i];
			Network& net = networks[i];

			c.set_target_checkpoint(c.reached_checkpoints() == checkpoints.size() ? nullptr : checkpoints[c.reached_checkpoints()]);

			c.update_inputs(net);
			net.update();
			const auto& results = net.outputs();

			c.set_drift(static_cast<float>(results.neurons[Axon_Drift].value));
			c.apply_torque(static_cast<float>(results.neurons[Axon_Steer_Right].value - results.neurons[Axon_Steer_Left].value));
			c.accelerate(static_cast<float>(results.neurons[Axon_Forward].value - results.neurons[Axon_Backwards].value));
			//c.feedback(static_cast<float>(results.neurons[Axon_Feedback].value));

			c.compute_raycasts(*wall_body);
		}

		float real_dt = dtclock.restart().asSeconds();
		w.set_dt(1.0f / 30.0f);
		w.step(speed, 4, 2).update();

		++ticks;
		total_time += w.dt();

		if (total_time > 42.0f)
		{
			mutate();
		}

		if (!fast_simulation || ticks % 30 == 0)
		{
			for (sf::Event ev; win.pollEvent(ev);)
			{
				switch (ev.type)
				{
				case sf::Event::Closed:
					win.close();
					break;

				case sf::Event::KeyPressed:
					if (ev.key.code == sf::Keyboard::Escape) { return 0; }
					else if (ev.key.code == sf::Keyboard::R) { return 2; }
					else if (ev.key.code == sf::Keyboard::M) { mutate(); }
					else if (ev.key.code == sf::Keyboard::F) { fast_simulation = fast_simulation ? false : true; }
					break;

				case sf::Event::MouseWheelScrolled:
					czoom = std::max(czoom - (ev.mouseWheelScroll.delta * .01f), 0.01f);
					break;

				default:
					break;
				}
			}

			win.clear(sf::Color{20, 20, 20});
			win.draw(checkpoint_vertices);
			win.draw(wall_vertices.data(), wall_vertices.size(), sf::Lines);
			w.render(win);

			const b2Vec2 b2target = top_car.get().GetPosition();
			w.update_view(win, sf::Vector2f{b2target.x, b2target.y}, czoom);

			sf::View cview{win.getView()};
			float ui_scale = 0.5f;
			win.setView(sf::View{sf::FloatRect{0.f, 0.f, static_cast<float>(win.getSize().x) * ui_scale, static_cast<float>(win.getSize().y) * ui_scale}});

			car_info.setString(fmt::format(
				"Generation #{}\n"
				"Speed: {:.1f}km/h\n"
				"Fitness: {:.0f}",
				mutator.current_generation,
				top_car.forward_velocity().Length() * 20.0f,
				top_car.fitness()
			));

			fps_info.setString(fmt::format(
				"{:.1f}ups",
				1.0f / real_dt
			));

			fps_info.setPosition(8.f, (0.5f * win.getSize().y) - 16.f);

			win.draw(car_info);
			win.draw(fps_info);

			win.setView(cview);

			win.display();
		}
	}

	return 0;
}

int main()
{
	// Define the render window
	sf::ContextSettings settings;
	settings.antialiasingLevel = 16;
	sf::RenderWindow win{sf::VideoMode{800, 600}, "Neural Network", sf::Style::Default, settings};
	win.setFramerateLimit(80);

	while (app(win) == 2);
}
