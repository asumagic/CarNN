#include "entities/car.hpp"
#include "entities/checkpoint.hpp"
#include "entities/wheel.hpp"
#include "imgui.h"
#include "neural/mutator.hpp"
#include "neural/network.hpp"
#include "neural/visualizer.hpp"
#include "randomutil.hpp"
#include "simulationunit.hpp"
#include "world.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <fmt/core.h>
#include <fstream>
#include <imgui-SFML.h>
#include <iostream>
#include <json/reader.h>
#include <json/value.h>

struct GuiWindows
{
	bool simulation_open = false;
	bool mutator_open    = false;
	bool draw_neural     = false;
};

int app(sf::RenderWindow& win)
{
	Simulation sim;

	GuiWindows windows;

	std::ifstream           race_config{"race.json", std::ios::binary};
	Json::Value             root;
	Json::CharReaderBuilder reader;
	Json::parseFromStream(reader, race_config, &root, nullptr);

	Body*                   wall_body;
	b2Vec2                  car_pos;
	std::vector<sf::Vertex> wall_vertices
		= sim.units[0].world.import_map(root["filepath"].asString(), wall_body, car_pos);

	for (std::size_t i = 1; i < sim.units.size(); ++i)
	{
		sim.units[i].world.import_map(root["filepath"].asString(), wall_body, car_pos);
	}

	b2BodyDef cp_bdef;
	cp_bdef.type = b2_staticBody;
	sf::VertexArray checkpoint_vertices{sf::Lines};
	{
		size_t i = 0;
		for (const Json::Value& cp : root["checkpoints"])
		{
			sf::Vector2f p1{
				cp["p1"].get(Json::ArrayIndex{0}, 0).asFloat() * 5.f,
				cp["p1"].get(Json::ArrayIndex{1}, 0).asFloat() * 5.f},
				p2{cp["p2"].get(Json::ArrayIndex{0}, 0).asFloat() * 5.f,
				   cp["p2"].get(Json::ArrayIndex{1}, 0).asFloat() * 5.f};

			sf::Vector2f center{p1 + (p2 - p1) / 2.f};

			// Extend the line by 5 pixels each side
			p1.x += (p1.x > center.x) ? 5.f : -5.f; // @todo compact if possible
			p2.x += (p2.x > center.x)
				? 5.f
				: -5.f; // for (float& x : {p1.x, p2.x} doesn't work since you can't get a reference to both
			p1.y += (p1.y > center.y) ? 5.f : -5.f;
			p2.y += (p2.y > center.y) ? 5.f : -5.f;

			const static sf::Color cp_col{0, 127, 0, 100};

			checkpoint_vertices.append(sf::Vertex{p1, cp_col});
			checkpoint_vertices.append(sf::Vertex{p2, cp_col});

			b2EdgeShape cp_shape;
			cp_shape.SetTwoSided(b2Vec2{p1.x, p1.y}, b2Vec2{p2.x, p2.y});

			b2FixtureDef cp_fdef;
			cp_fdef.shape    = &cp_shape;
			cp_fdef.isSensor = true;

			for (SimulationUnit& unit : sim.units)
			{
				Checkpoint& cpb = unit.world.add_body<Checkpoint>(cp_bdef);
				cpb.origin      = center;
				cpb.id          = i++;
				cpb.add_fixture(cp_fdef);

				unit.checkpoints.push_back(&cpb);
			}
		}
	}

	b2BodyDef bdef;
	bdef.type           = b2_dynamicBody;
	bdef.angularDamping = 0.01f;

	std::array<b2Vec2, 8> vertices
		= {{{-1.50f, -0.30f},
			{-1.00f, -1.90f},
			{-0.50f, -2.10f},
			{0.50f, -2.10f},
			{1.00f, -1.90f},
			{1.50f, -0.30f},
			{1.50f, 2.00f},
			{-1.50f, 2.00f}}};

	b2PolygonShape shape;
	shape.Set(vertices.data(), vertices.size()); // shape.SetAsBox(1.5f, 2.1f);

	b2FixtureDef fixdef;
	fixdef.shape             = &shape;
	fixdef.density           = 100.f;
	fixdef.friction          = 1.0f;
	fixdef.restitution       = 0.2f;
	fixdef.filter.groupIndex = -1;

	Mutator mutator;
	mutator.settings.load_from_file();

	std::vector<Car*>    cars;
	std::vector<Network> networks;

	float total_time = 0.0f;

	const auto reset_car = [&](Car& car) {
		car.transform(car_pos, static_cast<float>(0.5 * M_PI));
		car.reset();
	};

	const auto reset = [&] {
		total_time = 0.0f;

		for (auto& network : networks)
		{
			network.reset_values();
		}

		for (auto& car : cars)
		{
			reset_car(*car);
		}
	};

	const auto mutate = [&] {
		std::vector<NetworkResult> results;

		for (std::size_t i = 0; i < cars.size(); ++i)
		{
			results.push_back({&networks[i], cars[i]});
		}

		mutator.darwin(results);

		reset();
	};

	for (std::size_t i = 0; i < 24 * 5; ++i)
	{
		SimulationUnit& unit = sim.optimal_unit();
		Car&            car  = unit.world.add_body<Car>(bdef);
		car.unit             = &unit;
		cars.push_back(&car);

		car.with_color(sf::Color{200, 50, 0, 50}).add_fixture(fixdef);
		car.transform(car_pos, static_cast<float>(0.5 * M_PI));

		auto& last_network = networks.emplace_back(total_rays + 4, 6);
		mutator.randomize(last_network);

		auto& inputs  = last_network.inputs().neurons;
		auto& outputs = last_network.outputs().neurons;

		inputs[0].label = "vector to objective (x)";
		inputs[1].label = "vector to objective (y)";
		inputs[2].label = "velocity (forward)";
		inputs[3].label = "velocity (lateral)";
		for (std::size_t i = 4; i < inputs.size(); ++i)
		{
			inputs[i].label = fmt::format("lidar #{}", i - 4 + 1);
		}

		outputs[Axon_Forward].label     = "Forward";
		outputs[Axon_Backwards].label   = "Backwards";
		outputs[Axon_Brake].label       = "Brake force";
		outputs[Axon_Steer_Left].label  = "Left steering";
		outputs[Axon_Steer_Right].label = "Right steering";
		outputs[Axon_Drift].label       = "Drifting";
	}

	// Add the listener AFTER moving the car
	CarCheckpointListener listener;
	for (SimulationUnit& unit : sim.units)
	{
		unit.world.get().SetContactListener(&listener);
	}

	// Define the time data (delta time + simulation speed)
	sf::Clock dtclock;
	float     speed = 10.0f;

	// Define the camera settings
	float czoom = 0.1f;

	sf::Font infofnt;
	if (!infofnt.loadFromFile("Cantarell-Bold.ttf"))
		std::cerr << "WARNING: Failed to load font." << std::endl;

	bool        fast_simulation = false;
	std::size_t ticks           = 0;

	while (win.isOpen())
	{
		auto top_car_it = std::max_element(
			cars.begin(), cars.end(), [&](const Car* a, const Car* b) { return a->fitness() < b->fitness(); });

		Car&     top_car     = **top_car_it;
		Network& top_network = networks[std::distance(cars.begin(), top_car_it)];

// @TODO move this mess
#pragma omp parallel for
		for (std::size_t i = 0; i < cars.size(); ++i)
		{
			Car&     c   = *cars[i];
			Network& net = networks[i];

			if (c.dead)
			{
				continue;
			}

			c.set_target_checkpoint(c.unit->checkpoints.at(c.reached_checkpoints() % c.unit->checkpoints.size()));

			c.compute_raycasts(*wall_body);

			c.update_inputs(net);
			net.update();
			const auto& results = net.outputs();

			/*if (false)
			{
				if (sf::Joystick::isConnected(0))
				{
					c.steer(sf::Joystick::getAxisPosition(0, sf::Joystick::X) / 100.0f);
					c.accelerate(1.0f - (sf::Joystick::getAxisPosition(0, sf::Joystick::Y) / 200.0f + 0.5f)); // wtf
					c.brake(1.0f - (sf::Joystick::getAxisPosition(0, sf::Joystick::Z) / 200.0f + 0.5f));      // wtf
				}
			}
			else if (false)
			{
				const bool forward = sf::Keyboard::isKeyPressed(sf::Keyboard::Z);
				const bool left    = sf::Keyboard::isKeyPressed(sf::Keyboard::Q);
				const bool right   = sf::Keyboard::isKeyPressed(sf::Keyboard::D);
				const bool back    = sf::Keyboard::isKeyPressed(sf::Keyboard::S);
				const bool brake   = sf::Keyboard::isKeyPressed(sf::Keyboard::C);
				const bool drift   = sf::Keyboard::isKeyPressed(sf::Keyboard::Space);

				if (forward ^ back)
				{
					c.accelerate(forward ? 1.0f : -1.0f);
				}
				else
				{
					c.accelerate(0.0f);
				}

				if (left ^ right)
				{
					c.steer(right ? 1.0f : -1.0f);
				}
				else
				{
					c.steer(0.0f);
				}

				c.brake(brake ? 1.0f : 0.0f);

				c.set_drift(drift ? 1.0f : 0.0f);
			}
			else*/
			{
				c.set_drift(static_cast<float>(results.neurons[Axon_Drift].value));
				c.steer(static_cast<float>(
					results.neurons[Axon_Steer_Right].value - results.neurons[Axon_Steer_Left].value));
				c.accelerate(
					static_cast<float>(results.neurons[Axon_Forward].value - results.neurons[Axon_Backwards].value));
				c.brake(results.neurons[Axon_Brake].value);
			}
		}

		sf::Time time = dtclock.restart();

		float real_dt = time.asSeconds();

#pragma omp parallel for
		for (std::size_t i = 0; i < sim.units.size(); ++i)
		{
			SimulationUnit& unit = sim.units[i];
			unit.world.set_dt(1.0f / 30.0f);
			unit.world.step(speed, 16, 16).update();
		}

		++ticks;
		total_time += sim.units[0].world.dt();

		if (total_time > 60.0f * 5.0f)
		{
			mutate();
		}

		win.setFramerateLimit(fast_simulation ? 0 : 80);

		if (!fast_simulation || ticks % 60 == 0)
		{
			ImGui::SFML::Update(win, time);

			for (sf::Event ev; win.pollEvent(ev);)
			{
				ImGui::SFML::ProcessEvent(ev);

				switch (ev.type)
				{
				case sf::Event::Closed: win.close(); break;

				case sf::Event::KeyPressed:
					if (ev.key.code == sf::Keyboard::Escape)
					{
						return 0;
					}
					else if (ev.key.code == sf::Keyboard::R)
					{
						return 2;
					}
					else if (ev.key.code == sf::Keyboard::F)
					{
						fast_simulation = !fast_simulation;
					}
					else if (ev.key.code == sf::Keyboard::D)
					{
						top_network.dump(std::cout);
					}
					else if (ev.key.code == sf::Keyboard::S)
					{
						std::ofstream               os("nets.bin", std::ios::binary);
						cereal::BinaryOutputArchive archive(os);
						archive(networks);
					}
					else if (ev.key.code == sf::Keyboard::L)
					{
						std::ifstream              is("nets.bin", std::ios::binary);
						cereal::BinaryInputArchive archive(is);
						archive(networks);

						mutator = {};
						reset();
					}
					break;

				case sf::Event::MouseWheelScrolled:
					czoom = std::max(czoom - (ev.mouseWheelScroll.delta * .01f), 0.01f);
					break;

				default: break;
				}
			}

			win.clear(sf::Color{20, 20, 20});
			win.draw(checkpoint_vertices);
			win.draw(wall_vertices.data(), wall_vertices.size(), sf::Lines);
			for (auto& unit : sim.units)
			{
				unit.world.render(win);
			}

			const b2Vec2 b2target = top_car.get().GetPosition();
			top_car.world().update_view(win, sf::Vector2f{b2target.x, b2target.y}, czoom);
			/*sf::View view = win.getView();
			view.rotate(top_car.get().GetAngle() * (360.0f / (2.0f * 3.14159265359f)));
			win.setView(view);*/

			sf::View cview{win.getView()};
			float    ui_scale = 1.0f;
			win.setView(sf::View{sf::FloatRect{
				0.f,
				0.f,
				static_cast<float>(win.getSize().x) * ui_scale,
				static_cast<float>(win.getSize().y) * ui_scale}});

			if (windows.draw_neural)
			{
				Visualizer{top_network}.display(win, infofnt);
			}

			if (ImGui::BeginMainMenuBar())
			{
				ImGui::TextColored(
					ImVec4(1.0, 1.0, 1.0, 0.5), "%s", fmt::format("CarNN - {:6.1f}ups", 1.0f / real_dt).c_str());
				ImGui::SameLine();

				if (ImGui::BeginMenu("View"))
				{
					ImGui::MenuItem("Simulation", nullptr, &windows.simulation_open);
					ImGui::MenuItem("Mutator", nullptr, &windows.mutator_open);
					ImGui::MenuItem("Network viz", nullptr, &windows.draw_neural);
					ImGui::EndMenu();
				}

				ImGui::EndMainMenuBar();
			}

			if (windows.simulation_open)
			{
				if (ImGui::Begin("Simulation", &windows.simulation_open))
				{
					ImGui::Checkbox("Fast simulation", &fast_simulation);

					if (ImGui::Button("Mutate current pop."))
					{
						mutate();
					}

					if (ImGui::Button("Restart current run"))
					{
						reset();
					}
				}
				ImGui::End();
			}

			if (windows.mutator_open)
			{
				if (ImGui::Begin("Mutator controls", &windows.mutator_open))
				{
					if (ImGui::Button("Load"))
					{
						mutator.settings.load_from_file();
					}
					ImGui::SameLine();
					if (ImGui::Button("Save"))
					{
						mutator.settings.save();
					}
					ImGui::SameLine();
					if (ImGui::Button("Defaults"))
					{
						mutator.settings.load_defaults();
					}
					ImGui::Separator();

					ImGui::Text("Bias");
					ImGui::PushID("Bias");
					ImGui::InputFloat("Initial stddev", &mutator.settings.bias_initial_std_dev, 0.01, 0.1, "%.3f");
					ImGui::InputFloat(
						"Soft mutation stddev", &mutator.settings.bias_mutation_factor, 0.01, 0.1, "%.3f");
					ImGui::SliderFloat(
						"Soft mutation chance", &mutator.settings.bias_mutation_chance, 0.00, 0.99, "%.3f");
					ImGui::InputFloat(
						"Hard mutation stddev", &mutator.settings.bias_hard_mutation_factor, 0.01, 0.1, "%.3f");
					ImGui::SliderFloat(
						"Hard mutation chance", &mutator.settings.bias_hard_mutation_chance, 0.00, 0.99, "%.3f");
					ImGui::PopID();
					ImGui::Separator();

					ImGui::Text("Weight");
					ImGui::PushID("Weight");
					ImGui::InputFloat("Initial stddev", &mutator.settings.weight_initial_std_dev, 0.01, 0.1, "%.3f");
					ImGui::InputFloat(
						"Soft mutation stddev", &mutator.settings.weight_mutation_factor, 0.01, 0.1, "%.3f");
					ImGui::SliderFloat(
						"Soft mutation chance", &mutator.settings.weight_mutation_chance, 0.00, 0.99, "%.3f");
					ImGui::InputFloat(
						"Hard mutation stddev", &mutator.settings.weight_hard_mutation_factor, 0.01, 0.1, "%.3f");
					ImGui::SliderFloat(
						"Hard mutation chance", &mutator.settings.weight_hard_mutation_chance, 0.00, 0.99, "%.3f");
					ImGui::PopID();
					ImGui::Separator();

					ImGui::Text("Activation method");
					ImGui::PushID("Activation method");
					ImGui::SliderFloat(
						"Mutation chance", &mutator.settings.activation_mutation_chance, 0.00, 0.99, "%.3f");
					ImGui::PopID();
					ImGui::Separator();

					ImGui::Text("Neuron creation");
					ImGui::PushID("Neuron");
					ImGui::SliderFloat("Chance", &mutator.settings.neuron_creation_chance, 0.00, 0.99, "%.3f");
					ImGui::SliderFloat(
						"Extra synapse chance", &mutator.settings.extra_synapse_connection_chance, 0.00, 0.99, "%.3f");
					ImGui::PopID();
					ImGui::Separator();

					ImGui::Text("Network simplifier");
					ImGui::PushID("Net");
					ImGui::SliderFloat(
						"Synapse destruction chance", &mutator.settings.synapse_destruction_chance, 0.00, 0.99, "%.3f");
					ImGui::SliderFloat(
						"Conservative GC chance", &mutator.settings.conservative_gc_chance, 0.00, 1.0, "%.3f");
					ImGui::SliderFloat(
						"Aggressive GC chance", &mutator.settings.aggressive_gc_chance, 0.00, 1.0, "%.3f");
					ImGui::PopID();
					ImGui::Separator();

					ImGui::SliderInt("Survivors per round", &mutator.settings.round_survivors, 1, 30);
				}

				ImGui::End();
			}

			win.setView(cview);

			ImGui::SFML::Render(win);

			win.display();
		}
	}

	return 0;
}

int main()
{
	// Define the render window
	sf::ContextSettings settings;
	settings.antialiasingLevel = 4;
	sf::RenderWindow win{sf::VideoMode{800, 600}, "Neural Network", sf::Style::Default, settings};
	ImGui::SFML::Init(win);

	while (app(win) == 2)
		;

	ImGui::SFML::Shutdown();
}
