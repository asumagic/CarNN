#include "entities/car.hpp"
#include "entities/checkpoint.hpp"
#include "entities/wheel.hpp"
#include "imgui.h"
#include "neural/individual.hpp"
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
#include <spdlog/spdlog.h>
#include <tbb/tbb.h>

struct GuiWindows
{
	bool simulation_open = false;
	bool mutator_open    = false;
	bool draw_neural     = false;
};

class App
{
	public:
	App();
	~App();

	void run();

	private:
	static sf::ContextSettings default_context_settings();

	void advance_simulation(std::size_t ticks = 1);
	void frame();

	void tick(Individual& individual);
	void tick(SimulationUnit& unit);

	void start_new_run();
	void mutate_and_restart();

	void reset_individuals();
	void load_fonts();

	sf::RenderWindow _window;

	Simulation _sim;
	bool       _fast_simulation = false;

	std::vector<Individual> _population;

	Mutator _mutator;

	GuiWindows _gui;

	sf::Font _font;

	sf::Clock _frame_dt_clock;
	sf::Clock _sim_dt_clock;

	float _ups = 0.0f;

	// TODO: move camera stuff elsewhere
	float       _czoom              = 0.1f;
	Individual* _tracked_individual = nullptr;
};

App::App() : _window(sf::VideoMode(800, 600), "carnn", sf::Style::Default, default_context_settings())
{
	ImGui::SFML::Init(_window, false);
	load_fonts();
	_mutator.settings.load_from_file();
	reset_individuals();
}

App::~App() { ImGui::SFML::Shutdown(); }

void App::run()
{
	_tracked_individual = &_population[0];

	start_new_run();

	while (_window.isOpen())
	{
		advance_simulation(_fast_simulation ? 1000 : 1);
		frame();
	}
}

sf::ContextSettings App::default_context_settings()
{
	sf::ContextSettings ret;
	ret.antialiasingLevel = 4;
	return ret;
}

void App::advance_simulation(std::size_t ticks)
{
	tbb::parallel_for(tbb::blocked_range(_sim.units.begin(), _sim.units.end()), [&](const auto& range) {
		for (std::size_t i = 0; i < ticks; ++i)
		{
			for (SimulationUnit& unit : range)
			{
				tick(unit);

				// HACK:
				if (unit.seconds_elapsed > 60.0f * 5.0f)
				{
					return;
				}
			}
		}
	});

	// FIXME: this loses precision
	const float total_time = _sim.units[0].seconds_elapsed;

	if (total_time > 60.0f * 5.0f)
	{
		mutate_and_restart();
	}

	const sf::Time sim_time = _sim_dt_clock.restart();
	_ups                    = float(ticks) * 1.0f / sim_time.asSeconds();
}

void App::frame()
{
	_window.setFramerateLimit(!_fast_simulation ? 80 : 0);

	const sf::Time frame_time = _frame_dt_clock.restart();

	ImGui::SFML::Update(_window, frame_time);

	for (sf::Event ev; _window.pollEvent(ev);)
	{
		ImGui::SFML::ProcessEvent(ev);

		switch (ev.type)
		{
		case sf::Event::Closed:
		{
			_window.close();
			break;
		}

		case sf::Event::KeyPressed:
			if (ev.key.code == sf::Keyboard::Escape)
			{
				_window.close();
			}
			else if (ev.key.code == sf::Keyboard::R)
			{
				// FIXME: this should be reimplemented
				return;
			}
			else if (ev.key.code == sf::Keyboard::F)
			{
				_fast_simulation = !_fast_simulation;
			}
			else if (ev.key.code == sf::Keyboard::S)
			{
				std::ofstream               os("nets.bin", std::ios::binary);
				cereal::BinaryOutputArchive archive(os);
				archive(_population);
			}
			else if (ev.key.code == sf::Keyboard::L)
			{
				std::ifstream              is("nets.bin", std::ios::binary);
				cereal::BinaryInputArchive archive(is);
				archive(_population);

				start_new_run();
			}
			break;

		case sf::Event::MouseWheelScrolled:
		{
			_czoom = std::max(_czoom - (ev.mouseWheelScroll.delta * .01f), 0.01f);
			break;
		}

		default: break;
		}
	}

	_window.clear(sf::Color{20, 20, 20});
	_window.draw(_sim.checkpoint_vertices);
	_window.draw(_sim.wall_vertices.data(), _sim.wall_vertices.size(), sf::Lines);
	for (auto& unit : _sim.units)
	{
		unit.world.render(_window);
	}

	if (_tracked_individual != nullptr)
	{
		Car& car = *_sim.cars[_tracked_individual->car_id];

		const b2Vec2 b2target = car.get().GetPosition();
		car.world().update_view(_window, sf::Vector2f{b2target.x, b2target.y}, _czoom);
		/*sf::View view = win.getView();
		view.rotate(top_car.get().GetAngle() * (360.0f / (2.0f * 3.14159265359f)));
		win.setView(view);*/
	}

	sf::View world_view(_window.getView());

	float ui_scale = 1.0f;
	_window.setView(sf::View{
		sf::FloatRect{0.f, 0.f, float(_window.getSize().x) * ui_scale, float(_window.getSize().y) * ui_scale}});

	if (_tracked_individual != nullptr && _gui.draw_neural)
	{
		Visualizer(_tracked_individual->network).display(_window, _font);
	}

	if (ImGui::BeginMainMenuBar())
	{
		ImGui::TextColored(
			ImVec4(1.0, 1.0, 1.0, 0.5),
			"%s",
			fmt::format("CarNN - {:6.1f}fps - {:6.1f}ups", 1.0f / frame_time.asSeconds(), _ups).c_str());
		ImGui::SameLine();

		if (ImGui::BeginMenu("View"))
		{
			ImGui::MenuItem("Simulation", nullptr, &_gui.simulation_open);
			ImGui::MenuItem("Mutator", nullptr, &_gui.mutator_open);
			ImGui::MenuItem("Network viz", nullptr, &_gui.draw_neural);
			ImGui::EndMenu();
		}

		ImGui::EndMainMenuBar();
	}

	if (_gui.simulation_open)
	{
		if (ImGui::Begin("Simulation", &_gui.simulation_open))
		{
			ImGui::Checkbox("Fast simulation", &_fast_simulation);

			if (ImGui::Button("Mutate current pop."))
			{
				mutate_and_restart();
			}

			if (ImGui::Button("Restart current run"))
			{
				start_new_run();
			}
		}
		ImGui::End();
	}

	if (_gui.mutator_open)
	{
		if (ImGui::Begin("Mutator controls", &_gui.mutator_open))
		{
			if (ImGui::Button("Load"))
			{
				_mutator.settings.load_from_file();
			}
			ImGui::SameLine();
			if (ImGui::Button("Save"))
			{
				_mutator.settings.save();
			}
			ImGui::SameLine();
			if (ImGui::Button("Defaults"))
			{
				_mutator.settings.load_defaults();
			}
			ImGui::Separator();

			ImGui::Text("Bias");
			ImGui::PushID("Bias");
			ImGui::InputFloat("Initial stddev", &_mutator.settings.bias_initial_std_dev, 0.01, 0.1, "%.3f");
			ImGui::InputFloat("Soft mutation stddev", &_mutator.settings.bias_mutation_factor, 0.01, 0.1, "%.3f");
			ImGui::SliderFloat("Soft mutation chance", &_mutator.settings.bias_mutation_chance, 0.00, 0.99, "%.3f");
			ImGui::InputFloat("Hard mutation stddev", &_mutator.settings.bias_hard_mutation_factor, 0.01, 0.1, "%.3f");
			ImGui::SliderFloat(
				"Hard mutation chance", &_mutator.settings.bias_hard_mutation_chance, 0.00, 0.99, "%.3f");
			ImGui::PopID();
			ImGui::Separator();

			ImGui::Text("Weight");
			ImGui::PushID("Weight");
			ImGui::InputFloat("Initial stddev", &_mutator.settings.weight_initial_std_dev, 0.01, 0.1, "%.3f");
			ImGui::InputFloat("Soft mutation stddev", &_mutator.settings.weight_mutation_factor, 0.01, 0.1, "%.3f");
			ImGui::SliderFloat("Soft mutation chance", &_mutator.settings.weight_mutation_chance, 0.00, 0.99, "%.3f");
			ImGui::InputFloat(
				"Hard mutation stddev", &_mutator.settings.weight_hard_mutation_factor, 0.01, 0.1, "%.3f");
			ImGui::SliderFloat(
				"Hard mutation chance", &_mutator.settings.weight_hard_mutation_chance, 0.00, 0.99, "%.3f");
			ImGui::PopID();
			ImGui::Separator();

			ImGui::Text("Activation method");
			ImGui::PushID("Activation method");
			ImGui::SliderFloat("Mutation chance", &_mutator.settings.activation_mutation_chance, 0.00, 0.99, "%.3f");
			ImGui::PopID();
			ImGui::Separator();

			ImGui::Text("Neuron creation");
			ImGui::PushID("Neuron");
			ImGui::SliderFloat("Chance", &_mutator.settings.neuron_creation_chance, 0.00, 0.99, "%.3f");
			ImGui::SliderFloat(
				"Extra synapse chance", &_mutator.settings.extra_synapse_connection_chance, 0.00, 0.99, "%.3f");
			ImGui::PopID();
			ImGui::Separator();

			ImGui::SliderInt("Survivors per round", &_mutator.settings.round_survivors, 1, 100);
		}

		ImGui::End();
	}

	_window.setView(world_view);

	ImGui::SFML::Render(_window);

	_window.display();
}

void App::tick(Individual& individual)
{
	Car&     c   = *_sim.cars[individual.car_id];
	Network& net = individual.network;

	if (c.dead)
	{
		c.with_color(sf::Color{200, 0, 0, 60}, 255);
		return;
	}

	if (individual.survivor_from_last)
	{
		c.with_color(sf::Color{200, 50, 0, 200}, 255);
	}
	else
	{
		c.with_color(sf::Color{0, 0, 100, 40}, 255);
	}

	c.set_target_checkpoint(c.unit->checkpoints.at(c.reached_checkpoints() % c.unit->checkpoints.size()));

	c.compute_raycasts();

	c.update_inputs(net);
	net.update();
	const auto& results = net.outputs();

	/*if (false)
	{
		if (sf::Joystick::isConnected(0))
		{
			c.steer(sf::Joystick::getAxisPosition(0, sf::Joystick::X) / 100.0f);
			c.accelerate(1.0f - (sf::Joystick::getAxisPosition(0, sf::Joystick::Y) / 200.0f + 0.5f)); //
	wtf c.brake(1.0f - (sf::Joystick::getAxisPosition(0, sf::Joystick::Z) / 200.0f + 0.5f));      // wtf
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
		c.set_drift(static_cast<float>(results[Axon_Drift].value));
		c.steer(static_cast<float>(results[Axon_Steer_Right].value - results[Axon_Steer_Left].value));
		c.accelerate(static_cast<float>(results[Axon_Forward].value - results[Axon_Backwards].value));
		c.brake(results[Axon_Brake].value);
	}
}

void App::tick(SimulationUnit& unit)
{
	for (auto& car : unit.cars)
	{
		tick(*car->individual);
	}

	unit.world.set_dt(1.0f / 30.0f);
	unit.world.step(10.0f, 1, 1).update();
	++unit.ticks_elapsed;
	unit.seconds_elapsed += unit.world.dt();
}

void App::start_new_run()
{
	_sim = {};

	_tracked_individual = &_population[0];

	for (auto& individual : _population)
	{
		individual.network.reset_values();
		_sim.cars[individual.car_id]->individual = &individual;
	}
}

void App::mutate_and_restart()
{
	_mutator.darwin(_sim, _population);
	start_new_run();
}

void App::reset_individuals()
{
	_population.clear();
	_population.resize(_sim.cars.size());

	for (std::size_t i = 0; i < _population.size(); ++i)
	{
		Individual& individual = _population[i];

		individual.car_id = i;

		individual.network = Network(total_rays + 4, 6);
		_mutator.randomize(individual.network);
		/*
				auto& inputs  = individual.network.inputs().neurons;
				auto& outputs = individual.network.outputs().neurons;

				inputs[0].label = "vector to objective (x)";
				inputs[1].label = "vector to objective (y)";
				inputs[2].label = "velocity (forward)";
				inputs[3].label = "velocity (lateral)";
				for (std::size_t lidar_index = 4; lidar_index < inputs.size(); ++lidar_index)
				{
					inputs[lidar_index].label = fmt::format("lidar #{}", lidar_index - 4 + 1);
				}

				outputs[Axon_Forward].label     = "Forward";
				outputs[Axon_Backwards].label   = "Backwards";
				outputs[Axon_Brake].label       = "Brake force";
				outputs[Axon_Steer_Left].label  = "Left steering";
				outputs[Axon_Steer_Right].label = "Right steering";
				outputs[Axon_Drift].label       = "Drifting";*/
	}
}

void App::load_fonts()
{
	// TODO: cleaner error checking here

	const char* gui_font_path = "Tamzen8x16r.ttf";

	if (!_font.loadFromFile(gui_font_path))
	{
		spdlog::error("failed to load GUI font from file '{}'", gui_font_path);
	}

	{
		auto& io = ImGui::GetIO();
		io.Fonts->AddFontFromFileTTF("Tamzen8x16r.ttf", 16.0f);
		io.Fonts->AddFontFromFileTTF("Tamzen8x16b.ttf", 16.0f);

		ImGui::SFML::UpdateFontTexture();
	}
}

int main()
{
	tbb::task_scheduler_init init;

	App app;
	app.run();
}
