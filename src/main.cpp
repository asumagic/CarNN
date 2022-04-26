#include "imgui.h"
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <carnn/neural/network.hpp>
#include <carnn/neural/visualizer.hpp>
#include <carnn/sim/entities/car.hpp>
#include <carnn/sim/entities/wheel.hpp>
#include <carnn/sim/entities/checkpoint.hpp>
#include <carnn/sim/individual.hpp>
#include <carnn/sim/simulationunit.hpp>
#include <carnn/sim/world.hpp>
#include <carnn/training/mutator.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <carnn/util/maths.hpp>
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

enum class SimulationState
{
	Realtime,
	Fast,
	Paused
};

using namespace neural;
using namespace sim;
using namespace sim::entities;
using namespace training;

class App
{
	public:
	App();
	~App();

	void run();

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(_population), CEREAL_NVP(_mutator));
	}

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

	Simulation      _sim;
	SimulationState _simulation_state = SimulationState::Realtime;

	std::vector<Individual> _population;
	Mutator                 _mutator;

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
		switch (_simulation_state)
		{
		case SimulationState::Realtime:
		{
			advance_simulation(1);
			break;
		}

		case SimulationState::Fast:
		{
			advance_simulation(1000);
		}

		case SimulationState::Paused:
		default: break;
		}

		frame();
	}
}

sf::ContextSettings App::default_context_settings()
{
	sf::ContextSettings ret;
	ret.antialiasingLevel = 2;
	return ret;
}

void App::advance_simulation(std::size_t ticks)
{
	tbb::parallel_for(tbb::blocked_range(_sim.units.begin(), _sim.units.end()), [&](const auto& range) {
		for (SimulationUnit& unit : range)
		{
			for (std::size_t i = 0; i < ticks; ++i)
			{
				tick(unit);

				// HACK:
				if (unit.seconds_elapsed > 60.0f * 5.0f)
				{
					break;
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
	_window.setFramerateLimit(_simulation_state != SimulationState::Fast ? 80 : 0);

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
				_simulation_state
					= _simulation_state == SimulationState::Fast ? SimulationState::Realtime : SimulationState::Fast;
			}
			else if (ev.key.code == sf::Keyboard::S)
			{
				std::ofstream               os("nets.bin", std::ios::binary);
				cereal::BinaryOutputArchive archive(os);
				archive(*this);
			}
			else if (ev.key.code == sf::Keyboard::L)
			{
				std::ifstream              is("nets.bin", std::ios::binary);
				cereal::BinaryInputArchive archive(is);
				archive(*this);

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

	std::vector<Individual*> rendered_individuals(_population.size());
	for (std::size_t i = 0; i < _population.size(); ++i)
	{
		Individual& individual = _population[i];
		rendered_individuals[i] = &individual;

		Car& c = *_sim.cars[individual.car_id];
		if (!c.dead)
		{
			rendered_individuals.push_back(&individual);
		}
	}

	tbb::parallel_sort(
		rendered_individuals.begin(),
		rendered_individuals.end(),
		[&](const Individual* a, const Individual* b) {
			// TODO: pls make a method for this jesus
			Car& ac = *_sim.cars[a->car_id];
			Car& bc = *_sim.cars[b->car_id];
			return (ac.fitness() > bc.fitness());
		}
	);
	_tracked_individual = rendered_individuals[0];

	std::size_t rendered = 0;
	for (auto it = rendered_individuals.begin(); it != rendered_individuals.end(); ++it)
	{
		//if (*it != _tracked_individual) continue;

		Car& c = *_sim.cars[(*it)->car_id];

		if (c.dead)
		{
			c.with_color(sf::Color{200, 0, 0, 60}, 255);
		}

		if ((*it)->survivor_from_last)
		{
			c.with_color(sf::Color{200, 50, 0, 200}, 255);
		}
		else
		{
			c.with_color(sf::Color{0, 0, 100, 40}, 255);
		}

		c.render(_window);

		for (auto& wheel : c.get_wheels())
		{
			wheel->render(_window);
		}

		++rendered;

		if (rendered >= 100)
		{
			break;
		}
	}

	/*for (auto it = split_it; it != rendered_individuals.end(); ++it)
	{
		Car& c = *_sim.cars[(*it)->car_id];
		c.fast_render(_window);
	}*/

	/*for (auto& unit : _sim.units)
	{
		unit.world.render(_window);
	}*/

	if (_tracked_individual != nullptr)
	{
		Car& car = *_sim.cars[_tracked_individual->car_id];

		const b2Vec2 b2target = car.get().GetPosition();
		car.world().update_view(_window, sf::Vector2f{b2target.x, b2target.y}, _czoom);

		/*sf::View view = _window.getView();
		view.rotate(car.get().GetAngle() * (360.0f / (2.0f * 3.14159265359f)));
		_window.setView(view);*/
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
			if (ImGui::RadioButton("Realtime", _simulation_state == SimulationState::Realtime))
			{
				_simulation_state = SimulationState::Realtime;
			}

			ImGui::SameLine();
			if (ImGui::RadioButton("Fast", _simulation_state == SimulationState::Fast))
			{
				_simulation_state = SimulationState::Fast;
			}

			ImGui::SameLine();
			if (ImGui::RadioButton("Pause", _simulation_state == SimulationState::Paused))
			{
				_simulation_state = SimulationState::Paused;
			}

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

			const auto tooltip = [&](const char* text) {
				if (ImGui::IsItemHovered())
				{
					ImGui::SetNextWindowSize(ImVec2(800, -1));
					ImGui::BeginTooltip();
					ImGui::TextWrapped("%s", text);
					ImGui::EndTooltip();
				}
			};

			auto& cfg = _mutator.settings;

			ImGui::Text("Bias");
			ImGui::PushID("Bias");
			ImGui::InputFloat("Initial stddev", &cfg.bias_initial_std_dev, 0.01, 0.1, "%.3f");
			ImGui::InputFloat("Soft mutation stddev", &cfg.bias_mutation_factor, 0.01, 0.1, "%.3f");
			ImGui::SliderFloat("Soft mutation chance", &cfg.bias_mutation_chance, 0.00, 0.99, "%.3f");
			ImGui::InputFloat("Hard mutation stddev", &cfg.bias_hard_mutation_factor, 0.01, 0.1, "%.3f");
			ImGui::SliderFloat("Hard mutation chance", &cfg.bias_hard_mutation_chance, 0.00, 0.99, "%.3f");
			ImGui::PopID();
			ImGui::Separator();

			ImGui::Text("Weight");
			ImGui::PushID("Weight");
			ImGui::InputFloat("Initial stddev", &cfg.weight_initial_std_dev, 0.01, 0.1, "%.3f");
			ImGui::InputFloat("Soft mutation stddev", &cfg.weight_mutation_factor, 0.01, 0.1, "%.3f");
			ImGui::SliderFloat("Soft mutation chance", &cfg.weight_mutation_chance, 0.00, 0.99, "%.3f");
			ImGui::InputFloat("Hard mutation stddev", &cfg.weight_hard_mutation_factor, 0.01, 0.1, "%.3f");
			ImGui::SliderFloat("Hard mutation chance", &cfg.weight_hard_mutation_chance, 0.00, 0.99, "%.3f");
			ImGui::PopID();
			ImGui::Separator();

			ImGui::Text("Activation method");
			ImGui::PushID("Activation method");
			ImGui::SliderFloat("Mutation chance", &cfg.activation_mutation_chance, 0.00, 0.99, "%.3f");
			ImGui::PopID();
			ImGui::Separator();

			ImGui::Text("Neuron creation");
			ImGui::PushID("Neuron");
			ImGui::SliderFloat("Chance", &cfg.neuron_creation_chance, 0.00, 0.99, "%.3f");
			ImGui::SliderInt("Max extra synapses", &cfg.max_extra_synapses, 0, 100);
			ImGui::Separator();
			ImGui::PopID();

			ImGui::PushID("Crossing");
			ImGui::Text("Genome crossing");
			tooltip(
				"When generating a new population, the mutator selects a random pair of individuals.\n"
				"Their respective networks are then \"merged\" in order to generate a new individual.\n"
				"The two individuals in the pair can be the same one, meaning self-breeding is allowed.\n"
				"\n"
				"Breeding is \"unfair\" in the way that the 2nd individual in the pair has less chance to have its "
				"features imported into the new individual than the 1st.");

			ImGui::SliderFloat("Max imported synapses factor", &cfg.max_imported_synapses_factor, 0.0, 1.0);
			tooltip(
				"Whether hybridization occurs or not, an amount of random synapses from the 2nd network will be copied "
				"into the new individual.");

			ImGui::SliderFloat("Hybridization chance", &cfg.hybridization_chance, 0.0, 0.99);
			tooltip(
				"Hybridization occurs when trying to cross two networks with a different topology.\n"
				"In other words, given a breeding pair of networks (a, b), this is the chance that neurons present in "
				"b (but not present in a) to be copied into the new individual.");

			ImGui::SliderFloat("Max hybrid. divergence", &cfg.max_hybridization_divergence_factor, 0.0, 5.0);
			tooltip(
				"Defines the maximum possible divergence factor between two genomes for hybridization.\n"
				"Currently, the divergence factor is equal to the difference in neuron count between two genomes.");

			ImGui::PopID();
			ImGui::Separator();

			ImGui::SliderInt("Survivors per round", &cfg.round_survivors, 1, 100);
			tooltip("The size of the interbreeding population to select at the end of each round.");
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
		return;
	}

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
		for (std::size_t i = 0; i < total_rays; ++i)
		{
			c._ray_angles[i] = util::lerp(float(c._ray_angles[i]), std::clamp(results[Axon_FirstRay + i].value, 0.0f, 1.0f), 0.1f);
		}
	}
}

void App::tick(SimulationUnit& unit)
{
	for (auto& car : unit.cars)
	{
		tick(*car->individual);
	}

	unit.world.step(10.0f / 30.0f, 1, 1).update();
	++unit.ticks_elapsed;
	unit.seconds_elapsed += 1.0f / 30.0f;
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

		individual.network = Network(total_rays + 4, total_rays + 6);
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
