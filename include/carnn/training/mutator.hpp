#pragma once

#include <carnn/neural/activationmethod.hpp>
#include <carnn/neural/fwd.hpp>
#include <carnn/sim/fwd.hpp>
#include <carnn/training/settings.hpp>
#include <cereal/cereal.hpp>
#include <cstdint>
#include <vector>

namespace training
{
class Mutator
{
	public:
	Settings settings;

	float max_fitness                 = 0.0f;
	float fitness_evolution_threshold = 4.0f;

	std::size_t current_generation = 0;

	// TODO: less obviously magic value. this is because inputs and outputs have hardcoded ids so let's give them some
	//       space
	std::uint32_t current_evolution_id = 10000;

	neural::Network cross(neural::Network a, const neural::Network& b);

	void darwin(sim::Simulation& sim, std::vector<sim::Individual>& results);

	void create_random_neuron(neural::Network& network);
	void create_random_synapse(neural::Network& network);

	void mutate(neural::Network& network);

	void randomize(neural::Network& network);
	void randomize(neural::Neuron& neuron);
	void randomize(neural::Synapse& synapse);

	neural::ActivationMethod random_activation_method();

	std::uint32_t get_unique_evolution_id();

	double get_divergence_factor(const neural::Network& a, const neural::Network& b);

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(settings),
		   CEREAL_NVP(max_fitness),
		   CEREAL_NVP(fitness_evolution_threshold),
		   CEREAL_NVP(current_generation),
		   CEREAL_NVP(current_evolution_id));
	}
};
} // namespace training
