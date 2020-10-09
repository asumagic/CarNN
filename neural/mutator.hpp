#pragma once

#include "neuron.hpp"
#include <cereal/cereal.hpp>
#include <cstddef>
#include <vector>

class Network;
class Car;
struct Neuron;
struct Synapse;
class Simulation;
struct Individual;

enum class InitialTopology
{
	FullyConnected,
	PartiallyConnected,
	DisconnectedWithRandomNeurons,

	Total
};

struct MutatorSettings
{
	using Fp = float;

	Fp bias_initial_std_dev      = 0.05;
	Fp bias_mutation_factor      = 0.07;
	Fp bias_mutation_chance      = 0.7;
	Fp bias_hard_mutation_factor = 0.2;
	Fp bias_hard_mutation_chance = 0.2;

	Fp weight_initial_std_dev      = 0.05;
	Fp weight_mutation_factor      = 0.07;
	Fp weight_mutation_chance      = 0.7;
	Fp weight_hard_mutation_factor = 0.2;
	Fp weight_hard_mutation_chance = 0.2;

	Fp activation_mutation_chance = 0.3;

	Fp neuron_creation_chance          = 0.1;
	Fp extra_synapse_connection_chance = 0.5;

	Fp synapse_destruction_chance = 0.1;

	Fp conservative_gc_chance = 0.0;
	Fp aggressive_gc_chance   = 0.0;

	std::int32_t round_survivors = 10;

	bool load_from_file();
	bool save();
	void load_defaults();

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(bias_initial_std_dev),
		   CEREAL_NVP(bias_mutation_factor),
		   CEREAL_NVP(bias_mutation_chance),
		   CEREAL_NVP(bias_hard_mutation_factor),
		   CEREAL_NVP(bias_hard_mutation_chance),

		   CEREAL_NVP(weight_initial_std_dev),
		   CEREAL_NVP(weight_mutation_factor),
		   CEREAL_NVP(weight_mutation_chance),
		   CEREAL_NVP(weight_hard_mutation_factor),
		   CEREAL_NVP(weight_hard_mutation_chance),

		   CEREAL_NVP(activation_mutation_chance),

		   CEREAL_NVP(neuron_creation_chance),
		   CEREAL_NVP(extra_synapse_connection_chance),

		   CEREAL_NVP(synapse_destruction_chance),

		   CEREAL_NVP(conservative_gc_chance),
		   CEREAL_NVP(aggressive_gc_chance),

		   CEREAL_NVP(round_survivors));
	}
};

class Mutator
{
	public:
	MutatorSettings settings;

	float max_fitness                 = 0.0f;
	float fitness_evolution_threshold = 10.0f;

	std::size_t current_generation = 0;

	Network cross(const Network& a, const Network& b);

	void darwin(Simulation& sim, std::vector<Individual>& results);

	void create_random_neuron(Network& network);
	void create_random_synapse(Network& network);
	void destroy_random_synapse(Network& network);

	void mutate(Network& network);

	void randomize(Network& network);
	void randomize(Neuron& neuron);
	void randomize(Synapse& synapse);

	ActivationMethod random_activation_method();

	void gc(Network& network, bool aggressive);
};
