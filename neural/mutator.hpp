#pragma once

#include <cstddef>
#include <vector>

class Network;
class Car;
struct Neuron;
struct ForwardSynapse;

struct NetworkResult
{
	Network* network;
	Car*     car;

	bool operator<(const NetworkResult& other) const;
	bool operator>(const NetworkResult& other) const;
};

struct MutatorSettings
{
	double bias_initial_std_dev      = 0.2;
	double bias_mutation_factor      = 0.03;
	double bias_mutation_chance      = 0.8;
	double bias_hard_mutation_factor = 0.3;
	double bias_hard_mutation_chance = 0.1;

	double weight_initial_std_dev      = 0.2;
	double weight_mutation_factor      = 0.03;
	double weight_mutation_chance      = 0.8;
	double weight_hard_mutation_factor = 0.3;
	double weight_hard_mutation_chance = 0.1;

	double extra_synapse_connection_chance = 0.5;

	double neuron_creation_chance     = 0.3;
	double synapse_destruction_chance = 0.1;

	double conservative_gc_chance = 0.5;
	double aggressive_gc_chance   = 0.15;

	std::size_t round_survivors = 15;
};

class Mutator
{
	public:
	MutatorSettings settings;

	float max_fitness                 = 0.0f;
	float fitness_evolution_threshold = 100.0f;

	std::size_t current_generation = 0;

	Network cross(const Network& a, const Network& b);

	void darwin(std::vector<NetworkResult> results); // TODO span

	void create_random_neuron(Network& network);
	void create_random_synapse(Network& network);
	void destroy_random_synapse(Network& network);

	void mutate(Network& network);

	void randomize(Network& network);
	void randomize(Neuron& neuron);
	void randomize(ForwardSynapse& synapse);

	void gc(Network& network, bool aggressive);
};
