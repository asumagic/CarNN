#pragma once

#include <cstddef>
#include <vector>

class Network;
class Car;

struct NetworkResult
{
	Network* network;
	Car* car;

	bool operator<(const NetworkResult& other) const;
	bool operator>(const NetworkResult& other) const;
};

struct MutatorSettings
{
	double bias_mutation_factor = 1.0;
	double weight_mutation_factor = 0.3;

	double mutation_rate = 1.0;
	double hard_mutation_rate = 0.5;

	double extra_synapse_connection_chance = 0.3;

	std::size_t round_survivors = 4;
};

class Mutator
{
public:
	MutatorSettings settings;

	float max_fitness = 0.0f;
	float fitness_evolution_threshold = 100.0f;

	std::size_t current_generation = 0;

	Network cross(const Network& a, const Network& b);

	void darwin(std::vector<NetworkResult> results); // TODO span

	bool should_mutate() const;
	bool should_hard_mutate() const;

	void create_random_neuron(Network& network);
	void create_random_synapse(Network& network);
	void destroy_random_synapse(Network& network);

	void mutate(Network& network);
};
