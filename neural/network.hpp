#pragma once

#include "../maths.hpp"
#include "../randomutil.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "forwardsynapseidentifier.hpp"
#include "neuron.hpp"

class Car;

struct Neuron;
class Network;

struct NeuronLayer
{
	std::vector<Neuron> neurons;
};

class Network
{
	public:
	Network(std::size_t input_count, std::size_t output_count);

	void dump(std::ostream& stream = std::cout) const;

	NeuronLayer& inputs();
	NeuronLayer& outputs();

	NeuronIdentifier insert_random_neuron();
	NeuronIdentifier erase_random_neuron();

	Neuron&         neuron(NeuronIdentifier identifier);
	ForwardSynapse& synapse(ForwardSynapseIdentifier identifier);

	NeuronIdentifier         nth_neuron(std::size_t n, std::size_t first_layer = 0);
	ForwardSynapseIdentifier nth_synapse(std::size_t n, std::size_t first_layer = 0);

	std::size_t      neuron_count(std::size_t first_layer, std::size_t last_layer);
	NeuronIdentifier random_neuron(std::size_t first_layer, std::size_t last_layer);

	std::size_t              synapse_count(std::size_t first_layer, std::size_t last_layer);
	ForwardSynapseIdentifier random_synapse(std::size_t first_layer, std::size_t last_layer);

	std::array<NeuronLayer, 3> layers{};

	void update();

	void reset_values();

	private:
	void sanitize(NeuronIdentifier identifier);
	void sanitize(ForwardSynapseIdentifier identifier);
};

inline NeuronLayer& Network::inputs() { return layers.front(); }

inline NeuronLayer& Network::outputs() { return layers.back(); }

inline Neuron& Network::neuron(NeuronIdentifier identifier)
{
	sanitize(identifier);
	return layers[identifier.layer()].neurons[identifier.neuron_in_layer()];
}

inline ForwardSynapse& Network::synapse(ForwardSynapseIdentifier identifier)
{
	sanitize(identifier);
	return neuron(identifier.source_neuron).synapses[identifier.synapse_in_neuron];
}

inline void Network::sanitize([[maybe_unused]] NeuronIdentifier identifier)
{
	assert(identifier.layer() < layers.size());
	assert(identifier.neuron_in_layer() < layers[identifier.layer()].neurons.size());
}

inline void Network::sanitize([[maybe_unused]] ForwardSynapseIdentifier identifier)
{
	assert(identifier.synapse_in_neuron < neuron(identifier.source_neuron).synapses.size());
}
