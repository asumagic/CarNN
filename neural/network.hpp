#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <cassert>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include "../randomutil.hpp"
#include "../maths.hpp"

#include "neuron.hpp"
#include "forwardsynapseidentifier.hpp"

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
	Network() = default;
	Network(std::size_t input_count, std::size_t output_count, std::size_t max_layers);

	void dump(std::ostream& stream = std::cout) const;

	NeuronLayer& inputs();
	NeuronLayer& outputs();

	NeuronIdentifier insert_random_neuron();
	NeuronIdentifier erase_random_neuron();

	std::size_t neuron_count(std::size_t first_layer, std::size_t last_layer);

	Neuron& neuron(NeuronIdentifier identifier);
	ForwardSynapse& synapse(ForwardSynapseIdentifier identifier);

	NeuronIdentifier nth_neuron(std::size_t n, std::size_t first_layer = 0);
	ForwardSynapseIdentifier nth_synapse(std::size_t n, std::size_t first_layer = 0);

	std::vector<NeuronLayer> layers;

	void update();
};

inline NeuronLayer& Network::inputs()
{
	return layers.front();
}

inline NeuronLayer& Network::outputs()
{
	return layers.back();
}

inline std::size_t Network::neuron_count(std::size_t first_layer, std::size_t last_layer)
{
	std::size_t sum = 0;

	for (std::size_t i = first_layer; i <= last_layer; ++i)
	{
		sum += layers[i].neurons.size();
	}

	return sum;
}

inline Neuron& Network::neuron(NeuronIdentifier identifier)
{
	assert(identifier.layer() < layers.size());
	assert(identifier.neuron_in_layer() < layers[identifier.layer()].neurons.size());
	return layers[identifier.layer()].neurons[identifier.neuron_in_layer()];
}

inline ForwardSynapse& Network::synapse(ForwardSynapseIdentifier identifier)
{
	return neuron(identifier.source_neuron).synapses[identifier.synapse_in_neuron];
}

inline NeuronIdentifier Network::nth_neuron(std::size_t n, std::size_t first_layer)
{
	std::size_t neurons_found = 0;

	for (std::size_t layer_identifier = first_layer;; ++layer_identifier)
	{
		const NeuronLayer& layer = layers.at(layer_identifier);

		if (neurons_found + layer.neurons.size() > n)
		{
			return NeuronIdentifier{
				layer_identifier,
				n - neurons_found
			};
		}

		neurons_found += layer.neurons.size();
	}
}

inline ForwardSynapseIdentifier Network::nth_synapse(std::size_t n, std::size_t first_layer)
{
	std::size_t synapses_found = 0;

	for (std::size_t layer_identifier = first_layer;; ++layer_identifier)
	for (std::size_t neuron_identifier = 0; neuron_identifier < layers[layer_identifier].neurons.size(); ++neuron_identifier)
	{
		const Neuron& neuron = layers.at(layer_identifier).neurons.at(neuron_identifier);

		if (synapses_found + neuron.synapses.size() > n)
		{
			return ForwardSynapseIdentifier{
				{layer_identifier, neuron_identifier},
				n - synapses_found
			};
		}

		synapses_found += neuron.synapses.size();
	}
}

inline void Network::update()
{
	for (NeuronLayer& layer : layers)
	for (Neuron& neuron : layer.neurons)
	{
		// hack lol
		if (&layer != &layers.front())
		{
			neuron.compute_value();
			neuron.partial_activation = 0.0f;
		}

		neuron.propagate_forward(*this);
	}
}


#endif // NETWORK_HPP
