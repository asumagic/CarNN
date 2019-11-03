#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <cassert>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include "../randomutil.hpp"

class Car;

template<class T>
inline T sigmoid(T v)
{
	return 1.0 / (1.0 + std::exp(-v));
}

struct Neuron;
class Network;

struct NeuronIdentifier
{
	static constexpr std::size_t max_neurons_per_layer = 1024;

	std::size_t value;

	NeuronIdentifier() = default;

	explicit NeuronIdentifier(std::size_t value) :
		value{value}
	{}

	NeuronIdentifier(std::size_t layer, std::size_t neuron_in_layer) :
		value{layer * max_neurons_per_layer + neuron_in_layer}
	{
		assert(neuron_in_layer < max_neurons_per_layer);
	}

	std::size_t layer() const
	{
		return value / max_neurons_per_layer;
	}

	std::size_t neuron_in_layer() const
	{
		return value % max_neurons_per_layer;
	}
};

struct ForwardSynapseIdentifier
{
	NeuronIdentifier source_neuron;
	std::size_t synapse_in_neuron;
};

struct ForwardSynapse
{
	NeuronIdentifier forward_neuron_identifier;
	double weight;

	void randomize_parameters();
};

inline void ForwardSynapse::randomize_parameters()
{
	// TODO: do we really want a uniform distribution?
	weight = random_double(-2.0, 2.0);
}

struct Neuron
{
	double partial_activation = 0.0;
	double value;
	double bias = 0.0;
	std::vector<ForwardSynapse> synapses;

	void compute_value();
	void propagate_forward(Network& network);

	void randomize_parameters();
};

struct NeuronLayer
{
	std::vector<Neuron> neurons;
};

class Network
{
public:
	Network() = default;
	Network(std::size_t input_count, std::size_t output_count, std::size_t max_layers);

	void dump() const;

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

inline void Neuron::compute_value()
{
	value = sigmoid(partial_activation + bias);
}

inline void Neuron::propagate_forward(Network& network)
{
	for (ForwardSynapse& synapse : synapses)
	{
		Neuron& forward_neuron = network.neuron(synapse.forward_neuron_identifier);
		forward_neuron.partial_activation += value * synapse.weight;
	}
}

inline void Neuron::randomize_parameters()
{
	// TODO: do we really want a uniform distribution?
	bias = random_double(-3.0, 3.0);
}

inline Network::Network(std::size_t input_count, std::size_t output_count, std::size_t max_hidden) :
	layers(2 + max_hidden)
{
	auto &inputs = layers.front(), &outputs = layers.back();

	inputs.neurons.resize(input_count);
	outputs.neurons.resize(output_count);

	for (auto& input : inputs.neurons)
	{
		input.randomize_parameters();

		input.synapses.resize(output_count);

		for (std::size_t i = 0; i < output_count; ++i)
		{
			ForwardSynapse& synapse = input.synapses[i];
			synapse.forward_neuron_identifier = {layers.size() - 1, i};
			synapse.randomize_parameters();
		}
	}

	for (auto& output : outputs.neurons)
	{
		output.randomize_parameters();
	}
}

inline void Network::dump() const
{
	std::cout << "=========== GRAPHVIZ BEGIN\n";
	std::cout << "digraph G {\n";

	for (std::size_t layer_identifier = 0; layer_identifier < layers.size(); ++layer_identifier)
	for (std::size_t neuron_identifier = 0; neuron_identifier < layers[layer_identifier].neurons.size(); ++neuron_identifier)
	{
		NeuronIdentifier current_identifier{layer_identifier, neuron_identifier};
		const Neuron& current = layers[layer_identifier].neurons[neuron_identifier];

		std::cout << current_identifier.value << " [label=\"" << current.bias << "\"];\n";

		for (const ForwardSynapse& synapse : current.synapses)
		{
			std::cout << current_identifier.value << " -> " << synapse.forward_neuron_identifier.value << " [label=\"" << synapse.weight << "\"];\n";
		}
	}

	std::cout << "}\n";
	std::cout << "=========== GRAPHVIZ END\n";
}

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

struct NetworkResult
{
	Network* network;
	Car* car;

	bool operator<(const NetworkResult& other) const;
	bool operator>(const NetworkResult& other) const;
};

class Mutator
{
public:
	double bias_mutation_factor = 1.0;
	double weight_mutation_factor = 0.3;

	double mutation_rate = 1.0;
	double hard_mutation_rate = 0.5;

	double extra_synapse_connection_chance = 0.3;

	float max_fitness = 0.0f;
	float fitness_evolution_threshold = 100.0f;

	std::size_t current_generation = 0;

	std::size_t noah = 7;

	Network cross(const Network& a, const Network& b);

	void darwin(std::vector<NetworkResult> results);

	bool should_mutate() const
	{
		if (mutation_rate >= 1.0)
		{
			return true;
		}

		return random_double() < mutation_rate;
	}

	bool should_hard_mutate() const
	{
		if (hard_mutation_rate >= 0.9999)
		{
			throw std::runtime_error{"Hard mutation rate must be <1, otherwise the mutator would loop infinitely"};
		}

		return random_double() < hard_mutation_rate;
	}

	void create_random_neuron(Network& network)
	{
		std::size_t layer_identifier = random_int(1, network.layers.size() - 2);
		NeuronLayer& random_layer = network.layers[layer_identifier];

		Neuron& neuron = random_layer.neurons.emplace_back();
		neuron.randomize_parameters();

		const std::size_t predecessors = network.neuron_count(1, layer_identifier);
		const std::size_t successors = network.neuron_count(layer_identifier + 1, network.layers.size() - 1);

		// TODO: problem is that all this allows creation of duplicate synapses

		// Connect at least one input from the past layers to the new neuron
		do
		{
			const std::size_t random_neuron = random_int(0, predecessors - 1);
			Neuron& predecessor = network.neuron(network.nth_neuron(random_neuron, 0));

			ForwardSynapse& synapse = predecessor.synapses.emplace_back();
			synapse.forward_neuron_identifier = NeuronIdentifier{layer_identifier, random_layer.neurons.size() - 1};
			synapse.randomize_parameters();
		} while (random_double() < extra_synapse_connection_chance);

		// Connect this neuron to at least one neuron forward
		do
		{
			const std::size_t random_neuron = random_int(0, successors - 1);
			const NeuronIdentifier successor_identifier = network.nth_neuron(random_neuron, layer_identifier + 1);

			network.neuron(successor_identifier);

			ForwardSynapse& synapse = neuron.synapses.emplace_back();
			synapse.forward_neuron_identifier = successor_identifier;
			synapse.randomize_parameters();
		} while (random_double() < extra_synapse_connection_chance);
	}

	/*void create_random_synapse(Network& network)
	{

	}*/

	void destroy_random_synapse(Network& network)
	{
		// dis is garbage

		std::size_t synapse_count = 0;

		for (const auto& layer : network.layers)
		for (const auto& neuron : layer.neurons)
		{
			synapse_count += neuron.synapses.size();
		}

		std::size_t random_synapse = random_int(0, synapse_count - 1);
		ForwardSynapseIdentifier synapse_identifer = network.nth_synapse(random_synapse);

		Neuron& infected_neuron = network.neuron(synapse_identifer.source_neuron);
		infected_neuron.synapses.erase(infected_neuron.synapses.begin() + synapse_identifer.synapse_in_neuron);
	}

	void mutate(Network& network)
	{
		for (auto& layer : network.layers)
		for (auto& neuron : layer.neurons)
		{
			if (should_mutate())
			{
				neuron.bias = random_gauss_double(neuron.bias, bias_mutation_factor);
			}

			for (auto& synapse : neuron.synapses)
			{
				if (should_mutate())
				{
					synapse.weight = random_gauss_double(synapse.weight, weight_mutation_factor);
				}
			}
		}

		while (should_hard_mutate())
		{
			//auto type = random_int(0, 3);
			auto type = random_int(0, 1);

			switch (type)
			{
			case 0:
				create_random_neuron(network);
				break;

			case 1:
				destroy_random_synapse(network);
				break;

			/*case 2:
				create_random_synapse(network);
				break;*/
			}
		}
	}
};

#endif // NETWORK_HPP
