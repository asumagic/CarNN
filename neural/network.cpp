#include "network.hpp"

#include "../entities/car.hpp"

#include <fmt/core.h>

Network::Network(std::size_t input_count, std::size_t output_count, std::size_t max_hidden) :
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

void Network::dump(std::ostream& stream) const
{
	stream << "digraph G {\n";

	for (std::size_t layer_identifier = 0; layer_identifier < layers.size(); ++layer_identifier)
	for (std::size_t neuron_identifier = 0; neuron_identifier < layers[layer_identifier].neurons.size(); ++neuron_identifier)
	{
		NeuronIdentifier current_identifier{layer_identifier, neuron_identifier};
		const Neuron& current = layers[layer_identifier].neurons[neuron_identifier];

		stream << fmt::format(
			R"({} [label="{}"];)",
			current_identifier.value,
			current.bias
		);

		for (const ForwardSynapse& synapse : current.synapses)
		{
			stream << fmt::format(
				R"({} -> {} [label="{}"])",
				current_identifier.value,
				synapse.forward_neuron_identifier.value,
				synapse.weight
			);
		}
	}

	stream << "}\n";
}

void Network::update()
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

ForwardSynapseIdentifier Network::nth_synapse(std::size_t n, std::size_t first_layer)
{
	std::size_t synapses_found = 0;

	for (std::size_t layer_identifier = first_layer;; ++layer_identifier)
		for (std::size_t neuron_identifier = 0; neuron_identifier < layers[layer_identifier].neurons.size(); ++neuron_identifier)
		{
			const Neuron& neuron = layers.at(layer_identifier).neurons.at(neuron_identifier);

			if (synapses_found + neuron.synapses.size() > n)
			{
				return {
					{layer_identifier, neuron_identifier},
					n - synapses_found
				};
			}

			synapses_found += neuron.synapses.size();
		}
}

NeuronIdentifier Network::nth_neuron(std::size_t n, std::size_t first_layer)
{
	std::size_t neurons_found = 0;

	for (std::size_t layer_identifier = first_layer;; ++layer_identifier)
	{
		const NeuronLayer& layer = layers.at(layer_identifier);

		if (neurons_found + layer.neurons.size() > n)
		{
			return {
				layer_identifier,
						n - neurons_found
			};
		}

		neurons_found += layer.neurons.size();
	}
}

std::size_t Network::neuron_count(std::size_t first_layer, std::size_t last_layer) const
{
	std::size_t sum = 0;

	for (std::size_t i = first_layer; i <= last_layer; ++i)
	{
		sum += layers[i].neurons.size();
	}

	return sum;
}
