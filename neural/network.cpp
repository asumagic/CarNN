#include "network.hpp"

#include "../entities/car.hpp"

#include <fmt/core.h>

bool NetworkResult::operator<(const NetworkResult& other) const
{
	return car->fitness() < other.car->fitness();
}

bool NetworkResult::operator>(const NetworkResult& other) const
{
	return car->fitness() > other.car->fitness();
}

Network Mutator::cross(const Network& a, const Network& b)
{
	assert(a.layers.size() == b.layers.size());
	assert(a.layers.front().size() == b.layers.front().size());
	assert(a.layers.back().size() == b.layers.back().size());

	Network ret{a.layers.front().neurons.size(), a.layers.back().neurons.size(), a.layers.size() - 2};
	ret.layers = a.layers;

	for (std::size_t layer_identifier = 1; layer_identifier < b.layers.size() - 1; ++layer_identifier)
	{
		auto& ret_layer = ret.layers[layer_identifier].neurons;
		const auto& a_layer = a.layers[layer_identifier].neurons;
		const auto& b_layer = b.layers[layer_identifier].neurons;

		if (a_layer.size() < b_layer.size())
		{
			ret_layer.insert(ret_layer.end(), b_layer.begin() + a_layer.size(), b_layer.end());
		}
	}

	for (std::size_t layer_identifier = 0; layer_identifier < b.layers.size(); ++layer_identifier)
	{
		auto& ret_layer = ret.layers[layer_identifier].neurons;
		const auto& a_layer = a.layers[layer_identifier].neurons;
		const auto& b_layer = b.layers[layer_identifier].neurons;

		std::size_t common_neurons = std::min(a_layer.size(), b_layer.size());

		for (std::size_t i = 0; i < common_neurons; ++i)
		{
			auto& ret_neuron = ret_layer[i];
			const auto& a_neuron = a_layer[i];
			const auto& b_neuron = b_layer[i];

			if (a_neuron.synapses.size() < b_neuron.synapses.size())
			{
				ret_neuron.synapses.insert(
					ret_neuron.synapses.end(),
					b_neuron.synapses.begin() + a_neuron.synapses.size(),
					b_neuron.synapses.end()
				);
			}

			std::size_t common_synapses = std::min(a_neuron.synapses.size(), b_neuron.synapses.size());

			for (std::size_t j = 0; j < common_synapses; ++j)
			{
				if (random_bool() < 0.5)
				{
					ret_neuron.synapses[j] = b_neuron.synapses[j];
				}
			}
		}
	}

	return ret;
}

void Mutator::darwin(std::vector<NetworkResult> results)
{
	float new_max_fitness = std::max_element(results.begin(), results.end())->car->fitness();

	if (new_max_fitness >= max_fitness + fitness_evolution_threshold)
	{
		std::sort(results.begin(), results.end(), std::greater{});
		max_fitness = new_max_fitness;
		++current_generation;
	}

	for (std::size_t i = 0; i < noah; ++i)
	{
		results[i].car->with_color(sf::Color{200, 50, 0, 50});
	}

	for (std::size_t i = noah; i < results.size(); ++i)
	{
		results[i].car->with_color(sf::Color{0, 0, 100, 70});

		// TODO: this allows self breeding, do we allow it?
		*results[i].network = cross(
					*results[random_int(0, noah - 1)].network,
				*results[random_int(0, noah - 1)].network
				);

		mutate(*results[i].network);
	}
}

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
