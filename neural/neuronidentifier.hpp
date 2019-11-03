#pragma once

#include <cassert>
#include <cstddef>

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
