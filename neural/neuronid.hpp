#pragma once

#include <cassert>
#include <cstddef>

struct NeuronId
{
	static constexpr std::size_t max_neurons_per_layer = 1024;

	std::size_t value;

	NeuronId() = default;

	explicit NeuronId(std::size_t value) : value{value} {}

	NeuronId(std::size_t layer, std::size_t neuron_in_layer) : value{layer * max_neurons_per_layer + neuron_in_layer}
	{
		assert(neuron_in_layer < max_neurons_per_layer);
	}

	std::size_t layer() const { return value / max_neurons_per_layer; }

	std::size_t neuron_in_layer() const { return value % max_neurons_per_layer; }

	friend bool operator<(const NeuronId& a, const NeuronId& b);
	friend bool operator==(const NeuronId& a, const NeuronId& b);
};

inline bool operator<(const NeuronId& a, const NeuronId& b) { return a.value < b.value; }
inline bool operator==(const NeuronId& a, const NeuronId& b) { return a.value == b.value; }
