#pragma once

#include "synapse.hpp"
#include <vector>

class Network;

enum class ActivationMethod
{
	Sigmoid,
	LeakyRelu,

	Total
};

struct Neuron
{
	double           partial_activation = 0.0;
	double           value;
	double           bias              = 0.0;
	ActivationMethod activation_method = ActivationMethod::Sigmoid;

	std::vector<Synapse> synapses;

	void compute_value();
	void propagate_forward(Network& network);
};
