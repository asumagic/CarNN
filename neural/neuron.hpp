#pragma once

#include <vector>
#include "forwardsynapse.hpp"

class Network;

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
