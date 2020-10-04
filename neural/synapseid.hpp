#pragma once

#include "neuronid.hpp"

struct SynapseId
{
	NeuronId    source_neuron;
	std::size_t synapse_in_neuron;
};
