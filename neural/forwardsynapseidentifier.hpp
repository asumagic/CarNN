#pragma once

#include "neuronidentifier.hpp"

struct ForwardSynapseIdentifier
{
	NeuronIdentifier source_neuron;
	std::size_t      synapse_in_neuron;
};
