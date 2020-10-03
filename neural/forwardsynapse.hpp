#pragma once

#include "neuronidentifier.hpp"

struct ForwardSynapse
{
	NeuronIdentifier forward_neuron_identifier;
	double           weight;
};
