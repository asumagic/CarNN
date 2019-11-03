#pragma once

struct ForwardSynapseIdentifier
{
	NeuronIdentifier source_neuron;
	std::size_t synapse_in_neuron;
};
