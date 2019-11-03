#include "neuron.hpp"

#include "../maths.hpp"
#include "network.hpp"

void Neuron::compute_value()
{
	value = sigmoid(partial_activation + bias);
}

void Neuron::propagate_forward(Network& network)
{
	for (ForwardSynapse& synapse : synapses)
	{
		Neuron& forward_neuron = network.neuron(synapse.forward_neuron_identifier);
		forward_neuron.partial_activation += value * synapse.weight;
	}
}

void Neuron::randomize_parameters()
{
	// TODO: do we really want a uniform distribution?
	bias = random_double(-3.0, 3.0);
}
