#include "neuron.hpp"

#include "../maths.hpp"
#include "mutator.hpp"
#include "network.hpp"

void Neuron::compute_value()
{
	const auto v = partial_activation + bias;

	switch (activation_method)
	{
	case ActivationMethod::Sigmoid: value = 1.0 / (1.0 + std::exp(-v));
	case ActivationMethod::LeakyRelu: value = std::max(0.1 * v, v);
	default: break;
	}
}

void Neuron::propagate_forward(Network& network)
{
	for (ForwardSynapse& synapse : synapses)
	{
		Neuron& forward_neuron = network.neuron(synapse.forward_neuron_identifier);
		forward_neuron.partial_activation += value * synapse.weight;
	}
}
