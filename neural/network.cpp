#include "network.hpp"

Network::Network(size_t neurons, size_t outputs, size_t synapses) : _neurons(neurons), _axons(outputs)
{
	_synapses.reserve(synapses);

	for (Axon& axon : _axons)
	for (Neuron& neuron : _neurons)
	{
		axon.tie_neuron(neuron);
	}
}

Network& Network::update()
{
	for (Neuron& neuron : _neurons)
		neuron.update();

	for (Axon& axon : _axons)
		axon.update();

	return *this;
}

std::vector<double> Network::results()
{
	std::vector<double> results;

	results.reserve(_axons.size());

	for (Axon& axon : _axons)
		results.push_back(axon.read());

	return results;
}

Synapse& Network::add_synapse(double& input)
{
	_synapses.emplace_back(input);
	Synapse& syn = _synapses.back();

	for (Neuron& neuron : _neurons)
		neuron.add_synapse(syn);

	return syn;
}

void Network::render(sf::RenderTarget& target)
{
	for (size_t i = 0; i < _synapses.size(); ++i)
		_synapses[i].render(target, i);

	for (size_t i = 0; i < _neurons.size(); ++i)
		_neurons[i].render(target, i);

	for (size_t i = 0; i < _axons.size(); ++i)
		_axons[i].render(target, i);
}
