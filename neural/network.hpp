#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <cstdlib>
#include "neuron.hpp"
#include "synapse.hpp"
#include "axon.hpp"

class Network
{
public:
	Network(size_t neurons, size_t outputs, size_t synapses);

	Network& update();
	std::vector<double> results();

	Synapse& add_synapse(double& input); // UB if add_synapse is called more times than the total synapses count (c.f. 3rd argument)
	std::vector<Synapse>& get_synapses() { return _synapses; }

	void render(sf::RenderTarget& target);

private:
	std::vector<Synapse> _synapses;
	std::vector<Neuron> _neurons;
	std::vector<Axon> _axons;
};

#endif // NETWORK_HPP
