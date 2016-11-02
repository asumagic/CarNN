#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <cstdlib>
#include "synapse.hpp"
#include "axon.hpp"

class Axon;

// Individual synapse + synapse weight class for Neuron
struct SynapseWeighted
{
	SynapseWeighted(Synapse& syn, const double weight);

	Synapse& synapse;
	double weight;

	double get() const
	{
		return synapse.read() * weight;
	}
};

// NN neuron class
class Neuron
{
public:
	Neuron();

	sf::Vector2f screen_position(const sf::Vector2u window_size, const size_t index);
	void render(sf::RenderTarget& target, const size_t index);
	Neuron& update();
	double sigmoid();

	void add_synapse(Synapse& syn);

private:
	sf::CircleShape _shape;
	std::vector<SynapseWeighted> _synapses;
	double _bias,
		   _activation = 0.,
		   _sigmoid = 0.;
};

#endif // NEURON_HPP
