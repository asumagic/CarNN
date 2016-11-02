#ifndef AXON_HPP
#define AXON_HPP

#include "neuron.hpp"

class Neuron;

struct NeuronOutputWeighted
{
	NeuronOutputWeighted(Neuron& neuron, double weight);

	Neuron& n;
	double weight;

	double get();
};

// NN output
class Axon
{
public:
	Axon();

	inline double read() { return _value; }
	inline void write(const double in) { _value = in; }

	void tie_neuron(Neuron& n);

	sf::Vector2f screen_position(const sf::Vector2u window_size, const size_t index);

	void update();
	void render(sf::RenderTarget& target, const size_t index);

private:
	sf::CircleShape _shape;
	std::vector<NeuronOutputWeighted> _neurons;
	double _value = 0.;
};

#endif // AXON_HPP
