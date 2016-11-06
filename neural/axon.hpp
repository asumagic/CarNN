#ifndef AXON_HPP
#define AXON_HPP

#include "neuron.hpp"

class Neuron;

const size_t axon_history_size = 512;
const float graph_size = 196.f;

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
	Axon(const size_t index);

	inline double read() { return _value; }
	inline void write(const double in) { _value = in; }

	void tie_neuron(Neuron& n);

	void set_label(const std::string label);
	void set_font(sf::Font* fnt);
	sf::Vector2f screen_position(const sf::Vector2u window_size, const size_t index);

	void update(const size_t index);
	void render(sf::RenderTarget& target, const size_t index);

private:
	float graph_point_y(const float value, const size_t index);

	sf::CircleShape _shape;
	std::vector<NeuronOutputWeighted> _neurons;
	sf::VertexArray _weight_history;

	std::string _label = "Axon";
	sf::Font* _font;

	size_t _update_history_interval = 3;
	size_t _current_history_interval = _update_history_interval - 1;

	double _value = 0.;
};

#endif // AXON_HPP
