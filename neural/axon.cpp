#include "../randomutil.hpp"
#include "../maths.hpp"
#include "axon.hpp"

Axon::Axon() : _shape{5.} {}

void Axon::tie_neuron(Neuron& n)
{
	_neurons.emplace_back(n, random_double(0.0, 1.0));
}

sf::Vector2f Axon::screen_position(const sf::Vector2u window_size, const size_t index)
{
	return sf::Vector2f{static_cast<float>(window_size.x - ((index + 2) * 13)), 64.f};
}

void Axon::update()
{
	for (NeuronOutputWeighted& neuron : _neurons)
		_value = neuron.get();
}

void Axon::render(sf::RenderTarget& target, const size_t index)
{
	_shape.setPosition(screen_position(target.getSize(), index)); // @TODO : don't do this when not required
	if (_value >= 0. && _value <= 1.)
		_shape.setFillColor(sf::Color{0, static_cast<uint8_t>(lerp(50, 80, static_cast<float>(_value))), static_cast<uint8_t>(lerp(50, 255, static_cast<float>(_value)))});
	else
		_shape.setFillColor(sf::Color::Red);
	target.draw(_shape);
}

NeuronOutputWeighted::NeuronOutputWeighted(Neuron& neuron, double weight) : n(neuron), weight(weight) {}

double NeuronOutputWeighted::get()
{
	return n.sigmoid() * weight;
}
