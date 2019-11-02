#include "neuron.hpp"
#include "../randomutil.hpp"
#include "../maths.hpp"
#include <numeric>
#include <cmath>

Neuron::Neuron() : _shape{2.}, _bias(random_double(-10.0, 10.0)) {}

sf::Vector2f Neuron::screen_position(const sf::Vector2u window_size, const size_t index)
{
	float stair = static_cast<int>(index / 50);
	return sf::Vector2f{static_cast<float>(window_size.x - ((index - (stair * 50)) * 5)) - 8.f, 16.f + (stair * 5.f)};
}

void Neuron::render(sf::RenderTarget& target, const size_t index)
{
	_shape.setPosition(0.5f * screen_position(target.getSize(), index)); // @TODO : don't do this when not required
	if (_sigmoid >= 0. && _sigmoid <= 1.)
		_shape.setFillColor(sf::Color{0, static_cast<uint8_t>(lerp(40, 150, static_cast<float>(_sigmoid))), static_cast<uint8_t>(lerp(50, 255, static_cast<float>(_sigmoid)))});
	else
		_shape.setFillColor(sf::Color::Red);
	target.draw(_shape);
}

Neuron& Neuron::update()
{
	_activation = -_bias + std::accumulate(begin(_synapses), end(_synapses), 0., [](const double& c, const SynapseWeighted& syn) { return c + syn.get(); });
	_sigmoid = 1. / (1. + exp(-_activation));
	return *this;
}

double Neuron::sigmoid()
{
	return _sigmoid;
}

void Neuron::add_synapse(Synapse &syn)
{
	_synapses.emplace_back(syn, random_double(-1.0, 1.0));
}

SynapseWeighted::SynapseWeighted(Synapse &syn, const double weight) : synapse(syn), weight(weight) {}
