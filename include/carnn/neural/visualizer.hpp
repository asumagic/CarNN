#pragma once

#include <SFML/Graphics.hpp>
#include <carnn/neural/fwd.hpp>

namespace neural
{
class Visualizer
{
	public:
	Visualizer(const Network& network);

	void display(sf::RenderTarget& target, sf::Font& font);

	private:
	sf::Vector2f neuron_offset(NeuronPosition pos);

	const Network* _network;
};
} // namespace neural
