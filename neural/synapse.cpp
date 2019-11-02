#include "synapse.hpp"
#include "../maths.hpp"

Synapse::Synapse(double& ref) : _shape{2.}, _input_ref(ref) {}

void Synapse::render(sf::RenderTarget& target, const size_t index)
{
	_shape.setPosition(0.5f * sf::Vector2f{static_cast<float>(target.getSize().x - ((index + 2) * 6)), 8.f}); // @TODO : don't do this when not required

	if (_input_ref >= 0. && _input_ref <= 1.)
		_shape.setFillColor(sf::Color{0, static_cast<uint8_t>(lerp(40, 150, static_cast<float>(_input_ref))), static_cast<uint8_t>(lerp(50, 255, static_cast<float>(_input_ref)))});
	else
		_shape.setFillColor(sf::Color::Red);


	target.draw(_shape);
}
