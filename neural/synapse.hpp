#ifndef SYNAPSE_HPP
#define SYNAPSE_HPP

#include <SFML/Graphics.hpp>

// NN input
class Synapse
{
public:
	Synapse(double& ref);

	inline double read() { return _input_ref; }
	inline void write(const double in) { _input_ref = in; }

	void render(sf::RenderTarget& target, const size_t index);

private:
	sf::CircleShape _shape;
	double& _input_ref;
};

#endif // SYNAPSE_HPP
