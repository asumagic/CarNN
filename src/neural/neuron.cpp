#include "neuron.hpp"

#include "../maths.hpp"
#include "mutator.hpp"
#include "network.hpp"

void Neuron::compute_value()
{
	const auto v = partial_activation + bias;

	switch (activation_method)
	{
	case ActivationMethod::Sigmoid: value = 1.0 / (1.0 + std::exp(-v)); break;
	case ActivationMethod::LeakyRelu: value = std::max(NeuralFp(0.1) * v, v); break;
	case ActivationMethod::Sin: value = std::abs(v) < 0.01f ? 1.0f : std::sin(v * 2.0 * 3.14159265359) / v; break;
	default: break;
	}
}
