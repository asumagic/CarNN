#include <carnn/neural/neuron.hpp>

#include <carnn/training/mutator.hpp>
#include <carnn/util/maths.hpp>

namespace neural
{
void Neuron::compute_value()
{
	const auto v = partial_activation + bias;

	NeuralFp new_value = value;

	switch (activation_method)
	{
	case ActivationMethod::Sigmoid: new_value = 1.0 / (1.0 + std::exp(-v)); break;
	case ActivationMethod::LeakyRelu: new_value = std::max(NeuralFp(0.1) * v, v); break;
	case ActivationMethod::Sin: new_value = std::abs(v) < 0.01f ? 1.0f : std::sin(v * 2.0 * 3.14159265359) / v; break;
	default: break;
	}

	//value = util::lerp(value, new_value, 0.1f);
	value = new_value;
}
} // namespace neural
