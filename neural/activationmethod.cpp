#include "activationmethod.hpp"

std::string_view name(ActivationMethod method)
{
	switch (method)
	{
	case ActivationMethod::Sigmoid: return "sigmoid";
	case ActivationMethod::LeakyRelu: return "leaky RELU";
	case ActivationMethod::Sin: return "sinusoidal";
	case ActivationMethod::Total:
	default: return "<invalid>";
	}
}
