#include <carnn/neural/activationmethod.hpp>

std::string_view name(ActivationMethod method)
{
	switch (method)
	{
	case ActivationMethod::Sigmoid: return "sigmoid";
	case ActivationMethod::LeakyRelu: return "lRELU";
	case ActivationMethod::Sin: return "sin";
	case ActivationMethod::Total:
	default: return "<invalid>";
	}
}
