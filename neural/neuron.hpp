#pragma once

#include "synapse.hpp"
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <string>
#include <vector>

class Network;

enum class ActivationMethod
{
	Sigmoid,
	LeakyRelu,
	Sin,

	Total
};

struct Neuron
{
	double           partial_activation = 0.0;
	double           value;
	double           bias              = 0.0;
	ActivationMethod activation_method = ActivationMethod::Sigmoid;
	std::string      label;

	std::vector<Synapse> synapses;

	void compute_value();
	void propagate_forward(Network& network);

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(partial_activation),
		   CEREAL_NVP(bias),
		   CEREAL_NVP(activation_method),
		   CEREAL_NVP(label),
		   CEREAL_NVP(synapses));
	}
};
