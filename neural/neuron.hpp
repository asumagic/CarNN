#pragma once

#include "activationmethod.hpp"
#include "synapseid.hpp"
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <string>
#include <vector>

class Network;

struct Neuron
{
	std::uint32_t    evolution_id       = 0;
	double           partial_activation = 0.0;
	double           value;
	double           bias              = 0.0;
	ActivationMethod activation_method = ActivationMethod::Sigmoid;

	std::vector<SynapseId> synapses;

	Neuron() {}
	Neuron(std::uint32_t evolution_id) : evolution_id(evolution_id) {}

	Neuron(const Neuron&) = default;
	Neuron& operator=(const Neuron&) = default;

	Neuron(Neuron&&) = default;
	Neuron& operator=(Neuron&&) = default;

	void compute_value();
	void propagate_forward(Network& network);

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(evolution_id),
		   CEREAL_NVP(partial_activation),
		   CEREAL_NVP(bias),
		   CEREAL_NVP(activation_method),
		   CEREAL_NVP(synapses));
	}

	friend bool operator==(const Neuron& neuron, std::uint32_t evolution_id);
};

inline bool operator==(const Neuron& neuron, std::uint32_t evolution_id) { return neuron.evolution_id == evolution_id; }
