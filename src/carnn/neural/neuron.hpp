#pragma once

#include <carnn/neural/activationmethod.hpp>
#include <carnn/neural/types.hpp>
#include <cereal/cereal.hpp>

class Network;

struct Neuron
{
	NeuralFp         bias              = 0.0;
	std::uint32_t    evolution_id      = 0;
	ActivationMethod activation_method = ActivationMethod::Sigmoid;

	NeuralFp partial_activation = 0.0;
	NeuralFp value              = 0.0;

	Neuron() {}
	Neuron(std::uint32_t evolution_id) : evolution_id(evolution_id) {}

	Neuron(const Neuron&) = default;
	Neuron& operator=(const Neuron&) = default;

	Neuron(Neuron&&) = default;
	Neuron& operator=(Neuron&&) = default;

	void compute_value();

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(evolution_id), CEREAL_NVP(bias), CEREAL_NVP(activation_method));
	}

	friend bool operator==(const Neuron& neuron, std::uint32_t evolution_id)
	{
		return neuron.evolution_id == evolution_id;
	}
};
