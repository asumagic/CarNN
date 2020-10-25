#pragma once

#include <carnn/neural/neuronid.hpp>
#include <carnn/neural/types.hpp>
#include <cereal/cereal.hpp>

struct SynapseProperties
{
	NeuralFp weight;

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(weight));
	}
};

struct Synapse
{
	NeuronId          source, target;
	SynapseProperties properties;

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(source), CEREAL_NVP(target), CEREAL_NVP(properties));
	}
};
