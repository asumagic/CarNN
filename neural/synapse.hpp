#pragma once

#include "neuronid.hpp"
#include <cereal/cereal.hpp>

struct SynapseProperties
{
	double weight;

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
