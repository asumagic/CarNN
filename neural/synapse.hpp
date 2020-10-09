#pragma once

#include "neuronid.hpp"
#include <cereal/cereal.hpp>

struct Synapse
{
	NeuronId target;
	double   weight;

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(target, CEREAL_NVP(weight));
	}
};
