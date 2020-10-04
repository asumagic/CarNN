#pragma once

#include "neuronid.hpp"

struct Synapse
{
	NeuronId target;
	double   weight;
};
