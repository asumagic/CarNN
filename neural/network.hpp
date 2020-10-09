#pragma once

#include "../maths.hpp"
#include "neuron.hpp"
#include "synapseid.hpp"
#include <array>
#include <cassert>
#include <cereal/types/array.hpp>
#include <cereal/types/vector.hpp>
#include <cmath>
#include <iosfwd>
#include <vector>

class Car;

struct Neuron;
class Network;

struct Layer
{
	std::vector<Neuron> neurons;

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(neurons));
	}
};

class Network
{
	public:
	Network() = default;
	Network(std::size_t input_count, std::size_t output_count);

	void dump(std::ostream& stream) const;

	Layer& inputs();
	Layer& outputs();

	Neuron&  neuron(NeuronId identifier);
	Synapse& synapse(SynapseId identifier);

	NeuronId  nth_neuron(std::size_t n, std::size_t first_layer = 0);
	SynapseId nth_synapse(std::size_t n, std::size_t first_layer = 0);

	std::size_t neuron_count(std::size_t first_layer, std::size_t last_layer);
	NeuronId    random_neuron(std::size_t first_layer, std::size_t last_layer);

	std::size_t synapse_count(std::size_t first_layer, std::size_t last_layer);
	SynapseId   random_synapse(std::size_t first_layer, std::size_t last_layer);

	std::array<Layer, 3> layers{};

	void update();

	void reset_values();

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(layers));
	}

	private:
	void sanitize(NeuronId identifier);
	void sanitize(SynapseId identifier);
};

inline Layer& Network::inputs() { return layers.front(); }

inline Layer& Network::outputs() { return layers.back(); }

inline Neuron& Network::neuron(NeuronId identifier)
{
	sanitize(identifier);
	return layers[identifier.layer()].neurons[identifier.neuron_in_layer()];
}

inline Synapse& Network::synapse(SynapseId identifier)
{
	sanitize(identifier);
	return neuron(identifier.source_neuron).synapses[identifier.synapse_in_neuron];
}

inline void Network::sanitize([[maybe_unused]] NeuronId identifier)
{
	assert(identifier.layer() < layers.size());
	assert(identifier.neuron_in_layer() < layers[identifier.layer()].neurons.size());
}

inline void Network::sanitize([[maybe_unused]] SynapseId identifier)
{
	assert(identifier.synapse_in_neuron < neuron(identifier.source_neuron).synapses.size());
}
