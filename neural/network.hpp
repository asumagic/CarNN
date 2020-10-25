#pragma once

#include "neuron.hpp"
#include "synapse.hpp"
#include "synapseid.hpp"
#include <cereal/types/vector.hpp>
#include <gsl/span>
#include <vector>

struct Neuron;
class Network;

struct NeuronPosition
{
	std::size_t layer, neuron_in_layer;
};

class Network
{
	public:
	Network() = default;
	Network(std::size_t input_count, std::size_t output_count);

	Network(const Network&) = default;
	Network& operator=(const Network&) = default;

	Network(Network&&) = default;
	Network& operator=(Network&&) = default;

	gsl::span<Neuron>                inputs();
	gsl::span<Neuron>                hidden_layer();
	gsl::span<Neuron>                outputs();
	std::array<gsl::span<Neuron>, 3> layers();

	gsl::span<const Neuron>                inputs() const;
	gsl::span<const Neuron>                hidden_layer() const;
	gsl::span<const Neuron>                outputs() const;
	std::array<gsl::span<const Neuron>, 3> layers() const;

	NeuronPosition neuron_position(NeuronId id) const;

	Synapse&               create_synapse(NeuronId from, NeuronId to);
	Synapse&               get_or_create_synapse(NeuronId from, NeuronId to);
	[[nodiscard]] Synapse* get_synapse(NeuronId from, NeuronId to);
	[[nodiscard]] NeuronId get_neuron_id(std::uint32_t evolution_id) const;

	[[nodiscard]] NeuronId  random_neuron();
	[[nodiscard]] SynapseId random_synapse();

	std::vector<Neuron>  neurons;
	std::vector<Synapse> synapses;

	void update();

	void reset_values();

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(neurons), CEREAL_NVP(synapses));
	}

	private:
	std::size_t _input_count, _output_count;
};
