#pragma once

#include "../maths.hpp"
#include "neuron.hpp"
#include "synapse.hpp"
#include "synapseid.hpp"
#include <array>
#include <cassert>
#include <cereal/types/array.hpp>
#include <cereal/types/vector.hpp>
#include <cmath>
#include <gsl/span>
#include <iosfwd>
#include <vector>

class Car;

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

	void merge_with(const Network& other);

	NeuronPosition neuron_position(NeuronId id) const;

	SynapseId          create_synapse(NeuronId from, NeuronId to);
	[[nodiscard]] bool is_valid_synapse(SynapseId id) const;

	NeuronId           get_neuron_id(std::uint32_t evolution_id) const;
	[[nodiscard]] bool is_valid_neuron(NeuronId id) const;

	std::size_t neuron_count(std::size_t first_layer, std::size_t last_layer);
	NeuronId    random_neuron(std::size_t first_layer, std::size_t last_layer);

	SynapseId random_synapse();

	[[nodiscard]] bool is_valid() const;

	std::vector<Neuron>  neurons;
	std::vector<Synapse> synapses;

	void update();

	void reset_values();

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(neurons));
	}

	private:
	std::size_t _input_count, _output_count;
};
