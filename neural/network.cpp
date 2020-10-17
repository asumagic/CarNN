#include "network.hpp"

#include "../entities/car.hpp"
#include "../randomutil.hpp"
#include <fmt/core.h>

Network::Network(std::size_t input_count, std::size_t output_count) :
	neurons(input_count + output_count), _input_count(input_count), _output_count(output_count)
{
	// assign initial evolution ids for inputs and outputs, which necessarily matches the evolution ids of the other
	// networks
	for (std::size_t i = 0; i < neurons.size(); ++i)
	{
		neurons[i].evolution_id = std::uint32_t(i + 1);
	}

	for (std::size_t input_index = 0; input_index < inputs().size(); ++input_index)
	{
		for (std::size_t output_index = 0; output_index < output_count; ++output_index)
		{
			create_synapse(input_index, output_index + _input_count);
		}
	}
}

// TODO: move to its own utility file
template<class Dst, class Src>
gsl::span<Dst> span_const_cast(gsl::span<Src> src)
{
	return {const_cast<Dst*>(src.data()), src.size()};
}

gsl::span<Neuron> Network::inputs() { return span_const_cast<Neuron>(std::as_const(*this).inputs()); }
gsl::span<Neuron> Network::hidden_layer() { return span_const_cast<Neuron>(std::as_const(*this).hidden_layer()); }
gsl::span<Neuron> Network::outputs() { return span_const_cast<Neuron>(std::as_const(*this).outputs()); }
std::array<gsl::span<Neuron>, 3> Network::layers() { return {inputs(), hidden_layer(), outputs()}; }

gsl::span<const Neuron> Network::inputs() const { return {neurons.data(), _input_count}; }

gsl::span<const Neuron> Network::hidden_layer() const
{
	return {neurons.data() + _input_count + _output_count, neurons.size() - _input_count - _output_count};
}

gsl::span<const Neuron> Network::outputs() const { return {neurons.data() + _input_count, _output_count}; }

std::array<gsl::span<const Neuron>, 3> Network::layers() const { return {inputs(), hidden_layer(), outputs()}; }

NeuronPosition Network::neuron_position(NeuronId id) const
{
	if (id < _input_count)
	{
		return {0, id};
	}

	id -= _input_count;
	if (id < _output_count)
	{
		return {2, id};
	}

	id -= _output_count;
	return {1, id};
}

Synapse& Network::create_synapse(NeuronId from, NeuronId to)
{
	Synapse& synapse = synapses.emplace_back();
	synapse.source   = from;
	synapse.target   = to;
	return synapse;
}

Synapse& Network::get_or_create_synapse(NeuronId from, NeuronId to)
{
	if (Synapse* synapse = get_synapse(from, to); synapse != nullptr)
	{
		return *synapse;
	}

	return create_synapse(from, to);
}

Synapse* Network::get_synapse(NeuronId from, NeuronId to)
{
	const auto it = std::find_if(synapses.begin(), synapses.end(), [&](const Synapse& synapse) {
		return synapse.source == from && synapse.target == to;
	});

	if (it == synapses.end())
	{
		return nullptr;
	}

	return &*it;
}

NeuronId Network::get_neuron_id(uint32_t evolution_id) const
{
	const auto it = std::find(neurons.begin(), neurons.end(), evolution_id);
	return std::distance(neurons.begin(), it);
}

void Network::update()
{
	for (Neuron& neuron : neurons)
	{
		neuron.compute_value();
		neuron.partial_activation = 0.0;
	}

	for (Synapse& synapse : synapses)
	{
		neurons[synapse.target].partial_activation += neurons[synapse.source].value * synapse.properties.weight;
	}

	for (Neuron& neuron : outputs())
	{
		neuron.value = clamp(neuron.value, 0.0, 1.0);
	}
}

void Network::reset_values()
{
	for (Neuron& neuron : neurons)
	{
		neuron.value              = 0.0;
		neuron.partial_activation = 0.0;
	}
}

NeuronId Network::random_neuron() { return NeuronId(random_int(0, neurons.size() - 1)); }

SynapseId Network::random_synapse() { return random_int(0, synapses.size() - 1); }
