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

void Network::merge_with(const Network& other)
{
	// create missing neurons first (which are necessarily in the hidden layer).
	for (const Neuron& foreign_neuron : other.hidden_layer())
	{
		const NeuronId potential_match_id = get_neuron_id(foreign_neuron.evolution_id);

		if (potential_match_id == neurons.size()) // TODO: less garbage interface for this
		{
			Neuron clone = foreign_neuron;
			clone.synapses.clear();
			neurons.push_back(clone);
		}
	}

	// clone all b synapses
	for (const Neuron& foreign_neuron : other.neurons)
	{
		for (const SynapseId& foreign_synapse_id : foreign_neuron.synapses)
		{
			const NeuronId source_neuron_id = get_neuron_id(foreign_neuron.evolution_id);
			const NeuronId target_neuron_id
				= get_neuron_id(other.neurons[other.synapses[foreign_synapse_id].target].evolution_id);

			const SynapseId cloned_synapse_id = create_synapse(source_neuron_id, target_neuron_id);
			Synapse&        cloned_synapse    = synapses[cloned_synapse_id];

			cloned_synapse.weight = other.synapses[foreign_synapse_id].weight;
		}
	}
}

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

SynapseId Network::create_synapse(NeuronId from, NeuronId to)
{
	Synapse&        synapse = synapses.emplace_back();
	const SynapseId id      = synapses.size() - 1;

	neurons[from].synapses.push_back(id);
	synapse.target = to;

	return id;
}

bool Network::is_valid_synapse(SynapseId id) const
{
	return id < synapses.size() && synapses[id].target < neurons.size();
}

NeuronId Network::get_neuron_id(uint32_t evolution_id) const
{
	const auto it = std::find(neurons.begin(), neurons.end(), evolution_id);
	return std::distance(neurons.begin(), it);
}

bool Network::is_valid_neuron(NeuronId id) const
{
	if (id >= neurons.size())
	{
		return false;
	}

	const Neuron& neuron = neurons[id];
	return std::all_of(
		neuron.synapses.begin(), neuron.synapses.end(), [this](SynapseId id) { return is_valid_synapse(id); });
}

std::array<gsl::span<const Neuron>, 3> Network::layers() const
{
	return {
		inputs(),
		hidden_layer(),
		outputs(),
	};
}

void Network::update()
{
	for (Neuron& neuron : neurons)
	{
		neuron.compute_value();
		neuron.partial_activation = 0.0;
	}

	for (Neuron& neuron : neurons)
	{
		neuron.propagate_forward(*this);
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

std::size_t Network::neuron_count(std::size_t first_layer, std::size_t last_layer)
{
	std::size_t sum = 0;

	for (std::size_t i = first_layer; i <= last_layer; ++i)
	{
		sum += layers()[i].size();
	}

	return sum;
}

NeuronId Network::random_neuron(std::size_t first_layer, std::size_t last_layer)
{
	return NeuronId(random_int(0, neuron_count(first_layer, last_layer) - 1));
}

SynapseId Network::random_synapse() { return random_int(0, synapses.size() - 1); }

bool Network::is_valid() const
{
	for (std::size_t i = 0; i < neurons.size(); ++i)
	{
		if (!is_valid_neuron(NeuronId(i)))
		{
			return false;
		}
	}

	return true;
}
