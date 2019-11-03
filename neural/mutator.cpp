#include "mutator.hpp"

#include "../randomutil.hpp"
#include "../entities/car.hpp"

bool NetworkResult::operator<(const NetworkResult& other) const
{
	return car->fitness() < other.car->fitness();
}

bool NetworkResult::operator>(const NetworkResult& other) const
{
	return car->fitness() > other.car->fitness();
}

Network Mutator::cross(const Network& a, const Network& b)
{
	assert(a.layers.size() == b.layers.size());
	assert(a.layers.front().size() == b.layers.front().size());
	assert(a.layers.back().size() == b.layers.back().size());

	Network ret{a.layers.front().neurons.size(), a.layers.back().neurons.size(), a.layers.size() - 2};
	ret.layers = a.layers;

	for (std::size_t layer_identifier = 1; layer_identifier < b.layers.size() - 1; ++layer_identifier)
	{
		auto& ret_layer = ret.layers[layer_identifier].neurons;
		const auto& a_layer = a.layers[layer_identifier].neurons;
		const auto& b_layer = b.layers[layer_identifier].neurons;

		if (a_layer.size() < b_layer.size())
		{
			ret_layer.insert(ret_layer.end(), b_layer.begin() + a_layer.size(), b_layer.end());
		}
	}

	for (std::size_t layer_identifier = 0; layer_identifier < b.layers.size(); ++layer_identifier)
	{
		auto& ret_layer = ret.layers[layer_identifier].neurons;
		const auto& a_layer = a.layers[layer_identifier].neurons;
		const auto& b_layer = b.layers[layer_identifier].neurons;

		std::size_t common_neurons = std::min(a_layer.size(), b_layer.size());

		for (std::size_t i = 0; i < common_neurons; ++i)
		{
			auto& ret_neuron = ret_layer[i];
			const auto& a_neuron = a_layer[i];
			const auto& b_neuron = b_layer[i];

			if (a_neuron.synapses.size() < b_neuron.synapses.size())
			{
				ret_neuron.synapses.insert(
					ret_neuron.synapses.end(),
					b_neuron.synapses.begin() + a_neuron.synapses.size(),
					b_neuron.synapses.end()
				);
			}

			std::size_t common_synapses = std::min(a_neuron.synapses.size(), b_neuron.synapses.size());

			for (std::size_t j = 0; j < common_synapses; ++j)
			{
				if (random_bool() < 0.5)
				{
					ret_neuron.synapses[j] = b_neuron.synapses[j];
				}
			}
		}
	}

	return ret;
}

void Mutator::darwin(std::vector<NetworkResult> results)
{
	float new_max_fitness = std::max_element(results.begin(), results.end())->car->fitness();

	if (new_max_fitness >= max_fitness + fitness_evolution_threshold)
	{
		std::sort(results.begin(), results.end(), std::greater{});
		max_fitness = new_max_fitness;
		++current_generation;
	}

	for (std::size_t i = 0; i < settings.round_survivors; ++i)
	{
		results[i].car->with_color(sf::Color{200, 50, 0, 50});
	}

	for (std::size_t i = settings.round_survivors; i < results.size(); ++i)
	{
		results[i].car->with_color(sf::Color{0, 0, 100, 70});

		// TODO: this allows self breeding, do we allow it?
		*results[i].network = cross(
					*results[random_int(0, settings.round_survivors - 1)].network,
				*results[random_int(0, settings.round_survivors - 1)].network
				);

		mutate(*results[i].network);
	}
}

bool Mutator::should_mutate() const
{
	if (settings.mutation_rate >= 1.0)
	{
		return true;
	}

	return random_bool(settings.mutation_rate);
}

bool Mutator::should_hard_mutate() const
{
	if (settings.hard_mutation_rate >= 0.9999)
	{
		throw std::runtime_error{"Hard mutation rate must be <1, otherwise the mutator would loop infinitely"};
	}

	return random_bool(settings.hard_mutation_rate);
}

void Mutator::create_random_neuron(Network& network)
{
	std::size_t layer_identifier = random_int(1, network.layers.size() - 2);
	NeuronLayer& random_layer = network.layers[layer_identifier];

	Neuron& neuron = random_layer.neurons.emplace_back();
	neuron.randomize_parameters();

	const std::size_t predecessors = network.neuron_count(1, layer_identifier);
	const std::size_t successors = network.neuron_count(layer_identifier + 1, network.layers.size() - 1);

	// TODO: problem is that all this allows creation of duplicate synapses

	// Connect at least one input from the past layers to the new neuron
	do
	{
		const std::size_t random_neuron = random_int(0, predecessors - 1);
		Neuron& predecessor = network.neuron(network.nth_neuron(random_neuron, 0));

		ForwardSynapse& synapse = predecessor.synapses.emplace_back();
		synapse.forward_neuron_identifier = NeuronIdentifier{layer_identifier, random_layer.neurons.size() - 1};
		synapse.randomize_parameters();
	} while (random_bool(settings.extra_synapse_connection_chance));

	// Connect this neuron to at least one neuron forward
	do
	{
		const std::size_t random_neuron = random_int(0, successors - 1);
		const NeuronIdentifier successor_identifier = network.nth_neuron(random_neuron, layer_identifier + 1);

		network.neuron(successor_identifier);

		ForwardSynapse& synapse = neuron.synapses.emplace_back();
		synapse.forward_neuron_identifier = successor_identifier;
		synapse.randomize_parameters();
	} while (random_bool(settings.extra_synapse_connection_chance));
}

void Mutator::create_random_synapse([[maybe_unused]] Network& network)
{

}

void Mutator::destroy_random_synapse(Network& network)
{
	// dis is garbage

	std::size_t synapse_count = 0;

	for (const auto& layer : network.layers)
		for (const auto& neuron : layer.neurons)
		{
			synapse_count += neuron.synapses.size();
		}

	std::size_t random_synapse = random_int(0, synapse_count - 1);
	ForwardSynapseIdentifier synapse_identifer = network.nth_synapse(random_synapse);

	Neuron& infected_neuron = network.neuron(synapse_identifer.source_neuron);
	infected_neuron.synapses.erase(infected_neuron.synapses.begin() + synapse_identifer.synapse_in_neuron);
}

void Mutator::mutate(Network& network)
{
	for (auto& layer : network.layers)
		for (auto& neuron : layer.neurons)
		{
			if (should_mutate())
			{
				neuron.bias = random_gauss_double(neuron.bias, settings.bias_mutation_factor);
			}

			for (auto& synapse : neuron.synapses)
			{
				if (should_mutate())
				{
					synapse.weight = random_gauss_double(synapse.weight, settings.weight_mutation_factor);
				}
			}
		}

	while (should_hard_mutate())
	{
		//auto type = random_int(0, 3);
		auto type = random_int(0, 1);

		switch (type)
		{
		case 0:
			create_random_neuron(network);
		break;

		case 1:
			destroy_random_synapse(network);
		break;

		/*case 2:
			create_random_synapse(network);
			break;*/
		}
	}
}
