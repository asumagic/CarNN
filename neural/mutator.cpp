#include "mutator.hpp"

#include "../entities/car.hpp"
#include "../neural/network.hpp"
#include "../randomutil.hpp"

bool NetworkResult::operator<(const NetworkResult& other) const { return car->fitness() < other.car->fitness(); }

bool NetworkResult::operator>(const NetworkResult& other) const { return car->fitness() > other.car->fitness(); }

Network Mutator::cross(const Network& a, const Network& b)
{
	assert(a.layers.size() == b.layers.size());
	assert(a.layers.front().size() == b.layers.front().size());
	assert(a.layers.back().size() == b.layers.back().size());

	Network ret{a.layers.front().neurons.size(), a.layers.back().neurons.size()};
	ret.layers = a.layers;

	for (std::size_t layer_identifier = 1; layer_identifier < b.layers.size() - 1; ++layer_identifier)
	{
		auto&       ret_layer = ret.layers[layer_identifier].neurons;
		const auto& a_layer   = a.layers[layer_identifier].neurons;
		const auto& b_layer   = b.layers[layer_identifier].neurons;

		if (a_layer.size() < b_layer.size())
		{
			ret_layer.insert(ret_layer.end(), b_layer.begin() + a_layer.size(), b_layer.end());
		}
	}

	for (std::size_t layer_identifier = 0; layer_identifier < b.layers.size(); ++layer_identifier)
	{
		auto&       ret_layer = ret.layers[layer_identifier].neurons;
		const auto& a_layer   = a.layers[layer_identifier].neurons;
		const auto& b_layer   = b.layers[layer_identifier].neurons;

		std::size_t common_neurons = std::min(a_layer.size(), b_layer.size());

		for (std::size_t i = 0; i < common_neurons; ++i)
		{
			auto&       ret_neuron = ret_layer[i];
			const auto& a_neuron   = a_layer[i];
			const auto& b_neuron   = b_layer[i];

			if (a_neuron.synapses.size() < b_neuron.synapses.size())
			{
				ret_neuron.synapses.insert(
					ret_neuron.synapses.end(),
					b_neuron.synapses.begin() + a_neuron.synapses.size(),
					b_neuron.synapses.end());
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
	std::sort(results.begin(), results.end(), std::greater{});
	float new_max_fitness = results[0].car->fitness();

	if (new_max_fitness >= max_fitness + fitness_evolution_threshold)
	{
		max_fitness = new_max_fitness;
		++current_generation;

		for (std::size_t i = 0; i < settings.round_survivors; ++i)
		{
			results[i].car->with_color(sf::Color{200, 50, 0, 50});
			results[i].car->top_of_generation = true;
		}

		for (std::size_t i = settings.round_survivors; i < results.size(); ++i)
		{
			results[i].car->with_color(sf::Color{0, 0, 100, 70});
			results[i].car->top_of_generation = false;
		}
	}

	for (auto& result : results)
	{
		if (!result.car->top_of_generation)
		{
			// TODO: this allows self breeding, do we allow it?
			*result.network = cross(
				*results[random_int(0, settings.round_survivors - 1)].network,
				*results[random_int(0, settings.round_survivors - 1)].network);

			mutate(*result.network);
		}
	}
}

void Mutator::create_random_neuron(Network& network)
{
	Layer& hidden_layer = network.layers[1];

	Neuron& neuron = hidden_layer.neurons.emplace_back();
	randomize(neuron);

	do
	{
		const std::size_t random_neuron = random_int(0, network.inputs().neurons.size());
		Neuron&           predecessor   = network.neuron(network.nth_neuron(random_neuron, 0));

		Synapse& synapse = predecessor.synapses.emplace_back();
		synapse.target   = NeuronId{1, hidden_layer.neurons.size() - 1};
		randomize(synapse);
	} while (random_bool(settings.extra_synapse_connection_chance));

	// Connect this neuron to at least one neuron on the same layer or forward
	do
	{
		// i dont caaaaaaare
		const std::size_t random_neuron
			= random_int(0, network.layers[1].neurons.size() + network.layers[2].neurons.size() - 1);

		const NeuronId successor_identifier = network.nth_neuron(random_neuron, 1);

		Synapse& synapse = neuron.synapses.emplace_back();
		synapse.target   = successor_identifier;
		randomize(synapse);
	} while (random_bool(settings.extra_synapse_connection_chance));
}

void Mutator::create_random_synapse([[maybe_unused]] Network& network) {}

void Mutator::destroy_random_synapse(Network& network)
{
	// dis is garbage

	std::size_t synapse_count = 0;

	for (const auto& layer : network.layers)
		for (const auto& neuron : layer.neurons)
		{
			synapse_count += neuron.synapses.size();
		}

	std::size_t random_synapse    = random_int(0, synapse_count - 1);
	SynapseId   synapse_identifer = network.nth_synapse(random_synapse);

	Neuron& infected_neuron = network.neuron(synapse_identifer.source_neuron);
	infected_neuron.synapses.erase(infected_neuron.synapses.begin() + synapse_identifer.synapse_in_neuron);
}

void Mutator::mutate(Network& network)
{
	while (random_bool(settings.bias_mutation_chance))
	{
		auto& neuron = network.neuron(network.random_neuron(0, 2));
		neuron.bias  = random_gauss_double(neuron.bias, settings.bias_mutation_factor);
	}

	while (random_bool(settings.bias_hard_mutation_chance))
	{
		auto& neuron = network.neuron(network.random_neuron(0, 2));
		neuron.bias  = random_gauss_double(neuron.bias, settings.bias_hard_mutation_factor);
	}

	while (random_bool(settings.weight_mutation_chance))
	{
		auto& synapse  = network.synapse(network.random_synapse(0, 2));
		synapse.weight = random_gauss_double(synapse.weight, settings.weight_mutation_factor);
	}

	while (random_bool(settings.weight_hard_mutation_chance))
	{
		auto& synapse  = network.synapse(network.random_synapse(0, 2));
		synapse.weight = random_gauss_double(synapse.weight, settings.weight_hard_mutation_factor);
	}

	while (random_bool(settings.neuron_creation_chance))
	{
		create_random_neuron(network);
	}

	while (random_bool(settings.synapse_destruction_chance))
	{
		destroy_random_synapse(network);
	}

	if (random_bool(settings.aggressive_gc_chance))
	{
		gc(network, true);
	}
	else if (random_bool(settings.conservative_gc_chance))
	{
		gc(network, false);
	}
}

void Mutator::randomize(Network& network)
{
	for (auto& layer : network.layers)
	{
		for (auto& neuron : layer.neurons)
		{
			randomize(neuron);

			for (auto& synapse : neuron.synapses)
			{
				randomize(synapse);
			}
		}
	}
}

void Mutator::randomize(Neuron& neuron)
{
	neuron.bias              = random_gauss_double(0.0, settings.bias_initial_std_dev);
	neuron.activation_method = ActivationMethod(random_int(0, int(ActivationMethod::Total)));
}

void Mutator::randomize(Synapse& synapse)
{
	synapse.weight = random_gauss_double(0.0, settings.weight_initial_std_dev);
}

void Mutator::gc(Network& network, bool aggressive)
{
	for (auto& layer : network.layers)
	{
		for (auto& neuron : layer.neurons)
		{
			// we can't erase neurons because neuron identifiers must be stable.
			// but synapses are fine, as we don't refer to them permanently.

			std::sort(neuron.synapses.begin(), neuron.synapses.end(), [](const Synapse& a, const Synapse& b) {
				return a.target < b.target;
			});

			Synapse* last_identical_synapse = nullptr;

			for (auto synapse_it = neuron.synapses.begin(); synapse_it != neuron.synapses.end();)
			{
				if (synapse_it->weight == 0 || (aggressive && std::abs(synapse_it->weight) < 0.05))
				{
					synapse_it = neuron.synapses.erase(synapse_it);
					continue;
				}

				if (last_identical_synapse != nullptr && synapse_it->target == last_identical_synapse->target)
				{
					last_identical_synapse->weight += synapse_it->weight;
					synapse_it = neuron.synapses.erase(synapse_it);
					continue;
				}

				last_identical_synapse = &*synapse_it;
				++synapse_it;
			}
		}
	}
}
