#include "mutator.hpp"

#include "../entities/car.hpp"
#include "../neural/network.hpp"
#include "../randomutil.hpp"
#include "../simulationunit.hpp"
#include "individual.hpp"
#include <cereal/archives/json.hpp>
#include <fstream>
#include <spdlog/spdlog.h>

Network Mutator::cross(Network a, const Network& b)
{
	// TODO: this can probably be merged to a single loop

	assert(a.is_valid());
	assert(b.is_valid());

	a.merge_with(b);

	return a;
}

void Mutator::darwin(Simulation& sim, std::vector<Individual>& individuals)
{
	std::sort(individuals.begin(), individuals.end(), [&](const Individual& a, const Individual& b) {
		return sim.cars[a.car_id]->fitness() > sim.cars[b.car_id]->fitness();
	});

	float new_max_fitness = sim.cars.at(individuals[0].car_id)->fitness();

	if (new_max_fitness >= max_fitness + fitness_evolution_threshold)
	{
		spdlog::info(
			"entering generation {}: fitness {:.1f} exceeds old max {:.1f}",
			current_generation + 1,
			new_max_fitness,
			max_fitness);

		max_fitness = new_max_fitness;
		++current_generation;

		for (std::size_t i = 0; i < settings.round_survivors; ++i)
		{
			individuals[i].survivor_from_last = true;
		}

		for (std::size_t i = settings.round_survivors; i < individuals.size(); ++i)
		{
			individuals[i].survivor_from_last = false;
		}
	}
	else
	{
		spdlog::info(
			"did not enter new generation: fitness {:.1f} does not exceed old max {:.1f}",
			new_max_fitness,
			max_fitness);
	}

	for (auto& individual : individuals)
	{
		if (!individual.survivor_from_last)
		{
			// TODO: this allows self breeding, do we allow it?
			individual.network = cross(
				individuals[random_int(0, settings.round_survivors - 1)].network,
				individuals[random_int(0, settings.round_survivors - 1)].network);

			mutate(individual.network);
		}
	}
}

void Mutator::create_random_neuron(Network& network)
{
	Neuron&  neuron    = network.neurons.emplace_back(get_unique_evolution_id());
	NeuronId neuron_id = network.neurons.size() - 1;
	randomize(neuron);

	do
	{
		SynapseId synapse_id = network.create_synapse(network.random_neuron(0, 1), neuron_id);
		Synapse&  synapse    = network.synapses[synapse_id];
		randomize(synapse);
	} while (random_bool(settings.extra_synapse_connection_chance));

	// Connect this neuron to at least one neuron on the same layer or forward
	do
	{
		SynapseId synapse_id = network.create_synapse(neuron_id, network.random_neuron(1, 2));
		Synapse&  synapse    = network.synapses[synapse_id];
		randomize(synapse);
	} while (random_bool(settings.extra_synapse_connection_chance));
}

void Mutator::create_random_synapse([[maybe_unused]] Network& network) {}

void Mutator::mutate(Network& network)
{
	while (random_bool(settings.bias_mutation_chance))
	{
		auto& neuron = network.neurons[network.random_neuron(0, 2)];
		neuron.bias  = random_gauss_double(neuron.bias, settings.bias_mutation_factor);
	}

	while (random_bool(settings.bias_hard_mutation_chance))
	{
		auto& neuron = network.neurons[network.random_neuron(0, 2)];
		neuron.bias  = random_gauss_double(neuron.bias, settings.bias_hard_mutation_factor);
	}

	while (random_bool(settings.weight_mutation_chance))
	{
		auto& synapse  = network.synapses[network.random_synapse()];
		synapse.weight = random_gauss_double(synapse.weight, settings.weight_mutation_factor);
	}

	while (random_bool(settings.weight_hard_mutation_chance))
	{
		auto& synapse  = network.synapses[network.random_synapse()];
		synapse.weight = random_gauss_double(synapse.weight, settings.weight_hard_mutation_factor);
	}

	while (random_bool(settings.activation_mutation_chance))
	{
		auto& neuron             = network.neurons[network.random_neuron(0, 2)];
		neuron.activation_method = random_activation_method();
	}

	while (random_bool(settings.neuron_creation_chance))
	{
		create_random_neuron(network);
	}
}

void Mutator::randomize(Network& network)
{
	// const InitialTopology method = InitialTopology(random_int(0, int(InitialTopology::Total) - 1));
	const auto method = InitialTopology::FullyConnected;

	for (auto layer : network.layers())
	{
		for (auto& neuron : layer)
		{
			randomize(neuron);

			if (method == InitialTopology::PartiallyConnected)
			{
				neuron.synapses.erase(
					std::remove_if(
						neuron.synapses.begin(), neuron.synapses.end(), [&](const auto&) { return random_bool(0.7); }),
					neuron.synapses.end());
			}
			else if (method == InitialTopology::DisconnectedWithRandomNeurons)
			{
				neuron.synapses.clear();
			}

			for (const SynapseId synapse_id : neuron.synapses)
			{
				randomize(network.synapses[synapse_id]);
			}
		}
	}

	if (method == InitialTopology::DisconnectedWithRandomNeurons)
	{
		const std::size_t count = random_int(2, 20);
		for (std::size_t i = 0; i < count; ++i)
		{
			create_random_neuron(network);
		}
	}
}

void Mutator::randomize(Neuron& neuron)
{
	neuron.bias              = random_gauss_double(0.0, settings.bias_initial_std_dev);
	neuron.activation_method = ActivationMethod::LeakyRelu;
}

void Mutator::randomize(Synapse& synapse)
{
	synapse.weight = random_gauss_double(0.0, settings.weight_initial_std_dev);
}

ActivationMethod Mutator::random_activation_method()
{
	const double x = random_double();

	if (x < 0.6)
	{
		return ActivationMethod::Sigmoid;
	}

	if (x < 0.9)
	{
		return ActivationMethod::LeakyRelu;
	}

	return ActivationMethod::Sin;
}

uint32_t Mutator::get_unique_evolution_id() { return ++current_evolution_id; }

double Mutator::get_divergence_factor(const Network& a, const Network& b)
{
	double factor = 0.0;

	for (const Neuron& neuron : a.hidden_layer())
	{
		if (b.get_neuron_id(neuron.evolution_id) == b.neurons.size())
		{
			factor += 1.0;
		}
	}

	return factor;
}

bool MutatorSettings::load_from_file()
{
	spdlog::info("reloading mutator settings from file");

	try
	{
		std::ifstream            is("mutator.json", std::ios::binary);
		cereal::JSONInputArchive ar(is);
		serialize(ar);
	}
	catch (const cereal::Exception& e)
	{
		spdlog::error("exception occured while loading mutator settings: {}", e.what());
		return false;
	}

	return true;
}

bool MutatorSettings::save()
{
	spdlog::info("saving mutator settings to file");

	std::ofstream             os("mutator.json", std::ios::binary);
	cereal::JSONOutputArchive ar(os);
	serialize(ar);
	return true;
}

void MutatorSettings::load_defaults() { *this = {}; }
