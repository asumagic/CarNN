#include <carnn/training/mutator.hpp>

#include <carnn/neural/network.hpp>
#include <carnn/sim/entities/car.hpp>
#include <carnn/sim/individual.hpp>
#include <carnn/sim/simulationunit.hpp>
#include <carnn/util/random.hpp>
#include <spdlog/spdlog.h>
#include <fstream>

using namespace neural;

namespace training
{
Network Mutator::cross(Network a, const Network& b)
{
	// replace random synapses from a with ones from b
	// FIXME: O(n²) disaster below
	{
		const auto random_synapse_count = std::size_t(a.synapses.size() * settings.max_imported_synapses_factor);
		for (std::size_t i = 0; i < random_synapse_count; ++i)
		{
			Synapse& synapse = a.synapses[a.random_synapse()];

			const auto source_evoid = a.neurons[synapse.source].evolution_id;
			const auto target_evoid = a.neurons[synapse.target].evolution_id;

			// HACK: too lazy to do bullshit impl black magic because of const correctness
			if (Synapse* b_synapse
				= const_cast<Network&>(b).get_synapse(b.get_neuron_id(source_evoid), b.get_neuron_id(target_evoid));
				b_synapse != nullptr)
			{
				synapse.properties = b_synapse->properties;
			}
		}
	}

	if (util::random_bool(settings.hybridization_chance)
		&& get_divergence_factor(a, b) <= settings.max_hybridization_divergence_factor)
	{
		std::vector<NeuronId> b_imported_neurons;

		// create missing neurons first (which are necessarily in the hidden layer).
		for (std::size_t i = b.inputs().size() + b.outputs().size(); i < b.neurons.size(); ++i)
		{
			const Neuron& foreign_neuron = b.neurons[i];
			if (a.get_neuron_id(foreign_neuron.evolution_id)
				== a.neurons.size()) // TODO: less garbage interface for this
			{
				a.neurons.push_back(foreign_neuron);
				b_imported_neurons.push_back(i);
			}
		}

		// clone b synapses for imported neurons
		for (const Synapse& synapse : b.synapses)
		{
			if (std::find(b_imported_neurons.begin(), b_imported_neurons.end(), synapse.source)
					!= b_imported_neurons.end()
				|| std::find(b_imported_neurons.begin(), b_imported_neurons.end(), synapse.target)
					!= b_imported_neurons.end())
			{
				Synapse& cloned_synapse = a.create_synapse(
					a.get_neuron_id(b.neurons[synapse.source].evolution_id),
					a.get_neuron_id(b.neurons[synapse.target].evolution_id));

				cloned_synapse.properties = synapse.properties;
			}
		}
	}

	return a;
}

void Mutator::darwin(sim::Simulation& sim, std::vector<sim::Individual>& individuals)
{
	std::sort(individuals.begin(), individuals.end(), [&](const sim::Individual& a, const sim::Individual& b) {
		return sim.cars[a.car_id]->fitness() > sim.cars[b.car_id]->fitness();
	});

	float new_max_fitness = sim.cars.at(individuals[0].car_id)->fitness();

	std::ofstream fitness_csv("fitness.csv", std::ios::app);
	fitness_csv << time(nullptr) << ',' << new_max_fitness << '\n';

	if (new_max_fitness >= max_fitness + fitness_evolution_threshold)
	{
		spdlog::info(
			"entering generation {}: fitness {:.1f} exceeds old max {:.1f}",
			current_generation + 1,
			new_max_fitness,
			max_fitness);

		max_fitness = new_max_fitness;
		++current_generation;

		for (std::size_t i = 0; i < std::size_t(settings.round_survivors); ++i)
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
				individuals[util::random_int(0, settings.round_survivors - 1)].network,
				individuals[util::random_int(0, settings.round_survivors - 1)].network);

			mutate(individual.network);
		}
	}
}

void Mutator::create_random_neuron(Network& network)
{
	Neuron&  created_neuron    = network.neurons.emplace_back(get_unique_evolution_id());
	NeuronId created_neuron_id = network.neurons.size() - 1;
	randomize(created_neuron);

	std::size_t target_by_synapses = 0, source_of_synapses = 0;

	// this creates a number of synapses randomly. these synapses may connect the neuron as a source or target to any
	// other neuron, even itself (but does now allow creating duplicate synapses).

	const std::size_t extra_synapse_count
		= util::random_int(0, settings.max_extra_synapses); // TODO: gaussian distrib or something

	for (std::size_t i = 0; i < extra_synapse_count; ++i)
	{
		const NeuronId       random_neuron_id = network.random_neuron();
		const NeuronPosition pos              = network.neuron_position(random_neuron_id);

		bool is_target = false;
		switch (pos.layer)
		{
		case 0: is_target = false; break;
		case 1: is_target = util::random_bool(); break;
		case 2: is_target = true; break;
		}

		if (random_neuron_id != created_neuron_id)
		{
			if (is_target)
			{
				++target_by_synapses;
			}
			else
			{
				++source_of_synapses;
			}
		}

		if (is_target)
		{
			randomize(network.get_or_create_synapse(random_neuron_id, created_neuron_id));
		}
		else
		{
			randomize(network.get_or_create_synapse(created_neuron_id, random_neuron_id));
		}
	}

	// note that we use create_synapse and not get_or_create_synapse because we know for a fact such a synapse could
	// not have been made.

	// HACK: this should not assume the neuron ids of each layer
	if (target_by_synapses == 0)
	{
		randomize(network.create_synapse(util::random_int(0, network.inputs().size() - 1), created_neuron_id));
	}

	if (source_of_synapses == 0)
	{
		randomize(network.create_synapse(
			created_neuron_id,
			util::random_int(network.inputs().size(), network.inputs().size() + network.outputs().size() - 1)));
	}
}

void Mutator::create_random_synapse([[maybe_unused]] Network& network) {}

void Mutator::mutate(Network& network)
{
	while (util::random_bool(settings.bias_mutation_chance))
	{
		auto& neuron = network.neurons[network.random_neuron()];
		neuron.bias  = util::random_gauss_double(neuron.bias, settings.bias_mutation_factor);
	}

	while (util::random_bool(settings.bias_hard_mutation_chance))
	{
		auto& neuron = network.neurons[network.random_neuron()];
		neuron.bias  = util::random_gauss_double(neuron.bias, settings.bias_hard_mutation_factor);
	}

	while (util::random_bool(settings.weight_mutation_chance))
	{
		auto& synapse = network.synapses[network.random_synapse()];
		synapse.properties.weight
			= util::random_gauss_double(synapse.properties.weight, settings.weight_mutation_factor);
	}

	while (util::random_bool(settings.weight_hard_mutation_chance))
	{
		auto& synapse = network.synapses[network.random_synapse()];
		synapse.properties.weight
			= util::random_gauss_double(synapse.properties.weight, settings.weight_hard_mutation_factor);
	}

	while (util::random_bool(settings.activation_mutation_chance))
	{
		auto& neuron             = network.neurons[network.random_neuron()];
		neuron.activation_method = random_activation_method();
	}

	while (util::random_bool(settings.neuron_creation_chance))
	{
		create_random_neuron(network);
	}
}

void Mutator::randomize(Network& network)
{
	for (Neuron& neuron : network.neurons)
	{
		randomize(neuron);
	}

	for (Synapse& synapse : network.synapses)
	{
		randomize(synapse);
	}
}

void Mutator::randomize(Neuron& neuron)
{
	neuron.bias              = util::random_gauss_double(0.0, settings.bias_initial_std_dev);
	neuron.activation_method = ActivationMethod::LeakyRelu;
}

void Mutator::randomize(Synapse& synapse)
{
	synapse.properties.weight = util::random_gauss_double(0.0, settings.weight_initial_std_dev);
}

ActivationMethod Mutator::random_activation_method()
{
	const double x = util::random_double();

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
} // namespace training
