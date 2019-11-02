#include "network.hpp"

#include "../entities/car.hpp"

Network::Network(size_t outputs, size_t synapses)
{
	_synapses.reserve(synapses);
	_axons.reserve(outputs);

	for (size_t i = 0; i < outputs; ++i)
		_axons.emplace_back(i);

	_neuron_layers.emplace_back(_axons.size());

	for (Axon& axon : _axons)
	for (Neuron& neuron : _neuron_layers.back())
	{
		axon.tie_neuron(neuron);
	}
}

Network& Network::update()
{
	for (auto& layer : _neuron_layers)
	for (Neuron& neuron : layer)
		neuron.update();

	for (size_t i = 0; i < _axons.size(); ++i)
		_axons[i].update(i);

	return *this;
}

std::vector<double> Network::results()
{
	std::vector<double> results;

	results.reserve(_axons.size());

	for (Axon& axon : _axons)
		results.push_back(axon.read());

	return results;
}

Synapse& Network::add_synapse(double& input)
{
	_synapses.emplace_back(input);
	Synapse& syn = _synapses.back();

	for (Neuron& neuron : _neuron_layers.front())
		neuron.add_synapse(syn);

	return syn;
}

void Network::render(sf::RenderTarget& target)
{
	for (size_t i = 0; i < _synapses.size(); ++i)
		_synapses[i].render(target, i);

	{
		std::size_t i = 0;
		for (auto& layer : _neuron_layers)
		for (auto& neuron : layer)
		{
			neuron.render(target, ++i);
		}
	}

	for (size_t i = 0; i < _axons.size(); ++i)
		_axons[i].render(target, i);
}

bool proper::NetworkResult::operator<(const proper::NetworkResult& other) const
{
	return car->fitness() < other.car->fitness();
}

bool proper::NetworkResult::operator>(const proper::NetworkResult& other) const
{
	return car->fitness() > other.car->fitness();
}

proper::Network proper::Mutator::cross(const proper::Network& a, const proper::Network& b)
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

void proper::Mutator::darwin(std::vector<proper::NetworkResult> results)
{
	float new_max_fitness = std::max_element(results.begin(), results.end())->car->fitness();

	if (new_max_fitness >= max_fitness + fitness_evolution_threshold)
	{
		std::sort(results.begin(), results.end(), std::greater{});
		max_fitness = new_max_fitness;
		++current_generation;
	}

	for (std::size_t i = 0; i < noah; ++i)
	{
		results[i].car->with_color(sf::Color{200, 50, 0, 50});
	}

	for (std::size_t i = noah; i < results.size(); ++i)
	{
		results[i].car->with_color(sf::Color{0, 0, 100, 70});

		// TODO: this allows self breeding, do we allow it?
		*results[i].network = cross(
					*results[random_int(0, noah - 1)].network,
				*results[random_int(0, noah - 1)].network
				);

		mutate(*results[i].network);
	}

	results[0].network->dump();
}
