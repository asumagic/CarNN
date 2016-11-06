#include "ga.hpp"
#include <cassert>
#include <algorithm>
#include "../randomutil.hpp"

Genome Genome::crossover(Genome &other)
{
	assert(other.weights.size() == weights.size());

	Genome child;
	child.weights.resize(weights.size());
	for (size_t i = 0; i < weights.size(); ++i)
		child.weights[i] = (random_bool(0.5) ? weights[i] : other.weights[i]);

	return child;
}

void Genome::mutate(const double rate, const double range_min, const double range_max)
{
	// @TODO optimize
	for (double& gene : weights)
		if (random_double(range_min, rate) < rate)
			gene = random_double(range_min, range_max);
}

GeneticAlgorithm::GeneticAlgorithm(const double mutation_rate, size_t scoreboard_size) : _mutation_rate(mutation_rate), _population(scoreboard_size)
{
	assert(scoreboard_size >= 4);
}

std::vector<Genome> &GeneticAlgorithm::update()
{
	std::sort(begin(_population), end(_population)); // Worst first

	// Keep the best genome on the next iteration

	// Keep a non-mutated crossover of the two best ones
	_population[_population.size() - 1] = _population[_population.size() - 1].crossover(_population.back());

	// Crossover and mutate a half
	for (size_t i = _population.size() / 2; i < _population.size() - 2; ++i)
	{
		_population[i] = _population[_population.size() - 1].crossover(_population.back());
		_population[i].mutate(_mutation_rate, 0., 1.);
	}

	// Mutate the rest
	for (size_t i = 0; i < _population.size() / 2; ++i)
		_population[i].mutate(_mutation_rate, 0., 1.);

	return _population;
}
