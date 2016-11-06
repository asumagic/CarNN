#ifndef GA_HPP
#define GA_HPP

#include <functional>
#include <vector>

struct Genome
{
	int population;
	double fitness;
	std::vector<double> weights;

	bool operator<(const Genome& other) { return fitness < other.fitness; }

	Genome crossover(Genome& other);
	void mutate(const double rate, const double range_min = 0.0, const double range_max = 1.0);
};

class GeneticAlgorithm
{
public:
	GeneticAlgorithm(const double mutation_rate, size_t scoreboard_size);

	std::vector<Genome>& update();

private:
	const double _mutation_rate;
	std::vector<Genome> _population;
};

#endif // GA_HPP
