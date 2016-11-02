#ifndef GA_HPP
#define GA_HPP

class GeneticAlgorithm
{
public:
	GeneticAlgorithm(const double mutation_rate, const double children_rate);

private:
	const double _mutation_rate, _children_rate;
};

#endif // GA_HPP
