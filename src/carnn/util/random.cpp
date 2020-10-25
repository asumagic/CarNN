#include <carnn/util/random.hpp>

#include <random>

std::random_device device;
std::mt19937_64    mersenne(device());

double random_double(double min, double max)
{
	std::uniform_real_distribution<double> distrib(min, max);
	return distrib(mersenne);
}

double random_double() { return random_double(0.0, 1.0); }

double random_gauss_double(double mean, double stddev)
{
	std::normal_distribution<double> distrib(mean, stddev);
	return distrib(mersenne);
}

int random_int(int min, int max)
{
	std::uniform_int_distribution<int> distrib(min, max);
	return distrib(mersenne);
}

bool random_bool(const double probability)
{
	std::bernoulli_distribution distrib(probability);
	return distrib(mersenne);
}
