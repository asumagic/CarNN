#include "randomutil.hpp"

std::random_device device;
std::mt19937_64 mersenne(device());

double random_double(double min, double max)
{
	std::uniform_real_distribution<double> distrib(min, max);
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
