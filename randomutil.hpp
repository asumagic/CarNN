#ifndef RANDOMUTIL_HPP
#define RANDOMUTIL_HPP

#include <random>

extern std::random_device device;
extern std::mt19937_64 mersenne;

double random_double(double min, double max);
double random_double();
double random_gauss_double(double mean, double stddev);
int random_int(int min, int max);
bool random_bool(double probability = 0.5);

#endif // RANDOMUTIL_HPP
