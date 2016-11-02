#ifndef RANDOMUTIL_HPP
#define RANDOMUTIL_HPP

#include <random>

extern std::random_device device;
extern std::mt19937_64 mersenne;

double random_double(double min, double max);
int random_int(int min, int max);
bool random_bool(const double probability);

#endif // RANDOMUTIL_HPP
