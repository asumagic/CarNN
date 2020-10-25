#pragma once

namespace util
{
double random_double(double min, double max);
double random_double();
double random_gauss_double(double mean, double stddev);
int    random_int(int min, int max);
bool   random_bool(double probability = 0.5);
} // namespace util
