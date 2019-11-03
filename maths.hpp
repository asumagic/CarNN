#pragma once
#include <cmath>

template<class T, class U>
T lerp(T a, T b, U t)
{
	return (1.f - t) * a + (t * b);
}

template<class T>
inline T sigmoid(T v)
{
	return 1.0 / (1.0 + std::exp(-v));
}
