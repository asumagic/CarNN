#pragma once
#include <cmath>

template<class T>
T clamp(T x, T lower, T upper)
{
	return std::min(upper, std::max(lower, x));
}

template<class T>
T lerp(T a, T b, float t)
{
	t = clamp(t, 0.0f, 1.0f);
	return (1.f - t) * a + (t * b);
}
