#pragma once

#include <SFML/System/Vector2.hpp>
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

template<class T>
T pow2(T x)
{
	return x * x;
}

inline float distance(sf::Vector2f a, sf::Vector2f b) { return std::sqrt(pow2(a.x - b.x) + pow2(a.y - b.y)); }
