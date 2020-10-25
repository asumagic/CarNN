#pragma once

#include <SFML/Graphics/Color.hpp>
#include <SFML/System/Vector2.hpp>
#include <cmath>

namespace util
{
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

inline sf::Color lerp_rgb(sf::Color a, sf::Color b, float t)
{
	return {
		lerp(a.r, b.r, t),
		lerp(a.g, b.g, t),
		lerp(a.b, b.b, t),
		lerp(a.a, b.a, t),
	};
}

template<class T>
T pow2(T x)
{
	return x * x;
}

inline float distance(sf::Vector2f a, sf::Vector2f b) { return std::sqrt(pow2(a.x - b.x) + pow2(a.y - b.y)); }
} // namespace util
