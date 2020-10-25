#pragma once

#include <SFML/System/Vector2.hpp>

struct Line
{
	sf::Vector2f p1, p2;

	bool strictly_equal(const Line& other) const
	{
		return p1 == other.p1 && p2 == other.p2;
	}

	bool operator==(const Line& other) const
	{
		return strictly_equal(other) || strictly_equal({other.p2, other.p1});
	}

};
