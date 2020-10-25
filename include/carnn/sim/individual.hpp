#pragma once

#include <carnn/neural/network.hpp>
#include <cstdint>

struct Individual
{
	std::uint32_t car_id = 0;

	Network network;

	bool   survivor_from_last = false;
	double last_fitness       = 0.0;

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(car_id, network, survivor_from_last, last_fitness);
	}
};
