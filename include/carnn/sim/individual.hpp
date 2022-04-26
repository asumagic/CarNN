#pragma once

#include <carnn/neural/network.hpp>
#include <cstdint>

namespace sim
{
struct Individual
{
	std::uint32_t car_id = 0;

	neural::Network network;

	bool   survivor_from_last = false;

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(car_id, network, survivor_from_last);
	}
};
} // namespace sim
