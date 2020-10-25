#pragma once

#include <string_view>

namespace neural
{
enum class ActivationMethod : std::uint8_t
{
	Sigmoid,
	LeakyRelu,
	Sin,

	Total
};

std::string_view name(ActivationMethod method);
} // namespace neural
