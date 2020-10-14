#pragma once

#include <string_view>

enum class ActivationMethod
{
	Sigmoid,
	LeakyRelu,
	Sin,

	Total
};

std::string_view name(ActivationMethod method);
