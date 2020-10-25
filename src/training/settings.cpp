#include <carnn/training/mutator.hpp>

#include <cereal/archives/json.hpp>
#include <fstream>
#include <spdlog/spdlog.h>

namespace training
{
bool Settings::load_from_file()
{
	spdlog::info("reloading mutator settings from file");

	try
	{
		std::ifstream            is("mutator.json", std::ios::binary);
		cereal::JSONInputArchive ar(is);
		serialize(ar);
	}
	catch (const cereal::Exception& e)
	{
		spdlog::error("exception occured while loading mutator settings: {}", e.what());
		return false;
	}

	return true;
}

bool Settings::save()
{
	spdlog::info("saving mutator settings to file");

	std::ofstream             os("mutator.json", std::ios::binary);
	cereal::JSONOutputArchive ar(os);
	serialize(ar);
	return true;
}

void Settings::load_defaults() { *this = {}; }
} // namespace training
