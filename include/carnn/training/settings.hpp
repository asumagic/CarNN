#pragma once

#include <cereal/cereal.hpp>
#include <cstdint>

namespace training
{
struct Settings
{
	using Fp = float;

	Fp bias_initial_std_dev      = 0.05;
	Fp bias_mutation_factor      = 0.07;
	Fp bias_mutation_chance      = 0.7;
	Fp bias_hard_mutation_factor = 0.2;
	Fp bias_hard_mutation_chance = 0.2;

	Fp weight_initial_std_dev      = 0.05;
	Fp weight_mutation_factor      = 0.07;
	Fp weight_mutation_chance      = 0.7;
	Fp weight_hard_mutation_factor = 0.2;
	Fp weight_hard_mutation_chance = 0.2;

	Fp activation_mutation_chance = 0.3;

	Fp           neuron_creation_chance = 0.1;
	std::int32_t max_extra_synapses     = 10;

	Fp max_imported_synapses_factor        = 0.2;
	Fp hybridization_chance                = 0.2;
	Fp max_hybridization_divergence_factor = 2.0;

	Fp conservative_gc_chance = 0.0;
	Fp aggressive_gc_chance   = 0.0;

	std::int32_t round_survivors = 20;

	bool load_from_file();
	bool save();
	void load_defaults();

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(bias_initial_std_dev),
		   CEREAL_NVP(bias_mutation_factor),
		   CEREAL_NVP(bias_mutation_chance),
		   CEREAL_NVP(bias_hard_mutation_factor),
		   CEREAL_NVP(bias_hard_mutation_chance),

		   CEREAL_NVP(weight_initial_std_dev),
		   CEREAL_NVP(weight_mutation_factor),
		   CEREAL_NVP(weight_mutation_chance),
		   CEREAL_NVP(weight_hard_mutation_factor),
		   CEREAL_NVP(weight_hard_mutation_chance),

		   CEREAL_NVP(activation_mutation_chance),

		   CEREAL_NVP(neuron_creation_chance),
		   CEREAL_NVP(max_extra_synapses),

		   CEREAL_NVP(max_imported_synapses_factor),
		   CEREAL_NVP(hybridization_chance),
		   CEREAL_NVP(max_hybridization_divergence_factor),

		   CEREAL_NVP(round_survivors));
	}
};
}
