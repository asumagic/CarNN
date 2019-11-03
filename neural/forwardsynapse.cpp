#include "forwardsynapse.hpp"

#include "../randomutil.hpp"

void ForwardSynapse::randomize_parameters()
{
	// TODO: do we really want a uniform distribution?
	weight = random_double(-2.0, 2.0);
}
