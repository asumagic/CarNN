#include "checkpoint.hpp"

#include "../entities/car.hpp"

Checkpoint::Checkpoint(World& world, const b2BodyDef bdef, const bool do_render) : Body(world, bdef, do_render)
{
	set_type(BodyType::BodyCheckpoint);
}
