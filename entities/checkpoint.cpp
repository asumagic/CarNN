#include "checkpoint.hpp"

Checkpoint::Checkpoint(World& world, const b2BodyDef bdef, const bool do_render) : Body(world, bdef, do_render)
{
	set_type(BodyType::BodyCheckpoint);
}

void Checkpoint::set_id(const size_t id)
{
	_id = id;
}

size_t Checkpoint::get_id() const
{
	return _id;
}
