#ifndef CHECKPOINT_HPP
#define CHECKPOINT_HPP

#include "../body.hpp"

class Checkpoint : public Body
{
public:
	Checkpoint(World& world, const b2BodyDef bdef, const bool do_render = true);
	void set_id(const size_t id);
	size_t get_id() const;

private:
	size_t _id;
};

#endif // CHECKPOINT_HPP
