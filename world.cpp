#include "world.hpp"
#include "line.hpp"
#include "maths.hpp"

World::World(const b2Vec2 gravity) : _gravity{gravity}, _world{gravity} {}

World& World::step(const float speed, const int vel_it, const int pos_it)
{
	_world.Step(_dt * speed, vel_it, pos_it);
	return *this;
}

World& World::update()
{
	// Update bodies
	for (auto& b : _bodies)
		b->update();

	return *this;
}

World& World::render(sf::RenderTarget& target)
{
	// Render bodies
	for (auto& b : _bodies)
		b->render(target);

	return *this;
}

void World::set_dt(const float dt)
{
	_dt = dt;
	//_dt = lerp(_dt, dt, 0.8f);
}

float World::dt() const { return _dt; }

void World::update_view(sf::RenderTarget& target, sf::Vector2f origin, float czoom)
{
	sf::View new_view{
		lerp(target.getView().getCenter(), origin, 0.2f * _dt * 60.f), sf::Vector2f{target.getSize()} * czoom};
	target.setView(new_view);
}

std::vector<sf::Vertex> World::import_map(const std::string fname, Body*& wall, b2Vec2& car_origin)
{
	std::vector<sf::Vertex> ret;

	sf::Image map;
	if (!map.loadFromFile(fname))
	{
		return ret;
	}

	b2BodyDef bdef;
	bdef.type = b2_staticBody;
	wall      = &add_body(bdef);
	wall->set_type(BodyType::BodyWall);

	std::vector<Line> eliminated;
	sf::Vector2u      image_size = map.getSize();
	for (unsigned y = 1; y < image_size.y - 1; ++y)
		for (unsigned x = 1; x < image_size.x - 1; ++x)
		{
			const sf::Color main_pixel = map.getPixel(x, y);
			if (main_pixel == sf::Color::White)
			{
				for (unsigned yn = y - 1; yn < y + 2; ++yn)
					for (unsigned xn = x - 1; xn < x + 2; ++xn)
					{
						if (xn == x && yn == y)
						{
							continue;
						}

						Line ln{{x * world_scale, y * world_scale}, {xn * world_scale, yn * world_scale}};

						const sf::Color pixel = map.getPixel(xn, yn);
						if (pixel == sf::Color::White)
						{
							if (std::find(begin(eliminated), end(eliminated), ln) == end(eliminated))
							{
								b2EdgeShape wall_shape;
								wall_shape.SetTwoSided({ln.p1.x, ln.p1.y}, {ln.p2.x, ln.p2.y});

								b2FixtureDef fixdef;
								fixdef.shape = &wall_shape;

								wall->add_fixture(fixdef);

								ret.emplace_back(ln.p1, sf::Color::White);
								ret.emplace_back(ln.p2, sf::Color::White);

								eliminated.push_back(ln);
							}
						}
					}
			}
			else if (main_pixel.b == 255)
			{
				car_origin = {x * world_scale, y * world_scale};
			}
		}

	return ret;
}

b2World& World::get() { return _world; }
