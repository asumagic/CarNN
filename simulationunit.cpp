#include "simulationunit.hpp"

#include "entities/car.hpp"
#include "line.hpp"
#include "world.hpp"
#include <fstream>
#include <json/reader.h>
#include <json/value.h>
#include <spdlog/spdlog.h>

Simulation::Simulation() : units(std::thread::hardware_concurrency())
{
	spdlog::info("reinitializing simulation");
	load_map("map.png");
	load_checkpoints("race.json");
	init_cars();
}

void Simulation::load_map(const char* fname)
{
	spdlog::info("loading bitmap from file '{}'", fname);

	sf::Image map;
	if (!map.loadFromFile(fname))
	{
		return;
	}

	for (auto& unit : units)
	{
		b2BodyDef bdef;
		bdef.type = b2_staticBody;
		unit.wall = &unit.world.add_body(bdef);
		unit.wall->set_type(BodyType::BodyWall);
	}

	std::vector<Line> eliminated;
	sf::Vector2u      image_size = map.getSize();
	for (unsigned y = 1; y < image_size.y - 1; ++y)
	{
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

						Line ln{{x * World::scale, y * World::scale}, {xn * World::scale, yn * World::scale}};

						const sf::Color pixel = map.getPixel(xn, yn);
						if (pixel == sf::Color::White)
						{
							if (std::find(begin(eliminated), end(eliminated), ln) == end(eliminated))
							{
								b2EdgeShape wall_shape;
								wall_shape.SetTwoSided({ln.p1.x, ln.p1.y}, {ln.p2.x, ln.p2.y});

								b2FixtureDef fixdef;
								fixdef.shape = &wall_shape;

								for (auto& unit : units)
								{
									unit.wall->add_fixture(fixdef);
								}

								wall_vertices.emplace_back(ln.p1, sf::Color::White);
								wall_vertices.emplace_back(ln.p2, sf::Color::White);

								eliminated.push_back(ln);
							}
						}
					}
			}
			else if (main_pixel.b == 255)
			{
				car_origin = {x * World::scale, y * World::scale};
			}
		}
	}
}

void Simulation::load_checkpoints(const char* fname)
{
	spdlog::info("loading checkpoints from file '{}'", fname);

	std::ifstream           race_config{fname, std::ios::binary};
	Json::Value             root;
	Json::CharReaderBuilder reader;
	Json::parseFromStream(reader, race_config, &root, nullptr);

	b2BodyDef cp_bdef;
	cp_bdef.type = b2_staticBody;

	checkpoint_vertices = sf::VertexArray(sf::Lines);

	{
		size_t i = 0;
		for (const Json::Value& cp : root["checkpoints"])
		{
			sf::Vector2f p1{
				cp["p1"].get(Json::ArrayIndex{0}, 0).asFloat() * 5.f,
				cp["p1"].get(Json::ArrayIndex{1}, 0).asFloat() * 5.f},
				p2{cp["p2"].get(Json::ArrayIndex{0}, 0).asFloat() * 5.f,
				   cp["p2"].get(Json::ArrayIndex{1}, 0).asFloat() * 5.f};

			sf::Vector2f center{p1 + (p2 - p1) / 2.f};

			// Extend the line by 5 pixels each side
			p1.x += (p1.x > center.x) ? 5.f : -5.f; // @todo compact if possible
			p2.x += (p2.x > center.x)
				? 5.f
				: -5.f; // for (float& x : {p1.x, p2.x} doesn't work since you can't get a reference to both
			p1.y += (p1.y > center.y) ? 5.f : -5.f;
			p2.y += (p2.y > center.y) ? 5.f : -5.f;

			const static sf::Color cp_col{0, 127, 0, 100};

			checkpoint_vertices.append(sf::Vertex{p1, cp_col});
			checkpoint_vertices.append(sf::Vertex{p2, cp_col});

			b2EdgeShape cp_shape;
			cp_shape.SetTwoSided(b2Vec2{p1.x, p1.y}, b2Vec2{p2.x, p2.y});

			b2FixtureDef cp_fdef;
			cp_fdef.shape    = &cp_shape;
			cp_fdef.isSensor = true;

			for (SimulationUnit& unit : units)
			{
				Checkpoint& cpb = unit.world.add_body<Checkpoint>(cp_bdef);
				cpb.origin      = center;
				cpb.id          = i++;
				cpb.add_fixture(cp_fdef);

				unit.checkpoints.push_back(&cpb);
			}
		}
	}

	for (SimulationUnit& unit : units)
	{
		unit.world.get().SetContactListener(&unit.contact_listener);
	}
}

void Simulation::init_cars()
{
	spdlog::info("spawning cars");

	b2BodyDef bdef;
	bdef.type           = b2_dynamicBody;
	bdef.angularDamping = 0.01f;

	std::array<b2Vec2, 8> vertices
		= {{{-1.50f, -0.30f},
			{-1.00f, -1.90f},
			{-0.50f, -2.10f},
			{0.50f, -2.10f},
			{1.00f, -1.90f},
			{1.50f, -0.30f},
			{1.50f, 2.00f},
			{-1.50f, 2.00f}}};

	b2PolygonShape shape;
	shape.Set(vertices.data(), vertices.size()); // shape.SetAsBox(1.5f, 2.1f);

	b2FixtureDef fixdef;
	fixdef.shape             = &shape;
	fixdef.density           = 100.f;
	fixdef.friction          = 1.0f;
	fixdef.restitution       = 0.2f;
	fixdef.filter.groupIndex = -1;

	for (std::size_t i = 0; i < 24 * 15; ++i)
	{
		SimulationUnit& unit = optimal_unit();
		Car&            car  = unit.world.add_body<Car>(bdef);
		car.unit             = &unit;
		cars.push_back(&car);
		unit.cars.push_back(&car);

		car.with_color(sf::Color{200, 50, 0, 50}).add_fixture(fixdef);
		car.transform(car_origin, static_cast<float>(0.5 * M_PI));
	}
}

SimulationUnit& Simulation::optimal_unit()
{
	const auto it = std::min_element(units.begin(), units.end(), [](const auto& a, const auto& b) {
		return a.world.body_count() < b.world.body_count();
	});

	return *it;
}
