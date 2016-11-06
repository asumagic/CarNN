#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <fstream>
#include <jsoncpp/json/value.h>
#include <jsoncpp/json/reader.h>
#include "world.hpp"
#include "entities/checkpoint.hpp"
#include "entities/wheel.hpp"
#include "entities/car.hpp"
#include "neural/network.hpp"
#include "randomutil.hpp"

int app(sf::RenderWindow& win)
{
	World w{b2Vec2{0.f, 0.f}};

	std::ifstream race_config{"race.json", std::ios::binary};
	Json::Value root;
	Json::Reader reader;
	reader.parse(race_config, root, false);

	Body* wall_body;
	b2Vec2 car_pos;
	std::vector<sf::Vertex> wall_vertices = w.import_map(root["filepath"].asString(), wall_body, car_pos);

	b2BodyDef cp_bdef;
	cp_bdef.type = b2_staticBody;
	sf::VertexArray checkpoint_vertices{sf::Lines};
	{
		size_t i = 0;
		for (const Json::Value& cp : root["checkpoints"])
		{
			sf::Vector2f p1{cp["p1"].get(Json::ArrayIndex{0}, 0).asFloat() * 5.f, cp["p1"].get(Json::ArrayIndex{1}, 0).asFloat() * 5.f},
						 p2{cp["p2"].get(Json::ArrayIndex{0}, 0).asFloat() * 5.f, cp["p2"].get(Json::ArrayIndex{1}, 0).asFloat() * 5.f};

			sf::Vector2f center{p1 + (p2 - p1) / 2.f};

			// Extend the line by 5 pixels each side
			p1.x += (p1.x > center.x) ? 5.f : -5.f;  // @todo compact if possible
			p2.x += (p2.x > center.x) ? 5.f : -5.f;  // for (float& x : {p1.x, p2.x} doesn't work since you can't get a reference to both
			p1.y += (p1.y > center.y) ? 5.f : -5.f;
			p2.y += (p2.y > center.y) ? 5.f : -5.f;

			const static sf::Color cp_col{0, 127, 0, 100};

			checkpoint_vertices.append(sf::Vertex{p1, cp_col});
			checkpoint_vertices.append(sf::Vertex{p2, cp_col});

			b2EdgeShape cp_shape;
			cp_shape.Set(b2Vec2{p1.x, p1.y}, b2Vec2{p2.x, p2.y});

			b2FixtureDef cp_fdef;
			cp_fdef.shape = &cp_shape;
			cp_fdef.isSensor = true;

			Checkpoint& cpb = w.add_body<Checkpoint>(cp_bdef);
			cpb.set_id(i++);
			cpb.add_fixture(cp_fdef);
		}
	}

	/*** Define a car ***/
	b2BodyDef bdef;
	bdef.position = {0.f, 1.3f};
	bdef.type = b2_dynamicBody;

	std::array<b2Vec2, 8> vertices = {{
		{-1.50f, -0.30f},
		{-1.00f, -1.90f},
		{-0.50f, -2.10f},
		{ 0.50f, -2.10f},
		{ 1.00f, -1.90f},
		{ 1.50f, -0.30f},
		{ 1.50f,  2.00f},
		{-1.50f,  2.00f}
	}};

	b2PolygonShape shape;
	shape.Set(vertices.data(), vertices.size()); // shape.SetAsBox(1.5f, 2.1f);

	b2FixtureDef fixdef;
	fixdef.shape = &shape;
	fixdef.density = 50.f;
	fixdef.friction = 0.4f;
	fixdef.restitution = 0.2f;

	Car& c = w.add_body<Car>(bdef);
	c.with_color(sf::Color{200, 50, 0}).add_fixture(fixdef);
	c.transform(car_pos, static_cast<float>(0.5 * M_PI));

	// Add the listener AFTER moving the car
	CarCheckpointListener listener;
	w.get().SetContactListener(&listener);

	Network net{400, 5, total_rays + 2};
	c.add_synapses(net);

	// Define the time data (delta time + simulation speed)
	sf::Clock dtclock;
	float speed = 10.f;

	// Define the camera settings
	float czoom = 0.1f;

	sf::Font infofnt;
	if (!infofnt.loadFromFile("Cantarell-Bold.ttf"))
		std::cerr << "WARNING: Failed to load font." << std::endl;

	std::vector<Axon>& axons = net.axons();
	const std::array<const std::string, 5> labels {{ "Fwd", "Bck", "Left", "Right", "Drift" }};
	for (size_t i = 0; i < axons.size(); ++i)
	{
		axons[i].set_font(&infofnt);
		axons[i].set_label(labels[i]);
	}

	sf::Text car_info;
	car_info.setFont(infofnt);
	car_info.setColor(sf::Color::White);
	car_info.setCharacterSize(30);
	car_info.setScale(0.4f, 0.4f);
	car_info.setPosition(16.f, 16.f);
	car_info.setStyle(sf::Text::Bold);

	sf::Text fps_info;
	fps_info.setFont(infofnt);
	fps_info.setColor(sf::Color{80, 80, 80});
	fps_info.setCharacterSize(30);
	fps_info.setScale(0.4f, 0.4f);
	fps_info.setStyle(sf::Text::Bold);

	sf::Event ev;
	while (win.isOpen())
	{
		while (win.pollEvent(ev))
		{
			switch (ev.type)
			{
			case sf::Event::Closed:
				win.close();
				break;
			case sf::Event::KeyPressed:
				if (ev.key.code == sf::Keyboard::Escape)    { return 0; }
				else if (ev.key.code == sf::Keyboard::R)    { return 2; }
				break;
			case sf::Event::MouseWheelScrolled:
				czoom = std::max(czoom - (ev.mouseWheelScroll.delta * .01f), 0.01f);
				break;
			}
		}

		const b2Vec2 b2target = c.get().GetPosition();
		w.update_view(win, sf::Vector2f{b2target.x, b2target.y}, czoom);

		// @TODO move this mess
		const auto results = net.results();

		c.set_drift(static_cast<float>(results[Axon_Drift]));
		c.apply_torque(static_cast<float>(results[Axon_Steer_Right] - results[Axon_Steer_Left]));
		c.accelerate(static_cast<float>(results[Axon_Forward] - results[Axon_Backwards]));

		win.clear(sf::Color{20, 20, 20});
		c.compute_raycasts(*wall_body);
		win.draw(checkpoint_vertices);
		w.step(speed, 6, 2).update().render(win);
		win.draw(wall_vertices.data(), wall_vertices.size(), sf::Lines);
		w.set_dt(dtclock.restart().asSeconds());

		sf::View cview{win.getView()};
		win.setView(sf::View{sf::FloatRect{0.f, 0.f, static_cast<float>(win.getSize().x), static_cast<float>(win.getSize().y)}});

		car_info.setString(sf::String("Speed : ") + std::to_string(static_cast<unsigned short>(c.forward_velocity().Length() * 20.f)) + "km/h");

		fps_info.setString(std::to_string(static_cast<unsigned short>(1.f / w.dt())) + "fps");
		fps_info.setPosition(8.f, win.getSize().y - 16.f);

		net.update();
		net.render(win);

		win.draw(car_info);
		win.draw(fps_info);

		win.setView(cview);

		win.display();
	}

	return 0;
}

int main()
{
	// Define the render window
	sf::ContextSettings settings;
	settings.antialiasingLevel = 16;
	sf::RenderWindow win{sf::VideoMode{800, 600}, "Neural Network", sf::Style::Default, settings};
	win.setFramerateLimit(120);

	while (app(win) == 2);
}
