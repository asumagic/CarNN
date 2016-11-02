#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include "world.hpp"
#include "entities/wheel.hpp"
#include "entities/car.hpp"
#include "neural/network.hpp"
#include <iostream>
#include "randomutil.hpp"
int main()
{
	World w{b2Vec2{0.f, 0.f}};

	/*** Push box obstacles ***/
	std::vector<Body*> obstacles;
	b2Vec2 car_pos;
	std::vector<sf::Vertex> wall_vertices = w.import_map("race.png", obstacles, car_pos);
	/*obstacles.reserve(500);
	for (int i = 0; i < 250; ++i)
	{
		b2BodyDef bdef;
		bdef.position = {(float)(rand() % 100 + 10), (float)(rand() % 100 + 10)};
		bdef.type = b2_dynamicBody;

		b2PolygonShape shape;
		shape.SetAsBox(1.5f, 1.5f);

		b2FixtureDef fixdef;
		fixdef.shape = &shape;
		fixdef.density = 200.f;
		fixdef.friction = 0.2f;

		Body& box = w.add_body(bdef).with_color(sf::Color{static_cast<uint8_t>(rand() % 150), static_cast<uint8_t>(rand() % 150), static_cast<uint8_t>(rand() % 150)}); // @TODO fix this uglyness
		box.get().SetTransform(box.get().GetPosition(), (rand() % 10000) / 300.f); // @TODO fix this uglyness
		box.add_fixture(fixdef);
		obstacles.push_back(&box);
	}*/

	/*** Define a car ***/
	b2BodyDef bdef;
	bdef.position = {0.f, 1.3f};
	bdef.type = b2_dynamicBody;

	std::array<b2Vec2, 8> vertices = {{
		{-1.50f, -0.30f},
		{-1.00f, -2.00f},
		{-0.50f, -2.10f},
		{ 0.50f, -2.10f},
		{ 1.00f, -2.00f},
		{ 1.50f, -0.30f},
		{ 1.50f,  2.00f},
		{-1.50f,  2.00f}
	}};

	b2PolygonShape shape;
	shape.Set(vertices.data(), vertices.size()); // shape.SetAsBox(1.5f, 2.1f);

	b2FixtureDef fixdef;
	fixdef.shape = &shape;
	fixdef.density = 50.f;
	fixdef.friction = 0.2f;

	Car& c = w.add_body<Car>(bdef);
	c.with_color(sf::Color{200, 50, 0}).add_fixture(fixdef);
	c.transform(car_pos, static_cast<float>(0.5 * M_PI));

	Network net{200, 5, total_rays + 2};
	c.add_synapses(net);

	/*** Define rendering details ***/

	// Define the render window
	sf::ContextSettings settings;
	settings.antialiasingLevel = 16;
	sf::RenderWindow win{sf::VideoMode{800, 600}, "Neural Network", sf::Style::Default, settings};
	win.setFramerateLimit(120);

	// Define the time data (delta time + simulation speed)
	sf::Clock dtclock;
	float speed = 10.f;

	// Define the camera settings
	float czoom = 0.1f;

	sf::Font infofnt;
	if (!infofnt.loadFromFile("Cantarell-Bold.ttf"))
		std::cerr << "WARNING: Failed to load font." << std::endl;

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
				if (ev.key.code == sf::Keyboard::Escape)    win.close();
				else if (ev.key.code == sf::Keyboard::R)    { net = Network{200, 5, total_rays + 2}; c.add_synapses(net); }
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
		c.compute_raycasts(obstacles);
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
