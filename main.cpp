#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include "world.hpp"
#include "entities/wheel.hpp"
#include "entities/car.hpp"
#include <iostream>

int main()
{
	World w{b2Vec2{0.f, 0.f}};

	for (int i = 0; i < 1000; ++i)
	{
		b2BodyDef bdef;
		bdef.position = {(float)(rand() % 500 + 10), (float)(rand() % 500 + 10)};
		bdef.type = b2_dynamicBody;

		b2PolygonShape shape;
		shape.SetAsBox(1.5f, 1.5f);

		b2FixtureDef fixdef;
		fixdef.shape = &shape;
		fixdef.density = 2000.f;
		fixdef.friction = 0.2f;

		Body& box = w.add_body(bdef).with_color(sf::Color{static_cast<uint8_t>(rand() % 150), static_cast<uint8_t>(rand() % 150), static_cast<uint8_t>(rand() % 150)}); // @TODO fix this uglyness
		box.get().SetTransform(box.get().GetPosition(), (rand() % 10000) / 300.f); // @TODO fix this uglyness
		box.add_fixture(fixdef);
	}

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
	if (!infofnt.loadFromFile("Calibri.ttf"))
		std::cerr << "WARNING: Failed to load font." << std::endl;

	sf::Text car_info;
	car_info.setFont(infofnt);
	car_info.setColor(sf::Color::White);
	car_info.setCharacterSize(14);
	car_info.setPosition(16.f, 16.f);
	car_info.setStyle(sf::Text::Bold);

	bool use_joystick = false;
	if (sf::Joystick::isConnected(0))
	{
		std::cout << "Controller found." << std::endl;
		use_joystick = true;
	}

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
				break;
			case sf::Event::MouseWheelScrolled:
				czoom = std::max(czoom - (ev.mouseWheelScroll.delta * .01f), 0.04f);
				break;
			}
		}

		const b2Vec2 b2target = c.get().GetPosition();
		w.update_view(win, sf::Vector2f{b2target.x, b2target.y}, czoom, c.get().GetAngle());

		// @TODO move this mess
		c.set_drift(sf::Joystick::isButtonPressed(0, 0));

		int hpress; float by = 0.f;
		if (use_joystick)
		{
			by = sf::Joystick::getAxisPosition(0, sf::Joystick::Axis::X);
			if (by < 0)
				hpress = -1;
			else if (by > 0)
				hpress = 1;
			else
				hpress = 0;
		}
		else
		{
			hpress = -sf::Keyboard::isKeyPressed(sf::Keyboard::Left) + sf::Keyboard::isKeyPressed(sf::Keyboard::Right);
		}
		c.apply_torque(hpress == -1 ? Direction::Left : (hpress == 1 ? Direction::Right : Direction::None), by);

		int vpress;
		if (use_joystick)
			vpress = sf::Joystick::isButtonPressed(0, 5) - sf::Joystick::isButtonPressed(0, 7);
		else
			vpress = -sf::Keyboard::isKeyPressed(sf::Keyboard::Up) + sf::Keyboard::isKeyPressed(sf::Keyboard::Down);

		if (vpress != 0)
		{
			c.accelerate(vpress == -1 ? VDirection::Forward : VDirection::Backwards);
		}

		win.clear(sf::Color{20, 20, 20});
		w.step(speed, 6, 2).update().render(win);
		w.set_dt(dtclock.restart().asSeconds());

		sf::View cview{win.getView()};
		win.setView(sf::View{sf::FloatRect{0.f, 0.f, static_cast<float>(win.getSize().x), static_cast<float>(win.getSize().y)}});

		car_info.setString(sf::String("Framerate : ") + std::to_string(static_cast<unsigned short>(1.f / w.dt())) + sf::String("fps\nSpeed : ") +
						   std::to_string(static_cast<unsigned short>(c.forward_velocity().Length() * 20.f)) + "km/h");
		win.draw(car_info);

		win.setView(cview);

		win.display();
	}

	return 0;
}
