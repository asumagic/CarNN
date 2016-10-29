#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include "world.hpp"
#include <iostream>

int main()
{
	World w;

	{ // add a circle shape body in the world
		b2BodyDef bdef;
		bdef.position = {10.f, 10.f};
		bdef.type = b2_dynamicBody;

		b2CircleShape shape;
		shape.m_p.Set(0.f, 0.f);
		shape.m_radius = 1.f;

		b2FixtureDef fixdef;
		fixdef.shape = &shape;
		fixdef.density = 1.f;

		w.add_body(bdef).add_fixture(fixdef);
	}

	{ // add a circle shape body in the world
		b2BodyDef bdef;
		bdef.position = {-5.f, 10.f};
		bdef.type = b2_dynamicBody;

		b2CircleShape shape;
		shape.m_p.Set(0.f, 0.f);
		shape.m_radius = 1.f;

		b2FixtureDef fixdef;
		fixdef.shape = &shape;
		fixdef.density = 1.f;

		Body& b = w.add_body(bdef);
		b.add_fixture(fixdef);
		b.get().ApplyForce(b2Vec2{100.f, 0.f}, b2Vec2{0.f, 0.f}, true);
	}

	sf::ContextSettings settings;
	settings.antialiasingLevel = 4;

	sf::RenderWindow win{sf::VideoMode{800, 600}, "Neural Network"/*, sf::Style::Default, settings*/};
	win.setFramerateLimit(60);

	sf::Clock dtclock;

	float slowdown = .02f;

	sf::View view{sf::Vector2f{0.f, 0.f}, sf::Vector2f{80.f, 60.f}};
	win.setView(view);

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
				if (ev.key.code == sf::Keyboard::Escape) win.close();
				break;
			case sf::Event::Resized:
				win.setView(sf::View{sf::Vector2f{0.f, 0.f}, sf::Vector2f{ev.size.width / 10.f, ev.size.height / 10.f}});
				break;
			}
		}

		win.clear(sf::Color::Black);
		w.step(dtclock.restart().asSeconds() / slowdown, 6, 2).update().render(win);
		win.display();
	}

	return 0;
}
