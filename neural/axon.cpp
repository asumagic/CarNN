#include "../randomutil.hpp"
#include "../maths.hpp"
#include "axon.hpp"

static const float graph_margin = 5.f,
				   xoff = 16.f, yoff = 48.f,
				   trace_height = 96.f,
				   graph_height = trace_height + graph_margin * 2.f,
				   graph_margout = 6.f;

Axon::Axon(const size_t index) : _shape{5.}, _weight_history{sf::LinesStrip, axon_history_size}
{
	for (size_t i = 0; i < axon_history_size; ++i)
		_weight_history[i].position = {xoff + ((static_cast<float>(i) / static_cast<float>(axon_history_size)) * graph_size), graph_point_y(0.f, index)};
}

void Axon::tie_neuron(Neuron& n)
{
	_neurons.emplace_back(n, random_double(0.0, 1.0));
}

void Axon::set_label(const std::string label)
{
	_label = label;
}

void Axon::set_font(sf::Font* fnt)
{
	_font = fnt;
}

sf::Vector2f Axon::screen_position(const sf::Vector2u window_size, const size_t index)
{
	return sf::Vector2f{static_cast<float>(window_size.x - ((index + 2) * 13)), 64.f};
}

void Axon::update(const size_t index)
{
	if (++_current_history_interval > _update_history_interval)
	{
		for (size_t i = 1; i < axon_history_size; ++i)
			_weight_history[i - 1].position.y = _weight_history[i].position.y; // Shift previous vertices

		_weight_history[axon_history_size - 1].position.y = graph_point_y(static_cast<float>(_value), index);
		_current_history_interval = 0;

		for (size_t i = 0; i < axon_history_size - 5; ++i)
		{
			float cval = 1.f - ((_weight_history[i].position.y - yoff + graph_margin - (index * (graph_height + graph_margout))) / trace_height);
			_weight_history[i].color = sf::Color{0,
												 static_cast<uint8_t>(lerp(40, 70, cval)),
												 static_cast<uint8_t>(lerp(75, 200, cval))};
		}

		_weight_history[_weight_history.getVertexCount() - 1].color = sf::Color::White;
	}

	for (NeuronOutputWeighted& neuron : _neurons)
		_value = neuron.get();
}

void Axon::render(sf::RenderTarget& target, const size_t index)
{
	_shape.setPosition(screen_position(target.getSize(), index)); // @TODO : don't do this when not required
	if (_value >= 0. && _value <= 1.)
		_shape.setFillColor(sf::Color{0, static_cast<uint8_t>(lerp(50, 80, static_cast<float>(_value))), static_cast<uint8_t>(lerp(50, 255, static_cast<float>(_value)))});
	else
		_shape.setFillColor(sf::Color::Red);

	// @todo build the rectangle shape once
	sf::RectangleShape graph_back;
	graph_back.setPosition(xoff - graph_margin, yoff - graph_margin + index * (graph_height + graph_margout));
	graph_back.setSize(sf::Vector2f{graph_size + graph_margin * 2.f, graph_height});
	graph_back.setFillColor(sf::Color{0, 0, 0, 60});
	target.draw(graph_back);
	target.draw(_weight_history);

	if (_font && !_label.empty())
	{
		sf::Text text;
		text.setString(_label);
		text.setFont(*_font);
		text.setCharacterSize(10.f);
		text.setColor(sf::Color{255, 255, 255, 50});
		text.setRotation(90);
		text.setPosition(xoff + graph_size + 2.f * graph_margin + 6.f, yoff + (index * (graph_height + graph_margout)));
		target.draw(text);
	}

	target.draw(_shape);
}

float Axon::graph_point_y(const float value, const size_t index)
{
	return yoff - graph_margin + (index * (graph_height + graph_margout) + (1.f - value) * trace_height);
}

NeuronOutputWeighted::NeuronOutputWeighted(Neuron& neuron, double weight) : n(neuron), weight(weight) {}

double NeuronOutputWeighted::get()
{
	return n.sigmoid() * weight;
}
