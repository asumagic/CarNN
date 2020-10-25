#include <carnn/neural/visualizer.hpp>

#include <carnn/neural/network.hpp>
#include <carnn/util/maths.hpp>
#include <fmt/core.h>
#include <random>

constexpr float layer_padding       = 256.0f;
constexpr float neuron_area         = 700.0f;
constexpr float hidden_layer_jitter = 160.0f;

constexpr float neuron_radius = 12.0f;

Visualizer::Visualizer(const Network& network) : _network(&network) {}

void Visualizer::display(sf::RenderTarget& target, sf::Font& font)
{
	const sf::Vector2f global_origin(16.0f, 32.0f);

	const sf::Color inactive_color(255, 50, 0), active_color(0, 255, 50);

	for (const Synapse& synapse : _network->synapses)
	{
		const Neuron& neuron = _network->neurons[synapse.source];

		sf::Vector2f source_origin = global_origin + neuron_offset(_network->neuron_position(synapse.source));
		sf::Vector2f target_origin = global_origin + neuron_offset(_network->neuron_position(synapse.target));

		std::uint8_t alpha = lerp(255u, 64u, std::abs(synapse.properties.weight) * 4.0);

		sf::Color origin_color = lerp_rgb(inactive_color, active_color, neuron.value);
		sf::Color target_color = lerp_rgb(inactive_color, active_color, synapse.properties.weight);
		origin_color.a = target_color.a = alpha;

		std::array<sf::Vertex, 2> vertices{
			sf::Vertex(source_origin, origin_color), sf::Vertex(target_origin, target_color)};
		target.draw(vertices.data(), vertices.size(), sf::Lines);
	}

	for (std::size_t layer_id = 0; layer_id < _network->layers().size(); ++layer_id)
	{
		const auto layer = _network->layers()[layer_id];
		for (std::size_t neuron_id = 0; neuron_id < layer.size(); ++neuron_id)
		{
			const Neuron& neuron = layer[neuron_id];

			sf::Vector2f origin = global_origin + neuron_offset({layer_id, neuron_id});

			// the neuron itself
			sf::CircleShape shape(neuron_radius, 16);

			shape.setPosition(origin - sf::Vector2f(neuron_radius, neuron_radius));
			shape.setFillColor(lerp_rgb(inactive_color, active_color, neuron.value));

			target.draw(shape);

			sf::Text text;
			text.setFont(font);
			text.setCharacterSize(16);
			text.setPosition(origin);
			text.setString(fmt::format("{} (bias: {:.01f})", name(neuron.activation_method), neuron.bias));
			target.draw(text);
		}
	}
}

sf::Vector2f Visualizer::neuron_offset(NeuronPosition pos)
{
	float jitter = 0.0f;

	if (pos.layer == 1)
	{
		std::mt19937                   mt(pos.neuron_in_layer);
		std::uniform_real_distribution dist(-hidden_layer_jitter, hidden_layer_jitter);

		jitter = dist(mt);
	}

	return {
		std::round(float(pos.layer) * layer_padding + jitter),
		std::round(float(pos.neuron_in_layer) / float(_network->layers()[pos.layer].size()) * neuron_area)};
}
