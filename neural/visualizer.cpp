#include "visualizer.hpp"

#include "../maths.hpp"
#include "network.hpp"
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

	for (std::size_t layer_id = 0; layer_id < _network->layers().size(); ++layer_id)
	{
		const auto layer = _network->layers()[layer_id];
		for (std::size_t neuron_id = 0; neuron_id < layer.size(); ++neuron_id)
		{
			const Neuron& neuron = layer[neuron_id];

			sf::Vector2f origin = global_origin + neuron_offset({layer_id, neuron_id});

			// synapses
			for (std::size_t synapse_id = 0; synapse_id < neuron.synapses.size(); ++synapse_id)
			{
				const Synapse& synapse = _network->synapses[neuron.synapses[synapse_id]];

				sf::Vector2f target_origin = global_origin + neuron_offset(_network->neuron_position(synapse.target));

				std::uint8_t alpha = lerp(255u, 64u, std::abs(synapse.weight) * 4.0);

				sf::Color origin_color(
					lerp(inactive_color.r, active_color.r, neuron.value),
					lerp(inactive_color.g, active_color.g, neuron.value),
					lerp(inactive_color.b, active_color.b, neuron.value),
					alpha);
				sf::Color target_color(
					lerp(inactive_color.r, active_color.r, neuron.value * synapse.weight),
					lerp(inactive_color.g, active_color.g, neuron.value * synapse.weight),
					lerp(inactive_color.b, active_color.b, neuron.value * synapse.weight),
					alpha);

				std::array<sf::Vertex, 2> vertices{
					sf::Vertex(origin, origin_color), sf::Vertex(target_origin, target_color)};
				target.draw(vertices.data(), vertices.size(), sf::Lines);
			}

			// the neuron itself
			sf::CircleShape shape(neuron_radius, 16);

			shape.setPosition(origin - sf::Vector2f(neuron_radius, neuron_radius));
			shape.setFillColor(sf::Color(
				lerp(inactive_color.r, active_color.r, neuron.value),
				lerp(inactive_color.g, active_color.g, neuron.value),
				lerp(inactive_color.b, active_color.b, neuron.value)));

			target.draw(shape);

			sf::Text text;
			text.setFont(font);
			text.setCharacterSize(16);
			text.setPosition(origin);
			text.setString(fmt::format(
				"{}\nbias: {:.01f}\nevoid: {}", name(neuron.activation_method), neuron.bias, neuron.evolution_id));
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
