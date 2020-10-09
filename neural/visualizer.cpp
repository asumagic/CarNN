#include "visualizer.hpp"

#include "../maths.hpp"
#include "network.hpp"
#include <fmt/core.h>
#include <random>

constexpr float layer_padding       = 256.0f;
constexpr float neuron_area         = 1200.0f;
constexpr float hidden_layer_jitter = 160.0f;

constexpr float neuron_radius = 12.0f;

Visualizer::Visualizer(const Network& network) : _network(&network) {}

void Visualizer::display(sf::RenderTarget& target, sf::Font& font)
{
	const sf::Vector2f global_origin(16.0f, 128.0f);

	const sf::Color inactive_color(255, 50, 0), active_color(0, 255, 50);

	for (std::size_t layer_id = 0; layer_id < _network->layers.size(); ++layer_id)
	{
		const Layer& layer = _network->layers[layer_id];
		for (std::size_t neuron_id = 0; neuron_id < layer.neurons.size(); ++neuron_id)
		{
			const Neuron& neuron = layer.neurons[neuron_id];

			sf::Vector2f origin = global_origin + neuron_offset({layer_id, neuron_id});

			// synapses
			for (std::size_t synapse_id = 0; synapse_id < neuron.synapses.size(); ++synapse_id)
			{
				const Synapse& synapse = neuron.synapses[synapse_id];

				sf::Vector2f target_origin = global_origin + neuron_offset(synapse.target);

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

			// label
			std::string label;
			if (!neuron.label.empty())
			{
				label += fmt::format("{}\n", neuron.label);
			}

			const std::array<const char*, 3> activation_method_names{"sigmoid", "leaky RELU", "sin"};

			label += fmt::format(
				"{}\nbias: {:.01f}", activation_method_names[int(neuron.activation_method)], neuron.bias);

			sf::Text text;
			text.setFont(font);
			text.setCharacterSize(12);
			text.setPosition(origin);
			text.setString(label);
			target.draw(text);
		}
	}
}

sf::Vector2f Visualizer::neuron_offset(NeuronId id)
{
	float jitter = 0.0f;

	if (id.layer() == 1)
	{
		std::mt19937                   mt(1337 + id.neuron_in_layer());
		std::uniform_real_distribution dist(-hidden_layer_jitter, hidden_layer_jitter);

		jitter = dist(mt);
	}

	return {
		float(id.layer()) * layer_padding + jitter,
		float(id.neuron_in_layer()) / float(_network->layers[id.layer()].neurons.size()) * neuron_area};
}
