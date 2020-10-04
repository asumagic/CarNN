#pragma once

#include <SFML/Graphics.hpp>

#include "neuronid.hpp"

class Network;

class Visualizer
{
public:
    Visualizer(const Network& network);

    void display(sf::RenderTarget& target, sf::Font& font);

    private:
    sf::Vector2f neuron_offset(NeuronId id);

    const Network* _network;
};
