#include <model.h>
#include <layer.h>
#include <iostream>

void Model::add_layer(Layer* layer) {
    layer->set_layer_idx(layer_idx_);
    layer->label_parameters();
    layer_idx_++;
    layers_.push_back(layer);
} 

std::vector<Parameter*> Model::operator()(std::vector<Parameter*> x) {
    std::vector<Parameter*> output = x;
    for (Layer* layer : layers_) {
        output = (*layer)(output);
    }
    return output;
}

std::vector<Parameter*> Model::parameters() {
    std::vector<Parameter*> parameters;
    for (Layer* layer : layers_) {
        std::vector<Parameter*> layer_parameters = layer->parameters();
        parameters.insert(parameters.end(), layer_parameters.begin(), layer_parameters.end());
    }
    return parameters;
}