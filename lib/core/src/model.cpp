#include <model.h>
#include <layer.h>
#include <iostream>


Model::~Model() {
    for (int i = 0; i < layers_.size(); i++) {
        delete layers_[i];
        layers_[i] = nullptr;
    }
}

void Model::add_layer(Layer* layer) {
    layer->set_layer_idx(layer_idx_);
    layer->label_parameters();
    layer_idx_++;
    layers_.push_back(layer);
} 

std::vector<Parameter*> Model::operator()(std::vector<Parameter*> x) {
    std::vector<Parameter*> output = x;
    if (!train_) {
        for (Parameter* parameter : output) {
            parameter->eval();
        }
    }

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

void Model::train() {
    for (Layer* layer : layers_) {
        layer->train();
    }
}

void Model::eval() {
    for (Layer* layer : layers_) {
        layer->eval();
    }
}