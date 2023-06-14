#include <linear_layer.h>
#include <parameter.h>
#include <random>
#include <math.h>
#include <iostream>

LinearLayer::LinearLayer(int in_dim, int out_dim) :
    layer_idx_(-1) {
    neurons_.reserve(out_dim);
    // std::random_device rd;
    std::mt19937 gen(1);
    float k = 1.f / (float)in_dim;
    std::uniform_real_distribution<float> distribution(-std::sqrt(k), std::sqrt(k));
    
    for (int i = 0; i < out_dim; i++) {
        Neuron* neuron = new Neuron(in_dim);
        for (int i = 0; i < in_dim; i++) {
            neuron->weight_[i] = new Parameter(distribution(gen));
        }
        
        neuron->bias_ = new Parameter(distribution(gen));
        neurons_.push_back(neuron);
    }
}

void LinearLayer::label_parameters() {
    for (int neuron_idx = 0; neuron_idx < neurons_.size(); neuron_idx++) {
        Neuron* neuron = neurons_[neuron_idx];
        for (int weight_idx = 0; weight_idx < neuron->weight_.size(); weight_idx++) {
            neuron->weight_[weight_idx]->label_ = "weight_" + std::to_string(layer_idx_) + "_" + std::to_string(neuron_idx) + "_" + std::to_string(weight_idx);
        }
        neuron->bias_->label_ = "bias_" + std::to_string(layer_idx_) + "_" + std::to_string(neuron_idx);
        neuron->layer_idx_ = layer_idx_;
    }
}

void LinearLayer::set_layer_idx(int idx) {
    layer_idx_ = idx;
}

LinearLayer::~LinearLayer() {
    for (int i = 0; i < neurons_.size(); i++) {
        delete neurons_[i];
        neurons_[i] = nullptr;
    }
}

std::vector<Parameter*> LinearLayer::operator()(std::vector<Parameter*> x) {
    std::vector<Parameter*> output(neurons_.size());
    for (int i = 0; i < neurons_.size(); i++) {
        output[i] = (*neurons_[i])(x);
    }
    return output;
}

std::vector<Parameter*> LinearLayer::parameters() {
    std::vector<Parameter*> layer_parameters;
    for (Neuron* neuron : neurons_) {
        for (Parameter* parameter : neuron->weight_) {
            layer_parameters.push_back(parameter);
        }
        layer_parameters.push_back(neuron->bias_);
    }
    return layer_parameters;
}

void LinearLayer::train() {
    for (Neuron* neuron : neurons_) {
        neuron->train();
    }
}

void LinearLayer::eval() {
    for (Neuron* neuron : neurons_) {
        neuron->eval();
    }
}