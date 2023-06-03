#include <vector>
#include <iostream>

#include <neuron.h>
#include <parameter.h>

Neuron::Neuron(int in_dim) :
    neuron_idx_(-1) {
    weight_.resize(in_dim);
};

Neuron::~Neuron() {
    for (Parameter* parameter : weight_) {
        delete parameter;
        parameter = nullptr;
    }

    delete bias_;
    bias_ = nullptr;
}

Parameter* Neuron::operator()(std::vector<Parameter*> x) {
    Parameter* mult_sum = *x[0] * weight_[0];
    mult_sum->label_ = "neuron_mul_" + std::to_string(layer_idx_) + "_" + std::to_string(neuron_idx_) + "_" + std::to_string(0);
    for (int i = 1; i < x.size(); i++) {
        Parameter* mult = *x[i] * weight_[i];
        mult->label_ = "neuron_mul_" + std::to_string(layer_idx_) + "_" + std::to_string(neuron_idx_) + "_" + std::to_string(i);
        mult_sum = *mult_sum + mult;
        mult_sum->label_ = "neuron_sum_" + std::to_string(layer_idx_) + "_" + std::to_string(neuron_idx_) + "_" + std::to_string(i);
    }

    mult_sum = *mult_sum + bias_;
    mult_sum->label_ = "neuron_out_" + std::to_string(layer_idx_) + "_" + std::to_string(neuron_idx_);
    return mult_sum;
}