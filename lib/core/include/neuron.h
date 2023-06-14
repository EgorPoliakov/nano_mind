#pragma once
#include <parameter.h>
#include <vector>


class Neuron {
public:
    Neuron(int in_dim);
    ~Neuron();
    std::vector<Parameter*> weight_;
    Parameter* bias_;
    Parameter* operator()(std::vector<Parameter*> x);
    void train();
    void eval();
    int neuron_idx_;
    int layer_idx_;

private:
};