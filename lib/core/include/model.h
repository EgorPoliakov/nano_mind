#pragma once
#include <vector>
#include <layer.h>
#include <parameter.h>

class Model {
public:
    ~Model();
    void add_layer(Layer* layer);
    std::vector<Layer*> layers_;
    std::vector<Parameter*> operator()(std::vector<Parameter*> x);
    std::vector<Parameter*> parameters();
    void train();
    void eval();
    int layer_idx_;
    bool train_ = true;
};