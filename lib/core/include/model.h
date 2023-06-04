#pragma once
#include <vector>
#include <layer.h>
#include <parameter.h>

class Model {
public:
    void add_layer(Layer* layer);
    std::vector<Layer*> layers_;
    std::vector<Parameter*> operator()(std::vector<Parameter*> x);
    std::vector<Parameter*> parameters();
    int layer_idx_;
};