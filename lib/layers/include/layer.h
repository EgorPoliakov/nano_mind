#pragma once
#include <vector>
#include <parameter.h>


class Layer {
public:
    virtual ~Layer() {};
    virtual std::vector<Parameter*> operator()(std::vector<Parameter*> x) = 0;
    virtual void set_layer_idx(int idx) = 0;
    virtual void label_parameters() = 0;
    virtual std::vector<Parameter*> parameters() = 0;
    virtual void train() = 0;
    virtual void eval() = 0;
};