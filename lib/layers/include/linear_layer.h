#pragma once
#include <vector>
#include <neuron.h>
#include <parameter.h>
#include <layer.h>

class LinearLayer : public Layer {
public:
    LinearLayer(int in_dim, int out_dim);
    ~LinearLayer();
    std::vector<Parameter*> parameters() override;
    std::vector<Parameter*> operator()(std::vector<Parameter*> x) override;
    std::vector<Neuron*> neurons_;
    void label_parameters() override;
    void set_layer_idx(int idx) override;
    void train() override;
    void eval() override;
    int layer_idx_;
private:
    
};