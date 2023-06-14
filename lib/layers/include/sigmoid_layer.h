#pragma once
#include <vector>
#include <parameter.h>
#include <layer.h>

class SigmoidLayer : public Layer {
public:
    SigmoidLayer(int in_dim);
    std::vector<Parameter*> operator()(std::vector<Parameter*> x) override;
    void set_layer_idx(int idx) override;
    void label_parameters() override;
    std::vector<Parameter*> parameters() override;
    void train() override {};
    void eval() override {};
    int in_dim_;
    int layer_idx_;
};