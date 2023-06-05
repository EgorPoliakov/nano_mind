#pragma once
#include <layer.h>

class SoftmaxLayer : public Layer {
public:
    SoftmaxLayer(int in_dim);
    std::vector<Parameter*> operator()(std::vector<Parameter*> x) override;
    void set_layer_idx(int idx) override;
    void label_parameters() override;
    std::vector<Parameter*> parameters() override;
    int in_dim_;
    int layer_idx_;
};