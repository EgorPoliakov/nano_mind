#include <sigmoid_layer.h>

SigmoidLayer::SigmoidLayer(int in_dim) :
    in_dim_(in_dim), layer_idx_(-1) {
    
}

std::vector<Parameter*> SigmoidLayer::operator()(std::vector<Parameter*> x)  {
    std::vector<Parameter*> outputs;
    outputs.resize(in_dim_);
    for (int i = 0; i < x.size(); i++) {
        outputs[i] = x[i]->sigmoid();
        outputs[i]->label_ = "sigmoids_" + std::to_string(layer_idx_) + "_" + std::to_string(i);
    }
    return outputs;
}

void SigmoidLayer::set_layer_idx(int idx) {
    layer_idx_ = idx;
}

void SigmoidLayer::label_parameters() {
    return;
}

std::vector<Parameter*> SigmoidLayer::parameters() {
    return {};
}