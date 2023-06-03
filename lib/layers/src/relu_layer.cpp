#include <relu_layer.h>

ReLULayer::ReLULayer(int in_dim) :
    in_dim_(in_dim), layer_idx_(-1) {
    
}

std::vector<Parameter*> ReLULayer::operator()(std::vector<Parameter*> x)  {
    std::vector<Parameter*> outputs;
    outputs.resize(in_dim_);
    for (int i = 0; i < x.size(); i++) {
        outputs[i] = x[i]->relu();
        outputs[i]->label_ = "relu_" + std::to_string(layer_idx_) + "_" + std::to_string(i);
    }
    return outputs;
}

void ReLULayer::set_layer_idx(int idx) {
    layer_idx_ = idx;
}

void ReLULayer::label_parameters() {
    return;
}

std::vector<Parameter*> ReLULayer::parameters() {
    return {};
}