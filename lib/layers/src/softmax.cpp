#include <softmax.h>

SoftmaxLayer::SoftmaxLayer(int in_dim) :
    in_dim_(in_dim), layer_idx_(-1) {
    
}

std::vector<Parameter*> SoftmaxLayer::operator()(std::vector<Parameter*> x)  {
    std::vector<Parameter*> outputs;
    std::vector<Parameter*> exponents;
    Parameter* sum = x[0]->exp();
    exponents.push_back(sum);
    sum->label_ = "softmax_exp_" + std::to_string(layer_idx_) + "_" + std::to_string(0);
    outputs.resize(in_dim_);
    for (int i = 1; i < x.size(); i++) {
        Parameter* exp = x[i]->exp();
        exponents.push_back(exp);
        exp->label_ = "softmax_exp_" + std::to_string(layer_idx_) + "_" + std::to_string(i);
        sum = *sum + x[i]->exp();
        sum->label_ = "softmax_sum_" + std::to_string(layer_idx_) + "_" + std::to_string(i - 1);
    }

    for (int i = 0; i < x.size(); i++) {
        outputs[i] = *exponents[i] / sum;
        outputs[i]->label_ = "softmax_out" + std::to_string(layer_idx_) + "_" + std::to_string(i - 1);
    }

    return outputs;
}

void SoftmaxLayer::set_layer_idx(int idx) {
    layer_idx_ = idx;
}

void SoftmaxLayer::label_parameters() {
    return;
}

std::vector<Parameter*> SoftmaxLayer::parameters() {
    return {};
}