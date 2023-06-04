#include <parameter.h>
#include <vector>
#include <topological_sort.h>
#include <math.h>
#include <iostream>

Parameter::Parameter(float data, std::vector<Parameter*> children, char op) : 
    data_(data), children_(children), op_(op) {
    grad_ = 0.0f;
}

Parameter::Parameter() {
    data_ = 0.0f;
    children_ = {};
    op_ = 'n';
    grad_ = 0.0f;
}

Parameter::~Parameter() {
    
}

Parameter::Parameter(const Parameter& other) {
    backward_ = other.backward_;
    data_ = other.data_;
    children_ = other.children_;
    op_ = other.op_;
    grad_ = other.grad_;
}

void Parameter::backward() {
    std::vector<Parameter*> order = TopologicalSort::run(this);
    grad_ = 1.0f;
    for (int i = order.size() - 1; i >= 0; i--) {
        if (order[i]->backward_) {
            order[i]->backward_();
        }

        if (!order[i]->children_.empty()) {
            delete order[i];
            order[i] = nullptr;
        }
    }
}

Parameter* Parameter::operator+(Parameter* other) {
    Parameter* out = new Parameter(data_ + other->data_, {this, other}, '+');
    out->backward_ = [this, other, out]() {
        grad_ += out->grad_;
        other->grad_ += out->grad_;
    };

    return out;
}

Parameter* Parameter::operator-(Parameter* other) {
    Parameter* out = new Parameter(data_ - other->data_, {this, other}, '-');
    out->backward_ = [this, other, out]() {
        grad_ += out->grad_;
        other->grad_ += -out->grad_;
    };

    return out;
}

Parameter* Parameter::operator*(Parameter* other) {
    Parameter* out = new Parameter(data_ * other->data_, {this, other}, '*');
    out->backward_ = [this, other, out]() {
        grad_ += out->grad_ * other->data_;
        other->grad_ += out->grad_ * data_;
    };
    
    return out;
}

Parameter* Parameter::pow(int power) {
    Parameter* out = new Parameter(std::pow(data_, power), {this}, 'p');
    float d_power = power * data_;
    out->backward_ = [out, this, d_power]() {
        grad_ = out->grad_ * d_power;
    };
    return out;
}

Parameter* Parameter::tanh() {
    float tanh = (std::exp(2*data_) - 1)/(std::exp(2*data_) + 1);
    float d_tanh = 1 - std::pow(tanh, 2);
    Parameter* out = new Parameter(tanh, {this}, 't');
    out->backward_ = [out, this, d_tanh]() {
        grad_ += out->grad_ * d_tanh;
    };

    return out;
}

Parameter* Parameter::relu() {
    float relu = std::max(data_, 0.f);
    float d_relu = relu > 0.f ? 1.f : 0.f;
    Parameter* out = new Parameter(relu, {this}, 'r');
    out->backward_ = [out, this, d_relu]() {
        grad_ += out->grad_ * d_relu;
    };

    return out;
}

Parameter* Parameter::sigmoid() {
    float sigmoid = 1.f / (1.f + std::exp(-data_));
    float d_sigmoid = sigmoid * (1 - sigmoid);
    Parameter* out = new Parameter(sigmoid, {this}, 's');
    out->backward_ = [out, this, d_sigmoid]() {
        grad_ += out->grad_ * d_sigmoid;
    };
    return out;
}