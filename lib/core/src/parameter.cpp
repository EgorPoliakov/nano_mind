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

        if (!order[i]->children_.empty() || order[i]->op_ == 'd') {
            delete order[i];
            order[i] = nullptr;
        }
    }
}

Parameter* Parameter::operator+(Parameter* other) {
    if (train_) {
        Parameter* out = new Parameter(data_ + other->data_, {this, other}, '+');
        out->backward_ = [this, other, out]() {
            grad_ += out->grad_;
            other->grad_ += out->grad_;
        };
        return out;
    }

    Parameter* out = new Parameter(data_ + other->data_, {this, other}, '+');
    out->eval();
    return out;
}

Parameter* Parameter::operator-(Parameter* other) {
    if (train_) {
        Parameter* out = new Parameter(data_ - other->data_, {this, other}, '-');
        out->backward_ = [this, other, out]() {
            grad_ += out->grad_;
            other->grad_ += -out->grad_;
        };

        return out;
    }

    Parameter* out = new Parameter(data_ - other->data_, {this, other}, '-');
    out->eval();
    return out;
}

Parameter* Parameter::operator-() {
    if (train_) {
        Parameter* out = new Parameter(-data_, {this}, 'u');
        out->backward_ = [this, out]() {
            grad_ += -out->grad_;
        };
        return out;
    }

    Parameter* out = new Parameter(-data_, {this}, 'u');
    out->eval();
    return out;
}

Parameter* Parameter::operator*(Parameter* other) {
    if (train_) {
        Parameter* out = new Parameter(data_ * other->data_, {this, other}, '*');
        out->backward_ = [this, other, out]() {
            grad_ += out->grad_ * other->data_;
            other->grad_ += out->grad_ * data_;
        };
        
        return out;
    }

    Parameter* out = new Parameter(data_ * other->data_, {this, other}, '*');
    out->eval();
    return out;
}

Parameter* Parameter::operator/(Parameter* other) {
    if (train_) {
        Parameter* out = new Parameter(data_ / other->data_, {this, other}, '/');
        out->backward_ = [this, other, out]() {
            grad_ += out->grad_ * 1 / other->data_;
            other->grad_ += out->grad_ * -data_ / std::pow(other->data_, 2);
        };
        
        return out;
    }

    Parameter* out = new Parameter(data_ / other->data_, {this, other}, '/');
    out->eval();
    return out;
}

Parameter* Parameter::pow(int power) {
    if (train_) {
        Parameter* out = new Parameter(std::pow(data_, power), {this}, 'p');
        float d_power = power * data_;
        out->backward_ = [out, this, d_power]() {
            grad_ = out->grad_ * d_power;
        };
        return out;
    }
    
    Parameter* out = new Parameter(std::pow(data_, power), {this}, 'p');
    out->eval();
    return out;
}

Parameter* Parameter::log() {
    if (train_) {
        Parameter* out = new Parameter(std::log(data_), {this}, 'l');
        float d_log = 1.f / data_;
        out->backward_ = [out, this, d_log]() {
            grad_ = out->grad_ * d_log;
        };
        return out;
    }

    Parameter* out = new Parameter(std::log(data_), {this}, 'l');
    out->eval();
    return out;
}

Parameter* Parameter::tanh() {
    if (train_) {
        float tanh = (std::exp(2*data_) - 1)/(std::exp(2*data_) + 1);
        float d_tanh = 1 - std::pow(tanh, 2);
        Parameter* out = new Parameter(tanh, {this}, 't');
        out->backward_ = [out, this, d_tanh]() {
            grad_ += out->grad_ * d_tanh;
        };

        return out;
    }

    float tanh = (std::exp(2*data_) - 1)/(std::exp(2*data_) + 1);
    Parameter* out = new Parameter(tanh, {this}, 't');
    out->eval();
    return out;
}

Parameter* Parameter::relu() {
    if (train_) {
        float relu = std::max(data_, 0.f);
        float d_relu = relu > 0.f ? 1.f : 0.f;
        Parameter* out = new Parameter(relu, {this}, 'r');
        out->backward_ = [out, this, d_relu]() {
            grad_ += out->grad_ * d_relu;
        };

        return out;
    }

    float relu = std::max(data_, 0.f);
    Parameter* out = new Parameter(relu, {this}, 'r');
    out->eval();
    return out;
}

Parameter* Parameter::sigmoid() {
    if (train_) {
        float sigmoid = 1.f / (1.f + std::exp(-data_));
        float d_sigmoid = sigmoid * (1 - sigmoid);
        Parameter* out = new Parameter(sigmoid, {this}, 's');
        out->backward_ = [out, this, d_sigmoid]() {
            grad_ += out->grad_ * d_sigmoid;
        };
        return out;
    }

    float sigmoid = 1.f / (1.f + std::exp(-data_));
    Parameter* out = new Parameter(sigmoid, {this}, 's');
    out->eval();
    return out;
}

Parameter* Parameter::exp() {
    if (train_) {
        float exp = std::exp(data_);
        float d_exp = exp;
        Parameter* out = new Parameter(exp, {this}, 'e');
        out->backward_ = [out, this, d_exp]() {
            grad_ += out->grad_ * d_exp;
        };
        return out;
    }

    float exp = std::exp(data_);
    Parameter* out = new Parameter(exp, {this}, 'e');
    out->eval();
    return out;
}

void Parameter::train() {
    train_ = true;
}

void Parameter::eval() {
    train_ = false;
}