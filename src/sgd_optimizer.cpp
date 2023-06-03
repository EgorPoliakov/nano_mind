#include <sgd_optimizer.h>
#include <parameter.h>

SGDOptimizer::SGDOptimizer(std::vector<Parameter*> parameters, float learning_rate) :
    parameters_(parameters), learning_rate_(learning_rate) {
}

void SGDOptimizer::step() {
    for (Parameter* parameter : parameters_) {
        parameter->data_ -= learning_rate_ * parameter->grad_;
    }
}

void SGDOptimizer::zero_grad() {
    for (Parameter* parameter : parameters_) {
        parameter->grad_ = 0.f;
    }
}