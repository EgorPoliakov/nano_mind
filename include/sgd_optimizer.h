#pragma once
#include <vector>
#include <parameter.h>

class SGDOptimizer {
public:
    SGDOptimizer(std::vector<Parameter*> parameters, float learning_rate);
    void step();
    void zero_grad();
    std::vector<Parameter*> parameters_;
    float learning_rate_;
};