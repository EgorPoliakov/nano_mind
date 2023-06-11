#pragma once
#include <vector>
#include <parameter.h>
#include <softmax.h>

class CrossEntropyLoss {
public:
    CrossEntropyLoss(int in_dim);
    Parameter* operator()(std::vector<Parameter*> logits, Parameter* label);
    SoftmaxLayer softmax;

};