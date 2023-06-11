#include <gtest/gtest.h>
#include <parameter.h>
#include <cross_entropy_loss.h>
#include <vector>
#include <iostream>

float cross_entropy(std::vector<float> logits, float label) {
    std::vector<Parameter*> param_logits(logits.size());
    Parameter* param_label = new Parameter(label);

    for (int i = 0; i < logits.size(); i++) {
        param_logits[i] = new Parameter(logits[i]);
    }
    CrossEntropyLoss criterion(logits.size());
    Parameter* loss = criterion(param_logits, param_label);
    float result = loss->data_;

    loss->backward();
    loss = nullptr;

    delete param_label;
    param_label = nullptr;

    for (Parameter* parameter : param_logits) {
        delete parameter;
        parameter = nullptr;
    }

    return result;
}

TEST(LossFunctions, CrossEntropy) {
    EXPECT_NEAR(cross_entropy({1.4f, -4.5f, 0.3f}, 2), 1.389f, 0.001);
}