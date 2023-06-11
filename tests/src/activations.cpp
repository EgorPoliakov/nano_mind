#include <vector>
#include <gtest/gtest.h>
#include <parameter.h>
#include <softmax.h>

float sigmoid(float x) {
    Parameter* param = new Parameter(x);
    Parameter* out = param->sigmoid();
    float result = out->data_;

    delete param;
    delete out;
    return result;
}

float relu(float x) {
    Parameter* param = new Parameter(x);
    Parameter* out = param->relu();
    float result = out->data_;

    delete param;
    delete out;
    return result; 
}

float exp(float x) {
    Parameter* param = new Parameter(x);
    Parameter* out = param->exp();
    float result = out->data_;

    delete param;
    delete out;
    return result;
}

float log(float x) {
    Parameter* param = new Parameter(x);
    Parameter* out = param->log();
    float result = out->data_;

    delete param;
    delete out;
    return result;
}

std::vector<float> softmax(std::vector<float> x) {
    std::vector<Parameter*> input(x.size());
    for (int i = 0; i < x.size(); i++) {
        input[i] = new Parameter(x[i]);
    }

    SoftmaxLayer soft(x.size());
    std::vector<Parameter*> out = soft(input);
    std::vector<float> result;
    for (Parameter* parameter : out) {
        result.push_back(parameter->data_);
    }

    out[0]->backward();
    out[0] = nullptr;
    for (int i = 1; i < out.size(); i++) {
        delete out[i];
        out[i] = nullptr;
    }

    for (int i = 0; i < input.size(); i++) {
        delete input[i];
        input[i] = nullptr;
    }

    return result;
}


TEST(ActivationFunctions, Sigmoid) {
    EXPECT_NEAR(sigmoid(1.f), 0.731, 0.001);
}

TEST(ActivationFunctions, ReLUPositive) {
    EXPECT_EQ(relu(34.f), 34);
}

TEST(ActivationFunctions, ReLUNegative) {
    EXPECT_EQ(relu(-34.f), 0);
}

TEST(ActivationFunctions, Softmax) {
    std::vector<float> result = softmax({2.3f, 4.5f, 4.3f});
    std::vector<float> correct_result = {0.0574, 0.5183, 0.4243};
    for (int i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], correct_result[i], 0.001f);
    }
}

TEST(ActivationFunctions, Exp) {
    EXPECT_NEAR(exp(4.f), 54.598, 0.001);
}

TEST(ActivationFunctions, Log) {
    EXPECT_NEAR(log(4.f), 1.386, 0.001);
}