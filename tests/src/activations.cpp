#include <gtest/gtest.h>
#include <parameter.h>

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

TEST(ActivationFunctions, Sigmoid) {
    EXPECT_NEAR(sigmoid(1.f), 0.731, 0.001);
}

TEST(ActivationFunctions, ReLUPositive) {
    EXPECT_EQ(relu(34.f), 34);
}

TEST(ActivationFunctions, ReLUNegative) {
    EXPECT_EQ(relu(-34.f), 0);
}
