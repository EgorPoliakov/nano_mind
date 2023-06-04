#include <gtest/gtest.h>
#include <parameter.h>
#include <iostream>

float sigmoid_derivative(float x) {
    Parameter* param = new Parameter(x);
    Parameter* out = param->sigmoid();
    out->backward();

    float result = param->grad_;

    delete param;
    return result;
}

float relu_derivative(float x) {
    Parameter* param = new Parameter(x);
    Parameter* out = param->relu();
    out->backward();

    float result = param->grad_;

    delete param;
    return result;
}

std::pair<float, float> add_derivative(float a, float b) {
    Parameter* param_a = new Parameter(a);
    Parameter* param_b = new Parameter(b);
    Parameter* out = *param_a + param_b;
    out->backward();

    std::pair<float, float> result = {param_a->grad_, param_b->grad_};

    delete param_a;
    delete param_b;
    return result;
}

std::pair<float, float> subtract_derivative(float a, float b) {
    Parameter* param_a = new Parameter(a);
    Parameter* param_b = new Parameter(b);
    Parameter* out = *param_a - param_b;
    out->backward();

    std::pair<float, float> result = {param_a->grad_, param_b->grad_};

    delete param_a;
    delete param_b;
    return result;
}

std::pair<float, float> multiply_derivative(float a, float b) {
    Parameter* param_a = new Parameter(a);
    Parameter* param_b = new Parameter(b);
    Parameter* out = *param_a * param_b;
    out->backward();

    std::pair<float, float> result = {param_a->grad_, param_b->grad_};

    delete param_a;
    delete param_b;
    return result;
}

TEST(Derivatives, Sigmoid) {
    EXPECT_NEAR(sigmoid_derivative(1.f), 0.196, 0.001);
}

TEST(Derivatives, ReLUPositive) {
    EXPECT_EQ(relu_derivative(34.f), 1);
}

TEST(Derivatives, ReLUNegative) {
    EXPECT_EQ(relu_derivative(-34.f), 0);
}

TEST(Derivatives, Add) {
    std::pair<float, float> result = add_derivative(23.f, 42.f);
    EXPECT_EQ(result.first, 1);
    EXPECT_EQ(result.second, 1);
}

TEST(Derivatives, Subtract) {
    std::pair<float, float> result = subtract_derivative(23.f, 42.f);
    EXPECT_EQ(result.first, 1);
    EXPECT_EQ(result.second, -1);
}

TEST(Derivatives, Multiply) {
    std::pair<float, float> result = multiply_derivative(23.f, 42.f);
    EXPECT_EQ(result.first, 42);
    EXPECT_EQ(result.second, 23);
}
