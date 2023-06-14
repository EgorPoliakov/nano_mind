#include <gtest/gtest.h>
#include <parameter.h>
#include <softmax.h>
#include <cross_entropy_loss.h>
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

float exp_derivative(float x) {
    Parameter* param = new Parameter(x);
    Parameter* out = param->exp();
    out->backward();

    float result = param->grad_;

    delete param;
    return result;
}

float log_derivative(float x) {
    Parameter* param = new Parameter(x);
    Parameter* out = param->log();
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

float subtract_unary_derivative(float a) {
    Parameter* param_a = new Parameter(a);
    Parameter* out = -*param_a;
    out->backward();
    float result = param_a->grad_;

    delete param_a;
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

std::pair<float, float> divide_derivative(float a, float b) {
    Parameter* param_a = new Parameter(a);
    Parameter* param_b = new Parameter(b);
    Parameter* out = *param_a / param_b;
    out->backward();

    std::pair<float, float> result = {param_a->grad_, param_b->grad_};

    delete param_a;
    delete param_b;
    return result;
}

std::vector<float> softmax_derivative(std::vector<float> x) {
    std::vector<Parameter*> input(x.size());
    for (int i = 0; i < x.size(); i++) {
        input[i] = new Parameter(x[i]);
    }

    SoftmaxLayer soft(x.size());
    std::vector<Parameter*> out = soft(input);
    std::vector<float> result;

    out[1]->backward();
    out[1] = nullptr;
    
    delete out[0];
    delete out[2];
    out[0] = nullptr;
    out[2] = nullptr;

    for (Parameter* parameter : input) {
        result.push_back(parameter->grad_);
    }

    for (int i = 0; i < input.size(); i++) {
        delete input[i];
        input[i] = nullptr;
    }

    return result;
}

std::vector<float> cross_entropy_derivative(std::vector<float> x, float label) {
    std::vector<Parameter*> input(x.size());
    Parameter* param_label = new Parameter(label);

    for (int i = 0; i < x.size(); i++) {
        input[i] = new Parameter(x[i]);
    }

    CrossEntropyLoss criterion(x.size());
    Parameter* loss = criterion(input, param_label);
    std::vector<float> result;

    loss->backward();
    loss = nullptr;
    
    delete param_label;
    param_label = nullptr;

    for (Parameter* parameter : input) {
        result.push_back(parameter->grad_);
    }

    for (int i = 0; i < input.size(); i++) {
        delete input[i];
        input[i] = nullptr;
    }

    return result;
}

std::pair<float, float> eval_mode_multiply_derivative(float a, float b) {
    Parameter* param_a = new Parameter(a);
    Parameter* param_b = new Parameter(b);
    param_a->eval();
    param_b->eval();
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

TEST(Derivatives, Softmax) {
    std::vector<float> result = softmax_derivative({2.3f, 4.5f, 4.3f});
    std::vector<float> correct_result = {-0.0298,  0.2497, -0.2199};
    for (int i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], correct_result[i], 0.001f);
    }
}

TEST(Derivatives, CrossEntropy) {
    std::vector<float> result = cross_entropy_derivative({1.4f, -4.5f, 0.3f}, 2);
    std::vector<float> correct_result = {0.7487,  0.0021, -0.7508};
    for (int i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], correct_result[i], 0.0001f);
    }
}

TEST(Derivatives, Exp) {
    EXPECT_NEAR(exp_derivative(4.f), 54.598, 0.001);
}

TEST(Derivatives, Log) {
    EXPECT_NEAR(log_derivative(4.f), 0.25, 0.001);
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

TEST(Derivatives, SubtractUnary) {
    float result = subtract_unary_derivative(23.f);
    EXPECT_EQ(result, -1.f);
}

TEST(Derivatives, Multiply) {
    std::pair<float, float> result = multiply_derivative(23.f, 42.f);
    EXPECT_EQ(result.first, 42);
    EXPECT_EQ(result.second, 23);
}

TEST(Derivatives, Divide) {
    std::pair<float, float> result = divide_derivative(23.f, 42.f);
    EXPECT_NEAR(result.first, 0.023f, 0.001f);
    EXPECT_NEAR(result.second, -0.013f, 0.001f);
}

TEST(Derivatives, EvalModeMultiply) {
    std::pair<float, float> result = eval_mode_multiply_derivative(23.f, 42.f);
    EXPECT_EQ(result.first, 0);
    EXPECT_EQ(result.second, 0);
}
