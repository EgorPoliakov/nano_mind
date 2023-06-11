#include <gtest/gtest.h>
#include <parameter.h>

float add(float a, float b) {
    Parameter* param_a = new Parameter(a);
    Parameter* param_b = new Parameter(b);\
    Parameter* out = *param_a + param_b;
    float result = out->data_;

    delete param_a;
    delete param_b;
    delete out;
    return result;
}

float subtract_unary(float a) {
    Parameter* param_a = new Parameter(a);
    Parameter* out = -*param_a;
    float result = out->data_;

    delete param_a;
    delete out;
    return result;
}

float multiply(float a, float b) {
    Parameter* param_a = new Parameter(a);
    Parameter* param_b = new Parameter(b);
    Parameter* out = *param_a * param_b;
    float result = out->data_;

    delete param_a;
    delete param_b;
    delete out;
    return result;
}

float divide(float a, float b) {
    Parameter* param_a = new Parameter(a);
    Parameter* param_b = new Parameter(b);\
    Parameter* out = *param_a / param_b;
    float result = out->data_;

    delete param_a;
    delete param_b;
    delete out;
    return result;
}



TEST(ArithmeticFunctions, Add) {
    EXPECT_EQ(add(5.4f, 3.6f), 9.f);
}

TEST(ArithmeticFunctions, SubtractUnary) {
    EXPECT_EQ(subtract_unary(5.4f), -5.4f);
}

TEST(ArithmeticFunctions, Multiply) {
    EXPECT_EQ(multiply(5.4f, 3.6f), 19.44f);
}

TEST(ArithmeticFunctions, Divide) {
    EXPECT_NEAR(divide(5.4f, 3.6f), 1.5f, 0.001f);
}