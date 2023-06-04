#pragma once
#include <vector>
#include <string>
#include <functional>
#include <unordered_set>
class Parameter {
public:
    Parameter(float data, std::vector<Parameter*> children={}, char op='n');
    Parameter();
    ~Parameter();

    Parameter(const Parameter& other);

    void backward();

    Parameter* operator+(Parameter* other);
    // Parameter operator+(float other);

    Parameter* operator*(Parameter* other);
    // Parameter operator*(float other);

    Parameter* operator-(Parameter* other);

    Parameter* tanh();
    Parameter* relu();
    Parameter* sigmoid();
    Parameter* pow(int power);
    std::function<void()> backward_;
    float grad_;
    float data_;
    std::vector<Parameter*> children_;
    std::string label_;
private:
    char op_;
    
};