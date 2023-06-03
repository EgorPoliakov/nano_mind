#pragma once
#include <parameter.h>


class MSELoss {
public:
    Parameter* operator()(Parameter* prediction, Parameter* label);
    Parameter* subtraction_result_;
    Parameter* square_result_;
};