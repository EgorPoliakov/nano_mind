#include <mse_loss.h>
#include <parameter.h>

Parameter* MSELoss::operator()(Parameter* prediction, Parameter* label) {
    subtraction_result_ = *prediction - label;
    subtraction_result_->label_ = "mse_subtract";
    square_result_ = subtraction_result_->pow(2);
    square_result_->label_ = "mse_square";
    return square_result_;
}