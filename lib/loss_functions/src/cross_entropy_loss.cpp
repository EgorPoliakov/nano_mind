#include <cross_entropy_loss.h>
#include <vector>
#include <softmax.h>
#include <iostream>

CrossEntropyLoss::CrossEntropyLoss(int in_dim) :
    softmax(in_dim) {

}

Parameter* CrossEntropyLoss::operator()(std::vector<Parameter*> logits, Parameter* label) {
    float max_logit = logits[0]->data_;

    for (Parameter* parameter : logits) {
        max_logit = std::max(max_logit, parameter->data_);
    }

    Parameter* max_logit_parameter = new Parameter(max_logit, {}, 'd');
    std::vector<Parameter*> normalized_logits(logits.size());
    for (int i = 0; i < logits.size(); i++) {
        normalized_logits[i] = *logits[i] - max_logit_parameter;
    }

    std::vector<Parameter*> probabilities = softmax(normalized_logits);
    std::vector<Parameter*> log_probabilities(probabilities.size());

    for (int i = 0; i < probabilities.size(); i++) {
        log_probabilities[i] = probabilities[i]->log();
    }
    
    int class_idx = (int)label->data_;
    Parameter* loss = -*log_probabilities[class_idx];

    for (int i = 0; i < log_probabilities.size(); i++) {
        if (i != class_idx) {
            delete log_probabilities[i];
            delete probabilities[i];
        }
    }
    return loss;
}