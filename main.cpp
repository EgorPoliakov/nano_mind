#include <iostream>
#include <parameter.h>
#include <neuron.h>
#include <linear_layer.h>
#include <mse_loss.h>
#include <sgd_optimizer.h>
#include <relu_layer.h>
#include <model.h>

void print_matrix(const std::vector<std::vector<Parameter>>& matrix) {
    for (const std::vector<Parameter>& row : matrix) {
        for (Parameter param : row) {
            std::cout << param.data_ << " ";
        }
        std::cout << "\n";
    }
}

int main(int, char**) {
    std::vector<std::vector<Parameter*>> dataset = {
        {new Parameter(2)},
        {new Parameter(3)},
        {new Parameter(4)}
    };

    dataset[0][0]->label_ = "input";

    std::vector<Parameter*> labels = {
        new Parameter(4),
        new Parameter(9),
        new Parameter(16)
    };

    labels[0]->label_ = "label";

    MSELoss criterion;

    Model model;
    LinearLayer l1(1, 12);
    ReLULayer relu_layer(12);
    LinearLayer l2(12, 1);

    model.add_layer(&l1);
    model.add_layer(&relu_layer);
    model.add_layer(&l2);

    
    std::vector<Parameter*> params = model.parameters();

    std::cout << "Overall param: " << params.size() << std::endl;
    SGDOptimizer optimizer(params, 0.01);
    int iterations = 5000;
    for (int i = 0; i < iterations; i++) {
        for (int sample = 0; sample < dataset.size(); sample++) {
            std::vector<Parameter*> model_out = model(dataset[sample]);
            Parameter* loss = criterion(model_out[0], labels[sample]);
            std::cout << loss->data_ << std::endl;
            loss->backward();            
            optimizer.step();
            optimizer.zero_grad();
        }
    }

    for (Parameter* param : dataset[0]) {
        delete param;
    }

    delete labels[0];

    std::cout << "end training" << std::endl;
}
