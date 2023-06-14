#include <iostream>
#include <parameter.h>
#include <neuron.h>
#include <linear_layer.h>
#include <mse_loss.h>
#include <sgd_optimizer.h>
#include <relu_layer.h>
#include <model.h>
#include <cross_entropy_loss.h>
#include <mnist_reader.hpp>
#include <utils.h>



int main(int, char**) {
    int n_train = 1000;
    int n_test = 100;

    auto mnist_dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../mnist_dataset");
    int image_size = mnist_dataset.test_images[0].size();

    std::vector<std::vector<uint8_t>> dataset = {mnist_dataset.test_images.begin(), mnist_dataset.test_images.begin() + n_train};
    std::vector<std::vector<uint8_t>> test_dataset = {mnist_dataset.test_images.begin() + n_train, mnist_dataset.test_images.begin() + n_train + n_test};
    std::vector<uint8_t> labels = {mnist_dataset.test_labels.begin(), mnist_dataset.test_labels.begin() + n_train};
    std::vector<uint8_t> test_labels = {mnist_dataset.test_labels.begin() + n_train, mnist_dataset.test_labels.begin() + n_train + n_test};

    std::cout << "Total Training samples: " << dataset.size() << std::endl;
    std::cout << "Total Testing samples: " << test_dataset.size() << std::endl;
    
    int num_classes = 10;
    LinearLayer l1(image_size, 16);
    ReLULayer relu_layer1(16);
    LinearLayer l2(16, 32);
    ReLULayer relu_layer2(32);
    LinearLayer l3(32, 16);
    ReLULayer relu_layer3(16);
    LinearLayer l4(16, num_classes);

    std::vector<Layer*> layers = {
        &l1, 
        &relu_layer1, 
        &l2,
        &relu_layer2,
        &l3,
        &relu_layer3,
        &l4 
    };

    Model model;
    for (Layer* layer: layers) {
        model.add_layer(layer);
    }

    CrossEntropyLoss criterion(num_classes);
    
    std::vector<Parameter*> params = model.parameters();

    std::cout << "Overall param: " << params.size() << std::endl;
    SGDOptimizer optimizer(params, 0.01);

    int iterations = 30;
    for (int iter = 0; iter < iterations; iter++) {
        float total_loss = 0.f;
        float epoch_accuracy = 0.f;
        int correct_predictions = 0;

        model.train();
        for (int sample = 0; sample < dataset.size(); sample++) {
            std::vector<Parameter> input(image_size);
            std::vector<Parameter*> input_pointers(image_size);
            Parameter label = Parameter((float)labels[sample]);
            for (int i = 0; i < input.size(); i++) {
                input[i] = Parameter((float)dataset[sample][i] / 255.f);
                input_pointers[i] = &input[i];
            }

            std::vector<Parameter*> model_out = model(input_pointers);
            Parameter* loss = criterion(model_out, &label);

            loss->backward();            
            optimizer.step();
            optimizer.zero_grad();
        }

        model.eval();
        for (int sample = 0; sample < test_dataset.size(); sample++) {
            int image_size = test_dataset[sample].size();
            std::vector<Parameter> input(image_size);
            std::vector<Parameter*> input_pointers(image_size);
            Parameter label = Parameter((float)test_labels[sample]);
            for (int i = 0; i < input.size(); i++) {
                input[i] = Parameter((float)test_dataset[sample][i] / 255.f);
                input_pointers[i] = &input[i];
            }
            std::vector<Parameter*> model_out = model(input_pointers);
            Parameter* loss = criterion(model_out, &label);
            int predicted_label = nano_mind::argmax(model_out).second;

            if (predicted_label == label.data_) {
                correct_predictions++;
            }

            total_loss += loss->data_;
            loss->backward();            
        }

        epoch_accuracy += (float)correct_predictions / (float)test_dataset.size();
        std::cout << "Epoch accuracy: " << epoch_accuracy << std::endl;
        std::cout << "Epoch loss: " << total_loss / dataset.size() << std::endl;
    }
}
