#include <vector>
#include <gtest/gtest.h>
#include <neuron.h>
#include <iostream>

float neuron_forward(float x1, float x2) {
    
    std::vector<Parameter*> x = {new Parameter(x1), new Parameter(x2)};
    Neuron neuron(x.size());

    neuron.weight_[0] = new Parameter(3.f);    
    neuron.weight_[1] = new Parameter(2.f); 
    neuron.bias_ = new Parameter(1.f); 
    
    Parameter* out = neuron(x);
    float result = out->data_;
    
    delete out;
    
    return result;
}

TEST(Neurons, NeuronForward) {
    EXPECT_EQ(neuron_forward(2.3f, 4.3f), 16.5f);
}