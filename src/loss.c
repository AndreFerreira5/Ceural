#include "loss.h"

double mean_squared_error_loss(const size_t output_size, const double *network_outputs, const double *expected_output){
    double sum = 0;
    for(size_t output_neuron=0; output_neuron<output_size; ++output_neuron){
        sum += pow((network_outputs[output_neuron] - expected_output[output_neuron]), 2);
    }
    return (1/(double)output_size)*sum;
}


double multi_class_cross_entropy_loss(const size_t output_size, const double *network_outputs, const double *expected_output){
    double sum = 0;
    for(size_t output_neuron=0; output_neuron<output_size; ++output_neuron){
        sum += expected_output[output_neuron] * log(network_outputs[output_neuron]);
    }
    return -sum;
}


double binary_cross_entropy_loss(const size_t output_size, const double *network_outputs, const double *expected_output){
    double correct_class = -1;
    if(expected_output[0] == 1) correct_class = 0;
    if(expected_output[1] == 1) correct_class = 1;
    return -(correct_class*log(network_outputs[1]) + (1-correct_class)*log(1-network_outputs[1]));
}
