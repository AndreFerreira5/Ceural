#include "loss.h"


#define EPSILON 1e-10

double safe_log(double x) {
    return log(x < EPSILON ? EPSILON : x);
}


double mean_squared_error_loss(const size_t output_size, const double *network_output, const double *expected_output){
    double sum = 0;
    for(size_t output_neuron=0; output_neuron<output_size; ++output_neuron){
        sum += pow((network_output[output_neuron] - expected_output[output_neuron]), 2);
    }
    return (1/(double)output_size)*sum;
}


double mean_squared_error_loss_derivative(const double predicted, const double actual, const size_t output_size) {
    return (2.0 / (double)output_size) * (predicted - actual);
}


double multi_class_cross_entropy_loss(const size_t output_size, const double *network_output, const double *expected_output){
    double sum = 0;
    for(size_t output_neuron=0; output_neuron<output_size; ++output_neuron){
        sum += expected_output[output_neuron] * safe_log(network_output[output_neuron]);
    }
    return -sum;
}


double multi_class_cross_entropy_loss_derivative(const double predicted, const double actual) {
    if (predicted == 0) return 0; // handle log(0)
    return -actual / predicted;
}


double binary_cross_entropy_loss(const size_t output_size, const double *network_output, const double *expected_output){
    double correct_class = -1;
    if(output_size > 2)
        fprintf(stderr, "Using binary cross entropy on outputs with more than one class, this will not work as supposed! Consider changing to another loss function that supports multiple classes\n");
    if(expected_output[0] == 1 && expected_output[1] == 1)
        fprintf(stderr, "Expected output flags both classes as correct when only one should be the correct one\n");
    if(expected_output[0] == 1) correct_class = 0;
    if(expected_output[1] == 1) correct_class = 1;
    return -(correct_class*safe_log(network_output[1]) + (1-correct_class)*safe_log(1-network_output[1]));
}


double binary_cross_entropy_loss_derivative(double predicted, double actual) {
    if (predicted == 0 || predicted == 1) return 0; // handle log(0)
    return (predicted - actual) / (predicted * (1 - predicted));
}