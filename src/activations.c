#include "activations.h"
#include "utils.h"


double linear(double x){
    return x;
}


double linear_derivative(double x){
    return 1;
}


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}


double tanh(double x){
    double epx = exp(x);
    double enx = exp(-x);
    return (epx-enx)/(epx+enx);
}


double tanh_derivative(double x){
    double tanh_value = tanh(x);
    return 1 - (tanh_value*tanh_value);
}


double relu(double x) {
    return x > 0 ? x : 0;
}


double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}


double *softmax(size_t inputs_num, double *inputs){
    double *outputs = malloc(sizeof(double) * inputs_num);

    double max_input = inputs[0];
    for (size_t i = 1; i < inputs_num; i++) {
        if (inputs[i] > max_input) {
            max_input = inputs[i];
        }
    }

    double exp_sum = 0.0;
    for(size_t i=0; i<inputs_num; ++i){
        outputs[i] = exp(inputs[i] - max_input);
        exp_sum += outputs[i];
    }

    for(size_t i=0; i<inputs_num; ++i){
        outputs[i] /= exp_sum;
    }

    return outputs;
}
