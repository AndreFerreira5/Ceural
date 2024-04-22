#include "activations.h"
#include "utils.h"


double linear(const double x){
    return x;
}


double linear_derivative(const double x){
    return 1;
}


double sigmoid(const double x) {
    return 1.0 / (1.0 + exp(-x));
}


double sigmoid_derivative(const double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}


double tanh(const double x){
    double epx = exp(x);
    double enx = exp(-x);
    return (epx-enx)/(epx+enx);
}


double tanh_derivative(const double x){
    double tanh_value = tanh(x);
    return 1 - (tanh_value*tanh_value);
}


double relu(const double x) {
    return x > 0 ? x : 0;
}


double relu_derivative(const double x) {
    return x > 0 ? 1 : 0;
}


double *softmax(const size_t inputs_num, const double *inputs){
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

    if(exp_sum == 0.0){
        free(outputs);
        return NULL;
    }

    for(size_t i=0; i<inputs_num; ++i){
        outputs[i] /= exp_sum;
    }

    return outputs;
}


double *softmax_derivative(const size_t inputs_num, const double *inputs, const double* expected_outputs){
    double *softmax_activated_outputs = softmax(inputs_num, inputs);
    double *softmax_derivatives = malloc(sizeof(double) * inputs_num);
    for (size_t i = 0; i < inputs_num; i++) {
        softmax_derivatives[i] = softmax_activated_outputs[i] - (expected_outputs[i] == 1 ? 1.0 : 0.0);
    }
    free(softmax_activated_outputs);
    return softmax_derivatives;
}