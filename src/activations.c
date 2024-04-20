#include "activations.h"
#include "utils.h"


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}


double relu(double x) {
    return x > 0 ? x : 0;
}


double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
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