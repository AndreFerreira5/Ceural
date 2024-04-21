#ifndef DIGITS_NN_C_ACTIVATIONS_H
#define DIGITS_NN_C_ACTIVATIONS_H

#include "utils.h"

#define LINEAR_ACTIVATION 0
#define SIGMOID_ACTIVATION 1
#define TANH_ACTIVATION 2
#define RELU_ACTIVATION 3
#define SOFTMAX_ACTIVATION 4

double linear(double x);
double linear_derivative(double x);
double sigmoid(double x);
double sigmoid_derivative(double x);
double tanh(double x);
double tanh_derivative(double x);
double relu(double x);
double relu_derivative(double x);
double *softmax(size_t inputs_num, const double *inputs);
double *softmax_derivative(size_t inputs_num, const double *inputs, const double* expected_outputs);

#endif //DIGITS_NN_C_ACTIVATIONS_H
