#ifndef DIGITS_NN_C_ACTIVATIONS_H
#define DIGITS_NN_C_ACTIVATIONS_H

#include "utils.h"

double linear(double x);
double linear_derivative(double x);
double sigmoid(double x);
double sigmoid_derivative(double x);
double tanh(double x);
double tanh_derivative(double x);
double relu(double x);
double relu_derivative(double x);
double *softmax(size_t inputs_num, double *inputs);

#endif //DIGITS_NN_C_ACTIVATIONS_H
