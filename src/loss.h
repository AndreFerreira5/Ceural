#ifndef DIGITS_NN_C_LOSS_H
#define DIGITS_NN_C_LOSS_H

#include "utils.h"

#define MEAN_SQUARED_ERROR_LOSS 0
#define MULTI_CROSS_ENTROPY_LOSS 1
#define BINARY_CROSS_ENTROPY_LOSS 2

double mean_squared_error_loss(size_t output_size, const double *network_output, const double *expected_output);
double mean_squared_error_loss_derivative(double predicted, double actual, size_t output_size);
double multi_class_cross_entropy_loss(size_t output_size, const double *network_output, const double *expected_output);
double multi_class_cross_entropy_loss_derivative(double predicted, double actual);
double binary_cross_entropy_loss(size_t output_size, const double *network_output, const double *expected_output);
double binary_cross_entropy_loss_derivative(double predicted, double actual);

#endif //DIGITS_NN_C_LOSS_H
