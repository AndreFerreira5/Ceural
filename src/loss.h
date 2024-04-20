#ifndef DIGITS_NN_C_LOSS_H
#define DIGITS_NN_C_LOSS_H

#include "utils.h"

double mean_squared_error_loss(size_t output_size, const double *network_outputs, const double *expected_output);
double multi_class_cross_entropy_loss(size_t output_size, const double *network_outputs, const double *expected_output);
double binary_cross_entropy_loss(size_t output_size, const double *network_outputs, const double *expected_output);

#endif //DIGITS_NN_C_LOSS_H
