#ifndef DIGITS_NN_C_NN_CORE_H
#define DIGITS_NN_C_NN_CORE_H

#include "utils.h"


typedef struct {
    size_t size;
    size_t previous_layer_size;
    double **weights;
    double *biases;
    double *outputs;
    double (*activation)(double);
    double (*activation_derivative)(double);
} DenseLayer;


typedef struct{
    size_t input_layer_size;
    size_t dense_layers_num;
    DenseLayer *dense_layers;
    double (*loss)(size_t, const double*, const double*);
    double (*loss_derivative)(const double, const double);
    double learning_rate;
} NeuralNetwork;


/* Creates and returns a neural network with the provided layer sizes and activation type */
NeuralNetwork *create_neural_network(size_t input_layer_size,
                                     size_t dense_layers_num,
                                     const size_t *dense_layers_size,
                                     const int *dense_layers_activation_types,
                                     int loss_function,
                                     double learning_rate
                                     );

/* Deallocates the provided neural network */
void destroy_neural_network(NeuralNetwork *nn);

/* Feeds the provided input into the provided neural network and returns the output
 * If the neural network has X neurons, then the first X elements of the provided input will be fed to the network */
double *feedforward(NeuralNetwork *nn, const double *input);

/* Calculates the loss of the network, comparing the network output and the expected output using
 * the loss function of the network */
double calculate_loss(NeuralNetwork *nn, const double *network_output, const double *expected_output);

/* Propagates backwards through the network, calculating gradients and updating weights and biases
 * based on the network output and the expected output, using the activation functions attributed to the layers of the network*/
void backpropagation(NeuralNetwork *nn, const double *network_input, const double *expected_output);

#endif //DIGITS_NN_C_NN_CORE_H
