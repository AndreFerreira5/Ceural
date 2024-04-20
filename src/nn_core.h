#ifndef DIGITS_NN_C_NN_CORE_H
#define DIGITS_NN_C_NN_CORE_H

#include "utils.h"

#define LINEAR_ACTIVATION 0
#define SIGMOID_ACTIVATION 1
#define TANH_ACTIVATION 2
#define RELU_ACTIVATION 3
#define SOFTMAX_ACTIVATION 4

static inline const char* activation_to_string(int activation_type) {
    switch (activation_type) {
        case LINEAR_ACTIVATION:
            return "Linear";
        case SIGMOID_ACTIVATION:
            return "Sigmoid";
        case TANH_ACTIVATION:
            return "Tanh";
        case RELU_ACTIVATION:
            return "ReLU";
        case SOFTMAX_ACTIVATION:
            return "Softmax";
        default:
            return "Unknown";
    }
}


typedef struct {
    size_t size;
    int activation_type;
    size_t previous_layer_size;
    double **weights;
    double *biases;
} DenseLayer;


typedef struct{
    size_t input_layer_size;
    size_t dense_layers_num;
    DenseLayer *dense_layers;
} NeuralNetwork;


/* Creates and returns a neural network with the provided layer sizes and activation type */
NeuralNetwork *create_neural_network(size_t input_layer_size,
                                     size_t dense_layers_num,
                                     const size_t *dense_layers_size,
                                     const int *dense_layers_activation_types
                                     );

/* Deallocates the provided neural network */
void destroy_neural_network(NeuralNetwork *nn);

/* Feeds the provided input into the provided neural network and returns the output
 * If the neural network has X neurons, then the first X elements of the provided input will be fed to the network */
double *feedforward(NeuralNetwork *nn, const double *input);

#endif //DIGITS_NN_C_NN_CORE_H
