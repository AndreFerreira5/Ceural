#ifndef DIGITS_NN_C_NN_CORE_H
#define DIGITS_NN_C_NN_CORE_H

#include "utils.h"

#define RELU_ACTIVATION 0
#define SIGMOID_ACTIVATION 1
#define TANH_ACTIVATION 2

static inline const char* activation_to_string(int activation_type) {
    switch (activation_type) {
        case RELU_ACTIVATION:
            return "ReLU";
        case SIGMOID_ACTIVATION:
            return "Sigmoid";
        case TANH_ACTIVATION:
            return "Tanh";
        default:
            return "Unknown";
    }
}

typedef struct {
    size_t size;
} InputLayer;


typedef struct {
    size_t size;
} OutputLayer;


typedef struct {
    size_t size;
    size_t previous_layer_size;
    double **weights;
    double *biases;
} DenseLayer;


typedef struct{
    size_t dense_layers_num;

    InputLayer input_layer;
    DenseLayer *dense_layers;
    OutputLayer output_layer;
} NeuralNetwork;


/* Creates and returns a neural network with the provided layer sizes and activation type */
NeuralNetwork *create_neural_network(size_t input_layer_size,
                                     size_t output_layer_size,
                                     size_t dense_layers_num,
                                     const size_t dense_layers_size[],
                                     int activation_type);


void destroy_neural_network(NeuralNetwork *nn);

#endif //DIGITS_NN_C_NN_CORE_H
