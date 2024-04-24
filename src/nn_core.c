#include "nn_core.h"
#include "activations.h"
#include "loss.h"


double random_normal(double mean, double stddev) {
    double u1, u2, z0;
    u1 = rand() / (RAND_MAX + 1.0);
    u2 = rand() / (RAND_MAX + 1.0);
    z0 = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
    return z0 * stddev + mean;
}


double random_uniform(double min, double max) {
    return min + (double)rand() / ((double)RAND_MAX / (max - min));
}


void he_init_weights(size_t current_layer_size, size_t previous_layer_size, double** weights){
    double standard_deviation = sqrt(2. / (double)previous_layer_size);

    for(size_t current_layer_neuron=0; current_layer_neuron<current_layer_size; ++current_layer_neuron){
        for(size_t previous_layer_neuron=0; previous_layer_neuron<previous_layer_size; ++previous_layer_neuron){
            weights[current_layer_neuron][previous_layer_neuron] = random_normal(0, standard_deviation);
        }
    }
}


void gorlot_init_weights(size_t current_layer_size, size_t previous_layer_size, double** weights){
    double standard_deviation = sqrt(6. / (previous_layer_size + current_layer_size));

    for(size_t current_layer_neuron=0; current_layer_neuron<current_layer_size; ++current_layer_neuron){
        for(size_t previous_layer_neuron=0; previous_layer_neuron<previous_layer_size; ++previous_layer_neuron){
            weights[current_layer_neuron][previous_layer_neuron] = random_uniform(-standard_deviation, standard_deviation);
        }
    }
}


void init_biases(size_t current_layer_size, double *biases, const double bias_value){
    for(size_t current_layer_neuron=0; current_layer_neuron<current_layer_size; ++current_layer_neuron){
        biases[current_layer_neuron] = bias_value;
    }
}



NeuralNetwork *create_neural_network(const size_t input_layer_size, size_t dense_layers_num, const size_t *dense_layers_size, const int *dense_layers_activation_types, const int loss_function, const double learning_rate){
    if(input_layer_size <= 0){
        fprintf(stderr, "Invalid number of neurons for input layer\n");
        return NULL;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    nn->input_layer_size = input_layer_size;
    nn->learning_rate = learning_rate;

    switch(loss_function){
        default:
            fprintf(stderr, "Unrecognized loss function! Defaulting to Mean Squared Error\n");
        case MEAN_SQUARED_ERROR_LOSS:
            nn->loss = NULL;
            nn->loss_derivative = NULL;
            break;
        case MULTI_CROSS_ENTROPY_LOSS:
            nn->loss = multi_class_cross_entropy_loss;
            nn->loss_derivative = multi_class_cross_entropy_loss_derivative;
            break;
        case BINARY_CROSS_ENTROPY_LOSS:
            nn->loss = binary_cross_entropy_loss;
            nn->loss_derivative = binary_cross_entropy_loss_derivative;
            break;
    }

    /* Dense Layers Initialization */
    if(dense_layers_num){ // if number of dense layers is greater than 0
        // check if dense layers sizes are valid (>0)
        for(size_t i=0; i<dense_layers_num; ++i){
            if(dense_layers_size[i] <= 0){
                fprintf(stderr, "Wrong dense layer size of %zu: Dense layer size must be greater than 0\n", dense_layers_size[i]);
                return NULL;
            }
        }
        nn->dense_layers_num = dense_layers_num;
        nn->dense_layers = malloc(sizeof(DenseLayer) * dense_layers_num);

        size_t previous_layer_size = input_layer_size;

        // init each dense layer with provided sizes
        for(size_t layer=0; layer<dense_layers_num; ++layer){
            size_t dense_layer_size = dense_layers_size[layer]; // get layer size from arguments

            DenseLayer dense_layer;
            dense_layer.size = dense_layer_size;
            // malloc weights
            dense_layer.weights = malloc(sizeof(double*) * dense_layer_size);
            for(size_t current_layer_neuron=0; current_layer_neuron<dense_layer_size; ++current_layer_neuron){
                dense_layer.weights[current_layer_neuron] = malloc(sizeof(double) * previous_layer_size);
            }
            // malloc bias
            dense_layer.biases = malloc(sizeof(double) * dense_layer_size);

            switch(dense_layers_activation_types[layer]){
                default:
                    fprintf(stderr, "Activation type not recognized for layer %zu! Defaulting to ReLU\n", layer);
                case RELU_ACTIVATION:
                    dense_layer.activation = relu;
                    dense_layer.activation_derivative = relu_derivative;
                    he_init_weights(dense_layer_size, previous_layer_size, dense_layer.weights);
                    init_biases(dense_layer_size, dense_layer.biases, 0.01);
                    break;
                case LINEAR_ACTIVATION:
                    dense_layer.activation = linear;
                    dense_layer.activation_derivative = linear_derivative;
                    gorlot_init_weights(dense_layer_size, previous_layer_size, dense_layer.weights);
                    init_biases(dense_layer_size, dense_layer.biases, 0);
                    break;
                case SOFTMAX_ACTIVATION:
                    if(layer != dense_layers_num-1){
                        fprintf(stderr, "Softmax activation is not allowed in intermediate layers. It should only be used in the output layer. Defaulting to ReLU on layer %lu!\n", layer);
                        dense_layer.activation = relu;
                        dense_layer.activation_derivative = relu_derivative;
                        he_init_weights(dense_layer_size, previous_layer_size, dense_layer.weights);
                        init_biases(dense_layer_size, dense_layer.biases, 0.01);
                        break;
                    }
                    dense_layer.activation = NULL;
                    dense_layer.activation_derivative = NULL;
                    gorlot_init_weights(dense_layer_size, previous_layer_size, dense_layer.weights);
                    init_biases(dense_layer_size, dense_layer.biases, 0);
                    break;
                case SIGMOID_ACTIVATION:
                    dense_layer.activation = sigmoid;
                    dense_layer.activation_derivative = sigmoid_derivative;
                    gorlot_init_weights(dense_layer_size, previous_layer_size, dense_layer.weights);
                    init_biases(dense_layer_size, dense_layer.biases, 0);
                    break;
                case TANH_ACTIVATION:
                    nn->dense_layers[layer].activation = tanh;
                    nn->dense_layers[layer].activation_derivative = tanh_derivative;
                    gorlot_init_weights(dense_layer_size, previous_layer_size, dense_layer.weights);
                    init_biases(dense_layer_size, dense_layer.biases, 0);
                    break;
            }
            dense_layer.previous_layer_size = previous_layer_size;

            // assign created dense layer struct to nn dense layers array
            nn->dense_layers[layer] = dense_layer;

            // store the current dense layer size for the next dense layer creation
            previous_layer_size = dense_layer_size;
        }
    } else { // if number of dense layers is 0
        fprintf(stderr, "Network should have at least one dense layer to serve as the output layer\n");
        free(nn);
        return NULL;
    }


    gettimeofday(&end, NULL);
    long seconds = end.tv_sec - start.tv_sec;
    long useconds = end.tv_usec - start.tv_usec;
    double elapsed = seconds + useconds * 1E-6;
    fprintf(stdout, "Created neural network with %zu hidden layers in %.8f seconds\n", dense_layers_num,
                                                                                                        elapsed
                                                                                                        );

    return nn;
}


void destroy_neural_network(NeuralNetwork *nn){
    for(size_t dense_layer=0; dense_layer<nn->dense_layers_num; ++dense_layer){
        for(size_t dense_layer_neuron=0; dense_layer_neuron<nn->dense_layers[dense_layer].size; ++dense_layer_neuron)
            free(nn->dense_layers[dense_layer].weights[dense_layer_neuron]);

        free(nn->dense_layers[dense_layer].weights);
        free(nn->dense_layers[dense_layer].biases);
    }
    nn = NULL;
}


double weighted_sum(size_t previous_layer_size, size_t current_layer_size, double *input, double *weights, double bias){
    double weighted_sum = 0;
    for(size_t input_neuron=0; input_neuron<previous_layer_size; ++input_neuron){
        weighted_sum += input[input_neuron] * weights[input_neuron];
    }
    return weighted_sum;
}


double *feedforward(NeuralNetwork *nn, const double *input){

    size_t previous_layer_size = nn->input_layer_size;
    double *inputs = malloc(sizeof(double) * nn->input_layer_size);
    for(size_t i=0; i<nn->input_layer_size; ++i)
        inputs[i] = input[i];

    double *outputs = NULL;
    for(size_t current_layer=0; current_layer<nn->dense_layers_num; ++current_layer){

        if(nn->dense_layers[current_layer].activation == NULL){ // activation function is softmax
            double *biased_inputs = malloc(sizeof(double) * nn->dense_layers[current_layer].size);
            for(size_t current_layer_neuron=0; current_layer_neuron<nn->dense_layers[current_layer].size; ++current_layer_neuron){
                double weighted_sum_value = weighted_sum(previous_layer_size,
                                                         nn->dense_layers[current_layer].size,
                                                         inputs, nn->dense_layers[current_layer].weights[current_layer_neuron],
                                                         nn->dense_layers[current_layer].biases[current_layer_neuron]
                );
                double biased_value = weighted_sum_value + nn->dense_layers[current_layer].biases[current_layer_neuron];

                biased_inputs[current_layer_neuron] = biased_value;
            }
            outputs = softmax(nn->dense_layers[current_layer].size, biased_inputs);
            free(biased_inputs);
        } else {
            outputs = malloc(sizeof(double) * nn->dense_layers[current_layer].size);
            for(size_t current_layer_neuron=0; current_layer_neuron<nn->dense_layers[current_layer].size; ++current_layer_neuron){
                double weighted_sum_value = weighted_sum(previous_layer_size,
                                                         nn->dense_layers[current_layer].size,
                                                         inputs, nn->dense_layers[current_layer].weights[current_layer_neuron],
                                                         nn->dense_layers[current_layer].biases[current_layer_neuron]
                );
                double biased_value = weighted_sum_value + nn->dense_layers[current_layer].biases[current_layer_neuron];
                outputs[current_layer_neuron] = nn->dense_layers[current_layer].activation(biased_value);
            }
        }

        free(inputs);
        inputs = malloc(sizeof(double) * nn->dense_layers[current_layer].size);
        if(nn->dense_layers[current_layer].outputs != NULL){
            free(nn->dense_layers[current_layer].outputs);
        }
        nn->dense_layers[current_layer].outputs = malloc(sizeof(double) * nn->dense_layers[current_layer].size);
        memcpy(inputs, outputs, sizeof(double) * nn->dense_layers[current_layer].size);
        memcpy(nn->dense_layers[current_layer].outputs, outputs, sizeof(double) * nn->dense_layers[current_layer].size);
        free(outputs);
        outputs = NULL;
        previous_layer_size = nn->dense_layers[current_layer].size;
    }

    return inputs;
}


void backpropagation(NeuralNetwork *nn, const double *network_input, const double *expected_output){
    size_t last_layer_index = nn->dense_layers_num-1;



    double *deltas = malloc(sizeof(double) * nn->dense_layers[last_layer_index].size);
    double *softmax_derivatives = NULL;
    if(nn->dense_layers[last_layer_index].activation == NULL){
        softmax_derivatives = softmax_derivative(nn->dense_layers[last_layer_index].size, nn->dense_layers[last_layer_index].outputs, expected_output);
    }

    for(size_t neuron=0; neuron<nn->dense_layers[last_layer_index].size; ++neuron){
        double network_value = nn->dense_layers[last_layer_index].outputs[neuron];
        double expected_value = expected_output[neuron];
        double loss_derivative = nn->loss_derivative ? nn->loss_derivative(network_value, expected_value) :
                                 mean_squared_error_loss_derivative(network_value, expected_value, nn->dense_layers[last_layer_index].size);

        if(nn->dense_layers[last_layer_index].activation == NULL){
            deltas[neuron] = loss_derivative *
                             softmax_derivatives[neuron];
        } else {
            deltas[neuron] = loss_derivative *
                             nn->dense_layers[last_layer_index].activation_derivative(network_value);
        }
    }

    if(softmax_derivatives) free(softmax_derivatives);


    for(size_t layer=nn->dense_layers_num-2; ; --layer){
        if(layer == 0) break;

        DenseLayer *current_layer = &nn->dense_layers[layer];
        DenseLayer *next_layer = &nn->dense_layers[layer + 1];

        double *new_deltas = malloc(sizeof(double) * current_layer->size);
        for(size_t current_layer_neuron=0; current_layer_neuron<current_layer->size; ++current_layer_neuron){
            double sum = 0;
            for(size_t next_layer_neuron=0; next_layer_neuron<next_layer->size; ++next_layer_neuron){
                sum += deltas[next_layer_neuron] * next_layer->weights[next_layer_neuron][current_layer_neuron];
            }
            //if(current_layer->activation == NULL){}
                //new_deltas[current_layer_neuron] = sum * new_deltas[current_layer_neuron];
            //else
                new_deltas[current_layer_neuron] = sum * current_layer->activation_derivative(current_layer->outputs[current_layer_neuron]);
        }

        for(size_t next_layer_neuron=0; next_layer_neuron<next_layer->size; ++next_layer_neuron){
            for(size_t current_layer_neuron=0; current_layer_neuron<current_layer->size; ++current_layer_neuron){
                next_layer->weights[next_layer_neuron][current_layer_neuron] -= nn->learning_rate * deltas[next_layer_neuron] * current_layer->outputs[current_layer_neuron];
            }
            next_layer->biases[next_layer_neuron] -= nn->learning_rate * deltas[next_layer_neuron];
        }

        free(deltas);
        deltas = new_deltas;
    }

    DenseLayer *first_layer = &nn->dense_layers[0];
    for(size_t current_neuron=0; current_neuron<first_layer->size; ++current_neuron){
        for(size_t input_neuron=0; input_neuron<nn->input_layer_size; ++input_neuron){
            first_layer->weights[current_neuron][input_neuron] -= nn->learning_rate * deltas[current_neuron] * network_input[input_neuron];
        }
        first_layer->biases[current_neuron] -= nn->learning_rate * deltas[current_neuron];
    }

    free(deltas);
}


double calculate_loss(NeuralNetwork *nn, const double *network_output, const double *expected_output){
    if(nn->loss == NULL)
        return mean_squared_error_loss(nn->dense_layers[nn->dense_layers_num-1].size, network_output, expected_output);
    return nn->loss(nn->dense_layers[nn->dense_layers_num-1].size, network_output, expected_output);
}