#include "nn_core.h"
#include "activations.h"


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
    double standard_deviation = sqrt(2. / previous_layer_size);

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



NeuralNetwork *create_neural_network(const size_t input_layer_size, size_t dense_layers_num, const size_t *dense_layers_size, const int *dense_layers_activation_types){
    if(input_layer_size <= 0){
        fprintf(stderr, "Invalid number of neurons for input layer\n");
        return NULL;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));

    nn->input_layer_size = input_layer_size;

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
            dense_layer.activation_type = dense_layers_activation_types[layer];
            dense_layer.previous_layer_size = previous_layer_size;

            // malloc weights
            dense_layer.weights = malloc(sizeof(double*) * dense_layer_size);
            for(size_t current_layer_neuron=0; current_layer_neuron<dense_layer_size; ++current_layer_neuron){
                dense_layer.weights[current_layer_neuron] = malloc(sizeof(double) * previous_layer_size);
            }
            // malloc bias
            dense_layer.biases = malloc(sizeof(double) * dense_layer_size);

            dense_layer.activation_type = dense_layers_activation_types[layer];
            switch(dense_layer.activation_type){
                default:
                    fprintf(stderr, "Activation type not recognized! Defaulting to ReLU\n");
                case RELU_ACTIVATION:
                    he_init_weights(dense_layer_size, previous_layer_size, dense_layer.weights);
                    init_biases(dense_layer_size, dense_layer.biases, 0.01);
                    break;
                case LINEAR_ACTIVATION:
                case SOFTMAX_ACTIVATION:
                case SIGMOID_ACTIVATION:
                case TANH_ACTIVATION:
                    gorlot_init_weights(dense_layer_size, previous_layer_size, dense_layer.weights);
                    init_biases(dense_layer_size, dense_layer.biases, 0);
                    break;
            }

            // assign created dense layer struct to nn dense layers array
            nn->dense_layers[layer] = dense_layer;

            // store the current dense layer size for the next dense layer creation
            previous_layer_size = dense_layer_size;
        }
    } else { // if number of dense layers is 0
        nn->dense_layers_num = dense_layers_num;
        nn->dense_layers = NULL;
    }


    gettimeofday(&end, NULL);
    long seconds = end.tv_sec - start.tv_sec;
    long useconds = end.tv_usec - start.tv_usec;
    double elapsed = seconds + useconds * 1E-6;
    fprintf(stdout, "Created neural network with %zu hidden layers with in %.8f seconds\n", dense_layers_num,
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


// TODO fix potentially memory leak on 'outputs'
double *feedforward(NeuralNetwork *nn, const double *input){
    struct timeval start, end;
    gettimeofday(&start, NULL);


    size_t previous_layer_size = nn->input_layer_size;
    double *inputs = malloc(sizeof(double) * nn->input_layer_size);
    for(size_t i=0; i<nn->input_layer_size; ++i)
        inputs[i] = input[i];
    double *outputs;

    for(size_t current_layer=0; current_layer<nn->dense_layers_num; ++current_layer){
        outputs = malloc(sizeof(double) * nn->dense_layers[current_layer].size);
        int layer_activation_type = nn->dense_layers[current_layer].activation_type;

        double *biased_inputs;
        switch (layer_activation_type) {
            case LINEAR_ACTIVATION:
                for(size_t current_layer_neuron=0; current_layer_neuron<nn->dense_layers[current_layer].size; ++current_layer_neuron){
                    double weighted_sum_value = weighted_sum(previous_layer_size,
                                                             nn->dense_layers[current_layer].size,
                                                             inputs, nn->dense_layers[current_layer].weights[current_layer_neuron],
                                                             nn->dense_layers[current_layer].biases[current_layer_neuron]
                    );
                    double biased_value = weighted_sum_value + nn->dense_layers[current_layer].biases[current_layer_neuron];
                    outputs[current_layer_neuron] = linear(biased_value);
                }
                break;
            case SIGMOID_ACTIVATION:
                for(size_t current_layer_neuron=0; current_layer_neuron<nn->dense_layers[current_layer].size; ++current_layer_neuron){
                    double weighted_sum_value = weighted_sum(previous_layer_size,
                                                             nn->dense_layers[current_layer].size,
                                                             inputs, nn->dense_layers[current_layer].weights[current_layer_neuron],
                                                             nn->dense_layers[current_layer].biases[current_layer_neuron]
                    );
                    double biased_value = weighted_sum_value + nn->dense_layers[current_layer].biases[current_layer_neuron];
                    outputs[current_layer_neuron] = sigmoid(biased_value);
                }
                break;
            case TANH_ACTIVATION:
                for(size_t current_layer_neuron=0; current_layer_neuron<nn->dense_layers[current_layer].size; ++current_layer_neuron){
                    double weighted_sum_value = weighted_sum(previous_layer_size,
                                                             nn->dense_layers[current_layer].size,
                                                             inputs, nn->dense_layers[current_layer].weights[current_layer_neuron],
                                                             nn->dense_layers[current_layer].biases[current_layer_neuron]
                    );
                    double biased_value = weighted_sum_value + nn->dense_layers[current_layer].biases[current_layer_neuron];
                    outputs[current_layer_neuron] = tanh(biased_value);
                }
                break;
            case RELU_ACTIVATION:
                for(size_t current_layer_neuron=0; current_layer_neuron<nn->dense_layers[current_layer].size; ++current_layer_neuron){
                    double weighted_sum_value = weighted_sum(previous_layer_size,
                                                             nn->dense_layers[current_layer].size,
                                                             inputs, nn->dense_layers[current_layer].weights[current_layer_neuron],
                                                             nn->dense_layers[current_layer].biases[current_layer_neuron]
                    );
                    double biased_value = weighted_sum_value + nn->dense_layers[current_layer].biases[current_layer_neuron];
                    outputs[current_layer_neuron] = relu(biased_value);
                }
                break;
            case SOFTMAX_ACTIVATION:
                biased_inputs = malloc(sizeof(double) * nn->dense_layers[current_layer].size);
                for(size_t current_layer_neuron=0; current_layer_neuron<nn->dense_layers[current_layer].size; ++current_layer_neuron){
                    double weighted_sum_value = weighted_sum(previous_layer_size,
                                                             nn->dense_layers[current_layer].size,
                                                             inputs, nn->dense_layers[current_layer].weights[current_layer_neuron],
                                                             nn->dense_layers[current_layer].biases[current_layer_neuron]
                    );
                    double biased_value = weighted_sum_value + nn->dense_layers[current_layer].biases[current_layer_neuron];

                    biased_inputs[current_layer_neuron] = biased_value;
                }
                free(outputs);
                outputs = softmax(nn->dense_layers[current_layer].size, biased_inputs);
                free(biased_inputs);

                break;
            default:
                fprintf(stderr, "Error feedforwarding: Unknown type of activation -> %d\n", layer_activation_type);
                return NULL;
        }


        free(inputs);
        inputs = outputs;
        previous_layer_size = nn->dense_layers[current_layer].size;
    }


    gettimeofday(&end, NULL);
    long seconds = end.tv_sec - start.tv_sec;
    long useconds = end.tv_usec - start.tv_usec;
    double elapsed = seconds + useconds * 1E-6;
    fprintf(stdout, "Feedforward done in %fseconds\n", elapsed);

    return outputs;
}