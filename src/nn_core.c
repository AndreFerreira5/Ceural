#include "nn_core.h"


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


void relu_init_biases(size_t current_layer_size, double *biases){
    for(size_t current_layer_neuron=0; current_layer_neuron<current_layer_size; ++current_layer_neuron){
        biases[current_layer_neuron] = 0.01;
    }
}


void sigmoid_tanh_init_biases(size_t current_layer_size, double *biases){
    for(size_t current_layer_neuron=0; current_layer_neuron<current_layer_size; ++current_layer_neuron){
        biases[current_layer_neuron] = 0;
    }
}



NeuralNetwork* create_neural_network(const size_t input_layer_size, const size_t output_layer_size, size_t dense_layers_num, const size_t *dense_layers_size, int activation_type){
    if(input_layer_size <= 0){
        fprintf(stderr, "Invalid number of neurons for input layer\n");
        return NULL;
    }
    if(output_layer_size <= 0){
        fprintf(stderr, "Invalid number of neurons for output layer\n");
        return NULL;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));

    /* Input Layer Initialization */
    InputLayer input_layer;
    input_layer.size = input_layer_size;
    nn->input_layer = input_layer;

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
            dense_layer.previous_layer_size = previous_layer_size;

            // malloc weights
            dense_layer.weights = malloc(sizeof(double*) * dense_layer_size);
            for(size_t current_layer_neuron=0; current_layer_neuron<dense_layer_size; ++current_layer_neuron){
                dense_layer.weights[current_layer_neuron] = malloc(sizeof(double) * previous_layer_size);
            }
            // malloc bias
            dense_layer.biases = malloc(sizeof(double) * dense_layer_size);

            switch(activation_type){
                default:
                    fprintf(stderr, "Activation type not recognized! Defaulting to ReLU\n");
                case RELU_ACTIVATION:
                    he_init_weights(dense_layer_size, previous_layer_size, dense_layer.weights);
                    relu_init_biases(dense_layer_size, dense_layer.biases);
                    break;
                case SIGMOID_ACTIVATION:
                case TANH_ACTIVATION:
                    gorlot_init_weights(dense_layer_size, previous_layer_size, dense_layer.weights);
                    sigmoid_tanh_init_biases(dense_layer_size, dense_layer.biases);
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

    /* Output Layer Initialization */
    OutputLayer output_layer;
    output_layer.size = output_layer_size;
    nn->output_layer = output_layer;


    gettimeofday(&end, NULL);
    long seconds = end.tv_sec - start.tv_sec;
    long useconds = end.tv_usec - start.tv_usec;
    double elapsed = seconds + useconds * 1E-6;
    fprintf(stdout, "Created neural network with %zu hidden layers with %s activation in %.8f seconds\n", dense_layers_num,
                                                                                                        activation_to_string(activation_type),
                                                                                                        elapsed
                                                                                                        );

    return nn;
}