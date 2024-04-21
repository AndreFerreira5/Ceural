#include "nn_core.h"
#include "activations.h"
#include "loss.h"
#include "utils.h"

int main(){
    srand(time(NULL));

    size_t layers[] = {16, 16, 4};
    int layers_activations[] = {SIGMOID_ACTIVATION, SIGMOID_ACTIVATION, SOFTMAX_ACTIVATION};
    int loss_function = MULTI_CROSS_ENTROPY_LOSS;
    size_t layers_num = sizeof(layers)/sizeof(layers[0]);

    NeuralNetwork* nn = create_neural_network(4, layers_num, layers, layers_activations, loss_function, 0.00005);

    if(nn == NULL){
        fprintf(stderr, "Error creating neural network\n");
        exit(1);
    }

    for(size_t i=0; i<nn->dense_layers_num; ++i){
        fprintf(stdout, "Dense Layer n%zu has %zu neurons\n", i, nn->dense_layers[i].size);
        for(size_t j=0; j<nn->dense_layers[i].size; j++){
            fprintf(stdout, "Weights for neuron %zu: [", j);
            for(size_t x=0; x<nn->dense_layers[i].previous_layer_size; x++){
                fprintf(stdout, "%f, ", nn->dense_layers[i].weights[j][x]);
            }
            fprintf(stdout, "]\n");

            fprintf(stdout, "Biases for neuron %zu: [", j);
            for(size_t x=0; x<nn->dense_layers[i].size; x++){
                fprintf(stdout, "%f, ", nn->dense_layers[i].biases[x]);
            }
            fprintf(stdout, "]\n");
        }
    }
    double input[] = {0.23, 0.76, 0.93, 0.12};
    double expected_output[] = {0, 1, 0, 0};

    for(int i=0; i<10000; ++i){
        fprintf(stdout, "Epoch %d\n", i+1);

        double *network_output = feedforward(nn, input);
        fprintf(stdout, "[");
        for(size_t j=0; j<4; ++j){
            fprintf(stdout, "%f, ", network_output[j]);
        }
        fprintf(stdout, "]\n");

        double loss = calculate_loss(nn, network_output, expected_output);
        fprintf(stdout, "Loss: %f\n", loss);

        backpropagation(nn, input, expected_output);


        free(network_output);
    }

    destroy_neural_network(nn);

    return 0;
}