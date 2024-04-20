#include "nn_core.h"
#include "utils.h"

int main(){
    srand(time(NULL));

    size_t layers[] = {8, 8, 4};
    int layers_activations[] = {RELU_ACTIVATION, RELU_ACTIVATION, SOFTMAX_ACTIVATION};
    size_t layers_num = sizeof(layers)/sizeof(layers[0]);

    NeuralNetwork* nn = create_neural_network(4, layers_num, layers, layers_activations);

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

    double *feedforward_output = feedforward(nn, (double[]){0.23, 0.76, 0.93, 0.12});
    fprintf(stdout, "[");
    for(size_t i=0; i<8; ++i){
        fprintf(stdout, "%f, ", feedforward_output[i]);
    }
    fprintf(stdout, "]");
    destroy_neural_network(nn);

    return 0;
}