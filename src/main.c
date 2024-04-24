#include "nn_core.h"
#include "activations.h"
#include "loss.h"
#include "utils.h"
#include "data.h"

int main(){
    srand(time(NULL));

    mnist_handwritten_digits_data mnist_data = load_mnist_data("../data/mnist/handwritten-digits/train/train-images.idx3-ubyte",
                                                               "../data/mnist/handwritten-digits/train/train-labels.idx1-ubyte",
                                                               "../data/mnist/handwritten-digits/test/t10k-images.idx3-ubyte",
                                                               "../data/mnist/handwritten-digits/test/t10k-labels.idx1-ubyte"
                                                               );
    if(mnist_data.training_images.magic_number == -1){
        fprintf(stderr, "Error loading mnist data!\n");
        exit(1);
    }

    //int random = rand()/mnist_data.training_images.number_of_images;

    for(int photo=0; photo<10; ++photo){
        for(int i=0; i<mnist_data.training_images.number_of_rows; ++i){
            for(int j=0; j<mnist_data.training_images.number_of_columns; ++j){
                fprintf(stdout, "%.0f\t", mnist_data.training_images.images[photo][i*mnist_data.training_images.number_of_rows+j]*255);
            }
            fprintf(stdout, "\n");
        }
        fprintf(stdout, "\n");
        for(int i=0; i<10; ++i){
            fprintf(stdout, "%.0f ", mnist_data.training_labels.labels[photo][i]);
        }
        fprintf(stdout, "\n\n\n");
    }


    size_t layers[] = {64, 64, 10};
    int layers_activations[] = {RELU_ACTIVATION, RELU_ACTIVATION, SOFTMAX_ACTIVATION};
    int loss_function = MULTI_CROSS_ENTROPY_LOSS;
    size_t layers_num = sizeof(layers)/sizeof(layers[0]);
    double learning_rate = 0.000001;

    NeuralNetwork* nn = create_neural_network(mnist_data.training_images.number_of_rows * mnist_data.training_images.number_of_columns,
                                              layers_num,
                                              layers,
                                              layers_activations,
                                              loss_function,
                                              learning_rate);

    if(nn == NULL){
        fprintf(stderr, "Error creating neural network\n");
        exit(1);
    }


    size_t batch_size = 256;
    int epochs = 10000;

    for(int epoch = 0; epoch < epochs; epoch++) {
        // Shuffle the training data at the beginning of each epoch
        shuffle_training_data(mnist_data.training_images.images, mnist_data.training_labels.labels, mnist_data.training_images.number_of_images);

        double batch_loss;
        for(size_t i = 0; i < mnist_data.training_images.number_of_images; i += batch_size) {
            batch_loss = 0.0;
            for(size_t j = i; j < i + batch_size && j < mnist_data.training_images.number_of_images; j++) {
                double *network_output = feedforward(nn, mnist_data.training_images.images[j]);
                /*fprintf(stdout, "[");
                for(size_t x=0; x<10; ++x){
                    fprintf(stdout, "%f, ", network_output[x]);
                }
                fprintf(stdout, "]\n");*/

                batch_loss += calculate_loss(nn, network_output, mnist_data.training_labels.labels[j]);
                backpropagation(nn, mnist_data.training_images.images[j], mnist_data.training_labels.labels[j]);
                free(network_output);
            }
            batch_loss /= (double)batch_size;
        }
        fprintf(stdout, "Epoch %d, Batch Loss: %f\n", epoch + 1, batch_loss);
        int random = rand()/mnist_data.training_images.number_of_images;
        double *network_output = feedforward(nn, mnist_data.training_images.images[random]);

        fprintf(stdout, "Net Output: ");
        fprintf(stdout, "[");
        for(size_t x=0; x<10; ++x){
            fprintf(stdout, "%f, ", network_output[x]);
        }
        fprintf(stdout, "]\n");
        fprintf(stdout, "Expected Output: ");
        fprintf(stdout, "[");
        for(size_t x=0; x<10; ++x){
            fprintf(stdout, "%f, ", mnist_data.training_labels.labels[random][x]);
        }
        fprintf(stdout, "]\n");
        fprintf(stdout, "\n");
    }

    destroy_neural_network(nn);
    destroy_mnist_data(mnist_data);

    return 0;
}