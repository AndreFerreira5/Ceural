#ifndef DIGITS_NN_C_DATA_H
#define DIGITS_NN_C_DATA_H

#include "utils.h"


typedef struct{
    int32_t magic_number;
    int32_t number_of_items;
    double **labels;
} mnist_labels_set;


typedef struct{
    int32_t magic_number;
    int32_t number_of_images;
    int32_t number_of_rows;
    int32_t number_of_columns;
    double **images;
} mnist_images_set;


typedef struct{
    mnist_images_set training_images;
    mnist_labels_set training_labels;
    mnist_images_set test_images;
    mnist_labels_set test_labels;
} mnist_handwritten_digits_data;


mnist_handwritten_digits_data load_mnist_data(const char* training_images_filepath,
                                              const char* training_labels_filepath,
                                              const char* test_images_filepath,
                                              const char* test_labels_filepath
                                              );


void destroy_mnist_data(mnist_handwritten_digits_data mnist_data);


#endif //DIGITS_NN_C_DATA_H
