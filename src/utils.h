#ifndef DIGITS_NN_C_UTILS_H
#define DIGITS_NN_C_UTILS_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
#include <arpa/inet.h>


static inline void free_double_array(double **array, int count) {
    if (array != NULL) {
        for (int i = 0; i < count; ++i) {
            free(array[i]);
        }
        free(array);
    }
}


static inline void free_uint8_array(uint8_t **array, int count) {
    if (array != NULL) {
        for (int i = 0; i < count; ++i) {
            free(array[i]);
        }
        free(array);
    }
}


static inline double **convert_uint8_to_double(uint8_t **data, int data_size, int subdata_size){
    double **double_data = malloc(data_size * sizeof(double*));
    for (int i = 0; i < data_size; ++i) {
        double_data[i] = malloc(subdata_size * sizeof(double));
    }

    for(int i=0; i<data_size; ++i){
        for(int j = 0; j<subdata_size; ++j){
            double_data[i][j] = (double) data[i][j];
            //fprintf(stdout, "uint8: %d - double: %f\n", data[i][j], double_data[i][j]);
        }
    }

    return double_data;
}


static inline void normalize_double_data(double **data, int data_size, int subdata_size){
    if (data == NULL || data_size == 0 || subdata_size == 0) {
        return;
    }

    double max_value = 0;
    for (int i = 0; i < data_size; ++i) {
        for (int j = 0; j < subdata_size; ++j) {
            if (data[i][j] > max_value) {
                max_value = data[i][j];
            }
        }
    }

    if (max_value == 0) {
        return;
    }

    for (int i = 0; i < data_size; ++i) {
        for (int j = 0; j < subdata_size; ++j) {
            //fprintf(stdout, "double: %f - norm double: %.8f\n", data[i][j], (double) (data[i][j] / max_value));
            data[i][j] = (double) (data[i][j] / max_value);
        }
    }
}


static void swap_double_pointers(double** a, double** b) {
    double* temp = *a;
    *a = *b;
    *b = temp;
}


static inline void shuffle_training_data(double **images, double** labels, int data_size) {
    srand(time(NULL));

    for (int i = 0; i < data_size - 1; i++) {
        int j = i + rand() / (RAND_MAX / (data_size - i) + 1);
        swap_double_pointers(&images[i], &images[j]);
        swap_double_pointers(&labels[i], &labels[j]);
    }
}


#endif //DIGITS_NN_C_UTILS_H
