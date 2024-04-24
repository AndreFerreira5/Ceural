#include "data.h"


int read_from_file(void *var_pointer, size_t bytes_to_read, FILE* file){
    size_t result = fread(var_pointer, bytes_to_read, 1, file);
    if (result == 0) {
        if (feof(file)) {
            fprintf(stderr, "End of file reached or no data to read\n");
        } else if (ferror(file)) {
            fprintf(stderr, "Error reading from file");
        }
        fclose(file);
        return 1;
    }
    return 0;
}


mnist_images_set load_mnist_handwritten_images(const char* images_filepath){
    FILE *images_set_file = fopen(images_filepath, "rb");
    if (images_set_file == NULL) {
        fprintf(stderr, "Failed to open training images file!\n");
        return (mnist_images_set){-1};
    }

    mnist_images_set images_set;

    int32_t magic_number;
    if(read_from_file(&magic_number, sizeof(int32_t), images_set_file)) return (mnist_images_set){-1};
    magic_number = ntohl(magic_number);
    images_set.magic_number = magic_number;

    int32_t number_of_images;
    if(read_from_file(&number_of_images, sizeof(int32_t), images_set_file)) return (mnist_images_set){-1};
    number_of_images = ntohl(number_of_images);
    images_set.number_of_images = number_of_images;


    int32_t number_of_rows;
    if(read_from_file(&number_of_rows, sizeof(int32_t), images_set_file)) return (mnist_images_set){-1};
    number_of_rows = ntohl(number_of_rows);
    images_set.number_of_rows = number_of_rows;

    int32_t number_of_columns;
    if(read_from_file(&number_of_columns, sizeof(int32_t), images_set_file)) return (mnist_images_set){-1};
    number_of_columns = ntohl(number_of_columns);
    images_set.number_of_columns = number_of_columns;


    uint8_t **images_data = malloc(number_of_images * sizeof(uint8_t*));
    for (int i = 0; i < number_of_images; ++i) {
        images_data[i] = malloc(number_of_rows * number_of_columns * sizeof(uint8_t));
    }
    for(int32_t image=0; image<number_of_images; ++image){
        for(int32_t pixel=0; pixel<number_of_rows*number_of_columns; ++pixel){
            if(read_from_file(&images_data[image][pixel], sizeof(uint8_t), images_set_file)){
                free_uint8_array(images_data, number_of_images);
                return (mnist_images_set){-1};
            }
            //fprintf(stdout, "%d, ", images_data[image][pixel]);
        }
    }

    /*
    for(int32_t image=0; image<number_of_images; ++image){
        for(int32_t pixel=0; pixel<number_of_rows*number_of_columns; ++pixel){
            fprintf(stdout, "%hhu, ", images_data[image][pixel]);
        }
        fprintf(stdout, "\n\n");
    }*/


    fclose(images_set_file);
    double **double_images_data = convert_uint8_to_double(images_data, number_of_images, number_of_rows*number_of_columns);
    normalize_double_data(double_images_data, number_of_images, number_of_rows*number_of_columns);
    images_set.images = double_images_data;
    free_uint8_array(images_data, number_of_images);

    /*for(int32_t image=0; image<number_of_images; ++image){
        for(int32_t pixel=0; pixel<number_of_rows*number_of_columns; ++pixel){
            fprintf(stdout, "%f, ", images_set.images[image][pixel]);
        }
        fprintf(stdout, "\n\n");
    }*/
    return images_set;
}


mnist_labels_set load_mnist_handwritten_labels(const char* labels_filepath){
    FILE *labels_set_file = fopen(labels_filepath, "rb");
    if (labels_set_file == NULL) {
        fprintf(stderr, "Failed to open training images file!\n");
        return (mnist_labels_set){-1};
    }

    mnist_labels_set labels_set;

    int32_t magic_number;
    if(read_from_file(&magic_number, sizeof(int32_t), labels_set_file)) return (mnist_labels_set){-1};
    magic_number = ntohl(magic_number);
    labels_set.magic_number = magic_number;

    int32_t number_of_items;
    if(read_from_file(&number_of_items, sizeof(int32_t), labels_set_file)) return (mnist_labels_set){-1};
    number_of_items = ntohl(number_of_items);
    labels_set.number_of_items = number_of_items;


    uint8_t **labels_data = malloc(number_of_items * sizeof(uint8_t*));
    for (int i = 0; i < number_of_items; ++i) {
        labels_data[i] = malloc(10 * sizeof(uint8_t));
    }

    for(int32_t label=0; label<number_of_items; ++label){
        uint8_t label_value;
        if(read_from_file(&label_value, sizeof(uint8_t), labels_set_file)){
            free_uint8_array(labels_data, number_of_items);
            return (mnist_labels_set){-1};
        }

        for(int i = 0; i < 10; ++i){
            if (i == label_value) {
                labels_data[label][i] = 1;
            } else {
                labels_data[label][i] = 0;
            }
        }
    }

    fclose(labels_set_file);
    double **double_labels_data = convert_uint8_to_double(labels_data, number_of_items, 10);
    normalize_double_data(double_labels_data, number_of_items, 10);
    labels_set.labels = double_labels_data;
    free_uint8_array(labels_data, number_of_items);
    return labels_set;
}


mnist_handwritten_digits_data load_mnist_data(const char* training_images_filepath,
                                              const char* training_labels_filepath,
                                              const char* test_images_filepath,
                                              const char* test_labels_filepath
                                             ){
    mnist_handwritten_digits_data mnist_data;
    mnist_data.training_images = load_mnist_handwritten_images(training_images_filepath);
    if(mnist_data.training_images.magic_number == -1){
        fprintf(stderr, "Error loading mnist training images!\n");
        return (mnist_handwritten_digits_data){-1};
    }
    mnist_data.training_labels = load_mnist_handwritten_labels(training_labels_filepath);
    if(mnist_data.training_labels.magic_number == -1){
        fprintf(stderr, "Error loading mnist training labels!\n");
        return (mnist_handwritten_digits_data){-1};
    }
    mnist_data.test_images = load_mnist_handwritten_images(test_images_filepath);
    if(mnist_data.test_images.magic_number == -1){
        fprintf(stderr, "Error loading mnist test images!\n");
        return (mnist_handwritten_digits_data){-1};
    }
    mnist_data.test_labels = load_mnist_handwritten_labels(test_labels_filepath);
    if(mnist_data.test_labels.magic_number == -1){
        fprintf(stderr, "Error loading mnist test labels!\n");
        return (mnist_handwritten_digits_data){-1};
    }

    return mnist_data;
}


void destroy_mnist_data(mnist_handwritten_digits_data mnist_data){
    free_double_array(mnist_data.training_images.images, mnist_data.training_images.number_of_images);
    free_double_array(mnist_data.training_labels.labels, mnist_data.training_labels.number_of_items);
    free_double_array(mnist_data.test_images.images, mnist_data.test_images.number_of_images);
    free_double_array(mnist_data.test_labels.labels, mnist_data.test_labels.number_of_items);
}
