#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

// gcc -lm q1s.c -o q1s

#define MAX 100000.0

// Struct to store start and stop times
typedef struct {
    struct timespec start;
    struct timespec stop;
} Stopwatch;

/**
 * Multiplies a matrix by a vector where the matrix has dimensions m x m and the vector has dimensions m x 1
 * @param matrix: The matrix to multiply
 * @param vector: The vector to multiply
 * @param m: The dimensions of the matrix and vector
 * @return: The result of the matrix-vector multiplication
 */
double* matrix_vector_product(double* matrix, double* vector, int m) {
    // Dynamically allocate result vector and preallocate it to 0.0
    double* result = (double*) calloc(m, sizeof(double));
    if (!result) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    // Perform matrix-vector multiplication
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            result[i] += matrix[i * m + j] * vector[j];
        }
    }
    
    return result;
}

/** Seed the random number generator with entropy from /dev/urandom
 */
void seed_random() {
    int fd = open("/dev/urandom", O_RDONLY);
    unsigned int seed;
    read(fd, &seed, sizeof(seed));
    close(fd);
    srandom(seed);
}

/** Generate a random double between 0 and MAX
 * @return: A random double
 */
double random_double() {
    return ((double) random() / (double) RAND_MAX * MAX);
}

/**
 * Main function
 * @param argc: Number of arguments
 * @param argv: Array of arguments
 * @return: 0 if successful
 */
int main(int argc, char** argv) {
    // Declare and parse arguments to their variables
    unsigned long long m;
    if (argc < 2) {
        printf("Usage: %s <vector_dimension>\n", argv[0]);
        return -1;
    }
    sscanf(argv[1], "%llu", &m);

    // Declare our timer variables
    Stopwatch timer;
    double local_time;

    // Seed the random number generator
    seed_random();

    // Dynamically allocate memory for matrix and vector
    double* matrix = (double*) malloc(m * m * sizeof(double));
    double* vector = (double*) malloc(m * sizeof(double));

    if (!matrix || !vector) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Fill the matrix and vector with random numbers
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            matrix[i * m + j] = random_double();
        }
    }
    
    for (int i = 0; i < m; i++) {
        vector[i] = random_double();
    }

    // Start timer
    clock_gettime(CLOCK_MONOTONIC, &timer.start);

    // Multiply the matrix by the vector
    double *result = matrix_vector_product(matrix, vector, m);

    // Stop timer
    clock_gettime(CLOCK_MONOTONIC, &timer.stop);
    local_time = (timer.stop.tv_sec - timer.start.tv_sec) + (timer.stop.tv_nsec - timer.start.tv_nsec) / 1e9;

    // Print results
    printf("Serial Function\t%s\nVector Dimension\t%llu\nTime\t\t\t%lf seconds\n\n", "Matrix-Vector Product", m, local_time);

    // Free dynamically allocated memory
    free(matrix);
    free(vector);
    free(result);

    return 0;
}
