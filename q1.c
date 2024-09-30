#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>

// gcc -lm q1.c -o q1

#define MAX 100000.0

// Struct to store start and stop times
typedef struct {
    struct timespec start;
    struct timespec stop;
    double time;
} Stopwatch;

/**
 * Multiplies a matrix by a vector where the matrix has dimensions m x m and the vector has dimensions m x 1
 * @param matrix: The matrix to multiply
 * @param vector: The vector to multiply
 * @param m: The dimensions of the matrix and vector
 * @return: The result of the matrix-vector multiplication
 */
void matrix_vector_product(double* matrix, double* vector, double* result, int m) {
    memset(result, 0, m * sizeof(double));  // Zero out the result array

    // Perform matrix-vector multiplication
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            result[i] += matrix[i * m + j] * vector[j];
        }
    }
}

/** Seed the random number generator with entropy from /dev/urandom */
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

double calculate_time(Stopwatch timer) {
    return (timer.stop.tv_sec - timer.start.tv_sec) + (timer.stop.tv_nsec - timer.start.tv_nsec) / 1e9;
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
    int num_runs;
    if (argc < 3) {
        printf("Usage: %s <vector dimension> <number of runs to average>\n", argv[0]);
        return -1;
    }
    sscanf(argv[1], "%llu", &m);
    sscanf(argv[2], "%d", &num_runs);

    // Dynamically allocate memory for matrix, vector, and result
    double* matrix = (double*) malloc(m * m * sizeof(double));
    double* vector = (double*) malloc(m * sizeof(double));
    double* result = (double*) malloc(m * sizeof(double));

    if (!matrix || !vector || !result) {
        fprintf(stderr, "Memory allocation failed: matrices/vector!\n");
        exit(EXIT_FAILURE);
    }

    // Seed the random number generator
    seed_random();

    // Fill the matrix with random doubles between 0 and MAX
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            matrix[i * m + j] = random_double();
        }
    }

    // Fill the vector with random doubles between 0 and MAX
    for (int i = 0; i < m; i++) {
        vector[i] = random_double();
    }

    // Declare our timer variables
    Stopwatch *timers = (Stopwatch*) malloc(num_runs * sizeof(Stopwatch));
    if (!timers) {
        fprintf(stderr, "Memory allocation failed: timers!\n");
        exit(EXIT_FAILURE);
    }

    // Perform matrix-vector multiplication num_runs times
    for (int i = 0; i < num_runs; i++) {
        // Start timer
        clock_gettime(CLOCK_MONOTONIC, &timers[i].start);

        // Multiply the matrix by the vector, using the pre-allocated result array
        matrix_vector_product(matrix, vector, result, m);

        // Stop timer
        clock_gettime(CLOCK_MONOTONIC, &timers[i].stop);
        timers[i].time = calculate_time(timers[i]);
    }

    // Calculate the average time taken
    double avg_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
        avg_time += timers[i].time;
    }
    avg_time /= num_runs;

    // Print results
    printf("Serial Matrix-Vector Product\nVector Dimension:\t%llu\nAverage Time:\t%lf seconds\n\n", m, avg_time);

    // Free dynamically allocated memory
    free(matrix);
    free(vector);
    free(result);
    free(timers);

    return 0;
}
