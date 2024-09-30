#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>

// gcc q1s.c -o q1s
// ./q1s 25000 100

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
 * @param result: The result of the multiplication
 * @param rows: The number of rows in the matrix
 * @param cols: The number of columns in the matrix
 * @return: The result of the matrix-vector multiplication
 */
void matrix_vector_product(int* matrix, int* vector, int* result, int rows, int cols) {
    memset(result, 0, rows * sizeof(int));  // Zero out the result array

    // Perform matrix-vector multiplication
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * rows + j] * vector[j];
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

/** Calculate time in seconds
 * @param timer: The stopwatch struct
 * @return: The time in seconds
 */
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
    unsigned int m;
    int num_runs;

    if (argc < 3) {
        printf("Usage: %s <vector dimension> <number of runs to average>\n", argv[0]);
        return -1;
    }

    sscanf(argv[1], "%i", &m);
    sscanf(argv[2], "%d", &num_runs);

    // Dynamically allocate memory for matrix, vector, and result
    int* matrix = (int*) malloc(m * m * sizeof(int));
    int* vector = (int*) malloc(m * sizeof(int));
    int* result = (int*) malloc(m * sizeof(int));

    if (!matrix || !vector || !result) {
        fprintf(stderr, "Memory allocation failed: matrices/vector!\n");
        exit(EXIT_FAILURE);
    }

    // Seed the random number generator
    seed_random();

    // Fill the matrix with random integers between 0 and RAND_MAX
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            matrix[i * m + j] = rand();
        }
    }

    // Fill the vector with random integers between 0 and RAND_MAX
    for (int i = 0; i < m; i++) {
        vector[i] = rand();
    }

    // Declare our timer variables
    Stopwatch *timers = (Stopwatch*) malloc(num_runs * sizeof(Stopwatch));
    if (!timers) {
        fprintf(stderr, "Memory allocation failed: timers!\n");
        exit(EXIT_FAILURE);
    }

    // Perform matrix-vector multiplication num_runs times
    for (int i = 0; i < num_runs; i++) {
        clock_gettime(CLOCK_MONOTONIC, &timers[i].start); // Start timer

        // Multiply the matrix by the vector, using the pre-allocated result array
        matrix_vector_product(matrix, vector, result, m, m);

        
        clock_gettime(CLOCK_MONOTONIC, &timers[i].stop); // Stop timer
        timers[i].time = calculate_time(timers[i]); // Calculate time taken
    }

    // Calculate the average time taken across all runs
    double avg_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
        avg_time += timers[i].time;
    }
    avg_time /= num_runs;

    // Print results
    printf("Serial Matrix-Vector Product\nIterations:\t\t%d\nVector Dimension:\t%d\nAverage Time:\t\t%lf seconds\n\n", num_runs, m, avg_time);

    // Free allocated memory
    free(matrix);
    free(vector);
    free(result);
    free(timers);

    return 0;
}
