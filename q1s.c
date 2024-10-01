/**
 * Author: Jason Gardner
 * Date: 10/1/2024
 * Class: COP6616 Parallel Computing
 * Instructor: Scott Piersall
 * Assignment: Homework 2
 * Filename: q1s.c
 * 
 * THIS IS HALF OF THE REQUIRED IMPLEMENTATION FOR QUESTION 1
 * 
 * Description:
 * 1. (25 points)   Please implement an MPI program in C, C++, OR Python to calculate the
 *                  multiplication of a matrix and a vector (Using MPI Scatter and Gather). Specifically,
 * 
 *                  the MPI program can be implemented in the following way:
 *                      1.  Implement a Serial code solution to compute the multiplication of a matrix and a vector.
 *                          This is the basis for your computation of Speedup provided by parallelization. (You
 *                      should lookup the definition of Speedup in parallel computing carefully)
 *                      2. According to the input argument (the size of the vector) from main() function, generate
 *                          a matrix and a vector with random integer values, where the column size of matrix should
 *                          be equal to the size of the vector.
 *                      3. SCATTER: According to the number of processes from the input argument, split
 *                          the matrix into chunks (row-wise) with roughly equal size, then distribute chunks to all
 *                          processes using “scatter”. Additionally, the vector can be broadcasted to all processes.
 *                      4. Conduct product for the chunk of matrix and vector.
 *                      5. GATHER: The final result is collected on the master node using “gather”.
 *                      6. VERIFY CORRECTNESS: Make sure the result of your MPI code matches the
 *                          results of your serial code.
 *                      7. SPEEDUP: What is the speedup S of your approach? (Speedup is a specific
 *                          measurement and you should report it correctly) Combine the answer to this with your
 *                          experiments in the next step
 *                      8. EXPERIMENTS: You should run experiments consisting of running problem set
 *                          sizes over varying number of compute nodes. Discuss the relationship between increasing
 *                          the number of nodes to Speedup S in your submitted PDF
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>

// gcc q1s.c -o q1s
// ./q1s 25000 100

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
    // Check for correct number of arguments
    if (argc < 3) {
        printf("Usage: %s <vector dimension> <number of runs to average>\n", argv[0]);
        return -1;
    }

    // Declare and parse arguments to their variables
    unsigned int m;
    int num_runs;

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

    // Seed the random number generator and fill the matrix and vector
    seed_random();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            matrix[i * m + j] = rand();
        }
        vector[i] = rand();
    }

    // Declare our timer variables
    Stopwatch *stopwatches = (Stopwatch*) malloc(num_runs * sizeof(Stopwatch));
    if (!stopwatches) {
        fprintf(stderr, "Memory allocation failed: timers!\n");
        exit(EXIT_FAILURE);
    }

    // Perform matrix-vector multiplication num_runs times
    for (int i = 0; i < num_runs; i++) {
        clock_gettime(CLOCK_MONOTONIC, &stopwatches[i].start); // Start timer

        // Multiply the matrix by the vector, using the pre-allocated result array
        matrix_vector_product(matrix, vector, result, m, m);

        
        clock_gettime(CLOCK_MONOTONIC, &stopwatches[i].stop); // Stop timer
        stopwatches[i].time = calculate_time(stopwatches[i]); // Calculate time taken
    }

    // Calculate the average time taken across all runs
    double avg_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
        avg_time += stopwatches[i].time;
    }
    avg_time /= num_runs;

    // Print results
    printf("Serial Matrix-Vector Product\nIterations:\t\t%d\nVector Dimension:\t%d\nAverage Time:\t\t%lf seconds\n\n", num_runs, m, avg_time);

    // Free allocated memory
    free(matrix);
    free(vector);
    free(result);
    free(stopwatches);

    return 0;
}
