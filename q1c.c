/**
 * Author: Jason Gardner
 * Date: 10/1/2024
 * Class: COP6616 Parallel Computing
 * Instructor: Scott Piersall
 * Assignment: Homework 2
 * Filename: q1c.c
 * 
 * THIS IS THE FULL IMPLEMENTATION OF QUESTION 1
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
#include <mpi.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>

// mpicc q1c.c -o q1c
// mpirun -n 32 ./q1c 25000 100

// Struct to store start and stop times
typedef struct {
    struct timespec start;
    struct timespec stop;
    double time;
} Stopwatch;

/** Enum for accessing stopwatch indices for clarity */
enum {
    SERIAL = 0,
    PARALLEL = 1
};

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
 * Multiplies a matrix by a vector where the matrix has dimensions m x m and the vector has dimensions m x 1
 * @param matrix: The matrix to multiply
 * @param vector: The vector to multiply
 * @param result: The result of the multiplication
 * @param rows: The number of rows in the matrix
 * @param cols: The number of columns in the matrix
 */
void matrix_vector_product(int* matrix, int* vector, int* result, int rows, int cols) {
    memset(result, 0, rows * sizeof(int));  // Zero out the result array
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

/**
 * Multiplies a sub-matrix (local to each process) by a vector
 * @param local_matrix: The sub-matrix (local to each process)
 * @param vector: The full vector
 * @param local_result: The result of the multiplication (local to each process)
 * @param rows: The number of rows in the sub-matrix
 * @param cols: The number of columns in the matrix (same as the length of the vector)
 */
void local_matrix_vector_product(int* local_matrix, int* vector, int* local_result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            local_result[i] += local_matrix[i * cols + j] * vector[j];
        }
    }
}

/** Calculate the theoretical speedup using Amdahl's law
 * @param p: The number of processors
 * @param f_parallel: The fraction of the program that can be parallelized
 * @return: The theoretical speedup
 */
double amdahl_speedup(int p, double f_parallel) {
    return 1.0 / ((1.0 - f_parallel) + (f_parallel / p));
}

int main(int argc, char** argv) {
    int rank = 0, size = 1;
    // Check for correct number of arguments
    if (argc < 3) {
        if (rank == 0) {
            printf("Usage: %s <vector dimension> <number of runs to average>\n", argv[0]);
        }
        return -1;
    }

    // Declare and parse arguments to their variables
    int num_runs;
    unsigned int m;

    MPI_Init(&argc, &argv); // Initialize MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes    

    sscanf(argv[1], "%d", &m);
    sscanf(argv[2], "%d", &num_runs);

    // Allocate memory for matrix, vector, and results
    int* matrix = NULL;
    int* vector = (int*) malloc(m * sizeof(int));
    int* result_s = NULL;
    int* result_m = NULL;

    if (!vector) {
        fprintf(stderr, "Memory allocation failed: vector!\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (rank == 0) {
        matrix = (int*) malloc(m * m * sizeof(int));
        result_s = (int*) malloc(m * sizeof(int));
        result_m = (int*) malloc(m * sizeof(int));

        if (!matrix || !result_s || !result_m) {
            fprintf(stderr, "Memory allocation failed!\n");
            MPI_Finalize();
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
    }

    // Broadcast the vector to all processes
    MPI_Bcast(vector, m, MPI_INT, 0, MPI_COMM_WORLD);

    Stopwatch stopwatches[2]; // One for SERIAL, one for PARALLEL
    double total_serial_time = 0.0;
    double total_parallel_time = 0.0;

    if (rank == 0) {
        // Serial run
        clock_gettime(CLOCK_MONOTONIC, &stopwatches[SERIAL].start);
        for (int run = 0; run < num_runs; run++) {
            matrix_vector_product(matrix, vector, result_s, m, m);
        }
        clock_gettime(CLOCK_MONOTONIC, &stopwatches[SERIAL].stop);
        total_serial_time = calculate_time(stopwatches[SERIAL]) / num_runs;
    }

    // Parallel run
    int rows_per_proc = m / size;
    int remainder = m % size;
    int local_rows = (rank < remainder) ? rows_per_proc + 1 : rows_per_proc;
    int* local_matrix = (int*) malloc(local_rows * m * sizeof(int));
    int* local_result = (int*) malloc(local_rows * sizeof(int));

    // Scatter the matrix to all processes
    MPI_Scatter(matrix, local_rows * m, MPI_INT, local_matrix, local_rows * m, MPI_INT, 0, MPI_COMM_WORLD);

    clock_gettime(CLOCK_MONOTONIC, &stopwatches[PARALLEL].start);
    for (int run = 0; run < num_runs; run++) {
        local_matrix_vector_product(local_matrix, vector, local_result, local_rows, m);
    }
    clock_gettime(CLOCK_MONOTONIC, &stopwatches[PARALLEL].stop);
    total_parallel_time = calculate_time(stopwatches[PARALLEL]) / num_runs;

    // Gather results from all processes
    MPI_Gather(local_result, local_rows, MPI_INT, result_m, local_rows, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Compare results_s and result_m
        if (memcmp(result_s, result_m, m * sizeof(int)) == 0) {
            printf("Results match.\n");
        } else {
            printf("Results do not match!\n");
        }

        // Calculate actual and theoretical speedup
        double actual_speedup = total_serial_time / total_parallel_time;
        double theoretical_speedup = amdahl_speedup(size, 0.9); // Assuming 90% of the code is parallelizable

        printf("Serial Time: %lf\nParallel Time: %lf\nActual Speedup: %lf\nTheoretical Speedup (Amdahl's Law): %lf\n",
                total_serial_time, total_parallel_time, actual_speedup, theoretical_speedup);
    }

    free(matrix);
    free(vector);
    free(result_s);
    free(result_m);
    free(local_matrix);
    free(local_result);

    MPI_Finalize(); // Finalize MPI environment
    return 0;
}