/**
 * Author: Jason Gardner
 * Date: 10/1/2024
 * Class: COP6616 Parallel Computing
 * Instructor: Scott Piersall
 * Assignment: Homework 2
 * Filename: q4.c
 * 
 * THIS IS THE FULL IMPLEMENTATION OF QUESTION 1
 * 
 * 4. (35 points)  Find prime numbers using MPI. 
 *                     Develop an MPI program to find and print the first n prime numbers in the range [0…n]. 
 *                     Run your code using 2,4,8, and 16 nodes. 
 *                     The MPI communication methods you choose are up to you, however….you MUST distribute the work evenly among the nodes, 
 *                     not perform redundant or unnecessary calculations AND you should automatically skip even numbers. For your experiment, 
 *                    use the value of 2500000 for n. Your program should print out each prime number found (and the node that found it), 
 *                    and the master node (rank 0) should output the time elapsed for your program to run. Run the experiment for 2,4,8, 
 *                    and 16 nodes. Graph your results. Include the graph and a paragraph interpreting and explaining your experimental 
 *                    results in the PDF file report. Include your source code in the ZIP file for turn in.
 * 
 *                  Hints: There are a variety of ways to implement a solution to question 4. My hints are
 *                  •	Do not assign contiguous blocks of numbers to each node. That does NOT distribute the work evenly among the nodes. 
 *                      hink about this hint carefully. If you do not understand why assigning a contiguous block of numbers to each node 
 *                      is incorrect for this problem, please come see me to discuss….seriously. Solutions submitted which just assign 
 *                      contiguous blocks of numbers to nodes will receive zero credit
 *                  •	MPI_Reduce is your friend for this problem……..
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>

// mpicc q4.c -o q4
// mpirun -n 32 ./q4 25000

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
        local_result[i] = 0;
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

/**
 * Main function
 * @param argc: Number of arguments
 * @param argv: Array of arguments
 * @return: 0 if successful
 */
int main(int argc, char** argv) {
    int rank = 0, size = 1;

    MPI_Init(&argc, &argv); // Initialize MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    if (argc < 3) {
        if (rank == 0) {
            printf("Usage: %s <vector dimension> <number of runs to average>\n", argv[0]);
        }
        MPI_Finalize();
        return -1;
    }

    // Parse arguments
    int num_runs;
    unsigned int m;
    sscanf(argv[1], "%d", &m);
    sscanf(argv[2], "%d", &num_runs);

    // Allocate memory for vector and results
    int* vector = (int*) malloc(m * sizeof(int));
    int* result_s = NULL;
    int* result_m = NULL;
    int* matrix = NULL;

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

    // Determine the number of rows for each process
    int rows_per_proc = m / size;
    int remainder = m % size;
    int local_rows = (rank < remainder) ? rows_per_proc + 1 : rows_per_proc;

    // Allocate memory for local matrix and local result
    int* local_matrix = (int*) malloc(local_rows * m * sizeof(int));
    int* local_result = (int*) malloc(local_rows * sizeof(int));

    if (!local_matrix || !local_result) {
        fprintf(stderr, "Memory allocation failed for rank %d!\n", rank);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Manually send the matrix to each process
    if (rank == 0) {
        int offset = local_rows * m;  // The starting point for rank 0
        for (int r = 1; r < size; r++) {
            int rows_to_send = (r < remainder) ? (rows_per_proc + 1) : rows_per_proc;
            MPI_Send(matrix + offset, rows_to_send * m, MPI_INT, r, 0, MPI_COMM_WORLD);
            offset += rows_to_send * m;
        }
        memcpy(local_matrix, matrix, local_rows * m * sizeof(int));
    } else {
        // Non-root processes receive their portion of the matrix
        MPI_Recv(local_matrix, local_rows * m, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Perform parallel matrix-vector multiplication
    Stopwatch stopwatches[2];
    clock_gettime(CLOCK_MONOTONIC, &stopwatches[PARALLEL].start);
    for (int run = 0; run < num_runs; run++) {
        local_matrix_vector_product(local_matrix, vector, local_result, local_rows, m);
    }
    clock_gettime(CLOCK_MONOTONIC, &stopwatches[PARALLEL].stop);
    double total_parallel_time = calculate_time(stopwatches[PARALLEL]) / num_runs;

    // Rank 0 gathers the results from all processes
    int* recvcounts = (int*) malloc(size * sizeof(int));  // Array to store the number of elements to gather from each process
    int* displs = (int*) malloc(size * sizeof(int));      // Array to store the displacements for gathering
    int offset = 0;

    for (int r = 0; r < size; r++) {
        recvcounts[r] = (r < remainder) ? (rows_per_proc + 1) : rows_per_proc;
        recvcounts[r] *= 1;  // Multiply by 1 for 1D vector gathering
        displs[r] = offset;
        offset += recvcounts[r];
    }

    MPI_Gatherv(local_result, local_rows, MPI_INT, result_m, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // Serial run and comparison
    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &stopwatches[SERIAL].start);
        for (int run = 0; run < num_runs; run++) {
            matrix_vector_product(matrix, vector, result_s, m, m);
        }
        clock_gettime(CLOCK_MONOTONIC, &stopwatches[SERIAL].stop);
        double total_serial_time = calculate_time(stopwatches[SERIAL]) / num_runs;

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

    // Free allocated memory
    free(matrix);
    free(vector);
    free(result_s);
    free(result_m);
    free(local_matrix);
    free(local_result);
    free(recvcounts);
    free(displs);

    MPI_Finalize(); // Finalize MPI environment
    return 0;
}