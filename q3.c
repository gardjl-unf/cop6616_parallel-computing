/**
 * Author: Jason Gardner
 * Date: 10/1/2024
 * Class: COP6616 Parallel Computing
 * Instructor: Scott Piersall
 * Assignment: Homework 2
 * Filename: q3.c
 * 
 * THIS IS THE FULL IMPLEMENTATION OF QUESTION 1
 * 
 * Description:
 * 3. (35 points)   Computing Euclidean Distance Using OpenMP or MPI
 *                      In N-dimensional space, the Euclidean distance between two points P(p1, p2, p3,…, pn) and Q(q1, q2, q3,…, qn) is calculated as follows,  
 *                      √((p1-q1)^2+(p2-q2)^2+⋯+(pn-qn)^2 )
 * 
 *                      Please write an OpenMP or MPI program to compute the Euclidean distance between two N-dimensional points, where N is at least 1 million. 
 *                      Please initialize your vectors (denotes the two points) with random integers within a small range (0 to 99). This computation consists 
 *                      of a fully data-parallel phase (computing the square of the difference of the components for each dimension), a reduction (adding 
 *                      together all of these squares), and finally taking the square root of the reduced sum.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>

// mpicc q3.c -o q3
// mpirun -n 32 ./q3 1000000 100

#define min 0
#define max 99

// Struct to store start and stop times
typedef struct {
    struct timespec start;
    struct timespec stop;
    double time;
} Stopwatch;

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
 * Generate a random integer between MIN and MAX
 * @return: A random integer between MIN and MAX
 */
int generate_random_int() {
    return min + rand() % (max - min + 1);
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
    int* vector1 = (int*) malloc(m * sizeof(int));
    int* vector2 = (int*) malloc(m * sizeof(int));
    int result;

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