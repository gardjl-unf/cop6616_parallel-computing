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
#include <math.h>

// mpicc q3.c -o q3
// mpirun -n 32 ./q3 1000000 100

#define MIN 0
#define MAX 99

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

/** Calculate the theoretical speedup using Amdahl's law
 * @param p: The number of processors
 * @param f_parallel: The fraction of the program that can be parallelized
 * @return: The theoretical speedup
 */
double amdahl_speedup(int p, double f_parallel) {
    return 1.0 / ((1.0 - f_parallel) + (f_parallel / p));
}

/**
 * Calculate the partial Euclidean distance between two vectors
 * @param p: The first vector
 * @param q: The second vector
 * @param n: The number of dimensions
 * @return: The partial Euclidean distance
 */
double euclidean_distance(int* p, int* q, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (p[i] - q[i]) * (p[i] - q[i]);
    }
    return sum;
}

/**
 * Calculate a random vector value between MIN and MAX
 * @return: A random vector value between MIN and MAX
 */
int random_vector_value() {
    return MIN + rand() % (MAX - MIN + 1);
}

/**
 * Main function
 * @param argc: Number of arguments
 * @param argv: Array of arguments
 * @return: 0 if successful
 */
int main(int argc, char** argv) {
    int rank = 0, size = 1;
    
    // Parse arguments
    int num_runs;
    unsigned int m;
    double result_m;
    sscanf(argv[1], "%d", &m);
    sscanf(argv[2], "%d", &num_runs);

    MPI_Init(&argc, &argv); // Initialize MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    // Handle exit conditions
    if (rank == 0) {
        if (argc < 3) {
            printf("Usage: %s <vector dimension> <number of runs to average>\n", argv[0]);
        if (m < 1000000) {
            printf("Vector dimension must be at least 1 million (SEE REQUIREMENTS).\n");
        }
        MPI_Finalize();
        return -1;
    }

    // Allocate memory for vector and results
    int* vector1 = (int*) malloc(m * sizeof(int));
    int* vector2 = (int*) malloc(m * sizeof(int));
    double result_s;

    if (rank == 0) {
        double result_s;

        if (!vector1 || !vector2) {
            fprintf(stderr, "Memory allocation failed!\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        // Seed the random number generator and fill the vectors with random values between MIN and MAX
        seed_random();
        for (int i = 0; i < m; i++) {
            vector1[i] = random_vector_value();
            vector2[i] = random_vector_value();
        }
    }

    // Determine the number of rows for each process
    int rows_per_proc = m / size;
    int remainder = m % size;
    int local_rows = (rank < remainder) ? rows_per_proc + 1 : rows_per_proc;

    // Allocate memory for local vectors
    int* local_vector1 = (int*) malloc(local_rows * sizeof(int));
    int* local_vector2 = (int*) malloc(local_rows * sizeof(int));

    if (!local_vector1 || !local_vector2) {
        fprintf(stderr, "Memory allocation failed for rank %d!\n", rank);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    double local_result = 0.0;

    // Manually send the matrix to each process
    if (rank == 0) {
        int offset = local_rows * m;  // The starting point for rank 0
        for (int r = 1; r < size; r++) {
            int rows_to_send = (r < remainder) ? (rows_per_proc + 1) : rows_per_proc;
            MPI_Send(vector1 + offset, rows_to_send * m, MPI_INT, r, 0, MPI_COMM_WORLD);
            MPI_Send(vector2 + offset, rows_to_send * m, MPI_INT, r, 0, MPI_COMM_WORLD);
            offset += rows_to_send * m;
        }
        memcpy(local_vector1, vector1, local_rows * sizeof(int));
        memcpy(local_vector2, vector2, local_rows * sizeof(int));
    } else {
        // Non-root processes receive their portion of the matrix
        MPI_Recv(local_vector1, local_rows, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_vector2, local_rows, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Perform parallel matrix-vector multiplication
    Stopwatch stopwatches[2];
    clock_gettime(CLOCK_MONOTONIC, &stopwatches[PARALLEL].start);
    for (int run = 0; run < num_runs; run++) {
        local_result = euclidean_distance(local_vector1, local_vector2, local_rows);
    }
    clock_gettime(CLOCK_MONOTONIC, &stopwatches[PARALLEL].stop);
    double total_parallel_time = calculate_time(stopwatches[PARALLEL]) / num_runs;

    // Gather the result values from each process
    MPI_Reduce(&local_result, &result_m, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    result_m = sqrt(result_m);

    // Serial run and comparison
    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &stopwatches[SERIAL].start);
        for (int run = 0; run < num_runs; run++) {
            result_s = euclidean_distance(vector1, vector2, m);
        }
        result_s = sqrt(result_s);
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
    free(vector1);
    free(vector2);
    free(local_vector1);
    free(local_vector2);

    MPI_Finalize(); // Finalize MPI environment
    return 0;
}