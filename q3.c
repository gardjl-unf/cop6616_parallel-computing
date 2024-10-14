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

// mpicc -lm q3.c -o q3
// mpirun -n 32 ./q3 1000000 100

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define MIN 0
#define MAX 99
#define TOLERANCE 0.0001
#define PARALLELIZABLE_FRACTION 0.95

/** Struct to store start and stop times */
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

/** Calculate the theoretical speedup using Amdahl's law */
double amdahl_speedup(int p) {
    return 1.0 / ((1.0 - PARALLELIZABLE_FRACTION) + (PARALLELIZABLE_FRACTION / p));
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

/** Function to compare serial and parallel results with a tolerance */
_Bool compare_result(double result_s, double result_m) {
    return fabs(result_s - result_m) < TOLERANCE;
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

    int num_runs;
    unsigned int m;
    double result_m, result_s;
    sscanf(argv[1], "%d", &m);
    sscanf(argv[2], "%d", &num_runs);

    // Allocate vector memory on all processes
    int* vector1 = (int*) malloc(m * sizeof(int));
    int* vector2 = (int*) malloc(m * sizeof(int));

    if (!vector1 || !vector2) {
        fprintf(stderr, "Memory allocation failed on rank %d!\n", rank);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Root process initializes the vectors
    if (rank == 0) {
        seed_random();
        for (int i = 0; i < m; i++) {
            vector1[i] = random_vector_value();
            vector2[i] = random_vector_value();
        }
    }

    // Variables for tracking time
    double total_parallel_time = 0;
    double total_serial_time = 0;

    // Perform the Euclidean distance calculation for num_runs
    for (int run = 0; run < num_runs; run++) {
        Stopwatch parallel_timer, serial_timer;

        // Start parallel timing
        clock_gettime(CLOCK_MONOTONIC, &parallel_timer.start);

        // Broadcast vectors to all processes
        MPI_Bcast(vector1, m, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(vector2, m, MPI_INT, 0, MPI_COMM_WORLD);

        int rows_per_proc = m / size;
        int remainder = m % size;
        int local_rows = (rank < remainder) ? rows_per_proc + 1 : rows_per_proc;

        int* local_vector1 = (int*) malloc(local_rows * sizeof(int));
        int* local_vector2 = (int*) malloc(local_rows * sizeof(int));

        if (!local_vector1 || !local_vector2) {
            fprintf(stderr, "Memory allocation failed for rank %d!\n", rank);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        // Scatter portions of the vectors to each process manually
        if (rank == 0) {
            int offset = local_rows;
            for (int r = 1; r < size; r++) {
                int rows_to_send = (r < remainder) ? (rows_per_proc + 1) : rows_per_proc;
                MPI_Send(vector1 + offset, rows_to_send, MPI_INT, r, 0, MPI_COMM_WORLD);
                MPI_Send(vector2 + offset, rows_to_send, MPI_INT, r, 0, MPI_COMM_WORLD);
                offset += rows_to_send;
            }
            memcpy(local_vector1, vector1, local_rows * sizeof(int));
            memcpy(local_vector2, vector2, local_rows * sizeof(int));
        } else {
            MPI_Recv(local_vector1, local_rows, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(local_vector2, local_rows, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Perform parallel Euclidean distance calculation
        double local_result = euclidean_distance(local_vector1, local_vector2, local_rows);

        // Reduce partial results to the root process
        MPI_Reduce(&local_result, &result_m, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            result_m = sqrt(result_m);
        }

        // Stop parallel timing
        clock_gettime(CLOCK_MONOTONIC, &parallel_timer.stop);
        total_parallel_time += calculate_time(parallel_timer);

        // Free local vectors
        free(local_vector1);
        free(local_vector2);

        // Start serial timing (only on root process)
        if (rank == 0) {
            clock_gettime(CLOCK_MONOTONIC, &serial_timer.start);

            // Perform serial Euclidean distance calculation
            result_s = euclidean_distance(vector1, vector2, m);
            result_s = sqrt(result_s);

            clock_gettime(CLOCK_MONOTONIC, &serial_timer.stop);
            total_serial_time += calculate_time(serial_timer);

            // Compare results after each run
            if (!compare_result(result_s, result_m)) {
                printf("Results do not match in run %d!\n", run + 1);
            }
        }
    }

    // Calculate average times
    double average_parallel_time = total_parallel_time / num_runs;
    double average_serial_time = total_serial_time / num_runs;

    // Output results and speedup calculations
    if (rank == 0) {
        double actual_speedup = average_serial_time / average_parallel_time;
        double theoretical_speedup = amdahl_speedup(size);
        double speedup_ratio = (actual_speedup / theoretical_speedup) * 100;

        printf("Average Serial Time:\t\t\t%lfs\n", average_serial_time);
        printf("Average Parallel Time:\t\t\t%lfs\n", average_parallel_time);
        printf("Theoretical Speedup [Amdahl's Law]:\t%lf\n", theoretical_speedup);
        printf("Actual Speedup:\t\t\t\t%lf\n", actual_speedup);
        printf("Speedup Efficiency:\t\t\t%lf%%\n", speedup_ratio);
    }

    // Free allocated memory
    free(vector1);
    free(vector2);

    MPI_Finalize(); // Finalize MPI environment
    return 0;
}