#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>

// mpicc q1m.c -o q1m
// mpirun -n 32 ./q1m 25000 100

#define MAX 100000.0

// Struct to store start and stop times
typedef struct {
    struct timespec start;
    struct timespec stop;
    double time;
} Stopwatch;

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
    int rank, size;
    MPI_Init(&argc, &argv);  // Initialize MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get the total number of processes

    unsigned int m;
    int num_runs;
    if (argc < 3) {
        if (rank == 0) {
            printf("Usage: %s <vector dimension> <number of runs to average>\n", argv[0]);
        }
        MPI_Finalize();
        return -1;
    }

    sscanf(argv[1], "%d", &m);
    sscanf(argv[2], "%d", &num_runs);

    // Allocate memory for the vector (this will be used by all processes)
    int* vector = (int*) malloc(m * sizeof(int));
    if (!vector) {
        fprintf(stderr, "Memory allocation failed: vector!\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Root process initializes the matrix and the vector
    int* matrix = NULL;
    int* result = NULL;

    if (rank == 0) {
        matrix = (int*) malloc(m * m * sizeof(int));
        result = (int*) malloc(m * sizeof(int));

        if (!matrix || !result) {
            fprintf(stderr, "Memory allocation failed: matrix/result!\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        // Seed the random number generator and fill the matrix and vector
        seed_random();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                matrix[i * m + j] = rand();
            }
        }
        for (int i = 0; i < m; i++) {
            vector[i] = rand();
        }
    }

    // Broadcast the vector to all processes
    MPI_Bcast(vector, m, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine rows per process and handle uneven row distribution
    int rows_per_proc = m / size;  // Base number of rows each process will handle
    int remainder = m % size;  // Number of extra rows to distribute

    /** Create send_counts and displacements arrays for scattering:
     * https://www.mpich.org/static/docs/latest/www3/MPI_Scatterv.html
     * I struggled with this during the presentation and I found this stackoverflow post that 
     * helped me understand how to scatter unevenly.
     * https://stackoverflow.com/questions/9269399/sending-blocks-of-2d-array-in-c-using-mpi
     */
    int* send_counts = (int*) malloc(size * sizeof(int));
    int* displs = (int*) malloc(size * sizeof(int));
    int* recv_counts = (int*) malloc(size * sizeof(int));

    int displacement = 0;
    for (int i = 0; i < size; i++) {
        send_counts[i] = (i < remainder) ? (rows_per_proc + 1) * m : rows_per_proc * m;
        recv_counts[i] = (i < remainder) ? (rows_per_proc + 1) : rows_per_proc;
        displs[i] = displacement;
        displacement += send_counts[i];
    }

    // Allocate memory for the local matrix and local result
    int local_rows = recv_counts[rank];  // Number of rows for this process
    int* local_matrix = (int*) malloc(local_rows * m * sizeof(int));
    int* local_result = (int*) malloc(local_rows * sizeof(int));

    // Scatter the rows of the matrix to each process
    MPI_Scatterv(matrix, send_counts, displs, MPI_INT, local_matrix, local_rows * m, MPI_INT, 0, MPI_COMM_WORLD);

    // Run the matrix-vector product multiple times
    Stopwatch stopwatches;
    double total_local_time = 0.0;

    for (int run = 0; run < num_runs; run++) {
        // Start timer for each iteration
        clock_gettime(CLOCK_MONOTONIC, &stopwatches.start);

        // Perform the local matrix-vector multiplication
        local_matrix_vector_product(local_matrix, vector, local_result, local_rows, m);

        // Stop timer for each iteration
        clock_gettime(CLOCK_MONOTONIC, &stopwatches.stop);
        
        // Accumulate local time
        total_local_time += calculate_time(stopwatches);
    }

    // Calculate average local time over all iterations
    double avg_local_time = total_local_time / num_runs;

    // Gather the average local times to the root process
    double* all_times = NULL;
    if (rank == 0) {
        all_times = (double*) malloc(size * sizeof(double));
    }
    MPI_Gather(&avg_local_time, 1, MPI_DOUBLE, all_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Root process calculates the total and average times across all processes
    if (rank == 0) {
        double total_time = 0.0;
        for (int i = 0; i < size; i++) {
            total_time += all_times[i];
        }
        double avg_time = total_time / size;
        
        // Print the results
        printf("MPI Matrix-Vector Product\nIterations:\t\t%d\nVector Dimension:\t%d\nNumber of Processes:\t%d\nAverage Time:\t\t%lf seconds\n\n", num_runs, m, size, avg_time);

        free(all_times);
        free(matrix);
        free(result);
    }

    // Free allocated memory
    free(local_matrix);
    free(local_result);
    free(vector);
    free(send_counts);
    free(displs);
    free(recv_counts);

    MPI_Finalize();  // Finalize the MPI environment
    return 0;
}