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

// mpicc -lm q4.c -o q4
// mpirun -n 2 ./q4 2500000 25
// mpirun -n 4 ./q4 2500000 25
// mpirun -n 8 ./q4 2500000 25
// mpirun -n 16 ./q4 2500000 25

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#define TOLERANCE 0.0001
#define PARALLELIZABLE_FRACTION 0.99  // Assumed parallelizable portion for Amdahl's law

// Struct to store start and stop times
typedef struct {
    struct timespec start;
    struct timespec stop;
} Stopwatch;

/** Check if a number is prime */
int is_prime(int num) {
    if (num < 2) return 0;
    if (num % 2 == 0) return num == 2;  // 2 is the only even prime
    for (int i = 3; i <= sqrt(num); i += 2) {
        if (num % i == 0) return 0;
    }
    return 1;
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

// Comparison function for qsort (ascending order)
int compare_ints(const void* a, const void* b) {
    int int_a = *((int*)a);
    int int_b = *((int*)b);
    return (int_a > int_b) - (int_a < int_b);  // Returns -1, 0, or 1 for sorting
}

/**
 * Main function
 * @param argc: Number of arguments
 * @param argv: Array of arguments
 * @return: 0 if successful
 */
int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv); // Initialize MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    // Handle exit conditions
    if (argc < 3) {
        if (rank == 0) {
            printf("Usage: %s <upper bound n> <num_runs>\n", argv[0]);
        }
        MPI_Finalize();
        return -1;
    }

    // Parse arguments
    unsigned int n;
    int num_runs;
    sscanf(argv[1], "%u", &n);
    sscanf(argv[2], "%d", &num_runs);

    // Variables for tracking time
    double total_parallel_time = 0;
    double total_serial_time = 0;

    for (int run = 0; run < num_runs; run++) {
        Stopwatch parallel_timer, serial_timer;

        // Start timing for parallel execution
        clock_gettime(CLOCK_MONOTONIC, &parallel_timer.start);

        // Each node will gather primes it finds in its portion
        int* primes_local = (int*) malloc(n * sizeof(int));
        int prime_count = 0;

        // Interleaved work distribution across nodes
        for (int i = 3 + 2 * rank; i <= n; i += 2 * size) {
            if (is_prime(i)) {
                primes_local[prime_count++] = i;
                if (run == num_runs -1) {
                    printf("Node %d found prime: %d\n", rank, i);
                }
            }
        }

        // Collect results to rank 0
        int* primes_global = NULL;
        int* recvcounts = NULL;
        int* displs = NULL;

        if (rank == 0) {
            primes_global = (int*) malloc(n * sizeof(int));
            recvcounts = (int*) malloc(size * sizeof(int));
            displs = (int*) malloc(size * sizeof(int));

            primes_global[0] = 2;  // 2 is the only even prime
        }

        // Gather counts of primes from each process
        MPI_Gather(&prime_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        /** Set up displacements for gathering the actual primes into primes_global on rank 0
          * I'm still not entirely comfortable with this concept, but after a while I made it work.
          * q1 and q3 both had this put method at one point, but I was able to work around it and
          * find a concept that worked.  Here is the sole implementations using recvcounts and displs.
          */
        if (rank == 0) {
            int offset = 0;
            for (int i = 0; i < size; i++) {
                displs[i] = offset;
                offset += recvcounts[i];
            }
        }

        // Gather all primes to rank 0
        MPI_Gatherv(primes_local, prime_count, MPI_INT, primes_global, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

        // Stop timing for parallel execution
        clock_gettime(CLOCK_MONOTONIC, &parallel_timer.stop);
        total_parallel_time += calculate_time(parallel_timer);

        // Let's sort the prime numbers and write them to a file, as a treat for me
        if (rank == 0) {
            if (run == num_runs - 1) {
                // Sort primes before writing to file
                int total_primes = displs[size - 1] + recvcounts[size - 1];
                qsort(primes_global, total_primes, sizeof(int), compare_ints);

                // Write sorted primes to file primes_to_n.txt
                FILE *fp;
                char filename[30];
                sprintf(filename, "primes_to_%d.txt", n);
                fp = fopen(filename, "w");

                if (fp == NULL) {
                    fprintf(stderr, "Error opening file %s for writing.\n", filename);
                    exit(EXIT_FAILURE);
                }

                // Add 2 to the file
                fprintf(fp, "%s", "2 ");

                for (int i = 0; i < total_primes; i++) {
                    fprintf(fp, "%d ", primes_global[i]);
                }

                fclose(fp);

                // Free dynamically allocated memory on rank 0
                free(primes_global);
                free(recvcounts);
                free(displs);
            }
        }
        
        // Free dynamically allocated memory on all nodes
        free(primes_local);

        // Start timing for serial execution
        if (rank == 0) {
            clock_gettime(CLOCK_MONOTONIC, &serial_timer.start);

            int* serial_primes = (int*) malloc(n * sizeof(int));
            int serial_prime_count = 0;

            for (int i = 3; i <= n; i += 2) {
                if (is_prime(i)) {
                    serial_primes[serial_prime_count++] = i;
                }
            }

            clock_gettime(CLOCK_MONOTONIC, &serial_timer.stop);
            total_serial_time += calculate_time(serial_timer);

            free(serial_primes);
        }
        if (run == num_runs -1) {
            // Calculate average times
            double average_parallel_time = total_parallel_time / num_runs;
            double average_serial_time = total_serial_time / num_runs;
            // Output results and speedup calculations on rank 0
            if (rank == 0) {
                double actual_speedup = average_serial_time / average_parallel_time;
                double theoretical_speedup = amdahl_speedup(size);
                double speedup_ratio = (actual_speedup / theoretical_speedup) * 100;

                printf("Number of Primes Found to %d:\t%d\n", n, displs[size - 1] + recvcounts[size - 1] + 1);
                printf("Average Serial Time:\t\t\t%lfs\n", average_serial_time);
                printf("Average Parallel Time:\t\t\t%lfs\n", average_parallel_time);
                printf("Theoretical Speedup [Amdahl's Law]:\t%lf\n", theoretical_speedup);
                printf("Actual Speedup:\t\t\t\t%lf\n", actual_speedup);
                printf("Speedup Efficiency:\t\t\t%lf%%\n", speedup_ratio);
            }
        }
    }

    MPI_Finalize(); // Finalize MPI environment
    return 0;
}