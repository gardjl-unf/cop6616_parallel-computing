/**
 * Author: Jason Gardner
 * Date: 10/1/2024
 * Class: COP6616 Parallel Computing
 * Instructor: Scott Piersall
 * Assignment: Homework 2
 * Filename: q1m.c
 * 
 * THIS IS THE FULL IMPLEMENTATION OF QUESTION 2
 * THIS CODE IS NOT COMPILABLE!
 * 
 * Description:
 * 2.   (4 points)  For each of the following code segments, use OpenMP pragmas to make the loop parallel, 
 * or explain why the code segment is not suitable for parallel execution. You do not need to write 
 * executable C programs, please just add valid openMP pragmas. The code can be modified with keeping 
 * the same semantics for adding openMP pragmas.
 */

int main(int argc, char** argv) {

int* a; 
int* b;
int n, flag, x, p, k;

(1);
#pragma omp parallel for
for (int i=0; i < (int)sqrt(x); i++) {
    a[i] = 2.3 * i;
    if (i < 10) b[i] = a[i];
}

(2);
/** Because the flag variable can terminate execution of the loop based on a condition dependent on 
 *  prior iterations of the loop, it is not possible to parallelize this loop.
 * 
 *  I did find that using shared and flush(flag) would allow the loop to be 
 *  parallelized, but it is not a good practice.  The downsides are:
 * 
 *  High Overhead:  Frequent synchronization of flag can be costly
 *  Poor Performance:  Can waste computation, as loops in progress when flag is set to 1 will continue to run
 * 
 *  It also doesn't fit the requirements, as it requires a rework of the loop to be parallelized.
 *  Still, it was nifty to see how it could be done.  I hope the right answer is that it can't be parallelized.
 * 
 *  volatile int flag = 0;
 *  #pragma omp parallel for shared(flag) 
 *  for (i = 0; i < n; i++) {
 *      if (!flag) {  // Only proceed if flag is unset
 *         a[i] = 2.3 * i;
 *         
 *          if (a[i] < b[i]) {
 *              #pragma omp atomic write // Ensures that the write to flag is atomic (one process)
 *              flag = 1;
 *          }
 * 
 *          #pragma omp flush(flag)  // Flushes the value of flag to ensure that all threads see the same value
 *      }
 *  
 *      #pragma omp flush(flag)  // Flushes the value of flag to ensure that all threads see the same value
 *  }
 * 
 */
flag = 0;
for (int i = 0; (i < n) && (!flag); i++){
   a[i] = 2.3 * i;
   if (a[i] < b[i]) {
        flag = 1;
   }
}

(3);
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    a[i] = foo(i);
}

(4);
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    a[i] = foo(i);
    if (a[i] < b[i]) {
        a[i] = b[i];
    }
}

(5);
/**  Because the loop can be exited via the break statement that is dependent on the value of a[i] and b[i] with
 *  a[i] being calculated in an external function, it is not possible to parallelize this loop.
 * 
 *  Again, I found that using shared and flush(flag) or tasks to share the flag would allow the loop to be
 *  parallelized, but it is not a good practice as previously written.
 * 
 *  It requires a slight rework of the loop to be parallelized.
 * 
 *  volatile int flag = 0;
 * 
 *  #pragma omp parallel for shared(flag)
 *  for (int i = 0; i < n; i++) {
 *      if (!flag) {  // Only proceed if flag is unset
 *          a[i] = foo(i);
 * 
 *          if (a[i] < b[i]) {
 *              #pragma omp atomic write  // Ensures that the write to flag is atomic (one process)
 *              flag = 1;
 *          }
 *      }
 *      #pragma omp flush(flag)  // Ensure all threads see updated flag
 *  }
 * 
 */
for (int i = 0; i < n; i++){
    a[i] = foo(i);
    if (a[i] < b[i]) {
        break;
    }
}

(6);
int p = 0;
#pragma omp parallel for reduction(+:p)
for (int i = 0; i < n; i++) {
    p += a[i] * b[i];
}

(7);
/** Because the calculation for a[i] is dependent on a[i - k], it is not possible to parallelize this loop.
 * 
 * Working through the loop:
 * i starts at k and goes to 2 * k - 1, incrementing by 1
 * a[i] = a[i] + a[i-k]
 * 
 * When i = k, a[k] = a[k] + a[0]
 * When i = k + 1, a[k + 1] = a[k + 1] + a[1]
 * When i = k + 2, a[k + 2] = a[k + 2] + a[2]
 * ...
 * 
 * There is a linear dependency between the values of a[i] and a[i - k] that prevents parallelization.
 * There is no guarentee that a[i - k] will be calculated before a[i] in the loop.
 */
for (int i = k; i < 2 * k; i++){
    a[i] = a[i] + a[i-k];
}

(8);
#pragma omp parallel for
for (int i = k; i < n; i++) {
    a[i] = b * a[i - k];
}

}