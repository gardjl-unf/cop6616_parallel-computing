# cop6616_parallel-computing

*runonce*
echo "PATH=/cm/shared/apps/mvapich2/gcc/64/2.3.7/bin/:$PATH" >> ~/.bash_profile
source ~/.bash_profile

# Question 1
Compile:    mpicc q1c.c -o q1c
Run:        mpirun -n <num_nodes> ./q1c <matrix_dimension> <num_runs>

# Question 3
Compile:    mpicc -lm q3.c -o q3
Run:        mpirun -n <num_nodes> ./q3 <vector_dimension> <num_runs>

# Question 4
Compile:    mpicc -lm q4.c -o q4
Run:        mpirun -n <num_nodes> ./q4 <to_num> <num_runs>
