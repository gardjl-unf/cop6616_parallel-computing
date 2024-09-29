#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

// gcc test_randd.c -o test_randd

#define MAX 100000.0

double random_double() {
    return ((double) random() / (double) RAND_MAX * MAX);
}

void seed_random() {
    int fd = open("/dev/urandom", O_RDONLY);
    unsigned int seed;
    read(fd, &seed, sizeof(seed));
    close(fd);
    srandom(seed);
}

int main(int argc, char** argv) {
    seed_random();
    for (int i = 0; i < 10; i++) {
        printf("%lf\n", random_double());
    }
    return 0;
}