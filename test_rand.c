#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

// gcc test_rand.c -o test_rand

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
        printf("%d\n", rand());
    }
    return 0;
}