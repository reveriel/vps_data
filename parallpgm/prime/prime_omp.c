#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

int primes[10000 * 1000];


int count_primes(int n);

int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("usage : prime <n>\n");
        return -1;
    }
    int n = atoi(argv[1]);
    assert(n >= 2);
    int cnt = count_primes(n);
    printf("cnt = %d\n", cnt);
    return 0;
}


int is_prime(int n)
{
    if (n % 2 == 0) return 0;
    int bound = (int)ceil(sqrt(n));
    for (int i = 3; i < bound; i+= 2) {
        if (n % i == 0)
            return 0;
    }
    return 1;
}

// return the number of primes from 2 to n
int count_primes(int n)
{
    int cnt = 1;
#pragma omp parallel for
    for (int i = 2; i <= n; i ++) {
        if (is_prime(i)) {
            cnt++;
        }
    }
    return cnt;
}

