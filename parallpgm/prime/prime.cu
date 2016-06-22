#include <stdio.h>
//#include <math.h>
#include <stdlib.h>
#include <assert.h>


#include <cuda_runtime.h>

#include "config.h"
#include "../common/stopwatch.h"



char primes[N+10] = {0,0,1,1};


int count_primes(int n);

int main(int argc, char *argv[])
{
    //if (argc != 2) {
        //printf("usage : prime <n>\n");
        //return -1;
    //}
    //int n = atoi(argv[1]);
    int n = N;
    assert(n >= 2);
    stopwatch_restart();
    int cnt = count_primes(n);
    printf("time = %llu us\n", (long long unsigned)stopwatch_record());
    printf("cnt = %d\n", cnt);
    return 0;
}



__global__ void
is_prime_g(char *a, int n)
{
    // i is prime?
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        if (i % 2 == 0) {
            a[i] = 0;
            return;
        }
        int bound = (int)(sqrt((float)i));
        for (int j = 3; j <= bound; j += 2) {
            if (i % j == 0) {
                a[i] = 0;
                return;
            }
        }
        a[i] = 1;
    }
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
    char *d_a;
    int size = n * sizeof(char);
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&d_a, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failde to allocate device mem d_a (error code %s)\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_a, primes, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy from primes to d_a,(error code %s)\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock -1) / threadsPerBlock;
    is_prime_g<<<blocksPerGrid, threadsPerBlock>>>(d_a,n);

    err = cudaMemcpy(primes, d_a, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Faild to copy from d_a to prims, (error code %s)\n",
                cudaGetErrorString(err));
    }

    int cnt = 0;
    for (int i = 0; i <= n; i ++) {
        cnt += primes[i];
    }
    return cnt;
}

