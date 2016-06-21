#include <stdio.h>
#include <omp.h>

#define NUM_THREADS 4

static long num_steps = 100000000;
double step;

double cal_pi()
{
    double x, pi, sum = 0;
    step = 1./num_steps;
/*#pragma omp parallel for reduction(+:sum)*/ // wrong
#pragma omp parallel for private(x) reduction(+:sum)
    for (int i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum += 4. / (1. + x * x);
    }
    return pi = step * sum;
}

int main(int argc, char *argv[])
{
    omp_set_num_threads(NUM_THREADS);
    double pi = cal_pi();
    printf("pi = %.12lf\n", pi);
    return 0;
}


