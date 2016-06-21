#include <stdio.h>
#include <omp.h>


#define NUM_THREADS 4
static long num_steps = 100000000;
double step;

double cal_pi()
{
    double pi = 0;
    step = 1./num_steps;
    double sum[NUM_THREADS];
#pragma omp parallel
    {
        double x;
        int id = omp_get_thread_num();
        sum[id] = 0;
        for (int i = id; i < num_steps; i += NUM_THREADS) {
            x = (i + 0.5) * step;
            sum[id] += 4. / (1. + x * x);
        }
    }
    for (int i = 0; i < NUM_THREADS; i++)
        pi += sum[i] * step;
    return pi;

}

int main(int argc, char *argv[])
{
    omp_set_num_threads(NUM_THREADS);
    double pi = cal_pi();
    printf("pi = %.12lf\n", pi);
    return 0;
}


