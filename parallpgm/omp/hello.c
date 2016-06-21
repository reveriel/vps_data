#include <stdio.h>
#include <omp.h>



int a[1000];
int n = 1000;

void pooh(int id, int *a, int n)
{
    for (int i = 0; i < n; i++)
        a[i] = id;
}

int main()
{
    omp_set_num_threads(4);

#pragma omp parallel
    {
        int id = 0;
        id = omp_get_thread_num();
        pooh(id, a, n);
    }
    printf("all done\n");
    for (int i = 0; i < n; i++ )
        printf("%d\n", a[i]);
    return 0;
}

