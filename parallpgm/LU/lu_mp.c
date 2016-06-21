#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "config.h"

#include "../common/stopwatch.h"


double a[N][N];


#define A(i,j) (*((double*)a + (i) * N + (j)))

void fill_mat(double *a) {
    srand(time(0));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(i,j) = rand() / 12338272.;
        }
    }
}

void print_mat(double *a) {
    for (int i =0; i< N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%5.2f ", A(i,j));
        }
        printf("\n");
    }
}



int main(void) {

    fill_mat((double*)a);
    // print_mat((double*)a);

    stopwatch_restart();
    // omp_set_num_threads(4);
    for(int k = 0; k < N-1; k++){ //iterate over rows/columns for elimination
        // The "multiplier" is the factor by which a row is multiplied when
        //  being subtracted from another row.
        for(int row = k + 1; row < N; row++){
            // the multiplier only depends on (k,row),
            // it is invariant with respect to col
            double factor = a[row][k]/a[k][k];

            //Eliminate entries in sub (subtract rows)
            for(int col = k + 1; col < N; col++){ //column
                a[row][col] = a[row][col] - factor*a[k][col];
            }

            a[row][k] = factor;
        }
    }

    printf("c :time = %llu us\n", stopwatch_record());


    //////////////////////////////////////////////////////////////
    stopwatch_restart();

    // omp_set_num_threads(4);
    for(int k = 0; k < N-1; k++){ //iterate over rows/columns for elimination
        // The "multiplier" is the factor by which a row is multiplied when
        //  being subtracted from another row.
        for(int row = k + 1; row < N; row++){
            a[row][k] /= a[k][k];
        }
        // the multiplier only depends on (k,row),
        // it is invariant with respect to col

        //Eliminate entries in sub (subtract rows)
        #pragma omp parallel for shared(a,k)
        for (int i = k + 1; i < N; i++) {
            const double aik = a[i][k];
            for (int j = k + 1; j < N; j++) {
                a[i][j] -= aik * a[k][j];
            }
        }
    }

    printf("mp:time = %llu us\n", stopwatch_record());

    // print_mat((double*)a);

    return 0;

}
