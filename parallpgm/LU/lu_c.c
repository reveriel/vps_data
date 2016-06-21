#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../common/stopwatch.h"
#include "config.h"



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
    // assume L[i][i] == 1
    // #pragma omp parallel for
    // for (size_t j = 0; j < N; j++){
    //     for (size_t i = 0; i < N; i++){
    //         if(i <= j){
    //             double sum = 0;
    //             for (int k = 0; k < i; k++)
    //                 sum += a[i][k] * a[k][j];
    //             a[i][j] -= sum;
    //         }
    //         if(i > j){
    //             double sum = 0;
    //             for (int k = 0; k < j; k++)
    //                 sum += a[i][k] * a[k][j];
    //             a[i][j] = (a[i][j] - sum) / a[j][j];
    //         }
    //     }
    // }
    //

    //LU-decomposition based on Gaussian Elimination
    // - Arranged so that the multiplier doesn't have to be computed multiple times
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

    printf("time = %llu us\n", stopwatch_record());

    // print_mat((double*)a);

    return 0;

}
