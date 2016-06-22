#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <omp.h>
#include "config.h"
#include "../common/stopwatch.h"

/*#define M(row, col)  *(M.elements + (row) (*) M.width + col)*/
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;


//a h w  B h w    C 
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < B.width; j++) {
            C.elements[i * C.width + j] = 0;
            for (int k = 0; k < A.width; k++) {
                C.elements[i * C.width + j] +=
                    A.elements[i * A.width + k]
                    * B.elements[k * B.width + j];
            }
        }
    }
}


void fill_Matrix(Matrix A)
{
    for (int i = 0; i < A.height ; i++) {
        for (int j = 0; j < A.width; j++) {
            A.elements[i * A.width + j] = rand() / (float)RAND_MAX * 10;
        }
    }
}

void print_Matrix(Matrix A)
{
    /*for (int i = 0; i < A.height; i++) {*/
        /*for (int j = 0; j < A.width; j++) {*/
            /*printf("%4.1f ", A.elements[i * A.width + j]);*/
        /*}*/
        /*printf("\n");*/
    /*}*/
}

int main(int argc, char **argv)
{
//    if (argc != 2) {
 //       printf("usage: n\n");
  //      return -1;
   // }
    //int nnn = atoi(argv[1]);

    //int n = 1 << nnn;
    int n = N;

    srand(time(0));

    Matrix A, B, C;
    A.width = A.height = n;
    A.elements = (float *)malloc(sizeof(float) * n * n);
    B.width = B.height = n;
    B.elements = (float *)malloc(sizeof(float) * n * n);
    C.width = C.height = n;
    C.elements = (float *)malloc(sizeof(float) * n * n);


    fill_Matrix(A);
    //print_Matrix(A);
    //printf("\n");

    fill_Matrix(B);
    //print_Matrix(B);
    //printf("\n");


    stopwatch_restart();
    MatMul(A, B, C);
    printf("time = %llu us \n", (long long unsigned)stopwatch_record());

    //print_Matrix(C);

}
