#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*#define M(row, col)  *(M.elements + (row) (*) M.width + col)*/
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// thread block size
#define BLOCK_SIZE 16

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);


void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size  = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
            cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
            cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C.elements, d_C.elements, size,
            cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}


__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
     float Cvalue = 0;
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     for (int e = 0;  e < A.width; ++e)
         Cvalue += A.elements[row * A.width + e]
             * B.elements[e * B.width + col];
     C.elements[row * C.width + col] = Cvalue;
}



void init_Matrix(Matrix **ppA, int n)
{
    *ppA = (Matrix *)malloc(sizeof(Matrix));
    (*ppA)->width = (*ppA)->height = n;
    (*ppA)->elements = (float *)malloc(n * n * sizeof(float));
}

void fill_Matrix(Matrix *pA)
{
    srand(time(0));
    for (int i = 0; i < pA->height ; i++) {
        for (int j = 0; j < pA->width; j++) {
            pA->elements[i * pA->height + j] = rand() / RAND_MAX * 10;
        }
    }
}

void free_Matrix(Matrix *pA )
{
    free(pA->elements);
    free(pA);
}

void print_Matrix(Matrix *pA)
{
    for (int i = 0; i < pA->height; i++) {
        for (int j = 0; j < pA->width; j++) {
            printf("%4.1f ", pA->elements[i * pA->height + j]);
        }
        printf("\n");
    }
}

int main(void)
{
    int n = 1 << 3;
    Matrix *pA, *pB, *pC;
    init_Matrix(&pA, n);
    init_Matrix(&pB, n);
    init_Matrix(&pC, n);
    fill_Matrix(pA);
    print_Matrix(pA);
    printf("\n");
    fill_Matrix(pB);
    print_Matrix(pB);
    printf("\n");

    MatMul(*pA, *pB, *pC);

    MatMul(*pA, *pB, *pC);

    print_Matrix(pC);

}
