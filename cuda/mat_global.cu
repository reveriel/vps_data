#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*#define M(row, col)  *(M.elements + (row) (*) M.width + col)*/
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C);

#define BLOCK_SIZE 16

/*

void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    cudaError_t err;
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size  = A.width * A.height * sizeof(float);
    err =cudaMalloc(&d_A.elements, size);
    printf("malloc A: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_A.elements, A.elements, size,
            cudaMemcpyHostToDevice);
    printf("Copy A to device: %s\n",cudaGetErrorString(err));

    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    err =cudaMalloc(&d_B.elements, size);
    printf("malloc B: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_B.elements, B.elements, size,
            cudaMemcpyHostToDevice);
    printf("Copy B to device : %s\n", cudaGetErrorString(err));

    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    err = cudaMalloc(&d_C.elements, size);
    printf("malloc C : %s\n",cudaGetErrorString(err));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);


    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    cudaThreadSynchronize();

     err = cudaMemcpy(C.elements, d_C.elements, size,
            cudaMemcpyDeviceToHost);

    printf("Copy C off of device: %s\n",cudaGetErrorString(err));
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
} */



// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) {

    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaError_t err = cudaMalloc(&d_A.elements, size);
    /*printf("CUDA malloc A: %s\n",cudaGetErrorString(err));*/
    err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    /*printf("Copy A to device: %s\n",cudaGetErrorString(err));*/

    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    err = cudaMalloc(&d_B.elements, size);

    /*printf("CUDA malloc B: %s\n",cudaGetErrorString(err));*/
    err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    /*printf("Copy B to device: %s\n",cudaGetErrorString(err));*/
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    err = cudaMalloc(&d_C.elements, size);
    /*printf("CUDA malloc C: %s\n",cudaGetErrorString(err));*/

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x,
            (A.height + dimBlock.y - 1) / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    err = cudaThreadSynchronize();
    /*printf("Run kernel: %s\n", cudaGetErrorString(err));*/

    // Read C from device memory
    err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    /*printf("Copy C off of device: %s\n",cudaGetErrorString(err));*/

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    // cudaFree(d_C.elements);
}


/*

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
*/


// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
  // Each thread computes one element of C
  // by accumulating results into Cvalue
  float Cvalue = 0.0;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if(row > A.height || col > B.width) return;
  for (int e = 0; e < A.width; ++e)
    Cvalue += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]);
  C.elements[row * C.width + col] = Cvalue;
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
/*
    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < A.width; j++) {
            printf("%4.1f ", A.elements[i * A.width + j]);
        }
        printf("\n");
    }
    */
}

int main(int argc, char ** argv)
{

    if (argc != 2) {
        printf("usage: n\n");
        return -1;
    }
    int nnn = atoi(argv[1]);

    int n = 1 << nnn;
    srand(time(0));

    Matrix A, B, C;
    A.width = A.height = n;
    A.elements = (float *)malloc(sizeof(float) * n * n);
    B.width = B.height = n;
    B.elements = (float *)malloc(sizeof(float) * n * n);
    C.width = C.height = n;
    C.elements = (float *)malloc(sizeof(float) * n * n);


    fill_Matrix(A);
    /*print_Matrix(A);*/
    /*printf("\n");*/

    fill_Matrix(B);
    /*print_Matrix(B);*/
    /*printf("\n");*/

    MatMul(A, B, C);

    /*print_Matrix(C);*/

}
