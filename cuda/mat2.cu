#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// thread block size
#define BLOCK_SIZE 16



// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}


// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);






// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
/*
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);


    cudaError_t err = cudaThreadSynchronize();
    printf("Run kernel: %s\n", cudaGetErrorString(err));

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

*/
void MatMul(const Matrix A, const Matrix B, Matrix C) { 
  // Load A and B to device memory 
  Matrix d_A; 
  d_A.width = d_A.stride = A.width; 
  d_A.height = A.height; 
  size_t size = A.width * A.height * sizeof(float); 
  cudaError_t err = cudaMalloc(&d_A.elements, size); 
  printf("CUDA malloc A: %s\n",cudaGetErrorString(err)); 
  err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice); 
  printf("Copy A to device: %s\n",cudaGetErrorString(err)); 

  Matrix d_B; 
  d_B.width = d_B.stride = B.width; 
  d_B.height = B.height; 
  size = B.width * B.height * sizeof(float); 
  err = cudaMalloc(&d_B.elements, size); 
  printf("CUDA malloc B: %s\n",cudaGetErrorString(err));
  err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
  printf("Copy B to device: %s\n",cudaGetErrorString(err)); 

  // Allocate C in device memory 
  Matrix d_C; 
  d_C.width = d_C.stride = C.width; 
  d_C.height = C.height; 
  size = C.width * C.height * sizeof(float); 
  err = cudaMalloc(&d_C.elements, size); 
  printf("CUDA malloc C: %s\n",cudaGetErrorString(err));

  // Invoke kernel 
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y); 
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C); 
    err = cudaThreadSynchronize();
    printf("Run kernel: %s\n", cudaGetErrorString(err));

  // Read C from device memory 
  err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost); 
  printf("Copy C off of device: %s\n",cudaGetErrorString(err));

  // Free device memory
  cudaFree(d_A.elements); 
  cudaFree(d_B.elements); 
  cudaFree(d_C.elements); 
}

// Matrix multiplication kernel called by MatMul()
/*
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}
*/

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) { 
  // Block row and column 
  int blockRow = blockIdx.y; 
  int blockCol = blockIdx.x; 

  // Each thread block computes one sub-matrix Csub of C
  Matrix Csub = GetSubMatrix(C, blockRow, blockCol); 

  // Each thread computes one element of Csub 
  // by accumulating results into Cvalue 
  float Cvalue = 0.0; 

  // Thread row and column within Csub 
  int row = threadIdx.y; 
  int col = threadIdx.x; 

  // Loop over all the sub-matrices of A and B that are 
  // required to compute Csub 
  // Multiply each pair of sub-matrices together 
  // and accumulate the results 
  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
    // Get sub-matrix Asub of A 
    Matrix Asub = GetSubMatrix(A, blockRow, m); 

    // Get sub-matrix Bsub of B 
    Matrix Bsub = GetSubMatrix(B, m, blockCol); 

    // Shared memory used to store Asub and Bsub respectively 
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE]; 
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE]; 

    // Load Asub and Bsub from device memory to shared memory 
    // Each thread loads one element of each sub-matrix 
    As[row][col] = GetElement(Asub, row, col); 
    Bs[row][col] = GetElement(Bsub, row, col); 

    // Synchronize to make sure the sub-matrices are loaded 
    // before starting the computation 
    __syncthreads(); 

    // Multiply Asub and Bsub together 
    for (int e = 0; e < BLOCK_SIZE; ++e) 
      Cvalue += As[row][e] * Bs[e][col];
 
    // Synchronize to make sure that the preceding 
    // computation is done before loading two new 
    // sub-matrices of A and B in the next iteration 
    __syncthreads();  
  }

  // Write Csub to device memory 
  // Each thread writes one element 
  SetElement(Csub, row, col, Cvalue); 
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
    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < A.width; j++) {
            printf("%4.1f ", A.elements[i * A.width + j]);
        }
        printf("\n");
    }
}



int main(void)
{

    srand(time(0));
    int n = 1 << 3;
    Matrix A, B, C;
    A.width = A.height = n;
    A.elements = (float *)malloc(sizeof(float) * n * n);
    B.width = B.height = n;
    B.elements = (float *)malloc(sizeof(float) * n * n);
    C.width = C.height = n;
    C.elements = (float *)malloc(sizeof(float) * n * n);

    fill_Matrix(A);
    print_Matrix(A);
    printf("\n");

    fill_Matrix(B);
    print_Matrix(B);
    printf("\n");

    MatMul(A, B, C);

    print_Matrix(C);

}
