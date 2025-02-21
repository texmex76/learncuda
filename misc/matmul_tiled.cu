#include <cstdlib>
#include <iostream>
#include <stdexcept>

#define TILE_WIDTH 8

// Kernel for computing C = A * B
// A is an M x K matrix, B is a K x N matrix, and C is an M x N matrix.
__global__ void matmul_tiled(int M, int K, int N, float *A, float *B,
                             float *C) {
  // Allocate shared memory for a tile (submatrix) of A and B.
  __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
  __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

  // Compute the row and column indices of the C element this thread will
  // compute.
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  // Each thread will accumulate one element of C.
  float sum = 0.0f;

  // The total number of tiles we need to cover the K dimension.
  // We use integer division with rounding up.
  int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

  // Loop over the tiles along the K dimension.
  for (int t = 0; t < numTiles; t++) {
    // Index for the column of A that we need to load into shared memory.
    int A_col = t * TILE_WIDTH + threadIdx.x;
    // Each thread loads one element from A if it's within bounds.
    if (row < M && A_col < K)
      A_tile[threadIdx.y][threadIdx.x] = A[row * K + A_col];
    else
      A_tile[threadIdx.y][threadIdx.x] =
          0.0f; // Zero padding for out-of-bound threads.

    // Index for the row of B that we need to load.
    int B_row = t * TILE_WIDTH + threadIdx.y;
    // Each thread loads one element from B if within bounds.
    if (B_row < K && col < N)
      B_tile[threadIdx.y][threadIdx.x] = B[B_row * N + col];
    else
      B_tile[threadIdx.y][threadIdx.x] =
          0.0f; // Zero padding for out-of-bound threads.

    // Synchronize threads to ensure the tile is fully loaded before computing.
    __syncthreads();

    // Each thread computes a partial sum for its element in C by iterating over
    // the tile.
    for (int i = 0; i < TILE_WIDTH; i++) {
      sum += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
    }

    // Synchronize before loading the next tile.
    __syncthreads();
  }

  // After processing all tiles, write the result to C if the thread is within
  // C's bounds.
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

#define cudaSafeCall(call)                                                     \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(error) << std::endl;                     \
      throw std::runtime_error("CUDA Error");                                  \
    }                                                                          \
  } while (0)

int main(void) {
  try {
    int M = 200;
    int K = 150;
    int N = 409;

    float *a = (float *)malloc(M * K * sizeof(float));
    float *b = (float *)malloc(K * N * sizeof(float));
    float *c = (float *)malloc(M * N * sizeof(float));
    float *c_verify = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * K; i++) {
      a[i] = (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
      b[i] = (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        c_verify[i * N + j] = 0.0;
        for (int k = 0; k < K; k++) {
          c_verify[i * N + j] += a[i * K + k] * b[k * N + j];
        }
      }
    }

    float *a_d;
    float *b_d;
    float *c_d;
    cudaSafeCall(cudaMalloc((void **)&a_d, M * K * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&b_d, K * N * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&c_d, M * N * sizeof(float)));

    cudaSafeCall(
        cudaMemcpy(a_d, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(
        cudaMemcpy(b_d, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                   (M + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_tiled<<<numBlocks, threadsPerBlock>>>(M, K, N, a_d, b_d, c_d);
    cudaSafeCall(cudaPeekAtLastError());
    cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(
        cudaMemcpy(c, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; i++) {
      if (fabsf(c[i] - c_verify[i]) > 1e-6) {
        throw std::runtime_error("Matrices do not match.");
      }
    }
    cudaSafeCall(cudaFree(a_d));
    cudaSafeCall(cudaFree(b_d));
    cudaSafeCall(cudaFree(c_d));
    free(a);
    free(b);
    free(c);
    cudaDeviceReset();
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
