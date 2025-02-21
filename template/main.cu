#include <cstdlib>
#include <iostream>
#include <stdexcept>

__global__ void add(int n, float *a, float *b, float *c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
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
    int N = 4096;

    float *a = (float *)malloc(N * sizeof(float));
    float *b = (float *)malloc(N * sizeof(float));
    float *c = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
      a[i] = (float)rand() / (float)RAND_MAX;
      b[i] = (float)rand() / (float)RAND_MAX;
    }

    float *a_d;
    float *b_d;
    float *c_d;
    cudaSafeCall(cudaMalloc((void **)&a_d, N * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&b_d, N * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&c_d, N * sizeof(float)));

    cudaSafeCall(cudaMemcpy(a_d, a, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(b_d, b, N * sizeof(float), cudaMemcpyHostToDevice));

    int BLOCK_SIZE = 1024;
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add<<<gridSize, BLOCK_SIZE>>>(N, a_d, b_d, c_d);
    cudaSafeCall(cudaPeekAtLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    cudaSafeCall(cudaMemcpy(c, c_d, N * sizeof(float), cudaMemcpyDeviceToHost));

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
