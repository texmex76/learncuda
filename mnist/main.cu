#include "curand_kernel.h"
#include <__clang_cuda_math.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

__global__ void add(int n, float *a, float *b, float *c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

__global__ void forward(int batch_size, int n, int out_w, float *input,
                        float *weights, float *biases, float *output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < batch_size && col < out_w) {
    output[row * out_w + col] = biases[col];
    for (int i = 0; i < n; i++) {
      output[row * out_w + col] +=
          weights[i * out_w + col] * input[row * n + i];
    }
  }
}

__global__ void relu(int w, int h, float *input, float *output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && col < w) {
    float activation = input[row * w + col];
    output[row * w + col] = activation > 0.f ? activation : 0.f;
  }
}

__global__ void softmax(int w, int h, float *input, float *output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && col < w) {
    float maxval = input[row * w];
    for (int i = 1; i < w; i++) {
      maxval = max(maxval, input[row * w + i]);
    }
    float divisor = 0.f;
    for (int i = 0; i < w; i++) {
      divisor += exp(input[row * w + i] - maxval);
    }
    output[row * w + col] = exp(input[row * w + col] - maxval) / divisor;
  }
}

__global__ void cross_entropy(int w, int h, float *preds, float *real,
                              float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < h) {
    float loss = 0.f;
    for (int i = 0; i < w; i++) {
      loss -= real[idx * w + i] * log(max(1e-6, preds[idx * w + i]));
    }
    output[idx] = loss;
  }
}

__global__ void init_rand(int w, int h, float *mat) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && col < w) {
    curandState state;
    curand_init(42, row * w + col, 0, &state);
    mat[row * w + col] = curand_uniform(&state) * sqrtf(2.f / h);
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
