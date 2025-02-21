#include "curand_kernel.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

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

#define TILE_WIDTH 8
__global__ void matmul(int w, int h, float *a, float *b, float *c) {
  __shared__ float a_tile[TILE_WIDTH][TILE_WIDTH];
  __shared__ float b_tile[TILE_WIDTH][TILE_WIDTH];
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  float dot_prod = 0.f;

  int max_dim = w > h ? w : h;
  for (int tile_offset = 0; tile_offset < max_dim; tile_offset += TILE_WIDTH) {
    int a_chk = (tile_offset + tx) < w && row < h;
    a_tile[ty][tx] = a_chk ? a[row * w + tile_offset + tx] : 0.f;
    int b_chk = (tile_offset + ty) < h && col < w;
    b_tile[ty][tx] = b_chk ? b[(tile_offset + ty) * h + col] : 0.f;

    __syncthreads();
    for (int i = 0; i < TILE_WIDTH; i++) {
      dot_prod += a_tile[ty][i] * b_tile[i][tx];
    }
    __syncthreads();
  }
  if (row < h && col < w) {
    c[row * w + col] = dot_prod;
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
      maxval = fmaxf(maxval, input[row * w + i]);
    }
    float divisor = 0.f;
    for (int i = 0; i < w; i++) {
      divisor += expf(input[row * w + i] - maxval);
    }
    output[row * w + col] = expf(input[row * w + col] - maxval) / divisor;
  }
}

__global__ void cross_entropy(int w, int h, float *preds, float *real,
                              float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < h) {
    float loss = 0.f;
    for (int i = 0; i < w; i++) {
      loss -= real[idx * w + i] * logf(fmaxf(1e-6f, preds[idx * w + i]));
    }
    output[idx] = loss;
  }
}

__global__ void init_weights(int w, int h, float *mat) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && col < w) {
    curandState state;
    curand_init(42, row * w + col, 0, &state);
    mat[row * w + col] = (curand_uniform(&state) * 2.f - 1.f) * sqrtf(2.f / w);
  }
}

__global__ void init_biases(int n, float *mat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    curandState state;
    curand_init(42, idx, 0, &state);
    mat[idx] = curand_uniform(&state) * 0.01f;
  }
}

__global__ void cross_entropy_backwards(int w, int h, float *preds, float *real,
                                        float *output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && col < w) {
    output[row * w + col] = (preds[row * w + col] - real[row * w + col]) / h;
  }
}

__global__ void backwards(int batch_size, int n, int out_w, float *weights,
                          float *biases, float *d_l, float *out_d_l) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < batch_size && col < out_w) {
    float dl = 0.f;
    for (int i = 0; i < n; i++) {
      float w_val = weights[i * out_w + col];
      dl += w_val * d_l[row * n + i];
    }
    out_d_l[row * out_w + col] = dl;
  }
}

__global__ void relu_backwards(int w, int h, float *a, float *d_l, float *b) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && col < w) {
    float activation = a[row * w + col];
    b[row * w + col] = activation > 0.f ? d_l[row * w + col] : 0.f;
  }
}

__global__ void update_layer(int w, int h, int batch_size, float lr,
                             float *weights, float *biases, float *activations,
                             float *d_l) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && col < w) {
    float dw = 0.f;
    float db = 0.f;
    for (int i = 0; i < batch_size; i++) {
      float act = activations[i * h + row];
      float dl = d_l[i * w + col];
      dw += act * dl;
      db += dl;
    }
    weights[row * w + col] -= lr * dw / batch_size;
    biases[col] -= lr * db / batch_size;
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

// https://web.archive.org/web/20120516091853/https://yann.lecun.com/exdb/mnist/
int32_t readInt(std::ifstream &ifs) {
  int32_t value;
  ifs.read(reinterpret_cast<char *>(&value), sizeof(value));
  return __builtin_bswap32(value); // MNIST dataset is big-endian
}

int main(void) {
  try {
    const int num_samples = 60000;
    const int image_rows = 28;
    const int image_cols = 28;
    const int input_dim = image_rows * image_cols;
    const int hidden_dim = 128;
    const int output_dim = 10;
    const int batch_size = 64;
    const float learning_rate = 0.01f;
    const int num_iterations = 1000;

    std::ifstream image_file("data/train-images-idx3-ubyte", std::ios::binary);
    if (!image_file.is_open()) {
      throw std::runtime_error("Failed to open MNIST image file");
    }
    int magic_number = readInt(image_file);
    if (magic_number != 2051) {
      throw std::runtime_error("Training images magic number does not match.");
    }
    int num_images = readInt(image_file);
    int rows = readInt(image_file);
    int cols = readInt(image_file);
    if (num_images != num_samples || rows != image_rows || cols != image_cols) {
      throw std::runtime_error(
          "MNIST image file dimensions do not match expected values");
    }
    std::vector<float> train_images(num_images * rows * cols);
    for (int i = 0; i < num_images * rows * cols; i++) {
      unsigned char pixel = 0;
      image_file.read(reinterpret_cast<char *>(&pixel), 1);
      train_images[i] = pixel / 255.f;
    }
    image_file.close();

    std::ifstream label_file("data/train-labels-idx1-ubyte", std::ios::binary);
    if (!label_file.is_open()) {
      throw std::runtime_error("Failed to open MNIST label file");
    }
    magic_number = readInt(label_file);
    if (magic_number != 2049) {
      throw std::runtime_error("Training images magic number does not match.");
    }
    int num_labels = readInt(label_file);
    if (num_labels != num_samples) {
      throw std::runtime_error(
          "Number of labels does not match number of images");
    }
    std::vector<float> train_labels(num_labels * output_dim, 0.f);
    for (int i = 0; i < num_labels; i++) {
      unsigned char label = 0;
      label_file.read(reinterpret_cast<char *>(&label), 1);
      if (label >= output_dim) {
        throw std::runtime_error("Invalid label in MNIST label file");
      }
      train_labels[i * output_dim + label] = 1.f;
    }
    label_file.close();

    float *d_images, *d_labels;
    cudaSafeCall(
        cudaMalloc((void **)&d_images, batch_size * input_dim * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_labels,
                            batch_size * output_dim * sizeof(float)));

    // Layer 1 (Input -> Hidden)
    float *d_weights1, *d_biases1;
    cudaSafeCall(cudaMalloc((void **)&d_weights1,
                            input_dim * hidden_dim * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_biases1, hidden_dim * sizeof(float)));
    // Initialize biases
    int biasBlockSize = 256;
    int biasGridSize = (hidden_dim + biasBlockSize - 1) / biasBlockSize;
    init_biases<<<biasGridSize, biasBlockSize>>>(hidden_dim, d_biases1);
    // Initialize layer 1 weights
    dim3 blockDim2D(16, 16);
    dim3 gridDimWeights1((hidden_dim + blockDim2D.x - 1) / blockDim2D.x,
                         (input_dim + blockDim2D.y - 1) / blockDim2D.y);
    init_weights<<<gridDimWeights1, blockDim2D>>>(hidden_dim, input_dim,
                                                  d_weights1);
    cudaSafeCall(cudaPeekAtLastError());

    // Layer 2 (Hidden -> Output)
    float *d_weights2, *d_biases2;
    cudaSafeCall(cudaMalloc((void **)&d_weights2,
                            hidden_dim * output_dim * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_biases2, output_dim * sizeof(float)));
    // Initialize biases
    biasGridSize = (output_dim + biasBlockSize - 1) / biasBlockSize;
    init_biases<<<biasGridSize, biasBlockSize>>>(output_dim, d_biases1);
    // Init weights
    dim3 gridDimWeights2((output_dim + blockDim2D.x - 1) / blockDim2D.x,
                         (hidden_dim + blockDim2D.y - 1) / blockDim2D.y);
    init_weights<<<gridDimWeights2, blockDim2D>>>(output_dim, hidden_dim,
                                                  d_weights2);
    cudaSafeCall(cudaPeekAtLastError());

    // Intermediate buffers for activations, logits, predictions, and loss
    float *d_hidden_linear, *d_hidden_activation;
    cudaSafeCall(cudaMalloc((void **)&d_hidden_linear,
                            batch_size * hidden_dim * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_hidden_activation,
                            batch_size * hidden_dim * sizeof(float)));
    float *d_output_logits, *d_predictions;
    cudaSafeCall(cudaMalloc((void **)&d_output_logits,
                            batch_size * output_dim * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_predictions,
                            batch_size * output_dim * sizeof(float)));
    float *d_loss;
    cudaSafeCall(cudaMalloc((void **)&d_loss, batch_size * sizeof(float)));

    // Buffers for gradients
    float *d_grad_output; // Gradient from cross-entropy (output layer)
    cudaSafeCall(cudaMalloc((void **)&d_grad_output,
                            batch_size * output_dim * sizeof(float)));
    float *d_grad_hidden; // Gradient for hidden layer (after backpropagating
                          // layer 2)
    cudaSafeCall(cudaMalloc((void **)&d_grad_hidden,
                            batch_size * hidden_dim * sizeof(float)));

    int blockSize1D = 256;
    dim3 gridSizeLoss((batch_size + blockSize1D - 1) / blockSize1D);
    // For our 2D kernels we use blocks of size 16x16; grids are computed below
    // per kernel launch.
    dim3 block2D(16, 16);

    for (int iter = 0; iter < num_iterations; iter++) {
      int batch_start = (iter * batch_size) % num_samples;

      cudaSafeCall(cudaMemcpy(d_images, &train_images[batch_start * input_dim],
                              batch_size * input_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
      cudaSafeCall(cudaMemcpy(d_labels, &train_labels[batch_start * output_dim],
                              batch_size * output_dim * sizeof(float),
                              cudaMemcpyHostToDevice));

      // --- Forward Pass ---
      dim3 gridForward1((hidden_dim + block2D.x - 1) / block2D.x,
                        (batch_size + block2D.y - 1) / block2D.y);
      forward<<<gridForward1, block2D>>>(batch_size, input_dim, hidden_dim,
                                         d_images, d_weights1, d_biases1,
                                         d_hidden_linear);
      cudaSafeCall(cudaPeekAtLastError());

      relu<<<gridForward1, block2D>>>(hidden_dim, batch_size, d_hidden_linear,
                                      d_hidden_activation);
      cudaSafeCall(cudaPeekAtLastError());

      dim3 gridForward2((output_dim + block2D.x - 1) / block2D.x,
                        (batch_size + block2D.y - 1) / block2D.y);
      forward<<<gridForward2, block2D>>>(batch_size, hidden_dim, output_dim,
                                         d_hidden_activation, d_weights2,
                                         d_biases2, d_output_logits);
      cudaSafeCall(cudaPeekAtLastError());

      softmax<<<gridForward2, block2D>>>(output_dim, batch_size,
                                         d_output_logits, d_predictions);
      cudaSafeCall(cudaPeekAtLastError());

      cross_entropy<<<gridSizeLoss, blockSize1D>>>(
          output_dim, batch_size, d_predictions, d_labels, d_loss);
      cudaSafeCall(cudaPeekAtLastError());

      // --- Backward Pass ---
      cross_entropy_backwards<<<gridForward2, block2D>>>(
          output_dim, batch_size, d_predictions, d_labels, d_grad_output);
      cudaSafeCall(cudaPeekAtLastError());

      update_layer<<<gridForward2, block2D>>>(
          output_dim, hidden_dim, batch_size, learning_rate, d_weights2,
          d_biases2, d_hidden_activation, d_grad_output);
      cudaSafeCall(cudaPeekAtLastError());

      backwards<<<gridForward2, block2D>>>(batch_size, hidden_dim, output_dim,
                                           d_weights2, d_biases2, d_grad_output,
                                           d_grad_hidden);
      cudaSafeCall(cudaPeekAtLastError());

      relu_backwards<<<gridForward1, block2D>>>(hidden_dim, batch_size,
                                                d_hidden_linear, d_grad_hidden,
                                                d_hidden_linear);
      cudaSafeCall(cudaPeekAtLastError());

      update_layer<<<gridForward1, block2D>>>(
          hidden_dim, input_dim, batch_size, learning_rate, d_weights1,
          d_biases1, d_images, d_hidden_linear);
      cudaSafeCall(cudaPeekAtLastError());

      cudaSafeCall(cudaDeviceSynchronize());

      std::vector<float> loss_host(batch_size);
      cudaSafeCall(cudaMemcpy(loss_host.data(), d_loss,
                              batch_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
      float loss_sum = 0.f;
      for (float l : loss_host) {
        loss_sum += l;
      }
      float avg_loss = loss_sum / batch_size;
      std::cout << "Iteration " << iter << " Epoch "
                << iter * batch_size / num_samples << " - Loss: " << avg_loss
                << std::endl;
    }

    cudaSafeCall(cudaFree(d_images));
    cudaSafeCall(cudaFree(d_labels));
    cudaSafeCall(cudaFree(d_weights1));
    cudaSafeCall(cudaFree(d_biases1));
    cudaSafeCall(cudaFree(d_weights2));
    cudaSafeCall(cudaFree(d_biases2));
    cudaSafeCall(cudaFree(d_hidden_linear));
    cudaSafeCall(cudaFree(d_hidden_activation));
    cudaSafeCall(cudaFree(d_output_logits));
    cudaSafeCall(cudaFree(d_predictions));
    cudaSafeCall(cudaFree(d_loss));
    cudaSafeCall(cudaFree(d_grad_output));
    cudaSafeCall(cudaFree(d_grad_hidden));

    cudaSafeCall(cudaDeviceReset());
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
