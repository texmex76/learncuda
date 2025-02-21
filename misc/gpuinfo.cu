#include <cuda_runtime.h>
#include <iostream>

void printDeviceProperties(const cudaDeviceProp &prop, int deviceID) {
  std::cout << "========== GPU Device " << deviceID
            << " ==========" << std::endl;
  std::cout << "Device Name: " << prop.name << std::endl;
  std::cout << "UUID: ";
  for (int i = 0; i < 16; i++)
    std::cout << std::hex << (int)prop.uuid.bytes[i];
  std::cout << std::dec << std::endl;
  std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024)
            << " MB" << std::endl;
  std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024
            << " KB" << std::endl;
  std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
  std::cout << "Warp Size: " << prop.warpSize << std::endl;
  std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock
            << std::endl;
  std::cout << "Max Threads per Multiprocessor: "
            << prop.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "Multiprocessor Count: " << prop.multiProcessorCount
            << std::endl;
  std::cout << "Clock Rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
  std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits"
            << std::endl;
  std::cout << "L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB"
            << std::endl;
  std::cout << "Compute Capability: " << prop.major << "." << prop.minor
            << std::endl;
  std::cout << "ECC Enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
  std::cout << "PCI Bus ID: " << prop.pciBusID
            << ", Device ID: " << prop.pciDeviceID << std::endl;
  std::cout << "Concurrent Kernels Supported: "
            << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
  std::cout << "Unified Addressing: " << (prop.unifiedAddressing ? "Yes" : "No")
            << std::endl;
  std::cout << "Can Map Host Memory: " << (prop.canMapHostMemory ? "Yes" : "No")
            << std::endl;
  std::cout << "Max Grid Size: (" << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")"
            << std::endl;
  std::cout << "Max Block Dimensions: (" << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")"
            << std::endl;
  std::cout << "Max Texture 1D Size: " << prop.maxTexture1D << std::endl;
  std::cout << "Max Texture 2D Size: (" << prop.maxTexture2D[0] << ", "
            << prop.maxTexture2D[1] << ")" << std::endl;
  std::cout << "Max Texture 3D Size: (" << prop.maxTexture3D[0] << ", "
            << prop.maxTexture3D[1] << ", " << prop.maxTexture3D[2] << ")"
            << std::endl;
  std::cout << "Max Surface 2D Size: (" << prop.maxSurface2D[0] << ", "
            << prop.maxSurface2D[1] << ")" << std::endl;
  std::cout << "Supports Cooperative Launch: "
            << (prop.cooperativeLaunch ? "Yes" : "No") << std::endl;
  std::cout << "Supports Managed Memory: "
            << (prop.managedMemory ? "Yes" : "No") << std::endl;
  std::cout << "Supports Multi-GPU Board: "
            << (prop.isMultiGpuBoard ? "Yes" : "No") << std::endl;
  std::cout << "======================================" << std::endl;
}

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    std::cout << "No CUDA-capable device found!" << std::endl;
    return 1;
  }

  std::cout << "Found " << deviceCount << " CUDA-capable device(s)."
            << std::endl;

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printDeviceProperties(prop, i);
  }

  return 0;
}
