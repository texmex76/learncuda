Me trying to learn CUDA.

# Install CUDA

Head over to the official [website](https://developer.nvidia.com/cuda-toolkit) and follow the instructions.

# Debugging

```
compute-sanitizer --tool memcheck --leak-check full --target-processes all ./main
```
