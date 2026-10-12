# FlashAttention CUDA

A from-scratch CUDA project exploring GPU programming, matrix multiplication,
attention, and the optimization ideas behind FlashAttention.

## Progress

- [x] CUDA vector addition
- [x] CPU matrix multiplication
- [x] Naive CUDA matrix multiplication
- [x] Shared-memory tiled matrix multiplication
- [ ] CPU scaled dot-product attention
- [ ] Naive CUDA attention
- [ ] GPU softmax
- [ ] Online softmax
- [ ] FlashAttention-style forward kernel
- [ ] Correctness validation
- [ ] Profiling and optimization
- [ ] Benchmarks

## Goals

- Learn the CUDA execution model
- Understand GPU memory hierarchy
- Implement matrix multiplication from scratch
- Implement scaled dot-product attention
- Implement online softmax
- Build a FlashAttention-style CUDA kernel
- Profile kernels using NVIDIA tools
- Compare naive and optimized implementations