#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void addition_kernel(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

torch::Tensor addition_cuda(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    AT_ASSERTM(a.sizes() == b.sizes(), "a and b must have the same size");
    int n = a.numel();
    torch::Tensor out = torch::empty_like(a);
    addition_kernel<<<(n + 512 - 1) / 512, 512>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
