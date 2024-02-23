#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


__global__ void rgb_to_grayscale_kernel(const float* img, float* out, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 0.299f * img[i] + 0.587f * img[i + n] + 0.114f * img[i + 2 * n];
    }

}


torch::Tensor grey_scale_flattened(torch::Tensor image) {
  CHECK_INPUT(image);
  int n = a.numel();
  torch::Tensor out = torch::empty_like(a);
  rgb_to_grayscale_kernel<<<ceil(n / 1024.0), 1024>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "grey_scale_flattened",
    &grey_scale_flattened,
    "Flattens the image and converts it to grayscale (CUDA)", 
    py::arg("image")
  );
}