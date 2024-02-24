#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


__global__ void rgb_to_grayscale_flattened_kernel(const unsigned char* img, unsigned char* out, const int pixels_per_channel) {
  /* 
  Ok, so the image has 3 channels
  when we flatten it, we need to think about 
  the offset of rows as well as channels

  The formula for converting to grayscale is (from pmpp)
  “L=0.21*r+0.72*g+0.07*b”

  */  
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < pixels_per_channel) {
    out[i] = 0.21f * img[i] + 0.72f * img[i + pixels_per_channel] + 0.07f * img[i + 2 * pixels_per_channel];
  }

}


torch::Tensor grey_scale_flattened(const torch::Tensor image) {
  CHECK_INPUT(image);
  int height = image.size(1);
  int width = image.size(2);
  
  int pixels_per_channel = width * height;
  
  torch::Tensor out = torch::empty({height, width}, image.options());

  rgb_to_grayscale_flattened_kernel<<<ceil(pixels_per_channel / 1024.0), 1024>>>(
    image.data_ptr<unsigned char>(),
    out.data_ptr<unsigned char>(),
    pixels_per_channel
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "grey_scale_flattened",
    &grey_scale_flattened,
    "Flattens the image and converts it to grayscale (CUDA)"
  );
}