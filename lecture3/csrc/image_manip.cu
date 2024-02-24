#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


__global__ void rgb_to_grayscale_1d_kernel(const unsigned char* img, unsigned char* out, const int pixels_per_channel) {
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

__global__ void rgb_to_grayscale_2d_kernel(const unsigned char* img, unsigned char* out, const int height, const int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // moves along height
  int j = blockIdx.y * blockDim.y + threadIdx.y; // moves along width

  if (i < height && j < width) {
    int pixels_per_channel = height * width;
    int idx = i * width + j;
    out[idx] = 0.21f * img[idx] + 0.72f * img[idx + pixels_per_channel] + 0.07f * img[idx + 2 * pixels_per_channel];  
  }
}


torch::Tensor grey_scale_1d(const torch::Tensor image) {
  CHECK_INPUT(image);
  const int height = image.size(1);
  const int width = image.size(2);
  
  const int pixels_per_channel = width * height;
  
  torch::Tensor out = torch::empty({height, width}, image.options());

  rgb_to_grayscale_1d_kernel<<<ceil(pixels_per_channel / 1024.0), 1024>>>(
    image.data_ptr<unsigned char>(),
    out.data_ptr<unsigned char>(),
    pixels_per_channel
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor grey_scale_2d(const torch::Tensor image) {
  CHECK_INPUT(image);
  const int height = image.size(1);
  const int width = image.size(2);
  
  torch::Tensor out = torch::empty({height, width}, image.options());

  const dim3 blocks_in_grid(ceil(height / 32.0), ceil(width / 32.0), 1);
  const dim3 threads_in_block(32, 32, 1);

  rgb_to_grayscale_2d_kernel<<<blocks_in_grid, threads_in_block>>>(
    image.data_ptr<unsigned char>(),
    out.data_ptr<unsigned char>(),
    height,
    width
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  pybind11::module ops = m.def_submodule("ops", "Grey scale operations using CUDA");

  ops.def(
    "grey_scale_1d",
    &grey_scale_1d,
    "Flattens the image and uses 1d kernel to convert it to grayscale (CUDA)"
  );
  ops.def(
    "grey_scale_2d",
    &grey_scale_2d,
    "Flattens the image and uses 2d kernel to convert it to grayscale (CUDA)"
  );
}