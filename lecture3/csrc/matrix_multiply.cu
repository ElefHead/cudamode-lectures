#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


__global__
void matrix_multiply_1d_kernel(
    const float* M, const float* N, float* out, 
    const int M_height, const int width, 
    const int N_width
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M_height * N_width) {
        float total_value = 0.0;
        const int M_row = i / N_width;
        const int N_col = i % N_width;

        for (int j=0; j < width; ++j) {
            total_value += M[M_row * width + j] * N[j * N_width + N_col];
        }

        out[i] = total_value;
    }

}


__global__
void matrix_multiplication_2d_kernel(
    const float* M, const float* N, float* out,
    const int M_height, const int width, const int N_width
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if ((row < M_height) && (col < N_width)) {
        float total_value = 0.0;

        for (int i=0; i < width; ++i) {
            total_value += M[row * width + i] * N[i * N_width + col];
        }

        out[row*width + col] = total_value;
    }

}


torch::Tensor matrix_multiply_1d_op(
    const torch::Tensor M, const torch::Tensor N
) {
    CHECK_INPUT(M);
    CHECK_INPUT(N);

    const int M_height = M.size(0);
    const int M_width = M.size(1);
    const int N_height = N.size(0);
    const int N_width = N.size(1);

    AT_ASSERTM(M_width == N_height, "M width and N height should be the same");

    torch::Tensor out = torch::empty({M_height, N_width}, M.options());

    matrix_multiply_1d_kernel<<<ceil((M_height * N_width) / 1024.0), 1024>>>(
        M.data_ptr<float>(), 
        N.data_ptr<float>(),
        out.data_ptr<float>(),
        M_height, M_width, N_width
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


torch::Tensor matrix_multiply_2d_op(
    const torch::Tensor M, const torch::Tensor N
) {
    CHECK_INPUT(M);
    CHECK_INPUT(N);

    const int M_height = M.size(0);
    const int M_width = M.size(1);
    const int N_height = N.size(0);
    const int N_width = N.size(1);

    AT_ASSERTM(M_width == N_height, "M width and N height should be the same");

    torch::Tensor out = torch::empty({M_height, N_width}, M.options());

    dim3 blocks_in_grid(ceil(M_height / 32.0), ceil(N_width / 32.0), 1);
    dim3 threads_in_block(32, 32, 1);

    matrix_multiply_1d_kernel<<<blocks_in_grid, threads_in_block>>>(
        M.data_ptr<float>(), 
        N.data_ptr<float>(),
        out.data_ptr<float>(),
        M_height, M_width, N_width
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  pybind11::module ops = m.def_submodule("ops", "Matrix multiply operations using CUDA");

  ops.def(
    "matrix_multiply_1d_op",
    &matrix_multiply_1d_op,
    "Flattens the inputs and uses 1d kernel to calculate matrix multiplication output",
    py::arg("M"),
    py::arg("N")
  );

  ops.def(
    "matrix_multiply_2d_op",
    &matrix_multiply_2d_op,
    "Flattens the inputs and uses 1d kernel to calculate matrix multiplication output",
    py::arg("M"),
    py::arg("N")
  );

}