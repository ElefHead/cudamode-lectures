#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_WIDTH 32
#define BLOCK_SIZE_FLOAT 32.0


__global__
void matrix_multiply_tiled_kernel(
    float* M, float* N, float* out, 
    int M_height, int width, int N_width
) {
    __shared__ float M_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N_shared[TILE_WIDTH][TILE_WIDTH];

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    const int idr = threadIdx.y;
    const int idc = threadIdx.x;

    float value = 0.0f;

    for(int phase = 0; phase < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++phase) {
        
        const int M_col = phase * TILE_WIDTH + idc;
        const int N_row = phase * TILE_WIDTH + idr;

        M_shared[idr][idc] = (((row < M_height) && (M_col < width)) ? M[row * width + M_col] : 0.0);
        N_shared[idr][idc] = ((( N_row < width) && (col < N_width)) ? N[N_row * N_width + col] : 0.0);

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            value += M_shared[idr][i] * N_shared[i][idc];
        }

        __syncthreads();

    }

    if ((row < M_height) && (col < N_width)) {
        out[row * N_width + col] = value;
    }

}



torch::Tensor matrix_multiply_tiled(const torch::Tensor& M, const torch::Tensor& N) {
    CHECK_INPUT(M);
    CHECK_INPUT(N);

    const int M_height = M.size(0);
    const int M_width = M.size(1);
    const int N_width = N.size(1);

    AT_ASSERTM(M_width == N.size(0), "M width and N height should be the same");

    torch::Tensor out = torch::empty({M_height, N_width}, M.options());

    dim3 blocks_in_grid(ceil(N_width / BLOCK_SIZE_FLOAT), ceil(M_height / BLOCK_SIZE_FLOAT), 1);
    dim3 threads_in_block(TILE_WIDTH, TILE_WIDTH, 1);

    matrix_multiply_tiled_kernel<<<blocks_in_grid, threads_in_block>>>(
        M.data_ptr<float>(), 
        N.data_ptr<float>(),
        out.data_ptr<float>(),
        M_height, M_width, N_width
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;

}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  pybind11::module ops = m.def_submodule("ops", "Tiled Matrix multiplication operations using CUDA");

  ops.def(
    "matrix_multiply_tiled",
    &matrix_multiply_tiled,
    "Tiled Matrix Multiplication",
    py::arg("M"),
    py::arg("N")
  );

}