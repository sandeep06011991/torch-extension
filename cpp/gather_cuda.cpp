#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)





torch::Tensor gather_reduce_cuda(torch::Tensor a,torch::Tensor b);

PYBIND11_MODULE(cuda_kernel, m) {
  m.def("gather_cuda", &gather_reduce_cuda, "LLTM forward");
}
