#pragma once

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

torch::stable::Tensor get_cuda_view_from_cpu_tensor(
    torch::stable::Tensor& cpu_tensor);

inline torch::stable::Tensor weak_ref_tensor(torch::stable::Tensor& tensor) {
  STD_TORCH_CHECK(tensor.device().is_cuda(), "Tensor must be on CUDA device");
  return torch::stable::from_blob(tensor.mutable_data_ptr(), tensor.sizes(),
                                  tensor.strides(), tensor.device(),
                                  tensor.scalar_type());
}
