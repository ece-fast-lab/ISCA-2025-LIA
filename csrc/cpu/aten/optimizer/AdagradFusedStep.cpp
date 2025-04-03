#include "optimizer.h"

#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(adagrad_fused_step_kernel_stub);

std::tuple<at::Tensor, at::Tensor> adagrad_fused_step(
    const at::Tensor& param_,
    const at::Tensor& grad_,
    const at::Tensor& state_sum_,
    const at::Tensor& param2_,
    double step,
    double learning_rate,
    double weight_decay,
    double lr_decay,
    double eps) {
  RECORD_FUNCTION(
      "torch_ipex::adagrad_fused_step", c10::ArrayRef<c10::IValue>({}));

  TORCH_CHECK(
      learning_rate >= 0, "Expect learning rate >= 0.0, got ", learning_rate);
  TORCH_CHECK(lr_decay >= 0, "Expect lr_decay >=0.0 , got ", lr_decay);
  TORCH_CHECK(eps >= 0, "Expect eps >= 0.0, got ", eps);
  TORCH_CHECK(
      weight_decay >= 0, "Expect weight_decay >= 0.0, got ", weight_decay);

  TORCH_CHECK(
      param_.sizes() == grad_.sizes(),
      "Expect param and grad_ have the same sizes, param sizes: ",
      param_.sizes(),
      "; grad_ sizes: ",
      grad_.sizes());
  TORCH_CHECK(
      param_.sizes() == state_sum_.sizes(),
      "Expect param and state_sum have the same sizes, param sizes: ",
      param_.sizes(),
      "; state_sum sizes: ",
      state_sum_.sizes());
  TORCH_CHECK(
      param2_.numel() == 0 || param_.sizes() == param2_.sizes(),
      "Expect param and param2_ have the same sizes, param sizes: ",
      param_.sizes(),
      "; param2_ sizes: ",
      param2_.sizes());

  /*
  pointer to adagrad_fused_step_kernel_impl(
      param_,
      grad_,
      state_sum_,
      param2_,
      step,
      learning_rate,
      weight_decay,
      lr_decay,
      eps);
  */
  return adagrad_fused_step_kernel_stub(
      kCPU,
      param_,
      grad_,
      state_sum_,
      param2_,
      step,
      learning_rate,
      weight_decay,
      lr_decay,
      eps);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "adagrad_fused_step(Tensor(a!) param, Tensor grad, Tensor(b!) "
      "state_sum, Tensor trail, float step, float lr, float weight_decay, "
      "float lr_decay, float eps) -> (Tensor(a!), Tensor(b!))",
      torch_ipex::cpu::adagrad_fused_step);
}

} // namespace
