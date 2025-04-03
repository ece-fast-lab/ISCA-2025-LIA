#include "AddSoftmax.h"
#include <torch/all.h>

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(div_add_softmax_kernel_stub);
IPEX_DEFINE_DISPATCH(add_softmax_inplace_kernel_stub);

at::Tensor DivAddSoftmax(
    at::Tensor& a,
    const at::Tensor& b,
    const float& dim_per_head) {
  RECORD_FUNCTION("dil_addsoftmax", c10::ArrayRef<c10::IValue>({}));
  // pointer to div_add_softmax_kernel_impl(a, b, dim_per_head);
  return div_add_softmax_kernel_stub(kCPU, a, b, dim_per_head);
}

at::Tensor& AddSoftmax_(at::Tensor& a, const at::Tensor& b) {
  return add_softmax_inplace_kernel_stub(kCPU, a, b);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  // This custom op is needed because some models cannot be JIT-ed right now.
  // It will be removed after these models can be JIT-ed.
  m.def("add_softmax_(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  m.impl("add_softmax_", c10::DispatchKey::CPU, torch_ipex::cpu::AddSoftmax_);
}

} // namespace
