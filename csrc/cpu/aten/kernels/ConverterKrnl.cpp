#include <aten/Converter.h>
#include "vec/vec.h"

#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

#define BF16_2_FP32(dst, src, len) cvt_bf16_to_fp32(dst, src, len)
#define FP32_2_BF16(dst, src, len) cvt_fp32_to_bf16(dst, src, len)

namespace torch_ipex {
namespace cpu {

namespace {

void bf16_to_fp32(void* dst, const void* src, int len) {
  BF16_2_FP32((float*)dst, (at::BFloat16*)src, len);
}

void fp32_to_bf16(void* dst, const void* src, int len) {
  FP32_2_BF16((at::BFloat16*)dst, (float*)src, len);
}

at::Tensor cat_bfloat16_float_kernel_impl(
    const at::Tensor top_half_,
    const at::Tensor bottom_half_) {
  at::Tensor top_half = top_half_.contiguous();
  at::Tensor bottom_half = bottom_half_.contiguous();
  at::Tensor output = at::empty_strided(
      top_half_.sizes(),
      top_half_.strides(),
      top_half_.options().dtype(at::kFloat));
  at::Tensor output_contiguous = output.contiguous();
  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;
  at::BFloat16* top_half_data = top_half.data_ptr<at::BFloat16>();
  at::BFloat16* bottom_half_data = bottom_half.data_ptr<at::BFloat16>();
  float* output_data = output_contiguous.data_ptr<float>();
  int64_t grain_size = 512;
  at::parallel_for(
      0, top_half.numel(), grain_size, [&](int64_t begin, int64_t end) {
        // local pointers
        at::BFloat16* top_half_ptr = top_half_data + begin;
        at::BFloat16* bottom_half_ptr = bottom_half_data + begin;
        float* output_ptr = output_data + begin;
        const int64_t size = end - begin;
        int64_t d = 0;
        for (; d < size - (size % bVec::size()); d += bVec::size()) {
          bVec top_half_bvec = bVec::loadu(top_half_ptr + d);
          bVec bottom_half_bvec = bVec::loadu(bottom_half_ptr + d);
          fVec fvec, fvec2;
          std::tie(fvec, fvec2) =
              pack_bfloat16_float(top_half_bvec, bottom_half_bvec);
          fvec.store(output_ptr + d);
          fvec2.store(output_ptr + d + fVec::size());
        }
        for (; d < size; d++) {
          output_ptr[d] =
              at::vec::pack_bfloat16_float(top_half_ptr[d], bottom_half_ptr[d]);
        }
      });
  if (!output.is_contiguous()) {
    output.copy_(output_contiguous);
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor> split_float_bfloat16_kernel_impl(
    const at::Tensor tensor_) {
  auto tensor = tensor_.contiguous();
  auto top_half = at::empty_strided(
      tensor_.sizes(),
      tensor_.strides(),
      tensor_.options().dtype(at::kBFloat16));
  auto top_half_contiguous = top_half.contiguous();
  auto bottom_half = at::empty_strided(
      tensor_.sizes(),
      tensor_.strides(),
      tensor_.options().dtype(at::kBFloat16));
  auto bottom_half_contiguous = bottom_half.contiguous();
  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;
  at::BFloat16* top_half_data = top_half_contiguous.data_ptr<at::BFloat16>();
  at::BFloat16* bottom_half_data =
      bottom_half_contiguous.data_ptr<at::BFloat16>();
  float* tensor_data = tensor.data_ptr<float>();
  int64_t grain_size = 512;
  at::parallel_for(
      0, top_half.numel(), grain_size, [&](int64_t begin, int64_t end) {
        // local pointers
        at::BFloat16* top_half_ptr = top_half_data + begin;
        at::BFloat16* bottom_half_ptr = bottom_half_data + begin;
        float* tensor_ptr = tensor_data + begin;
        const int64_t size = end - begin;
        int64_t d = 0;
        for (; d < size - (size % bVec::size()); d += bVec::size()) {
          fVec fvec = fVec::loadu(tensor_ptr + d);
          fVec fvec2 = fVec::loadu(tensor_ptr + d + fVec::size());
          bVec top_half_bvec, bottom_half_bvec;
          std::tie(top_half_bvec, bottom_half_bvec) =
              unpack_float_bfloat16(fvec, fvec2);
          top_half_bvec.store(top_half_ptr + d);
          bottom_half_bvec.store(bottom_half_ptr + d);
        }
        for (; d < size; d++) {
          at::BFloat16 top_half_val;
          at::BFloat16 bottom_half_val;
          std::tie(top_half_val, bottom_half_val) =
              unpack_float_bfloat16(tensor_ptr[d]);
          top_half_ptr[d] = top_half_val;
          bottom_half_ptr[d] = bottom_half_val;
        }
      });
  if (!top_half.is_contiguous()) {
    top_half.copy_(top_half_contiguous);
    bottom_half.copy_(bottom_half_contiguous);
  }
  return std::tie(top_half, bottom_half);
}

} // anonymous namespace

IPEX_REGISTER_DISPATCH(
    cat_bfloat16_float_kernel_stub,
    &cat_bfloat16_float_kernel_impl);
IPEX_REGISTER_DISPATCH(
    split_float_bfloat16_kernel_stub,
    &split_float_bfloat16_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
