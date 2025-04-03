#pragma once

#include <immintrin.h>

#include <ATen/ATen.h>

namespace torch_ipex {
namespace cpu {
namespace kernel {

#if defined(CPU_CAPABILITY_AVX512)

inline void update_batch_kernel_impl(
    const __m512i& max_symbols_epi32,
    const __m512i& flag_1_epi32,
    const __m512i& flag_0_epi32,
    const __m512i& blank_id_epi32,
    const __m512i& k_right_epi64,
    const __m512i& k_left_epi64,
    const __m512i& out_lens_epi32,
    const __m512i& sos_epi32,
    __m512i& lable_col_epi32,
    __m512i& symbols_added_epi32,
    __m512i& time_idxs_epi32,
    __m512i& blankness_out_epi32,
    __m512i& blankvec_out_epi32,
    __m512i& not_blank_out_epi32,
    __m512i& label_to_put_out_right_epi64,
    __m512i& label_to_put_out_left_epi64) {
  auto k_epi32 = _mm512_castsi256_si512(_mm512_cvtusepi64_epi32(k_right_epi64));
  k_epi32 =
      _mm512_inserti64x4(k_epi32, _mm512_cvtusepi64_epi32(k_left_epi64), 1);

  // blankness = k.eq(self._blank_id)
  auto blankness_eq_mask = _mm512_cmpeq_epi32_mask(k_epi32, blank_id_epi32);
  // symbols_added *= blankness.logical_not()
  symbols_added_epi32 = _mm512_mask_mullo_epi32(
      symbols_added_epi32,
      blankness_eq_mask,
      symbols_added_epi32,
      flag_0_epi32);

  // time_idxs = time_idxs + blankness
  time_idxs_epi32 = _mm512_mask_add_epi32(
      time_idxs_epi32, blankness_eq_mask, time_idxs_epi32, flag_1_epi32);
  // blank_vec = time_idxs.ge(out_lens)
  auto blank_vec_ge_mask =
      _mm512_cmpge_epi32_mask(time_idxs_epi32, out_lens_epi32);

  // not_blank = tmp_blank_vec.eq(0)
  auto temp1 = _mm512_kor(blankness_eq_mask, blank_vec_ge_mask);
  auto temp2 = _mm512_cmpeq_epi32_mask(flag_1_epi32, flag_1_epi32);
  auto not_blank_mask = _mm512_kxor(temp1, temp2);
  not_blank_out_epi32 = _mm512_mask_add_epi32(
      flag_0_epi32, not_blank_mask, flag_0_epi32, flag_1_epi32);

  // label_col += not_blank
  lable_col_epi32 = _mm512_mask_add_epi32(
      lable_col_epi32, not_blank_mask, lable_col_epi32, flag_1_epi32);
  // symbols_added += not_blank
  symbols_added_epi32 = _mm512_mask_add_epi32(
      symbols_added_epi32, not_blank_mask, symbols_added_epi32, flag_1_epi32);

  auto symbols_ge_mask =
      _mm512_cmpge_epi32_mask(symbols_added_epi32, max_symbols_epi32);

  // time_idxs += need_add
  time_idxs_epi32 = _mm512_mask_add_epi32(
      time_idxs_epi32, symbols_ge_mask, time_idxs_epi32, flag_1_epi32);
  // symbols_added *= symbols_added.lt(max_symbols)
  symbols_added_epi32 = _mm512_mask_mullo_epi32(
      symbols_added_epi32, symbols_ge_mask, symbols_added_epi32, flag_0_epi32);

  // blankness.logical_or_(need_add)
  auto blankness_out_mask = _mm512_kor(blankness_eq_mask, symbols_ge_mask);
  blankness_out_epi32 = _mm512_mask_add_epi32(
      flag_0_epi32, blankness_out_mask, flag_0_epi32, flag_1_epi32);
  blankvec_out_epi32 = _mm512_mask_add_epi32(
      flag_0_epi32, blank_vec_ge_mask, flag_0_epi32, flag_1_epi32);

  // (k-self._SOS)*not_blank
  auto label_to_put_epi32 = _mm512_sub_epi32(k_epi32, sos_epi32);
  label_to_put_epi32 =
      _mm512_mullo_epi32(label_to_put_epi32, not_blank_out_epi32);

  __m256i vlow = _mm512_castsi512_si256(label_to_put_epi32);
  __m256i vhigh = _mm512_extracti64x4_epi64(label_to_put_epi32, 1);

  label_to_put_out_right_epi64 = _mm512_cvtepi32_epi64(vlow);
  label_to_put_out_left_epi64 = _mm512_cvtepi32_epi64(vhigh);
}

inline void update_batch_kernel(
    const at::Tensor& k,
    const at::Tensor& out_lens,
    at::Tensor label_col,
    at::Tensor symbols_added,
    at::Tensor time_idxs,
    at::Tensor blankness_out,
    at::Tensor blankvec_out,
    at::Tensor not_blank_out,
    at::Tensor label_to_put_out,
    int max_symbols,
    int blank_id,
    int len,
    int _SOS) {
  void* k_ptr = static_cast<int64_t*>(k.data_ptr());
  void* out_lens_ptr = static_cast<int32_t*>(out_lens.data_ptr());
  void* lable_col_ptr = static_cast<int32_t*>(label_col.data_ptr());
  void* symbols_added_ptr = static_cast<int32_t*>(symbols_added.data_ptr());
  void* time_idxs_ptr = static_cast<int32_t*>(time_idxs.data_ptr());
  void* blankness_out_ptr = static_cast<int32_t*>(blankness_out.data_ptr());
  void* blankvec_out_ptr = static_cast<int32_t*>(blankvec_out.data_ptr());
  void* not_blank_out_ptr = static_cast<int32_t*>(not_blank_out.data_ptr());
  void* label_to_put_out_ptr =
      static_cast<int64_t*>(label_to_put_out.data_ptr());

  auto max_symbols_epi32 = _mm512_set1_epi32(max_symbols);
  auto flag_1_epi32 = _mm512_set1_epi32(1);
  auto flag_0_epi32 = _mm512_set1_epi32(0);
  auto sos_epi32 = _mm512_set1_epi32(_SOS);
  auto blank_id_epi32 = _mm512_set1_epi32(blank_id);
  auto blankness_out_epi32 = _mm512_set1_epi32(0);
  auto blankvec_out_epi32 = _mm512_set1_epi32(0);
  auto not_blank_out_epi32 = _mm512_set1_epi32(0);
  auto label_to_put_out_right_epi64 = _mm512_set1_epi64(0);
  auto label_to_put_out_left_epi64 = _mm512_set1_epi64(0);

  int i = 0;
  for (; i <= len - 16; i += 16) {
    auto k_right_epi64 = _mm512_load_epi64((void*)((int64_t*)k_ptr + i + 0));
    auto k_left_epi64 = _mm512_load_epi64((void*)((int64_t*)k_ptr + i + 8));
    auto out_lens_epi32 =
        _mm512_load_epi32((void*)((int32_t*)out_lens_ptr + i));
    auto lable_col_epi32 =
        _mm512_load_epi32((void*)((int32_t*)lable_col_ptr + i));
    auto symbols_added_epi32 =
        _mm512_load_epi32((void*)((int32_t*)symbols_added_ptr + i));
    auto time_idxs_epi32 =
        _mm512_load_epi32((void*)((int32_t*)time_idxs_ptr + i));

    update_batch_kernel_impl(
        max_symbols_epi32,
        flag_1_epi32,
        flag_0_epi32,
        blank_id_epi32,
        k_right_epi64,
        k_left_epi64,
        out_lens_epi32,
        sos_epi32,
        lable_col_epi32,
        symbols_added_epi32,
        time_idxs_epi32,
        blankness_out_epi32,
        blankvec_out_epi32,
        not_blank_out_epi32,
        label_to_put_out_right_epi64,
        label_to_put_out_left_epi64);

    _mm512_store_epi32(
        (void*)((int32_t*)symbols_added_ptr + i), symbols_added_epi32);
    _mm512_store_epi32((void*)((int32_t*)time_idxs_ptr + i), time_idxs_epi32);
    _mm512_store_epi32((void*)((int32_t*)lable_col_ptr + i), lable_col_epi32);
    _mm512_store_epi32(
        (void*)((int32_t*)blankness_out_ptr + i), blankness_out_epi32);
    _mm512_store_epi32(
        (void*)((int32_t*)blankvec_out_ptr + i), blankvec_out_epi32);
    _mm512_store_epi32(
        (void*)((int32_t*)not_blank_out_ptr + i), not_blank_out_epi32);
    _mm512_store_epi64(
        (void*)((int64_t*)label_to_put_out_ptr + i + 0),
        label_to_put_out_right_epi64);
    _mm512_store_epi64(
        (void*)((int64_t*)label_to_put_out_ptr + i + 8),
        label_to_put_out_left_epi64);
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto k_right_epi64 =
        _mm512_maskz_load_epi64(mask, (void*)((int64_t*)k_ptr + i + 0));
    auto k_left_epi64 =
        _mm512_maskz_load_epi64(mask >> 8, (void*)((int64_t*)k_ptr + i + 8));
    auto out_lens_epi32 =
        _mm512_maskz_load_epi32(mask, (void*)((int32_t*)out_lens_ptr + i));
    auto lable_col_epi32 =
        _mm512_maskz_load_epi32(mask, (void*)((int32_t*)lable_col_ptr + i));
    auto symbols_added_epi32 =
        _mm512_maskz_load_epi32(mask, (void*)((int32_t*)symbols_added_ptr + i));
    auto time_idxs_epi32 =
        _mm512_maskz_load_epi32(mask, (void*)((int32_t*)time_idxs_ptr + i));

    update_batch_kernel_impl(
        max_symbols_epi32,
        flag_1_epi32,
        flag_0_epi32,
        blank_id_epi32,
        k_right_epi64,
        k_left_epi64,
        out_lens_epi32,
        sos_epi32,
        lable_col_epi32,
        symbols_added_epi32,
        time_idxs_epi32,
        blankness_out_epi32,
        blankvec_out_epi32,
        not_blank_out_epi32,
        label_to_put_out_right_epi64,
        label_to_put_out_left_epi64);

    _mm512_mask_store_epi32(
        (void*)((int32_t*)symbols_added_ptr + i), mask, symbols_added_epi32);
    _mm512_mask_store_epi32(
        (void*)((int32_t*)time_idxs_ptr + i), mask, time_idxs_epi32);
    _mm512_mask_store_epi32(
        (void*)((int32_t*)lable_col_ptr + i), mask, lable_col_epi32);
    _mm512_mask_store_epi32(
        (void*)((int32_t*)blankness_out_ptr + i), mask, blankness_out_epi32);
    _mm512_mask_store_epi32(
        (void*)((int32_t*)blankvec_out_ptr + i), mask, blankvec_out_epi32);
    _mm512_mask_store_epi32(
        (void*)((int32_t*)not_blank_out_ptr + i), mask, not_blank_out_epi32);
    _mm512_mask_store_epi64(
        (void*)((int64_t*)label_to_put_out_ptr + i + 0),
        mask,
        label_to_put_out_right_epi64);
    _mm512_mask_store_epi64(
        (void*)((int64_t*)label_to_put_out_ptr + i + 8),
        mask >> 8,
        label_to_put_out_left_epi64);
  }
}

inline bool should_update_feature(const at::Tensor& blankness_out, int len) {
  // if blankness_out.nonzero().size(0) > 0, return true; else return false
  void* blankness_out_ptr = static_cast<int32_t*>(blankness_out.data_ptr());
  int i = 0;
  for (; i <= len - 16; i += 16) {
    auto blankness_out_epi32 =
        _mm512_load_epi32((void*)((int32_t*)blankness_out_ptr + i));
    if (_mm512_reduce_add_epi32(blankness_out_epi32) != 0) {
      return true;
    }
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto blankness_out_epi32 =
        _mm512_maskz_load_epi32(mask, (void*)((int32_t*)blankness_out_ptr + i));
    if (_mm512_reduce_add_epi32(blankness_out_epi32) != 0) {
      return true;
    }
  }

  return false;
}

inline bool all_time_idxs_processed_kernel(
    const at::Tensor& blankvec_out,
    int len) {
  // if blank_vec.nonzero().size(0) == batch_size, return true; else return
  // false
  void* blankvec_out_ptr = static_cast<int32_t*>(blankvec_out.data_ptr());

  int sum = 0;
  int i = 0;
  for (; i <= len - 16; i += 16) {
    auto blankvec_out_epi32 =
        _mm512_load_epi32((void*)((int32_t*)blankvec_out_ptr + i));
    sum += _mm512_reduce_add_epi32(blankvec_out_epi32);
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto blankvec_out_epi32 =
        _mm512_maskz_load_epi32(mask, (void*)((int32_t*)blankvec_out_ptr + i));
    sum += _mm512_reduce_add_epi32(blankvec_out_epi32);
  }
  return (sum == len);
}

#endif

} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
