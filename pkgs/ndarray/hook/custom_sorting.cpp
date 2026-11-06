// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "custom_sorting.h"
#include "ndarray_common.h"
#include <atomic>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wundef"
#endif
#include "hwy/contrib/sort/vqsort.h"
#include "hwy/highway.h"
#include "hwy/per_target.h"
#include "hwy/contrib/sort/vqsort-inl.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
static_assert(HWY_TARGET == HWY_AVX2, "Expected HWY_AVX2 static target when compiled with AVX2+FMA+F16C");
static_assert(HWY_NATIVE_FMA == 1, "Expected hardware FMA to be enabled when compiled with -mfma");
#endif
#include <type_traits>

static thread_local int g_ndarray_oom_flag = 0;

// Process-global counter for tests that observe `NativeFinalizer` callbacks.
// The VM may run those on a GC helper thread rather than the mutator, so a
// thread-local flag written there would be invisible to the test.
static std::atomic<int> g_ndarray_finalizer_hits{0};

extern "C" {
void ndarray_set_oom_flag(void) {
    g_ndarray_oom_flag = 1;
}

int ndarray_consume_oom_flag(void) {
    int prev = g_ndarray_oom_flag;
    g_ndarray_oom_flag = 0;
    return prev;
}

void ndarray_test_finalizer_hit(void *token) {
    (void)token;
    g_ndarray_finalizer_hits.fetch_add(1, std::memory_order_relaxed);
}

int ndarray_test_consume_finalizer_hits(void) {
    return g_ndarray_finalizer_hits.exchange(0, std::memory_order_relaxed);
}
}

// ----------------------------------------------------------------------------
// Struct definitions for Complex number representations
// ----------------------------------------------------------------------------

typedef struct {
    double real;
    double imag;
} complex128_t;

typedef struct {
    float real;
    float imag;
} complex64_t;

// ----------------------------------------------------------------------------
// Inlined Comparators for Direct Sorters
// ----------------------------------------------------------------------------

static inline int compare_double_inline(double a, double b) {
    int nan_a = isnan(a);
    int nan_b = isnan(b);
    if (nan_a && nan_b) return 0;
    if (nan_a) return 1;
    if (nan_b) return -1;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_double_fast(double a, double b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_float_fast(float a, float b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_float_inline(float a, float b) {
    int nan_a = isnan(a);
    int nan_b = isnan(b);
    if (nan_a && nan_b) return 0;
    if (nan_a) return 1;
    if (nan_b) return -1;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_int64_inline(long long a, long long b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_int32_inline(int a, int b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_int16_inline(int16_t a, int16_t b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_uint8_inline(uint8_t a, uint8_t b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_int8_inline(int8_t a, int8_t b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_uint16_inline(uint16_t a, uint16_t b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_uint32_inline(uint32_t a, uint32_t b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_uint64_inline(uint64_t a, uint64_t b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline bool is_nan_float16(uint16_t bits) {
    return ((bits & 0x7C00) == 0x7C00) && ((bits & 0x03FF) != 0);
}

static inline float decode_f16_to_f32(uint16_t bits) {
    return hwy::F32FromF16Mem(&bits);
}

static inline int compare_float16_inline(uint16_t a_bits, uint16_t b_bits) {
    bool nan_a = is_nan_float16(a_bits);
    bool nan_b = is_nan_float16(b_bits);
    if (nan_a && nan_b) return 0;
    if (nan_a) return 1;
    if (nan_b) return -1;
    float a = decode_f16_to_f32(a_bits);
    float b = decode_f16_to_f32(b_bits);
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

struct Float16Less {
    bool operator()(uint16_t a_bits, uint16_t b_bits) const {
        bool nan_a = is_nan_float16(a_bits);
        bool nan_b = is_nan_float16(b_bits);
        if (nan_a || nan_b) {
            return !nan_a && nan_b;
        }
        return decode_f16_to_f32(a_bits) < decode_f16_to_f32(b_bits);
    }
};

static inline bool is_nan_bfloat16(uint16_t bits) {
    return ((bits & 0x7F80) == 0x7F80) && ((bits & 0x007F) != 0);
}

static inline float decode_bf16_to_f32(uint16_t bits) {
    return hwy::F32FromBF16Mem(&bits);
}

static inline int compare_bfloat16_inline(uint16_t a_bits, uint16_t b_bits) {
    bool nan_a = is_nan_bfloat16(a_bits);
    bool nan_b = is_nan_bfloat16(b_bits);
    if (nan_a && nan_b) return 0;
    if (nan_a) return 1;
    if (nan_b) return -1;
    float a = decode_bf16_to_f32(a_bits);
    float b = decode_bf16_to_f32(b_bits);
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

struct BFloat16Less {
    bool operator()(uint16_t a_bits, uint16_t b_bits) const {
        bool nan_a = is_nan_bfloat16(a_bits);
        bool nan_b = is_nan_bfloat16(b_bits);
        if (nan_a || nan_b) {
            return !nan_a && nan_b;
        }
        return decode_bf16_to_f32(a_bits) < decode_bf16_to_f32(b_bits);
    }
};

static inline int compare_double_with_nan(double a, double b) {
    bool a_nan = std::isnan(a);
    bool b_nan = std::isnan(b);
    if (a_nan && b_nan) return 0;
    if (a_nan) return 1;
    if (b_nan) return -1;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_float_with_nan(float a, float b) {
    bool a_nan = std::isnan(a);
    bool b_nan = std::isnan(b);
    if (a_nan && b_nan) return 0;
    if (a_nan) return 1;
    if (b_nan) return -1;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_complex128_inline(complex128_t ca, complex128_t cb) {
    int cmp_real = compare_double_with_nan(ca.real, cb.real);
    if (cmp_real != 0) return cmp_real;
    return compare_double_with_nan(ca.imag, cb.imag);
}

static inline int compare_complex64_inline(complex64_t ca, complex64_t cb) {
    int cmp_real = compare_float_with_nan(ca.real, cb.real);
    if (cmp_real != 0) return cmp_real;
    return compare_float_with_nan(ca.imag, cb.imag);
}

// Lexicographical comparison for complex numbers
template<typename T>
static inline bool comp_complex_impl(T a, T b) {
    bool nan_ar = std::isnan(a.real);
    bool nan_br = std::isnan(b.real);
    if (nan_ar && nan_br) {
        bool nan_ai = std::isnan(a.imag);
        bool nan_bi = std::isnan(b.imag);
        if (nan_ai && nan_bi) return false;
        if (nan_ai) return false;
        if (nan_bi) return true;
        return a.imag < b.imag;
    } else if (nan_ar) {
        return false;
    } else if (nan_br) {
        return true;
    } else if (a.real != b.real) {
        return a.real < b.real;
    }
    
    bool nan_ai = std::isnan(a.imag);
    bool nan_bi = std::isnan(b.imag);
    if (nan_ai && nan_bi) return false;
    if (nan_ai) return false;
    if (nan_bi) return true;
    return a.imag < b.imag;
}

// Equivalence for complex
template<typename T>
static inline bool eq_complex_impl(T a, T b) {
    bool eq_r = (a.real == b.real) || (std::isnan(a.real) && std::isnan(b.real));
    bool eq_i = (a.imag == b.imag) || (std::isnan(a.imag) && std::isnan(b.imag));
    return eq_r && eq_i;
}

// ----------------------------------------------------------------------------
// C++ Templates for sorting, searching, etc.
// ----------------------------------------------------------------------------

template <typename T>
inline bool is_nonzero(T val) {
    return val != T(0);
}
template <>
inline bool is_nonzero<double>(double val) {
    uint64_t b;
    std::memcpy(&b, &val, sizeof(b));
    return (b & 0x7FFFFFFFFFFFFFFFULL) != 0;
}
template <>
inline bool is_nonzero<float>(float val) {
    uint32_t b;
    std::memcpy(&b, &val, sizeof(b));
    return (b & 0x7FFFFFFFU) != 0;
}
template <>
inline bool is_nonzero<complex128_t>(complex128_t val) {
    return is_nonzero<double>(val.real) || is_nonzero<double>(val.imag);
}
template <>
inline bool is_nonzero<complex64_t>(complex64_t val) {
    return is_nonzero<float>(val.real) || is_nonzero<float>(val.imag);
}

template <typename T>
static inline int standard_compare(T a, T b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

template <typename T>
static inline bool is_nan_check(T val) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::isnan(val);
    }
    return false;
}

// ----------------------------------------------------------------------------
// Order-preserving unsigned key bijections for SIMD VQSort argsort
// ----------------------------------------------------------------------------

static inline uint16_t sortable_key_half(uint16_t bits) {
    if ((bits & 0x7FFFu) == 0) return 0x8000u; // canonicalize -0.0 and +0.0
    return (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                            : static_cast<uint16_t>(bits | 0x8000u);
}

template <typename T>
static inline uint32_t sortable_key32(T val) {
    if constexpr (std::is_same_v<T, float>) {
        uint32_t bits;
        std::memcpy(&bits, &val, sizeof(uint32_t));
        if ((bits << 1) == 0) bits = 0; // canonicalize -0.0f and +0.0f branchlessly
        uint32_t mask = static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31) | 0x80000000u;
        return bits ^ mask;
    } else if constexpr (sizeof(T) == 4 && std::is_signed_v<T>) {
        return static_cast<uint32_t>(val) ^ 0x80000000u;
    } else if constexpr (sizeof(T) == 4 && !std::is_signed_v<T>) {
        return static_cast<uint32_t>(val);
    } else if constexpr (sizeof(T) == 2 && std::is_signed_v<T>) {
        return static_cast<uint32_t>(static_cast<uint16_t>(val) ^ 0x8000u);
    } else {
        return static_cast<uint32_t>(static_cast<uint16_t>(val));
    }
}

template <typename T>
static inline uint64_t sortable_key64(T val) {
    if constexpr (std::is_same_v<T, double>) {
        uint64_t bits;
        std::memcpy(&bits, &val, sizeof(uint64_t));
        if ((bits << 1) == 0) bits = 0; // canonicalize -0.0 and +0.0 branchlessly
        uint64_t mask = static_cast<uint64_t>(static_cast<int64_t>(bits) >> 63) | 0x8000000000000000ULL;
        return bits ^ mask;
    } else if constexpr (std::is_signed_v<T>) {
        return static_cast<uint64_t>(val) ^ 0x8000000000000000ULL;
    } else {
        return static_cast<uint64_t>(val);
    }
}

static inline uint32_t to_sortable_bits32(float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(uint32_t));
    uint32_t mask = static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31) | 0x80000000u;
    return bits ^ mask;
}

static inline float from_sortable_bits32(uint32_t key) {
    uint32_t mask = static_cast<uint32_t>(static_cast<int32_t>(~key) >> 31) | 0x80000000u;
    uint32_t bits = key ^ mask;
    float val;
    std::memcpy(&val, &bits, sizeof(float));
    return val;
}

static inline uint64_t to_sortable_bits64(double val) {
    uint64_t bits;
    std::memcpy(&bits, &val, sizeof(uint64_t));
    uint64_t mask = static_cast<uint64_t>(static_cast<int64_t>(bits) >> 63) | 0x8000000000000000ULL;
    return bits ^ mask;
}

static inline double from_sortable_bits64(uint64_t key) {
    uint64_t mask = static_cast<uint64_t>(static_cast<int64_t>(~key) >> 63) | 0x8000000000000000ULL;
    uint64_t bits = key ^ mask;
    double val;
    std::memcpy(&val, &bits, sizeof(double));
    return val;
}

// ----------------------------------------------------------------------------
// Highway SIMD Vectorized Quickselect (VQSelect) & Helpers
// ----------------------------------------------------------------------------

HWY_BEFORE_NAMESPACE();
namespace ndarray_hwy {
namespace hn = hwy::HWY_NAMESPACE;

#if VQSORT_ENABLED
namespace hd = hwy::HWY_NAMESPACE::detail;

template <class Traits, typename T>
HWY_INLINE size_t SelectPivotRank(Traits st, const T *HWY_RESTRICT samples,
                                  size_t target_lanes, size_t num_lanes) {
    constexpr size_t kSampleLanes = hd::Constants::SampleLanes<T>();
    constexpr size_t N1 = st.LanesPerKey();
    constexpr size_t kNumSamples = kSampleLanes / N1;

    const double frac = static_cast<double>(target_lanes) / static_cast<double>(num_lanes);
    const double biased = 0.5 + 0.65 * (frac - 0.5);
    size_t target_sample = static_cast<size_t>(biased * static_cast<double>(kNumSamples));
    if (target_sample < 1) target_sample = 1;
    if (target_sample > kNumSamples - 2) target_sample = kNumSamples - 2;
    const size_t rank_mid = target_sample * N1;

    size_t rank_prev = rank_mid - N1;
    for (; st.Equal1(samples + rank_prev, samples + rank_mid); rank_prev -= N1) {
        if (rank_prev == 0) return 0;
    }

    size_t rank_next = rank_prev + N1;
    for (; st.Equal1(samples + rank_next, samples + rank_mid); rank_next += N1) {
        if (rank_next == kSampleLanes - N1) return rank_prev;
    }

    const size_t excess_if_median = rank_next - rank_mid;
    const size_t excess_if_prev = rank_mid - rank_prev;
    return excess_if_median < excess_if_prev ? rank_mid : rank_prev;
}

template <class D, class Traits, typename T>
HWY_NOINLINE void RecurseSelect(D d, Traits st, T *HWY_RESTRICT keys,
                                const size_t num, const int64_t base_key,
                                const int64_t *HWY_RESTRICT k_list,
                                const int64_t k_start, const int64_t k_end,
                                T *HWY_RESTRICT buf,
                                uint64_t *HWY_RESTRICT state,
                                const size_t remaining_levels) {
    if (k_start > k_end || num == 0) return;

    const size_t N = hn::Lanes(d);
    constexpr size_t kLPK = st.LanesPerKey();
    if (HWY_UNLIKELY(num <= hd::Constants::BaseCaseNumLanes<kLPK>(N))) {
        hd::BaseCase(d, st, keys, num, buf);
        return;
    }

    hd::DrawSamples(d, st, keys, num, buf, state);

    hn::Vec<D> pivot;
    hd::PivotResult result = hd::PivotResult::kNormal;
    if (HWY_UNLIKELY(hd::UnsortedSampleEqual(d, st, buf))) {
        pivot = st.SetKey(d, buf);
        size_t idx_second = 0;
        if (HWY_UNLIKELY(hd::AllEqual(d, st, pivot, keys, num, &idx_second))) {
            return;
        }
        const hn::Vec<D> second = st.SetKey(d, keys + idx_second);
        hn::Vec<D> third;
        if (HWY_UNLIKELY(!st.IsKV() &&
                         hd::PartitionIfTwoKeys(d, st, pivot, keys, num, idx_second,
                                                second, third, buf))) {
            return;
        }
        pivot = hd::ChoosePivotForEqualSamples(d, st, keys, num, buf, second, third,
                                               result);
    } else {
        hd::SortSamples(d, st, buf);
        if (HWY_UNLIKELY(!st.IsKV() &&
                         hd::PartitionIfTwoSamples(d, st, keys, num, buf))) {
            return;
        }
        const int64_t mid_k = k_list[k_start + (k_end - k_start) / 2];
        const size_t target_lanes = static_cast<size_t>(mid_k - base_key) * kLPK;
        const size_t pivot_rank = SelectPivotRank(st, buf, target_lanes, num);
        pivot = st.SetKey(d, buf + pivot_rank);
    }

    if (HWY_UNLIKELY(remaining_levels == 0)) {
        hd::HeapSort(st, keys, num);
        return;
    }

    const size_t bound = hd::Partition(d, st, keys, num, pivot, buf);
    const int64_t split_key = base_key + static_cast<int64_t>(bound / kLPK);

    int64_t mid = k_start;
    while (mid <= k_end && k_list[mid] < split_key) {
        mid++;
    }

    if (mid > k_start && HWY_LIKELY(result != hd::PivotResult::kIsFirst)) {
        RecurseSelect(d, st, keys, bound, base_key, k_list, k_start, mid - 1,
                      buf, state, remaining_levels - 1);
    }
    if (mid <= k_end && HWY_LIKELY(result != hd::PivotResult::kWasLast)) {
        RecurseSelect(d, st, keys + bound, num - bound, split_key, k_list, mid, k_end,
                      buf, state, remaining_levels - 1);
    }
}

template <class D, class Traits, typename T>
void Select(D d, Traits st, T *HWY_RESTRICT keys, size_t num,
            const int64_t *HWY_RESTRICT k_list, int64_t k_start, int64_t k_end,
            T *HWY_RESTRICT buf) {
#if HWY_MAX_BYTES > 64
    if (HWY_UNLIKELY(hn::Lanes(d) > 64 / sizeof(T))) {
        return Select(hn::CappedTag<T, 64 / sizeof(T)>(), st, keys, num, k_list, k_start, k_end, buf);
    }
#endif
    if (!hd::HandleSpecialCases(d, st, keys, num, buf)) {
        uint64_t *HWY_RESTRICT state = hwy::detail::GetGeneratorStateStatic();
        const size_t max_levels = 50;
        RecurseSelect(d, st, keys, num, 0, k_list, k_start, k_end, buf, state, max_levels);
    }
}

template <typename T>
void VQSelectStatic(T *HWY_RESTRICT keys, size_t num,
                    const int64_t *HWY_RESTRICT k_list, int64_t k_start, int64_t k_end) {
    using Adapter = hd::KeyAdapter<T>;
    using Order = typename Adapter::Ascending;
    const hd::SharedTraits<typename Adapter::template Traits<Order>> st;
    using LaneType = typename decltype(st)::LaneType;
    const hn::SortTag<LaneType> d;
    constexpr size_t kLPK = st.LanesPerKey();
    HWY_ALIGN LaneType buf[hwy::SortConstants::BufBytes<LaneType, kLPK>(HWY_MAX_BYTES) / sizeof(LaneType)];
    Select(d, st, reinterpret_cast<LaneType *>(keys), num * kLPK, k_list, k_start, k_end, buf);
}
#endif  // VQSORT_ENABLED

template <typename T>
int64_t FindFirstNaN(const T *HWY_RESTRICT src, int64_t size) {
    if constexpr (!std::is_floating_point_v<T>) {
        return -1;
    } else {
        const hn::ScalableTag<T> d;
        const int64_t N = hn::Lanes(d);
        int64_t i = 0;
        for (; i + 4 * N <= size; i += 4 * N) {
            const auto m0 = hn::IsNaN(hn::LoadU(d, src + i));
            const auto m1 = hn::IsNaN(hn::LoadU(d, src + i + N));
            const auto m2 = hn::IsNaN(hn::LoadU(d, src + i + 2 * N));
            const auto m3 = hn::IsNaN(hn::LoadU(d, src + i + 3 * N));
            const auto m_any = hn::Or(hn::Or(m0, m1), hn::Or(m2, m3));
            if (HWY_UNLIKELY(!hn::AllFalse(d, m_any))) {
                for (int64_t k = 0; k < 4 * N; k++) {
                    if (std::isnan(src[i + k])) return i + k;
                }
            }
        }
        for (; i < size; i++) {
            if (HWY_UNLIKELY(std::isnan(src[i]))) return i;
        }
        return -1;
    }
}

template <typename T>
bool CopyAndCheckNaN64(const T *HWY_RESTRICT src, T *HWY_RESTRICT dst, int64_t size) {
    if constexpr (!std::is_floating_point_v<T>) {
        std::memcpy(dst, src, static_cast<size_t>(size) * sizeof(T));
        return false;
    } else {
        const hn::ScalableTag<T> d;
        const int64_t N = hn::Lanes(d);
        int64_t i = 0;
        auto any_nan0 = hn::Zero(d);
        auto any_nan1 = hn::Zero(d);
        for (; i + 4 * N <= size; i += 4 * N) {
            const auto v0 = hn::LoadU(d, src + i);
            const auto v1 = hn::LoadU(d, src + i + N);
            const auto v2 = hn::LoadU(d, src + i + 2 * N);
            const auto v3 = hn::LoadU(d, src + i + 3 * N);
            hn::StoreU(v0, d, dst + i);
            hn::StoreU(v1, d, dst + i + N);
            hn::StoreU(v2, d, dst + i + 2 * N);
            hn::StoreU(v3, d, dst + i + 3 * N);
            any_nan0 = hn::Or(any_nan0, hn::Or(hn::VecFromMask(d, hn::IsNaN(v0)), hn::VecFromMask(d, hn::IsNaN(v1))));
            any_nan1 = hn::Or(any_nan1, hn::Or(hn::VecFromMask(d, hn::IsNaN(v2)), hn::VecFromMask(d, hn::IsNaN(v3))));
        }
        any_nan0 = hn::Or(any_nan0, any_nan1);
        if (HWY_UNLIKELY(!hn::AllFalse(d, hn::MaskFromVec(any_nan0)))) {
            return true;
        }
        for (; i < size; i++) {
            T v = src[i];
            if (HWY_UNLIKELY(std::isnan(v))) return true;
            dst[i] = v;
        }
        return false;
    }
}

template <typename T>
void ArgPartitionAroundPivot64(const T *HWY_RESTRICT data,
                               int64_t *HWY_RESTRICT indices,
                               int64_t size, T pivot) {
    using HwT = std::conditional_t<std::is_same_v<T, long long>, int64_t,
                std::conditional_t<std::is_same_v<T, unsigned long long>, uint64_t, T>>;
    const hn::ScalableTag<HwT> d_val;
    const hn::Repartition<int64_t, decltype(d_val)> d_idx;
    const int64_t N = hn::Lanes(d_val);

    int64_t left = 0;
    int64_t right = size;
    int64_t first_eq = -1;
    int64_t i = 0;

    if constexpr (hn::CompressIsPartition<int64_t>::value) {
        if (i <= size - 2 * N) {
            const auto v_pivot = hn::Set(d_val, static_cast<HwT>(pivot));
            auto v_idx = hn::Iota(d_idx, i);
            const auto v_step = hn::Set(d_idx, N);
            const int64_t simd_limit = size - 2 * N;
            for (; i <= simd_limit; i += N) {
                const auto v = hn::LoadU(d_val, reinterpret_cast<const HwT *>(data + i));
                const auto eq = hn::Eq(v, v_pivot);
                if (HWY_LIKELY(hn::AllFalse(d_val, eq))) {
                    const auto gt = hn::RebindMask(d_idx, hn::Gt(v, v_pivot));
                    const auto lr = hn::CompressNot(v_idx, gt);
                    const int64_t n_gt = static_cast<int64_t>(hn::CountTrue(d_idx, gt));
                    hn::StoreU(lr, d_idx, indices + left);
                    hn::StoreU(lr, d_idx, indices + right - N);
                    left += (N - n_gt);
                    right -= n_gt;
                } else {
                    for (int64_t k = 0; k < N; ++k) {
                        T val = data[i + k];
                        if (val < pivot) {
                            indices[left++] = i + k;
                        } else if (val > pivot) {
                            indices[--right] = i + k;
                        } else {
                            first_eq = i + k;
                        }
                    }
                }
                v_idx = hn::Add(v_idx, v_step);
            }
        }
    } else {
        while (i < size && (size - right) < N) {
            T v = data[i];
            if (v < pivot) {
                indices[left++] = i;
            } else if (v > pivot) {
                indices[--right] = i;
            } else {
                first_eq = i;
            }
            i++;
        }

        if (i <= size - 2 * N) {
            const auto v_pivot = hn::Set(d_val, static_cast<HwT>(pivot));
            auto v_idx = hn::Iota(d_idx, i);
            const auto v_step = hn::Set(d_idx, N);
            const int64_t simd_limit = size - 2 * N;
            for (; i <= simd_limit; i += N) {
                const auto v = hn::LoadU(d_val, reinterpret_cast<const HwT *>(data + i));
                const auto lt = hn::RebindMask(d_idx, hn::Lt(v, v_pivot));
                const auto gt = hn::RebindMask(d_idx, hn::Gt(v, v_pivot));
                const size_t n_lt = hn::CompressStore(v_idx, lt, d_idx, indices + left);
                left += static_cast<int64_t>(n_lt);
                const size_t n_gt = hn::CountTrue(d_idx, gt);
                right -= static_cast<int64_t>(n_gt);
                (void)hn::CompressBlendedStore(v_idx, gt, d_idx, indices + right);
                if (HWY_UNLIKELY(static_cast<int64_t>(n_lt + n_gt) != N)) {
                    const auto eq = hn::RebindMask(d_idx, hn::Eq(v, v_pivot));
                    first_eq = i + static_cast<int64_t>(hn::FindKnownFirstTrue(d_idx, eq));
                }
                v_idx = hn::Add(v_idx, v_step);
            }
        }
    }

    for (; i < size; i++) {
        T v = data[i];
        if (v < pivot) {
            indices[left++] = i;
        } else if (v > pivot) {
            indices[--right] = i;
        } else {
            first_eq = i;
        }
    }

    if (right - left == 1) {
        indices[left] = first_eq;
    } else if (right > left) {
        for (int64_t j = 0; j < size; j++) {
            if (data[j] == pivot) {
                indices[left++] = j;
            }
        }
    }
}

} // namespace ndarray_hwy
HWY_AFTER_NAMESPACE();

template <typename T, typename Compare>
static void multi_nth_element(T *arr, int64_t left, int64_t right, const int64_t *k_list, int64_t k_start, int64_t k_end, Compare cmp) {
    if (k_start > k_end || left >= right) return;
    int64_t mid_k_idx = k_start + (k_end - k_start) / 2;
    int64_t k = k_list[mid_k_idx];
    if (k < left) {
        multi_nth_element(arr, left, right, k_list, mid_k_idx + 1, k_end, cmp);
        return;
    }
    if (k > right) {
        multi_nth_element(arr, left, right, k_list, k_start, mid_k_idx - 1, cmp);
        return;
    }
    std::nth_element(arr + left, arr + k, arr + right + 1, cmp);
    if (k > left) {
        multi_nth_element(arr, left, k - 1, k_list, k_start, mid_k_idx - 1, cmp);
    }
    if (k < right) {
        multi_nth_element(arr, k + 1, right, k_list, mid_k_idx + 1, k_end, cmp);
    }
}

template <typename Compare>
static void arg_multi_nth_element(int64_t *indices, int64_t left, int64_t right, const int64_t *k_list, int64_t k_start, int64_t k_end, Compare cmp) {
    if (k_start > k_end || left >= right) return;
    int64_t mid_k_idx = k_start + (k_end - k_start) / 2;
    int64_t k = k_list[mid_k_idx];
    if (k < left) {
        arg_multi_nth_element(indices, left, right, k_list, mid_k_idx + 1, k_end, cmp);
        return;
    }
    if (k > right) {
        arg_multi_nth_element(indices, left, right, k_list, k_start, mid_k_idx - 1, cmp);
        return;
    }
    std::nth_element(indices + left, indices + k, indices + right + 1, cmp);
    if (k > left) {
        arg_multi_nth_element(indices, left, k - 1, k_list, k_start, mid_k_idx - 1, cmp);
    }
    if (k < right) {
        arg_multi_nth_element(indices, k + 1, right, k_list, mid_k_idx + 1, k_end, cmp);
    }
}

template <typename T>
static inline void vqselect_dispatch(T *array, int64_t valid_len, const int64_t *k_list, int64_t k_start, int64_t k_end) {
#if VQSORT_ENABLED
    if constexpr (std::is_same_v<T, long long> || std::is_same_v<T, int64_t>) {
        ndarray_hwy::VQSelectStatic(reinterpret_cast<int64_t *>(array), static_cast<size_t>(valid_len), k_list, k_start, k_end);
    } else if constexpr (std::is_same_v<T, unsigned long long> || std::is_same_v<T, uint64_t>) {
        ndarray_hwy::VQSelectStatic(reinterpret_cast<uint64_t *>(array), static_cast<size_t>(valid_len), k_list, k_start, k_end);
    } else {
        ndarray_hwy::VQSelectStatic(array, static_cast<size_t>(valid_len), k_list, k_start, k_end);
    }
#else
    if (k_end == k_start) {
        std::nth_element(array, array + k_list[k_start], array + valid_len);
    } else {
        multi_nth_element(array, 0, valid_len - 1, k_list, k_start, k_end, std::less<T>());
    }
#endif
}

template <typename T>
static void partition_impl(T *array, int64_t size, const int64_t *k_list, int64_t k_size) {
    if (array == nullptr || size <= 1 || k_list == nullptr || k_size <= 0) return;

    int64_t valid_len = size;
    if constexpr (std::is_floating_point_v<T>) {
        int64_t first_nan = -1;
        for (int64_t i = 0; i < size; i++) {
            if (std::isnan(array[i])) {
                first_nan = i;
                break;
            }
        }
        if (first_nan != -1) {
            int64_t write_pos = first_nan;
            for (int64_t i = first_nan + 1; i < size; i++) {
                if (!std::isnan(array[i])) {
                    std::swap(array[write_pos], array[i]);
                    write_pos++;
                }
            }
            valid_len = write_pos;
        }
    }

    if (valid_len <= 1) return;

    NoThrowBuffer<int64_t> sorted_k;
    const int64_t *k_ptr = k_list;
    int64_t num_k = k_size;
    if (!std::is_sorted(k_list, k_list + k_size)) {
        sorted_k.assign(k_list, k_list + k_size);
        if (!sorted_k.ok()) return;
        std::sort(sorted_k.begin(), sorted_k.end());
        k_ptr = sorted_k.data();
    }

    int64_t k_start = 0;
    while (k_start < num_k && k_ptr[k_start] < 0) k_start++;
    int64_t k_end = k_start;
    while (k_end < num_k && k_ptr[k_end] < valid_len) k_end++;

    if (k_end <= k_start) return;

    if constexpr (sizeof(T) >= 2) {
        if (valid_len >= 64) {
            if constexpr (std::is_same_v<T, float>) {
                uint32_t *u_ptr = reinterpret_cast<uint32_t *>(array);
                for (int64_t i = 0; i < valid_len; i++) {
                    u_ptr[i] = to_sortable_bits32(array[i]);
                }
                vqselect_dispatch(u_ptr, valid_len, k_ptr, k_start, k_end - 1);
                for (int64_t i = 0; i < valid_len; i++) {
                    array[i] = from_sortable_bits32(u_ptr[i]);
                }
            } else if constexpr (std::is_same_v<T, double>) {
                uint64_t *u_ptr = reinterpret_cast<uint64_t *>(array);
                for (int64_t i = 0; i < valid_len; i++) {
                    u_ptr[i] = to_sortable_bits64(array[i]);
                }
                vqselect_dispatch(u_ptr, valid_len, k_ptr, k_start, k_end - 1);
                for (int64_t i = 0; i < valid_len; i++) {
                    array[i] = from_sortable_bits64(u_ptr[i]);
                }
            } else {
                vqselect_dispatch(array, valid_len, k_ptr, k_start, k_end - 1);
            }
            return;
        }
    }

    if (k_end - k_start == 1) {
        std::nth_element(array, array + k_ptr[k_start], array + valid_len);
    } else {
        multi_nth_element(array, 0, valid_len - 1, k_ptr, k_start, k_end - 1, std::less<T>());
    }
}

template <typename T>
static void argpartition_impl(const T *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size) {
    if (data == nullptr || indices == nullptr || size <= 0 || k_list == nullptr || k_size <= 0) return;
    if (size == 1) {
        indices[0] = 0;
        return;
    }

    NoThrowBuffer<int64_t> sorted_k;
    const int64_t *k_ptr = k_list;
    int64_t num_k = k_size;
    if (!std::is_sorted(k_list, k_list + k_size)) {
        sorted_k.assign(k_list, k_list + k_size);
        if (!sorted_k.ok()) return;
        std::sort(sorted_k.begin(), sorted_k.end());
        k_ptr = sorted_k.data();
    }

    // Fast path for 16-bit and 32-bit types (zero heap allocation, in-place packed uint64_t VQSelect)
    if constexpr (sizeof(T) >= 2 && sizeof(T) <= 4) {
        if (size >= 64 && size <= 0xFFFFFFFFLL) {
            uint64_t *packed = reinterpret_cast<uint64_t *>(indices);
            int64_t valid_len = size;
            if constexpr (std::is_floating_point_v<T>) {
                int64_t left = 0;
                int64_t right = size - 1;
                for (int64_t i = 0; i < size; i++) {
                    T v = data[i];
                    if (HWY_UNLIKELY(std::isnan(v))) {
                        indices[right--] = i;
                    } else {
                        packed[left++] = (static_cast<uint64_t>(sortable_key32(v)) << 32) | static_cast<uint32_t>(i);
                    }
                }
                valid_len = left;
                if (valid_len < size) {
                    std::reverse(indices + valid_len, indices + size);
                }
            } else {
                for (int64_t i = 0; i < size; i++) {
                    packed[i] = (static_cast<uint64_t>(sortable_key32(data[i])) << 32) | static_cast<uint32_t>(i);
                }
            }
            if (valid_len <= 1) {
                if (valid_len == 1) indices[0] = static_cast<int64_t>(packed[0] & 0xFFFFFFFFULL);
                return;
            }
            int64_t k_start = 0;
            while (k_start < num_k && k_ptr[k_start] < 0) k_start++;
            int64_t k_end = k_start;
            while (k_end < num_k && k_ptr[k_end] < valid_len) k_end++;
            if (k_end > k_start) {
                vqselect_dispatch(packed, valid_len, k_ptr, k_start, k_end - 1);
            }
            for (int64_t i = 0; i < valid_len; i++) {
                indices[i] = static_cast<int64_t>(packed[i] & 0xFFFFFFFFULL);
            }
            return;
        }
    }

    // Fast path for 64-bit types
    if constexpr (sizeof(T) == 8) {
        if (size >= 64) {
            // Zero-allocation single-k fast path using indices buffer in-place as scratch
            int64_t k_0 = 0;
            while (k_0 < num_k && k_ptr[k_0] < 0) k_0++;
            int64_t k_1 = k_0;
            while (k_1 < num_k && k_ptr[k_1] < size) k_1++;

            if (k_1 - k_0 == 1) {
                T *k_buf_inplace = reinterpret_cast<T *>(indices);
                const bool has_nan = ndarray_hwy::CopyAndCheckNaN64(data, k_buf_inplace, size);
                if (HWY_LIKELY(!has_nan)) {
                    vqselect_dispatch(k_buf_inplace, size, k_ptr, k_0, k_0);
                    const T pivot = k_buf_inplace[k_ptr[k_0]];
                    ndarray_hwy::ArgPartitionAroundPivot64(data, indices, size, pivot);
                    return;
                }
            }

#if VQSORT_ENABLED
            // Multi-k or NaN-containing path for 64-bit types via VQSelect on K64V64
            NoThrowBuffer<hwy::K64V64> kv;
            if (kv.try_resize_uninit(static_cast<size_t>(size))) {
                hwy::K64V64 *kv_ptr = kv.data();
                int64_t valid_len = size;
                if constexpr (std::is_floating_point_v<T>) {
                    int64_t left = 0;
                    int64_t right = size - 1;
                    for (int64_t i = 0; i < size; i++) {
                        T v = data[i];
                        if (HWY_UNLIKELY(std::isnan(v))) {
                            indices[right--] = i;
                        } else {
                            kv_ptr[left].key = sortable_key64(v);
                            kv_ptr[left].value = static_cast<uint64_t>(i);
                            left++;
                        }
                    }
                    valid_len = left;
                    if (valid_len < size) {
                        std::reverse(indices + valid_len, indices + size);
                    }
                } else {
                    for (int64_t i = 0; i < size; i++) {
                        kv_ptr[i].key = sortable_key64(data[i]);
                        kv_ptr[i].value = static_cast<uint64_t>(i);
                    }
                }

                if (valid_len <= 1) {
                    if (valid_len == 1) indices[0] = static_cast<int64_t>(kv_ptr[0].value);
                    return;
                }

                int64_t k_start = 0;
                while (k_start < num_k && k_ptr[k_start] < 0) k_start++;
                int64_t k_end = k_start;
                while (k_end < num_k && k_ptr[k_end] < valid_len) k_end++;

                if (k_end > k_start) {
                    ndarray_hwy::VQSelectStatic(kv_ptr, static_cast<size_t>(valid_len), k_ptr, k_start, k_end - 1);
                }
                for (int64_t i = 0; i < valid_len; i++) {
                    indices[i] = static_cast<int64_t>(kv_ptr[i].value);
                }
                return;
            }
#endif
        }
    }

    // Fallback if scratch allocation fails or size < 64
    int64_t valid_len = size;
    if constexpr (std::is_floating_point_v<T>) {
        int64_t left = 0;
        int64_t right = size - 1;
        for (int64_t i = 0; i < size; i++) {
            if (std::isnan(data[i])) {
                indices[right--] = i;
            } else {
                indices[left++] = i;
            }
        }
        valid_len = left;
        if (valid_len < size) {
            std::reverse(indices + valid_len, indices + size);
        }
    } else {
        for (int64_t i = 0; i < size; i++) {
            indices[i] = i;
        }
    }

    if (valid_len <= 1) return;

    int64_t k_start = 0;
    while (k_start < num_k && k_ptr[k_start] < 0) k_start++;
    int64_t k_end = k_start;
    while (k_end < num_k && k_ptr[k_end] < valid_len) k_end++;

    if (k_end <= k_start) return;

    auto cmp = [data](int64_t a, int64_t b) {
        return data[a] < data[b];
    };

    if (k_end - k_start == 1) {
        std::nth_element(indices, indices + k_ptr[k_start], indices + valid_len, cmp);
    } else {
        arg_multi_nth_element(indices, 0, valid_len - 1, k_ptr, k_start, k_end - 1, cmp);
    }
}

template <typename T>
static void argsort_impl(const T *data, int64_t *indices, int64_t size, int kind) {
    if (data == nullptr || indices == nullptr || size <= 0) return;
    if (size == 1) {
        indices[0] = 0;
        return;
    }

    // Fast pre-check on data directly before writing indices (exits in ~2 elements on random inputs)
    bool is_sorted = true;
    bool is_rev_sorted = true;
    bool seen_nan = false;
    if constexpr (std::is_floating_point_v<T>) {
        seen_nan = std::isnan(data[0]);
    }
    if (!seen_nan) {
        for (int64_t i = 0; i < size - 1; i++) {
            T val_cur = data[i];
            T val_next = data[i + 1];
            if constexpr (std::is_floating_point_v<T>) {
                if (std::isnan(val_next)) {
                    seen_nan = true;
                    is_sorted = false;
                    is_rev_sorted = false;
                    break;
                }
            }
            if (val_cur > val_next) {
                is_sorted = false;
                if (!is_rev_sorted) break;
            }
            if (val_cur <= val_next) {
                is_rev_sorted = false;
                if (!is_sorted) break;
            }
        }
        if (is_sorted) {
            for (int64_t i = 0; i < size; i++) indices[i] = i;
            return;
        }
        if (is_rev_sorted) {
            for (int64_t i = 0; i < size; i++) indices[i] = size - 1 - i;
            return;
        }
    }

    // Fast path for 8-bit integer types (O(N) stable counting argsort, zero heap allocation)
    if constexpr (sizeof(T) == 1 && !std::is_floating_point_v<T>) {
        if (size > 32 && (kind == 0 || kind == 1)) {
            int64_t counts[257] = {0};
            for (int64_t i = 0; i < size; i++) {
                uint8_t bucket = static_cast<uint8_t>(data[i]) ^ (std::is_signed_v<T> ? 0x80u : 0x00u);
                counts[bucket + 1]++;
            }
            for (int b = 0; b < 256; b++) {
                counts[b + 1] += counts[b];
            }
            for (int64_t i = 0; i < size; i++) {
                uint8_t bucket = static_cast<uint8_t>(data[i]) ^ (std::is_signed_v<T> ? 0x80u : 0x00u);
                indices[counts[bucket]++] = i;
            }
            return;
        }
    }

    // Fast path for 16-bit and 32-bit types: single pass packs (key << 32) | idx into front of indices
    // and places NaN indices at back, followed by in-place VQSort (zero heap allocation).
    if constexpr (sizeof(T) >= 2 && sizeof(T) <= 4) {
        if (size > 32 && size <= 0xFFFFFFFFLL && (kind == 0 || kind == 1)) {
            uint64_t *packed = reinterpret_cast<uint64_t *>(indices);
            int64_t valid_len = size;
            if constexpr (std::is_floating_point_v<T>) {
                int64_t left = 0;
                int64_t right = size - 1;
                for (int64_t i = 0; i < size; i++) {
                    T v = data[i];
                    if (HWY_UNLIKELY(std::isnan(v))) {
                        indices[right--] = i;
                    } else {
                        packed[left++] = (static_cast<uint64_t>(sortable_key32(v)) << 32) | static_cast<uint32_t>(i);
                    }
                }
                valid_len = left;
                if (valid_len < size) {
                    std::reverse(indices + valid_len, indices + size);
                }
            } else {
                for (int64_t i = 0; i < size; i++) {
                    packed[i] = (static_cast<uint64_t>(sortable_key32(data[i])) << 32) | static_cast<uint32_t>(i);
                }
            }
            if (valid_len > 1) {
                hwy::VQSort(packed, static_cast<size_t>(valid_len), hwy::SortAscending());
            }
            for (int64_t i = 0; i < valid_len; i++) {
                indices[i] = static_cast<int64_t>(packed[i] & 0xFFFFFFFFULL);
            }
            return;
        }
    }

    // Fast path for 64-bit types: single pass fills K64V64/uint128_t for non-NaNs and places NaNs at back of indices.
    if constexpr (sizeof(T) == 8) {
        if (size > 32 && (kind == 0 || kind == 1)) {
            constexpr size_t kMaxThreadLocalScratch = 65536;
            if (kind == 0) {
                static thread_local NoThrowBuffer<hwy::K64V64> tl_kv;
                NoThrowBuffer<hwy::K64V64> heap_kv;
                NoThrowBuffer<hwy::K64V64> &kv = (static_cast<size_t>(size) <= kMaxThreadLocalScratch) ? tl_kv : heap_kv;
                if (kv.size() >= static_cast<size_t>(size) || kv.try_resize_uninit(static_cast<size_t>(size))) {
                    hwy::K64V64 *kv_ptr = kv.data();
                    int64_t valid_len = size;
                    if constexpr (std::is_floating_point_v<T>) {
                        const int64_t first_nan = seen_nan ? 0 : ndarray_hwy::FindFirstNaN(data, size);
                        if (HWY_LIKELY(first_nan == -1)) {
                            for (int64_t i = 0; i < size; i++) {
                                kv_ptr[i].key = sortable_key64(data[i]);
                                kv_ptr[i].value = static_cast<uint64_t>(i);
                            }
                        } else {
                            for (int64_t i = 0; i < first_nan; i++) {
                                kv_ptr[i].key = sortable_key64(data[i]);
                                kv_ptr[i].value = static_cast<uint64_t>(i);
                            }
                            int64_t left = first_nan;
                            int64_t right = size - 1;
                            for (int64_t i = first_nan; i < size; i++) {
                                T v = data[i];
                                if (HWY_UNLIKELY(std::isnan(v))) {
                                    indices[right--] = i;
                                } else {
                                    kv_ptr[left].key = sortable_key64(v);
                                    kv_ptr[left].value = static_cast<uint64_t>(i);
                                    left++;
                                }
                            }
                            valid_len = left;
                            if (valid_len < size) {
                                std::reverse(indices + valid_len, indices + size);
                            }
                        }
                    } else {
                        for (int64_t i = 0; i < size; i++) {
                            kv_ptr[i].key = sortable_key64(data[i]);
                            kv_ptr[i].value = static_cast<uint64_t>(i);
                        }
                    }
                    if (valid_len > 1) {
                        hwy::VQSort(kv_ptr, static_cast<size_t>(valid_len), hwy::SortAscending());
                    }
                    for (int64_t i = 0; i < valid_len; i++) {
                        indices[i] = static_cast<int64_t>(kv_ptr[i].value);
                    }
                    return;
                }
            } else {
                static thread_local NoThrowBuffer<hwy::uint128_t> tl_u128;
                NoThrowBuffer<hwy::uint128_t> heap_u128;
                NoThrowBuffer<hwy::uint128_t> &u128 = (static_cast<size_t>(size) <= kMaxThreadLocalScratch) ? tl_u128 : heap_u128;
                if (u128.size() >= static_cast<size_t>(size) || u128.try_resize_uninit(static_cast<size_t>(size))) {
                    hwy::uint128_t *u_ptr = u128.data();
                    int64_t valid_len = size;
                    if constexpr (std::is_floating_point_v<T>) {
                        const int64_t first_nan = seen_nan ? 0 : ndarray_hwy::FindFirstNaN(data, size);
                        if (HWY_LIKELY(first_nan == -1)) {
                            for (int64_t i = 0; i < size; i++) {
                                u_ptr[i].hi = sortable_key64(data[i]);
                                u_ptr[i].lo = static_cast<uint64_t>(i);
                            }
                        } else {
                            for (int64_t i = 0; i < first_nan; i++) {
                                u_ptr[i].hi = sortable_key64(data[i]);
                                u_ptr[i].lo = static_cast<uint64_t>(i);
                            }
                            int64_t left = first_nan;
                            int64_t right = size - 1;
                            for (int64_t i = first_nan; i < size; i++) {
                                T v = data[i];
                                if (HWY_UNLIKELY(std::isnan(v))) {
                                    indices[right--] = i;
                                } else {
                                    u_ptr[left].hi = sortable_key64(v);
                                    u_ptr[left].lo = static_cast<uint64_t>(i);
                                    left++;
                                }
                            }
                            valid_len = left;
                            if (valid_len < size) {
                                std::reverse(indices + valid_len, indices + size);
                            }
                        }
                    } else {
                        for (int64_t i = 0; i < size; i++) {
                            u_ptr[i].hi = sortable_key64(data[i]);
                            u_ptr[i].lo = static_cast<uint64_t>(i);
                        }
                    }
                    if (valid_len > 1) {
                        hwy::VQSort(u_ptr, static_cast<size_t>(valid_len), hwy::SortAscending());
                    }
                    for (int64_t i = 0; i < valid_len; i++) {
                        indices[i] = static_cast<int64_t>(u_ptr[i].lo);
                    }
                    return;
                }
            }
        }
    }

    int64_t valid_len = size;
    if constexpr (std::is_floating_point_v<T>) {
        int64_t left = 0;
        int64_t right = size - 1;
        for (int64_t i = 0; i < size; i++) {
            if (std::isnan(data[i])) {
                indices[right--] = i;
            } else {
                indices[left++] = i;
            }
        }
        valid_len = left;
        if (valid_len < size) {
            std::reverse(indices + valid_len, indices + size);
        }
    } else {
        for (int64_t i = 0; i < size; i++) {
            indices[i] = i;
        }
    }

    if (valid_len <= 1) return;

    auto cmp = [data](int64_t a, int64_t b) {
        return data[a] < data[b];
    };

    if (kind == 0) {
        std::sort(indices, indices + valid_len, cmp);
    } else if (kind == 1) {
        std::stable_sort(indices, indices + valid_len, cmp);
    } else {
        std::make_heap(indices, indices + valid_len, cmp);
        std::sort_heap(indices, indices + valid_len, cmp);
    }
}

template <typename T>
static void sort_float_impl(T *array, int64_t size, int kind) {
    if (array == nullptr || size <= 1) return;

    const int64_t first_nan = ndarray_hwy::FindFirstNaN(array, size);
    int64_t non_nan_size = size;
    if (first_nan != -1) {
        if (kind == 1) { // stable sort: preserve NaN order
            NoThrowBuffer<T> nans;
            if (!nans.push_back(array[first_nan])) return;
            int64_t write_pos = first_nan;
            for (int64_t i = first_nan + 1; i < size; i++) {
                if (std::isnan(array[i])) {
                    if (!nans.push_back(array[i])) return;
                } else {
                    array[write_pos++] = array[i];
                }
            }
            for (size_t i = 0; i < nans.size(); i++) {
                array[write_pos + i] = nans[i];
            }
            non_nan_size = write_pos;
        } else { // unstable sort: in-place partition
            int64_t write_pos = first_nan;
            for (int64_t i = first_nan + 1; i < size; i++) {
                if (!std::isnan(array[i])) {
                    std::swap(array[write_pos], array[i]);
                    write_pos++;
                }
            }
            non_nan_size = write_pos;
        }
    }

    if (non_nan_size <= 1) return;

    bool is_sorted = true;
    bool is_rev_sorted = true;
    for (int64_t i = 0; i < non_nan_size - 1; i++) {
        if (array[i] > array[i + 1]) {
            is_sorted = false;
            if (!is_rev_sorted) break;
        }
        if (array[i] <= array[i + 1]) {
            is_rev_sorted = false;
            if (!is_sorted) break;
        }
    }
    if (is_sorted) return;
    if (is_rev_sorted) {
        std::reverse(array, array + non_nan_size);
        return;
    }

    if (kind == 0) {
        if (non_nan_size < 128) {
            std::sort(array, array + non_nan_size);
        } else {
            hwy::VQSort(array, static_cast<size_t>(non_nan_size), hwy::SortAscending());
        }
    } else if (kind == 2) {
        std::make_heap(array, array + non_nan_size);
        std::sort_heap(array, array + non_nan_size);
    } else {
        std::stable_sort(array, array + non_nan_size);
    }
}

template <typename T>
static void sort_int_impl(T *array, int64_t size, int kind) {
    if (array == nullptr || size <= 1) return;

    bool is_sorted = true;
    bool is_rev_sorted = true;
    for (int64_t i = 0; i < size - 1; i++) {
        if (array[i] > array[i + 1]) {
            is_sorted = false;
            if (!is_rev_sorted) break;
        }
        if (array[i] <= array[i + 1]) {
            is_rev_sorted = false;
            if (!is_sorted) break;
        }
    }
    if (is_sorted) return;
    if (is_rev_sorted) {
        std::reverse(array, array + size);
        return;
    }

    if (kind == 0) {
        if (size < 128) {
            std::sort(array, array + size);
        } else {
            if constexpr (std::is_same_v<T, long long> || std::is_same_v<T, int64_t>) {
                hwy::VQSort((int64_t *)array, static_cast<size_t>(size), hwy::SortAscending());
            } else if constexpr (std::is_same_v<T, unsigned long long> || std::is_same_v<T, uint64_t>) {
                hwy::VQSort((uint64_t *)array, static_cast<size_t>(size), hwy::SortAscending());
            } else {
                hwy::VQSort(array, static_cast<size_t>(size), hwy::SortAscending());
            }
        }
    } else if (kind == 2) {
        std::make_heap(array, array + size);
        std::sort_heap(array, array + size);
    } else {
        std::stable_sort(array, array + size);
    }
}


template <typename T, typename Compare>
static void searchsorted(const T *arr, int64_t size, const T *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter, Compare cmp) {
    if (arr == nullptr || values == nullptr || out_indices == nullptr || num_values <= 0) return;
    for (int64_t v_idx = 0; v_idx < num_values; v_idx++) {
        T val = values[v_idx];
        int64_t low = 0;
        int64_t high = size;
        while (low < high) {
            int64_t mid = low + (high - low) / 2;
            T mid_val = (sorter != nullptr) ? arr[sorter[mid]] : arr[mid];
            int comp = cmp(mid_val, val);
            if (side_left) {
                if (comp < 0) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            } else {
                if (comp <= 0) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
        }
        out_indices[v_idx] = low;
    }
}

HWY_BEFORE_NAMESPACE();
namespace ndarray_hwy {

template <typename T>
static int64_t CountNonZeroContiguous(const T *src, int64_t n) {
    int64_t count = 0;
    int64_t i = 0;
    if constexpr (std::is_same_v<T, double>) {
        const hn::ScalableTag<double> df;
        using DI = hn::Rebind<int64_t, decltype(df)>;
        using DU = hn::Rebind<uint64_t, decltype(df)>;
        const DI di;
        const DU du;
        const int64_t L = hn::Lanes(df);
        const auto abs_mask = hn::Set(du, 0x7FFFFFFFFFFFFFFFULL);
        const auto zero = hn::Zero(du);
        auto acc0 = hn::Zero(di);
        auto acc1 = hn::Zero(di);
        auto acc2 = hn::Zero(di);
        auto acc3 = hn::Zero(di);
        for (; i + 4 * L <= n; i += 4 * L) {
            auto v0 = hn::And(hn::BitCast(du, hn::LoadU(df, src + i)), abs_mask);
            auto v1 = hn::And(hn::BitCast(du, hn::LoadU(df, src + i + L)), abs_mask);
            auto v2 = hn::And(hn::BitCast(du, hn::LoadU(df, src + i + 2 * L)), abs_mask);
            auto v3 = hn::And(hn::BitCast(du, hn::LoadU(df, src + i + 3 * L)), abs_mask);
            acc0 = hn::Sub(acc0, hn::BitCast(di, hn::VecFromMask(du, v0 != zero)));
            acc1 = hn::Sub(acc1, hn::BitCast(di, hn::VecFromMask(du, v1 != zero)));
            acc2 = hn::Sub(acc2, hn::BitCast(di, hn::VecFromMask(du, v2 != zero)));
            acc3 = hn::Sub(acc3, hn::BitCast(di, hn::VecFromMask(du, v3 != zero)));
        }
        count += hn::ReduceSum(di, hn::Add(hn::Add(acc0, acc1), hn::Add(acc2, acc3)));
        for (; i + L <= n; i += L) {
            auto v = hn::And(hn::BitCast(du, hn::LoadU(df, src + i)), abs_mask);
            count += hn::CountTrue(du, v != zero);
        }
    } else if constexpr (std::is_same_v<T, float>) {
        const hn::ScalableTag<float> df;
        using DU = hn::Rebind<uint32_t, decltype(df)>;
        const DU du;
        const int64_t L = hn::Lanes(df);
        const auto abs_mask = hn::Set(du, 0x7FFFFFFFU);
        const auto zero = hn::Zero(du);
        for (; i + L <= n; i += L) {
            auto v = hn::And(hn::BitCast(du, hn::LoadU(df, src + i)), abs_mask);
            count += hn::CountTrue(du, v != zero);
        }
    } else if constexpr (std::is_integral_v<T>) {
        using LaneT = std::conditional_t<
            sizeof(T) == 8, int64_t,
            std::conditional_t<sizeof(T) == 4, int32_t,
            std::conditional_t<sizeof(T) == 2, int16_t, uint8_t>>>;
        const hn::ScalableTag<LaneT> d;
        const int64_t L = hn::Lanes(d);
        const LaneT *p = reinterpret_cast<const LaneT *>(src);
        const auto zero = hn::Zero(d);
        for (; i + 4 * L <= n; i += 4 * L) {
            count += hn::CountTrue(d, hn::LoadU(d, p + i) != zero) +
                     hn::CountTrue(d, hn::LoadU(d, p + i + L) != zero) +
                     hn::CountTrue(d, hn::LoadU(d, p + i + 2 * L) != zero) +
                     hn::CountTrue(d, hn::LoadU(d, p + i + 3 * L) != zero);
        }
        for (; i + L <= n; i += L) {
            count += hn::CountTrue(d, hn::LoadU(d, p + i) != zero);
        }
    }
    for (; i < n; i++) {
        if (is_nonzero(src[i])) count++;
    }
    return count;
}

template <typename T>
static void ToBoolMaskContiguous(const T *src, int64_t n, unsigned char *dest) {
    int64_t i = 0;
    if constexpr (std::is_same_v<T, double> || (std::is_integral_v<T> && sizeof(T) == 8)) {
        const hn::ScalableTag<uint64_t> du;
        using DU32 = hn::Rebind<uint32_t, decltype(du)>;
        using DU8 = hn::Rebind<uint8_t, decltype(du)>;
        const DU32 du32;
        const DU8 du8;
        const int64_t L = hn::Lanes(du);
        const uint64_t *p = reinterpret_cast<const uint64_t *>(src);
        const auto zero = hn::Zero(du);
        const auto one = hn::Set(du, 1);
        const auto abs_mask = hn::Set(du, std::is_same_v<T, double> ? 0x7FFFFFFFFFFFFFFFULL : 0xFFFFFFFFFFFFFFFFULL);
#if HWY_TARGET != HWY_SCALAR
        using DU32Repart = hn::Repartition<uint32_t, decltype(du)>;
        using DU16Repart = hn::Repartition<uint16_t, decltype(du)>;
        using DU8Half = hn::Rebind<uint8_t, DU16Repart>;
        const DU32Repart du32_repart;
        const DU16Repart du16_repart;
        const DU8Half du8_half;
        for (; i + 4 * L <= n; i += 4 * L) {
            const auto v0 = hn::And(hn::VecFromMask(du, hn::And(hn::LoadU(du, p + i), abs_mask) != zero), one);
            const auto v1 = hn::And(hn::VecFromMask(du, hn::And(hn::LoadU(du, p + i + L), abs_mask) != zero), one);
            const auto v2 = hn::And(hn::VecFromMask(du, hn::And(hn::LoadU(du, p + i + 2 * L), abs_mask) != zero), one);
            const auto v3 = hn::And(hn::VecFromMask(du, hn::And(hn::LoadU(du, p + i + 3 * L), abs_mask) != zero), one);
            const auto v32_01 = hn::OrderedDemote2To(du32_repart, v0, v1);
            const auto v32_23 = hn::OrderedDemote2To(du32_repart, v2, v3);
            const auto v16_0123 = hn::OrderedDemote2To(du16_repart, v32_01, v32_23);
            const auto v8_0123 = hn::DemoteTo(du8_half, v16_0123);
            hn::StoreU(v8_0123, du8_half, dest + i);
        }
#endif
        for (; i + L <= n; i += L) {
            const auto v = hn::And(hn::LoadU(du, p + i), abs_mask);
            const auto v01 = hn::And(hn::VecFromMask(du, v != zero), one);
            const auto v8 = hn::DemoteTo(du8, hn::DemoteTo(du32, v01));
            hn::StoreU(v8, du8, dest + i);
        }
    } else if constexpr (std::is_same_v<T, float> || (std::is_integral_v<T> && sizeof(T) == 4)) {
        const hn::ScalableTag<uint32_t> du;
        using DU8 = hn::Rebind<uint8_t, decltype(du)>;
        const DU8 du8;
        const int64_t L = hn::Lanes(du);
        const uint32_t *p = reinterpret_cast<const uint32_t *>(src);
        const auto zero = hn::Zero(du);
        const auto one = hn::Set(du, 1);
        const auto abs_mask = hn::Set(du, std::is_same_v<T, float> ? 0x7FFFFFFFU : 0xFFFFFFFFU);
#if HWY_TARGET != HWY_SCALAR
        using DU16Repart = hn::Repartition<uint16_t, decltype(du)>;
        using DU8Repart = hn::Repartition<uint8_t, decltype(du)>;
        const DU16Repart du16_repart;
        const DU8Repart du8_repart;
        for (; i + 4 * L <= n; i += 4 * L) {
            const auto v0 = hn::And(hn::VecFromMask(du, hn::And(hn::LoadU(du, p + i), abs_mask) != zero), one);
            const auto v1 = hn::And(hn::VecFromMask(du, hn::And(hn::LoadU(du, p + i + L), abs_mask) != zero), one);
            const auto v2 = hn::And(hn::VecFromMask(du, hn::And(hn::LoadU(du, p + i + 2 * L), abs_mask) != zero), one);
            const auto v3 = hn::And(hn::VecFromMask(du, hn::And(hn::LoadU(du, p + i + 3 * L), abs_mask) != zero), one);
            const auto v16_01 = hn::OrderedDemote2To(du16_repart, v0, v1);
            const auto v16_23 = hn::OrderedDemote2To(du16_repart, v2, v3);
            const auto v8_0123 = hn::OrderedDemote2To(du8_repart, v16_01, v16_23);
            hn::StoreU(v8_0123, du8_repart, dest + i);
        }
#endif
        for (; i + L <= n; i += L) {
            const auto v = hn::And(hn::LoadU(du, p + i), abs_mask);
            const auto v01 = hn::And(hn::VecFromMask(du, v != zero), one);
            const auto v8 = hn::DemoteTo(du8, v01);
            hn::StoreU(v8, du8, dest + i);
        }
    } else if constexpr (std::is_integral_v<T> && sizeof(T) == 2) {
        const hn::ScalableTag<uint16_t> du;
        using DU8 = hn::Rebind<uint8_t, decltype(du)>;
        const DU8 du8;
        const int64_t L = hn::Lanes(du);
        const uint16_t *p = reinterpret_cast<const uint16_t *>(src);
        const auto zero = hn::Zero(du);
        const auto one = hn::Set(du, 1);
        for (; i + L <= n; i += L) {
            const auto v = hn::LoadU(du, p + i);
            const auto v01 = hn::And(hn::VecFromMask(du, v != zero), one);
            const auto v8 = hn::DemoteTo(du8, v01);
            hn::StoreU(v8, du8, dest + i);
        }
    } else if constexpr (std::is_integral_v<T> && sizeof(T) == 1) {
        const hn::ScalableTag<uint8_t> du;
        const int64_t L = hn::Lanes(du);
        const uint8_t *p = reinterpret_cast<const uint8_t *>(src);
        const auto zero = hn::Zero(du);
        const auto one = hn::Set(du, 1);
        for (; i + L <= n; i += L) {
            const auto v = hn::LoadU(du, p + i);
            const auto v01 = hn::And(hn::VecFromMask(du, v != zero), one);
            hn::StoreU(v01, du, dest + i);
        }
    }
    for (; i < n; i++) {
        dest[i] = is_nonzero(src[i]) ? 1 : 0;
    }
}

static int64_t UnpackMaskImpl(const uint8_t *mask_ptr, int64_t size, int64_t *out_indices) {
    const hn::ScalableTag<int64_t> d;
    using Rebind8 = hn::Rebind<uint8_t, decltype(d)>;
    const Rebind8 d8;
    const int64_t L = hn::Lanes(d);

    int64_t total_count = 0;
    int64_t j = 0;
    if (L <= 64) {
        alignas(64) int64_t buf[128];
        auto v_base = hn::Iota(d, 0);
        const auto zero = hn::Zero(d);

        while (j + L <= size) {
            int64_t chunk_end = std::min<int64_t>(size - L, j + 64 - L);
            int64_t chunk_count = 0;
            for (; j <= chunk_end; j += L) {
                auto mask_bytes = hn::LoadU(d8, mask_ptr + j);
                auto mask_i64 = hn::PromoteTo(d, mask_bytes);
                auto mask = (mask_i64 != zero);
                auto v_index = hn::Add(v_base, hn::Set(d, j));
                chunk_count += hn::CompressStore(v_index, mask, d, buf + chunk_count);
            }
            if (chunk_count > 0) {
                std::memcpy(out_indices + total_count, buf, static_cast<size_t>(chunk_count) * sizeof(int64_t));
                total_count += chunk_count;
            }
        }
    }
    for (; j < size; j++) {
        if (mask_ptr[j] != 0) {
            out_indices[total_count++] = j;
        }
    }
    return total_count;
}

template <typename T>
static int64_t ArgMinMax1D(const T *src, int64_t n, int is_max) {
    T best_val = src[0];
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(best_val)) return 0;
    }
    int64_t best_idx = 0;
    int64_t i = 1;

    const hn::ScalableTag<T> d;
    const int64_t L = hn::Lanes(d);

    if (n >= 1 + 4 * L) {
        auto v_best = hn::Set(d, best_val);
        if (is_max) {
            for (; i + 4 * L <= n; i += 4 * L) {
                const auto v0 = hn::LoadU(d, src + i);
                const auto v1 = hn::LoadU(d, src + i + L);
                const auto v2 = hn::LoadU(d, src + i + 2 * L);
                const auto v3 = hn::LoadU(d, src + i + 3 * L);
                if constexpr (std::is_floating_point_v<T>) {
                    const auto vmax = hn::Max(hn::Max(v0, v1), hn::Max(v2, v3));
                    const auto gt = hn::Gt(vmax, v_best);
                    const auto nan01 = hn::Or(hn::IsNaN(v0), hn::IsNaN(v1));
                    const auto nan23 = hn::Or(hn::IsNaN(v2), hn::IsNaN(v3));
                    const auto any = hn::Or(gt, hn::Or(nan01, nan23));
                    if (HWY_UNLIKELY(!hn::AllFalse(d, any))) {
                        for (int64_t k = i; k < i + 4 * L; k++) {
                            T val = src[k];
                            if (std::isnan(val)) return k;
                            if (val > best_val) {
                                best_val = val;
                                best_idx = k;
                            }
                        }
                        v_best = hn::Set(d, best_val);
                    }
                } else {
                    const auto vmax = hn::Max(hn::Max(v0, v1), hn::Max(v2, v3));
                    if (HWY_UNLIKELY(!hn::AllFalse(d, hn::Gt(vmax, v_best)))) {
                        for (int64_t k = i; k < i + 4 * L; k++) {
                            T val = src[k];
                            if (val > best_val) {
                                best_val = val;
                                best_idx = k;
                            }
                        }
                        v_best = hn::Set(d, best_val);
                    }
                }
            }
        } else {
            for (; i + 4 * L <= n; i += 4 * L) {
                const auto v0 = hn::LoadU(d, src + i);
                const auto v1 = hn::LoadU(d, src + i + L);
                const auto v2 = hn::LoadU(d, src + i + 2 * L);
                const auto v3 = hn::LoadU(d, src + i + 3 * L);
                if constexpr (std::is_floating_point_v<T>) {
                    const auto vmin = hn::Min(hn::Min(v0, v1), hn::Min(v2, v3));
                    const auto lt = hn::Lt(vmin, v_best);
                    const auto nan01 = hn::Or(hn::IsNaN(v0), hn::IsNaN(v1));
                    const auto nan23 = hn::Or(hn::IsNaN(v2), hn::IsNaN(v3));
                    const auto any = hn::Or(lt, hn::Or(nan01, nan23));
                    if (HWY_UNLIKELY(!hn::AllFalse(d, any))) {
                        for (int64_t k = i; k < i + 4 * L; k++) {
                            T val = src[k];
                            if (std::isnan(val)) return k;
                            if (val < best_val) {
                                best_val = val;
                                best_idx = k;
                            }
                        }
                        v_best = hn::Set(d, best_val);
                    }
                } else {
                    const auto vmin = hn::Min(hn::Min(v0, v1), hn::Min(v2, v3));
                    if (HWY_UNLIKELY(!hn::AllFalse(d, hn::Lt(vmin, v_best)))) {
                        for (int64_t k = i; k < i + 4 * L; k++) {
                            T val = src[k];
                            if (val < best_val) {
                                best_val = val;
                                best_idx = k;
                            }
                        }
                        v_best = hn::Set(d, best_val);
                    }
                }
            }
        }
    }

    if (is_max) {
        for (; i < n; i++) {
            T val = src[i];
            if constexpr (std::is_floating_point_v<T>) {
                if (std::isnan(val)) return i;
            }
            if (val > best_val) {
                best_val = val;
                best_idx = i;
            }
        }
    } else {
        for (; i < n; i++) {
            T val = src[i];
            if constexpr (std::is_floating_point_v<T>) {
                if (std::isnan(val)) return i;
            }
            if (val < best_val) {
                best_val = val;
                best_idx = i;
            }
        }
    }
    return best_idx;
}

} // namespace ndarray_hwy
HWY_AFTER_NAMESPACE();

template <typename T, typename Compare>
static void argminmax(
    const T *src,
    const int64_t *stridesSrc,
    int64_t *dest,
    const int64_t *stridesDest,
    const int64_t *shape,
    int rank,
    int axis,
    int is_max,
    int is_contiguous,
    Compare cmp
) {
    if (src == nullptr || dest == nullptr || shape == nullptr || stridesSrc == nullptr || stridesDest == nullptr || rank <= 0) return;
    if (is_contiguous && axis == -1) {
        int64_t n = shape[0];
        if (n <= 0) return;
        dest[0] = ndarray_hwy::ArgMinMax1D<T>(src, n, is_max);
        return;
    }
    if (axis < 0 || axis >= rank || shape[axis] <= 0) return;
    int64_t dest_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) {
            if (shape[d] <= 0) return;
            dest_size *= shape[d];
        }
    }
    NoThrowBuffer<int64_t> coord_dest_vec, strides_dest_vec, shape_dest_vec;
    int64_t coord_dest_stack[32] = {0};
    int64_t strides_dest_stack[32] = {0};
    int64_t shape_dest_stack[32] = {0};
    int64_t *coord_dest = coord_dest_stack;
    int64_t *strides_dest_clean = strides_dest_stack;
    int64_t *shape_dest_clean = shape_dest_stack;
    if (rank > 32) {
        coord_dest_vec.assign(rank, 0);
        strides_dest_vec.assign(rank, 0);
        shape_dest_vec.assign(rank, 0);
        if (!coord_dest_vec.ok() || !strides_dest_vec.ok() || !shape_dest_vec.ok()) return;
        coord_dest = coord_dest_vec.data();
        strides_dest_clean = strides_dest_vec.data();
        shape_dest_clean = shape_dest_vec.data();
    }
    int rank_dest = 0;
    for (int d = 0; d < rank; d++) {
        if (d != axis) {
            shape_dest_clean[rank_dest] = shape[d];
            strides_dest_clean[rank_dest] = stridesDest[rank_dest];
            rank_dest++;
        }
    }
    const int64_t axis_len = shape[axis];
    const int64_t axis_stride = stridesSrc[axis];
    for (int64_t el = 0; el < dest_size; el++) {
        int64_t dest_offset = 0;
        for (int d = 0; d < rank_dest; d++) {
            dest_offset += coord_dest[d] * strides_dest_clean[d];
        }
        int64_t base_src_offset = 0;
        int rank_dest_idx = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                base_src_offset += coord_dest[rank_dest_idx] * stridesSrc[d];
                rank_dest_idx++;
            }
        }
        if (axis_stride == 1) {
            dest[dest_offset] = ndarray_hwy::ArgMinMax1D<T>(src + base_src_offset, axis_len, is_max);
        } else {
            int64_t best_idx = 0;
            T best_val = src[base_src_offset];
            if (is_nan_check(best_val)) {
                dest[dest_offset] = 0;
            } else {
                for (int64_t i = 1; i < axis_len; i++) {
                    int64_t src_offset = base_src_offset + i * axis_stride;
                    T val = src[src_offset];
                    if (is_nan_check(val)) {
                        best_idx = i;
                        break;
                    }
                    if (is_max) {
                        if (cmp(val, best_val) > 0) {
                            best_val = val;
                            best_idx = i;
                        }
                    } else {
                        if (cmp(val, best_val) < 0) {
                            best_val = val;
                            best_idx = i;
                        }
                    }
                }
                dest[dest_offset] = best_idx;
            }
        }
        if (rank_dest > 0) {
            for (int d = rank_dest - 1; d >= 0; d--) {
                coord_dest[d]++;
                if (coord_dest[d] < shape_dest_clean[d]) break;
                coord_dest[d] = 0;
            }
        }
    }
}

template <typename T>
static void count_nonzero(
    const T *src,
    const int64_t *stridesSrc,
    int64_t *dest,
    const int64_t *stridesDest,
    const int64_t *shape,
    int rank,
    int axis,
    int is_contiguous
) {
    if (src == nullptr || dest == nullptr || shape == nullptr || stridesSrc == nullptr || stridesDest == nullptr || rank <= 0) return;
    if (axis < 0) {
        if (is_contiguous) {
            dest[0] = ndarray_hwy::CountNonZeroContiguous(src, shape[0]);
            return;
        }
        int64_t total_size = 1;
        for (int d = 0; d < rank; d++) {
            if (shape[d] <= 0) {
                dest[0] = 0;
                return;
            }
            total_size *= shape[d];
        }
        NoThrowBuffer<int64_t> coord_vec;
        int64_t coord_stack[32] = {0};
        int64_t *coord = coord_stack;
        if (rank > 32) {
            coord_vec.assign(rank, 0);
            if (!coord_vec.ok()) return;
            coord = coord_vec.data();
        }
        int64_t count = 0;
        int64_t offset = 0;
        for (int64_t el = 0; el < total_size; el++) {
            if (is_nonzero(src[offset])) count++;
            for (int d = rank - 1; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offset += stridesSrc[d];
                    break;
                }
                coord[d] = 0;
                offset -= (shape[d] - 1) * stridesSrc[d];
            }
        }
        dest[0] = count;
        return;
    }
    if (axis >= rank) return;
    int64_t dest_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) {
            if (shape[d] <= 0) return;
            dest_size *= shape[d];
        }
    }
    NoThrowBuffer<int64_t> coord_dest_vec, strides_dest_vec, shape_dest_vec;
    int64_t coord_dest_stack[32] = {0};
    int64_t strides_dest_stack[32] = {0};
    int64_t shape_dest_stack[32] = {0};
    int64_t *coord_dest = coord_dest_stack;
    int64_t *strides_dest_clean = strides_dest_stack;
    int64_t *shape_dest_clean = shape_dest_stack;
    if (rank > 32) {
        coord_dest_vec.assign(rank, 0);
        strides_dest_vec.assign(rank, 0);
        shape_dest_vec.assign(rank, 0);
        if (!coord_dest_vec.ok() || !strides_dest_vec.ok() || !shape_dest_vec.ok()) return;
        coord_dest = coord_dest_vec.data();
        strides_dest_clean = strides_dest_vec.data();
        shape_dest_clean = shape_dest_vec.data();
    }
    int rank_dest = 0;
    for (int d = 0; d < rank; d++) {
        if (d != axis) {
            shape_dest_clean[rank_dest] = shape[d];
            strides_dest_clean[rank_dest] = stridesDest[rank_dest];
            rank_dest++;
        }
    }
    for (int64_t el = 0; el < dest_size; el++) {
        int64_t dest_offset = 0;
        for (int d = 0; d < rank_dest; d++) {
            dest_offset += coord_dest[d] * strides_dest_clean[d];
        }
        int64_t count = 0;
        int64_t base_src_offset = 0;
        int rank_dest_idx = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                base_src_offset += coord_dest[rank_dest_idx] * stridesSrc[d];
                rank_dest_idx++;
            }
        }
        for (int64_t i = 0; i < shape[axis]; i++) {
            int64_t src_offset = base_src_offset + i * stridesSrc[axis];
            if (is_nonzero(src[src_offset])) count++;
        }
        dest[dest_offset] = count;
        if (rank_dest > 0) {
            for (int d = rank_dest - 1; d >= 0; d--) {
                coord_dest[d]++;
                if (coord_dest[d] < shape_dest_clean[d]) break;
                coord_dest[d] = 0;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Public Sorters with Kind Routing
// ----------------------------------------------------------------------------

extern "C" void native_sort_double(double *array, int64_t size, int kind) {
    sort_float_impl(array, size, kind);
}

extern "C" void native_sort_float(float *array, int64_t size, int kind) {
    sort_float_impl(array, size, kind);
}

extern "C" void native_sort_int64(long long *array, int64_t size, int kind) {
    sort_int_impl(array, size, kind);
}

extern "C" void native_sort_int32(int *array, int64_t size, int kind) {
    sort_int_impl(array, size, kind);
}

extern "C" void native_sort_int16(int16_t *array, int64_t size, int kind) {
    sort_int_impl(array, size, kind);
}

extern "C" void native_sort_uint8(uint8_t *array, int64_t size, int kind) {
    if (array == nullptr || size <= 1) return;

    bool is_sorted = true;
    bool is_rev_sorted = true;
    for (int64_t i = 0; i < size - 1; i++) {
        if (array[i] > array[i + 1]) {
            is_sorted = false;
            if (!is_rev_sorted) break;
        }
        if (array[i] <= array[i + 1]) {
            is_rev_sorted = false;
            if (!is_sorted) break;
        }
    }
    if (is_sorted) return;
    if (is_rev_sorted) {
        std::reverse(array, array + size);
        return;
    }

    if (kind == 0 || kind == 1) {
        if (size > 32) {
            int64_t counts[256] = {0};
            for (int64_t i = 0; i < size; i++) {
                counts[array[i]]++;
            }
            int64_t idx = 0;
            for (int val = 0; val < 256; val++) {
                int64_t c = counts[val];
                if (c > 0) {
                    memset(array + idx, val, static_cast<size_t>(c));
                    idx += c;
                }
            }
        } else {
            std::sort(array, array + size);
        }
    } else if (kind == 2) {
        std::make_heap(array, array + size);
        std::sort_heap(array, array + size);
    }
}

extern "C" void native_sort_complex128(double *array, int64_t size, int kind) {
    if (array == nullptr || size <= 1) return;
    complex128_t *carr = (complex128_t *)array;
    if (kind == 0) {
        std::sort(carr, carr + size, comp_complex_impl<complex128_t>);
    } else if (kind == 2) {
        std::make_heap(carr, carr + size, comp_complex_impl<complex128_t>);
        std::sort_heap(carr, carr + size, comp_complex_impl<complex128_t>);
    } else {
        std::stable_sort(carr, carr + size, comp_complex_impl<complex128_t>);
    }
}

extern "C" void native_sort_complex64(float *array, int64_t size, int kind) {
    if (array == nullptr || size <= 1) return;
    complex64_t *carr = (complex64_t *)array;
    if (kind == 0) {
        std::sort(carr, carr + size, comp_complex_impl<complex64_t>);
    } else if (kind == 2) {
        std::make_heap(carr, carr + size, comp_complex_impl<complex64_t>);
        std::sort_heap(carr, carr + size, comp_complex_impl<complex64_t>);
    } else {
        std::stable_sort(carr, carr + size, comp_complex_impl<complex64_t>);
    }
}

extern "C" void native_sort_int8(int8_t *array, int64_t size, int kind) {
    if (array == nullptr || size <= 1) return;

    bool is_sorted = true;
    bool is_rev_sorted = true;
    for (int64_t i = 0; i < size - 1; i++) {
        if (array[i] > array[i + 1]) {
            is_sorted = false;
            if (!is_rev_sorted) break;
        }
        if (array[i] <= array[i + 1]) {
            is_rev_sorted = false;
            if (!is_sorted) break;
        }
    }
    if (is_sorted) return;
    if (is_rev_sorted) {
        std::reverse(array, array + size);
        return;
    }

    if (kind == 0 || kind == 1) {
        if (size > 32) {
            int64_t counts[256] = {0};
            for (int64_t i = 0; i < size; i++) {
                counts[(uint8_t)(array[i] + 128)]++;
            }
            int64_t idx = 0;
            for (int val = 0; val < 256; val++) {
                int64_t c = counts[val];
                if (c > 0) {
                    memset(array + idx, (int8_t)(val - 128), static_cast<size_t>(c));
                    idx += c;
                }
            }
        } else {
            if (kind == 0) std::sort(array, array + size);
            else std::stable_sort(array, array + size);
        }
    } else if (kind == 2) {
        std::make_heap(array, array + size);
        std::sort_heap(array, array + size);
    }
}

extern "C" void native_sort_uint16(uint16_t *array, int64_t size, int kind) {
    sort_int_impl(array, size, kind);
}

extern "C" void native_sort_uint32(uint32_t *array, int64_t size, int kind) {
    sort_int_impl(array, size, kind);
}

extern "C" void native_sort_uint64(uint64_t *array, int64_t size, int kind) {
    sort_int_impl(array, size, kind);
}

extern "C" void native_sort_float16(uint16_t *array, int64_t size, int kind) {
    if (array == nullptr || size <= 1) return;

    int64_t first_nan = -1;
    for (int64_t i = 0; i < size; i++) {
        if (is_nan_float16(array[i])) {
            first_nan = i;
            break;
        }
    }
    int64_t non_nan_size = size;
    if (first_nan != -1) {
        if (kind == 1) {
            NoThrowBuffer<uint16_t> nans;
            if (!nans.push_back(array[first_nan])) return;
            int64_t write_pos = first_nan;
            for (int64_t i = first_nan + 1; i < size; i++) {
                if (is_nan_float16(array[i])) {
                    if (!nans.push_back(array[i])) return;
                } else {
                    array[write_pos++] = array[i];
                }
            }
            for (size_t i = 0; i < nans.size(); i++) {
                array[write_pos + i] = nans[i];
            }
            non_nan_size = write_pos;
        } else {
            int64_t write_pos = first_nan;
            for (int64_t i = first_nan + 1; i < size; i++) {
                if (!is_nan_float16(array[i])) {
                    std::swap(array[write_pos], array[i]);
                    write_pos++;
                }
            }
            non_nan_size = write_pos;
        }
    }

    if (non_nan_size <= 1) return;

    bool is_sorted = true;
    bool is_rev_sorted = true;
    for (int64_t i = 0; i < non_nan_size - 1; i++) {
        float cur = decode_f16_to_f32(array[i]);
        float next = decode_f16_to_f32(array[i + 1]);
        if (cur > next) {
            is_sorted = false;
            if (!is_rev_sorted) break;
        }
        if (cur <= next) {
            is_rev_sorted = false;
            if (!is_sorted) break;
        }
    }
    if (is_sorted) return;
    if (is_rev_sorted) {
        std::reverse(array, array + non_nan_size);
        return;
    }

    if (kind == 0) {
        if (non_nan_size >= 128 && hwy::HaveFloat16()) {
            hwy::VQSort((hwy::float16_t *)array, static_cast<size_t>(non_nan_size), hwy::SortAscending());
        } else {
            std::sort(array, array + non_nan_size, Float16Less());
        }
    } else if (kind == 2) {
        std::make_heap(array, array + non_nan_size, Float16Less());
        std::sort_heap(array, array + non_nan_size, Float16Less());
    } else {
        std::stable_sort(array, array + non_nan_size, Float16Less());
    }
}

extern "C" void native_sort_bfloat16(uint16_t *array, int64_t size, int kind) {
    if (array == nullptr || size <= 1) return;

    int64_t first_nan = -1;
    for (int64_t i = 0; i < size; i++) {
        if (is_nan_bfloat16(array[i])) {
            first_nan = i;
            break;
        }
    }
    int64_t non_nan_size = size;
    if (first_nan != -1) {
        if (kind == 1) {
            NoThrowBuffer<uint16_t> nans;
            if (!nans.push_back(array[first_nan])) return;
            int64_t write_pos = first_nan;
            for (int64_t i = first_nan + 1; i < size; i++) {
                if (is_nan_bfloat16(array[i])) {
                    if (!nans.push_back(array[i])) return;
                } else {
                    array[write_pos++] = array[i];
                }
            }
            for (size_t i = 0; i < nans.size(); i++) {
                array[write_pos + i] = nans[i];
            }
            non_nan_size = write_pos;
        } else {
            int64_t write_pos = first_nan;
            for (int64_t i = first_nan + 1; i < size; i++) {
                if (!is_nan_bfloat16(array[i])) {
                    std::swap(array[write_pos], array[i]);
                    write_pos++;
                }
            }
            non_nan_size = write_pos;
        }
    }

    if (non_nan_size <= 1) return;

    bool is_sorted = true;
    bool is_rev_sorted = true;
    for (int64_t i = 0; i < non_nan_size - 1; i++) {
        float cur = decode_bf16_to_f32(array[i]);
        float next = decode_bf16_to_f32(array[i + 1]);
        if (cur > next) {
            is_sorted = false;
            if (!is_rev_sorted) break;
        }
        if (cur <= next) {
            is_rev_sorted = false;
            if (!is_sorted) break;
        }
    }
    if (is_sorted) return;
    if (is_rev_sorted) {
        std::reverse(array, array + non_nan_size);
        return;
    }

    if (kind == 0) {
        std::sort(array, array + non_nan_size, BFloat16Less());
    } else if (kind == 2) {
        std::make_heap(array, array + non_nan_size, BFloat16Less());
        std::sort_heap(array, array + non_nan_size, BFloat16Less());
    } else {
        std::stable_sort(array, array + non_nan_size, BFloat16Less());
    }
}

// ----------------------------------------------------------------------------
// Public Argsort Sorters with Kind Parameter
// ----------------------------------------------------------------------------

extern "C" void native_argsort_double(const double *data, int64_t *indices, int64_t size, int kind) {
    argsort_impl(data, indices, size, kind);
}

extern "C" void native_argsort_float(const float *data, int64_t *indices, int64_t size, int kind) {
    argsort_impl(data, indices, size, kind);
}

extern "C" void native_argsort_int64(const long long *data, int64_t *indices, int64_t size, int kind) {
    argsort_impl(data, indices, size, kind);
}

extern "C" void native_argsort_int32(const int *data, int64_t *indices, int64_t size, int kind) {
    argsort_impl(data, indices, size, kind);
}

extern "C" void native_argsort_int16(const int16_t *data, int64_t *indices, int64_t size, int kind) {
    argsort_impl(data, indices, size, kind);
}

extern "C" void native_argsort_uint8(const uint8_t *data, int64_t *indices, int64_t size, int kind) {
    argsort_impl(data, indices, size, kind);
}

extern "C" void native_argsort_int8(const int8_t *data, int64_t *indices, int64_t size, int kind) {
    argsort_impl(data, indices, size, kind);
}

extern "C" void native_argsort_uint16(const uint16_t *data, int64_t *indices, int64_t size, int kind) {
    argsort_impl(data, indices, size, kind);
}

extern "C" void native_argsort_uint32(const uint32_t *data, int64_t *indices, int64_t size, int kind) {
    argsort_impl(data, indices, size, kind);
}

extern "C" void native_argsort_uint64(const uint64_t *data, int64_t *indices, int64_t size, int kind) {
    argsort_impl(data, indices, size, kind);
}

extern "C" void native_argsort_float16(const uint16_t *data, int64_t *indices, int64_t size, int kind) {
    if (data == nullptr || indices == nullptr || size <= 0) return;
    if (size == 1) {
        indices[0] = 0;
        return;
    }

    int64_t left = 0;
    int64_t right = size - 1;
    for (int64_t i = 0; i < size; i++) {
        if (is_nan_float16(data[i])) {
            indices[right--] = i;
        } else {
            indices[left++] = i;
        }
    }
    int64_t valid_len = left;
    if (valid_len < size) {
        std::reverse(indices + valid_len, indices + size);
    }

    if (valid_len <= 1) return;

    bool is_sorted = true;
    bool is_rev_sorted = true;
    for (int64_t i = 0; i < valid_len - 1; i++) {
        uint16_t k_cur = sortable_key_half(data[indices[i]]);
        uint16_t k_next = sortable_key_half(data[indices[i + 1]]);
        if (k_cur > k_next) {
            is_sorted = false;
            if (!is_rev_sorted) break;
        }
        if (k_cur <= k_next) {
            is_rev_sorted = false;
            if (!is_sorted) break;
        }
    }
    if (is_sorted) return;
    if (is_rev_sorted) {
        std::reverse(indices, indices + valid_len);
        return;
    }

    if (kind != 2 && static_cast<uint64_t>(size) <= 0xFFFFFFFFULL) {
        uint64_t *packed = reinterpret_cast<uint64_t *>(indices);
        for (int64_t i = 0; i < valid_len; i++) {
            uint32_t idx = static_cast<uint32_t>(indices[i]);
            uint32_t key = static_cast<uint32_t>(sortable_key_half(data[idx]));
            packed[i] = (static_cast<uint64_t>(key) << 32) | idx;
        }
        hwy::VQSort(packed, static_cast<size_t>(valid_len), hwy::SortAscending());
        for (int64_t i = 0; i < valid_len; i++) {
            indices[i] = static_cast<int64_t>(packed[i] & 0xFFFFFFFFULL);
        }
        return;
    }

    auto cmp = [data](int64_t a, int64_t b) {
        return sortable_key_half(data[a]) < sortable_key_half(data[b]);
    };

    if (kind == 0) {
        std::sort(indices, indices + valid_len, cmp);
    } else if (kind == 1) {
        std::stable_sort(indices, indices + valid_len, cmp);
    } else {
        std::make_heap(indices, indices + valid_len, cmp);
        std::sort_heap(indices, indices + valid_len, cmp);
    }
}

extern "C" void native_argsort_bfloat16(const uint16_t *data, int64_t *indices, int64_t size, int kind) {
    if (data == nullptr || indices == nullptr || size <= 0) return;
    if (size == 1) {
        indices[0] = 0;
        return;
    }

    int64_t left = 0;
    int64_t right = size - 1;
    for (int64_t i = 0; i < size; i++) {
        if (is_nan_bfloat16(data[i])) {
            indices[right--] = i;
        } else {
            indices[left++] = i;
        }
    }
    int64_t valid_len = left;
    if (valid_len < size) {
        std::reverse(indices + valid_len, indices + size);
    }

    if (valid_len <= 1) return;

    bool is_sorted = true;
    bool is_rev_sorted = true;
    for (int64_t i = 0; i < valid_len - 1; i++) {
        uint16_t k_cur = sortable_key_half(data[indices[i]]);
        uint16_t k_next = sortable_key_half(data[indices[i + 1]]);
        if (k_cur > k_next) {
            is_sorted = false;
            if (!is_rev_sorted) break;
        }
        if (k_cur <= k_next) {
            is_rev_sorted = false;
            if (!is_sorted) break;
        }
    }
    if (is_sorted) return;
    if (is_rev_sorted) {
        std::reverse(indices, indices + valid_len);
        return;
    }

    if (kind != 2 && static_cast<uint64_t>(size) <= 0xFFFFFFFFULL) {
        uint64_t *packed = reinterpret_cast<uint64_t *>(indices);
        for (int64_t i = 0; i < valid_len; i++) {
            uint32_t idx = static_cast<uint32_t>(indices[i]);
            uint32_t key = static_cast<uint32_t>(sortable_key_half(data[idx]));
            packed[i] = (static_cast<uint64_t>(key) << 32) | idx;
        }
        hwy::VQSort(packed, static_cast<size_t>(valid_len), hwy::SortAscending());
        for (int64_t i = 0; i < valid_len; i++) {
            indices[i] = static_cast<int64_t>(packed[i] & 0xFFFFFFFFULL);
        }
        return;
    }

    auto cmp = [data](int64_t a, int64_t b) {
        return sortable_key_half(data[a]) < sortable_key_half(data[b]);
    };

    if (kind == 0) {
        std::sort(indices, indices + valid_len, cmp);
    } else if (kind == 1) {
        std::stable_sort(indices, indices + valid_len, cmp);
    } else {
        std::make_heap(indices, indices + valid_len, cmp);
        std::sort_heap(indices, indices + valid_len, cmp);
    }
}

// ----------------------------------------------------------------------------
// Public Partition Sorters
// ----------------------------------------------------------------------------

extern "C" void native_partition_double(double *array, int64_t size, const int64_t *k_list, int64_t k_size) {
    partition_impl(array, size, k_list, k_size);
}

extern "C" void native_partition_float(float *array, int64_t size, const int64_t *k_list, int64_t k_size) {
    partition_impl(array, size, k_list, k_size);
}

extern "C" void native_partition_int64(long long *array, int64_t size, const int64_t *k_list, int64_t k_size) {
    partition_impl(array, size, k_list, k_size);
}

extern "C" void native_partition_int32(int *array, int64_t size, const int64_t *k_list, int64_t k_size) {
    partition_impl(array, size, k_list, k_size);
}

extern "C" void native_partition_int16(int16_t *array, int64_t size, const int64_t *k_list, int64_t k_size) {
    partition_impl(array, size, k_list, k_size);
}

extern "C" void native_partition_uint8(uint8_t *array, int64_t size, const int64_t *k_list, int64_t k_size) {
    partition_impl(array, size, k_list, k_size);
}

extern "C" void native_partition_complex128(double *array, int64_t size, const int64_t *k_list, int64_t k_size) {
    if (array == nullptr || size <= 1 || k_list == nullptr || k_size <= 0) return;
    complex128_t *carr = (complex128_t *)array;
    NoThrowBuffer<int64_t> sorted_k;
    const int64_t *k_ptr = k_list;
    int64_t num_k = k_size;
    if (!std::is_sorted(k_list, k_list + k_size)) {
        sorted_k.assign(k_list, k_list + k_size);
        if (!sorted_k.ok()) return;
        std::sort(sorted_k.begin(), sorted_k.end());
        k_ptr = sorted_k.data();
    }
    if (num_k == 1) {
        if (k_ptr[0] >= 0 && k_ptr[0] < size) {
            std::nth_element(carr, carr + k_ptr[0], carr + size, comp_complex_impl<complex128_t>);
        }
    } else {
        multi_nth_element(carr, 0, size - 1, k_ptr, 0, num_k - 1, comp_complex_impl<complex128_t>);
    }
}

extern "C" void native_partition_complex64(float *array, int64_t size, const int64_t *k_list, int64_t k_size) {
    if (array == nullptr || size <= 1 || k_list == nullptr || k_size <= 0) return;
    complex64_t *carr = (complex64_t *)array;
    NoThrowBuffer<int64_t> sorted_k;
    const int64_t *k_ptr = k_list;
    int64_t num_k = k_size;
    if (!std::is_sorted(k_list, k_list + k_size)) {
        sorted_k.assign(k_list, k_list + k_size);
        if (!sorted_k.ok()) return;
        std::sort(sorted_k.begin(), sorted_k.end());
        k_ptr = sorted_k.data();
    }
    if (num_k == 1) {
        if (k_ptr[0] >= 0 && k_ptr[0] < size) {
            std::nth_element(carr, carr + k_ptr[0], carr + size, comp_complex_impl<complex64_t>);
        }
    } else {
        multi_nth_element(carr, 0, size - 1, k_ptr, 0, num_k - 1, comp_complex_impl<complex64_t>);
    }
}

extern "C" void native_partition_int8(int8_t *array, int64_t size, const int64_t *k_list, int64_t k_size) {
    partition_impl(array, size, k_list, k_size);
}

extern "C" void native_partition_uint16(uint16_t *array, int64_t size, const int64_t *k_list, int64_t k_size) {
    partition_impl(array, size, k_list, k_size);
}

extern "C" void native_partition_uint32(uint32_t *array, int64_t size, const int64_t *k_list, int64_t k_size) {
    partition_impl(array, size, k_list, k_size);
}

extern "C" void native_partition_uint64(uint64_t *array, int64_t size, const int64_t *k_list, int64_t k_size) {
    partition_impl(array, size, k_list, k_size);
}

extern "C" void native_partition_float16(uint16_t *array, int64_t size, const int64_t *k_list, int64_t k_size) {
    if (array == nullptr || size <= 1 || k_list == nullptr || k_size <= 0) return;

    int64_t valid_len = size;
    int64_t first_nan = -1;
    for (int64_t i = 0; i < size; i++) {
        if (is_nan_float16(array[i])) {
            first_nan = i;
            break;
        }
    }
    if (first_nan != -1) {
        int64_t write_pos = first_nan;
        for (int64_t i = first_nan + 1; i < size; i++) {
            if (!is_nan_float16(array[i])) {
                std::swap(array[write_pos], array[i]);
                write_pos++;
            }
        }
        valid_len = write_pos;
    }

    if (valid_len <= 1) return;

    NoThrowBuffer<int64_t> sorted_k;
    const int64_t *k_ptr = k_list;
    int64_t num_k = k_size;
    if (!std::is_sorted(k_list, k_list + k_size)) {
        sorted_k.assign(k_list, k_list + k_size);
        if (!sorted_k.ok()) return;
        std::sort(sorted_k.begin(), sorted_k.end());
        k_ptr = sorted_k.data();
    }

    int64_t k_start = 0;
    while (k_start < num_k && k_ptr[k_start] < 0) k_start++;
    int64_t k_end = k_start;
    while (k_end < num_k && k_ptr[k_end] < valid_len) k_end++;

    if (k_end <= k_start) return;

    auto cmp = Float16Less();
    if (k_end - k_start == 1) {
        std::nth_element(array, array + k_ptr[k_start], array + valid_len, cmp);
    } else {
        multi_nth_element(array, 0, valid_len - 1, k_ptr, k_start, k_end - 1, cmp);
    }
}

extern "C" void native_partition_bfloat16(uint16_t *array, int64_t size, const int64_t *k_list, int64_t k_size) {
    if (array == nullptr || size <= 1 || k_list == nullptr || k_size <= 0) return;

    int64_t valid_len = size;
    int64_t first_nan = -1;
    for (int64_t i = 0; i < size; i++) {
        if (is_nan_bfloat16(array[i])) {
            first_nan = i;
            break;
        }
    }
    if (first_nan != -1) {
        int64_t write_pos = first_nan;
        for (int64_t i = first_nan + 1; i < size; i++) {
            if (!is_nan_bfloat16(array[i])) {
                std::swap(array[write_pos], array[i]);
                write_pos++;
            }
        }
        valid_len = write_pos;
    }

    if (valid_len <= 1) return;

    NoThrowBuffer<int64_t> sorted_k;
    const int64_t *k_ptr = k_list;
    int64_t num_k = k_size;
    if (!std::is_sorted(k_list, k_list + k_size)) {
        sorted_k.assign(k_list, k_list + k_size);
        if (!sorted_k.ok()) return;
        std::sort(sorted_k.begin(), sorted_k.end());
        k_ptr = sorted_k.data();
    }

    int64_t k_start = 0;
    while (k_start < num_k && k_ptr[k_start] < 0) k_start++;
    int64_t k_end = k_start;
    while (k_end < num_k && k_ptr[k_end] < valid_len) k_end++;

    if (k_end <= k_start) return;

    auto cmp = BFloat16Less();
    if (k_end - k_start == 1) {
        std::nth_element(array, array + k_ptr[k_start], array + valid_len, cmp);
    } else {
        multi_nth_element(array, 0, valid_len - 1, k_ptr, k_start, k_end - 1, cmp);
    }
}

// ----------------------------------------------------------------------------
// Public Argpartition Sorters
// ----------------------------------------------------------------------------

extern "C" void native_argpartition_double(const double *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size) {
    argpartition_impl(data, indices, size, k_list, k_size);
}

extern "C" void native_argpartition_float(const float *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size) {
    argpartition_impl(data, indices, size, k_list, k_size);
}

extern "C" void native_argpartition_int64(const long long *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size) {
    argpartition_impl(data, indices, size, k_list, k_size);
}

extern "C" void native_argpartition_int32(const int *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size) {
    argpartition_impl(data, indices, size, k_list, k_size);
}

extern "C" void native_argpartition_int16(const int16_t *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size) {
    argpartition_impl(data, indices, size, k_list, k_size);
}

extern "C" void native_argpartition_uint8(const uint8_t *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size) {
    argpartition_impl(data, indices, size, k_list, k_size);
}

extern "C" void native_argpartition_complex128(const double *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size) {
    if (data == nullptr || indices == nullptr || size <= 0 || k_list == nullptr || k_size <= 0) return;
    for (int64_t i = 0; i < size; i++) indices[i] = i;
    if (size <= 1) return;
    const complex128_t *cdata = (const complex128_t *)data;
    NoThrowBuffer<int64_t> sorted_k;
    const int64_t *k_ptr = k_list;
    int64_t num_k = k_size;
    if (!std::is_sorted(k_list, k_list + k_size)) {
        sorted_k.assign(k_list, k_list + k_size);
        if (!sorted_k.ok()) return;
        std::sort(sorted_k.begin(), sorted_k.end());
        k_ptr = sorted_k.data();
    }
    auto cmp = [cdata](int64_t a, int64_t b) {
        return comp_complex_impl(cdata[a], cdata[b]);
    };
    if (num_k == 1) {
        if (k_ptr[0] >= 0 && k_ptr[0] < size) {
            std::nth_element(indices, indices + k_ptr[0], indices + size, cmp);
        }
    } else {
        arg_multi_nth_element(indices, 0, size - 1, k_ptr, 0, num_k - 1, cmp);
    }
}

extern "C" void native_argpartition_complex64(const float *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size) {
    if (data == nullptr || indices == nullptr || size <= 0 || k_list == nullptr || k_size <= 0) return;
    for (int64_t i = 0; i < size; i++) indices[i] = i;
    if (size <= 1) return;
    const complex64_t *cdata = (const complex64_t *)data;
    NoThrowBuffer<int64_t> sorted_k;
    const int64_t *k_ptr = k_list;
    int64_t num_k = k_size;
    if (!std::is_sorted(k_list, k_list + k_size)) {
        sorted_k.assign(k_list, k_list + k_size);
        if (!sorted_k.ok()) return;
        std::sort(sorted_k.begin(), sorted_k.end());
        k_ptr = sorted_k.data();
    }
    auto cmp = [cdata](int64_t a, int64_t b) {
        return comp_complex_impl(cdata[a], cdata[b]);
    };
    if (num_k == 1) {
        if (k_ptr[0] >= 0 && k_ptr[0] < size) {
            std::nth_element(indices, indices + k_ptr[0], indices + size, cmp);
        }
    } else {
        arg_multi_nth_element(indices, 0, size - 1, k_ptr, 0, num_k - 1, cmp);
    }
}

extern "C" void native_argpartition_int8(const int8_t *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size) {
    argpartition_impl(data, indices, size, k_list, k_size);
}

extern "C" void native_argpartition_uint16(const uint16_t *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size) {
    argpartition_impl(data, indices, size, k_list, k_size);
}

extern "C" void native_argpartition_uint32(const uint32_t *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size) {
    argpartition_impl(data, indices, size, k_list, k_size);
}

extern "C" void native_argpartition_uint64(const uint64_t *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size) {
    argpartition_impl(data, indices, size, k_list, k_size);
}

extern "C" void native_argpartition_float16(const uint16_t *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size) {
    if (data == nullptr || indices == nullptr || size <= 0 || k_list == nullptr || k_size <= 0) return;
    if (size == 1) {
        indices[0] = 0;
        return;
    }

    int64_t left = 0;
    int64_t right = size - 1;
    for (int64_t i = 0; i < size; i++) {
        if (is_nan_float16(data[i])) {
            indices[right--] = i;
        } else {
            indices[left++] = i;
        }
    }
    int64_t valid_len = left;
    if (valid_len < size) {
        std::reverse(indices + valid_len, indices + size);
    }

    if (valid_len <= 1) return;

    NoThrowBuffer<int64_t> sorted_k;
    const int64_t *k_ptr = k_list;
    int64_t num_k = k_size;
    if (!std::is_sorted(k_list, k_list + k_size)) {
        sorted_k.assign(k_list, k_list + k_size);
        if (!sorted_k.ok()) return;
        std::sort(sorted_k.begin(), sorted_k.end());
        k_ptr = sorted_k.data();
    }

    int64_t k_start = 0;
    while (k_start < num_k && k_ptr[k_start] < 0) k_start++;
    int64_t k_end = k_start;
    while (k_end < num_k && k_ptr[k_end] < valid_len) k_end++;

    if (k_end <= k_start) return;

    if (valid_len >= 64 && static_cast<uint64_t>(size) <= 0xFFFFFFFFULL) {
        uint64_t *packed = reinterpret_cast<uint64_t *>(indices);
        for (int64_t i = 0; i < valid_len; i++) {
            uint32_t idx = static_cast<uint32_t>(indices[i]);
            uint32_t key = static_cast<uint32_t>(sortable_key_half(data[idx]));
            packed[i] = (static_cast<uint64_t>(key) << 32) | idx;
        }
        vqselect_dispatch(packed, valid_len, k_ptr, k_start, k_end - 1);
        for (int64_t i = 0; i < valid_len; i++) {
            indices[i] = static_cast<int64_t>(packed[i] & 0xFFFFFFFFULL);
        }
        return;
    }

    auto cmp = [data](int64_t a, int64_t b) {
        return sortable_key_half(data[a]) < sortable_key_half(data[b]);
    };

    if (k_end - k_start == 1) {
        std::nth_element(indices, indices + k_ptr[k_start], indices + valid_len, cmp);
    } else {
        arg_multi_nth_element(indices, 0, valid_len - 1, k_ptr, k_start, k_end - 1, cmp);
    }
}

extern "C" void native_argpartition_bfloat16(const uint16_t *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size) {
    if (data == nullptr || indices == nullptr || size <= 0 || k_list == nullptr || k_size <= 0) return;
    if (size == 1) {
        indices[0] = 0;
        return;
    }

    int64_t left = 0;
    int64_t right = size - 1;
    for (int64_t i = 0; i < size; i++) {
        if (is_nan_bfloat16(data[i])) {
            indices[right--] = i;
        } else {
            indices[left++] = i;
        }
    }
    int64_t valid_len = left;
    if (valid_len < size) {
        std::reverse(indices + valid_len, indices + size);
    }

    if (valid_len <= 1) return;

    NoThrowBuffer<int64_t> sorted_k;
    const int64_t *k_ptr = k_list;
    int64_t num_k = k_size;
    if (!std::is_sorted(k_list, k_list + k_size)) {
        sorted_k.assign(k_list, k_list + k_size);
        if (!sorted_k.ok()) return;
        std::sort(sorted_k.begin(), sorted_k.end());
        k_ptr = sorted_k.data();
    }

    int64_t k_start = 0;
    while (k_start < num_k && k_ptr[k_start] < 0) k_start++;
    int64_t k_end = k_start;
    while (k_end < num_k && k_ptr[k_end] < valid_len) k_end++;

    if (k_end <= k_start) return;

    if (valid_len >= 64 && static_cast<uint64_t>(size) <= 0xFFFFFFFFULL) {
        uint64_t *packed = reinterpret_cast<uint64_t *>(indices);
        for (int64_t i = 0; i < valid_len; i++) {
            uint32_t idx = static_cast<uint32_t>(indices[i]);
            uint32_t key = static_cast<uint32_t>(sortable_key_half(data[idx]));
            packed[i] = (static_cast<uint64_t>(key) << 32) | idx;
        }
        vqselect_dispatch(packed, valid_len, k_ptr, k_start, k_end - 1);
        for (int64_t i = 0; i < valid_len; i++) {
            indices[i] = static_cast<int64_t>(packed[i] & 0xFFFFFFFFULL);
        }
        return;
    }

    auto cmp = [data](int64_t a, int64_t b) {
        return sortable_key_half(data[a]) < sortable_key_half(data[b]);
    };

    if (k_end - k_start == 1) {
        std::nth_element(indices, indices + k_ptr[k_start], indices + valid_len, cmp);
    } else {
        arg_multi_nth_element(indices, 0, valid_len - 1, k_ptr, k_start, k_end - 1, cmp);
    }
}

// ----------------------------------------------------------------------------
// Public Searchsorted (Binary Search) functions
// ----------------------------------------------------------------------------

extern "C" void native_searchsorted_double(const double *array, int64_t size, const double *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, compare_double_inline);
}

extern "C" void native_searchsorted_float(const float *array, int64_t size, const float *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, compare_float_inline);
}

extern "C" void native_searchsorted_int64(const long long *array, int64_t size, const long long *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, compare_int64_inline);
}

extern "C" void native_searchsorted_int32(const int *array, int64_t size, const int *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, compare_int32_inline);
}

extern "C" void native_searchsorted_int16(const int16_t *array, int64_t size, const int16_t *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, compare_int16_inline);
}

extern "C" void native_searchsorted_uint8(const uint8_t *array, int64_t size, const uint8_t *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, compare_uint8_inline);
}

extern "C" void native_searchsorted_complex128(const double *array, int64_t size, const double *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter) {
    searchsorted((const complex128_t *)array, size, (const complex128_t *)values, out_indices, num_values, side_left, sorter, compare_complex128_inline);
}

extern "C" void native_searchsorted_complex64(const float *array, int64_t size, const float *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter) {
    searchsorted((const complex64_t *)array, size, (const complex64_t *)values, out_indices, num_values, side_left, sorter, compare_complex64_inline);
}

extern "C" void native_searchsorted_int8(const int8_t *array, int64_t size, const int8_t *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, standard_compare<int8_t>);
}

extern "C" void native_searchsorted_uint16(const uint16_t *array, int64_t size, const uint16_t *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, standard_compare<uint16_t>);
}

extern "C" void native_searchsorted_uint32(const uint32_t *array, int64_t size, const uint32_t *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, standard_compare<uint32_t>);
}

extern "C" void native_searchsorted_uint64(const uint64_t *array, int64_t size, const uint64_t *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, standard_compare<uint64_t>);
}

extern "C" void native_searchsorted_float16(const uint16_t *array, int64_t size, const uint16_t *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, compare_float16_inline);
}

extern "C" void native_searchsorted_bfloat16(const uint16_t *array, int64_t size, const uint16_t *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, compare_bfloat16_inline);
}

// ----------------------------------------------------------------------------
// Utility Operations
// ----------------------------------------------------------------------------

extern "C" int custom_memcmp(const void *s1, const void *s2, size_t n) {
    if (s1 == nullptr || s2 == nullptr) return s1 == s2 ? 0 : (s1 == nullptr ? -1 : 1);
    return memcmp(s1, s2, n);
}

extern "C" void native_zero_memory(void *ptr, size_t bytes) {
    if (ptr == nullptr || bytes <= 0) return;
    memset(ptr, 0, bytes);
}

extern "C" void custom_memcpy(void *dest, const void *src, size_t n) {
    if (dest == nullptr || src == nullptr || n <= 0) return;
    memmove(dest, src, n);
}

extern "C" void native_collect_nonzero_coords(
    const unsigned char *cond,
    int64_t total_size,
    const int64_t *shape,
    const int64_t *strides,
    int rank,
    int64_t **out_coords
) {
    if (cond == nullptr || shape == nullptr || strides == nullptr || out_coords == nullptr || total_size <= 0 || rank <= 0) return;
    if (rank == 1 && strides[0] == 1) {
        ndarray_hwy::UnpackMaskImpl(cond, total_size, out_coords[0]);
        return;
    }
    if (rank == 2 && strides[0] == shape[1] && strides[1] == 1) {
        const int64_t rows = shape[0];
        const int64_t cols = shape[1];
        int64_t *out_r = out_coords[0];
        int64_t *out_c = out_coords[1];
        int64_t write_idx = 0;
        for (int64_t r = 0; r < rows; r++) {
            int64_t row_count = ndarray_hwy::UnpackMaskImpl(cond + r * cols, cols, out_c + write_idx);
            int64_t *r_ptr = out_r + write_idx;
            for (int64_t k = 0; k < row_count; k++) {
                r_ptr[k] = r;
            }
            write_idx += row_count;
        }
        return;
    }
    NoThrowBuffer<int64_t> coord_vec;
    int64_t coord_stack[32] = {0};
    int64_t *coord = coord_stack;
    if (rank > 32) {
        coord_vec.assign(rank, 0);
        if (!coord_vec.ok()) return;
        coord = coord_vec.data();
    }
    int64_t offset = 0;
    int64_t write_idx = 0;

    for (int64_t el = 0; el < total_size; el++) {
        if (cond[offset]) {
            for (int d = 0; d < rank; d++) {
                out_coords[d][write_idx] = coord[d];
            }
            write_idx++;
        }

        // Advance odometer multidimensional walk
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offset += strides[d];
                break;
            }
            coord[d] = 0;
            offset -= (shape[d] - 1) * strides[d];
        }
    }
}

extern "C" void native_collect_nonzero_coords_grouped(
    const unsigned char *cond,
    int64_t total_size,
    const int64_t *shape,
    const int64_t *strides,
    int rank,
    int64_t *out_coords
) {
    if (cond == nullptr || shape == nullptr || strides == nullptr || out_coords == nullptr || total_size <= 0 || rank <= 0) return;
    if (rank == 1 && strides[0] == 1) {
        ndarray_hwy::UnpackMaskImpl(cond, total_size, out_coords);
        return;
    }
    if (rank == 2 && strides[0] == shape[1] && strides[1] == 1) {
        const int64_t rows = shape[0];
        const int64_t cols = shape[1];
        int64_t col_buf[256];
        int64_t write_idx = 0;
        for (int64_t r = 0; r < rows; r++) {
            const unsigned char *row = cond + r * cols;
            for (int64_t c0 = 0; c0 < cols; c0 += 256) {
                int64_t chunk = std::min<int64_t>(256, cols - c0);
                int64_t cnt = ndarray_hwy::UnpackMaskImpl(row + c0, chunk, col_buf);
                for (int64_t k = 0; k < cnt; k++) {
                    out_coords[write_idx * 2] = r;
                    out_coords[write_idx * 2 + 1] = c0 + col_buf[k];
                    write_idx++;
                }
            }
        }
        return;
    }
    NoThrowBuffer<int64_t> coord_vec;
    int64_t coord_stack[32] = {0};
    int64_t *coord = coord_stack;
    if (rank > 32) {
        coord_vec.assign(rank, 0);
        if (!coord_vec.ok()) return;
        coord = coord_vec.data();
    }
    int64_t offset = 0;
    int64_t write_idx = 0;

    for (int64_t el = 0; el < total_size; el++) {
        if (cond[offset]) {
            for (int d = 0; d < rank; d++) {
                out_coords[write_idx * rank + d] = coord[d];
            }
            write_idx++;
        }

        // Advance odometer multidimensional walk
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offset += strides[d];
                break;
            }
            coord[d] = 0;
            offset -= (shape[d] - 1) * strides[d];
        }
    }
}

template <typename T>
static void to_bool_mask(
    const T *src,
    int64_t size,
    const int64_t *shape,
    const int64_t *strides,
    int rank,
    int is_contiguous,
    unsigned char *dest
) {
    if (src == nullptr || dest == nullptr || size <= 0) return;
    if (is_contiguous) {
        ndarray_hwy::ToBoolMaskContiguous(src, size, dest);
        return;
    }
    if (shape == nullptr || strides == nullptr || rank <= 0) return;
    NoThrowBuffer<int64_t> coord_vec;
    int64_t coord_stack[32] = {0};
    int64_t *coord = coord_stack;
    if (rank > 32) {
        coord_vec.assign(rank, 0);
        if (!coord_vec.ok()) {
            std::memset(dest, 0, static_cast<size_t>(size));
            return;
        }
        coord = coord_vec.data();
    }
    int64_t offset = 0;
    for (int64_t i = 0; i < size; i++) {
        dest[i] = is_nonzero(src[offset]) ? 1 : 0;
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offset += strides[d];
                break;
            }
            coord[d] = 0;
            offset -= (shape[d] - 1) * strides[d];
        }
    }
}

extern "C" void native_to_bool_mask_double(const void *src, int64_t size, const int64_t *shape, const int64_t *strides, int rank, int is_contiguous, unsigned char *dest) {
    to_bool_mask((const double *)src, size, shape, strides, rank, is_contiguous, dest);
}
extern "C" void native_to_bool_mask_float(const void *src, int64_t size, const int64_t *shape, const int64_t *strides, int rank, int is_contiguous, unsigned char *dest) {
    to_bool_mask((const float *)src, size, shape, strides, rank, is_contiguous, dest);
}
extern "C" void native_to_bool_mask_int64(const void *src, int64_t size, const int64_t *shape, const int64_t *strides, int rank, int is_contiguous, unsigned char *dest) {
    to_bool_mask((const long long *)src, size, shape, strides, rank, is_contiguous, dest);
}
extern "C" void native_to_bool_mask_int32(const void *src, int64_t size, const int64_t *shape, const int64_t *strides, int rank, int is_contiguous, unsigned char *dest) {
    to_bool_mask((const int *)src, size, shape, strides, rank, is_contiguous, dest);
}
extern "C" void native_to_bool_mask_complex128(const void *src, int64_t size, const int64_t *shape, const int64_t *strides, int rank, int is_contiguous, unsigned char *dest) {
    to_bool_mask((const complex128_t *)src, size, shape, strides, rank, is_contiguous, dest);
}
extern "C" void native_to_bool_mask_complex64(const void *src, int64_t size, const int64_t *shape, const int64_t *strides, int rank, int is_contiguous, unsigned char *dest) {
    to_bool_mask((const complex64_t *)src, size, shape, strides, rank, is_contiguous, dest);
}
extern "C" void native_to_bool_mask_uint8(const void *src, int64_t size, const int64_t *shape, const int64_t *strides, int rank, int is_contiguous, unsigned char *dest) {
    to_bool_mask((const unsigned char *)src, size, shape, strides, rank, is_contiguous, dest);
}
extern "C" void native_to_bool_mask_int16(const void *src, int64_t size, const int64_t *shape, const int64_t *strides, int rank, int is_contiguous, unsigned char *dest) {
    to_bool_mask((const short *)src, size, shape, strides, rank, is_contiguous, dest);
}

extern "C" void native_argminmax_double(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_max, int is_contiguous) {
    argminmax((const double *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_max, is_contiguous, standard_compare<double>);
}
extern "C" void native_argminmax_float(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_max, int is_contiguous) {
    argminmax((const float *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_max, is_contiguous, standard_compare<float>);
}
extern "C" void native_argminmax_int64(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_max, int is_contiguous) {
    argminmax((const int64_t *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_max, is_contiguous, standard_compare<int64_t>);
}
extern "C" void native_argminmax_int32(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_max, int is_contiguous) {
    argminmax((const int32_t *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_max, is_contiguous, standard_compare<int32_t>);
}
extern "C" void native_argminmax_uint8(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_max, int is_contiguous) {
    argminmax((const uint8_t *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_max, is_contiguous, standard_compare<uint8_t>);
}
extern "C" void native_argminmax_int16(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_max, int is_contiguous) {
    argminmax((const int16_t *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_max, is_contiguous, standard_compare<int16_t>);
}

extern "C" void native_count_nonzero_double(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_contiguous) {
    count_nonzero((const double *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_contiguous);
}
extern "C" void native_count_nonzero_float(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_contiguous) {
    count_nonzero((const float *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_contiguous);
}
extern "C" void native_count_nonzero_int64(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_contiguous) {
    count_nonzero((const long long *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_contiguous);
}
extern "C" void native_count_nonzero_int32(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_contiguous) {
    count_nonzero((const int *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_contiguous);
}
extern "C" void native_count_nonzero_uint8(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_contiguous) {
    count_nonzero((const unsigned char *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_contiguous);
}
extern "C" void native_count_nonzero_int16(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_contiguous) {
    count_nonzero((const short *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_contiguous);
}

extern "C" void native_count_nonzero_complex128(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_contiguous) {
    count_nonzero((const complex128_t *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_contiguous);
}

extern "C" void native_count_nonzero_complex64(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_contiguous) {
    count_nonzero((const complex64_t *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_contiguous);
}

extern "C" {
int64_t unpack_mask_c(
    const uint8_t *mask_ptr,
    int64_t size,
    int64_t stride,
    int64_t *out_indices
) {
    if (mask_ptr == nullptr || out_indices == nullptr || size <= 0) return 0;

    if (stride == 1) {
        return ndarray_hwy::UnpackMaskImpl(mask_ptr, size, out_indices);
    } else {
        int64_t count = 0;
        for (int64_t j = 0; j < size; j++) {
            if (mask_ptr[j * stride] != 0) {
                out_indices[count++] = j;
            }
        }
        return count;
    }
}
}

extern "C" {
int64_t native_count_mask(const uint8_t *mask, int64_t size) {
    if (mask == nullptr || size <= 0) return 0;
    return ndarray_hwy::CountNonZeroContiguous(mask, size);
}

void native_apply_mask(
    int dtype,
    const void *src,
    const uint8_t *mask,
    void *dest,
    int64_t size
) {
    if (src == nullptr || mask == nullptr || dest == nullptr || size <= 0) return;
    
    switch (dtype) {
        case DTYPE_FLOAT32: { // float32
            const float *s = (const float *)src;
            float *d = (float *)dest;
            int64_t count = 0;
            for (int64_t i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_FLOAT64: { // float64
            const double *s = (const double *)src;
            double *d = (double *)dest;
            int64_t count = 0;
            for (int64_t i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_COMPLEX64: { // complex64
            struct C64 { float real, imag; };
            const C64 *s = (const C64 *)src;
            C64 *d = (C64 *)dest;
            int64_t count = 0;
            for (int64_t i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_COMPLEX128: { // complex128
            struct C128 { double real, imag; };
            const C128 *s = (const C128 *)src;
            C128 *d = (C128 *)dest;
            int64_t count = 0;
            for (int64_t i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_INT32: { // int32
            const int32_t *s = (const int32_t *)src;
            int32_t *d = (int32_t *)dest;
            int64_t count = 0;
            for (int64_t i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_INT64: { // int64
            const int64_t *s = (const int64_t *)src;
            int64_t *d = (int64_t *)dest;
            int64_t count = 0;
            for (int64_t i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_UINT8: { // uint8
            const uint8_t *s = (const uint8_t *)src;
            uint8_t *d = (uint8_t *)dest;
            int64_t count = 0;
            for (int64_t i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_INT16: { // int16
            const int16_t *s = (const int16_t *)src;
            int16_t *d = (int16_t *)dest;
            int64_t count = 0;
            for (int64_t i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_UINT64: { // uint64
            const uint64_t *s = (const uint64_t *)src;
            uint64_t *d = (uint64_t *)dest;
            int64_t count = 0;
            for (int64_t i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_UINT32: { // uint32
            const uint32_t *s = (const uint32_t *)src;
            uint32_t *d = (uint32_t *)dest;
            int64_t count = 0;
            for (int64_t i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_UINT16: { // uint16
            const uint16_t *s = (const uint16_t *)src;
            uint16_t *d = (uint16_t *)dest;
            int64_t count = 0;
            for (int64_t i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_INT8: { // int8
            const int8_t *s = (const int8_t *)src;
            int8_t *d = (int8_t *)dest;
            int64_t count = 0;
            for (int64_t i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16: {
            const uint16_t *s = (const uint16_t *)src;
            uint16_t *d = (uint16_t *)dest;
            int64_t count = 0;
            for (int64_t i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_BOOLEAN: { // boolean
            const uint8_t *s = (const uint8_t *)src;
            uint8_t *d = (uint8_t *)dest;
            int64_t count = 0;
            for (int64_t i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        default:
            return;
    }
}
}

static inline bool comp_double_impl(double a, double b) {
    bool nan_a = std::isnan(a);
    bool nan_b = std::isnan(b);
    if (nan_a && nan_b) return false;
    if (nan_a) return false;
    if (nan_b) return true;
    return a < b;
}

static inline bool eq_double_impl(double a, double b) {
    return (a == b) || (std::isnan(a) && std::isnan(b));
}

static int64_t unique_double_fast(const double *src, double *dest, int64_t size) {
    if (size <= 0) return 0;
    if (dest != src) memmove(dest, src, static_cast<size_t>(size) * sizeof(double));
    double *non_nan_end = std::partition(dest, dest + size, [](double x) {
        return !std::isnan(x);
    });
    int64_t non_nan_size = non_nan_end - dest;
    if (non_nan_size > 1) {
        hwy::VQSort(dest, static_cast<size_t>(non_nan_size), hwy::SortAscending());
    }
    int64_t write_idx = 0;
    if (non_nan_size > 0) {
        for (int64_t read_idx = 1; read_idx < non_nan_size; read_idx++) {
            if (dest[read_idx] != dest[write_idx]) {
                write_idx++;
                dest[write_idx] = dest[read_idx];
            }
        }
        write_idx++;
    }
    if (non_nan_size < size) {
        dest[write_idx++] = std::numeric_limits<double>::quiet_NaN();
    }
    return write_idx;
}

static int64_t unique_float_fast(const float *src, float *dest, int64_t size) {
    if (size <= 0) return 0;
    if (dest != src) memmove(dest, src, static_cast<size_t>(size) * sizeof(float));
    float *non_nan_end = std::partition(dest, dest + size, [](float x) {
        return !std::isnan(x);
    });
    int64_t non_nan_size = non_nan_end - dest;
    if (non_nan_size > 1) {
        hwy::VQSort(dest, static_cast<size_t>(non_nan_size), hwy::SortAscending());
    }
    int64_t write_idx = 0;
    if (non_nan_size > 0) {
        for (int64_t read_idx = 1; read_idx < non_nan_size; read_idx++) {
            if (dest[read_idx] != dest[write_idx]) {
                write_idx++;
                dest[write_idx] = dest[read_idx];
            }
        }
        write_idx++;
    }
    if (non_nan_size < size) {
        dest[write_idx++] = std::numeric_limits<float>::quiet_NaN();
    }
    return write_idx;
}

static int64_t unique_int32_fast(const int32_t *src, int32_t *dest, int64_t size) {
    if (size <= 0) return 0;
    if (dest != src) memmove(dest, src, static_cast<size_t>(size) * sizeof(int32_t));
    if (size > 1) {
        hwy::VQSort(dest, static_cast<size_t>(size), hwy::SortAscending());
    }
    int64_t write_idx = 0;
    for (int64_t read_idx = 1; read_idx < size; read_idx++) {
        if (dest[read_idx] != dest[write_idx]) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

static int64_t unique_int64_fast(const int64_t *src, int64_t *dest, int64_t size) {
    if (size <= 0) return 0;
    if (dest != src) memmove(dest, src, static_cast<size_t>(size) * sizeof(int64_t));
    if (size > 1) {
        hwy::VQSort((int64_t *)dest, static_cast<size_t>(size), hwy::SortAscending());
    }
    int64_t write_idx = 0;
    for (int64_t read_idx = 1; read_idx < size; read_idx++) {
        if (dest[read_idx] != dest[write_idx]) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

static int64_t unique_int16_fast(const int16_t *src, int16_t *dest, int64_t size) {
    if (size <= 0) return 0;
    if (dest != src) memmove(dest, src, static_cast<size_t>(size) * sizeof(int16_t));
    if (size > 1) {
        hwy::VQSort(dest, static_cast<size_t>(size), hwy::SortAscending());
    }
    int64_t write_idx = 0;
    for (int64_t read_idx = 1; read_idx < size; read_idx++) {
        if (dest[read_idx] != dest[write_idx]) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

static int64_t unique_uint16_fast(const uint16_t *src, uint16_t *dest, int64_t size) {
    if (size <= 0) return 0;
    if (dest != src) memmove(dest, src, static_cast<size_t>(size) * sizeof(uint16_t));
    if (size > 1) {
        hwy::VQSort(dest, static_cast<size_t>(size), hwy::SortAscending());
    }
    int64_t write_idx = 0;
    for (int64_t read_idx = 1; read_idx < size; read_idx++) {
        if (dest[read_idx] != dest[write_idx]) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

static int64_t unique_uint32_fast(const uint32_t *src, uint32_t *dest, int64_t size) {
    if (size <= 0) return 0;
    if (dest != src) memmove(dest, src, static_cast<size_t>(size) * sizeof(uint32_t));
    if (size > 1) {
        hwy::VQSort(dest, static_cast<size_t>(size), hwy::SortAscending());
    }
    int64_t write_idx = 0;
    for (int64_t read_idx = 1; read_idx < size; read_idx++) {
        if (dest[read_idx] != dest[write_idx]) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

static int64_t unique_uint64_fast(const uint64_t *src, uint64_t *dest, int64_t size) {
    if (size <= 0) return 0;
    if (dest != src) memmove(dest, src, static_cast<size_t>(size) * sizeof(uint64_t));
    if (size > 1) {
        hwy::VQSort((uint64_t *)dest, static_cast<size_t>(size), hwy::SortAscending());
    }
    int64_t write_idx = 0;
    for (int64_t read_idx = 1; read_idx < size; read_idx++) {
        if (dest[read_idx] != dest[write_idx]) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

static int64_t unique_uint8_fast(const uint8_t *src, uint8_t *dest, int64_t size) {
    if (size <= 0) return 0;
    bool present[256] = {false};
    for (int64_t i = 0; i < size; i++) {
        present[src[i]] = true;
    }
    int64_t count = 0;
    for (int v = 0; v < 256; v++) {
        if (present[v]) {
            dest[count++] = (uint8_t)v;
        }
    }
    return count;
}

static int64_t unique_int8_fast(const int8_t *src, int8_t *dest, int64_t size) {
    if (size <= 0) return 0;
    bool present[256] = {false};
    for (int64_t i = 0; i < size; i++) {
        present[static_cast<uint8_t>(src[i] ^ 0x80)] = true;
    }
    int64_t count = 0;
    for (int v = 0; v < 256; v++) {
        if (present[v]) {
            dest[count++] = static_cast<int8_t>(v ^ 0x80);
        }
    }
    return count;
}

static int64_t unique_complex128_fast(const complex128_t *src, complex128_t *dest, int64_t size) {
    if (size <= 0) return 0;
    if (dest != src) memmove(dest, src, static_cast<size_t>(size) * sizeof(complex128_t));
    std::sort(dest, dest + size, comp_complex_impl<complex128_t>);
    int64_t write_idx = 0;
    for (int64_t read_idx = 1; read_idx < size; read_idx++) {
        if (!eq_complex_impl(dest[read_idx], dest[write_idx])) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

static int64_t unique_complex64_fast(const complex64_t *src, complex64_t *dest, int64_t size) {
    if (size <= 0) return 0;
    if (dest != src) memmove(dest, src, static_cast<size_t>(size) * sizeof(complex64_t));
    std::sort(dest, dest + size, comp_complex_impl<complex64_t>);
    int64_t write_idx = 0;
    for (int64_t read_idx = 1; read_idx < size; read_idx++) {
        if (!eq_complex_impl(dest[read_idx], dest[write_idx])) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

template<typename T, typename Comp, typename Eq>
static int64_t unique_template(const T *src, T *dest, int64_t size,
                           int64_t *out_index, int64_t *out_inverse, int64_t *out_counts,
                           Comp comp, Eq eq) {
    if (size <= 0) return 0;
    
    NoThrowBuffer<int64_t> idx(static_cast<size_t>(size));
    if (!idx.ok()) {
        ndarray_set_oom_flag();
        return -4;
    }
    for (int64_t i = 0; i < size; i++) idx[static_cast<size_t>(i)] = i;
    
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
        return comp(src[a], src[b]);
    });

    const uint8_t *s_bytes = reinterpret_cast<const uint8_t *>(src);
    const uint8_t *d_bytes = reinterpret_cast<const uint8_t *>(dest);
    size_t byte_len = static_cast<size_t>(size) * sizeof(T);
    bool overlaps = (d_bytes < s_bytes + byte_len) && (s_bytes < d_bytes + byte_len);
    NoThrowBuffer<T> unique_buf(overlaps ? static_cast<size_t>(size) : 0);
    if (overlaps && !unique_buf.ok()) {
        ndarray_set_oom_flag();
        return -4;
    }
    T *out_dest = overlaps ? unique_buf.data() : dest;
    
    int64_t write_idx = 0;
    out_dest[0] = src[idx[0]];
    if (out_index) out_index[0] = idx[0];
    if (out_inverse) out_inverse[idx[0]] = 0;
    
    int64_t current_count = 1;
    
    for (int64_t read_idx = 1; read_idx < size; read_idx++) {
        if (!eq(src[idx[static_cast<size_t>(read_idx)]], src[idx[static_cast<size_t>(read_idx - 1)]])) {
            if (out_counts) out_counts[write_idx] = current_count;
            write_idx++;
            out_dest[write_idx] = src[idx[static_cast<size_t>(read_idx)]];
            if (out_index) out_index[write_idx] = idx[static_cast<size_t>(read_idx)];
            if (out_inverse) out_inverse[idx[static_cast<size_t>(read_idx)]] = write_idx;
            current_count = 1;
        } else {
            if (out_inverse) out_inverse[idx[static_cast<size_t>(read_idx)]] = write_idx;
            current_count++;
        }
    }
    if (out_counts) out_counts[write_idx] = current_count;
    if (overlaps) {
        memcpy(dest, out_dest, static_cast<size_t>(write_idx + 1) * sizeof(T));
    }
    
    return write_idx + 1;
}

static inline double decode_fp16_sort(uint16_t bits) {
    uint16_t sign = (bits >> 15) & 0x1;
    uint16_t exp16 = (bits >> 10) & 0x1F;
    uint16_t frac16 = bits & 0x3FF;
    if (exp16 == 0x1F) {
        if (frac16 == 0) {
            return sign ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
        } else {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    if (exp16 == 0) {
        if (frac16 == 0) return sign ? -0.0 : 0.0;
        double val = (double)frac16 / 1024.0 * 6.103515625e-5;
        return sign ? -val : val;
    }
    uint64_t exp64 = (uint64_t)(exp16 - 15 + 1023);
    uint64_t frac64 = (uint64_t)frac16 << 42;
    uint64_t f64Bits = ((uint64_t)sign << 63) | (exp64 << 52) | frac64;
    double d;
    memcpy(&d, &f64Bits, sizeof(double));
    return d;
}

static inline double decode_bf16_sort(uint16_t bits) {
    uint32_t f32Bits = (uint32_t)bits << 16;
    float f;
    memcpy(&f, &f32Bits, sizeof(float));
    return (double)f;
}

static inline bool comp_fp16_impl(uint16_t a, uint16_t b) {
    return comp_double_impl(decode_fp16_sort(a), decode_fp16_sort(b));
}

static inline bool eq_fp16_impl(uint16_t a, uint16_t b) {
    return eq_double_impl(decode_fp16_sort(a), decode_fp16_sort(b));
}

static inline bool comp_bf16_impl(uint16_t a, uint16_t b) {
    return comp_double_impl(decode_bf16_sort(a), decode_bf16_sort(b));
}

static inline bool eq_bf16_impl(uint16_t a, uint16_t b) {
    return eq_double_impl(decode_bf16_sort(a), decode_bf16_sort(b));
}

static int64_t unique_fp16_fast(const uint16_t *src, uint16_t *dest, int64_t size) {
    if (size <= 0) return 0;
    if (dest != src) memmove(dest, src, static_cast<size_t>(size) * sizeof(uint16_t));
    std::sort(dest, dest + size, comp_fp16_impl);
    int64_t write_idx = 0;
    for (int64_t read_idx = 1; read_idx < size; read_idx++) {
        if (!eq_fp16_impl(dest[read_idx], dest[write_idx])) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

static int64_t unique_bf16_fast(const uint16_t *src, uint16_t *dest, int64_t size) {
    if (size <= 0) return 0;
    if (dest != src) memmove(dest, src, static_cast<size_t>(size) * sizeof(uint16_t));
    std::sort(dest, dest + size, comp_bf16_impl);
    int64_t write_idx = 0;
    for (int64_t read_idx = 1; read_idx < size; read_idx++) {
        if (!eq_bf16_impl(dest[read_idx], dest[write_idx])) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

template<typename T>
static int64_t unique_scalar_fast(const T *src, T *dest, int64_t size) {
    if (size <= 0) return 0;
    if (dest != src) memmove(dest, src, static_cast<size_t>(size) * sizeof(T));
    std::sort(dest, dest + size);
    int64_t write_idx = 0;
    for (int64_t read_idx = 1; read_idx < size; read_idx++) {
        if (dest[read_idx] != dest[write_idx]) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

extern "C" {
int64_t ndarray_unique(const void *src, void *dest, int64_t size, int dtype,
                   int64_t *out_index, int64_t *out_inverse, int64_t *out_counts) {
    if (src == nullptr || dest == nullptr || size <= 0) return 0;
    
    bool has_optional = (out_index != nullptr || out_inverse != nullptr || out_counts != nullptr);
    if (!has_optional) {
        switch (dtype) {
            case DTYPE_FLOAT64:
                return unique_double_fast((const double *)src, (double *)dest, size);
            case DTYPE_FLOAT32:
                return unique_float_fast((const float *)src, (float *)dest, size);
            case DTYPE_INT64:
                return unique_int64_fast((const int64_t *)src, (int64_t *)dest, size);
            case DTYPE_INT32:
                return unique_int32_fast((const int32_t *)src, (int32_t *)dest, size);
            case DTYPE_INT16:
                return unique_int16_fast((const int16_t *)src, (int16_t *)dest, size);
            case DTYPE_INT8:
                return unique_int8_fast((const int8_t *)src, (int8_t *)dest, size);
            case DTYPE_UINT64:
                return unique_uint64_fast((const uint64_t *)src, (uint64_t *)dest, size);
            case DTYPE_UINT32:
                return unique_uint32_fast((const uint32_t *)src, (uint32_t *)dest, size);
            case DTYPE_UINT16:
                return unique_uint16_fast((const uint16_t *)src, (uint16_t *)dest, size);
            case DTYPE_UINT8:
            case DTYPE_BOOLEAN:
                return unique_uint8_fast((const uint8_t *)src, (uint8_t *)dest, size);
            case DTYPE_FLOAT16:
                return unique_fp16_fast((const uint16_t *)src, (uint16_t *)dest, size);
            case DTYPE_BFLOAT16:
                return unique_bf16_fast((const uint16_t *)src, (uint16_t *)dest, size);
            case DTYPE_COMPLEX128:
                return unique_complex128_fast((const complex128_t *)src, (complex128_t *)dest, size);
            case DTYPE_COMPLEX64:
                return unique_complex64_fast((const complex64_t *)src, (complex64_t *)dest, size);
            default:
                return -2;
        }
    }
    
    switch (dtype) {
        case DTYPE_FLOAT64:
            return unique_template<double>(
                (const double *)src, (double *)dest, size,
                out_index, out_inverse, out_counts,
                comp_double_impl, eq_double_impl
            );
        case DTYPE_FLOAT32:
            return unique_template<float>(
                (const float *)src, (float *)dest, size,
                out_index, out_inverse, out_counts,
                [](float a, float b) {
                    bool nan_a = std::isnan(a);
                    bool nan_b = std::isnan(b);
                    if (nan_a && nan_b) return false;
                    if (nan_a) return false;
                    if (nan_b) return true;
                    return a < b;
                },
                [](float a, float b) {
                    return (a == b) || (std::isnan(a) && std::isnan(b));
                }
            );
        case DTYPE_INT64:
            return unique_template<int64_t>(
                (const int64_t *)src, (int64_t *)dest, size,
                out_index, out_inverse, out_counts,
                std::less<int64_t>(), std::equal_to<int64_t>()
            );
        case DTYPE_INT32:
            return unique_template<int32_t>(
                (const int32_t *)src, (int32_t *)dest, size,
                out_index, out_inverse, out_counts,
                std::less<int32_t>(), std::equal_to<int32_t>()
            );
        case DTYPE_INT16:
            return unique_template<int16_t>(
                (const int16_t *)src, (int16_t *)dest, size,
                out_index, out_inverse, out_counts,
                std::less<int16_t>(), std::equal_to<int16_t>()
            );
        case DTYPE_INT8:
            return unique_template<int8_t>(
                (const int8_t *)src, (int8_t *)dest, size,
                out_index, out_inverse, out_counts,
                std::less<int8_t>(), std::equal_to<int8_t>()
            );
        case DTYPE_UINT64:
            return unique_template<uint64_t>(
                (const uint64_t *)src, (uint64_t *)dest, size,
                out_index, out_inverse, out_counts,
                std::less<uint64_t>(), std::equal_to<uint64_t>()
            );
        case DTYPE_UINT32:
            return unique_template<uint32_t>(
                (const uint32_t *)src, (uint32_t *)dest, size,
                out_index, out_inverse, out_counts,
                std::less<uint32_t>(), std::equal_to<uint32_t>()
            );
        case DTYPE_UINT16:
            return unique_template<uint16_t>(
                (const uint16_t *)src, (uint16_t *)dest, size,
                out_index, out_inverse, out_counts,
                std::less<uint16_t>(), std::equal_to<uint16_t>()
            );
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            return unique_template<uint8_t>(
                (const uint8_t *)src, (uint8_t *)dest, size,
                out_index, out_inverse, out_counts,
                std::less<uint8_t>(), std::equal_to<uint8_t>()
            );
        case DTYPE_FLOAT16:
            return unique_template<uint16_t>(
                (const uint16_t *)src, (uint16_t *)dest, size,
                out_index, out_inverse, out_counts,
                comp_fp16_impl, eq_fp16_impl
            );
        case DTYPE_BFLOAT16:
            return unique_template<uint16_t>(
                (const uint16_t *)src, (uint16_t *)dest, size,
                out_index, out_inverse, out_counts,
                comp_bf16_impl, eq_bf16_impl
            );
        case DTYPE_COMPLEX128:
            return unique_template<complex128_t>(
                (const complex128_t *)src, (complex128_t *)dest, size,
                out_index, out_inverse, out_counts,
                comp_complex_impl<complex128_t>, eq_complex_impl<complex128_t>
            );
        case DTYPE_COMPLEX64:
            return unique_template<complex64_t>(
                (const complex64_t *)src, (complex64_t *)dest, size,
                out_index, out_inverse, out_counts,
                comp_complex_impl<complex64_t>, eq_complex_impl<complex64_t>
            );
        default:
            return -2;
    }
}
}
