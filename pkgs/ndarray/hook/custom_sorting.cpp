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
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>
#define VQSORT_ENABLED 1
#include "hwy/contrib/sort/vqsort.h"
#include "hwy/highway.h"
#include "hwy/per_target.h"
#include <type_traits>

static thread_local int g_ndarray_oom_flag = 0;

extern "C" {
void ndarray_set_oom_flag(void) {
    g_ndarray_oom_flag = 1;
}

int ndarray_consume_oom_flag(void) {
    int prev = g_ndarray_oom_flag;
    g_ndarray_oom_flag = 0;
    return prev;
}
}

template <typename T>
struct NoThrowBuffer {
    T *ptr_ = nullptr;
    size_t size_ = 0;
    size_t cap_ = 0;
    bool ok_ = true;

    NoThrowBuffer() noexcept = default;
    explicit NoThrowBuffer(size_t n) noexcept {
        resize(n);
    }
    NoThrowBuffer(size_t n, T val) noexcept {
        assign(n, val);
    }
    ~NoThrowBuffer() noexcept {
        std::free(ptr_);
    }
    NoThrowBuffer(const NoThrowBuffer &) = delete;
    NoThrowBuffer &operator=(const NoThrowBuffer &) = delete;

    bool resize(size_t n) noexcept {
        std::free(ptr_);
        ptr_ = nullptr;
        size_ = 0;
        cap_ = 0;
        if (n == 0) {
            ok_ = true;
            return true;
        }
        if (n > static_cast<size_t>(-1) / sizeof(T)) {
            ok_ = false;
            ndarray_set_oom_flag();
            return false;
        }
        ptr_ = static_cast<T *>(std::calloc(n, sizeof(T)));
        if (!ptr_) {
            ok_ = false;
            ndarray_set_oom_flag();
            return false;
        }
        size_ = n;
        cap_ = n;
        ok_ = true;
        return true;
    }

    bool assign(size_t n, T val) noexcept {
        if (!resize(n)) return false;
        const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&val);
        bool is_zero = true;
        for (size_t b = 0; b < sizeof(T); ++b) {
            if (bytes[b] != 0) {
                is_zero = false;
                break;
            }
        }
        if (!is_zero) {
            for (size_t i = 0; i < n; ++i) {
                ptr_[i] = val;
            }
        }
        return true;
    }

    bool assign(const T *first, const T *last) noexcept {
        size_t n = static_cast<size_t>(last - first);
        if (!resize(n)) return false;
        if (n > 0 && first != nullptr) {
            std::memcpy(ptr_, first, n * sizeof(T));
        }
        return true;
    }

    bool push_back(const T &val) noexcept {
        if (size_ == cap_) {
            size_t new_cap = cap_ == 0 ? 8 : (cap_ < 1024 ? cap_ * 2 : cap_ + cap_ / 2);
            if (new_cap <= cap_ || new_cap > static_cast<size_t>(-1) / sizeof(T)) {
                ok_ = false;
                ndarray_set_oom_flag();
                return false;
            }
            T *new_ptr = static_cast<T *>(std::realloc(ptr_, new_cap * sizeof(T)));
            if (!new_ptr) {
                ok_ = false;
                ndarray_set_oom_flag();
                return false;
            }
            ptr_ = new_ptr;
            cap_ = new_cap;
        }
        ptr_[size_++] = val;
        return true;
    }

    T *data() noexcept { return ptr_; }
    const T *data() const noexcept { return ptr_; }
    T *begin() noexcept { return ptr_; }
    T *end() noexcept { return ptr_ + size_; }
    const T *begin() const noexcept { return ptr_; }
    const T *end() const noexcept { return ptr_ + size_; }
    size_t size() const noexcept { return size_; }
    bool ok() const noexcept { return ok_; }
    T &operator[](size_t i) noexcept { return ptr_[i]; }
    const T &operator[](size_t i) const noexcept { return ptr_[i]; }
};

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
// Thread-Local globals for Argsort (Indirect Sorting) data tracking
// ----------------------------------------------------------------------------

static thread_local const double *global_double_data = nullptr;
static thread_local const float *global_float_data = nullptr;
static thread_local const long long *global_int64_data = nullptr;
static thread_local const int *global_int32_data = nullptr;
static thread_local const int16_t *global_int16_data = nullptr;
static thread_local const uint8_t *global_uint8_data = nullptr;

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
// Instantiations of Christopher Swenson's TimSort from third_party/timsort
// ----------------------------------------------------------------------------

#define SORT_NAME tim_double
#define SORT_TYPE double
#define SORT_CMP(x, y) compare_double_inline(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_fast_double
#define SORT_TYPE double
#define SORT_CMP(x, y) compare_double_fast(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_float
#define SORT_TYPE float
#define SORT_CMP(x, y) compare_float_inline(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_fast_float
#define SORT_TYPE float
#define SORT_CMP(x, y) compare_float_fast(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_int64
#define SORT_TYPE long long
#define SORT_CMP(x, y) compare_int64_inline(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_int32
#define SORT_TYPE int
#define SORT_CMP(x, y) compare_int32_inline(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_int16
#define SORT_TYPE int16_t
#define SORT_CMP(x, y) compare_int16_inline(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_uint8
#define SORT_TYPE uint8_t
#define SORT_CMP(x, y) compare_uint8_inline(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_complex128
#define SORT_TYPE complex128_t
#define SORT_CMP(x, y) compare_complex128_inline(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_complex64
#define SORT_TYPE complex64_t
#define SORT_CMP(x, y) compare_complex64_inline(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

// ----------------------------------------------------------------------------
// Comparators for Stable Indirect Sorters (Argsort)
// ----------------------------------------------------------------------------

static inline int compare_indices_double_timsort(int64_t idx_a, int64_t idx_b) {
    double val_a = global_double_data[idx_a];
    double val_b = global_double_data[idx_b];
    int nan_a = isnan(val_a);
    int nan_b = isnan(val_b);
    if (nan_a && nan_b) {
        if (idx_a < idx_b) return -1;
        if (idx_a > idx_b) return 1;
        return 0;
    }
    if (nan_a) return 1;
    if (nan_b) return -1;
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

static inline int compare_indices_float_timsort(int64_t idx_a, int64_t idx_b) {
    float val_a = global_float_data[idx_a];
    float val_b = global_float_data[idx_b];
    int nan_a = isnan(val_a);
    int nan_b = isnan(val_b);
    if (nan_a && nan_b) {
        if (idx_a < idx_b) return -1;
        if (idx_a > idx_b) return 1;
        return 0;
    }
    if (nan_a) return 1;
    if (nan_b) return -1;
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

static inline int compare_indices_int64_timsort(int64_t idx_a, int64_t idx_b) {
    long long val_a = global_int64_data[idx_a];
    long long val_b = global_int64_data[idx_b];
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

static inline int compare_indices_int32_timsort(int64_t idx_a, int64_t idx_b) {
    int val_a = global_int32_data[idx_a];
    int val_b = global_int32_data[idx_b];
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

static inline int compare_indices_int16_timsort(int64_t idx_a, int64_t idx_b) {
    int16_t val_a = global_int16_data[idx_a];
    int16_t val_b = global_int16_data[idx_b];
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

static inline int compare_indices_uint8_timsort(int64_t idx_a, int64_t idx_b) {
    uint8_t val_a = global_uint8_data[idx_a];
    uint8_t val_b = global_uint8_data[idx_b];
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

// ----------------------------------------------------------------------------
// Instantiations of Christopher Swenson's TimSort for Argsort
// ----------------------------------------------------------------------------

#define SORT_NAME tim_indices_double
#define SORT_TYPE int64_t
#define SORT_CMP(x, y) compare_indices_double_timsort(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_indices_float
#define SORT_TYPE int64_t
#define SORT_CMP(x, y) compare_indices_float_timsort(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_indices_int64
#define SORT_TYPE int64_t
#define SORT_CMP(x, y) compare_indices_int64_timsort(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_indices_int32
#define SORT_TYPE int64_t
#define SORT_CMP(x, y) compare_indices_int32_timsort(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_indices_int16
#define SORT_TYPE int64_t
#define SORT_CMP(x, y) compare_indices_int16_timsort(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_indices_uint8
#define SORT_TYPE int64_t
#define SORT_CMP(x, y) compare_indices_uint8_timsort(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

// ----------------------------------------------------------------------------
// C++ Templates for sorting, searching, etc.
// ----------------------------------------------------------------------------

template <typename T>
inline bool is_nonzero(T val) {
    return val != T(0);
}
template <>
inline bool is_nonzero<complex128_t>(complex128_t val) {
    return val.real != 0.0 || val.imag != 0.0;
}
template <>
inline bool is_nonzero<complex64_t>(complex64_t val) {
    return val.real != 0.0f || val.imag != 0.0f;
}

template <typename T>
static inline int standard_compare(T a, T b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

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

    auto cmp = std::less<T>();
    if (k_end - k_start == 1) {
        std::nth_element(array, array + k_ptr[k_start], array + valid_len, cmp);
    } else {
        multi_nth_element(array, 0, valid_len - 1, k_ptr, k_start, k_end - 1, cmp);
    }
}

template <typename T>
static void argpartition_impl(const T *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size) {
    if (data == nullptr || indices == nullptr || size <= 0 || k_list == nullptr || k_size <= 0) return;
    if (size == 1) {
        indices[0] = 0;
        return;
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

    // O(N) pre-pass to check if already non-decreasing or strictly decreasing
    bool is_sorted = true;
    bool is_rev_sorted = true;
    for (int64_t i = 0; i < valid_len - 1; i++) {
        T val_cur = data[indices[i]];
        T val_next = data[indices[i + 1]];
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
        return;
    }
    if (is_rev_sorted) {
        std::reverse(indices, indices + valid_len);
        return;
    }

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

    int64_t first_nan = -1;
    for (int64_t i = 0; i < size; i++) {
        if (std::isnan(array[i])) {
            first_nan = i;
            break;
        }
    }
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
        for (int64_t i = 0; i < size; i++) {
            dest[i] = is_nonzero(src[i]) ? 1 : 0;
        }
        return;
    }
    if (shape == nullptr || strides == nullptr || rank <= 0) return;
    NoThrowBuffer<int64_t> coord_vec;
    int64_t coord_stack[32] = {0};
    int64_t *coord = coord_stack;
    if (rank > 32) {
        coord_vec.assign(rank, 0);
        if (!coord_vec.ok()) return;
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

template <typename T>
static inline bool is_nan_check(T val) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::isnan(val);
    }
    return false;
}

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
        T val0 = src[0];
        if (is_nan_check(val0)) {
            dest[0] = 0;
            return;
        }
        int64_t idx0 = 0, idx1 = 0, idx2 = 0, idx3 = 0;
        T val1 = val0, val2 = val0, val3 = val0;
        int64_t i = 1;
        if (is_max) {
            for (; i + 3 < n; i += 4) {
                T v0 = src[i], v1 = src[i + 1], v2 = src[i + 2], v3 = src[i + 3];
                if (is_nan_check(v0) || is_nan_check(v1) || is_nan_check(v2) || is_nan_check(v3)) break;
                if (v0 > val0) { val0 = v0; idx0 = i; }
                if (v1 > val1) { val1 = v1; idx1 = i + 1; }
                if (v2 > val2) { val2 = v2; idx2 = i + 2; }
                if (v3 > val3) { val3 = v3; idx3 = i + 3; }
            }
            if (val1 > val0 || (val1 == val0 && idx1 < idx0)) { val0 = val1; idx0 = idx1; }
            if (val2 > val0 || (val2 == val0 && idx2 < idx0)) { val0 = val2; idx0 = idx2; }
            if (val3 > val0 || (val3 == val0 && idx3 < idx0)) { val0 = val3; idx0 = idx3; }
            for (; i < n; i++) {
                T val = src[i];
                if (is_nan_check(val)) {
                    idx0 = i;
                    break;
                }
                if (cmp(val, val0) > 0) {
                    val0 = val;
                    idx0 = i;
                }
            }
        } else {
            for (; i + 3 < n; i += 4) {
                T v0 = src[i], v1 = src[i + 1], v2 = src[i + 2], v3 = src[i + 3];
                if (is_nan_check(v0) || is_nan_check(v1) || is_nan_check(v2) || is_nan_check(v3)) break;
                if (v0 < val0) { val0 = v0; idx0 = i; }
                if (v1 < val1) { val1 = v1; idx1 = i + 1; }
                if (v2 < val2) { val2 = v2; idx2 = i + 2; }
                if (v3 < val3) { val3 = v3; idx3 = i + 3; }
            }
            if (val1 < val0 || (val1 == val0 && idx1 < idx0)) { val0 = val1; idx0 = idx1; }
            if (val2 < val0 || (val2 == val0 && idx2 < idx0)) { val0 = val2; idx0 = idx2; }
            if (val3 < val0 || (val3 == val0 && idx3 < idx0)) { val0 = val3; idx0 = idx3; }
            for (; i < n; i++) {
                T val = src[i];
                if (is_nan_check(val)) {
                    idx0 = i;
                    break;
                }
                if (cmp(val, val0) < 0) {
                    val0 = val;
                    idx0 = i;
                }
            }
        }
        dest[0] = idx0;
        return;
    }
    int64_t dest_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) dest_size *= shape[d];
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
        int64_t best_idx = 0;
        int64_t base_src_offset = 0;
        int rank_dest_idx = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                base_src_offset += coord_dest[rank_dest_idx] * stridesSrc[d];
                rank_dest_idx++;
            }
        }
        T best_val = src[base_src_offset];
        if (is_nan_check(best_val)) {
            dest[dest_offset] = 0;
        } else {
            for (int64_t i = 1; i < shape[axis]; i++) {
                int64_t src_offset = base_src_offset + i * stridesSrc[axis];
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
    if (is_contiguous && axis == -1) {
        int64_t count = 0;
        for (int64_t i = 0; i < shape[0]; i++) {
            if (is_nonzero(src[i])) count++;
        }
        dest[0] = count;
        return;
    }
    int64_t dest_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) dest_size *= shape[d];
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
        float val_cur = decode_f16_to_f32(data[indices[i]]);
        float val_next = decode_f16_to_f32(data[indices[i + 1]]);
        if (val_cur > val_next) {
            is_sorted = false;
            if (!is_rev_sorted) break;
        }
        if (val_cur <= val_next) {
            is_rev_sorted = false;
            if (!is_sorted) break;
        }
    }
    if (is_sorted) return;
    if (is_rev_sorted) {
        std::reverse(indices, indices + valid_len);
        return;
    }

    auto cmp = [data](int64_t a, int64_t b) {
        return decode_f16_to_f32(data[a]) < decode_f16_to_f32(data[b]);
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
        float val_cur = decode_bf16_to_f32(data[indices[i]]);
        float val_next = decode_bf16_to_f32(data[indices[i + 1]]);
        if (val_cur > val_next) {
            is_sorted = false;
            if (!is_rev_sorted) break;
        }
        if (val_cur <= val_next) {
            is_rev_sorted = false;
            if (!is_sorted) break;
        }
    }
    if (is_sorted) return;
    if (is_rev_sorted) {
        std::reverse(indices, indices + valid_len);
        return;
    }

    auto cmp = [data](int64_t a, int64_t b) {
        return decode_bf16_to_f32(data[a]) < decode_bf16_to_f32(data[b]);
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

    auto cmp = [data](int64_t a, int64_t b) {
        return decode_f16_to_f32(data[a]) < decode_f16_to_f32(data[b]);
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

    auto cmp = [data](int64_t a, int64_t b) {
        return decode_bf16_to_f32(data[a]) < decode_bf16_to_f32(data[b]);
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
    argminmax((const long long *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_max, is_contiguous, standard_compare<long long>);
}
extern "C" void native_argminmax_int32(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_max, int is_contiguous) {
    argminmax((const int *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_max, is_contiguous, standard_compare<int>);
}
extern "C" void native_argminmax_uint8(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_max, int is_contiguous) {
    argminmax((const unsigned char *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_max, is_contiguous, standard_compare<unsigned char>);
}
extern "C" void native_argminmax_int16(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_max, int is_contiguous) {
    argminmax((const short *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_max, is_contiguous, standard_compare<short>);
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

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

int64_t UnpackMaskImpl(const uint8_t *mask_ptr, int64_t size, int64_t *out_indices) {
    const hn::ScalableTag<int64_t> d;
    using Rebind8 = hn::Rebind<uint8_t, decltype(d)>;
    const Rebind8 d8;
    const int64_t L = hn::Lanes(d);

    int64_t count = 0;
    int64_t j = 0;
    int64_t limit = size - L;
    auto v_base = hn::Iota(d, 0);

    for (; j <= limit; j += L) {
        auto mask_bytes = hn::LoadU(d8, mask_ptr + j);
        auto mask_i64 = hn::PromoteTo(d, mask_bytes);
        auto mask = (mask_i64 != hn::Zero(d));
        auto v_index = hn::Add(v_base, hn::Set(d, j));
        count += hn::CompressStore(v_index, mask, d, out_indices + count);
    }
    for (; j < size; j++) {
        if (mask_ptr[j] != 0) {
            out_indices[count++] = j;
        }
    }
    return count;
}

} // namespace HWY_NAMESPACE
} // namespace hwy
HWY_AFTER_NAMESPACE();

extern "C" {
int64_t unpack_mask_c(
    const uint8_t *mask_ptr,
    int64_t size,
    int64_t stride,
    int64_t *out_indices
) {
    if (mask_ptr == nullptr || out_indices == nullptr || size <= 0) return 0;

    if (stride == 1) {
        return hwy::HWY_NAMESPACE::UnpackMaskImpl(mask_ptr, size, out_indices);
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
    int64_t count = 0;
    for (int64_t i = 0; i < size; i++) {
        if (mask[i] != 0) count++;
    }
    return count;
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
            abort();
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
    memcpy(dest, src, static_cast<size_t>(size) * sizeof(double));
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
    memcpy(dest, src, static_cast<size_t>(size) * sizeof(float));
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
    memcpy(dest, src, static_cast<size_t>(size) * sizeof(int32_t));
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
    memcpy(dest, src, static_cast<size_t>(size) * sizeof(int64_t));
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
    memcpy(dest, src, static_cast<size_t>(size) * sizeof(int16_t));
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
    memcpy(dest, src, static_cast<size_t>(size) * sizeof(uint16_t));
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
    memcpy(dest, src, static_cast<size_t>(size) * sizeof(uint32_t));
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
    memcpy(dest, src, static_cast<size_t>(size) * sizeof(uint64_t));
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
    memcpy(dest, src, static_cast<size_t>(size) * sizeof(complex128_t));
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
    memcpy(dest, src, static_cast<size_t>(size) * sizeof(complex64_t));
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
    
    int64_t write_idx = 0;
    dest[0] = src[idx[0]];
    if (out_index) out_index[0] = idx[0];
    if (out_inverse) out_inverse[idx[0]] = 0;
    
    int64_t current_count = 1;
    
    for (int64_t read_idx = 1; read_idx < size; read_idx++) {
        if (!eq(src[idx[static_cast<size_t>(read_idx)]], src[idx[static_cast<size_t>(read_idx - 1)]])) {
            if (out_counts) out_counts[write_idx] = current_count;
            write_idx++;
            dest[write_idx] = src[idx[static_cast<size_t>(read_idx)]];
            if (out_index) out_index[write_idx] = idx[static_cast<size_t>(read_idx)];
            if (out_inverse) out_inverse[idx[static_cast<size_t>(read_idx)]] = write_idx;
            current_count = 1;
        } else {
            if (out_inverse) out_inverse[idx[static_cast<size_t>(read_idx)]] = write_idx;
            current_count++;
        }
    }
    if (out_counts) out_counts[write_idx] = current_count;
    
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
    memcpy(dest, src, static_cast<size_t>(size) * sizeof(uint16_t));
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
    memcpy(dest, src, static_cast<size_t>(size) * sizeof(uint16_t));
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
    memcpy(dest, src, static_cast<size_t>(size) * sizeof(T));
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
                abort();
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
            abort();
    }
}
}
