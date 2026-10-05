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

#ifndef NDARRAY_COMMON_H
#define NDARRAY_COMMON_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>
#include <type_traits>

#if defined(_MSC_VER)
#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT restrict
#endif

#ifndef NDARRAY_EXPORT
#if defined(_WIN32)
#define NDARRAY_EXPORT __declspec(dllexport)
#else
#define NDARRAY_EXPORT __attribute__((visibility("default"), used))
#endif
#endif

#ifndef DTYPE_FLOAT64
#define DTYPE_FLOAT64 0
#define DTYPE_FLOAT32 1
#define DTYPE_FLOAT16 2
#define DTYPE_BFLOAT16 3
#define DTYPE_INT64 4
#define DTYPE_INT32 5
#define DTYPE_INT16 6
#define DTYPE_INT8 7
#define DTYPE_UINT64 8
#define DTYPE_UINT32 9
#define DTYPE_UINT16 10
#define DTYPE_UINT8 11
#define DTYPE_COMPLEX128 12
#define DTYPE_COMPLEX64 13
#define DTYPE_BOOLEAN 14
#endif

extern "C" {
void ndarray_set_oom_flag(void);
int ndarray_consume_oom_flag(void);
int get_and_reset_division_error(void);
}

#ifdef __cplusplus

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

    bool allocate(size_t n) noexcept {
        return resize(n);
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
    explicit operator bool() const noexcept { return ok_ && ptr_ != nullptr; }
    T &operator[](size_t i) noexcept { return ptr_[i]; }
    const T &operator[](size_t i) const noexcept { return ptr_[i]; }
};

constexpr int STACK_RANK_LIMIT = 32;

#define DECLARE_RANK_BUFFER(type, name, rank_expr) \
    type name##_stack[32]; \
    int64_t _r_##name = static_cast<int64_t>(rank_expr); \
    NoThrowBuffer<type> name##_heap(_r_##name > 32 ? static_cast<size_t>(_r_##name) : 0); \
    if (_r_##name > 32 && !name##_heap.ok()) { \
        ndarray_set_oom_flag(); \
        return; \
    } \
    type *name = (_r_##name > 32) ? name##_heap.data() : name##_stack; \
    if (_r_##name <= 32) { \
        if (_r_##name > 0) { \
            std::memset(name##_stack, 0, static_cast<size_t>(_r_##name) * sizeof(type)); \
        } else { \
            name##_stack[0] = static_cast<type>(0); \
        } \
    }

#define DECLARE_RANK_BUFFER_RET(type, name, rank_expr, ret_val) \
    type name##_stack[32]; \
    int64_t _r_##name = static_cast<int64_t>(rank_expr); \
    NoThrowBuffer<type> name##_heap(_r_##name > 32 ? static_cast<size_t>(_r_##name) : 0); \
    if (_r_##name > 32 && !name##_heap.ok()) { \
        ndarray_set_oom_flag(); \
        return ret_val; \
    } \
    type *name = (_r_##name > 32) ? name##_heap.data() : name##_stack; \
    if (_r_##name <= 32) { \
        if (_r_##name > 0) { \
            std::memset(name##_stack, 0, static_cast<size_t>(_r_##name) * sizeof(type)); \
        } else { \
            name##_stack[0] = static_cast<type>(0); \
        } \
    }

static inline bool pad_ranges_overlap(
    const void *src, int64_t src_min_elem, int64_t src_max_elem,
    const void *dest, int64_t dest_min_elem, int64_t dest_max_elem,
    size_t elem_size
) {
    if (src == nullptr || dest == nullptr || elem_size == 0) return false;
    const uint8_t *s_min = static_cast<const uint8_t *>(src) + src_min_elem * static_cast<int64_t>(elem_size);
    const uint8_t *s_max = static_cast<const uint8_t *>(src) + (src_max_elem + 1) * static_cast<int64_t>(elem_size);
    const uint8_t *d_min = static_cast<const uint8_t *>(dest) + dest_min_elem * static_cast<int64_t>(elem_size);
    const uint8_t *d_max = static_cast<const uint8_t *>(dest) + (dest_max_elem + 1) * static_cast<int64_t>(elem_size);
    return (s_min < d_max && d_min < s_max);
}

static inline bool strided_buffers_any_overlap(
    const void *ptr1, const int64_t *strides1, const int64_t *shape1, int64_t rank1, size_t elem_size1,
    const void *ptr2, const int64_t *strides2, const int64_t *shape2, int64_t rank2, size_t elem_size2
) {
    if (ptr1 == nullptr || ptr2 == nullptr || rank1 < 0 || rank2 < 0) return false;
    int64_t min_off1 = 0, max_off1 = 0;
    for (int64_t d = 0; d < rank1; d++) {
        if (shape1 == nullptr || shape1[d] <= 0) return false;
        int64_t s1 = strides1 != nullptr ? strides1[d] : 0;
        int64_t extent = shape1[d] - 1;
        if (s1 < 0) min_off1 += extent * s1;
        else max_off1 += extent * s1;
    }
    int64_t min_off2 = 0, max_off2 = 0;
    for (int64_t d = 0; d < rank2; d++) {
        if (shape2 == nullptr || shape2[d] <= 0) return false;
        int64_t s2 = strides2 != nullptr ? strides2[d] : 0;
        int64_t extent = shape2[d] - 1;
        if (s2 < 0) min_off2 += extent * s2;
        else max_off2 += extent * s2;
    }
    const uint8_t *min1 = static_cast<const uint8_t *>(ptr1) + min_off1 * static_cast<int64_t>(elem_size1);
    const uint8_t *max1 = static_cast<const uint8_t *>(ptr1) + max_off1 * static_cast<int64_t>(elem_size1) + (elem_size1 - 1);
    const uint8_t *min2 = static_cast<const uint8_t *>(ptr2) + min_off2 * static_cast<int64_t>(elem_size2);
    const uint8_t *max2 = static_cast<const uint8_t *>(ptr2) + max_off2 * static_cast<int64_t>(elem_size2) + (elem_size2 - 1);
    return (min1 <= max2 && min2 <= max1);
}

template <typename T>
static inline T ndarray_wrapping_add(T a, T b) noexcept {
    using UT = typename std::make_unsigned<T>::type;
    return static_cast<T>(static_cast<UT>(a) + static_cast<UT>(b));
}

template <typename T>
static inline T ndarray_wrapping_sub(T a, T b) noexcept {
    using UT = typename std::make_unsigned<T>::type;
    return static_cast<T>(static_cast<UT>(a) - static_cast<UT>(b));
}

template <typename T>
static inline T ndarray_wrapping_mul(T a, T b) noexcept {
    using UT = typename std::make_unsigned<T>::type;
    return static_cast<T>(static_cast<UT>(a) * static_cast<UT>(b));
}

template <typename T>
static inline T ndarray_wrapping_neg(T a) noexcept {
    using UT = typename std::make_unsigned<T>::type;
    return static_cast<T>(static_cast<UT>(0) - static_cast<UT>(a));
}

#endif // __cplusplus

#endif // NDARRAY_COMMON_H
