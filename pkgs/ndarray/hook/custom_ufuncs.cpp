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

/* ============================================================================
 * ndarray CUSTOM NATIVE EXTENSIONS LIBRARY (custom_ufuncs.c)
 * ============================================================================
 * Contains optimized contiguous vector mathematics, multi-dimensional strided
 * odometer walks, custom complex trigonometric algorithms, pocketfft wrappers,
 * and high-speed sorting utilities.
 *
 * Design Rules:
 * 1. Thread-safe and platform-independent standard C99 arithmetic.
 * 2. Zero-allocation sweeps offloading all layout loops to raw pointer space.
 * 3. Systematically aligned logic groupings mapped directly to FFI bindings.
 * ============================================================================
 */

#include "custom_ufuncs.h"
#include "ndarray_common.h"
#include <cstring>
#include <math.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdlib.h>
#include <complex>
#include <stdio.h>
#include "custom_sorting.h"
#include <type_traits>
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <immintrin.h>
#endif

#define MAKE_BINARY_OP_NAME(code, name) name,
static const char* BINARY_OP_NAMES[] = {
    FOR_EACH_BINARY_OP(MAKE_BINARY_OP_NAME)
};
#undef MAKE_BINARY_OP_NAME

const char* get_binary_op_name(int index) {
    if (index < 0 || index >= OP_COUNT) return nullptr;
    return BINARY_OP_NAMES[index];
}

int get_binary_op_count(void) {
    return (int)OP_COUNT;
}

int get_binary_op_enum_val(int index) {
    if (index < 0 || index >= (int)OP_COUNT) return -1;
    return index;
}

#ifndef VECTORIZED_TARGETS
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__APPLE__) && (defined(__x86_64__) || defined(__i386__)) && !defined(_WIN32)
#define VECTORIZED_TARGETS __attribute__((target_clones("avx512f", "avx2", "sse4.2", "default")))
#else
#define VECTORIZED_TARGETS
#endif
#endif
// ============================================================================
// C++ TEMPLATES FOR CASTING, SET OPERATIONS, AND STRIDED LOOPS
// ============================================================================

// Forward declarations of helper functions defined later
static inline void xoshiro256_seed(uint64_t seed, uint64_t s[4]);
static inline uint64_t xoshiro256_next(uint64_t s[4]);
static void fill_secure_bytes(void *dest, size_t size);

template <typename T>
inline T logaddexp_op(T x, T y) {
    if (std::isnan(x) || std::isnan(y)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (x == y) {
        return x + std::log(static_cast<T>(2.0));
    }
    T max_val = std::max(x, y);
    T min_val = std::min(x, y);
    return max_val + std::log1p(std::exp(min_val - max_val));
}

template <typename T>
inline T logaddexp2_op(T x, T y) {
    if (std::isnan(x) || std::isnan(y)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (x == y) {
        return x + static_cast<T>(1.0);
    }
    T max_val = std::max(x, y);
    T min_val = std::min(x, y);
    T ln2 = std::log(static_cast<T>(2.0));
    return max_val + std::log1p(std::exp((min_val - max_val) * ln2)) / ln2;
}

// --- Integer Exponentiation Helper ---
template <typename T>
struct ipow_helper {
    static inline T pow(T base, T exp) {
        if constexpr (std::is_signed<T>::value) {
            if (exp < 0) {
                if (base == 1) return 1;
                if (base == -1) return (exp % 2 == 0) ? 1 : -1;
                return 0;
            }
        }
        using UT = typename std::make_unsigned<T>::type;
        UT result = 1;
        UT b = static_cast<UT>(base);
        UT e = static_cast<UT>(exp);
        while (e > 0) {
            if (e & 1) result = static_cast<UT>(result * b);
            b = static_cast<UT>(b * b);
            e >>= 1;
        }
        return static_cast<T>(result);
    }
};

template <typename T>
inline T ipow(T base, T exp) {
    return ipow_helper<T>::pow(base, exp);
}

// --- Float16 and BFloat16 Helpers ---
static inline uint16_t encode_float16(double value) {
    uint64_t f64Bits;
    memcpy(&f64Bits, &value, sizeof(double));
    uint16_t sign = (uint16_t)((f64Bits >> 63) & 0x1);
    int exp64 = (int)((f64Bits >> 52) & 0x7FF);
    uint64_t frac64 = f64Bits & 0xFFFFFFFFFFFFFULL;

    if (exp64 == 0x7FF) {
        if (frac64 == 0) {
            return (uint16_t)((sign << 15) | 0x7C00);
        } else {
            return (uint16_t)((sign << 15) | 0x7E00);
        }
    }
    if (exp64 == 0) {
        return (uint16_t)(sign << 15);
    }
    int exp = exp64 - 1023;
    if (exp > 15) {
        return (uint16_t)((sign << 15) | 0x7C00);
    }
    if (exp < -25) {
        return (uint16_t)(sign << 15);
    }
    uint64_t mantissa = 0x10000000000000ULL | frac64;
    if (exp < -14) {
        int shift = 42 + (-14 - exp);
        uint64_t halfBit = 1ULL << (shift - 1);
        uint64_t mask = halfBit - 1ULL;
        uint16_t frac16 = (uint16_t)(mantissa >> shift);
        uint64_t rem = mantissa & (halfBit | mask);
        if (rem > halfBit || (rem == halfBit && (frac16 & 1) != 0)) {
            frac16++;
        }
        return (uint16_t)((sign << 15) | frac16);
    }
    uint16_t exp16 = (uint16_t)(exp + 15);
    uint16_t frac16 = (uint16_t)((mantissa >> 42) & 0x3FF);
    uint64_t halfBit = 1ULL << 41;
    uint64_t mask = halfBit - 1ULL;
    uint64_t rem = mantissa & (halfBit | mask);
    uint16_t combined = (uint16_t)((exp16 << 10) | frac16);
    if (rem > halfBit || (rem == halfBit && (frac16 & 1) != 0)) {
        combined++;
    }
    return (uint16_t)((sign << 15) | combined);
}

static inline double decode_float16(uint16_t bits) {
    uint16_t sign = (bits >> 15) & 0x1;
    uint16_t exp16 = (bits >> 10) & 0x1F;
    uint16_t frac16 = bits & 0x3FF;

    if (exp16 == 0x1F) {
        if (frac16 == 0) {
            return sign == 1 ? -std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::infinity();
        } else {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    if (exp16 == 0) {
        if (frac16 == 0) {
            return sign == 1 ? -0.0 : 0.0;
        }
        double val = (double)frac16 / 1024.0 * 6.103515625e-5;
        return sign == 1 ? -val : val;
    }
    uint64_t exp64 = (uint64_t)(exp16 - 15 + 1023);
    uint64_t frac64 = ((uint64_t)frac16) << 42;
    uint64_t f64Bits = (((uint64_t)sign) << 63) | (exp64 << 52) | frac64;
    double res;
    memcpy(&res, &f64Bits, sizeof(double));
    return res;
}

static inline uint16_t encode_bfloat16(double value) {
    uint64_t f64Bits;
    memcpy(&f64Bits, &value, sizeof(double));
    uint16_t sign = (uint16_t)((f64Bits >> 63) & 0x1);
    int exp64 = (int)((f64Bits >> 52) & 0x7FF);
    uint64_t frac64 = f64Bits & 0xFFFFFFFFFFFFFULL;

    if (exp64 == 0x7FF) {
        if (frac64 == 0) {
            return (uint16_t)((sign << 15) | 0x7F80);
        } else {
            return (uint16_t)((sign << 15) | 0x7FC0);
        }
    }
    if (exp64 == 0) {
        return (uint16_t)(sign << 15);
    }
    int exp = exp64 - 1023;
    if (exp > 127) {
        return (uint16_t)((sign << 15) | 0x7F80);
    }
    if (exp < -134) {
        return (uint16_t)(sign << 15);
    }
    uint64_t mantissa = 0x10000000000000ULL | frac64;
    if (exp < -126) {
        int shift = 45 + (-126 - exp);
        uint64_t halfBit = 1ULL << (shift - 1);
        uint64_t mask = halfBit - 1ULL;
        uint16_t frac16 = (uint16_t)(mantissa >> shift);
        uint64_t rem = mantissa & (halfBit | mask);
        if (rem > halfBit || (rem == halfBit && (frac16 & 1) != 0)) {
            frac16++;
        }
        return (uint16_t)((sign << 15) | frac16);
    }
    uint16_t exp16 = (uint16_t)(exp + 127);
    uint16_t frac16 = (uint16_t)((mantissa >> 45) & 0x7F);
    uint64_t halfBit = 1ULL << 44;
    uint64_t mask = halfBit - 1ULL;
    uint64_t rem = mantissa & (halfBit | mask);
    uint16_t combined = (uint16_t)((exp16 << 7) | frac16);
    if (rem > halfBit || (rem == halfBit && (frac16 & 1) != 0)) {
        combined++;
    }
    return (uint16_t)((sign << 15) | combined);
}

static inline double decode_bfloat16(uint16_t bits) {
    uint32_t f32Bits = ((uint32_t)bits) << 16;
    float f;
    memcpy(&f, &f32Bits, sizeof(float));
    return (double)f;
}

struct bfloat16_t;

struct float16_t {
    uint16_t bits;
    float16_t() : bits(0) {}
    float16_t(double v) : bits(encode_float16(v)) {}
    float16_t(float v) : bits(encode_float16((double)v)) {}
    template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
    float16_t(T v) : bits(encode_float16((double)v)) {}
    explicit float16_t(bfloat16_t v);
    operator double() const { return decode_float16(bits); }
    static inline float16_t from_bits(uint16_t b) { float16_t f; f.bits = b; return f; }
};

struct bfloat16_t {
    uint16_t bits;
    bfloat16_t() : bits(0) {}
    bfloat16_t(double v) : bits(encode_bfloat16(v)) {}
    bfloat16_t(float v) : bits(encode_bfloat16((double)v)) {}
    template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
    bfloat16_t(T v) : bits(encode_bfloat16((double)v)) {}
    explicit bfloat16_t(float16_t v) : bits(encode_bfloat16(decode_float16(v.bits))) {}
    operator double() const { return decode_bfloat16(bits); }
    static inline bfloat16_t from_bits(uint16_t b) { bfloat16_t f; f.bits = b; return f; }
};

inline float16_t::float16_t(bfloat16_t v) : bits(encode_float16(decode_bfloat16(v.bits))) {}

inline bool operator==(const float16_t& x, const float16_t& y) { return (double)x == (double)y; }
inline bool operator!=(const float16_t& x, const float16_t& y) { return (double)x != (double)y; }
inline bool operator<(const float16_t& x, const float16_t& y) { return (double)x < (double)y; }
inline bool operator<=(const float16_t& x, const float16_t& y) { return (double)x <= (double)y; }
inline bool operator>(const float16_t& x, const float16_t& y) { return (double)x > (double)y; }
inline bool operator>=(const float16_t& x, const float16_t& y) { return (double)x >= (double)y; }

inline bool operator==(const bfloat16_t& x, const bfloat16_t& y) { return (double)x == (double)y; }
inline bool operator!=(const bfloat16_t& x, const bfloat16_t& y) { return (double)x != (double)y; }
inline bool operator<(const bfloat16_t& x, const bfloat16_t& y) { return (double)x < (double)y; }
inline bool operator<=(const bfloat16_t& x, const bfloat16_t& y) { return (double)x <= (double)y; }
inline bool operator>(const bfloat16_t& x, const bfloat16_t& y) { return (double)x > (double)y; }
inline bool operator>=(const bfloat16_t& x, const bfloat16_t& y) { return (double)x >= (double)y; }

static_assert(sizeof(float16_t) == 2, "float16_t must be 2 bytes");
static_assert(sizeof(bfloat16_t) == 2, "bfloat16_t must be 2 bytes");

// --- Saturating Float-to-Int Helper ---
template <typename IntT>
static inline IntT saturating_float_to_int(double val) {
    if (std::isnan(val)) return static_cast<IntT>(0);
    if constexpr (std::is_same<IntT, int64_t>::value) {
        if (val >= 9223372036854775808.0) return INT64_MAX;
        if (val <= -9223372036854775808.0) return INT64_MIN;
        return static_cast<int64_t>(val);
    } else if constexpr (std::is_same<IntT, uint64_t>::value) {
        if (val <= 0.0) return 0ULL;
        if (val >= 18446744073709551616.0) return UINT64_MAX;
        if (val >= 9223372036854775808.0) {
            return static_cast<uint64_t>(val - 9223372036854775808.0) + 9223372036854775808ULL;
        }
        return static_cast<uint64_t>(val);
    } else {
        constexpr double kMin = static_cast<double>(std::numeric_limits<IntT>::min());
        constexpr double kMax = static_cast<double>(std::numeric_limits<IntT>::max());
        if (val >= kMax) return std::numeric_limits<IntT>::max();
        if (val <= kMin) return std::numeric_limits<IntT>::min();
        return static_cast<IntT>(val);
    }
}

// --- Casting Templates ---
template <typename DestType, typename SrcType>
struct caster {
    static inline DestType cast(SrcType val) {
        return static_cast<DestType>(val);
    }
};

struct bool_byte_t { uint8_t v; };

template <typename SrcType>
struct caster<bool_byte_t, SrcType> {
    static inline bool_byte_t cast(SrcType val) {
        return bool_byte_t{static_cast<uint8_t>(val != 0 ? 1 : 0)};
    }
};
template <>
struct caster<bool_byte_t, float16_t> {
    static inline bool_byte_t cast(float16_t val) {
        return bool_byte_t{static_cast<uint8_t>(decode_float16(val.bits) != 0.0f ? 1 : 0)};
    }
};
template <>
struct caster<bool_byte_t, bfloat16_t> {
    static inline bool_byte_t cast(bfloat16_t val) {
        return bool_byte_t{static_cast<uint8_t>(decode_bfloat16(val.bits) != 0.0f ? 1 : 0)};
    }
};
template <>
struct caster<bool_byte_t, cpx_t> {
    static inline bool_byte_t cast(cpx_t val) {
        return bool_byte_t{static_cast<uint8_t>((val.r != 0.0 || val.i != 0.0) ? 1 : 0)};
    }
};
template <>
struct caster<bool_byte_t, cpx_f_t> {
    static inline bool_byte_t cast(cpx_f_t val) {
        return bool_byte_t{static_cast<uint8_t>((val.r != 0.0f || val.i != 0.0f) ? 1 : 0)};
    }
};
template <>
struct caster<bool_byte_t, bool_byte_t> {
    static inline bool_byte_t cast(bool_byte_t val) {
        return bool_byte_t{static_cast<uint8_t>(val.v != 0 ? 1 : 0)};
    }
};

// Casting FROM complex to scalar (returns real part cast to DestType)
template <typename DestType>
struct caster<DestType, cpx_t> {
    static inline DestType cast(cpx_t val) { return static_cast<DestType>(val.r); }
};

template <typename DestType>
struct caster<DestType, cpx_f_t> {
    static inline DestType cast(cpx_f_t val) { return static_cast<DestType>(val.r); }
};

// Specialize float/complex -> integer casts to use saturating + NaN->0 semantics
#define DEFINE_SATURATING_INT_CASTERS(INT_T) \
template <> struct caster<INT_T, double> { \
    static inline INT_T cast(double val) { return saturating_float_to_int<INT_T>(val); } \
}; \
template <> struct caster<INT_T, float> { \
    static inline INT_T cast(float val) { return saturating_float_to_int<INT_T>(static_cast<double>(val)); } \
}; \
template <> struct caster<INT_T, float16_t> { \
    static inline INT_T cast(float16_t val) { return saturating_float_to_int<INT_T>(decode_float16(val.bits)); } \
}; \
template <> struct caster<INT_T, bfloat16_t> { \
    static inline INT_T cast(bfloat16_t val) { return saturating_float_to_int<INT_T>(decode_bfloat16(val.bits)); } \
}; \
template <> struct caster<INT_T, cpx_t> { \
    static inline INT_T cast(cpx_t val) { return saturating_float_to_int<INT_T>(val.r); } \
}; \
template <> struct caster<INT_T, cpx_f_t> { \
    static inline INT_T cast(cpx_f_t val) { return saturating_float_to_int<INT_T>(static_cast<double>(val.r)); } \
};

DEFINE_SATURATING_INT_CASTERS(int64_t)
DEFINE_SATURATING_INT_CASTERS(int32_t)
DEFINE_SATURATING_INT_CASTERS(int16_t)
DEFINE_SATURATING_INT_CASTERS(int8_t)
DEFINE_SATURATING_INT_CASTERS(uint64_t)
DEFINE_SATURATING_INT_CASTERS(uint32_t)
DEFINE_SATURATING_INT_CASTERS(uint16_t)
DEFINE_SATURATING_INT_CASTERS(uint8_t)
#undef DEFINE_SATURATING_INT_CASTERS

// Casting FROM scalar to complex (sets real part, imag is 0)
template <typename SrcType>
struct caster<cpx_t, SrcType> {
    static inline cpx_t cast(SrcType val) { return cpx_t{static_cast<double>(val), 0.0}; }
};

template <typename SrcType>
struct caster<cpx_f_t, SrcType> {
    static inline cpx_f_t cast(SrcType val) { return cpx_f_t{static_cast<float>(val), 0.0f}; }
};

// Complex to Complex casts
template <>
struct caster<cpx_t, cpx_t> {
    static inline cpx_t cast(cpx_t val) { return val; }
};

template <>
struct caster<cpx_f_t, cpx_f_t> {
    static inline cpx_f_t cast(cpx_f_t val) { return val; }
};

template <>
struct caster<cpx_t, cpx_f_t> {
    static inline cpx_t cast(cpx_f_t val) { return cpx_t{(double)val.r, (double)val.i}; }
};

template <>
struct caster<cpx_f_t, cpx_t> {
    static inline cpx_f_t cast(cpx_t val) { return cpx_f_t{(float)val.r, (float)val.i}; }
};

inline bool contiguous_buffers_overlap_not_identical(const void* p1, size_t bytes1, const void* p2, size_t bytes2) {
    if (p1 == nullptr || p2 == nullptr || p1 == p2) return false;
    return (static_cast<const uint8_t*>(p1) < static_cast<const uint8_t*>(p2) + bytes2 &&
            static_cast<const uint8_t*>(p2) < static_cast<const uint8_t*>(p1) + bytes1);
}

static inline bool strided_buffers_overlap_not_identical(
    const void *ptr1, const int64_t *strides1, size_t elem_size1,
    const void *ptr2, const int64_t *strides2, size_t elem_size2,
    const int64_t *shape, int rank
) {
    if (ptr1 == nullptr || ptr2 == nullptr || rank < 0) return false;
    if (rank == 0) {
        if (ptr1 == ptr2 && elem_size1 == elem_size2) return false;
        const uint8_t *min1 = static_cast<const uint8_t *>(ptr1);
        const uint8_t *max1 = min1 + elem_size1 - 1;
        const uint8_t *min2 = static_cast<const uint8_t *>(ptr2);
        const uint8_t *max2 = min2 + elem_size2 - 1;
        return (min1 <= max2 && min2 <= max1);
    }
    if (strides1 == nullptr || strides2 == nullptr || shape == nullptr) return false;

    intptr_t min_off1 = 0, max_off1 = 0;
    intptr_t min_off2 = 0, max_off2 = 0;
    for (int d = 0; d < rank; d++) {
        if (shape[d] <= 0) return false;
        intptr_t s1 = strides1[d];
        intptr_t s2 = strides2[d];
        intptr_t extent = shape[d] - 1;
        if (s1 < 0) min_off1 += extent * s1;
        else max_off1 += extent * s1;
        if (s2 < 0) min_off2 += extent * s2;
        else max_off2 += extent * s2;
    }

    const uint8_t *min1 = static_cast<const uint8_t *>(ptr1) + min_off1 * (intptr_t)elem_size1;
    const uint8_t *max1 = static_cast<const uint8_t *>(ptr1) + max_off1 * (intptr_t)elem_size1 + (elem_size1 - 1);
    const uint8_t *min2 = static_cast<const uint8_t *>(ptr2) + min_off2 * (intptr_t)elem_size2;
    const uint8_t *max2 = static_cast<const uint8_t *>(ptr2) + max_off2 * (intptr_t)elem_size2 + (elem_size2 - 1);

    if (!(min1 <= max2 && min2 <= max1)) {
        return false;
    }

    bool same_strides = (elem_size1 == elem_size2);
    if (same_strides) {
        for (int d = 0; d < rank; d++) {
            if (shape[d] > 1) {
                if (strides1[d] != strides2[d] || strides1[d] == 0) {
                    same_strides = false;
                    break;
                }
            }
        }
    }

    return !(ptr1 == ptr2 && same_strides);
}

template <typename SrcType, typename DestType>
static void s_cast_generic_impl(
    const void *src_ptr, const int64_t *stridesSrc,
    void *dest_ptr,
    const int64_t *shape, int rank, int64_t total_elements
) {
    const SrcType *src = (const SrcType *)src_ptr;
    DestType *dest = (DestType *)dest_ptr;
    if (rank == 0) {
        dest[0] = caster<DestType, SrcType>::cast(src[0]);
        return;
    }
    if (rank <= 0) return;
    DECLARE_RANK_BUFFER(int64_t, stridesDest, rank);
    int64_t s = 1;
    for (int d = rank - 1; d >= 0; d--) {
        stridesDest[d] = s;
        s *= shape[d];
    }
    if (strided_buffers_overlap_not_identical(src, stridesSrc, sizeof(SrcType), dest, stridesDest, sizeof(DestType), shape, rank)) {
        NoThrowBuffer<DestType> temp(total_elements);
        if (!temp.ok()) return;
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetSrc = 0;
        for (int64_t el = 0; el < total_elements; el++) {
            temp[el] = caster<DestType, SrcType>::cast(src[offsetSrc]);
            for (int d = rank - 1; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetSrc += stridesSrc[d];
                    break;
                }
                coord[d] = 0;
                offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            }
        }
        std::memcpy(dest, temp.data(), total_elements * sizeof(DestType));
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        dest[el] = caster<DestType, SrcType>::cast(src[offsetSrc]);
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}

template <typename T>
static inline T real_sinc(T x) {
    if (std::isinf(x)) return (T)0.0;
    const T pi = (T)3.14159265358979323846;
    T abs_x = std::abs(x);
    if (abs_x < (T)1e-4) {
        T x2 = x * x;
        T pi2 = pi * pi;
        return (T)1.0 - (pi2 * x2) / (T)6.0 + (pi2 * pi2 * x2 * x2) / (T)120.0;
    } else {
        return std::sin(pi * x) / (pi * x);
    }
}

template <typename T>
static inline std::complex<T> complex_sinc_impl(std::complex<T> cz) {
    if (std::isinf(std::abs(cz)) || std::isinf(cz.real()) || std::isinf(cz.imag())) {
        return std::complex<T>((T)0.0, (T)0.0);
    }
    const T pi = (T)3.14159265358979323846;
    T abs_z = std::abs(cz);
    if (abs_z < (T)1e-4) {
        std::complex<T> z2 = cz * cz;
        T pi2 = pi * pi;
        return (T)1.0 - (pi2 * z2) / (T)6.0 + (pi2 * pi2 * z2 * z2) / (T)120.0;
    } else {
        std::complex<T> pi_z = pi * cz;
        return std::sin(pi_z) / pi_z;
    }
}

// --- Set Operations Templates ---
template <typename T>
struct set_comparator {
    static inline int compare(T a, T b) {
        if (a < b) return -1;
        if (a > b) return 1;
        return 0;
    }
};

template <>
struct set_comparator<double> {
    static inline int compare(double a, double b) {
        bool nan_a = std::isnan(a);
        bool nan_b = std::isnan(b);
        if (nan_a && nan_b) return 0;
        if (nan_a) return 1;
        if (nan_b) return -1;
        if (a < b) return -1;
        if (a > b) return 1;
        return 0;
    }
};

template <>
struct set_comparator<float> {
    static inline int compare(float a, float b) {
        return set_comparator<double>::compare((double)a, (double)b);
    }
};

template <>
struct set_comparator<float16_t> {
    static inline int compare(float16_t a, float16_t b) {
        return set_comparator<double>::compare((double)a, (double)b);
    }
};

template <>
struct set_comparator<bfloat16_t> {
    static inline int compare(bfloat16_t a, bfloat16_t b) {
        return set_comparator<double>::compare((double)a, (double)b);
    }
};

template <>
struct set_comparator<cpx_t> {
    static inline int compare(cpx_t a, cpx_t b) {
        int cmp_r = set_comparator<double>::compare(a.r, b.r);
        if (cmp_r != 0) return cmp_r;
        return set_comparator<double>::compare(a.i, b.i);
    }
};

template <>
struct set_comparator<cpx_f_t> {
    static inline int compare(cpx_f_t a, cpx_f_t b) {
        int cmp_r = set_comparator<float>::compare(a.r, b.r);
        if (cmp_r != 0) return cmp_r;
        return set_comparator<float>::compare(a.i, b.i);
    }
};

template <typename T>
struct set_is_nan {
    static inline bool check(T) { return false; }
};
template <>
struct set_is_nan<double> {
    static inline bool check(double a) { return std::isnan(a); }
};
template <>
struct set_is_nan<float> {
    static inline bool check(float a) { return std::isnan(a); }
};
template <>
struct set_is_nan<float16_t> {
    static inline bool check(float16_t a) { return std::isnan((double)a); }
};
template <>
struct set_is_nan<bfloat16_t> {
    static inline bool check(bfloat16_t a) { return std::isnan((double)a); }
};
template <>
struct set_is_nan<cpx_t> {
    static inline bool check(cpx_t a) { return std::isnan(a.r) || std::isnan(a.i); }
};
template <>
struct set_is_nan<cpx_f_t> {
    static inline bool check(cpx_f_t a) { return std::isnan(a.r) || std::isnan(a.i); }
};

template <typename T>
static int64_t intersect1d_impl(const T *ar1, int64_t size1, const T *ar2, int64_t size2, T *dest) {
    int64_t i = 0, j = 0, k = 0;
    while (i < size1 && j < size2) {
        if (set_is_nan<T>::check(ar1[i])) {
            i++;
            continue;
        }
        if (set_is_nan<T>::check(ar2[j])) {
            j++;
            continue;
        }
        int cmp = set_comparator<T>::compare(ar1[i], ar2[j]);
        if (cmp < 0) {
            i++;
        } else if (cmp > 0) {
            j++;
        } else {
            dest[k++] = ar1[i];
            i++;
            j++;
        }
    }
    return k;
}

template <typename T>
static int64_t setdiff1d_impl(const T *ar1, int64_t size1, const T *ar2, int64_t size2, T *dest) {
    int64_t i = 0, j = 0, k = 0;
    while (i < size1) {
        if (j >= size2) {
            dest[k++] = ar1[i++];
            continue;
        }
        int cmp = set_comparator<T>::compare(ar1[i], ar2[j]);
        if (cmp < 0) {
            dest[k++] = ar1[i++];
        } else if (cmp > 0) {
            j++;
        } else {
            i++;
        }
    }
    return k;
}

template <typename T>
static int64_t setxor1d_impl(const T *ar1, int64_t size1, const T *ar2, int64_t size2, T *dest) {
    int64_t i = 0, j = 0, k = 0;
    while (i < size1 || j < size2) {
        if (i >= size1) {
            dest[k++] = ar2[j++];
            continue;
        }
        if (j >= size2) {
            dest[k++] = ar1[i++];
            continue;
        }
        int cmp = set_comparator<T>::compare(ar1[i], ar2[j]);
        if (cmp < 0) {
            dest[k++] = ar1[i++];
        } else if (cmp > 0) {
            dest[k++] = ar2[j++];
        } else {
            i++;
            j++;
        }
    }
    return k;
}

template <typename T>
static int64_t union1d_impl(const T *ar1, int64_t size1, const T *ar2, int64_t size2, T *dest) {
    int64_t i = 0, j = 0, k = 0;
    while (i < size1 || j < size2) {
        if (i >= size1) {
            dest[k++] = ar2[j++];
            continue;
        }
        if (j >= size2) {
            dest[k++] = ar1[i++];
            continue;
        }
        int cmp = set_comparator<T>::compare(ar1[i], ar2[j]);
        if (cmp < 0) {
            dest[k++] = ar1[i++];
        } else if (cmp > 0) {
            dest[k++] = ar2[j++];
        } else {
            dest[k++] = ar1[i];
            i++;
            j++;
        }
    }
    return k;
}

template <typename T>
static void isin_impl(const T *ar1, int64_t size1, const T *ar2, int64_t size2, uint8_t *dest, int invert) {
    for (int64_t i = 0; i < size1; i++) {
        T val = ar1[i];
        int found = 0;
        if (!set_is_nan<T>::check(val)) {
            int64_t low = 0;
            int64_t high = size2 - 1;
            while (low <= high) {
                int64_t mid = low + (high - low) / 2;
                if (set_is_nan<T>::check(ar2[mid])) {
                    high = mid - 1;
                    continue;
                }
                int cmp = set_comparator<T>::compare(ar2[mid], val);
                if (cmp < 0) {
                    low = mid + 1;
                } else if (cmp > 0) {
                    high = mid - 1;
                } else {
                    found = 1;
                    break;
                }
            }
        }
        dest[i] = invert ? !found : found;
    }
}

// --- Strided Loop Templates ---
template <typename T, typename Op>
static void strided_cum_op_impl(
    const T *src, const int64_t *stridesSrc,
    T *res, const int64_t *stridesRes,
    const int64_t *shape, int rank, int axis, Op op
) {
    if (src == nullptr || res == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t total_size = 1;
    for (int d = 0; d < rank; d++) {
        if (shape[d] <= 0) return;
        total_size *= shape[d];
    }
    if (strided_buffers_overlap_not_identical(src, stridesSrc, sizeof(T), res, stridesRes, sizeof(T), shape, rank)) {
        DECLARE_RANK_BUFFER(int64_t, tempStrides, rank);
        int64_t s = 1;
        for (int d = rank - 1; d >= 0; d--) {
            tempStrides[d] = s;
            s *= shape[d];
        }
        NoThrowBuffer<T> temp(total_size);
        if (!temp.ok()) return;
        strided_cum_op_impl(src, stridesSrc, temp.data(), tempStrides, shape, rank, axis, op);
        DECLARE_RANK_BUFFER(int64_t, coord_out, rank);
        for (int64_t i = 0; i < total_size; i++) {
            int64_t offsetRes = 0;
            for (int d = 0; d < rank; d++) {
                offsetRes += coord_out[d] * stridesRes[d];
            }
            res[offsetRes] = temp[i];
            for (int d = rank - 1; d >= 0; d--) {
                coord_out[d]++;
                if (coord_out[d] < shape[d]) break;
                coord_out[d] = 0;
            }
        }
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetSrc = 0;
        int64_t offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                offsetRes += coord[d] * stridesRes[d];
            }
        }
        T acc;
        for (int64_t i = 0; i < shape[axis]; i++) {
            int64_t idxSrc = offsetSrc + i * stridesSrc[axis];
            int64_t idxRes = offsetRes + i * stridesRes[axis];
            acc = (i == 0) ? src[idxSrc] : op(acc, src[idxSrc]);
            res[idxRes] = acc;
        }
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

template <typename T>
static void strided_unwrap_op_impl(
    const T *src, const int64_t *stridesSrc,
    T *res, const int64_t *stridesRes,
    const int64_t *shape, int rank, int axis, T discont
) {
    if (src == nullptr || res == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t total_size = 1;
    for (int d = 0; d < rank; d++) {
        if (shape[d] <= 0) return;
        total_size *= shape[d];
    }
    if (strided_buffers_overlap_not_identical(src, stridesSrc, sizeof(T), res, stridesRes, sizeof(T), shape, rank)) {
        DECLARE_RANK_BUFFER(int64_t, tempStrides, rank);
        int64_t s = 1;
        for (int d = rank - 1; d >= 0; d--) {
            tempStrides[d] = s;
            s *= shape[d];
        }
        NoThrowBuffer<T> temp(total_size);
        if (!temp.ok()) return;
        strided_unwrap_op_impl(src, stridesSrc, temp.data(), tempStrides, shape, rank, axis, discont);
        DECLARE_RANK_BUFFER(int64_t, coord_out, rank);
        for (int64_t i = 0; i < total_size; i++) {
            int64_t offsetRes = 0;
            for (int d = 0; d < rank; d++) {
                offsetRes += coord_out[d] * stridesRes[d];
            }
            res[offsetRes] = temp[i];
            for (int d = rank - 1; d >= 0; d--) {
                coord_out[d]++;
                if (coord_out[d] < shape[d]) break;
                coord_out[d] = 0;
            }
        }
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    const T pi = (T)3.14159265358979323846;
    const T two_pi = 2 * pi;

    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetSrc = 0;
        int64_t offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                offsetRes += coord[d] * stridesRes[d];
            }
        }
        T prev_val = 0;
        T cum_correction = 0;
        for (int64_t i = 0; i < shape[axis]; i++) {
            int64_t idxSrc = offsetSrc + i * stridesSrc[axis];
            int64_t idxRes = offsetRes + i * stridesRes[axis];
            T val = src[idxSrc];
            if (i == 0) {
                res[idxRes] = val;
            } else {
                T dd = val - prev_val;
                T ddmod = dd + pi;
                T q = std::floor(ddmod / two_pi);
                ddmod = ddmod - q * two_pi;
                ddmod -= pi;
                if (std::abs(ddmod + pi) < 1e-9 && dd > 0) {
                    ddmod = pi;
                }
                T ph_correct = ddmod - dd;
                if (std::abs(dd) < discont) {
                    ph_correct = 0;
                }
                cum_correction += ph_correct;
                res[idxRes] = val + cum_correction;
            }
            prev_val = val;
        }
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

template <typename T, typename Op>
static void strided_diff_op_impl(
    const T *src, const int64_t *stridesSrc,
    T *res, const int64_t *stridesRes,
    const int64_t *shape, int rank, int axis, Op op
) {
    if (src == nullptr || res == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    if (shape[axis] <= 1) return;
    DECLARE_RANK_BUFFER(int64_t, resShape, rank);
    int64_t total_res_size = 1;
    for (int d = 0; d < rank; d++) {
        if (shape[d] <= 0) return;
        resShape[d] = (d == axis) ? (shape[d] - 1) : shape[d];
        total_res_size *= resShape[d];
    }
    if (total_res_size <= 0) return;
    if (strided_buffers_any_overlap(src, stridesSrc, shape, rank, sizeof(T), res, stridesRes, resShape, rank, sizeof(T))) {
        DECLARE_RANK_BUFFER(int64_t, tempStrides, rank);
        int64_t s = 1;
        for (int d = rank - 1; d >= 0; d--) {
            tempStrides[d] = s;
            s *= resShape[d];
        }
        NoThrowBuffer<T> temp(total_res_size);
        if (!temp.ok()) {
            ndarray_set_oom_flag();
            return;
        }
        strided_diff_op_impl(src, stridesSrc, temp.data(), tempStrides, shape, rank, axis, op);
        DECLARE_RANK_BUFFER(int64_t, coord_out, rank);
        for (int64_t i = 0; i < total_res_size; i++) {
            int64_t offsetRes = 0;
            for (int d = 0; d < rank; d++) {
                offsetRes += coord_out[d] * stridesRes[d];
            }
            res[offsetRes] = temp[i];
            for (int d = rank - 1; d >= 0; d--) {
                coord_out[d]++;
                if (coord_out[d] < resShape[d]) break;
                coord_out[d] = 0;
            }
        }
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetSrc = 0;
        int64_t offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                offsetRes += coord[d] * stridesRes[d];
            }
        }
        for (int64_t i = 0; i < shape[axis] - 1; i++) {
            int64_t idxSrcCurr = offsetSrc + i * stridesSrc[axis];
            int64_t idxSrcNext = offsetSrc + (i + 1) * stridesSrc[axis];
            int64_t idxRes = offsetRes + i * stridesRes[axis];
            res[idxRes] = op(src[idxSrcNext], src[idxSrcCurr]);
        }
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

static inline bool mask_overlaps_strided_buffer(
    const uint8_t *mask, int64_t total_size,
    const void *res, const int64_t *stridesRes, size_t elem_size_res,
    const int64_t *shape, int rank
) {
    if (mask == nullptr || res == nullptr || total_size <= 0 || rank < 0) return false;
    const uint8_t *min_m = mask;
    const uint8_t *max_m = mask + total_size - 1;
    if (rank == 0) {
        const uint8_t *min_r = static_cast<const uint8_t *>(res);
        const uint8_t *max_r = min_r + elem_size_res - 1;
        return (min_m <= max_r && min_r <= max_m);
    }
    if (stridesRes == nullptr || shape == nullptr) return false;
    intptr_t min_off = 0, max_off = 0;
    for (int d = 0; d < rank; d++) {
        if (shape[d] <= 0) return false;
        intptr_t s = stridesRes[d];
        intptr_t extent = shape[d] - 1;
        if (s < 0) min_off += extent * s;
        else max_off += extent * s;
    }
    const uint8_t *min_r = static_cast<const uint8_t *>(res) + min_off * (intptr_t)elem_size_res;
    const uint8_t *max_r = static_cast<const uint8_t *>(res) + max_off * (intptr_t)elem_size_res + (elem_size_res - 1);
    return (min_m <= max_r && min_r <= max_m);
}

static inline bool mask_overlaps_contiguous_buffer(const uint8_t *mask, int64_t size, const void *res, size_t bytes_res) {
    if (mask == nullptr || res == nullptr || size <= 0 || bytes_res == 0) return false;
    const uint8_t *p1 = mask;
    const uint8_t *p2 = static_cast<const uint8_t *>(res);
    return (p1 < p2 + bytes_res && p2 < p1 + (size_t)size);
}

template <bool HAS_MASK, typename T, typename Op>
static inline void strided_unary_op_loop(
    const T *src, const int64_t *stridesSrc,
    T *res, const int64_t *stridesRes,
    const int64_t *shape, int rank,
    const uint8_t *mask, Op op
) {
    if (rank < 0) return;
    int64_t total_size = 1;
    for (int d = 0; d < rank; d++) total_size *= shape[d];
    if (total_size <= 0) return;

    bool overlap = strided_buffers_overlap_not_identical(
        src, stridesSrc, sizeof(T),
        res, stridesRes, sizeof(T),
        shape, rank
    ) || (HAS_MASK && mask_overlaps_strided_buffer(mask, total_size, res, stridesRes, sizeof(T), shape, rank));

    if (overlap) {
        using TempT = typename std::conditional<std::is_same<T, bool>::value, uint8_t, T>::type;
        NoThrowBuffer<TempT> temp(total_size);
        if (!temp.ok()) {
            ndarray_set_oom_flag();
            return;
        }
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        for (int64_t i = 0; i < total_size; i++) {
            int64_t offsetSrc = 0;
            int64_t offsetRes = 0;
            for (int d = 0; d < rank; d++) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (HAS_MASK) offsetRes += coord[d] * stridesRes[d];
            }
            if (!HAS_MASK || mask[i]) {
                temp[i] = static_cast<TempT>(op(src[offsetSrc]));
            } else {
                temp[i] = static_cast<TempT>(res[offsetRes]);
            }
            for (int d = rank - 1; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) break;
                coord[d] = 0;
            }
        }
        DECLARE_RANK_BUFFER(int64_t, coord_out, rank);
        for (int64_t i = 0; i < total_size; i++) {
            int64_t offsetRes = 0;
            for (int d = 0; d < rank; d++) {
                offsetRes += coord_out[d] * stridesRes[d];
            }
            res[offsetRes] = static_cast<T>(temp[i]);
            for (int d = rank - 1; d >= 0; d--) {
                coord_out[d]++;
                if (coord_out[d] < shape[d]) break;
                coord_out[d] = 0;
            }
        }
        return;
    }

    if (!HAS_MASK) {
        if (rank == 0) {
            res[0] = op(src[0]);
            return;
        }
        if (rank == 1) {
            const int64_t sSrc = stridesSrc[0];
            const int64_t sRes = stridesRes[0];
            if (sSrc == 1 && sRes == 1) {
                for (int64_t i = 0; i < total_size; i++) {
                    res[i] = op(src[i]);
                }
                return;
            }
            for (int64_t i = 0; i < total_size; i++) {
                res[i * sRes] = op(src[i * sSrc]);
            }
            return;
        }
        if (rank == 2 && stridesSrc[0] == 1 && stridesSrc[1] == shape[0] &&
            stridesRes[0] == shape[1] && stridesRes[1] == 1) {
            const int64_t M = shape[0];
            const int64_t N = shape[1];
            constexpr int64_t TILE = 32;
            for (int64_t i0 = 0; i0 < M; i0 += TILE) {
                int64_t bh = std::min(TILE, M - i0);
                for (int64_t j0 = 0; j0 < N; j0 += TILE) {
                    int64_t bw = std::min(TILE, N - j0);
                    T tile[TILE][TILE];
                    for (int64_t jj = 0; jj < bw; jj++) {
                        const T *src_row = src + (j0 + jj) * M + i0;
                        for (int64_t ii = 0; ii < bh; ii++) {
                            tile[jj][ii] = op(src_row[ii]);
                        }
                    }
                    for (int64_t ii = 0; ii < bh; ii++) {
                        T *res_row = res + (i0 + ii) * N + j0;
                        for (int64_t jj = 0; jj < bw; jj++) {
                            res_row[jj] = tile[jj][ii];
                        }
                    }
                }
            }
            return;
        }
        if (rank >= 2) {
            const int64_t inner_n = shape[rank - 1];
            const int64_t sSrc = stridesSrc[rank - 1];
            const int64_t sRes = stridesRes[rank - 1];
            int64_t outer_elements = 1;
            for (int d = 0; d < rank - 1; d++) outer_elements *= shape[d];

            DECLARE_RANK_BUFFER(int64_t, coord, rank);
            int64_t offsetSrc = 0, offsetRes = 0;

            if (sSrc == 1 && sRes == 1) {
                for (int64_t out_idx = 0; out_idx < outer_elements; out_idx++) {
                    const T *rowSrc = src + offsetSrc;
                    T *rowRes = res + offsetRes;
                    for (int64_t j = 0; j < inner_n; j++) {
                        rowRes[j] = op(rowSrc[j]);
                    }
                    for (int d = rank - 2; d >= 0; d--) {
                        coord[d]++;
                        if (coord[d] < shape[d]) {
                            offsetSrc += stridesSrc[d];
                            offsetRes += stridesRes[d];
                            break;
                        }
                        coord[d] = 0;
                        offsetSrc -= (shape[d] - 1) * stridesSrc[d];
                        offsetRes -= (shape[d] - 1) * stridesRes[d];
                    }
                }
                return;
            }

            for (int64_t out_idx = 0; out_idx < outer_elements; out_idx++) {
                const T *rowSrc = src + offsetSrc;
                T *rowRes = res + offsetRes;
                for (int64_t j = 0; j < inner_n; j++) {
                    rowRes[j * sRes] = op(rowSrc[j * sSrc]);
                }
                for (int d = rank - 2; d >= 0; d--) {
                    coord[d]++;
                    if (coord[d] < shape[d]) {
                        offsetSrc += stridesSrc[d];
                        offsetRes += stridesRes[d];
                        break;
                    }
                    coord[d] = 0;
                    offsetSrc -= (shape[d] - 1) * stridesSrc[d];
                    offsetRes -= (shape[d] - 1) * stridesRes[d];
                }
            }
            return;
        }
    }

    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    for (int64_t i = 0; i < total_size; i++) {
        if (!HAS_MASK || mask[i]) {
            int64_t offsetSrc = 0;
            int64_t offsetRes = 0;
            for (int d = 0; d < rank; d++) {
                offsetSrc += coord[d] * stridesSrc[d];
                offsetRes += coord[d] * stridesRes[d];
            }
            res[offsetRes] = op(src[offsetSrc]);
        }
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

template <typename T, typename Op>
static void strided_unary_op_impl(
    const T *src, const int64_t *stridesSrc,
    T *res, const int64_t *stridesRes,
    const int64_t *shape, int rank,
    const uint8_t *mask, Op op
) {
    if (src == nullptr || res == nullptr || (rank > 0 && shape == nullptr) || rank < 0) return;
    if (mask != nullptr) {
        strided_unary_op_loop<true>(src, stridesSrc, res, stridesRes, shape, rank, mask, op);
    } else {
        strided_unary_op_loop<false>(src, stridesSrc, res, stridesRes, shape, rank, nullptr, op);
    }
}

template <bool HAS_MASK, typename T, typename Op>
static inline void v_unary_loop(const T *src, T *res, int64_t size, const uint8_t *mask, Op op) {
    for (int64_t i = 0; i < size; i++) {
        if (HAS_MASK) {
            if (mask[i]) {
                res[i] = op(src[i]);
            }
        } else {
            res[i] = op(src[i]);
        }
    }
}

template <typename T, typename Op>
static void v_unary_impl(const T *src, T *res, int64_t size, const uint8_t *mask, Op op) {
    if (src == nullptr || res == nullptr || size <= 0) return;
    if (contiguous_buffers_overlap_not_identical(src, (size_t)size * sizeof(T), res, (size_t)size * sizeof(T)) ||
        mask_overlaps_contiguous_buffer(mask, size, res, (size_t)size * sizeof(T))) {
        using TempT = typename std::conditional<std::is_same<T, bool>::value, uint8_t, T>::type;
        NoThrowBuffer<TempT> temp(size);
        if (!temp.ok()) {
            ndarray_set_oom_flag();
            return;
        }
        if (mask != nullptr) {
            std::memcpy(temp.data(), res, (size_t)size * sizeof(T));
            v_unary_loop<true>(src, reinterpret_cast<T *>(temp.data()), size, mask, op);
        } else {
            v_unary_loop<false>(src, reinterpret_cast<T *>(temp.data()), size, nullptr, op);
        }
        std::memcpy(res, temp.data(), (size_t)size * sizeof(T));
        return;
    }
    if (mask != nullptr) {
        v_unary_loop<true>(src, res, size, mask, op);
    } else {
        v_unary_loop<false>(src, res, size, nullptr, op);
    }
}

template <bool HAS_MASK, typename T_IN, typename T_OUT, typename Op>
static inline void strided_unary_cast_loop(
    const T_IN *src, const int64_t *stridesSrc,
    T_OUT *res, const int64_t *stridesRes,
    const int64_t *shape, int rank,
    const uint8_t *mask, Op op
) {
    if (rank < 0) return;
    int64_t total_size = 1;
    for (int d = 0; d < rank; d++) total_size *= shape[d];
    if (total_size <= 0) return;

    bool overlap = strided_buffers_overlap_not_identical(
        src, stridesSrc, sizeof(T_IN),
        res, stridesRes, sizeof(T_OUT),
        shape, rank
    ) || (HAS_MASK && mask_overlaps_strided_buffer(mask, total_size, res, stridesRes, sizeof(T_OUT), shape, rank));

    if (overlap) {
        using TempT = typename std::conditional<std::is_same<T_OUT, bool>::value, uint8_t, T_OUT>::type;
        NoThrowBuffer<TempT> temp(total_size);
        if (!temp.ok()) {
            ndarray_set_oom_flag();
            return;
        }
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        for (int64_t i = 0; i < total_size; i++) {
            int64_t offsetSrc = 0;
            int64_t offsetRes = 0;
            for (int d = 0; d < rank; d++) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (HAS_MASK) offsetRes += coord[d] * stridesRes[d];
            }
            if (!HAS_MASK || mask[i]) {
                temp[i] = static_cast<TempT>(op(src[offsetSrc]));
            } else {
                temp[i] = static_cast<TempT>(res[offsetRes]);
            }
            for (int d = rank - 1; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) break;
                coord[d] = 0;
            }
        }
        DECLARE_RANK_BUFFER(int64_t, coord_out, rank);
        for (int64_t i = 0; i < total_size; i++) {
            int64_t offsetRes = 0;
            for (int d = 0; d < rank; d++) {
                offsetRes += coord_out[d] * stridesRes[d];
            }
            res[offsetRes] = static_cast<T_OUT>(temp[i]);
            for (int d = rank - 1; d >= 0; d--) {
                coord_out[d]++;
                if (coord_out[d] < shape[d]) break;
                coord_out[d] = 0;
            }
        }
        return;
    }

    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    for (int64_t i = 0; i < total_size; i++) {
        if (!HAS_MASK || mask[i]) {
            int64_t offsetSrc = 0;
            int64_t offsetRes = 0;
            for (int d = 0; d < rank; d++) {
                offsetSrc += coord[d] * stridesSrc[d];
                offsetRes += coord[d] * stridesRes[d];
            }
            res[offsetRes] = op(src[offsetSrc]);
        }
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

template <typename T_IN, typename T_OUT, typename Op>
static void strided_unary_cast_impl(
    const T_IN *src, const int64_t *stridesSrc,
    T_OUT *res, const int64_t *stridesRes,
    const int64_t *shape, int rank,
    const uint8_t *mask, Op op
) {
    if (src == nullptr || res == nullptr || (rank > 0 && shape == nullptr) || rank < 0) return;
    if (mask != nullptr) {
        strided_unary_cast_loop<true>(src, stridesSrc, res, stridesRes, shape, rank, mask, op);
    } else {
        strided_unary_cast_loop<false>(src, stridesSrc, res, stridesRes, shape, rank, nullptr, op);
    }
}

template <bool HAS_MASK, typename T_IN, typename T_OUT, typename Op>
static inline void v_unary_cast_loop(const T_IN *src, T_OUT *res, int64_t size, const uint8_t *mask, Op op) {
    for (int64_t i = 0; i < size; i++) {
        if (HAS_MASK) {
            if (mask[i]) {
                res[i] = op(src[i]);
            }
        } else {
            res[i] = op(src[i]);
        }
    }
}

template <typename T_IN, typename T_OUT, typename Op>
static void v_unary_cast_impl(const T_IN *src, T_OUT *res, int64_t size, const uint8_t *mask, Op op) {
    if (src == nullptr || res == nullptr || size <= 0) return;
    if (contiguous_buffers_overlap_not_identical(src, (size_t)size * sizeof(T_IN), res, (size_t)size * sizeof(T_OUT)) ||
        (static_cast<const void *>(src) == static_cast<const void *>(res) && sizeof(T_IN) != sizeof(T_OUT)) ||
        mask_overlaps_contiguous_buffer(mask, size, res, (size_t)size * sizeof(T_OUT))) {
        using TempT = typename std::conditional<std::is_same<T_OUT, bool>::value, uint8_t, T_OUT>::type;
        NoThrowBuffer<TempT> temp(size);
        if (!temp.ok()) {
            ndarray_set_oom_flag();
            return;
        }
        if (mask != nullptr) {
            std::memcpy(temp.data(), res, (size_t)size * sizeof(T_OUT));
            v_unary_cast_loop<true>(src, reinterpret_cast<T_OUT *>(temp.data()), size, mask, op);
        } else {
            v_unary_cast_loop<false>(src, reinterpret_cast<T_OUT *>(temp.data()), size, nullptr, op);
        }
        std::memcpy(res, temp.data(), (size_t)size * sizeof(T_OUT));
        return;
    }
    if (mask != nullptr) {
        v_unary_cast_loop<true>(src, res, size, mask, op);
    } else {
        v_unary_cast_loop<false>(src, res, size, nullptr, op);
    }
}

template <typename T>
static void v_fill_impl(T * RESTRICT res, T value, int64_t size) {
    if (res == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = value;
    }
}

template <typename T>
static void s_fill_impl(T * RESTRICT res, const int64_t * RESTRICT strides, const int64_t * RESTRICT shape, int rank, T value) {
    if (res == nullptr || strides == nullptr || shape == nullptr || rank < 0) return;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        res[0] = value;
        return;
    }
    NoThrowBuffer<int64_t> coord_vec;
    int64_t coord_stack[32] = {0};
    int64_t *coord = coord_stack;
    if (rank > 32) {
        coord_vec.assign(rank, 0);
        if (!coord_vec.ok()) {
            ndarray_set_oom_flag();
            return;
        }
        coord = coord_vec.data();
    }
    int64_t offset = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        res[offset] = value;
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

template <typename T, typename BoundsType, typename RangeType>
static void v_randint_impl(T *res, int64_t size, BoundsType low, BoundsType high, uint64_t seed) {
    if (res == nullptr || size <= 0 || low >= high) return;
    uint64_t s[4];
    xoshiro256_seed(seed, s);
    uint64_t low_u = static_cast<uint64_t>(static_cast<int64_t>(low));
    uint64_t high_u = static_cast<uint64_t>(static_cast<int64_t>(high));
    uint64_t range = high_u - low_u;
    for (int64_t i = 0; i < size; i++) {
        uint64_t offset = (range == 0) ? xoshiro256_next(s) : (xoshiro256_next(s) % range);
        res[i] = static_cast<T>(low_u + offset);
    }
}

template <bool HAS_MASK, typename T1, typename T2, typename TRes, typename Op>
static inline void strided_binary_op_loop(
    const T1 *a, const int64_t *stridesA,
    const T2 *b, const int64_t *stridesB,
    TRes *res, const int64_t *stridesRes,
    const int64_t *shape, int rank,
    const uint8_t *mask, Op op
) {
    if (rank < 0) return;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (total_elements <= 0) return;

    bool overlap =
        strided_buffers_overlap_not_identical(a, stridesA, sizeof(T1), res, stridesRes, sizeof(TRes), shape, rank) ||
        strided_buffers_overlap_not_identical(b, stridesB, sizeof(T2), res, stridesRes, sizeof(TRes), shape, rank) ||
        (HAS_MASK && mask_overlaps_strided_buffer(mask, total_elements, res, stridesRes, sizeof(TRes), shape, rank));

    if (overlap) {
        using TempT = typename std::conditional<std::is_same<TRes, bool>::value, uint8_t, TRes>::type;
        NoThrowBuffer<TempT> temp(total_elements);
        if (!temp.ok()) {
            ndarray_set_oom_flag();
            return;
        }
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetA = 0, offsetB = 0, offsetResIn = 0;
        for (int64_t el = 0; el < total_elements; el++) {
            if (!HAS_MASK || mask[el]) {
                temp[el] = static_cast<TempT>(op(a[offsetA], b[offsetB]));
            } else {
                temp[el] = static_cast<TempT>(res[offsetResIn]);
            }
            for (int d = rank - 1; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetA += stridesA[d];
                    offsetB += stridesB[d];
                    offsetResIn += stridesRes[d];
                    break;
                }
                coord[d] = 0;
                offsetA -= (shape[d] - 1) * stridesA[d];
                offsetB -= (shape[d] - 1) * stridesB[d];
                offsetResIn -= (shape[d] - 1) * stridesRes[d];
            }
        }
        DECLARE_RANK_BUFFER(int64_t, coord_out, rank);
        int64_t offsetRes = 0;
        for (int64_t el = 0; el < total_elements; el++) {
            res[offsetRes] = static_cast<TRes>(temp[el]);
            for (int d = rank - 1; d >= 0; d--) {
                coord_out[d]++;
                if (coord_out[d] < shape[d]) {
                    offsetRes += stridesRes[d];
                    break;
                }
                coord_out[d] = 0;
                offsetRes -= (shape[d] - 1) * stridesRes[d];
            }
        }
        return;
    }

    if (!HAS_MASK) {
        if (rank == 0) {
            res[0] = op(a[0], b[0]);
            return;
        }
        if (rank == 1) {
            const int64_t sA = stridesA[0];
            const int64_t sB = stridesB[0];
            const int64_t sR = stridesRes[0];
            if (sA == 1 && sB == 0 && sR == 1) {
                const T2 b0 = b[0];
                for (int64_t i = 0; i < total_elements; i++) {
                    res[i] = op(a[i], b0);
                }
                return;
            }
            if (sA == 0 && sB == 1 && sR == 1) {
                const T1 a0 = a[0];
                for (int64_t i = 0; i < total_elements; i++) {
                    res[i] = op(a0, b[i]);
                }
                return;
            }
            if (sA == 1 && sB == 1 && sR == 1) {
                for (int64_t i = 0; i < total_elements; i++) {
                    res[i] = op(a[i], b[i]);
                }
                return;
            }
            for (int64_t i = 0; i < total_elements; i++) {
                res[i * sR] = op(a[i * sA], b[i * sB]);
            }
            return;
        }
        if (rank >= 2) {
            const int64_t inner_n = shape[rank - 1];
            const int64_t sA = stridesA[rank - 1];
            const int64_t sB = stridesB[rank - 1];
            const int64_t sR = stridesRes[rank - 1];
            int64_t outer_elements = 1;
            for (int d = 0; d < rank - 1; d++) outer_elements *= shape[d];

            DECLARE_RANK_BUFFER(int64_t, coord, rank);
            int64_t offsetA = 0, offsetB = 0, offsetRes = 0;

            if (sA == 1 && sB == 0 && sR == 1) {
                for (int64_t out_idx = 0; out_idx < outer_elements; out_idx++) {
                    const T2 b0 = b[offsetB];
                    const T1 *rowA = a + offsetA;
                    TRes *rowRes = res + offsetRes;
                    for (int64_t j = 0; j < inner_n; j++) {
                        rowRes[j] = op(rowA[j], b0);
                    }
                    for (int d = rank - 2; d >= 0; d--) {
                        coord[d]++;
                        if (coord[d] < shape[d]) {
                            offsetA += stridesA[d];
                            offsetB += stridesB[d];
                            offsetRes += stridesRes[d];
                            break;
                        }
                        coord[d] = 0;
                        offsetA -= (shape[d] - 1) * stridesA[d];
                        offsetB -= (shape[d] - 1) * stridesB[d];
                        offsetRes -= (shape[d] - 1) * stridesRes[d];
                    }
                }
                return;
            }
            if (sA == 0 && sB == 1 && sR == 1) {
                for (int64_t out_idx = 0; out_idx < outer_elements; out_idx++) {
                    const T1 a0 = a[offsetA];
                    const T2 *rowB = b + offsetB;
                    TRes *rowRes = res + offsetRes;
                    for (int64_t j = 0; j < inner_n; j++) {
                        rowRes[j] = op(a0, rowB[j]);
                    }
                    for (int d = rank - 2; d >= 0; d--) {
                        coord[d]++;
                        if (coord[d] < shape[d]) {
                            offsetA += stridesA[d];
                            offsetB += stridesB[d];
                            offsetRes += stridesRes[d];
                            break;
                        }
                        coord[d] = 0;
                        offsetA -= (shape[d] - 1) * stridesA[d];
                        offsetB -= (shape[d] - 1) * stridesB[d];
                        offsetRes -= (shape[d] - 1) * stridesRes[d];
                    }
                }
                return;
            }
            if (sA == 1 && sB == 1 && sR == 1) {
                for (int64_t out_idx = 0; out_idx < outer_elements; out_idx++) {
                    const T1 *rowA = a + offsetA;
                    const T2 *rowB = b + offsetB;
                    TRes *rowRes = res + offsetRes;
                    for (int64_t j = 0; j < inner_n; j++) {
                        rowRes[j] = op(rowA[j], rowB[j]);
                    }
                    for (int d = rank - 2; d >= 0; d--) {
                        coord[d]++;
                        if (coord[d] < shape[d]) {
                            offsetA += stridesA[d];
                            offsetB += stridesB[d];
                            offsetRes += stridesRes[d];
                            break;
                        }
                        coord[d] = 0;
                        offsetA -= (shape[d] - 1) * stridesA[d];
                        offsetB -= (shape[d] - 1) * stridesB[d];
                        offsetRes -= (shape[d] - 1) * stridesRes[d];
                    }
                }
                return;
            }
            for (int64_t out_idx = 0; out_idx < outer_elements; out_idx++) {
                const T1 *rowA = a + offsetA;
                const T2 *rowB = b + offsetB;
                TRes *rowRes = res + offsetRes;
                for (int64_t j = 0; j < inner_n; j++) {
                    rowRes[j * sR] = op(rowA[j * sA], rowB[j * sB]);
                }
                for (int d = rank - 2; d >= 0; d--) {
                    coord[d]++;
                    if (coord[d] < shape[d]) {
                        offsetA += stridesA[d];
                        offsetB += stridesB[d];
                        offsetRes += stridesRes[d];
                        break;
                    }
                    coord[d] = 0;
                    offsetA -= (shape[d] - 1) * stridesA[d];
                    offsetB -= (shape[d] - 1) * stridesB[d];
                    offsetRes -= (shape[d] - 1) * stridesRes[d];
                }
            }
            return;
        }
    }

    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetA = 0, offsetB = 0, offsetRes = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        if (!HAS_MASK || mask[el]) {
            res[offsetRes] = op(a[offsetA], b[offsetB]);
        }
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetA += stridesA[d];
                offsetB += stridesB[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetA -= (shape[d] - 1) * stridesA[d];
            offsetB -= (shape[d] - 1) * stridesB[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

template <typename T1, typename T2, typename TRes, typename Op>
static void strided_binary_op_impl(
    const T1 *a, const int64_t *stridesA,
    const T2 *b, const int64_t *stridesB,
    TRes *res, const int64_t *stridesRes,
    const int64_t *shape, int rank,
    const uint8_t *mask, Op op
) {
    if (a == nullptr || b == nullptr || res == nullptr || (rank > 0 && (shape == nullptr || stridesA == nullptr || stridesB == nullptr || stridesRes == nullptr)) || rank < 0) return;
    if (mask != nullptr) {
        strided_binary_op_loop<true>(a, stridesA, b, stridesB, res, stridesRes, shape, rank, mask, op);
    } else {
        strided_binary_op_loop<false>(a, stridesA, b, stridesB, res, stridesRes, shape, rank, nullptr, op);
    }
}

template <bool HAS_MASK, typename T1, typename T2, typename TRes, typename Op>
static inline void strided_binary_cast_loop(
    const T1 *a, const int64_t *stridesA,
    const T2 *b, const int64_t *stridesB,
    TRes *res, const int64_t *stridesRes,
    const int64_t *shape, int rank,
    const uint8_t *mask, Op op
) {
    strided_binary_op_loop<HAS_MASK, T1, T2, TRes, Op>(
        a, stridesA, b, stridesB, res, stridesRes, shape, rank, mask, op
    );
}

template <typename T1, typename T2, typename TRes, typename Op>
static void strided_binary_cast_impl(
    const T1 *a, const int64_t *stridesA,
    const T2 *b, const int64_t *stridesB,
    TRes *res, const int64_t *stridesRes,
    const int64_t *shape, int rank,
    const uint8_t *mask, Op op
) {
    strided_binary_op_impl<T1, T2, TRes, Op>(
        a, stridesA, b, stridesB, res, stridesRes, shape, rank, mask, op
    );
}

template <bool HAS_MASK, typename T1, typename T2, typename TRes, typename Op>
static inline void v_binary_loop(const T1 *a, const T2 *b, TRes *res, int64_t size, const uint8_t *mask, Op op) {
    for (int64_t i = 0; i < size; i++) {
        if (HAS_MASK) {
            if (mask[i]) {
                res[i] = op(a[i], b[i]);
            }
        } else {
            res[i] = op(a[i], b[i]);
        }
    }
}

template <typename T1, typename T2, typename TRes, typename Op>
static void v_binary_impl(const T1 *a, const T2 *b, TRes *res, int64_t size, const uint8_t *mask, Op op) {
    if (a == nullptr || b == nullptr || res == nullptr || size <= 0) return;
    bool overlap_a = contiguous_buffers_overlap_not_identical(a, (size_t)size * sizeof(T1), res, (size_t)size * sizeof(TRes)) ||
                     (static_cast<const void *>(a) == static_cast<const void *>(res) && sizeof(T1) != sizeof(TRes));
    bool overlap_b = contiguous_buffers_overlap_not_identical(b, (size_t)size * sizeof(T2), res, (size_t)size * sizeof(TRes)) ||
                     (static_cast<const void *>(b) == static_cast<const void *>(res) && sizeof(T2) != sizeof(TRes));
    bool overlap_mask = mask_overlaps_contiguous_buffer(mask, size, res, (size_t)size * sizeof(TRes));
    if (overlap_a || overlap_b || overlap_mask) {
        using TempT = typename std::conditional<std::is_same<TRes, bool>::value, uint8_t, TRes>::type;
        NoThrowBuffer<TempT> temp(size);
        if (!temp.ok()) {
            ndarray_set_oom_flag();
            return;
        }
        if (mask != nullptr) {
            std::memcpy(temp.data(), res, (size_t)size * sizeof(TRes));
            v_binary_loop<true>(a, b, reinterpret_cast<TRes *>(temp.data()), size, mask, op);
        } else {
            v_binary_loop<false>(a, b, reinterpret_cast<TRes *>(temp.data()), size, nullptr, op);
        }
        std::memcpy(res, temp.data(), (size_t)size * sizeof(TRes));
        return;
    }
    if (mask != nullptr) {
        v_binary_loop<true>(a, b, res, size, mask, op);
    } else {
        v_binary_loop<false>(a, b, res, size, nullptr, op);
    }
}

template <typename T>
static void s_where_impl(
    const unsigned char *cond, const int64_t *stridesCond,
    const T *x, const int64_t *stridesX,
    const T *y, const int64_t *stridesY,
    T *res, const int64_t *stridesRes,
    const int64_t *shape, int rank
) {
    if (cond == nullptr || x == nullptr || y == nullptr || res == nullptr || rank < 0) return;
    if (rank == 0) {
        res[0] = cond[0] ? x[0] : y[0];
        return;
    }
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) {
        if (shape[i] <= 0) return;
        total_elements *= shape[i];
    }
    bool overlap =
        strided_buffers_overlap_not_identical(cond, stridesCond, sizeof(unsigned char), res, stridesRes, sizeof(T), shape, rank) ||
        strided_buffers_overlap_not_identical(x, stridesX, sizeof(T), res, stridesRes, sizeof(T), shape, rank) ||
        strided_buffers_overlap_not_identical(y, stridesY, sizeof(T), res, stridesRes, sizeof(T), shape, rank);
    if (overlap) {
        NoThrowBuffer<T> temp(total_elements);
        if (!temp.ok()) {
            ndarray_set_oom_flag();
            return;
        }
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetCond = 0, offsetX = 0, offsetY = 0;
        for (int64_t el = 0; el < total_elements; el++) {
            temp[el] = cond[offsetCond] ? x[offsetX] : y[offsetY];
            for (int d = rank - 1; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetCond += stridesCond[d];
                    offsetX    += stridesX[d];
                    offsetY    += stridesY[d];
                    break;
                }
                coord[d] = 0;
                offsetCond -= (shape[d] - 1) * stridesCond[d];
                offsetX    -= (shape[d] - 1) * stridesX[d];
                offsetY    -= (shape[d] - 1) * stridesY[d];
            }
        }
        DECLARE_RANK_BUFFER(int64_t, coord_out, rank);
        int64_t offsetRes = 0;
        for (int64_t el = 0; el < total_elements; el++) {
            res[offsetRes] = temp[el];
            for (int d = rank - 1; d >= 0; d--) {
                coord_out[d]++;
                if (coord_out[d] < shape[d]) {
                    offsetRes += stridesRes[d];
                    break;
                }
                coord_out[d] = 0;
                offsetRes -= (shape[d] - 1) * stridesRes[d];
            }
        }
        return;
    }
    int is_contiguous = 1;
    int64_t expected_stride = 1;
    for (int64_t i = rank - 1; i >= 0; i--) {
        if (stridesCond[i] != expected_stride ||
            stridesX[i] != expected_stride ||
            stridesY[i] != expected_stride ||
            stridesRes[i] != expected_stride) {
            is_contiguous = 0;
            break;
        }
        expected_stride *= shape[i];
    }
    if (is_contiguous) {
        for (int64_t i = 0; i < total_elements; i++) {
            res[i] = cond[i] ? x[i] : y[i];
        }
        return;
    }
    if (rank == 1) {
        const int64_t sCond = stridesCond[0];
        const int64_t sX = stridesX[0];
        const int64_t sY = stridesY[0];
        const int64_t sRes = stridesRes[0];
        if (sCond == 1 && sRes == 1) {
            if (sX == 1 && sY == 1) {
                for (int64_t i = 0; i < total_elements; i++) res[i] = cond[i] ? x[i] : y[i];
                return;
            }
            if (sX == 1 && sY == 0) {
                const T y0 = y[0];
                for (int64_t i = 0; i < total_elements; i++) res[i] = cond[i] ? x[i] : y0;
                return;
            }
            if (sX == 0 && sY == 1) {
                const T x0 = x[0];
                for (int64_t i = 0; i < total_elements; i++) res[i] = cond[i] ? x0 : y[i];
                return;
            }
            if (sX == 0 && sY == 0) {
                const T x0 = x[0], y0 = y[0];
                for (int64_t i = 0; i < total_elements; i++) res[i] = cond[i] ? x0 : y0;
                return;
            }
            for (int64_t i = 0; i < total_elements; i++) res[i] = cond[i] ? x[i * sX] : y[i * sY];
            return;
        }
        for (int64_t i = 0; i < total_elements; i++) {
            res[i * sRes] = cond[i * sCond] ? x[i * sX] : y[i * sY];
        }
        return;
    }

    if (rank >= 2) {
        const int64_t inner_n = shape[rank - 1];
        const int64_t sCond = stridesCond[rank - 1];
        const int64_t sX = stridesX[rank - 1];
        const int64_t sY = stridesY[rank - 1];
        const int64_t sRes = stridesRes[rank - 1];
        int64_t outer_elements = 1;
        for (int d = 0; d < rank - 1; d++) outer_elements *= shape[d];

        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetCond = 0, offsetX = 0, offsetY = 0, offsetRes = 0;

        auto step_outer = [&]() {
            for (int d = rank - 2; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetCond += stridesCond[d];
                    offsetX    += stridesX[d];
                    offsetY    += stridesY[d];
                    offsetRes  += stridesRes[d];
                    break;
                }
                coord[d] = 0;
                offsetCond -= (shape[d] - 1) * stridesCond[d];
                offsetX    -= (shape[d] - 1) * stridesX[d];
                offsetY    -= (shape[d] - 1) * stridesY[d];
                offsetRes  -= (shape[d] - 1) * stridesRes[d];
            }
        };

        if (sCond == 1 && sRes == 1) {
            if (sX == 1 && sY == 1) {
                for (int64_t out_idx = 0; out_idx < outer_elements; out_idx++) {
                    const unsigned char *rowCond = cond + offsetCond;
                    const T *rowX = x + offsetX;
                    const T *rowY = y + offsetY;
                    T *rowRes = res + offsetRes;
                    for (int64_t j = 0; j < inner_n; j++) {
                        rowRes[j] = rowCond[j] ? rowX[j] : rowY[j];
                    }
                    step_outer();
                }
                return;
            }
            if (sX == 1 && sY == 0) {
                for (int64_t out_idx = 0; out_idx < outer_elements; out_idx++) {
                    const unsigned char *rowCond = cond + offsetCond;
                    const T *rowX = x + offsetX;
                    const T y0 = y[offsetY];
                    T *rowRes = res + offsetRes;
                    for (int64_t j = 0; j < inner_n; j++) {
                        rowRes[j] = rowCond[j] ? rowX[j] : y0;
                    }
                    step_outer();
                }
                return;
            }
            if (sX == 0 && sY == 1) {
                for (int64_t out_idx = 0; out_idx < outer_elements; out_idx++) {
                    const unsigned char *rowCond = cond + offsetCond;
                    const T x0 = x[offsetX];
                    const T *rowY = y + offsetY;
                    T *rowRes = res + offsetRes;
                    for (int64_t j = 0; j < inner_n; j++) {
                        rowRes[j] = rowCond[j] ? x0 : rowY[j];
                    }
                    step_outer();
                }
                return;
            }
            if (sX == 0 && sY == 0) {
                for (int64_t out_idx = 0; out_idx < outer_elements; out_idx++) {
                    const unsigned char *rowCond = cond + offsetCond;
                    const T x0 = x[offsetX];
                    const T y0 = y[offsetY];
                    T *rowRes = res + offsetRes;
                    for (int64_t j = 0; j < inner_n; j++) {
                        rowRes[j] = rowCond[j] ? x0 : y0;
                    }
                    step_outer();
                }
                return;
            }
            for (int64_t out_idx = 0; out_idx < outer_elements; out_idx++) {
                const unsigned char *rowCond = cond + offsetCond;
                const T *rowX = x + offsetX;
                const T *rowY = y + offsetY;
                T *rowRes = res + offsetRes;
                for (int64_t j = 0; j < inner_n; j++) {
                    rowRes[j] = rowCond[j] ? rowX[j * sX] : rowY[j * sY];
                }
                step_outer();
            }
            return;
        }

        for (int64_t out_idx = 0; out_idx < outer_elements; out_idx++) {
            const unsigned char *rowCond = cond + offsetCond;
            const T *rowX = x + offsetX;
            const T *rowY = y + offsetY;
            T *rowRes = res + offsetRes;
            for (int64_t j = 0; j < inner_n; j++) {
                rowRes[j * sRes] = rowCond[j * sCond] ? rowX[j * sX] : rowY[j * sY];
            }
            step_outer();
        }
        return;
    }

    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetCond = 0, offsetX = 0, offsetY = 0, offsetRes = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        res[offsetRes] = cond[offsetCond] ? x[offsetX] : y[offsetY];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetCond += stridesCond[d];
                offsetX    += stridesX[d];
                offsetY    += stridesY[d];
                offsetRes  += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetCond -= (shape[d] - 1) * stridesCond[d];
            offsetX    -= (shape[d] - 1) * stridesX[d];
            offsetY    -= (shape[d] - 1) * stridesY[d];
            offsetRes  -= (shape[d] - 1) * stridesRes[d];
        }
    }
}




// Macro to define strided binary operations
#define DEFINE_STRIDED_BINARY_OP(name, typeA, typeB, typeResult, op) \
void name(const typeA *a, const int64_t *stridesA, \
          const typeB *b, const int64_t *stridesB, \
          typeResult *res, const int64_t *stridesRes, \
          const int64_t *shape, int rank, \
          const uint8_t *mask) { \
    strided_binary_op_impl(a, stridesA, b, stridesB, res, stridesRes, shape, rank, mask, [](typeA x, typeB y) { return op(x, y); }); \
}

#include <complex>

static inline cpx_t cpx_add_cast(cpx_t x, cpx_t y) {
    std::complex<double> cx(x.r, x.i);
    std::complex<double> cy(y.r, y.i);
    std::complex<double> cres = cx + cy;
    return {cres.real(), cres.imag()};
}

static inline cpx_t cpx_sub_cast(cpx_t x, cpx_t y) {
    std::complex<double> cx(x.r, x.i);
    std::complex<double> cy(y.r, y.i);
    std::complex<double> cres = cx - cy;
    return {cres.real(), cres.imag()};
}

static inline cpx_t cpx_mul_cast(cpx_t x, cpx_t y) {
    std::complex<double> cx(x.r, x.i);
    std::complex<double> cy(y.r, y.i);
    std::complex<double> cres = cx * cy;
    return {cres.real(), cres.imag()};
}

static inline cpx_t cpx_div_cast(cpx_t x, cpx_t y) {
    std::complex<double> cx(x.r, x.i);
    std::complex<double> cy(y.r, y.i);
    std::complex<double> cres = cx / cy;
    return {cres.real(), cres.imag()};
}

static inline cpx_f_t cpx_add_f_cast(cpx_f_t x, cpx_f_t y) {
    std::complex<float> cx(x.r, x.i);
    std::complex<float> cy(y.r, y.i);
    std::complex<float> cres = cx + cy;
    return {cres.real(), cres.imag()};
}

static inline cpx_f_t cpx_sub_f_cast(cpx_f_t x, cpx_f_t y) {
    std::complex<float> cx(x.r, x.i);
    std::complex<float> cy(y.r, y.i);
    std::complex<float> cres = cx - cy;
    return {cres.real(), cres.imag()};
}

static inline cpx_f_t cpx_mul_f_cast(cpx_f_t x, cpx_f_t y) {
    std::complex<float> cx(x.r, x.i);
    std::complex<float> cy(y.r, y.i);
    std::complex<float> cres = cx * cy;
    return {cres.real(), cres.imag()};
}

static inline cpx_f_t cpx_div_f_cast(cpx_f_t x, cpx_f_t y) {
    std::complex<float> cx(x.r, x.i);
    std::complex<float> cy(y.r, y.i);
    std::complex<float> cres = cx / cy;
    return {cres.real(), cres.imag()};
}

static inline cpx_t cpx_add(cpx_t a, cpx_t b) {
    return (cpx_t){a.r + b.r, a.i + b.i};
}
static inline cpx_t cpx_sub(cpx_t a, cpx_t b) {
    return (cpx_t){a.r - b.r, a.i - b.i};
}
static inline cpx_t cpx_mul(cpx_t a, cpx_t b) {
    return (cpx_t){a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}
static inline cpx_t cpx_div(cpx_t x, cpx_t y) {
    std::complex<double> cx(x.r, x.i);
    std::complex<double> cy(y.r, y.i);
    std::complex<double> cres = cx / cy;
    return {cres.real(), cres.imag()};
}
static inline cpx_f_t cpx_add_f(cpx_f_t a, cpx_f_t b) {
    return (cpx_f_t){a.r + b.r, a.i + b.i};
}
static inline cpx_f_t cpx_sub_f(cpx_f_t a, cpx_f_t b) {
    return (cpx_f_t){a.r - b.r, a.i - b.i};
}
static inline cpx_f_t cpx_mul_f(cpx_f_t a, cpx_f_t b) {
    return (cpx_f_t){a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}
static inline cpx_f_t cpx_div_f(cpx_f_t x, cpx_f_t y) {
    std::complex<float> cx(x.r, x.i);
    std::complex<float> cy(y.r, y.i);
    std::complex<float> cres = cx / cy;
    return {cres.real(), cres.imag()};
}

// 1. DOUBLE PRECISION (FLOAT64) FLAT CONTIGUOUS KERNELS
// ============================================================================

#define IMPLEMENT_V_BINARY(name, type, op) \
VECTORIZED_TARGETS \
void v_##name##_##type(const type *a, const type *b, type *res, int64_t size, const uint8_t *mask) { \
    v_binary_impl(a, b, res, size, mask, [](type x, type y) { return x op y; }); \
}

#define IMPLEMENT_V_UNARY(name, type, func) \
VECTORIZED_TARGETS \
void v_##name##_##type(const type *src, type *res, int64_t size, const uint8_t *mask) { \
    v_unary_impl(src, res, size, mask, [](type x) { return func(x); }); \
}

#define IMPLEMENT_V_BINARY_FUNC(name, type, func) \
VECTORIZED_TARGETS \
void v_##name##_##type(const type *a, const type *b, type *res, int64_t size, const uint8_t *mask) { \
    v_binary_impl(a, b, res, size, mask, [](type x, type y) { return func(x, y); }); \
}

IMPLEMENT_V_BINARY(add, double, +)
IMPLEMENT_V_BINARY(sub, double, -)
IMPLEMENT_V_BINARY(mul, double, *)
IMPLEMENT_V_BINARY(div, double, /)

IMPLEMENT_V_UNARY(sin, double, sin)
IMPLEMENT_V_UNARY(sinc, double, real_sinc)
IMPLEMENT_V_UNARY(cos, double, cos)
IMPLEMENT_V_UNARY(exp, double, exp)
IMPLEMENT_V_UNARY(log, double, log)
IMPLEMENT_V_UNARY(sinh, double, sinh)
IMPLEMENT_V_UNARY(cosh, double, cosh)
IMPLEMENT_V_UNARY(tanh, double, tanh)
IMPLEMENT_V_UNARY(asinh, double, asinh)
IMPLEMENT_V_UNARY(acosh, double, acosh)
IMPLEMENT_V_UNARY(atanh, double, atanh)
IMPLEMENT_V_UNARY(asin, double, asin)
IMPLEMENT_V_UNARY(acos, double, acos)
IMPLEMENT_V_UNARY(atan, double, atan)

IMPLEMENT_V_BINARY_FUNC(atan2, double, atan2)

IMPLEMENT_V_UNARY(expm1, double, expm1)
IMPLEMENT_V_UNARY(expm1, float, expm1f)
IMPLEMENT_V_UNARY(log1p, double, log1p)
IMPLEMENT_V_UNARY(log1p, float, log1pf)
IMPLEMENT_V_UNARY(rint, double, rint)
IMPLEMENT_V_UNARY(rint, float, rintf)
IMPLEMENT_V_UNARY(trunc, double, trunc)
IMPLEMENT_V_UNARY(trunc, float, truncf)

IMPLEMENT_V_BINARY_FUNC(logaddexp, double, logaddexp_op<double>)
IMPLEMENT_V_BINARY_FUNC(logaddexp, float, logaddexp_op<float>)
IMPLEMENT_V_BINARY_FUNC(logaddexp2, double, logaddexp2_op<double>)
IMPLEMENT_V_BINARY_FUNC(logaddexp2, float, logaddexp2_op<float>)


// ============================================================================
// OPTIMIZED REDUCTION KERNELS (AVX2 SIMD & UNROLLED ACCUMULATION)
// ============================================================================

#if defined(__AVX2__)
#define HAS_AVX2_REDUCTIONS 1
#else
#define HAS_AVX2_REDUCTIONS 0
#endif

#if HAS_AVX2_REDUCTIONS
static inline double hsum_avx2_pd(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    __m128d v128 = _mm_add_pd(vlow, vhigh);
    __m128d s = _mm_hadd_pd(v128, v128);
    return _mm_cvtsd_f64(s);
}

static inline float hsum_avx2_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 v128 = _mm_add_ps(vlow, vhigh);
    v128 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
    v128 = _mm_add_ss(v128, _mm_shuffle_ps(v128, v128, 1));
    return _mm_cvtss_f32(v128);
}
#endif

// ----------------------------------------------------------------------------
// 1D Contiguous Reductions
// ----------------------------------------------------------------------------

double r_sum_double(const double *src, int64_t size) {
    if (src == nullptr || size <= 0) return 0.0;
    int64_t i = 0;
#if HAS_AVX2_REDUCTIONS
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();

    for (; i + 15 < size; i += 16) {
        acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(src + i));
        acc1 = _mm256_add_pd(acc1, _mm256_loadu_pd(src + i + 4));
        acc2 = _mm256_add_pd(acc2, _mm256_loadu_pd(src + i + 8));
        acc3 = _mm256_add_pd(acc3, _mm256_loadu_pd(src + i + 12));
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    acc0 = _mm256_add_pd(acc0, acc2);

    for (; i + 3 < size; i += 4) {
        acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(src + i));
    }
    double total = hsum_avx2_pd(acc0);
#else
    double total = 0.0;
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    for (; i + 3 < size; i += 4) {
        a0 += src[i];
        a1 += src[i + 1];
        a2 += src[i + 2];
        a3 += src[i + 3];
    }
    total = (a0 + a1) + (a2 + a3);
#endif
    for (; i < size; i++) {
        total += src[i];
    }
    return total;
}

double r_prod_double(const double *src, int64_t size) {
    if (src == nullptr || size <= 0) return 1.0;
    double acc0 = 1.0, acc1 = 1.0, acc2 = 1.0, acc3 = 1.0;
    int64_t i = 0;
    for (; i + 3 < size; i += 4) {
        acc0 *= src[i];
        acc1 *= src[i + 1];
        acc2 *= src[i + 2];
        acc3 *= src[i + 3];
    }
    double acc = (acc0 * acc1) * (acc2 * acc3);
    for (; i < size; i++) {
        acc *= src[i];
    }
    return acc;
}

double r_mean_double(const double *src, int64_t size) {
    if (src == nullptr || size <= 0) return NAN;
    return r_sum_double(src, size) / (double)size;
}

double r_var_double(const double *src, int64_t size, int ddof) {
    if (src == nullptr || size <= 0) return NAN;
    double m = r_sum_double(src, size) / (double)size;
    int64_t i = 0;
#if HAS_AVX2_REDUCTIONS
    __m256d vm = _mm256_set1_pd(m);
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();

    for (; i + 15 < size; i += 16) {
        __m256d d0 = _mm256_sub_pd(_mm256_loadu_pd(src + i), vm);
        __m256d d1 = _mm256_sub_pd(_mm256_loadu_pd(src + i + 4), vm);
        __m256d d2 = _mm256_sub_pd(_mm256_loadu_pd(src + i + 8), vm);
        __m256d d3 = _mm256_sub_pd(_mm256_loadu_pd(src + i + 12), vm);

        acc0 = _mm256_fmadd_pd(d0, d0, acc0);
        acc1 = _mm256_fmadd_pd(d1, d1, acc1);
        acc2 = _mm256_fmadd_pd(d2, d2, acc2);
        acc3 = _mm256_fmadd_pd(d3, d3, acc3);
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    acc0 = _mm256_add_pd(acc0, acc2);

    for (; i + 3 < size; i += 4) {
        __m256d d0 = _mm256_sub_pd(_mm256_loadu_pd(src + i), vm);
        acc0 = _mm256_fmadd_pd(d0, d0, acc0);
    }
    double sum_sq = hsum_avx2_pd(acc0);
#else
    double sum_sq = 0.0;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (; i + 3 < size; i += 4) {
        double d0 = src[i] - m;
        double d1 = src[i + 1] - m;
        double d2 = src[i + 2] - m;
        double d3 = src[i + 3] - m;
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    sum_sq = (s0 + s1) + (s2 + s3);
#endif
    for (; i < size; i++) {
        double d = src[i] - m;
        sum_sq += d * d;
    }
    if (size <= ddof) {
        return (sum_sq == 0.0 || std::isnan(sum_sq)) ? NAN : INFINITY;
    }
    return sum_sq / (double)(size - ddof);
}

double r_std_double(const double *src, int64_t size, int ddof) {
    double v = r_var_double(src, size, ddof);
    return std::sqrt(v);
}

float r_sum_float(const float *src, int64_t size) {
    if (src == nullptr || size <= 0) return 0.0f;
    int64_t i = 0;
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    for (; i + 3 < size; i += 4) {
        a0 += (double)src[i];
        a1 += (double)src[i + 1];
        a2 += (double)src[i + 2];
        a3 += (double)src[i + 3];
    }
    double total = (a0 + a1) + (a2 + a3);
    for (; i < size; i++) {
        total += (double)src[i];
    }
    return (float)total;
}

float r_prod_float(const float *src, int64_t size) {
    if (src == nullptr || size <= 0) return 1.0f;
    float acc0 = 1.0f, acc1 = 1.0f, acc2 = 1.0f, acc3 = 1.0f;
    int64_t i = 0;
    for (; i + 3 < size; i += 4) {
        acc0 *= src[i];
        acc1 *= src[i + 1];
        acc2 *= src[i + 2];
        acc3 *= src[i + 3];
    }
    float acc = (acc0 * acc1) * (acc2 * acc3);
    for (; i < size; i++) {
        acc *= src[i];
    }
    return acc;
}

float r_mean_float(const float *src, int64_t size) {
    if (src == nullptr || size <= 0) return NAN;
    return r_sum_float(src, size) / (float)size;
}

double r_mean_float_to_double(const float *src, int64_t size) {
    if (src == nullptr || size <= 0) return NAN;
    return (double)r_sum_float(src, size) / (double)size;
}

float r_var_float(const float *src, int64_t size, int ddof) {
    if (src == nullptr || size <= 0) return NAN;
    float m = r_sum_float(src, size) / (float)size;
    int64_t i = 0;
#if HAS_AVX2_REDUCTIONS
    __m256 vm = _mm256_set1_ps(m);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (; i + 31 < size; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(src + i), vm);
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(src + i + 8), vm);
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(src + i + 16), vm);
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(src + i + 24), vm);

        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    for (; i + 7 < size; i += 8) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(src + i), vm);
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    }
    float sum_sq = hsum_avx2_ps(acc0);
#else
    float sum_sq = 0.0f;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (; i + 3 < size; i += 4) {
        float d0 = src[i] - m;
        float d1 = src[i + 1] - m;
        float d2 = src[i + 2] - m;
        float d3 = src[i + 3] - m;
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    sum_sq = (s0 + s1) + (s2 + s3);
#endif
    for (; i < size; i++) {
        float d = src[i] - m;
        sum_sq += d * d;
    }
    if (size <= ddof) {
        return (sum_sq == 0.0f || std::isnan(sum_sq)) ? NAN : INFINITY;
    }
    return sum_sq / (float)(size - ddof);
}

double r_var_float_to_double(const float *src, int64_t size, int ddof) {
    if (src == nullptr || size <= 0) return NAN;
    double m = (double)r_sum_float(src, size) / (double)size;
    double sum_sq = 0.0;
    for (int64_t i = 0; i < size; i++) {
        double d = (double)src[i] - m;
        sum_sq += d * d;
    }
    if (size <= ddof) {
        return (sum_sq == 0.0 || std::isnan(sum_sq)) ? NAN : INFINITY;
    }
    return sum_sq / (double)(size - ddof);
}

float r_std_float(const float *src, int64_t size, int ddof) {
    float v = r_var_float(src, size, ddof);
    return std::sqrt(v);
}

double r_std_float_to_double(const float *src, int64_t size, int ddof) {
    double v = r_var_float_to_double(src, size, ddof);
    return std::sqrt(v);
}

template <typename T>
static inline double r_mean_int_impl(const T *src, int64_t size) {
    if (src == nullptr || size <= 0) return NAN;
    double sum = 0.0;
    for (int64_t i = 0; i < size; i++) sum += (double)src[i];
    return sum / (double)size;
}

template <typename T>
static inline double r_var_int_impl(const T *src, int64_t size, int ddof) {
    if (src == nullptr || size <= 0) return NAN;
    double m = r_mean_int_impl(src, size);
    double sum_sq = 0.0;
    for (int64_t i = 0; i < size; i++) {
        double d = (double)src[i] - m;
        sum_sq += d * d;
    }
    if (size <= ddof) {
        return (sum_sq == 0.0 || std::isnan(sum_sq)) ? NAN : INFINITY;
    }
    return sum_sq / (double)(size - ddof);
}

double r_mean_int64_to_double(const int64_t *src, int64_t size) { return r_mean_int_impl(src, size); }
double r_mean_int32_to_double(const int32_t *src, int64_t size) { return r_mean_int_impl(src, size); }
double r_mean_uint8_to_double(const uint8_t *src, int64_t size) { return r_mean_int_impl(src, size); }
double r_mean_int16_to_double(const int16_t *src, int64_t size) { return r_mean_int_impl(src, size); }

double r_var_int64_to_double(const int64_t *src, int64_t size, int ddof) { return r_var_int_impl(src, size, ddof); }
double r_var_int32_to_double(const int32_t *src, int64_t size, int ddof) { return r_var_int_impl(src, size, ddof); }
double r_var_uint8_to_double(const uint8_t *src, int64_t size, int ddof) { return r_var_int_impl(src, size, ddof); }
double r_var_int16_to_double(const int16_t *src, int64_t size, int ddof) { return r_var_int_impl(src, size, ddof); }

double r_std_int64_to_double(const int64_t *src, int64_t size, int ddof) { return std::sqrt(r_var_int_impl(src, size, ddof)); }
double r_std_int32_to_double(const int32_t *src, int64_t size, int ddof) { return std::sqrt(r_var_int_impl(src, size, ddof)); }
double r_std_uint8_to_double(const uint8_t *src, int64_t size, int ddof) { return std::sqrt(r_var_int_impl(src, size, ddof)); }
double r_std_int16_to_double(const int16_t *src, int64_t size, int ddof) { return std::sqrt(r_var_int_impl(src, size, ddof)); }

typedef struct { double r; double i; } cpx_add_t;
static inline cpx_t cpx_add_red(cpx_t a, cpx_t b) { return cpx_t{a.r + b.r, a.i + b.i}; }
static inline cpx_f_t cpx_add_f_red(cpx_f_t a, cpx_f_t b) { return cpx_f_t{a.r + b.r, a.i + b.i}; }

cpx_t r_sum_complex128(const cpx_t *src, int64_t size) {
    if (src == nullptr || size <= 0) return cpx_t{0.0, 0.0};
    double re = 0.0, im = 0.0;
    for (int64_t i = 0; i < size; i++) {
        re += src[i].r;
        im += src[i].i;
    }
    return cpx_t{re, im};
}

cpx_f_t r_sum_complex64(const cpx_f_t *src, int64_t size) {
    if (src == nullptr || size <= 0) return cpx_f_t{0.0f, 0.0f};
    float re = 0.0f, im = 0.0f;
    for (int64_t i = 0; i < size; i++) {
        re += src[i].r;
        im += src[i].i;
    }
    return cpx_f_t{re, im};
}

cpx_t r_mean_complex128(const cpx_t *src, int64_t size) {
    if (src == nullptr || size <= 0) return cpx_t{NAN, NAN};
    cpx_t s = r_sum_complex128(src, size);
    return cpx_t{s.r / (double)size, s.i / (double)size};
}

cpx_f_t r_mean_complex64(const cpx_f_t *src, int64_t size) {
    if (src == nullptr || size <= 0) return cpx_f_t{NAN, NAN};
    cpx_f_t s = r_sum_complex64(src, size);
    return cpx_f_t{s.r / (float)size, s.i / (float)size};
}

cpx_t r_mean_complex64_to_complex128(const cpx_f_t *src, int64_t size) {
    if (src == nullptr || size <= 0) return cpx_t{NAN, NAN};
    cpx_f_t s = r_sum_complex64(src, size);
    return cpx_t{(double)s.r / (double)size, (double)s.i / (double)size};
}

// ----------------------------------------------------------------------------
// Multi-D & Axis Reductions
// ----------------------------------------------------------------------------

template <typename T>
static inline void fill_reduction_dest(T *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, T val) {
    if (dest == nullptr || shape == nullptr || (rank > 1 && stridesDest == nullptr)) return;
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) {
            if (shape[d] <= 0) return;
            outer_size *= shape[d];
        }
    }
    if (outer_size <= 0) return;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetDest = 0;
    for (int64_t o = 0; o < outer_size; o++) {
        dest[offsetDest] = val;
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) {
                offsetDest += stridesDest[targetD];
                break;
            }
            coord[d] = 0;
            offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

static inline bool is_reduction_dest_contiguous(const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    if (rank <= 1) return true;
    if (stridesDest == nullptr) return false;
    int64_t expected_stride = 1;
    for (int d = rank - 1; d >= 0; d--) {
        if (d == axis) continue;
        int td = (d < axis) ? d : (d - 1);
        if (shape[d] > 1 && stridesDest[td] != expected_stride) {
            return false;
        }
        expected_stride *= shape[d];
    }
    return true;
}

void s_sum_double(const double *src, const int64_t *stridesSrc,
                  double *dest, const int64_t *stridesDest,
                  const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        fill_reduction_dest<double>(dest, stridesDest, shape, rank, axis, 0.0);
        return;
    }

    if (rank == 2 && axis == 0 && stridesSrc[1] == 1 && stridesDest[0] == 1) {
        int64_t M = shape[0];
        int64_t N = shape[1];
        int64_t S = stridesSrc[0];
        memcpy(dest, src, N * sizeof(double));
        for (int64_t r = 1; r < M; r++) {
            const double *row = src + r * S;
            int64_t c = 0;
#if HAS_AVX2_REDUCTIONS
            for (; c + 15 < N; c += 16) {
                _mm256_storeu_pd(dest + c, _mm256_add_pd(_mm256_loadu_pd(dest + c), _mm256_loadu_pd(row + c)));
                _mm256_storeu_pd(dest + c + 4, _mm256_add_pd(_mm256_loadu_pd(dest + c + 4), _mm256_loadu_pd(row + c + 4)));
                _mm256_storeu_pd(dest + c + 8, _mm256_add_pd(_mm256_loadu_pd(dest + c + 8), _mm256_loadu_pd(row + c + 8)));
                _mm256_storeu_pd(dest + c + 12, _mm256_add_pd(_mm256_loadu_pd(dest + c + 12), _mm256_loadu_pd(row + c + 12)));
            }
            for (; c + 3 < N; c += 4) {
                _mm256_storeu_pd(dest + c, _mm256_add_pd(_mm256_loadu_pd(dest + c), _mm256_loadu_pd(row + c)));
            }
#endif
            for (; c < N; c++) {
                dest[c] += row[c];
            }
        }
        return;
    }

    if (axis == rank - 1 && stridesSrc[rank - 1] == 1) {
        int64_t L = shape[rank - 1];
        int64_t outer_size = 1;
        for (int d = 0; d < rank - 1; d++) outer_size *= shape[d];
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetSrc = 0;
        int64_t offsetDest = 0;
        for (int64_t o = 0; o < outer_size; o++) {
            dest[offsetDest] = r_sum_double(src + offsetSrc, L);
            for (int d = rank - 2; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetSrc += stridesSrc[d];
                    offsetDest += stridesDest[d];
                    break;
                }
                coord[d] = 0;
                offsetSrc -= (shape[d] - 1) * stridesSrc[d];
                offsetDest -= (shape[d] - 1) * stridesDest[d];
            }
        }
        return;
    }

    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    int64_t offsetDest = 0;
    int64_t stride_axis = stridesSrc[axis];

    for (int64_t o = 0; o < outer_size; o++) {
        const double *axis_ptr = src + offsetSrc;
        double sum = 0.0;
        for (int64_t i = 0; i < size_axis; i++) {
            sum += axis_ptr[i * stride_axis];
        }
        dest[offsetDest] = sum;

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDest += stridesDest[targetD];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

void s_mean_double(const double *src, const int64_t *stridesSrc,
                   double *dest, const int64_t *stridesDest,
                   const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        fill_reduction_dest<double>(dest, stridesDest, shape, rank, axis, NAN);
        return;
    }

    if (rank == 2 && axis == 0 && stridesSrc[1] == 1 && stridesDest[0] == 1) {
        int64_t M = shape[0];
        int64_t N = shape[1];
        s_sum_double(src, stridesSrc, dest, stridesDest, shape, rank, 0);
        double scale = 1.0 / (double)M;
        int64_t c = 0;
#if HAS_AVX2_REDUCTIONS
        __m256d vscale = _mm256_set1_pd(scale);
        for (; c + 15 < N; c += 16) {
            _mm256_storeu_pd(dest + c, _mm256_mul_pd(_mm256_loadu_pd(dest + c), vscale));
            _mm256_storeu_pd(dest + c + 4, _mm256_mul_pd(_mm256_loadu_pd(dest + c + 4), vscale));
            _mm256_storeu_pd(dest + c + 8, _mm256_mul_pd(_mm256_loadu_pd(dest + c + 8), vscale));
            _mm256_storeu_pd(dest + c + 12, _mm256_mul_pd(_mm256_loadu_pd(dest + c + 12), vscale));
        }
        for (; c + 3 < N; c += 4) {
            _mm256_storeu_pd(dest + c, _mm256_mul_pd(_mm256_loadu_pd(dest + c), vscale));
        }
#endif
        for (; c < N; c++) {
            dest[c] *= scale;
        }
        return;
    }

    if (axis == rank - 1 && stridesSrc[rank - 1] == 1) {
        int64_t L = shape[rank - 1];
        int64_t outer_size = 1;
        for (int d = 0; d < rank - 1; d++) outer_size *= shape[d];
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetSrc = 0;
        int64_t offsetDest = 0;
        for (int64_t o = 0; o < outer_size; o++) {
            dest[offsetDest] = r_mean_double(src + offsetSrc, L);
            for (int d = rank - 2; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetSrc += stridesSrc[d];
                    offsetDest += stridesDest[d];
                    break;
                }
                coord[d] = 0;
                offsetSrc -= (shape[d] - 1) * stridesSrc[d];
                offsetDest -= (shape[d] - 1) * stridesDest[d];
            }
        }
        return;
    }

    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    int64_t offsetDest = 0;
    int64_t stride_axis = stridesSrc[axis];
    double inv_size = 1.0 / (double)size_axis;

    for (int64_t o = 0; o < outer_size; o++) {
        const double *axis_ptr = src + offsetSrc;
        double sum = 0.0;
        for (int64_t i = 0; i < size_axis; i++) {
            sum += axis_ptr[i * stride_axis];
        }
        dest[offsetDest] = sum * inv_size;

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDest += stridesDest[targetD];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

void s_var_double(const double *src, const int64_t *stridesSrc,
                  double *dest, const int64_t *stridesDest,
                  const int64_t *shape, int rank, int axis, int ddof) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        int64_t outer_size = 1;
        for (int d = 0; d < rank; d++) if (d != axis) outer_size *= shape[d];
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetDest = 0;
        for (int64_t o = 0; o < outer_size; o++) {
            dest[offsetDest] = NAN;
            for (int d = rank - 1; d >= 0; d--) {
                if (d == axis) continue;
                coord[d]++;
                int targetD = (d < axis) ? d : (d - 1);
                if (coord[d] < shape[d]) {
                    offsetDest += stridesDest[targetD];
                    break;
                }
                coord[d] = 0;
                offsetDest -= (shape[d] - 1) * stridesDest[targetD];
            }
        }
        return;
    }

    int res_rank = rank - 1;
    DECLARE_RANK_BUFFER(int64_t, res_shape, std::max(1, res_rank));
    int64_t outer_total = 1;
    for (int d = 0, rd = 0; d < rank; d++) {
        if (d != axis) {
            if (shape[d] <= 0) return;
            res_shape[rd++] = shape[d];
            outer_total *= shape[d];
        }
    }
    if (outer_total <= 0) return;
    if (strided_buffers_any_overlap(src, stridesSrc, shape, rank, sizeof(double), dest, stridesDest, res_shape, res_rank, sizeof(double))) {
        DECLARE_RANK_BUFFER(int64_t, temp_strides, std::max(1, res_rank));
        int64_t s = 1;
        for (int d = res_rank - 1; d >= 0; d--) {
            temp_strides[d] = s;
            s *= res_shape[d];
        }
        NoThrowBuffer<double> temp(outer_total);
        if (!temp.ok()) {
            ndarray_set_oom_flag();
            return;
        }
        s_var_double(src, stridesSrc, temp.data(), temp_strides, shape, rank, axis, ddof);
        DECLARE_RANK_BUFFER(int64_t, coord_out, std::max(1, res_rank));
        for (int64_t i = 0; i < outer_total; i++) {
            int64_t offsetDest = 0;
            for (int d = 0; d < res_rank; d++) {
                offsetDest += coord_out[d] * stridesDest[d];
            }
            dest[offsetDest] = temp[i];
            for (int d = res_rank - 1; d >= 0; d--) {
                coord_out[d]++;
                if (coord_out[d] < res_shape[d]) break;
                coord_out[d] = 0;
            }
        }
        return;
    }

    if (rank == 2 && axis == 0 && stridesSrc[1] == 1 && stridesDest[0] == 1) {
        int64_t M = shape[0];
        int64_t N = shape[1];
        int64_t S = stridesSrc[0];
        NoThrowBuffer<double> mean_buf_vec;
        double stack_buf[1024];
        double *mean_buf = stack_buf;
        if (N > 1024) {
            mean_buf_vec.resize(N);
            if (!mean_buf_vec.ok()) {
                ndarray_set_oom_flag();
                return;
            }
            mean_buf = mean_buf_vec.data();
        }

        s_mean_double(src, stridesSrc, mean_buf, stridesDest, shape, rank, 0);

        memset(dest, 0, N * sizeof(double));
        for (int64_t r = 0; r < M; r++) {
            const double *row = src + r * S;
            int64_t c = 0;
#if HAS_AVX2_REDUCTIONS
            for (; c + 15 < N; c += 16) {
                __m256d vm0 = _mm256_loadu_pd(mean_buf + c);
                __m256d vs0 = _mm256_loadu_pd(row + c);
                __m256d vd0 = _mm256_loadu_pd(dest + c);
                __m256d df0 = _mm256_sub_pd(vs0, vm0);
                vd0 = _mm256_fmadd_pd(df0, df0, vd0);
                _mm256_storeu_pd(dest + c, vd0);

                __m256d vm1 = _mm256_loadu_pd(mean_buf + c + 4);
                __m256d vs1 = _mm256_loadu_pd(row + c + 4);
                __m256d vd1 = _mm256_loadu_pd(dest + c + 4);
                __m256d df1 = _mm256_sub_pd(vs1, vm1);
                vd1 = _mm256_fmadd_pd(df1, df1, vd1);
                _mm256_storeu_pd(dest + c + 4, vd1);

                __m256d vm2 = _mm256_loadu_pd(mean_buf + c + 8);
                __m256d vs2 = _mm256_loadu_pd(row + c + 8);
                __m256d vd2 = _mm256_loadu_pd(dest + c + 8);
                __m256d df2 = _mm256_sub_pd(vs2, vm2);
                vd2 = _mm256_fmadd_pd(df2, df2, vd2);
                _mm256_storeu_pd(dest + c + 8, vd2);

                __m256d vm3 = _mm256_loadu_pd(mean_buf + c + 12);
                __m256d vs3 = _mm256_loadu_pd(row + c + 12);
                __m256d vd3 = _mm256_loadu_pd(dest + c + 12);
                __m256d df3 = _mm256_sub_pd(vs3, vm3);
                vd3 = _mm256_fmadd_pd(df3, df3, vd3);
                _mm256_storeu_pd(dest + c + 12, vd3);
            }
            for (; c + 3 < N; c += 4) {
                __m256d vm0 = _mm256_loadu_pd(mean_buf + c);
                __m256d vs0 = _mm256_loadu_pd(row + c);
                __m256d vd0 = _mm256_loadu_pd(dest + c);
                __m256d df0 = _mm256_sub_pd(vs0, vm0);
                vd0 = _mm256_fmadd_pd(df0, df0, vd0);
                _mm256_storeu_pd(dest + c, vd0);
            }
#endif
            for (; c < N; c++) {
                double diff = row[c] - mean_buf[c];
                dest[c] += diff * diff;
            }
        }
        if (M <= ddof) {
            for (int64_t c = 0; c < N; c++) {
                dest[c] = (dest[c] == 0.0 || std::isnan(dest[c])) ? NAN : INFINITY;
            }
            return;
        }
        double scale = 1.0 / (double)(M - ddof);
        int64_t c = 0;
#if HAS_AVX2_REDUCTIONS
        __m256d vscale = _mm256_set1_pd(scale);
        for (; c + 15 < N; c += 16) {
            _mm256_storeu_pd(dest + c, _mm256_mul_pd(_mm256_loadu_pd(dest + c), vscale));
            _mm256_storeu_pd(dest + c + 4, _mm256_mul_pd(_mm256_loadu_pd(dest + c + 4), vscale));
            _mm256_storeu_pd(dest + c + 8, _mm256_mul_pd(_mm256_loadu_pd(dest + c + 8), vscale));
            _mm256_storeu_pd(dest + c + 12, _mm256_mul_pd(_mm256_loadu_pd(dest + c + 12), vscale));
        }
        for (; c + 3 < N; c += 4) {
            _mm256_storeu_pd(dest + c, _mm256_mul_pd(_mm256_loadu_pd(dest + c), vscale));
        }
#endif
        for (; c < N; c++) {
            dest[c] *= scale;
        }
        return;
    }

    if (axis == rank - 1 && stridesSrc[rank - 1] == 1) {
        int64_t L = shape[rank - 1];
        int64_t outer_size = 1;
        for (int d = 0; d < rank - 1; d++) outer_size *= shape[d];
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetSrc = 0;
        int64_t offsetDest = 0;
        for (int64_t o = 0; o < outer_size; o++) {
            dest[offsetDest] = r_var_double(src + offsetSrc, L, ddof);
            for (int d = rank - 2; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetSrc += stridesSrc[d];
                    offsetDest += stridesDest[d];
                    break;
                }
                coord[d] = 0;
                offsetSrc -= (shape[d] - 1) * stridesSrc[d];
                offsetDest -= (shape[d] - 1) * stridesDest[d];
            }
        }
        return;
    }

    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    int64_t offsetDest = 0;
    int64_t stride_axis = stridesSrc[axis];
    double inv_size = 1.0 / (double)size_axis;
    double inv_denom = (size_axis > ddof) ? (1.0 / (double)(size_axis - ddof)) : 0.0;

    for (int64_t o = 0; o < outer_size; o++) {
        const double *axis_ptr = src + offsetSrc;
        double sum = 0.0;
        for (int64_t i = 0; i < size_axis; i++) {
            sum += axis_ptr[i * stride_axis];
        }
        double mean_val = sum * inv_size;
        double sum_sq = 0.0;
        for (int64_t i = 0; i < size_axis; i++) {
            double diff = axis_ptr[i * stride_axis] - mean_val;
            sum_sq += diff * diff;
        }
        if (size_axis <= ddof) {
            dest[offsetDest] = (sum_sq == 0.0 || std::isnan(sum_sq)) ? NAN : INFINITY;
        } else {
            dest[offsetDest] = sum_sq * inv_denom;
        }

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDest += stridesDest[targetD];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

void s_std_double(const double *src, const int64_t *stridesSrc,
                  double *dest, const int64_t *stridesDest,
                  const int64_t *shape, int rank, int axis, int ddof) {
    s_var_double(src, stridesSrc, dest, stridesDest, shape, rank, axis, ddof);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    bool is_dest_contiguous = is_reduction_dest_contiguous(stridesDest, shape, rank, axis);
    if (is_dest_contiguous) {
        int64_t c = 0;
#if HAS_AVX2_REDUCTIONS
        for (; c + 15 < outer_size; c += 16) {
            _mm256_storeu_pd(dest + c, _mm256_sqrt_pd(_mm256_loadu_pd(dest + c)));
            _mm256_storeu_pd(dest + c + 4, _mm256_sqrt_pd(_mm256_loadu_pd(dest + c + 4)));
            _mm256_storeu_pd(dest + c + 8, _mm256_sqrt_pd(_mm256_loadu_pd(dest + c + 8)));
            _mm256_storeu_pd(dest + c + 12, _mm256_sqrt_pd(_mm256_loadu_pd(dest + c + 12)));
        }
        for (; c + 3 < outer_size; c += 4) {
            _mm256_storeu_pd(dest + c, _mm256_sqrt_pd(_mm256_loadu_pd(dest + c)));
        }
#endif
        for (; c < outer_size; c++) {
            dest[c] = std::sqrt(dest[c]);
        }
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetDest = 0;
    for (int64_t o = 0; o < outer_size; o++) {
        dest[offsetDest] = std::sqrt(dest[offsetDest]);
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) {
                offsetDest += stridesDest[targetD];
                break;
            }
            coord[d] = 0;
            offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

void s_sum_float(const float *src, const int64_t *stridesSrc,
                 float *dest, const int64_t *stridesDest,
                 const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        fill_reduction_dest<float>(dest, stridesDest, shape, rank, axis, 0.0f);
        return;
    }

    if (axis == rank - 1 && stridesSrc[rank - 1] == 1) {
        int64_t L = shape[rank - 1];
        int64_t outer_size = 1;
        for (int d = 0; d < rank - 1; d++) outer_size *= shape[d];
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetSrc = 0;
        int64_t offsetDest = 0;
        for (int64_t o = 0; o < outer_size; o++) {
            dest[offsetDest] = r_sum_float(src + offsetSrc, L);
            for (int d = rank - 2; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetSrc += stridesSrc[d];
                    offsetDest += stridesDest[d];
                    break;
                }
                coord[d] = 0;
                offsetSrc -= (shape[d] - 1) * stridesSrc[d];
                offsetDest -= (shape[d] - 1) * stridesDest[d];
            }
        }
        return;
    }

    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    int64_t offsetDest = 0;
    int64_t stride_axis = stridesSrc[axis];

    for (int64_t o = 0; o < outer_size; o++) {
        const float *axis_ptr = src + offsetSrc;
        double sum = 0.0;
        for (int64_t i = 0; i < size_axis; i++) {
            sum += (double)axis_ptr[i * stride_axis];
        }
        dest[offsetDest] = (float)sum;

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDest += stridesDest[targetD];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

void s_mean_float(const float *src, const int64_t *stridesSrc,
                  float *dest, const int64_t *stridesDest,
                  const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        fill_reduction_dest<float>(dest, stridesDest, shape, rank, axis, NAN);
        return;
    }

    if (rank == 2 && axis == 0 && stridesSrc[1] == 1 && stridesDest[0] == 1) {
        int64_t M = shape[0];
        int64_t N = shape[1];
        s_sum_float(src, stridesSrc, dest, stridesDest, shape, rank, 0);
        float scale = 1.0f / (float)M;
        int64_t c = 0;
#if HAS_AVX2_REDUCTIONS
        __m256 vscale = _mm256_set1_ps(scale);
        for (; c + 31 < N; c += 32) {
            _mm256_storeu_ps(dest + c, _mm256_mul_ps(_mm256_loadu_ps(dest + c), vscale));
            _mm256_storeu_ps(dest + c + 8, _mm256_mul_ps(_mm256_loadu_ps(dest + c + 8), vscale));
            _mm256_storeu_ps(dest + c + 16, _mm256_mul_ps(_mm256_loadu_ps(dest + c + 16), vscale));
            _mm256_storeu_ps(dest + c + 24, _mm256_mul_ps(_mm256_loadu_ps(dest + c + 24), vscale));
        }
        for (; c + 7 < N; c += 8) {
            _mm256_storeu_ps(dest + c, _mm256_mul_ps(_mm256_loadu_ps(dest + c), vscale));
        }
#endif
        for (; c < N; c++) {
            dest[c] *= scale;
        }
        return;
    }

    if (axis == rank - 1 && stridesSrc[rank - 1] == 1) {
        int64_t L = shape[rank - 1];
        int64_t outer_size = 1;
        for (int d = 0; d < rank - 1; d++) outer_size *= shape[d];
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetSrc = 0;
        int64_t offsetDest = 0;
        for (int64_t o = 0; o < outer_size; o++) {
            dest[offsetDest] = r_mean_float(src + offsetSrc, L);
            for (int d = rank - 2; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetSrc += stridesSrc[d];
                    offsetDest += stridesDest[d];
                    break;
                }
                coord[d] = 0;
                offsetSrc -= (shape[d] - 1) * stridesSrc[d];
                offsetDest -= (shape[d] - 1) * stridesDest[d];
            }
        }
        return;
    }

    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    int64_t offsetDest = 0;
    int64_t stride_axis = stridesSrc[axis];
    float inv_size = 1.0f / (float)size_axis;

    for (int64_t o = 0; o < outer_size; o++) {
        const float *axis_ptr = src + offsetSrc;
        double sum = 0.0;
        for (int64_t i = 0; i < size_axis; i++) {
            sum += (double)axis_ptr[i * stride_axis];
        }
        dest[offsetDest] = (float)(sum * (double)inv_size);

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDest += stridesDest[targetD];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

void s_var_float(const float *src, const int64_t *stridesSrc,
                 float *dest, const int64_t *stridesDest,
                 const int64_t *shape, int rank, int axis, int ddof) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        int64_t outer_size = 1;
        for (int d = 0; d < rank; d++) if (d != axis) outer_size *= shape[d];
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetDest = 0;
        for (int64_t o = 0; o < outer_size; o++) {
            dest[offsetDest] = NAN;
            for (int d = rank - 1; d >= 0; d--) {
                if (d == axis) continue;
                coord[d]++;
                int targetD = (d < axis) ? d : (d - 1);
                if (coord[d] < shape[d]) {
                    offsetDest += stridesDest[targetD];
                    break;
                }
                coord[d] = 0;
                offsetDest -= (shape[d] - 1) * stridesDest[targetD];
            }
        }
        return;
    }

    int res_rank = rank - 1;
    DECLARE_RANK_BUFFER(int64_t, res_shape, std::max(1, res_rank));
    int64_t outer_total = 1;
    for (int d = 0, rd = 0; d < rank; d++) {
        if (d != axis) {
            if (shape[d] <= 0) return;
            res_shape[rd++] = shape[d];
            outer_total *= shape[d];
        }
    }
    if (outer_total <= 0) return;
    if (strided_buffers_any_overlap(src, stridesSrc, shape, rank, sizeof(float), dest, stridesDest, res_shape, res_rank, sizeof(float))) {
        DECLARE_RANK_BUFFER(int64_t, temp_strides, std::max(1, res_rank));
        int64_t s = 1;
        for (int d = res_rank - 1; d >= 0; d--) {
            temp_strides[d] = s;
            s *= res_shape[d];
        }
        NoThrowBuffer<float> temp(outer_total);
        if (!temp.ok()) {
            ndarray_set_oom_flag();
            return;
        }
        s_var_float(src, stridesSrc, temp.data(), temp_strides, shape, rank, axis, ddof);
        DECLARE_RANK_BUFFER(int64_t, coord_out, std::max(1, res_rank));
        for (int64_t i = 0; i < outer_total; i++) {
            int64_t offsetDest = 0;
            for (int d = 0; d < res_rank; d++) {
                offsetDest += coord_out[d] * stridesDest[d];
            }
            dest[offsetDest] = temp[i];
            for (int d = res_rank - 1; d >= 0; d--) {
                coord_out[d]++;
                if (coord_out[d] < res_shape[d]) break;
                coord_out[d] = 0;
            }
        }
        return;
    }

    if (rank == 2 && axis == 0 && stridesSrc[1] == 1 && stridesDest[0] == 1) {
        int64_t M = shape[0];
        int64_t N = shape[1];
        int64_t S = stridesSrc[0];
        NoThrowBuffer<float> mean_buf_vec;
        float stack_buf[1024];
        float *mean_buf = stack_buf;
        if (N > 1024) {
            mean_buf_vec.resize(N);
            if (!mean_buf_vec.ok()) {
                ndarray_set_oom_flag();
                return;
            }
            mean_buf = mean_buf_vec.data();
        }

        s_mean_float(src, stridesSrc, mean_buf, stridesDest, shape, rank, 0);

        memset(dest, 0, N * sizeof(float));
        for (int64_t r = 0; r < M; r++) {
            const float *row = src + r * S;
            int64_t c = 0;
#if HAS_AVX2_REDUCTIONS
            for (; c + 31 < N; c += 32) {
                __m256 vm0 = _mm256_loadu_ps(mean_buf + c);
                __m256 vs0 = _mm256_loadu_ps(row + c);
                __m256 vd0 = _mm256_loadu_ps(dest + c);
                __m256 df0 = _mm256_sub_ps(vs0, vm0);
                vd0 = _mm256_fmadd_ps(df0, df0, vd0);
                _mm256_storeu_ps(dest + c, vd0);

                __m256 vm1 = _mm256_loadu_ps(mean_buf + c + 8);
                __m256 vs1 = _mm256_loadu_ps(row + c + 8);
                __m256 vd1 = _mm256_loadu_ps(dest + c + 8);
                __m256 df1 = _mm256_sub_ps(vs1, vm1);
                vd1 = _mm256_fmadd_ps(df1, df1, vd1);
                _mm256_storeu_ps(dest + c + 8, vd1);

                __m256 vm2 = _mm256_loadu_ps(mean_buf + c + 16);
                __m256 vs2 = _mm256_loadu_ps(row + c + 16);
                __m256 vd2 = _mm256_loadu_ps(dest + c + 16);
                __m256 df2 = _mm256_sub_ps(vs2, vm2);
                vd2 = _mm256_fmadd_ps(df2, df2, vd2);
                _mm256_storeu_ps(dest + c + 16, vd2);

                __m256 vm3 = _mm256_loadu_ps(mean_buf + c + 24);
                __m256 vs3 = _mm256_loadu_ps(row + c + 24);
                __m256 vd3 = _mm256_loadu_ps(dest + c + 24);
                __m256 df3 = _mm256_sub_ps(vs3, vm3);
                vd3 = _mm256_fmadd_ps(df3, df3, vd3);
                _mm256_storeu_ps(dest + c + 24, vd3);
            }
            for (; c + 7 < N; c += 8) {
                __m256 vm0 = _mm256_loadu_ps(mean_buf + c);
                __m256 vs0 = _mm256_loadu_ps(row + c);
                __m256 vd0 = _mm256_loadu_ps(dest + c);
                __m256 df0 = _mm256_sub_ps(vs0, vm0);
                vd0 = _mm256_fmadd_ps(df0, df0, vd0);
                _mm256_storeu_ps(dest + c, vd0);
            }
#endif
            for (; c < N; c++) {
                float diff = row[c] - mean_buf[c];
                dest[c] += diff * diff;
            }
        }
        if (M <= ddof) {
            for (int64_t c = 0; c < N; c++) {
                dest[c] = (dest[c] == 0.0f || std::isnan(dest[c])) ? NAN : INFINITY;
            }
            return;
        }
        float scale = 1.0f / (float)(M - ddof);
        int64_t c = 0;
#if HAS_AVX2_REDUCTIONS
        __m256 vscale = _mm256_set1_ps(scale);
        for (; c + 31 < N; c += 32) {
            _mm256_storeu_ps(dest + c, _mm256_mul_ps(_mm256_loadu_ps(dest + c), vscale));
            _mm256_storeu_ps(dest + c + 8, _mm256_mul_ps(_mm256_loadu_ps(dest + c + 8), vscale));
            _mm256_storeu_ps(dest + c + 16, _mm256_mul_ps(_mm256_loadu_ps(dest + c + 16), vscale));
            _mm256_storeu_ps(dest + c + 24, _mm256_mul_ps(_mm256_loadu_ps(dest + c + 24), vscale));
        }
        for (; c + 7 < N; c += 8) {
            _mm256_storeu_ps(dest + c, _mm256_mul_ps(_mm256_loadu_ps(dest + c), vscale));
        }
#endif
        for (; c < N; c++) {
            dest[c] *= scale;
        }
        return;
    }

    if (axis == rank - 1 && stridesSrc[rank - 1] == 1) {
        int64_t L = shape[rank - 1];
        int64_t outer_size = 1;
        for (int d = 0; d < rank - 1; d++) outer_size *= shape[d];
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetSrc = 0;
        int64_t offsetDest = 0;
        for (int64_t o = 0; o < outer_size; o++) {
            dest[offsetDest] = r_var_float(src + offsetSrc, L, ddof);
            for (int d = rank - 2; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetSrc += stridesSrc[d];
                    offsetDest += stridesDest[d];
                    break;
                }
                coord[d] = 0;
                offsetSrc -= (shape[d] - 1) * stridesSrc[d];
                offsetDest -= (shape[d] - 1) * stridesDest[d];
            }
        }
        return;
    }

    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    int64_t offsetDest = 0;
    int64_t stride_axis = stridesSrc[axis];
    float inv_size = 1.0f / (float)size_axis;
    float inv_denom = (size_axis > ddof) ? (1.0f / (float)(size_axis - ddof)) : 0.0f;

    for (int64_t o = 0; o < outer_size; o++) {
        const float *axis_ptr = src + offsetSrc;
        double sum = 0.0;
        for (int64_t i = 0; i < size_axis; i++) {
            sum += (double)axis_ptr[i * stride_axis];
        }
        float mean_val = (float)(sum * (double)inv_size);
        float sum_sq = 0.0f;
        for (int64_t i = 0; i < size_axis; i++) {
            float diff = axis_ptr[i * stride_axis] - mean_val;
            sum_sq += diff * diff;
        }
        if (size_axis <= ddof) {
            dest[offsetDest] = (sum_sq == 0.0f || std::isnan(sum_sq)) ? NAN : INFINITY;
        } else {
            dest[offsetDest] = sum_sq * inv_denom;
        }

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDest += stridesDest[targetD];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

void s_std_float(const float *src, const int64_t *stridesSrc,
                 float *dest, const int64_t *stridesDest,
                 const int64_t *shape, int rank, int axis, int ddof) {
    s_var_float(src, stridesSrc, dest, stridesDest, shape, rank, axis, ddof);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    bool is_dest_contiguous = is_reduction_dest_contiguous(stridesDest, shape, rank, axis);
    if (is_dest_contiguous) {
        int64_t c = 0;
#if HAS_AVX2_REDUCTIONS
        for (; c + 31 < outer_size; c += 32) {
            _mm256_storeu_ps(dest + c, _mm256_sqrt_ps(_mm256_loadu_ps(dest + c)));
            _mm256_storeu_ps(dest + c + 8, _mm256_sqrt_ps(_mm256_loadu_ps(dest + c + 8)));
            _mm256_storeu_ps(dest + c + 16, _mm256_sqrt_ps(_mm256_loadu_ps(dest + c + 16)));
            _mm256_storeu_ps(dest + c + 24, _mm256_sqrt_ps(_mm256_loadu_ps(dest + c + 24)));
        }
        for (; c + 7 < outer_size; c += 8) {
            _mm256_storeu_ps(dest + c, _mm256_sqrt_ps(_mm256_loadu_ps(dest + c)));
        }
#endif
        for (; c < outer_size; c++) {
            dest[c] = std::sqrt(dest[c]);
        }
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetDest = 0;
    for (int64_t o = 0; o < outer_size; o++) {
        dest[offsetDest] = std::sqrt(dest[offsetDest]);
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) {
                offsetDest += stridesDest[targetD];
                break;
            }
            coord[d] = 0;
            offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

void s_mean_float_to_double(const float *src, const int64_t *stridesSrc,
                            double *dest, const int64_t *stridesDest,
                            const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        fill_reduction_dest<double>(dest, stridesDest, shape, rank, axis, NAN);
        return;
    }

    if (axis == rank - 1 && stridesSrc[rank - 1] == 1) {
        int64_t L = shape[rank - 1];
        int64_t outer_size = 1;
        for (int d = 0; d < rank - 1; d++) outer_size *= shape[d];
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetSrc = 0;
        int64_t offsetDest = 0;
        for (int64_t o = 0; o < outer_size; o++) {
            dest[offsetDest] = r_mean_float_to_double(src + offsetSrc, L);
            for (int d = rank - 2; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetSrc += stridesSrc[d];
                    offsetDest += stridesDest[d];
                    break;
                }
                coord[d] = 0;
                offsetSrc -= (shape[d] - 1) * stridesSrc[d];
                offsetDest -= (shape[d] - 1) * stridesDest[d];
            }
        }
        return;
    }

    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    int64_t offsetDest = 0;
    int64_t stride_axis = stridesSrc[axis];
    double inv_size = 1.0 / (double)size_axis;

    for (int64_t o = 0; o < outer_size; o++) {
        const float *axis_ptr = src + offsetSrc;
        double sum = 0.0;
        for (int64_t i = 0; i < size_axis; i++) {
            sum += (double)axis_ptr[i * stride_axis];
        }
        dest[offsetDest] = sum * inv_size;

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDest += stridesDest[targetD];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

void s_var_float_to_double(const float *src, const int64_t *stridesSrc,
                           double *dest, const int64_t *stridesDest,
                           const int64_t *shape, int rank, int axis, int ddof) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        int64_t outer_size = 1;
        for (int d = 0; d < rank; d++) if (d != axis) outer_size *= shape[d];
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetDest = 0;
        for (int64_t o = 0; o < outer_size; o++) {
            dest[offsetDest] = NAN;
            for (int d = rank - 1; d >= 0; d--) {
                if (d == axis) continue;
                coord[d]++;
                int targetD = (d < axis) ? d : (d - 1);
                if (coord[d] < shape[d]) {
                    offsetDest += stridesDest[targetD];
                    break;
                }
                coord[d] = 0;
                offsetDest -= (shape[d] - 1) * stridesDest[targetD];
            }
        }
        return;
    }

    if (axis == rank - 1 && stridesSrc[rank - 1] == 1) {
        int64_t L = shape[rank - 1];
        int64_t outer_size = 1;
        for (int d = 0; d < rank - 1; d++) outer_size *= shape[d];
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetSrc = 0;
        int64_t offsetDest = 0;
        for (int64_t o = 0; o < outer_size; o++) {
            dest[offsetDest] = r_var_float_to_double(src + offsetSrc, L, ddof);
            for (int d = rank - 2; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetSrc += stridesSrc[d];
                    offsetDest += stridesDest[d];
                    break;
                }
                coord[d] = 0;
                offsetSrc -= (shape[d] - 1) * stridesSrc[d];
                offsetDest -= (shape[d] - 1) * stridesDest[d];
            }
        }
        return;
    }

    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    int64_t offsetDest = 0;
    int64_t stride_axis = stridesSrc[axis];
    double inv_size = 1.0 / (double)size_axis;
    double inv_denom = (size_axis > ddof) ? (1.0 / (double)(size_axis - ddof)) : 0.0;

    for (int64_t o = 0; o < outer_size; o++) {
        const float *axis_ptr = src + offsetSrc;
        double sum = 0.0;
        for (int64_t i = 0; i < size_axis; i++) {
            sum += (double)axis_ptr[i * stride_axis];
        }
        double mean_val = sum * inv_size;
        double sum_sq = 0.0;
        for (int64_t i = 0; i < size_axis; i++) {
            double diff = (double)axis_ptr[i * stride_axis] - mean_val;
            sum_sq += diff * diff;
        }
        if (size_axis <= ddof) {
            dest[offsetDest] = (sum_sq == 0.0 || std::isnan(sum_sq)) ? NAN : INFINITY;
        } else {
            dest[offsetDest] = sum_sq * inv_denom;
        }

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDest += stridesDest[targetD];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

void s_std_float_to_double(const float *src, const int64_t *stridesSrc,
                           double *dest, const int64_t *stridesDest,
                           const int64_t *shape, int rank, int axis, int ddof) {
    s_var_float_to_double(src, stridesSrc, dest, stridesDest, shape, rank, axis, ddof);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetDest = 0;
    for (int64_t o = 0; o < outer_size; o++) {
        dest[offsetDest] = std::sqrt(dest[offsetDest]);
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) {
                offsetDest += stridesDest[targetD];
                break;
            }
            coord[d] = 0;
            offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

template <typename T>
static inline void s_mean_int_impl(const T *src, const int64_t *stridesSrc,
                                   double *dest, const int64_t *stridesDest,
                                   const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        int64_t outer_size = 1;
        for (int d = 0; d < rank; d++) if (d != axis) outer_size *= shape[d];
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetDest = 0;
        for (int64_t o = 0; o < outer_size; o++) {
            dest[offsetDest] = NAN;
            for (int d = rank - 1; d >= 0; d--) {
                if (d == axis) continue;
                coord[d]++;
                int targetD = (d < axis) ? d : (d - 1);
                if (coord[d] < shape[d]) {
                    offsetDest += stridesDest[targetD];
                    break;
                }
                coord[d] = 0;
                offsetDest -= (shape[d] - 1) * stridesDest[targetD];
            }
        }
        return;
    }

    if (axis == rank - 1 && stridesSrc[rank - 1] == 1) {
        int64_t L = shape[rank - 1];
        int64_t outer_size = 1;
        for (int d = 0; d < rank - 1; d++) outer_size *= shape[d];
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetSrc = 0;
        int64_t offsetDest = 0;
        for (int64_t o = 0; o < outer_size; o++) {
            dest[offsetDest] = r_mean_int_impl(src + offsetSrc, L);
            for (int d = rank - 2; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetSrc += stridesSrc[d];
                    offsetDest += stridesDest[d];
                    break;
                }
                coord[d] = 0;
                offsetSrc -= (shape[d] - 1) * stridesSrc[d];
                offsetDest -= (shape[d] - 1) * stridesDest[d];
            }
        }
        return;
    }

    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    int64_t offsetDest = 0;
    int64_t stride_axis = stridesSrc[axis];
    double inv_size = 1.0 / (double)size_axis;

    for (int64_t o = 0; o < outer_size; o++) {
        const T *axis_ptr = src + offsetSrc;
        double sum = 0.0;
        for (int64_t i = 0; i < size_axis; i++) {
            sum += (double)axis_ptr[i * stride_axis];
        }
        dest[offsetDest] = sum * inv_size;

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDest += stridesDest[targetD];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

template <typename T>
static inline void s_var_int_impl(const T *src, const int64_t *stridesSrc,
                                  double *dest, const int64_t *stridesDest,
                                  const int64_t *shape, int rank, int axis, int ddof) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        int64_t outer_size = 1;
        for (int d = 0; d < rank; d++) if (d != axis) outer_size *= shape[d];
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetDest = 0;
        for (int64_t o = 0; o < outer_size; o++) {
            dest[offsetDest] = NAN;
            for (int d = rank - 1; d >= 0; d--) {
                if (d == axis) continue;
                coord[d]++;
                int targetD = (d < axis) ? d : (d - 1);
                if (coord[d] < shape[d]) {
                    offsetDest += stridesDest[targetD];
                    break;
                }
                coord[d] = 0;
                offsetDest -= (shape[d] - 1) * stridesDest[targetD];
            }
        }
        return;
    }

    if (axis == rank - 1 && stridesSrc[rank - 1] == 1) {
        int64_t L = shape[rank - 1];
        int64_t outer_size = 1;
        for (int d = 0; d < rank - 1; d++) outer_size *= shape[d];
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetSrc = 0;
        int64_t offsetDest = 0;
        for (int64_t o = 0; o < outer_size; o++) {
            dest[offsetDest] = r_var_int_impl(src + offsetSrc, L, ddof);
            for (int d = rank - 2; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetSrc += stridesSrc[d];
                    offsetDest += stridesDest[d];
                    break;
                }
                coord[d] = 0;
                offsetSrc -= (shape[d] - 1) * stridesSrc[d];
                offsetDest -= (shape[d] - 1) * stridesDest[d];
            }
        }
        return;
    }

    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    int64_t offsetDest = 0;
    int64_t stride_axis = stridesSrc[axis];
    double inv_size = 1.0 / (double)size_axis;
    double inv_denom = (size_axis > ddof) ? (1.0 / (double)(size_axis - ddof)) : 0.0;

    for (int64_t o = 0; o < outer_size; o++) {
        const T *axis_ptr = src + offsetSrc;
        double sum = 0.0;
        for (int64_t i = 0; i < size_axis; i++) {
            sum += (double)axis_ptr[i * stride_axis];
        }
        double mean_val = sum * inv_size;
        double sum_sq = 0.0;
        for (int64_t i = 0; i < size_axis; i++) {
            double diff = (double)axis_ptr[i * stride_axis] - mean_val;
            sum_sq += diff * diff;
        }
        if (size_axis <= ddof) {
            dest[offsetDest] = (sum_sq == 0.0 || std::isnan(sum_sq)) ? NAN : INFINITY;
        } else {
            dest[offsetDest] = sum_sq * inv_denom;
        }

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDest += stridesDest[targetD];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

void s_mean_int64_to_double(const int64_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) { s_mean_int_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis); }
void s_mean_int32_to_double(const int32_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) { s_mean_int_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis); }
void s_mean_uint8_to_double(const uint8_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) { s_mean_int_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis); }
void s_mean_int16_to_double(const int16_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) { s_mean_int_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis); }

void s_var_int64_to_double(const int64_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof) { s_var_int_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, ddof); }
void s_var_int32_to_double(const int32_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof) { s_var_int_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, ddof); }
void s_var_uint8_to_double(const uint8_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof) { s_var_int_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, ddof); }
void s_var_int16_to_double(const int16_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof) { s_var_int_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, ddof); }

void s_std_int64_to_double(const int64_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof) {
    s_var_int64_to_double(src, stridesSrc, dest, stridesDest, shape, rank, axis, ddof);
    int64_t outer_size = 1; for (int d = 0; d < rank; d++) if (d != axis) outer_size *= shape[d];
    DECLARE_RANK_BUFFER(int64_t, coord, rank); int64_t offsetDest = 0;
    for (int64_t o = 0; o < outer_size; o++) {
        dest[offsetDest] = std::sqrt(dest[offsetDest]);
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++; int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) { offsetDest += stridesDest[targetD]; break; }
            coord[d] = 0; offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}
void s_std_int32_to_double(const int32_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof) {
    s_var_int32_to_double(src, stridesSrc, dest, stridesDest, shape, rank, axis, ddof);
    int64_t outer_size = 1; for (int d = 0; d < rank; d++) if (d != axis) outer_size *= shape[d];
    DECLARE_RANK_BUFFER(int64_t, coord, rank); int64_t offsetDest = 0;
    for (int64_t o = 0; o < outer_size; o++) {
        dest[offsetDest] = std::sqrt(dest[offsetDest]);
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++; int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) { offsetDest += stridesDest[targetD]; break; }
            coord[d] = 0; offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}
void s_std_uint8_to_double(const uint8_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof) {
    s_var_uint8_to_double(src, stridesSrc, dest, stridesDest, shape, rank, axis, ddof);
    int64_t outer_size = 1; for (int d = 0; d < rank; d++) if (d != axis) outer_size *= shape[d];
    DECLARE_RANK_BUFFER(int64_t, coord, rank); int64_t offsetDest = 0;
    for (int64_t o = 0; o < outer_size; o++) {
        dest[offsetDest] = std::sqrt(dest[offsetDest]);
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++; int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) { offsetDest += stridesDest[targetD]; break; }
            coord[d] = 0; offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}
void s_std_int16_to_double(const int16_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof) {
    s_var_int16_to_double(src, stridesSrc, dest, stridesDest, shape, rank, axis, ddof);
    int64_t outer_size = 1; for (int d = 0; d < rank; d++) if (d != axis) outer_size *= shape[d];
    DECLARE_RANK_BUFFER(int64_t, coord, rank); int64_t offsetDest = 0;
    for (int64_t o = 0; o < outer_size; o++) {
        dest[offsetDest] = std::sqrt(dest[offsetDest]);
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++; int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) { offsetDest += stridesDest[targetD]; break; }
            coord[d] = 0; offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

void s_sum_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        fill_reduction_dest<cpx_t>(dest, stridesDest, shape, rank, axis, cpx_t{0.0, 0.0});
        return;
    }
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) if (d != axis) outer_size *= shape[d];
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    int64_t offsetDest = 0;
    int64_t stride_axis = stridesSrc[axis];
    for (int64_t o = 0; o < outer_size; o++) {
        const cpx_t *axis_ptr = src + offsetSrc;
        double re = 0.0, im = 0.0;
        for (int64_t i = 0; i < size_axis; i++) {
            re += axis_ptr[i * stride_axis].r;
            im += axis_ptr[i * stride_axis].i;
        }
        dest[offsetDest] = cpx_t{re, im};
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++; int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) { offsetSrc += stridesSrc[d]; offsetDest += stridesDest[targetD]; break; }
            coord[d] = 0; offsetSrc -= (shape[d] - 1) * stridesSrc[d]; offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

void s_sum_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        fill_reduction_dest<cpx_f_t>(dest, stridesDest, shape, rank, axis, cpx_f_t{0.0f, 0.0f});
        return;
    }
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) if (d != axis) outer_size *= shape[d];
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    int64_t offsetDest = 0;
    int64_t stride_axis = stridesSrc[axis];
    for (int64_t o = 0; o < outer_size; o++) {
        const cpx_f_t *axis_ptr = src + offsetSrc;
        float re = 0.0f, im = 0.0f;
        for (int64_t i = 0; i < size_axis; i++) {
            re += axis_ptr[i * stride_axis].r;
            im += axis_ptr[i * stride_axis].i;
        }
        dest[offsetDest] = cpx_f_t{re, im};
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++; int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) { offsetSrc += stridesSrc[d]; offsetDest += stridesDest[targetD]; break; }
            coord[d] = 0; offsetSrc -= (shape[d] - 1) * stridesSrc[d]; offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

void s_mean_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        fill_reduction_dest<cpx_t>(dest, stridesDest, shape, rank, axis, cpx_t{NAN, NAN});
        return;
    }
    s_sum_complex128(src, stridesSrc, dest, stridesDest, shape, rank, axis);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) if (d != axis) outer_size *= shape[d];
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetDest = 0;
    double inv_size = 1.0 / (double)size_axis;
    for (int64_t o = 0; o < outer_size; o++) {
        dest[offsetDest].r *= inv_size;
        dest[offsetDest].i *= inv_size;
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++; int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) { offsetDest += stridesDest[targetD]; break; }
            coord[d] = 0; offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

void s_mean_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        fill_reduction_dest<cpx_f_t>(dest, stridesDest, shape, rank, axis, cpx_f_t{NAN, NAN});
        return;
    }
    s_sum_complex64(src, stridesSrc, dest, stridesDest, shape, rank, axis);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) if (d != axis) outer_size *= shape[d];
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetDest = 0;
    float inv_size = 1.0f / (float)size_axis;
    for (int64_t o = 0; o < outer_size; o++) {
        dest[offsetDest].r *= inv_size;
        dest[offsetDest].i *= inv_size;
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++; int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) { offsetDest += stridesDest[targetD]; break; }
            coord[d] = 0; offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

void s_mean_complex64_to_complex128(const cpx_f_t *src, const int64_t *stridesSrc, cpx_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        fill_reduction_dest<cpx_t>(dest, stridesDest, shape, rank, axis, cpx_t{NAN, NAN});
        return;
    }
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) if (d != axis) outer_size *= shape[d];
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    int64_t offsetDest = 0;
    int64_t stride_axis = stridesSrc[axis];
    double inv_size = 1.0 / (double)size_axis;
    for (int64_t o = 0; o < outer_size; o++) {
        const cpx_f_t *axis_ptr = src + offsetSrc;
        double sum_r = 0.0, sum_i = 0.0;
        for (int64_t i = 0; i < size_axis; i++) {
            sum_r += (double)axis_ptr[i * stride_axis].r;
            sum_i += (double)axis_ptr[i * stride_axis].i;
        }
        dest[offsetDest] = cpx_t{sum_r * inv_size, sum_i * inv_size};
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++; int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) { offsetSrc += stridesSrc[d]; offsetDest += stridesDest[targetD]; break; }
            coord[d] = 0; offsetSrc -= (shape[d] - 1) * stridesSrc[d]; offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

// ============================================================================
// 2. DOUBLE PRECISION (FLOAT64) GENERIC ND STRIDED BROADCASTING KERNELS
// ============================================================================

// Macros to define the boilerplate iterative odometer index advancements!
#define ADVANCE_ODOMETER_LOOP \
    for (int d = rank - 1; d >= 0; d--) { \
        coord[d]++; \
        if (coord[d] < shape[d]) break; \
        coord[d] = 0; \
    }

void s_add_double(const double *a, const int64_t *stridesA,
                  const double *b, const int64_t *stridesB,
                  double *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank) {
    if (rank > 0 && (stridesA == nullptr || stridesB == nullptr || stridesRes == nullptr)) return;
    strided_binary_op_impl(a, stridesA, b, stridesB, res, stridesRes, shape, rank, nullptr, [](double x, double y) { return x + y; });
}

void s_sub_double(const double *a, const int64_t *stridesA,
                  const double *b, const int64_t *stridesB,
                  double *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank) {
    if (rank > 0 && (stridesA == nullptr || stridesB == nullptr || stridesRes == nullptr)) return;
    strided_binary_op_impl(a, stridesA, b, stridesB, res, stridesRes, shape, rank, nullptr, [](double x, double y) { return x - y; });
}

void s_mul_double(const double *a, const int64_t *stridesA,
                  const double *b, const int64_t *stridesB,
                  double *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank) {
    if (rank > 0 && (stridesA == nullptr || stridesB == nullptr || stridesRes == nullptr)) return;
    strided_binary_op_impl(a, stridesA, b, stridesB, res, stridesRes, shape, rank, nullptr, [](double x, double y) { return x * y; });
}

void s_div_double(const double *a, const int64_t *stridesA,
                  const double *b, const int64_t *stridesB,
                  double *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank) {
    if (rank > 0 && (stridesA == nullptr || stridesB == nullptr || stridesRes == nullptr)) return;
    strided_binary_op_impl(a, stridesA, b, stridesB, res, stridesRes, shape, rank, nullptr, [](double x, double y) { return x / y; });
}

// ============================================================================
// 3. COMPLEX128 VECTOR KERNELS (CONTIGUOUS & STRIDED)
// ============================================================================

void v_add_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int64_t size, const uint8_t *mask) {
    v_binary_impl(a, b, res, size, mask, cpx_add);
}

void v_add_complex64(const cpx_f_t *a, const cpx_f_t *b, cpx_f_t *res, int64_t size, const uint8_t *mask) {
    v_binary_impl(a, b, res, size, mask, cpx_add_f);
}

void v_sub_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int64_t size, const uint8_t *mask) {
    v_binary_impl(a, b, res, size, mask, cpx_sub);
}

void v_sub_complex64(const cpx_f_t *a, const cpx_f_t *b, cpx_f_t *res, int64_t size, const uint8_t *mask) {
    v_binary_impl(a, b, res, size, mask, cpx_sub_f);
}

void v_mul_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int64_t size, const uint8_t *mask) {
    v_binary_impl(a, b, res, size, mask, cpx_mul);
}

void v_mul_complex64(const cpx_f_t *a, const cpx_f_t *b, cpx_f_t *res, int64_t size, const uint8_t *mask) {
    v_binary_impl(a, b, res, size, mask, cpx_mul_f);
}

void v_div_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int64_t size, const uint8_t *mask) {
    v_binary_impl(a, b, res, size, mask, cpx_div);
}

void v_div_complex64(const cpx_f_t *a, const cpx_f_t *b, cpx_f_t *res, int64_t size, const uint8_t *mask) {
    v_binary_impl(a, b, res, size, mask, cpx_div_f);
}

void s_add_complex(const cpx_t *a, const int64_t *stridesA,
                  const cpx_t *b, const int64_t *stridesB,
                  cpx_t *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank) {
    if (rank > 0 && (stridesA == nullptr || stridesB == nullptr || stridesRes == nullptr)) return;
    strided_binary_op_impl(a, stridesA, b, stridesB, res, stridesRes, shape, rank, nullptr, cpx_add);
}

void s_sub_complex(const cpx_t *a, const int64_t *stridesA,
                  const cpx_t *b, const int64_t *stridesB,
                  cpx_t *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank) {
    if (rank > 0 && (stridesA == nullptr || stridesB == nullptr || stridesRes == nullptr)) return;
    strided_binary_op_impl(a, stridesA, b, stridesB, res, stridesRes, shape, rank, nullptr, cpx_sub);
}

void s_mul_complex(const cpx_t *a, const int64_t *stridesA,
                  const cpx_t *b, const int64_t *stridesB,
                  cpx_t *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank) {
    if (rank > 0 && (stridesA == nullptr || stridesB == nullptr || stridesRes == nullptr)) return;
    strided_binary_op_impl(a, stridesA, b, stridesB, res, stridesRes, shape, rank, nullptr, cpx_mul);
}

void s_div_complex(const cpx_t *a, const int64_t *stridesA,
                  const cpx_t *b, const int64_t *stridesB,
                  cpx_t *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank) {
    if (rank > 0 && (stridesA == nullptr || stridesB == nullptr || stridesRes == nullptr)) return;
    strided_binary_op_impl(a, stridesA, b, stridesB, res, stridesRes, shape, rank, nullptr, cpx_div);
}

// ============================================================================
// 4. SINGLE PRECISION (FLOAT32) VECTOR CONTIGUOUS KERNELS
// ============================================================================

void v_add_float(const float *a, const float *b, float *res, int64_t size, const uint8_t *mask) {
    v_binary_impl(a, b, res, size, mask, [](float x, float y) { return x + y; });
}

void v_sub_float(const float *a, const float *b, float *res, int64_t size, const uint8_t *mask) {
    v_binary_impl(a, b, res, size, mask, [](float x, float y) { return x - y; });
}

void v_mul_float(const float *a, const float *b, float *res, int64_t size, const uint8_t *mask) {
    v_binary_impl(a, b, res, size, mask, [](float x, float y) { return x * y; });
}

void v_div_float(const float *a, const float *b, float *res, int64_t size, const uint8_t *mask) {
    v_binary_impl(a, b, res, size, mask, [](float x, float y) { return x / y; });
}

void v_sin_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::sin(x); });
}

void v_sinc_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return real_sinc<float>(x); });
}

void v_cos_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::cos(x); });
}

void v_exp_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::exp(x); });
}

void v_log_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::log(x); });
}

void v_sinh_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::sinh(x); });
}

void v_cosh_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::cosh(x); });
}

void v_tanh_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::tanh(x); });
}

void v_asinh_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::asinh(x); });
}

void v_acosh_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::acosh(x); });
}

void v_atanh_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::atanh(x); });
}

void v_asin_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::asin(x); });
}

void v_acos_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::acos(x); });
}

void v_atan_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::atan(x); });
}

void v_atan2_float(const float *y, const float *x, float *res, int64_t size, const uint8_t *mask) {
    v_binary_impl(y, x, res, size, mask, [](float yv, float xv) { return std::atan2(yv, xv); });
}



// ============================================================================
// 5. ADDITIONAL MATH, ROUNDING & CLIPPING KERNELS (SIMD AUTOVECTORIZABLE)
// ============================================================================

void v_sqrt_double(const double *src, double *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](double x) { return std::sqrt(x); });
}

void v_tan_double(const double *src, double *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](double x) { return std::tan(x); });
}

void v_abs_double(const double *src, double *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](double x) { return std::fabs(x); });
}

void v_ceil_double(const double *src, double *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](double x) { return std::ceil(x); });
}

void v_floor_double(const double *src, double *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](double x) { return std::floor(x); });
}

void v_round_double(const double *src, double *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](double x) { return std::rint(x); });
}

void v_clip_double(const double *src, double *res, double min_val, double max_val, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [min_val, max_val](double val) {
        if (std::isnan(val) || std::isnan(min_val) || std::isnan(max_val)) return static_cast<double>(NAN);
        if (val < min_val) val = min_val;
        if (val > max_val) val = max_val;
        return val;
    });
}

void v_sqrt_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::sqrt(x); });
}

void v_tan_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::tan(x); });
}

void v_abs_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::fabs(x); });
}

void v_ceil_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::ceil(x); });
}

void v_floor_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::floor(x); });
}

void v_round_float(const float *src, float *res, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [](float x) { return std::nearbyint(x); });
}

void v_clip_float(const float *src, float *res, float min_val, float max_val, int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, [min_val, max_val](float val) {
        if (std::isnan(val) || std::isnan(min_val) || std::isnan(max_val)) return static_cast<float>(NAN);
        if (val < min_val) val = min_val;
        if (val > max_val) val = max_val;
        return val;
    });
}

// ============================================================================
// 6. GENERIC ND STRIDED BROADCASTING TERNARY "where" KERNELS (RANK <= 8)
// ============================================================================

#define DEFINE_S_WHERE(NAME, TYPE) \
void s_where_##NAME(const unsigned char *cond, const int64_t *stridesCond, \
                    const TYPE *x, const int64_t *stridesX, \
                    const TYPE *y, const int64_t *stridesY, \
                    TYPE *res, const int64_t *stridesRes, \
                    const int64_t *shape, int rank) { \
    s_where_impl<TYPE>(cond, stridesCond, x, stridesX, y, stridesY, res, stridesRes, shape, rank); \
}

DEFINE_S_WHERE(double, double)
DEFINE_S_WHERE(float, float)
DEFINE_S_WHERE(int64, int64_t)
DEFINE_S_WHERE(int32, int32_t)
DEFINE_S_WHERE(uint8, uint8_t)
DEFINE_S_WHERE(int16, int16_t)
DEFINE_S_WHERE(complex128, cpx_t)
DEFINE_S_WHERE(complex64, cpx_f_t)

#undef DEFINE_S_WHERE

static inline uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline void xoshiro256_seed(uint64_t seed, uint64_t s[4]) {
    uint64_t x = seed;
    s[0] = splitmix64(&x);
    s[1] = splitmix64(&x);
    s[2] = splitmix64(&x);
    s[3] = splitmix64(&x);
}

static inline uint64_t rotl(const uint64_t x, int64_t k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoshiro256_next(uint64_t s[4]) {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rotl(s[3], 45);

    return result;
}

void v_normal_double(double *res, int64_t size, double loc, double scale, uint64_t seed) {
    if (res == nullptr || size <= 0) return;
    if (std::isnan(loc) || std::isnan(scale) || scale < 0.0) {
        for (int64_t i = 0; i < size; i++) res[i] = NAN;
        return;
    }
    if (scale == 0.0) {
        for (int64_t i = 0; i < size; i++) res[i] = loc;
        return;
    }

    uint64_t s[4];
    xoshiro256_seed(seed, s);

    int64_t i = 0;
    while (i < size) {
        double u1;
        do {
            u1 = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
        } while (u1 == 0.0);

        double u2 = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);

        double mag = scale * std::sqrt(-2.0 * std::log(u1));
        double angle = 2.0 * M_PI * u2;

        res[i] = loc + mag * std::cos(angle);
        if (i + 1 < size) {
            res[i + 1] = loc + mag * std::sin(angle);
        }
        i += 2;
    }
}

void v_normal_float(float *res, int64_t size, float loc, float scale, uint64_t seed) {
    if (res == nullptr || size <= 0) return;
    if (std::isnan(loc) || std::isnan(scale) || scale < 0.0f) {
        for (int64_t i = 0; i < size; i++) res[i] = NAN;
        return;
    }
    if (scale == 0.0f) {
        for (int64_t i = 0; i < size; i++) res[i] = loc;
        return;
    }

    uint64_t s[4];
    xoshiro256_seed(seed, s);

    int64_t i = 0;
    while (i < size) {
        float u1;
        do {
            u1 = (float)((double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0));
        } while (u1 == 0.0f);

        float u2 = (float)((double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0));

        float mag = scale * std::sqrt(-2.0f * std::log(u1));
        float angle = 2.0f * (float)M_PI * u2;

        res[i] = loc + mag * std::cos(angle);
        if (i + 1 < size) {
            res[i + 1] = loc + mag * std::sin(angle);
        }
        i += 2;
    }
}

void v_uniform_double(double *res, int64_t size, uint64_t seed) {
    if (res == nullptr || size <= 0) return;

    uint64_t s[4];
    xoshiro256_seed(seed, s);

    for (int64_t i = 0; i < size; i++) {
        res[i] = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
    }
}

void v_uniform_float(float *res, int64_t size, uint64_t seed) {
    if (res == nullptr || size <= 0) return;

    uint64_t s[4];
    xoshiro256_seed(seed, s);

    for (int64_t i = 0; i < size; i++) {
        res[i] = (float)((double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0));
    }
}

#define IMPLEMENT_V_RANDINT(TYPE_NAME, TYPE, BOUNDS_TYPE, RANGE_TYPE) \
void v_randint_##TYPE_NAME(TYPE *res, int64_t size, BOUNDS_TYPE low, BOUNDS_TYPE high, uint64_t seed) { \
    v_randint_impl<TYPE, BOUNDS_TYPE, RANGE_TYPE>(res, size, low, high, seed); \
}

IMPLEMENT_V_RANDINT(int64, int64_t, int64_t, int64_t)
IMPLEMENT_V_RANDINT(int32, int32_t, int32_t, int32_t)
IMPLEMENT_V_RANDINT(int16, int16_t, int, int)
IMPLEMENT_V_RANDINT(uint8, uint8_t, int, int)

#define IMPLEMENT_V_FILL(TYPE_NAME, TYPE) \
void v_fill_##TYPE_NAME(TYPE * RESTRICT res, TYPE value, int64_t size) { \
    v_fill_impl(res, value, size); \
}

#define IMPLEMENT_S_FILL(TYPE_NAME, TYPE) \
void s_fill_##TYPE_NAME(TYPE * RESTRICT res, const int64_t * RESTRICT strides, const int64_t * RESTRICT shape, int rank, TYPE value) { \
    s_fill_impl<TYPE>(res, strides, shape, rank, value); \
}

extern "C" {
IMPLEMENT_V_FILL(double, double)
IMPLEMENT_V_FILL(float, float)
IMPLEMENT_V_FILL(int64, int64_t)
IMPLEMENT_V_FILL(int32, int32_t)
IMPLEMENT_V_FILL(int16, int16_t)
IMPLEMENT_V_FILL(uint8, uint8_t)
IMPLEMENT_V_FILL(boolean, uint8_t)

IMPLEMENT_S_FILL(double, double)
IMPLEMENT_S_FILL(float, float)
IMPLEMENT_S_FILL(int64, int64_t)
IMPLEMENT_S_FILL(int32, int32_t)
IMPLEMENT_S_FILL(int16, int16_t)
IMPLEMENT_S_FILL(uint8, uint8_t)
IMPLEMENT_S_FILL(boolean, uint8_t)

void v_fill_complex128(cpx_t * RESTRICT res, double valR, double valI, int64_t size) {
    if (res == nullptr || size <= 0) return;
    cpx_t val = {valR, valI};
    for (int64_t i = 0; i < size; i++) {
        res[i] = val;
    }
}

void v_fill_complex64(cpx_f_t * RESTRICT res, float valR, float valI, int64_t size) {
    if (res == nullptr || size <= 0) return;
    cpx_f_t val = {valR, valI};
    for (int64_t i = 0; i < size; i++) {
        res[i] = val;
    }
}

void s_fill_complex128(cpx_t * RESTRICT res, const int64_t * RESTRICT strides, const int64_t * RESTRICT shape, int rank, double valR, double valI) {
    cpx_t val = {valR, valI};
    s_fill_impl<cpx_t>(res, strides, shape, rank, val);
}

void s_fill_complex64(cpx_f_t * RESTRICT res, const int64_t * RESTRICT strides, const int64_t * RESTRICT shape, int rank, float valR, float valI) {
    cpx_f_t val = {valR, valI};
    s_fill_impl<cpx_f_t>(res, strides, shape, rank, val);
}
}

void v_linspace_double(double *res, double start, double step, int64_t size) {
    if (res == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = start + i * step;
    }
}

void v_linspace_float(float *res, float start, float step, int64_t size) {
    if (res == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = start + i * step;
    }
}

void v_linspace_complex128(cpx_t *res, double startR, double startI, double stepR, double stepI, int64_t size) {
    if (res == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i].r = startR + i * stepR;
        res[i].i = startI + i * stepI;
    }
}

void v_linspace_complex64(cpx_f_t *res, float startR, float startI, float stepR, float stepI, int64_t size) {
    if (res == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i].r = startR + i * stepR;
        res[i].i = startI + i * stepI;
    }
}

void v_linspace_int64(int64_t *res, double start, double step, int64_t size) {
    if (res == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = saturating_float_to_int<int64_t>(std::floor(start + i * step));
    }
}

void v_linspace_int32(int32_t *res, double start, double step, int64_t size) {
    if (res == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = saturating_float_to_int<int32_t>(std::floor(start + i * step));
    }
}

void v_linspace_int16(int16_t *res, double start, double step, int64_t size) {
    if (res == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = saturating_float_to_int<int16_t>(std::floor(start + i * step));
    }
}

void v_linspace_uint8(uint8_t *res, double start, double step, int64_t size) {
    if (res == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = saturating_float_to_int<uint8_t>(std::floor(start + i * step));
    }
}

void v_logspace_double(double *res, double start, double step, double base, int64_t size) {
    if (res == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = std::pow(base, start + i * step);
    }
}

void v_logspace_float(float *res, float start, float step, float base, int64_t size) {
    if (res == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = std::pow(base, start + i * step);
    }
}

void v_logspace_complex128(cpx_t *res, double startR, double startI, double stepR, double stepI, double baseR, double baseI, int64_t size) {
    if (res == nullptr || size <= 0) return;
    std::complex<double> cbase(baseR, baseI);
    for (int64_t i = 0; i < size; i++) {
        std::complex<double> cexp(startR + i * stepR, startI + i * stepI);
        std::complex<double> cres = std::pow(cbase, cexp);
        res[i].r = cres.real();
        res[i].i = cres.imag();
    }
}

void v_logspace_complex64(cpx_f_t *res, float startR, float startI, float stepR, float stepI, float baseR, float baseI, int64_t size) {
    if (res == nullptr || size <= 0) return;
    std::complex<float> cbase(baseR, baseI);
    for (int64_t i = 0; i < size; i++) {
        std::complex<float> cexp(startR + i * stepR, startI + i * stepI);
        std::complex<float> cres = std::pow(cbase, cexp);
        res[i].r = cres.real();
        res[i].i = cres.imag();
    }
}

void v_geomspace_double(double *res, double logStart, double step, double sign, int64_t size) {
    if (res == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = sign * std::pow(10.0, logStart + i * step);
    }
}

void v_geomspace_float(float *res, float logStart, float step, float sign, int64_t size) {
    if (res == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = sign * std::pow(10.0f, logStart + i * step);
    }
}

void v_geomspace_complex128(cpx_t *res, double logStartR, double logStartI, double stepR, double stepI, int64_t size) {
    if (res == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        std::complex<double> cexp(logStartR + i * stepR, logStartI + i * stepI);
        std::complex<double> cres = std::pow(10.0, cexp);
        res[i].r = cres.real();
        res[i].i = cres.imag();
    }
}

void v_geomspace_complex64(cpx_f_t *res, float logStartR, float logStartI, float stepR, float stepI, int64_t size) {
    if (res == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        std::complex<float> cexp(logStartR + i * stepR, logStartI + i * stepI);
        std::complex<float> cres = std::pow(10.0f, cexp);
        res[i].r = cres.real();
        res[i].i = cres.imag();
    }
}

// Helper functions to perform rolling FNV-1a hash updates
static inline void hash_double(uint32_t *hash, double val) {
    if (val == 0.0) {
        val = 0.0;
    }
    if (val != val) { // isNaN
        const uint64_t nan_bits = 0x7ff8000000000000ULL;
        const uint8_t *bytes = (const uint8_t*)&nan_bits;
        for (int64_t i = 0; i < 8; i++) {
            *hash ^= bytes[i];
            *hash *= 16777619U;
        }
        return;
    }
    const uint8_t *bytes = (const uint8_t*)&val;
    for (int64_t i = 0; i < 8; i++) {
        *hash ^= bytes[i];
        *hash *= 16777619U;
    }
}

static inline void hash_float(uint32_t *hash, float val) {
    if (val == 0.0f) {
        val = 0.0f;
    }
    if (val != val) { // isNaN
        const uint32_t nan_bits = 0x7fc00000U;
        const uint8_t *bytes = (const uint8_t*)&nan_bits;
        for (int64_t i = 0; i < 4; i++) {
            *hash ^= bytes[i];
            *hash *= 16777619U;
        }
        return;
    }
    const uint8_t *bytes = (const uint8_t*)&val;
    for (int64_t i = 0; i < 4; i++) {
        *hash ^= bytes[i];
        *hash *= 16777619U;
    }
}

static inline void hash_int64(uint32_t *hash, int64_t val) {
    const uint8_t *bytes = (const uint8_t*)&val;
    for (int64_t i = 0; i < 8; i++) {
        *hash ^= bytes[i];
        *hash *= 16777619U;
    }
}

static inline void hash_int32(uint32_t *hash, int32_t val) {
    const uint8_t *bytes = (const uint8_t*)&val;
    for (int64_t i = 0; i < 4; i++) {
        *hash ^= bytes[i];
        *hash *= 16777619U;
    }
}

static inline void hash_int16(uint32_t *hash, int16_t val) {
    const uint8_t *bytes = (const uint8_t*)&val;
    for (int64_t i = 0; i < 2; i++) {
        *hash ^= bytes[i];
        *hash *= 16777619U;
    }
}

static inline void hash_uint8(uint32_t *hash, uint8_t val) {
    *hash ^= val;
    *hash *= 16777619U;
}

static inline void hash_boolean(uint32_t *hash, uint8_t val) {
    uint8_t b = val ? 1 : 0;
    *hash ^= b;
    *hash *= 16777619U;
}

// ============================================================================
// 7. NATIVE C HIGH-SPEED STRIDED FLATTENING/COPYING KERNELS
// ============================================================================

void s_flatten_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *shape, int rank) {
    if (src == nullptr || dest == nullptr || rank < 0) return;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        dest[0] = src[0];
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        dest[el] = src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}

void s_flatten_float(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *shape, int rank) {
    if (src == nullptr || dest == nullptr || rank < 0) return;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        dest[0] = src[0];
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        dest[el] = src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}

void s_flatten_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *shape, int rank) {
    if (src == nullptr || dest == nullptr || rank < 0) return;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        dest[0] = src[0];
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        dest[el] = src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}

void s_flatten_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *shape, int rank) {
    if (src == nullptr || dest == nullptr || rank < 0) return;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        dest[0] = src[0];
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        dest[el] = src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}

void s_flatten_complex128(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *shape, int rank) {
    if (src == nullptr || dest == nullptr || rank < 0) return;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        dest[0] = src[0];
        dest[1] = src[1];
        return;
    }
    const cpx_t *c_src = (const cpx_t*)src;
    cpx_t *c_dest = (cpx_t*)dest;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        c_dest[el] = c_src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}

void s_flatten_complex64(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *shape, int rank) {
    if (src == nullptr || dest == nullptr || rank < 0) return;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        dest[0] = src[0];
        dest[1] = src[1];
        return;
    }
    const cpx_f_t *c_src = (const cpx_f_t*)src;
    cpx_f_t *c_dest = (cpx_f_t*)dest;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        c_dest[el] = c_src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}

void s_flatten_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *shape, int rank) {
    if (src == nullptr || dest == nullptr || rank < 0) return;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        dest[0] = src[0];
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        dest[el] = src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}

void s_flatten_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *shape, int rank) {
    if (src == nullptr || dest == nullptr || rank < 0) return;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        dest[0] = src[0];
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        dest[el] = src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}

// ============================================================================
// 8. NATIVE C HIGH-SPEED ELEMENTS HASHING KERNELS
// ============================================================================

uint32_t s_hash_double(const double *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous) {
    if (a == nullptr) return 0;
    uint32_t hash = 2166136261U;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (is_contiguous) {
        for (int64_t i = 0; i < total_elements; i++) {
            hash_double(&hash, a[i]);
        }
        return hash;
    }
    if (rank <= 0) {
        if (rank == 0) {
            hash_double(&hash, a[0]);
        }
        return hash;
    }
    DECLARE_RANK_BUFFER_RET(int64_t, coord, rank, 0);
    int64_t offset = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        hash_double(&hash, a[offset]);
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
    return hash;
}

uint32_t s_hash_float(const float *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous) {
    if (a == nullptr) return 0;
    uint32_t hash = 2166136261U;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (is_contiguous) {
        for (int64_t i = 0; i < total_elements; i++) {
            hash_float(&hash, a[i]);
        }
        return hash;
    }
    if (rank <= 0) {
        if (rank == 0) {
            hash_float(&hash, a[0]);
        }
        return hash;
    }
    DECLARE_RANK_BUFFER_RET(int64_t, coord, rank, 0);
    int64_t offset = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        hash_float(&hash, a[offset]);
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
    return hash;
}

uint32_t s_hash_int64(const int64_t *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous) {
    if (a == nullptr) return 0;
    uint32_t hash = 2166136261U;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (is_contiguous) {
        for (int64_t i = 0; i < total_elements; i++) {
            hash_int64(&hash, a[i]);
        }
        return hash;
    }
    if (rank <= 0) {
        if (rank == 0) {
            hash_int64(&hash, a[0]);
        }
        return hash;
    }
    DECLARE_RANK_BUFFER_RET(int64_t, coord, rank, 0);
    int64_t offset = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        hash_int64(&hash, a[offset]);
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
    return hash;
}

uint32_t s_hash_int32(const int32_t *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous) {
    if (a == nullptr) return 0;
    uint32_t hash = 2166136261U;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (is_contiguous) {
        for (int64_t i = 0; i < total_elements; i++) {
            hash_int32(&hash, a[i]);
        }
        return hash;
    }
    if (rank <= 0) {
        if (rank == 0) {
            hash_int32(&hash, a[0]);
        }
        return hash;
    }
    DECLARE_RANK_BUFFER_RET(int64_t, coord, rank, 0);
    int64_t offset = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        hash_int32(&hash, a[offset]);
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
    return hash;
}

uint32_t s_hash_int16(const int16_t *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous) {
    if (a == nullptr) return 0;
    uint32_t hash = 2166136261U;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (is_contiguous) {
        for (int64_t i = 0; i < total_elements; i++) {
            hash_int16(&hash, a[i]);
        }
        return hash;
    }
    if (rank <= 0) {
        if (rank == 0) {
            hash_int16(&hash, a[0]);
        }
        return hash;
    }
    DECLARE_RANK_BUFFER_RET(int64_t, coord, rank, 0);
    int64_t offset = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        hash_int16(&hash, a[offset]);
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
    return hash;
}

uint32_t s_hash_uint8(const uint8_t *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous) {
    if (a == nullptr) return 0;
    uint32_t hash = 2166136261U;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (is_contiguous) {
        for (int64_t i = 0; i < total_elements; i++) {
            hash_uint8(&hash, a[i]);
        }
        return hash;
    }
    if (rank <= 0) {
        if (rank == 0) {
            hash_uint8(&hash, a[0]);
        }
        return hash;
    }
    DECLARE_RANK_BUFFER_RET(int64_t, coord, rank, 0);
    int64_t offset = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        hash_uint8(&hash, a[offset]);
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
    return hash;
}

uint32_t s_hash_complex128(const double *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous) {
    if (a == nullptr) return 0;
    uint32_t hash = 2166136261U;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    const cpx_t *c_a = (const cpx_t*)a;
    if (is_contiguous) {
        for (int64_t i = 0; i < total_elements; i++) {
            hash_double(&hash, c_a[i].r);
            hash_double(&hash, c_a[i].i);
        }
        return hash;
    }
    if (rank <= 0) {
        if (rank == 0) {
            hash_double(&hash, c_a[0].r);
            hash_double(&hash, c_a[0].i);
        }
        return hash;
    }
    DECLARE_RANK_BUFFER_RET(int64_t, coord, rank, 0);
    int64_t offset = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        hash_double(&hash, c_a[offset].r);
        hash_double(&hash, c_a[offset].i);
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
    return hash;
}

uint32_t s_hash_complex64(const float *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous) {
    if (a == nullptr) return 0;
    uint32_t hash = 2166136261U;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    const cpx_f_t *c_a = (const cpx_f_t*)a;
    if (is_contiguous) {
        for (int64_t i = 0; i < total_elements; i++) {
            hash_float(&hash, c_a[i].r);
            hash_float(&hash, c_a[i].i);
        }
        return hash;
    }
    if (rank <= 0) {
        if (rank == 0) {
            hash_float(&hash, c_a[0].r);
            hash_float(&hash, c_a[0].i);
        }
        return hash;
    }
    DECLARE_RANK_BUFFER_RET(int64_t, coord, rank, 0);
    int64_t offset = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        hash_float(&hash, c_a[offset].r);
        hash_float(&hash, c_a[offset].i);
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
    return hash;
}

uint32_t s_hash_boolean(const uint8_t *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous) {
    if (a == nullptr) return 0;
    uint32_t hash = 2166136261U;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (is_contiguous) {
        for (int64_t i = 0; i < total_elements; i++) {
            hash_boolean(&hash, a[i]);
        }
        return hash;
    }
    if (rank <= 0) {
        if (rank == 0) {
            hash_boolean(&hash, a[0]);
        }
        return hash;
    }
    DECLARE_RANK_BUFFER_RET(int64_t, coord, rank, 0);
    int64_t offset = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        hash_boolean(&hash, a[offset]);
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
    return hash;
}

// ============================================================================
// 9. NATIVE C HIGH-SPEED RANDOM DISTRIBUTION GENERATORS
// ============================================================================

template <typename T, typename NextDoubleFn>
static inline void v_poisson_impl(T *res, int64_t size, double lam, NextDoubleFn next_double) {
    if (res == nullptr || size <= 0) return;
    if (std::isnan(lam) || std::isinf(lam) || lam <= 0.0) {
        for (int64_t i = 0; i < size; i++) res[i] = 0;
        return;
    }

    if (lam < 30.0) {
        double limit = std::exp(-lam);
        for (int64_t i = 0; i < size; i++) {
            int64_t k = 0;
            double p = 1.0;
            do {
                k++;
                double u = next_double();
                p *= u;
            } while (p > limit);
            res[i] = saturating_float_to_int<T>((double)(k - 1));
        }
    } else {
        double stddev = std::sqrt(lam);
        int64_t i = 0;
        while (i < size) {
            double u1;
            do {
                u1 = next_double();
            } while (u1 == 0.0);

            double u2 = next_double();

            double mag = std::sqrt(-2.0 * std::log(u1));
            double angle = 2.0 * M_PI * u2;

            double z0 = mag * std::cos(angle);
            double z1 = mag * std::sin(angle);

            double val0 = lam + stddev * z0;
            res[i] = val0 < 0.0 ? 0 : saturating_float_to_int<T>(std::round(val0));

            if (i + 1 < size) {
                double val1 = lam + stddev * z1;
                res[i + 1] = val1 < 0.0 ? 0 : saturating_float_to_int<T>(std::round(val1));
            }
            i += 2;
        }
    }
}

void v_poisson_int64(int64_t *res, int64_t size, double lam, uint64_t seed) {
    uint64_t s[4];
    xoshiro256_seed(seed, s);
    v_poisson_impl<int64_t>(res, size, lam, [&s]() {
        return (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
    });
}

void v_poisson_int32(int32_t *res, int64_t size, double lam, uint64_t seed) {
    uint64_t s[4];
    xoshiro256_seed(seed, s);
    v_poisson_impl<int32_t>(res, size, lam, [&s]() {
        return (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
    });
}

template <typename T, typename NextDoubleFn>
static inline void v_binomial_impl(T *res, int64_t size, int n, double p, NextDoubleFn next_double) {
    if (res == nullptr || size <= 0) return;
    if (n <= 0 || std::isnan(p) || p <= 0.0 || p > 1.0) {
        T fill_val = (n > 0 && p == 1.0) ? saturating_float_to_int<T>((double)n) : 0;
        for (int64_t i = 0; i < size; i++) res[i] = fill_val;
        return;
    }
    if (p == 1.0) {
        T fill_val = saturating_float_to_int<T>((double)n);
        for (int64_t i = 0; i < size; i++) res[i] = fill_val;
        return;
    }

    if (n < 50) {
        for (int64_t i = 0; i < size; i++) {
            int successes = 0;
            for (int64_t t = 0; t < n; t++) {
                double u = next_double();
                if (u < p) {
                    successes++;
                }
            }
            res[i] = saturating_float_to_int<T>((double)successes);
        }
    } else {
        double mean = (double)n * p;
        double stddev = std::sqrt((double)n * p * (1.0 - p));

        if (stddev == 0.0) {
            T fill_val = saturating_float_to_int<T>(std::round(mean));
            for (int64_t i = 0; i < size; i++) {
                res[i] = fill_val;
            }
        } else {
            int64_t i = 0;
            while (i < size) {
                double u1;
                do {
                    u1 = next_double();
                } while (u1 == 0.0);

                double u2 = next_double();

                double mag = std::sqrt(-2.0 * std::log(u1));
                double angle = 2.0 * M_PI * u2;

                double z0 = mag * std::cos(angle);
                double z1 = mag * std::sin(angle);

                double val0 = std::round(mean + stddev * z0);
                if (val0 < 0.0) val0 = 0.0;
                if (val0 > (double)n) val0 = (double)n;
                res[i] = saturating_float_to_int<T>(val0);

                if (i + 1 < size) {
                    double val1 = std::round(mean + stddev * z1);
                    if (val1 < 0.0) val1 = 0.0;
                    if (val1 > (double)n) val1 = (double)n;
                    res[i + 1] = saturating_float_to_int<T>(val1);
                }
                i += 2;
            }
        }
    }
}

void v_binomial_int64(int64_t *res, int64_t size, int n, double p, uint64_t seed) {
    uint64_t s[4];
    xoshiro256_seed(seed, s);
    v_binomial_impl<int64_t>(res, size, n, p, [&s]() {
        return (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
    });
}

void v_binomial_int32(int32_t *res, int64_t size, int n, double p, uint64_t seed) {
    uint64_t s[4];
    xoshiro256_seed(seed, s);
    v_binomial_impl<int32_t>(res, size, n, p, [&s]() {
        return (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
    });
}

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

typedef LONG (WINAPI *BCryptGenRandomFunc)(
    void* hAlgorithm,
    unsigned char* pbBuffer,
    ULONG cbBuffer,
    ULONG dwFlags
);

static void fill_secure_bytes_win(void *dest, size_t size) {
    size_t offset = 0;
    HMODULE hBcrypt = LoadLibraryA("bcrypt.dll");
    if (hBcrypt != nullptr) {
        BCryptGenRandomFunc pBCryptGenRandom = (BCryptGenRandomFunc)GetProcAddress(hBcrypt, "BCryptGenRandom");
        if (pBCryptGenRandom != nullptr) {
            unsigned char *out = (unsigned char *)dest;
            while (offset < size) {
                size_t rem = size - offset;
                ULONG chunk = (rem > 0x7FFFFFFFULL) ? 0x7FFFFFFFUL : (ULONG)rem;
                if (pBCryptGenRandom(nullptr, out + offset, chunk, 0x00000002) < 0) {
                    break;
                }
                offset += chunk;
            }
        }
        FreeLibrary(hBcrypt);
    }
    if (offset < size) {
        uint64_t state = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dest)) ^ static_cast<uint64_t>(size) ^ 0x9e3779b97f4a7c15ULL;
        unsigned char *out = (unsigned char *)dest;
        while (offset < size) {
            uint64_t r = splitmix64(&state);
            size_t chunk = std::min(size - offset, sizeof(uint64_t));
            std::memcpy(out + offset, &r, chunk);
            offset += chunk;
        }
    }
}
#else
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#endif

static void fill_secure_bytes(void *dest, size_t size) {
#ifdef _WIN32
    fill_secure_bytes_win(dest, size);
#else
    size_t bytes_read = 0;
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        while (bytes_read < size) {
            ssize_t res = read(fd, (char *)dest + bytes_read, size - bytes_read);
            if (res < 0) {
                if (errno == EINTR) continue;
                break;
            }
            if (res == 0) break;
            bytes_read += static_cast<size_t>(res);
        }
        close(fd);
    }
    if (bytes_read < size) {
        uint64_t state = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dest)) ^ static_cast<uint64_t>(size) ^ 0x9e3779b97f4a7c15ULL;
        unsigned char *out = (unsigned char *)dest;
        while (bytes_read < size) {
            uint64_t r = splitmix64(&state);
            size_t chunk = std::min(size - bytes_read, sizeof(uint64_t));
            std::memcpy(out + bytes_read, &r, chunk);
            bytes_read += chunk;
        }
    }
#endif
}

struct SecureRng {
    uint64_t buf[64];
    int idx = 64;

    uint64_t next_uint64() {
        if (idx >= 64) {
            fill_secure_bytes(buf, sizeof(buf));
            idx = 0;
        }
        return buf[idx++];
    }

    double next_double() {
        return (double)(next_uint64() >> 11) * (1.0 / 9007199254740992.0);
    }
};

void v_secure_poisson_int64(int64_t *res, int64_t size, double lam) {
    SecureRng rng;
    v_poisson_impl<int64_t>(res, size, lam, [&rng]() { return rng.next_double(); });
}

void v_secure_poisson_int32(int32_t *res, int64_t size, double lam) {
    SecureRng rng;
    v_poisson_impl<int32_t>(res, size, lam, [&rng]() { return rng.next_double(); });
}

void v_secure_binomial_int64(int64_t *res, int64_t size, int n, double p) {
    SecureRng rng;
    v_binomial_impl<int64_t>(res, size, n, p, [&rng]() { return rng.next_double(); });
}

void v_secure_binomial_int32(int32_t *res, int64_t size, int n, double p) {
    SecureRng rng;
    v_binomial_impl<int32_t>(res, size, n, p, [&rng]() { return rng.next_double(); });
}

void v_secure_uniform_double(double *res, int64_t size) {
    if (res == nullptr || size <= 0) return;
    fill_secure_bytes(res, static_cast<size_t>(size) * sizeof(double));
    for (int64_t i = 0; i < size; i++) {
        uint64_t bits = 0;
        std::memcpy(&bits, &res[i], sizeof(bits));
        res[i] = (double)(bits >> 11) * (1.0 / 9007199254740992.0);
    }
}

void v_secure_uniform_float(float *res, int64_t size) {
    if (res == nullptr || size <= 0) return;
    fill_secure_bytes(res, static_cast<size_t>(size) * sizeof(float));
    for (int64_t i = 0; i < size; i++) {
        uint32_t bits = 0;
        std::memcpy(&bits, &res[i], sizeof(bits));
        res[i] = (float)((double)(bits >> 5) * (1.0 / 134217728.0));
    }
}

template <typename T, typename BoundsType, typename RangeType>
static void v_secure_randint_impl(T *res, int64_t size, BoundsType low, BoundsType high) {
    if (res == nullptr || size <= 0 || low >= high) return;
    SecureRng rng;
    uint64_t low_u = static_cast<uint64_t>(static_cast<int64_t>(low));
    uint64_t high_u = static_cast<uint64_t>(static_cast<int64_t>(high));
    uint64_t range = high_u - low_u;
    for (int64_t i = 0; i < size; i++) {
        uint64_t raw = rng.next_uint64();
        uint64_t offset = (range == 0) ? raw : (raw % range);
        res[i] = static_cast<T>(low_u + offset);
    }
}

#define IMPLEMENT_V_SECURE_RANDINT(TYPE_NAME, TYPE, BOUNDS_TYPE, RANGE_TYPE) \
void v_secure_randint_##TYPE_NAME(TYPE *res, int64_t size, BOUNDS_TYPE low, BOUNDS_TYPE high) { \
    v_secure_randint_impl<TYPE, BOUNDS_TYPE, RANGE_TYPE>(res, size, low, high); \
}

IMPLEMENT_V_SECURE_RANDINT(int64, int64_t, int64_t, int64_t)
IMPLEMENT_V_SECURE_RANDINT(int32, int32_t, int32_t, int32_t)
IMPLEMENT_V_SECURE_RANDINT(int16, int16_t, int, int)
IMPLEMENT_V_SECURE_RANDINT(uint8, uint8_t, int, int)

void v_secure_normal_double(double *res, int64_t size, double loc, double scale) {
    if (res == nullptr || size <= 0) return;
    if (std::isnan(scale) || scale < 0.0 || std::isnan(loc)) {
        for (int64_t i = 0; i < size; i++) res[i] = NAN;
        return;
    }
    if (scale == 0.0) {
        for (int64_t i = 0; i < size; i++) res[i] = loc;
        return;
    }
    v_secure_uniform_double(res, size);
    double extra_u2 = 0.5;
    if (size % 2 == 1) {
        uint64_t extra_bits = 0;
        fill_secure_bytes(&extra_bits, sizeof(extra_bits));
        extra_u2 = (double)(extra_bits >> 11) * (1.0 / 9007199254740992.0);
    }
    int64_t i = 0;
    while (i < size) {
        double u1 = res[i];
        if (u1 == 0.0) u1 = 1e-15;
        double u2 = (i + 1 < size) ? res[i + 1] : extra_u2;
        double mag = scale * std::sqrt(-2.0 * std::log(u1));
        double angle = 2.0 * M_PI * u2;
        res[i] = loc + mag * std::cos(angle);
        if (i + 1 < size) {
            res[i + 1] = loc + mag * std::sin(angle);
        }
        i += 2;
    }
}

void v_secure_normal_float(float *res, int64_t size, float loc, float scale) {
    if (res == nullptr || size <= 0) return;
    if (std::isnan(scale) || scale < 0.0f || std::isnan(loc)) {
        for (int64_t i = 0; i < size; i++) res[i] = NAN;
        return;
    }
    if (scale == 0.0f) {
        for (int64_t i = 0; i < size; i++) res[i] = loc;
        return;
    }
    v_secure_uniform_float(res, size);
    float extra_u2 = 0.5f;
    if (size % 2 == 1) {
        uint32_t extra_bits = 0;
        fill_secure_bytes(&extra_bits, sizeof(extra_bits));
        extra_u2 = (float)((double)(extra_bits >> 5) * (1.0 / 134217728.0));
    }
    int64_t i = 0;
    while (i < size) {
        float u1 = res[i];
        if (u1 == 0.0f) u1 = 1e-15f;
        float u2 = (i + 1 < size) ? res[i + 1] : extra_u2;
        float mag = scale * std::sqrt(-2.0f * std::log(u1));
        float angle = 2.0f * (float)M_PI * u2;
        res[i] = loc + mag * std::cos(angle);
        if (i + 1 < size) {
            res[i + 1] = loc + mag * std::sin(angle);
        }
        i += 2;
    }
}

void v_tril_double(const double *src, double *res, int64_t batch_count, int64_t rows, int64_t cols, int64_t k) {
    if (src == nullptr || res == nullptr || batch_count <= 0 || rows <= 0 || cols <= 0) return;
    int64_t matrix_size = rows * cols;
    for (int64_t b = 0; b < batch_count; b++) {
        const double *s_mat = src + b * matrix_size;
        double *r_mat = res + b * matrix_size;
        for (int64_t r = 0; r < rows; r++) {
            for (int64_t c = 0; c < cols; c++) {
                int64_t idx = r * cols + c;
                r_mat[idx] = (c - r <= k) ? s_mat[idx] : 0.0;
            }
        }
    }
}

void v_tril_float(const float *src, float *res, int64_t batch_count, int64_t rows, int64_t cols, int64_t k) {
    if (src == nullptr || res == nullptr || batch_count <= 0 || rows <= 0 || cols <= 0) return;
    int64_t matrix_size = rows * cols;
    for (int64_t b = 0; b < batch_count; b++) {
        const float *s_mat = src + b * matrix_size;
        float *r_mat = res + b * matrix_size;
        for (int64_t r = 0; r < rows; r++) {
            for (int64_t c = 0; c < cols; c++) {
                int64_t idx = r * cols + c;
                r_mat[idx] = (c - r <= k) ? s_mat[idx] : 0.0f;
            }
        }
    }
}

void v_triu_double(const double *src, double *res, int64_t batch_count, int64_t rows, int64_t cols, int64_t k) {
    if (src == nullptr || res == nullptr || batch_count <= 0 || rows <= 0 || cols <= 0) return;
    int64_t matrix_size = rows * cols;
    for (int64_t b = 0; b < batch_count; b++) {
        const double *s_mat = src + b * matrix_size;
        double *r_mat = res + b * matrix_size;
        for (int64_t r = 0; r < rows; r++) {
            for (int64_t c = 0; c < cols; c++) {
                int64_t idx = r * cols + c;
                r_mat[idx] = (c - r >= k) ? s_mat[idx] : 0.0;
            }
        }
    }
}

void v_triu_float(const float *src, float *res, int64_t batch_count, int64_t rows, int64_t cols, int64_t k) {
    if (src == nullptr || res == nullptr || batch_count <= 0 || rows <= 0 || cols <= 0) return;
    int64_t matrix_size = rows * cols;
    for (int64_t b = 0; b < batch_count; b++) {
        const float *s_mat = src + b * matrix_size;
        float *r_mat = res + b * matrix_size;
        for (int64_t r = 0; r < rows; r++) {
            for (int64_t c = 0; c < cols; c++) {
                int64_t idx = r * cols + c;
                r_mat[idx] = (c - r >= k) ? s_mat[idx] : 0.0f;
            }
        }
    }
}

#define DEFINE_STRIDED_CUM_OP(FUNCNAME, T, OP) \
void FUNCNAME(const T *src, const int64_t *stridesSrc, \
              T *res, const int64_t *stridesRes, \
              const int64_t *shape, int rank, int axis) { \
    strided_cum_op_impl(src, stridesSrc, res, stridesRes, shape, rank, axis, [](T a, T b) { return OP(a, b); }); \
}

// standard real ops
#define OP_ADD(x, y) ((x) + (y))
#define OP_MUL(x, y) ((x) * (y))
#define OP_ADD_I64(x, y) ((int64_t)((uint64_t)(x) + (uint64_t)(y)))
#define OP_ADD_I32(x, y) ((int32_t)((uint32_t)(x) + (uint32_t)(y)))
#define OP_ADD_I16(x, y) ((int16_t)((uint16_t)(x) + (uint16_t)(y)))
#define OP_MUL_I64(x, y) ((int64_t)((uint64_t)(x) * (uint64_t)(y)))
#define OP_MUL_I32(x, y) ((int32_t)((uint32_t)(x) * (uint32_t)(y)))
#define OP_MUL_I16(x, y) ((int16_t)((uint16_t)(x) * (uint16_t)(y)))
#define OP_MIN(x, y) (((x) < (y)) ? (x) : (y))
#define OP_MAX(x, y) (((x) > (y)) ? (x) : (y))
#define OP_MIN_FLOAT(x, y) ((std::isnan(x) || std::isnan(y)) ? ((x) + (y)) : (((x) < (y)) ? (x) : (y)))
#define OP_MAX_FLOAT(x, y) ((std::isnan(x) || std::isnan(y)) ? ((x) + (y)) : (((x) > (y)) ? (x) : (y)))

DEFINE_STRIDED_CUM_OP(s_cumsum_double, double, OP_ADD)
DEFINE_STRIDED_CUM_OP(s_cumsum_float, float, OP_ADD)
DEFINE_STRIDED_CUM_OP(s_cumsum_int64, int64_t, OP_ADD_I64)
DEFINE_STRIDED_CUM_OP(s_cumsum_int32, int32_t, OP_ADD_I32)
DEFINE_STRIDED_CUM_OP(s_cumsum_int16, int16_t, OP_ADD_I16)
DEFINE_STRIDED_CUM_OP(s_cumsum_uint8, uint8_t, OP_ADD)

DEFINE_STRIDED_CUM_OP(s_cumprod_double, double, OP_MUL)
DEFINE_STRIDED_CUM_OP(s_cumprod_float, float, OP_MUL)
DEFINE_STRIDED_CUM_OP(s_cumprod_int64, int64_t, OP_MUL_I64)
DEFINE_STRIDED_CUM_OP(s_cumprod_int32, int32_t, OP_MUL_I32)
DEFINE_STRIDED_CUM_OP(s_cumprod_int16, int16_t, OP_MUL_I16)
DEFINE_STRIDED_CUM_OP(s_cumprod_uint8, uint8_t, OP_MUL)

DEFINE_STRIDED_CUM_OP(s_cummin_double, double, OP_MIN_FLOAT)
DEFINE_STRIDED_CUM_OP(s_cummin_float, float, OP_MIN_FLOAT)
DEFINE_STRIDED_CUM_OP(s_cummin_int64, int64_t, OP_MIN)
DEFINE_STRIDED_CUM_OP(s_cummin_int32, int32_t, OP_MIN)
DEFINE_STRIDED_CUM_OP(s_cummin_int16, int16_t, OP_MIN)
DEFINE_STRIDED_CUM_OP(s_cummin_uint8, uint8_t, OP_MIN)

DEFINE_STRIDED_CUM_OP(s_cummax_double, double, OP_MAX_FLOAT)
DEFINE_STRIDED_CUM_OP(s_cummax_float, float, OP_MAX_FLOAT)
DEFINE_STRIDED_CUM_OP(s_cummax_int64, int64_t, OP_MAX)
DEFINE_STRIDED_CUM_OP(s_cummax_int32, int32_t, OP_MAX)
DEFINE_STRIDED_CUM_OP(s_cummax_int16, int16_t, OP_MAX)
DEFINE_STRIDED_CUM_OP(s_cummax_uint8, uint8_t, OP_MAX)

DEFINE_STRIDED_CUM_OP(s_cumsum_complex128, cpx_t, cpx_add)
DEFINE_STRIDED_CUM_OP(s_cumsum_complex64, cpx_f_t, cpx_add_f)
DEFINE_STRIDED_CUM_OP(s_cumprod_complex128, cpx_t, cpx_mul)
DEFINE_STRIDED_CUM_OP(s_cumprod_complex64, cpx_f_t, cpx_mul_f)

#define DEFINE_STRIDED_DIFF_OP(FUNCNAME, T, SUB_OP) \
void FUNCNAME(const T *src, const int64_t *stridesSrc, \
              T *res, const int64_t *stridesRes, \
              const int64_t *shape, int rank, int axis) { \
    strided_diff_op_impl(src, stridesSrc, res, stridesRes, shape, rank, axis, [](T a, T b) { return SUB_OP(a, b); }); \
}

#define SUB_REAL(x, y) ((x) - (y))
#define SUB_INT64(x, y) ((int64_t)((uint64_t)(x) - (uint64_t)(y)))
#define SUB_INT32(x, y) ((int32_t)((uint32_t)(x) - (uint32_t)(y)))
#define SUB_INT16(x, y) ((int16_t)((uint16_t)(x) - (uint16_t)(y)))
#define SUB_INT8(x, y) ((int8_t)((uint8_t)(x) - (uint8_t)(y)))
#define SUB_UINT64(x, y) ((uint64_t)(x) - (uint64_t)(y))
#define SUB_UINT32(x, y) ((uint32_t)(x) - (uint32_t)(y))
#define SUB_UINT16(x, y) ((uint16_t)((uint16_t)(x) - (uint16_t)(y)))
#define SUB_UINT8(x, y) ((uint8_t)((uint8_t)(x) - (uint8_t)(y)))
#define SUB_BOOL(x, y) ((((x) != 0) != ((y) != 0)) ? 1 : 0)

DEFINE_STRIDED_DIFF_OP(s_diff_double, double, SUB_REAL)
DEFINE_STRIDED_DIFF_OP(s_diff_float, float, SUB_REAL)
DEFINE_STRIDED_DIFF_OP(s_diff_int64, int64_t, SUB_INT64)
DEFINE_STRIDED_DIFF_OP(s_diff_int32, int32_t, SUB_INT32)
DEFINE_STRIDED_DIFF_OP(s_diff_int16, int16_t, SUB_INT16)
DEFINE_STRIDED_DIFF_OP(s_diff_int8, int8_t, SUB_INT8)
DEFINE_STRIDED_DIFF_OP(s_diff_uint64, uint64_t, SUB_UINT64)
DEFINE_STRIDED_DIFF_OP(s_diff_uint32, uint32_t, SUB_UINT32)
DEFINE_STRIDED_DIFF_OP(s_diff_uint16, uint16_t, SUB_UINT16)
DEFINE_STRIDED_DIFF_OP(s_diff_uint8, uint8_t, SUB_UINT8)
DEFINE_STRIDED_DIFF_OP(s_diff_boolean, uint8_t, SUB_BOOL)
DEFINE_STRIDED_DIFF_OP(s_diff_complex128, cpx_t, cpx_sub)
DEFINE_STRIDED_DIFF_OP(s_diff_complex64, cpx_f_t, cpx_sub_f)

#define DEFINE_STRIDED_UNARY_OP(FUNCNAME, T, OP) \
void FUNCNAME(const T *src, const int64_t *stridesSrc, \
              T *res, const int64_t *stridesRes, \
              const int64_t *shape, int rank, \
              const uint8_t *mask) { \
    strided_unary_op_impl(src, stridesSrc, res, stridesRes, shape, rank, mask, [](T x) { return OP(x); }); \
}

#define OP_SIN_D(x) std::sin(x)
#define OP_SIN_F(x) std::sin(x)
#define OP_COS_D(x) std::cos(x)
#define OP_COS_F(x) std::cos(x)

DEFINE_STRIDED_UNARY_OP(s_sin_double, double, OP_SIN_D)
DEFINE_STRIDED_UNARY_OP(s_sin_float, float, OP_SIN_F)
DEFINE_STRIDED_UNARY_OP(s_cos_double, double, OP_COS_D)
DEFINE_STRIDED_UNARY_OP(s_cos_float, float, OP_COS_F)

#define OP_SINC_D(x) real_sinc<double>(x)
#define OP_SINC_F(x) real_sinc<float>(x)
DEFINE_STRIDED_UNARY_OP(s_sinc_double, double, OP_SINC_D)
DEFINE_STRIDED_UNARY_OP(s_sinc_float, float, OP_SINC_F)

// complex trig helper definitions
static inline cpx_t cpx_sin(cpx_t z) {
    return (cpx_t){std::sin(z.r) * std::cosh(z.i), std::cos(z.r) * std::sinh(z.i)};
}
static inline cpx_f_t cpx_sin_f(cpx_f_t z) {
    return (cpx_f_t){std::sin(z.r) * std::cosh(z.i), std::cos(z.r) * std::sinh(z.i)};
}

static inline cpx_t cpx_sinc(cpx_t z) {
    std::complex<double> cz(z.r, z.i);
    auto cres = complex_sinc_impl(cz);
    return {cres.real(), cres.imag()};
}

static inline cpx_f_t cpx_sinc_f(cpx_f_t z) {
    std::complex<float> cz(z.r, z.i);
    auto cres = complex_sinc_impl(cz);
    return {cres.real(), cres.imag()};
}

static inline cpx_t cpx_cos(cpx_t z) {
    return (cpx_t){std::cos(z.r) * std::cosh(z.i), -std::sin(z.r) * std::sinh(z.i)};
}
static inline cpx_f_t cpx_cos_f(cpx_f_t z) {
    return (cpx_f_t){std::cos(z.r) * std::cosh(z.i), -std::sin(z.r) * std::sinh(z.i)};
}

static inline cpx_t cpx_tan(cpx_t z) {
    if (!std::isnan(z.r) && !std::isinf(z.r) && std::abs(z.i) > 350.0) {
        return (cpx_t){0.0, std::copysign(1.0, z.i)};
    }
    std::complex<double> cz(z.r, z.i);
    std::complex<double> res = std::tan(cz);
    return (cpx_t){res.real(), res.imag()};
}
static inline cpx_f_t cpx_tan_f(cpx_f_t z) {
    if (!std::isnan(z.r) && !std::isinf(z.r) && std::abs(z.i) > 40.0f) {
        return (cpx_f_t){0.0f, std::copysign(1.0f, z.i)};
    }
    std::complex<float> cz(z.r, z.i);
    std::complex<float> res = std::tan(cz);
    return (cpx_f_t){res.real(), res.imag()};
}

static inline cpx_t cpx_asin(cpx_t z) {
    std::complex<double> cz(z.r, z.i);
    std::complex<double> res = std::asin(cz);
    return (cpx_t){res.real(), res.imag()};
}

static inline cpx_f_t cpx_asin_f(cpx_f_t z) {
    std::complex<float> cz(z.r, z.i);
    std::complex<float> res = std::asin(cz);
    return (cpx_f_t){res.real(), res.imag()};
}

static inline cpx_t cpx_acos(cpx_t z) {
    std::complex<double> cz(z.r, z.i);
    std::complex<double> res = std::acos(cz);
    return (cpx_t){res.real(), res.imag()};
}

static inline cpx_f_t cpx_acos_f(cpx_f_t z) {
    std::complex<float> cz(z.r, z.i);
    std::complex<float> res = std::acos(cz);
    return (cpx_f_t){res.real(), res.imag()};
}

static inline cpx_t cpx_atan(cpx_t z) {
    std::complex<double> cz(z.r, z.i);
    std::complex<double> res = std::atan(cz);
    return (cpx_t){res.real(), res.imag()};
}

static inline cpx_f_t cpx_atan_f(cpx_f_t z) {
    std::complex<float> cz(z.r, z.i);
    std::complex<float> res = std::atan(cz);
    return (cpx_f_t){res.real(), res.imag()};
}

static inline cpx_t cpx_sinh(cpx_t z) {
    return (cpx_t){std::sinh(z.r) * std::cos(z.i), std::cosh(z.r) * std::sin(z.i)};
}
static inline cpx_f_t cpx_sinh_f(cpx_f_t z) {
    return (cpx_f_t){std::sinh(z.r) * std::cos(z.i), std::cosh(z.r) * std::sin(z.i)};
}

static inline cpx_t cpx_cosh(cpx_t z) {
    return (cpx_t){std::cosh(z.r) * std::cos(z.i), std::sinh(z.r) * std::sin(z.i)};
}
static inline cpx_f_t cpx_cosh_f(cpx_f_t z) {
    return (cpx_f_t){std::cosh(z.r) * std::cos(z.i), std::sinh(z.r) * std::sin(z.i)};
}

static inline cpx_t cpx_tanh(cpx_t z) {
    if (!std::isnan(z.i) && !std::isinf(z.i) && std::abs(z.r) > 350.0) {
        return (cpx_t){std::copysign(1.0, z.r), 0.0};
    }
    std::complex<double> cz(z.r, z.i);
    std::complex<double> res = std::tanh(cz);
    return (cpx_t){res.real(), res.imag()};
}
static inline cpx_f_t cpx_tanh_f(cpx_f_t z) {
    if (!std::isnan(z.i) && !std::isinf(z.i) && std::abs(z.r) > 40.0f) {
        return (cpx_f_t){std::copysign(1.0f, z.r), 0.0f};
    }
    std::complex<float> cz(z.r, z.i);
    std::complex<float> res = std::tanh(cz);
    return (cpx_f_t){res.real(), res.imag()};
}

static inline cpx_t cpx_asinh(cpx_t z) {
    std::complex<double> cz(z.r, z.i);
    std::complex<double> res = std::asinh(cz);
    return (cpx_t){res.real(), res.imag()};
}
static inline cpx_f_t cpx_asinh_f(cpx_f_t z) {
    std::complex<float> cz(z.r, z.i);
    std::complex<float> res = std::asinh(cz);
    return (cpx_f_t){res.real(), res.imag()};
}

static inline cpx_t cpx_log2(cpx_t z) {
    std::complex<double> cz(z.r, z.i);
    std::complex<double> res = std::log(cz) / std::log(2.0);
    return {res.real(), res.imag()};
}
static inline cpx_f_t cpx_log2_f(cpx_f_t z) {
    std::complex<float> cz(z.r, z.i);
    std::complex<float> res = std::log(cz) / std::log(2.0f);
    return {res.real(), res.imag()};
}
static inline cpx_t cpx_log10(cpx_t z) {
    std::complex<double> cz(z.r, z.i);
    std::complex<double> res = std::log(cz) / std::log(10.0);
    return {res.real(), res.imag()};
}
static inline cpx_f_t cpx_log10_f(cpx_f_t z) {
    std::complex<float> cz(z.r, z.i);
    std::complex<float> res = std::log(cz) / std::log(10.0f);
    return {res.real(), res.imag()};
}
static inline cpx_t cpx_reciprocal(cpx_t z) {
    std::complex<double> cz(z.r, z.i);
    std::complex<double> res = 1.0 / cz;
    return {res.real(), res.imag()};
}
static inline cpx_f_t cpx_reciprocal_f(cpx_f_t z) {
    std::complex<float> cz(z.r, z.i);
    std::complex<float> res = 1.0f / cz;
    return {res.real(), res.imag()};
}

static inline cpx_t cpx_exp(cpx_t z) {
    std::complex<double> cz(z.r, z.i);
    std::complex<double> cres = std::exp(cz);
    return {cres.real(), cres.imag()};
}

static inline cpx_f_t cpx_exp_f(cpx_f_t z) {
    std::complex<float> cz(z.r, z.i);
    std::complex<float> cres = std::exp(cz);
    return {cres.real(), cres.imag()};
}

static inline cpx_t cpx_log(cpx_t z) {
    std::complex<double> cz(z.r, z.i);
    std::complex<double> cres = std::log(cz);
    return {cres.real(), cres.imag()};
}

static inline cpx_f_t cpx_log_f(cpx_f_t z) {
    std::complex<float> cz(z.r, z.i);
    std::complex<float> cres = std::log(cz);
    return {cres.real(), cres.imag()};
}

static inline cpx_t cpx_sqrt(cpx_t z) {
    std::complex<double> cz(z.r, z.i);
    std::complex<double> cres = std::sqrt(cz);
    return {cres.real(), cres.imag()};
}

static inline cpx_f_t cpx_sqrt_f(cpx_f_t z) {
    std::complex<float> cz(z.r, z.i);
    std::complex<float> cres = std::sqrt(cz);
    return {cres.real(), cres.imag()};
}

static inline cpx_t cpx_acosh(cpx_t z) {
    std::complex<double> cz(z.r, z.i);
    std::complex<double> cres = std::acosh(cz);
    return {cres.real(), cres.imag()};
}

static inline cpx_f_t cpx_acosh_f(cpx_f_t z) {
    std::complex<float> cz(z.r, z.i);
    std::complex<float> cres = std::acosh(cz);
    return {cres.real(), cres.imag()};
}

static inline cpx_t cpx_expm1(cpx_t z) {
    if (z.i == 0.0) {
        return {std::expm1(z.r), 0.0};
    }
    double expm1_a = std::expm1(z.r);
    double sin_b_2 = std::sin(z.i / 2.0);
    double real_part = expm1_a * std::cos(z.i) - 2.0 * sin_b_2 * sin_b_2;
    double imag_part = std::exp(z.r) * std::sin(z.i);
    return {real_part, imag_part};
}

static inline cpx_f_t cpx_expm1_f(cpx_f_t z) {
    if (z.i == 0.0f) {
        return {std::expm1(z.r), 0.0f};
    }
    float expm1_a = std::expm1(z.r);
    float sin_b_2 = std::sin(z.i / 2.0f);
    float real_part = expm1_a * std::cos(z.i) - 2.0f * sin_b_2 * sin_b_2;
    float imag_part = std::exp(z.r) * std::sin(z.i);
    return {real_part, imag_part};
}

static inline cpx_t cpx_log1p(cpx_t z) {
    double r2 = z.r * z.r;
    double i2 = z.i * z.i;
    double u = 2.0 * z.r + r2 + i2;
    double real_part;
    if (std::abs(u) < 0.5) {
        real_part = 0.5 * std::log1p(u);
    } else {
        real_part = std::log(std::hypot(1.0 + z.r, z.i));
    }
    double imag_part = std::atan2(z.i, 1.0 + z.r);
    return {real_part, imag_part};
}

static inline cpx_f_t cpx_log1p_f(cpx_f_t z) {
    float r2 = z.r * z.r;
    float i2 = z.i * z.i;
    float u = 2.0f * z.r + r2 + i2;
    float real_part;
    if (std::abs(u) < 0.5f) {
        real_part = 0.5f * std::log1p(u);
    } else {
        real_part = std::log(std::hypot(1.0f + z.r, z.i));
    }
    float imag_part = std::atan2(z.i, 1.0f + z.r);
    return {real_part, imag_part};
}

#define DEFINE_UNARY_CAST_VEC(FUNCNAME, T_IN, T_OUT, OP) \
void FUNCNAME(const T_IN *src, T_OUT *res, int64_t size, const uint8_t *mask) { \
    v_unary_cast_impl<T_IN, T_OUT>(src, res, size, mask, [](T_IN x) { return OP(x); }); \
}

#define DEFINE_STRIDED_UNARY_CAST_OP(FUNCNAME, T_IN, T_OUT, OP) \
void FUNCNAME(const T_IN *src, const int64_t *stridesSrc, \
              T_OUT *res, const int64_t *stridesRes, \
              const int64_t *shape, int rank, [[maybe_unused]] const uint8_t *mask) { \
    strided_unary_cast_impl<T_IN, T_OUT>(src, stridesSrc, res, stridesRes, shape, rank, mask, [](T_IN x) { return OP(x); }); \
}

#define DEFINE_COMPLEX_UNARY_VEC(FUNCNAME, T, OP) \
void FUNCNAME(const T *src, T *res, int64_t size, const uint8_t *mask) { \
    v_unary_impl(src, res, size, mask, [](T x) { return OP(x); }); \
}

DEFINE_COMPLEX_UNARY_VEC(v_expm1_complex128, cpx_t, cpx_expm1)
DEFINE_COMPLEX_UNARY_VEC(v_expm1_complex64, cpx_f_t, cpx_expm1_f)
DEFINE_COMPLEX_UNARY_VEC(v_log1p_complex128, cpx_t, cpx_log1p)
DEFINE_COMPLEX_UNARY_VEC(v_log1p_complex64, cpx_f_t, cpx_log1p_f)
DEFINE_COMPLEX_UNARY_VEC(v_sin_complex128, cpx_t, cpx_sin)
DEFINE_COMPLEX_UNARY_VEC(v_sin_complex64, cpx_f_t, cpx_sin_f)
DEFINE_COMPLEX_UNARY_VEC(v_sinc_complex128, cpx_t, cpx_sinc)
DEFINE_COMPLEX_UNARY_VEC(v_sinc_complex64, cpx_f_t, cpx_sinc_f)
DEFINE_COMPLEX_UNARY_VEC(v_cos_complex128, cpx_t, cpx_cos)
DEFINE_COMPLEX_UNARY_VEC(v_cos_complex64, cpx_f_t, cpx_cos_f)
DEFINE_COMPLEX_UNARY_VEC(v_tan_complex128, cpx_t, cpx_tan)
DEFINE_COMPLEX_UNARY_VEC(v_tan_complex64, cpx_f_t, cpx_tan_f)

DEFINE_COMPLEX_UNARY_VEC(v_asin_complex128, cpx_t, cpx_asin)
DEFINE_COMPLEX_UNARY_VEC(v_asin_complex64, cpx_f_t, cpx_asin_f)
DEFINE_COMPLEX_UNARY_VEC(v_acos_complex128, cpx_t, cpx_acos)
DEFINE_COMPLEX_UNARY_VEC(v_acos_complex64, cpx_f_t, cpx_acos_f)
DEFINE_COMPLEX_UNARY_VEC(v_atan_complex128, cpx_t, cpx_atan)
DEFINE_COMPLEX_UNARY_VEC(v_atan_complex64, cpx_f_t, cpx_atan_f)

DEFINE_COMPLEX_UNARY_VEC(v_sinh_complex128, cpx_t, cpx_sinh)
DEFINE_COMPLEX_UNARY_VEC(v_sinh_complex64, cpx_f_t, cpx_sinh_f)
DEFINE_COMPLEX_UNARY_VEC(v_cosh_complex128, cpx_t, cpx_cosh)
DEFINE_COMPLEX_UNARY_VEC(v_cosh_complex64, cpx_f_t, cpx_cosh_f)
DEFINE_COMPLEX_UNARY_VEC(v_tanh_complex128, cpx_t, cpx_tanh)
DEFINE_COMPLEX_UNARY_VEC(v_tanh_complex64, cpx_f_t, cpx_tanh_f)
DEFINE_COMPLEX_UNARY_VEC(v_asinh_complex128, cpx_t, cpx_asinh)
DEFINE_COMPLEX_UNARY_VEC(v_asinh_complex64, cpx_f_t, cpx_asinh_f)
DEFINE_COMPLEX_UNARY_VEC(v_exp_complex128, cpx_t, cpx_exp)
DEFINE_COMPLEX_UNARY_VEC(v_exp_complex64, cpx_f_t, cpx_exp_f)
DEFINE_COMPLEX_UNARY_VEC(v_log_complex128, cpx_t, cpx_log)
DEFINE_COMPLEX_UNARY_VEC(v_log_complex64, cpx_f_t, cpx_log_f)
DEFINE_COMPLEX_UNARY_VEC(v_sqrt_complex128, cpx_t, cpx_sqrt)
DEFINE_COMPLEX_UNARY_VEC(v_sqrt_complex64, cpx_f_t, cpx_sqrt_f)
DEFINE_COMPLEX_UNARY_VEC(v_acosh_complex128, cpx_t, cpx_acosh)
DEFINE_COMPLEX_UNARY_VEC(v_acosh_complex64, cpx_f_t, cpx_acosh_f)

DEFINE_STRIDED_UNARY_OP(s_expm1_complex128, cpx_t, cpx_expm1)
DEFINE_STRIDED_UNARY_OP(s_expm1_complex64, cpx_f_t, cpx_expm1_f)
DEFINE_STRIDED_UNARY_OP(s_log1p_complex128, cpx_t, cpx_log1p)
DEFINE_STRIDED_UNARY_OP(s_log1p_complex64, cpx_f_t, cpx_log1p_f)

DEFINE_STRIDED_UNARY_OP(s_sin_complex128, cpx_t, cpx_sin)
DEFINE_STRIDED_UNARY_OP(s_sin_complex64, cpx_f_t, cpx_sin_f)
DEFINE_STRIDED_UNARY_OP(s_sinc_complex128, cpx_t, cpx_sinc)
DEFINE_STRIDED_UNARY_OP(s_sinc_complex64, cpx_f_t, cpx_sinc_f)
DEFINE_STRIDED_UNARY_OP(s_cos_complex128, cpx_t, cpx_cos)
DEFINE_STRIDED_UNARY_OP(s_cos_complex64, cpx_f_t, cpx_cos_f)
DEFINE_STRIDED_UNARY_OP(s_tan_complex128, cpx_t, cpx_tan)
DEFINE_STRIDED_UNARY_OP(s_tan_complex64, cpx_f_t, cpx_tan_f)

DEFINE_STRIDED_UNARY_OP(s_asin_complex128, cpx_t, cpx_asin)
DEFINE_STRIDED_UNARY_OP(s_asin_complex64, cpx_f_t, cpx_asin_f)
DEFINE_STRIDED_UNARY_OP(s_acos_complex128, cpx_t, cpx_acos)
DEFINE_STRIDED_UNARY_OP(s_acos_complex64, cpx_f_t, cpx_acos_f)
DEFINE_STRIDED_UNARY_OP(s_atan_complex128, cpx_t, cpx_atan)
DEFINE_STRIDED_UNARY_OP(s_atan_complex64, cpx_f_t, cpx_atan_f)

DEFINE_STRIDED_UNARY_OP(s_sinh_complex128, cpx_t, cpx_sinh)
DEFINE_STRIDED_UNARY_OP(s_sinh_complex64, cpx_f_t, cpx_sinh_f)
DEFINE_STRIDED_UNARY_OP(s_cosh_complex128, cpx_t, cpx_cosh)
DEFINE_STRIDED_UNARY_OP(s_cosh_complex64, cpx_f_t, cpx_cosh_f)
DEFINE_STRIDED_UNARY_OP(s_tanh_complex128, cpx_t, cpx_tanh)
DEFINE_STRIDED_UNARY_OP(s_tanh_complex64, cpx_f_t, cpx_tanh_f)
DEFINE_STRIDED_UNARY_OP(s_asinh_complex128, cpx_t, cpx_asinh)
DEFINE_STRIDED_UNARY_OP(s_asinh_complex64, cpx_f_t, cpx_asinh_f)
DEFINE_STRIDED_UNARY_OP(s_exp_complex128, cpx_t, cpx_exp)
DEFINE_STRIDED_UNARY_OP(s_exp_complex64, cpx_f_t, cpx_exp_f)
DEFINE_STRIDED_UNARY_OP(s_log_complex128, cpx_t, cpx_log)
DEFINE_STRIDED_UNARY_OP(s_log_complex64, cpx_f_t, cpx_log_f)
DEFINE_STRIDED_UNARY_OP(s_sqrt_complex128, cpx_t, cpx_sqrt)
DEFINE_STRIDED_UNARY_OP(s_sqrt_complex64, cpx_f_t, cpx_sqrt_f)
DEFINE_STRIDED_UNARY_OP(s_acosh_complex128, cpx_t, cpx_acosh)
DEFINE_STRIDED_UNARY_OP(s_acosh_complex64, cpx_f_t, cpx_acosh_f)

static inline double cpx_abs(cpx_t z) {
    return std::hypot(z.r, z.i);
}

static inline float cpx_abs_f(cpx_f_t z) {
    return std::hypot(z.r, z.i);
}

void v_abs_complex128(const cpx_t *src, double *res, int64_t size, const uint8_t *mask) {
    if (src == nullptr || res == nullptr || size <= 0) return;
    v_unary_cast_impl(src, res, size, mask, cpx_abs);
}

void v_abs_complex64(const cpx_f_t *src, float *res, int64_t size, const uint8_t *mask) {
    if (src == nullptr || res == nullptr || size <= 0) return;
    v_unary_cast_impl(src, res, size, mask, cpx_abs_f);
}

void s_abs_complex128(const cpx_t *src, const int64_t *stridesSrc,
                      double *res, const int64_t *stridesRes,
                      const int64_t *shape, int rank, const uint8_t *mask) {
    strided_unary_cast_impl(src, stridesSrc, res, stridesRes, shape, rank, mask, cpx_abs);
}

void s_abs_complex64(const cpx_f_t *src, const int64_t *stridesSrc,
                     float *res, const int64_t *stridesRes,
                     const int64_t *shape, int rank, const uint8_t *mask) {
    strided_unary_cast_impl(src, stridesSrc, res, stridesRes, shape, rank, mask, cpx_abs_f);
}

static inline double cpx_angle(cpx_t z) {
    return std::atan2(z.i, z.r);
}

static inline float cpx_angle_f(cpx_f_t z) {
    return std::atan2(z.i, z.r);
}

DEFINE_UNARY_CAST_VEC(v_angle_complex128, cpx_t, double, cpx_angle)
DEFINE_UNARY_CAST_VEC(v_angle_complex64, cpx_f_t, float, cpx_angle_f)

DEFINE_STRIDED_UNARY_CAST_OP(s_angle_complex128, cpx_t, double, cpx_angle)
DEFINE_STRIDED_UNARY_CAST_OP(s_angle_complex64, cpx_f_t, float, cpx_angle_f)

void s_unwrap_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis, double discont) {
    strided_unwrap_op_impl(src, stridesSrc, res, stridesRes, shape, rank, axis, discont);
}

void s_unwrap_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis, float discont) {
    strided_unwrap_op_impl(src, stridesSrc, res, stridesRes, shape, rank, axis, discont);
}

#define OP_ASIN_D(x) std::asin(x)
#define OP_ASIN_F(x) std::asin(x)
#define OP_ACOS_D(x) std::acos(x)
#define OP_ACOS_F(x) std::acos(x)
#define OP_ATAN_D(x) std::atan(x)
#define OP_ATAN_F(x) std::atan(x)

DEFINE_STRIDED_UNARY_OP(s_asin_double, double, OP_ASIN_D)
DEFINE_STRIDED_UNARY_OP(s_asin_float, float, OP_ASIN_F)
DEFINE_STRIDED_UNARY_OP(s_acos_double, double, OP_ACOS_D)
DEFINE_STRIDED_UNARY_OP(s_acos_float, float, OP_ACOS_F)
DEFINE_STRIDED_UNARY_OP(s_atan_double, double, OP_ATAN_D)
DEFINE_STRIDED_UNARY_OP(s_atan_float, float, OP_ATAN_F)

void s_atan2_double(const double *y, const int64_t *stridesY,
                   const double *x, const int64_t *stridesX,
                   double *res, const int64_t *stridesRes,
                   const int64_t *shape, int rank, const uint8_t *mask) {
    strided_binary_op_impl(y, stridesY, x, stridesX, res, stridesRes, shape, rank, mask, [](double yv, double xv) { return std::atan2(yv, xv); });
}

void s_atan2_float(const float *y, const int64_t *stridesY,
                  const float *x, const int64_t *stridesX,
                  float *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank, const uint8_t *mask) {
    strided_binary_op_impl(y, stridesY, x, stridesX, res, stridesRes, shape, rank, mask, [](float yv, float xv) { return std::atan2(yv, xv); });
}

#define OP_TAN_D(x) std::tan(x)
#define OP_TAN_F(x) std::tan(x)
#define OP_EXP_D(x) std::exp(x)
#define OP_EXP_F(x) std::exp(x)
#define OP_LOG_D(x) std::log(x)
#define OP_LOG_F(x) std::log(x)

DEFINE_STRIDED_UNARY_OP(s_tan_double, double, OP_TAN_D)
DEFINE_STRIDED_UNARY_OP(s_tan_float, float, OP_TAN_F)
DEFINE_STRIDED_UNARY_OP(s_exp_double, double, OP_EXP_D)
DEFINE_STRIDED_UNARY_OP(s_exp_float, float, OP_EXP_F)
DEFINE_STRIDED_UNARY_OP(s_log_double, double, OP_LOG_D)
DEFINE_STRIDED_UNARY_OP(s_log_float, float, OP_LOG_F)

#define OP_SINH_D(x) std::sinh(x)
#define OP_SINH_F(x) std::sinh(x)
#define OP_COSH_D(x) std::cosh(x)
#define OP_COSH_F(x) std::cosh(x)
#define OP_TANH_D(x) std::tanh(x)
#define OP_TANH_F(x) std::tanh(x)

DEFINE_STRIDED_UNARY_OP(s_sinh_double, double, OP_SINH_D)
DEFINE_STRIDED_UNARY_OP(s_sinh_float, float, OP_SINH_F)
DEFINE_STRIDED_UNARY_OP(s_cosh_double, double, OP_COSH_D)
DEFINE_STRIDED_UNARY_OP(s_cosh_float, float, OP_COSH_F)
DEFINE_STRIDED_UNARY_OP(s_tanh_double, double, OP_TANH_D)
DEFINE_STRIDED_UNARY_OP(s_tanh_float, float, OP_TANH_F)

#define OP_ASINH_D(x) std::asinh(x)
#define OP_ASINH_F(x) std::asinh(x)
#define OP_ACOSH_D(x) std::acosh(x)
#define OP_ACOSH_F(x) std::acosh(x)
#define OP_ATANH_D(x) std::atanh(x)
#define OP_ATANH_F(x) std::atanh(x)

DEFINE_STRIDED_UNARY_OP(s_asinh_double, double, OP_ASINH_D)
DEFINE_STRIDED_UNARY_OP(s_asinh_float, float, OP_ASINH_F)
DEFINE_STRIDED_UNARY_OP(s_acosh_double, double, OP_ACOSH_D)
DEFINE_STRIDED_UNARY_OP(s_acosh_float, float, OP_ACOSH_F)
DEFINE_STRIDED_UNARY_OP(s_atanh_double, double, OP_ATANH_D)
DEFINE_STRIDED_UNARY_OP(s_atanh_float, float, OP_ATANH_F)

static inline cpx_t cpx_atanh(cpx_t z) {
    std::complex<double> cz(z.r, z.i);
    std::complex<double> res = std::atanh(cz);
    return (cpx_t){res.real(), res.imag()};
}

static inline cpx_f_t cpx_atanh_f(cpx_f_t z) {
    std::complex<float> cz(z.r, z.i);
    std::complex<float> res = std::atanh(cz);
    return (cpx_f_t){res.real(), res.imag()};
}

DEFINE_COMPLEX_UNARY_VEC(v_atanh_complex128, cpx_t, cpx_atanh)
DEFINE_COMPLEX_UNARY_VEC(v_atanh_complex64, cpx_f_t, cpx_atanh_f)
DEFINE_STRIDED_UNARY_OP(s_atanh_complex128, cpx_t, cpx_atanh)
DEFINE_STRIDED_UNARY_OP(s_atanh_complex64, cpx_f_t, cpx_atanh_f)

void v_hypot_complex128(const cpx_t *x1, const cpx_t *x2, double *res,int64_t size, const uint8_t *mask) {
    v_binary_impl(x1, x2, res, size, mask, [](cpx_t a, cpx_t b) {
        return std::hypot(std::hypot(a.r, a.i), std::hypot(b.r, b.i));
    });
}

void v_hypot_complex64(const cpx_f_t *x1, const cpx_f_t *x2, float *res,int64_t size, const uint8_t *mask) {
    v_binary_impl(x1, x2, res, size, mask, [](cpx_f_t a, cpx_f_t b) {
        return std::hypot(std::hypot(a.r, a.i), std::hypot(b.r, b.i));
    });
}

void s_hypot_complex128(const cpx_t *x1, const int64_t *stridesX1, const cpx_t *x2, const int64_t *stridesX2, double *res, const int64_t *stridesRes, const int64_t *shape,int rank, const uint8_t *mask) {
    strided_binary_op_impl(x1, stridesX1, x2, stridesX2, res, stridesRes, shape, rank, mask, [](cpx_t a, cpx_t b) {
        return std::hypot(std::hypot(a.r, a.i), std::hypot(b.r, b.i));
    });
}

void s_hypot_complex64(const cpx_f_t *x1, const int64_t *stridesX1, const cpx_f_t *x2, const int64_t *stridesX2, float *res, const int64_t *stridesRes, const int64_t *shape,int rank, const uint8_t *mask) {
    strided_binary_op_impl(x1, stridesX1, x2, stridesX2, res, stridesRes, shape, rank, mask, [](cpx_f_t a, cpx_f_t b) {
        return std::hypot(std::hypot(a.r, a.i), std::hypot(b.r, b.i));
    });
}

static inline cpx_t cpx_pow(cpx_t z1, cpx_t z2) {
    if (z2.r == 0.0 && z2.i == 0.0) return (cpx_t){1.0, 0.0};
    double mag = std::hypot(z1.r, z1.i);
    if (mag == 0.0) {
        if (z2.r > 0.0 && z2.i == 0.0) return (cpx_t){0.0, 0.0};
        return (cpx_t){NAN, NAN};
    }
    double L = std::log(mag);
    double theta = std::atan2(z1.i, z1.r);
    double R = z2.r * L - z2.i * theta;
    double imag_val = z2.i * L + z2.r * theta;
    double eR = std::exp(R);
    return (cpx_t){eR * std::cos(imag_val), eR * std::sin(imag_val)};
}

static inline cpx_f_t cpx_pow_f(cpx_f_t z1, cpx_f_t z2) {
    if (z2.r == 0.0f && z2.i == 0.0f) return (cpx_f_t){1.0f, 0.0f};
    float mag = std::hypot(z1.r, z1.i);
    if (mag == 0.0f) {
        if (z2.r > 0.0f && z2.i == 0.0f) return (cpx_f_t){0.0f, 0.0f};
        return (cpx_f_t){NAN, NAN};
    }
    float L = std::log(mag);
    float theta = std::atan2(z1.i, z1.r);
    float R = z2.r * L - z2.i * theta;
    float imag_val = z2.i * L + z2.r * theta;
    float eR = std::exp(R);
    return (cpx_f_t){eR * std::cos(imag_val), eR * std::sin(imag_val)};
}


void v_pow_complex128(const cpx_t *x1, const cpx_t *x2, cpx_t *res,int64_t size, const uint8_t *mask) {
    v_binary_impl(x1, x2, res, size, mask, cpx_pow);
}

void v_pow_complex64(const cpx_f_t *x1, const cpx_f_t *x2, cpx_f_t *res,int64_t size, const uint8_t *mask) {
    v_binary_impl(x1, x2, res, size, mask, cpx_pow_f);
}

void s_pow_complex128(const cpx_t *x1, const int64_t *stridesX1, const cpx_t *x2, const int64_t *stridesX2, cpx_t *res, const int64_t *stridesRes, const int64_t *shape,int rank, const uint8_t *mask) {
    strided_binary_op_impl(x1, stridesX1, x2, stridesX2, res, stridesRes, shape, rank, mask, cpx_pow);
}

void s_pow_complex64(const cpx_f_t *x1, const int64_t *stridesX1, const cpx_f_t *x2, const int64_t *stridesX2, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape,int rank, const uint8_t *mask) {
    strided_binary_op_impl(x1, stridesX1, x2, stridesX2, res, stridesRes, shape, rank, mask, cpx_pow_f);
}

static inline cpx_t cpx_conj_op(cpx_t z) { return (cpx_t){z.r, -z.i}; }
static inline cpx_f_t cpx_conj_f_op(cpx_f_t z) { return (cpx_f_t){z.r, -z.i}; }

void v_conj_complex128(const cpx_t *src, cpx_t *res,int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, cpx_conj_op);
}

void v_conj_complex64(const cpx_f_t *src, cpx_f_t *res,int64_t size, const uint8_t *mask) {
    v_unary_impl(src, res, size, mask, cpx_conj_f_op);
}

void s_conj_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape,int rank, const uint8_t *mask) {
    strided_unary_op_impl(src, stridesSrc, res, stridesRes, shape, rank, mask, cpx_conj_op);
}

void s_conj_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape,int rank, const uint8_t *mask) {
    strided_unary_op_impl(src, stridesSrc, res, stridesRes, shape, rank, mask, cpx_conj_f_op);
}

/* ============================================================================
 * CROSS-TYPE BINARY MATH FFI HELPERS IMPLEMENTATION
 * ============================================================================
 */

static inline cpx_t cpx_from_double(double v) { return (cpx_t){v, 0.0}; }
static inline cpx_t cpx_from_float(float v) { return (cpx_t){(double)v, 0.0}; }
static inline cpx_t cpx_from_int64(int64_t v) { return (cpx_t){(double)v, 0.0}; }
static inline cpx_t cpx_from_int32(int32_t v) { return (cpx_t){(double)v, 0.0}; }
static inline cpx_t cpx_from_uint8(uint8_t v) { return (cpx_t){(double)v, 0.0}; }
static inline cpx_t cpx_from_int16(int16_t v) { return (cpx_t){(double)v, 0.0}; }
static inline cpx_t cpx_from_cpx(cpx_t v) { return v; }

static inline cpx_t cpx_from_cpx64(cpx_f_t v) { return (cpx_t){(double)v.r, (double)v.i}; }

static inline cpx_f_t cpx_f_from_cpx(cpx_t v) { return (cpx_f_t){(float)v.r, (float)v.i}; }


#define EXPR_double(OP, Ta, Tb, x, y, OP_SYM) ((double)x OP_SYM (double)y)
#define EXPR_float(OP, Ta, Tb, x, y, OP_SYM) ((float)x OP_SYM (float)y)
#define EXPR_int64(OP, Ta, Tb, x, y, OP_SYM) ((int64_t)((uint64_t)(int64_t)(x) OP_SYM (uint64_t)(int64_t)(y)))
#define EXPR_int32(OP, Ta, Tb, x, y, OP_SYM) ((int32_t)((uint32_t)(int32_t)(x) OP_SYM (uint32_t)(int32_t)(y)))
#define EXPR_int16(OP, Ta, Tb, x, y, OP_SYM) ((int16_t)((uint32_t)(uint16_t)(int16_t)(x) OP_SYM (uint32_t)(uint16_t)(int16_t)(y)))
#define EXPR_uint8(OP, Ta, Tb, x, y, OP_SYM) ((uint8_t)((uint32_t)(uint8_t)(x) OP_SYM (uint32_t)(uint8_t)(y)))
#define EXPR_cpx(OP, Ta, Tb, x, y, OP_SYM) cpx_##OP(cpx_from_##Ta(x), cpx_from_##Tb(y))
#define EXPR_cpx64(OP, Ta, Tb, x, y, OP_SYM) cpx_f_from_cpx(cpx_##OP(cpx_from_##Ta(x), cpx_from_##Tb(y)))

template <typename Ta, typename Tb, typename Tr, typename Lambda>
static inline void v_dispatch_add(const Ta *a, const Tb *b, Tr *res, int64_t size, const uint8_t *mask, [[maybe_unused]] Lambda fn) {
    v_binary_impl(a, b, res, size, mask, fn);
}
template <typename Lambda>
static inline void v_dispatch_add(const cpx_t *a, const cpx_t *b, cpx_t *res, int64_t size, const uint8_t *mask, [[maybe_unused]] Lambda fn) {
    v_add_complex(a, b, res, size, mask);
}
template <typename Lambda>
static inline void v_dispatch_add(const cpx_f_t *a, const cpx_f_t *b, cpx_f_t *res, int64_t size, const uint8_t *mask, [[maybe_unused]] Lambda fn) {
    v_add_complex64(a, b, res, size, mask);
}

template <typename Ta, typename Tb, typename Tr, typename Lambda>
static inline void v_dispatch_sub(const Ta *a, const Tb *b, Tr *res, int64_t size, const uint8_t *mask, [[maybe_unused]] Lambda fn) {
    v_binary_impl(a, b, res, size, mask, fn);
}
template <typename Lambda>
static inline void v_dispatch_sub(const cpx_t *a, const cpx_t *b, cpx_t *res, int64_t size, const uint8_t *mask, [[maybe_unused]] Lambda fn) {
    v_sub_complex(a, b, res, size, mask);
}
template <typename Lambda>
static inline void v_dispatch_sub(const cpx_f_t *a, const cpx_f_t *b, cpx_f_t *res, int64_t size, const uint8_t *mask, [[maybe_unused]] Lambda fn) {
    v_sub_complex64(a, b, res, size, mask);
}

template <typename Ta, typename Tb, typename Tr, typename Lambda>
static inline void v_dispatch_mul(const Ta *a, const Tb *b, Tr *res, int64_t size, const uint8_t *mask, [[maybe_unused]] Lambda fn) {
    v_binary_impl(a, b, res, size, mask, fn);
}
template <typename Lambda>
static inline void v_dispatch_mul(const cpx_t *a, const cpx_t *b, cpx_t *res, int64_t size, const uint8_t *mask, [[maybe_unused]] Lambda fn) {
    v_mul_complex(a, b, res, size, mask);
}
template <typename Lambda>
static inline void v_dispatch_mul(const cpx_f_t *a, const cpx_f_t *b, cpx_f_t *res, int64_t size, const uint8_t *mask, [[maybe_unused]] Lambda fn) {
    v_mul_complex64(a, b, res, size, mask);
}

template <typename Ta, typename Tb, typename Tr, typename Lambda>
static inline void v_dispatch_div(const Ta *a, const Tb *b, Tr *res, int64_t size, const uint8_t *mask, [[maybe_unused]] Lambda fn) {
    v_binary_impl(a, b, res, size, mask, fn);
}
template <typename Lambda>
static inline void v_dispatch_div(const cpx_t *a, const cpx_t *b, cpx_t *res, int64_t size, const uint8_t *mask, [[maybe_unused]] Lambda fn) {
    v_div_complex(a, b, res, size, mask);
}
template <typename Lambda>
static inline void v_dispatch_div(const cpx_f_t *a, const cpx_f_t *b, cpx_f_t *res, int64_t size, const uint8_t *mask, [[maybe_unused]] Lambda fn) {
    v_div_complex64(a, b, res, size, mask);
}

#define DEFINE_V_UFUNC(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, OP_SYM) \
void v_##OP##_##Ta_tok##_##Tb_tok##_##Tr_tok(const Ta *a, const Tb *b, Tr *res, int64_t size, const uint8_t *mask) { \
    v_dispatch_##OP(a, b, res, size, mask, [](Ta x, Tb y) { \
        return EXPR_##Tr_tok(OP, Ta_tok, Tb_tok, x, y, OP_SYM); \
    }); \
}

#define DEFINE_S_UFUNC(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, OP_SYM) \
void s_##OP##_##Ta_tok##_##Tb_tok##_##Tr_tok(const Ta *a, const int64_t *stridesA, \
                                             const Tb *b, const int64_t *stridesB, \
                                             Tr *res, const int64_t *stridesRes, \
                                             const int64_t *shape, int rank, \
                                             const uint8_t *mask) { \
    strided_binary_op_impl(a, stridesA, b, stridesB, res, stridesRes, shape, rank, mask, [](Ta x, Tb y) { \
        return EXPR_##Tr_tok(OP, Ta_tok, Tb_tok, x, y, OP_SYM); \
    }); \
}

#define DEFINE_V_S_UFUNC(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, OP_SYM) \
  DEFINE_V_UFUNC(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, OP_SYM) \
  DEFINE_S_UFUNC(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, OP_SYM)

#define BUILD_ADD_COMBINATIONS(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr) \
  DEFINE_V_S_UFUNC(add, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, +)

#define BUILD_SUB_COMBINATIONS(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr) \
  DEFINE_V_S_UFUNC(sub, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, -)

#define BUILD_MUL_COMBINATIONS(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr) \
  DEFINE_V_S_UFUNC(mul, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, *)

#define BUILD_DIV_COMBINATIONS(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr) \
  DEFINE_V_S_UFUNC(div, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, /)

GENERATE_COMMUTATIVE_COMBINATIONS(add, BUILD_ADD_COMBINATIONS)
GENERATE_OP_COMBINATIONS(sub, BUILD_SUB_COMBINATIONS)
GENERATE_COMMUTATIVE_COMBINATIONS(mul, BUILD_MUL_COMBINATIONS)
GENERATE_DIV_COMBINATIONS(div, BUILD_DIV_COMBINATIONS)

void cast_uint8_to_double(const uint8_t *src, double *dst, int64_t size) {
    if (src == nullptr || dst == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        dst[i] = (double)src[i];
    }
}

void cast_int16_to_double(const int16_t *src, double *dst, int64_t size) {
    if (src == nullptr || dst == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        dst[i] = (double)src[i];
    }
}

void cast_double_to_uint8(const double *src, uint8_t *dst, int64_t size) {
    if (src == nullptr || dst == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        dst[i] = saturating_float_to_int<uint8_t>(src[i]);
    }
}

void cast_double_to_int16(const double *src, int16_t *dst, int64_t size) {
    if (src == nullptr || dst == nullptr || size <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        dst[i] = saturating_float_to_int<int16_t>(src[i]);
    }
}

void s_cast_uint8_to_double(const uint8_t *src, const int64_t *stridesSrc, double *dst, const int64_t *stridesDst, const int64_t *shape, int rank) {
    if (src == nullptr || dst == nullptr || (rank > 0 && shape == nullptr) || rank < 0) return;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0, offsetDst = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        dst[offsetDst] = (double)src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDst += stridesDst[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDst -= (shape[d] - 1) * stridesDst[d];
        }
    }
}

void s_cast_int16_to_double(const int16_t *src, const int64_t *stridesSrc, double *dst, const int64_t *stridesDst, const int64_t *shape, int rank) {
    if (src == nullptr || dst == nullptr || (rank > 0 && shape == nullptr) || rank < 0) return;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0, offsetDst = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        dst[offsetDst] = (double)src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDst += stridesDst[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDst -= (shape[d] - 1) * stridesDst[d];
        }
    }
}

void s_cast_double_to_uint8(const double *src, const int64_t *stridesSrc, uint8_t *dst, const int64_t *stridesDst, const int64_t *shape, int rank) {
    if (src == nullptr || dst == nullptr || (rank > 0 && shape == nullptr) || rank < 0) return;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0, offsetDst = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        dst[offsetDst] = saturating_float_to_int<uint8_t>(src[offsetSrc]);
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDst += stridesDst[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDst -= (shape[d] - 1) * stridesDst[d];
        }
    }
}

void s_cast_double_to_int16(const double *src, const int64_t *stridesSrc, int16_t *dst, const int64_t *stridesDst, const int64_t *shape, int rank) {
    if (src == nullptr || dst == nullptr || (rank > 0 && shape == nullptr) || rank < 0) return;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0, offsetDst = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        dst[offsetDst] = saturating_float_to_int<int16_t>(src[offsetSrc]);
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDst += stridesDst[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDst -= (shape[d] - 1) * stridesDst[d];
        }
    }
}

/* ============================================================================
 * copy_and_cast_strided
 * ============================================================================
 * A fused native C kernel that flattens any strided multi-dimensional view
 * (ranks 0 to 8) and casts its elements to the target data type in a single
 * contiguous pass.
 *
 * RATIONALE & DESIGN TRADE-OFFS (LOOP FUSION OPTIMIZATION):
 * - While it is modular to split this into a type-independent blind copy
 *   (memcpy elements along strides) followed by a contiguous linear cast sweep,
 *   that modularity requires allocating temporary contiguous arrays on the unmanaged
 *   heap, writing to them, reading them back to cast, and disposing them.
 * - This fused implementation performs both layout flattening and type upcasting
 *   simultaneously in a single native pass directly into the destination FFI buffer,
 *   bypassing intermediate allocations and halving memory bus traffic.
 * - Type dispatching is done via nested switch blocks. Since type tags remain
 *   constant throughout the entire loop, CPU branch prediction completely
 *   eliminates the branching overhead inside the hot loop.
 */
[[maybe_unused]] static void copy_and_cast_strided(
    int src_type, const void *src_ptr, const int64_t *strides_src,
    int dest_type, void *dest_ptr, const int64_t *shape, int rank) {
    
    if (src_ptr == nullptr || dest_ptr == nullptr || rank < 0) return;

    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];

    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetSrc = 0;

    for (int64_t el = 0; el < total_elements; el++) {
        double r_val = 0.0;
        double i_val = 0.0;
        int64_t i64_val = 0;
        uint64_t u64_val = 0;
        bool is_int_src = false;
        bool is_uint64_src = false;

        if (rank == 0) {
            offsetSrc = 0;
        }

        switch (src_type) {
            case 0: // float64
                r_val = ((const double*)src_ptr)[offsetSrc];
                break;
            case 1: // float32
                r_val = (double)((const float*)src_ptr)[offsetSrc];
                break;
            case 2: // float16
                r_val = decode_float16(((const uint16_t*)src_ptr)[offsetSrc]);
                break;
            case 3: // bfloat16
                r_val = decode_bfloat16(((const uint16_t*)src_ptr)[offsetSrc]);
                break;
            case 4: // int64
                i64_val = ((const int64_t*)src_ptr)[offsetSrc];
                r_val = (double)i64_val;
                is_int_src = true;
                break;
            case 5: // int32
                i64_val = (int64_t)((const int32_t*)src_ptr)[offsetSrc];
                r_val = (double)i64_val;
                is_int_src = true;
                break;
            case 6: // int16
                i64_val = (int64_t)((const int16_t*)src_ptr)[offsetSrc];
                r_val = (double)i64_val;
                is_int_src = true;
                break;
            case 7: // int8
                i64_val = (int64_t)((const int8_t*)src_ptr)[offsetSrc];
                r_val = (double)i64_val;
                is_int_src = true;
                break;
            case 8: // uint64
                u64_val = ((const uint64_t*)src_ptr)[offsetSrc];
                r_val = (double)u64_val;
                is_uint64_src = true;
                break;
            case 9: // uint32
                i64_val = (int64_t)((const uint32_t*)src_ptr)[offsetSrc];
                r_val = (double)i64_val;
                is_int_src = true;
                break;
            case 10: // uint16
                i64_val = (int64_t)((const uint16_t*)src_ptr)[offsetSrc];
                r_val = (double)i64_val;
                is_int_src = true;
                break;
            case 11: // uint8
                i64_val = (int64_t)((const uint8_t*)src_ptr)[offsetSrc];
                r_val = (double)i64_val;
                is_int_src = true;
                break;
            case 12: // complex128
                r_val = ((const cpx_t*)src_ptr)[offsetSrc].r;
                i_val = ((const cpx_t*)src_ptr)[offsetSrc].i;
                break;
            case 13: // complex64
                r_val = (double)((const cpx_f_t*)src_ptr)[offsetSrc].r;
                i_val = (double)((const cpx_f_t*)src_ptr)[offsetSrc].i;
                break;
            case 14: // boolean
                i64_val = ((const uint8_t*)src_ptr)[offsetSrc] ? 1 : 0;
                r_val = (double)i64_val;
                is_int_src = true;
                break;
            default:
                return;
        }

        switch (dest_type) {
            case 0: // float64
                ((double*)dest_ptr)[el] = is_uint64_src ? (double)u64_val : (is_int_src ? (double)i64_val : r_val);
                break;
            case 1: // float32
                ((float*)dest_ptr)[el] = is_uint64_src ? (float)u64_val : (is_int_src ? (float)i64_val : (float)r_val);
                break;
            case 2: // float16
                ((uint16_t*)dest_ptr)[el] = encode_float16(is_uint64_src ? (double)u64_val : (is_int_src ? (double)i64_val : r_val));
                break;
            case 3: // bfloat16
                ((uint16_t*)dest_ptr)[el] = encode_bfloat16(is_uint64_src ? (double)u64_val : (is_int_src ? (double)i64_val : r_val));
                break;
            case 4: // int64
                if (is_uint64_src) {
                    ((int64_t*)dest_ptr)[el] = (int64_t)u64_val;
                } else if (is_int_src) {
                    ((int64_t*)dest_ptr)[el] = i64_val;
                } else {
                    ((int64_t*)dest_ptr)[el] = saturating_float_to_int<int64_t>(r_val);
                }
                break;
            case 5: // int32
                if (is_uint64_src) {
                    ((int32_t*)dest_ptr)[el] = (int32_t)u64_val;
                } else if (is_int_src) {
                    ((int32_t*)dest_ptr)[el] = (int32_t)i64_val;
                } else {
                    ((int32_t*)dest_ptr)[el] = saturating_float_to_int<int32_t>(r_val);
                }
                break;
            case 6: // int16
                if (is_uint64_src) {
                    ((int16_t*)dest_ptr)[el] = (int16_t)u64_val;
                } else if (is_int_src) {
                    ((int16_t*)dest_ptr)[el] = (int16_t)i64_val;
                } else {
                    ((int16_t*)dest_ptr)[el] = saturating_float_to_int<int16_t>(r_val);
                }
                break;
            case 7: // int8
                if (is_uint64_src) {
                    ((int8_t*)dest_ptr)[el] = (int8_t)u64_val;
                } else if (is_int_src) {
                    ((int8_t*)dest_ptr)[el] = (int8_t)i64_val;
                } else {
                    ((int8_t*)dest_ptr)[el] = saturating_float_to_int<int8_t>(r_val);
                }
                break;
            case 8: // uint64
                if (is_uint64_src) {
                    ((uint64_t*)dest_ptr)[el] = u64_val;
                } else if (is_int_src) {
                    ((uint64_t*)dest_ptr)[el] = (uint64_t)i64_val;
                } else {
                    ((uint64_t*)dest_ptr)[el] = saturating_float_to_int<uint64_t>(r_val);
                }
                break;
            case 9: // uint32
                if (is_uint64_src) {
                    ((uint32_t*)dest_ptr)[el] = (uint32_t)u64_val;
                } else if (is_int_src) {
                    ((uint32_t*)dest_ptr)[el] = (uint32_t)i64_val;
                } else {
                    ((uint32_t*)dest_ptr)[el] = saturating_float_to_int<uint32_t>(r_val);
                }
                break;
            case 10: // uint16
                if (is_uint64_src) {
                    ((uint16_t*)dest_ptr)[el] = (uint16_t)u64_val;
                } else if (is_int_src) {
                    ((uint16_t*)dest_ptr)[el] = (uint16_t)i64_val;
                } else {
                    ((uint16_t*)dest_ptr)[el] = saturating_float_to_int<uint16_t>(r_val);
                }
                break;
            case 11: // uint8
                if (is_uint64_src) {
                    ((uint8_t*)dest_ptr)[el] = (uint8_t)u64_val;
                } else if (is_int_src) {
                    ((uint8_t*)dest_ptr)[el] = (uint8_t)i64_val;
                } else {
                    ((uint8_t*)dest_ptr)[el] = saturating_float_to_int<uint8_t>(r_val);
                }
                break;
            case 12: // complex128
                ((cpx_t*)dest_ptr)[el].r = is_uint64_src ? (double)u64_val : (is_int_src ? (double)i64_val : r_val);
                ((cpx_t*)dest_ptr)[el].i = i_val;
                break;
            case 13: // complex64
                ((cpx_f_t*)dest_ptr)[el].r = is_uint64_src ? (float)u64_val : (is_int_src ? (float)i64_val : (float)r_val);
                ((cpx_f_t*)dest_ptr)[el].i = (float)i_val;
                break;
            case 14: // boolean
                if (is_uint64_src) {
                    ((uint8_t*)dest_ptr)[el] = (u64_val != 0) ? 1 : 0;
                } else if (is_int_src) {
                    ((uint8_t*)dest_ptr)[el] = (i64_val != 0) ? 1 : 0;
                } else {
                    ((uint8_t*)dest_ptr)[el] = (r_val != 0.0 || i_val != 0.0) ? 1 : 0;
                }
                break;
            default:
                return;
        }

        if (rank > 0) {
            for (int d = rank - 1; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetSrc += strides_src[d];
                    break;
                }
                coord[d] = 0;
                offsetSrc -= (shape[d] - 1) * strides_src[d];
            }
        }
    }
}

#include <string.h>

static void copy_advanced_recursive_c(
    const char *src_ptr,
    char **dest_ptr_ref,
    const int64_t *src_strides,
    const int64_t *src_shape,
    int rank,
    int byte_width,
    const int *types,
    const int64_t *index_vals,
    const int64_t *slice_starts,
    const int64_t *slice_stops,
    const int64_t *slice_steps,
    int64_t **indices_ptrs,
    const int64_t *indices_lens,
    int src_dim,
    int64_t current_offset
) {
    if (src_dim == rank) {
        memcpy(*dest_ptr_ref, src_ptr + current_offset, byte_width);
        *dest_ptr_ref += byte_width;
        return;
    }

    int type = types[src_dim];
    int64_t stride = src_strides[src_dim];

    if (type == 0) { // Index
        int64_t idx = index_vals[src_dim];
        copy_advanced_recursive_c(
            src_ptr, dest_ptr_ref, src_strides, src_shape, rank, byte_width,
            types, index_vals, slice_starts, slice_stops, slice_steps,
            indices_ptrs, indices_lens, src_dim + 1,
            current_offset + idx * stride * byte_width
        );
    } else if (type == 1) { // Slice
        int64_t start = slice_starts[src_dim];
        int64_t stop = slice_stops[src_dim];
        int64_t step = slice_steps[src_dim];
        
        if (step > 0) {
            for (int64_t idx = start; idx < stop; idx += step) {
                copy_advanced_recursive_c(
                    src_ptr, dest_ptr_ref, src_strides, src_shape, rank, byte_width,
                    types, index_vals, slice_starts, slice_stops, slice_steps,
                    indices_ptrs, indices_lens, src_dim + 1,
                    current_offset + idx * stride * byte_width
                );
            }
        } else {
            for (int64_t idx = start; idx > stop; idx += step) {
                copy_advanced_recursive_c(
                    src_ptr, dest_ptr_ref, src_strides, src_shape, rank, byte_width,
                    types, index_vals, slice_starts, slice_stops, slice_steps,
                    indices_ptrs, indices_lens, src_dim + 1,
                    current_offset + idx * stride * byte_width
                );
            }
        }
    } else if (type == 2) { // Indices
        int64_t *indices = indices_ptrs[src_dim];
        int64_t len = indices_lens[src_dim];
        for (int64_t i = 0; i < len; i++) {
            int64_t idx = indices[i];
            copy_advanced_recursive_c(
                src_ptr, dest_ptr_ref, src_strides, src_shape, rank, byte_width,
                types, index_vals, slice_starts, slice_stops, slice_steps,
                indices_ptrs, indices_lens, src_dim + 1,
                current_offset + idx * stride * byte_width
            );
        }
    }
}

void copy_advanced_c(
    const void *src_ptr,
    void *dest_ptr,
    const int64_t *src_strides,
    const int64_t *src_shape,
    int rank,
    int byte_width,
    const int *types,
    const int64_t *index_vals,
    const int64_t *slice_starts,
    const int64_t *slice_stops,
    const int64_t *slice_steps,
    int64_t **indices_ptrs,
    const int64_t *indices_lens
) {
    if (src_ptr == nullptr || dest_ptr == nullptr || rank <= 0) return;
    char *dest_ptr_cursor = (char *)dest_ptr;
    copy_advanced_recursive_c(
        (const char *)src_ptr,
        &dest_ptr_cursor,
        src_strides,
        src_shape,
        rank,
        byte_width,
        types,
        index_vals,
        slice_starts,
        slice_stops,
        slice_steps,
        indices_ptrs,
        indices_lens,
        0,
        0
    );
}

/* ============================================================================
 * SECTION 10: NumPy-Compatible Universal Functions (ufuncs)
 * ============================================================================
 */

static thread_local int division_error_flag = 0;

int get_and_reset_division_error(void) {
    int err = division_error_flag;
    division_error_flag = 0;
    return err;
}

static inline double double_floordiv(double a, double b) {
    if (b == 0.0) {
        if (a == 0.0 || std::isnan(a)) return NAN;
        return std::copysign(INFINITY, a) * std::copysign(1.0, b);
    }
    if (std::isnan(a) || std::isnan(b) || std::isinf(a)) {
        return NAN;
    }
    if (std::isinf(b)) {
        if (a == 0.0) return std::copysign(0.0, a) * std::copysign(1.0, b);
        return ((std::signbit(a) != 0) == (std::signbit(b) != 0)) ? 0.0 : -1.0;
    }
    if (a == 0.0) {
        return std::copysign(0.0, a) * std::copysign(1.0, b);
    }
    double mod = std::fmod(a, b);
    double div = (a - mod) / b;
    if (mod != 0.0) {
        if ((b < 0.0) != (mod < 0.0)) {
            mod += b;
            div -= 1.0;
        }
    }
    double floordiv;
    if (div != 0.0) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5) {
            floordiv += 1.0;
        }
    } else {
        floordiv = std::copysign(0.0, a / b);
    }
    return floordiv;
}
static inline float float_floordiv(float a, float b) {
    if (b == 0.0f) {
        if (a == 0.0f || std::isnan(a)) return NAN;
        return std::copysign(INFINITY, a) * std::copysign(1.0f, b);
    }
    if (std::isnan(a) || std::isnan(b) || std::isinf(a)) {
        return NAN;
    }
    if (std::isinf(b)) {
        if (a == 0.0f) return std::copysign(0.0f, a) * std::copysign(1.0f, b);
        return ((std::signbit(a) != 0) == (std::signbit(b) != 0)) ? 0.0f : -1.0f;
    }
    if (a == 0.0f) {
        return std::copysign(0.0f, a) * std::copysign(1.0f, b);
    }
    float mod = std::fmod(a, b);
    float div = (a - mod) / b;
    if (mod != 0.0f) {
        if ((b < 0.0f) != (mod < 0.0f)) {
            mod += b;
            div -= 1.0f;
        }
    }
    float floordiv;
    if (div != 0.0f) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5f) {
            floordiv += 1.0f;
        }
    } else {
        floordiv = std::copysign(0.0f, a / b);
    }
    return floordiv;
}
static inline int64_t int64_floordiv(int64_t x, int64_t y) {
    if (y == 0) {
        division_error_flag = 1;
        return 0;
    }
    if (x == INT64_MIN && y == -1) {
        return INT64_MIN;
    }
    int64_t res = x / y;
    int64_t rem = x % y;
    if (rem != 0 && ((x < 0) ^ (y < 0))) {
        res--;
    }
    return res;
}
static inline int32_t int32_floordiv(int32_t x, int32_t y) {
    if (y == 0) {
        division_error_flag = 1;
        return 0;
    }
    if (x == INT32_MIN && y == -1) {
        return INT32_MIN;
    }
    int32_t res = x / y;
    int32_t rem = x % y;
    if (rem != 0 && ((x < 0) ^ (y < 0))) {
        res--;
    }
    return res;
}

static inline double double_remainder(double x, double y) {
    if (y == 0.0 || std::isnan(x) || std::isnan(y) || std::isinf(x)) return NAN;
    double rem = std::fmod(x, y);
    if (rem != 0.0) {
        if ((rem < 0.0) != (y < 0.0)) {
            rem += y;
        }
    } else {
        rem = std::copysign(0.0, y);
    }
    return rem;
}
static inline float float_remainder(float x, float y) {
    if (y == 0.0f || std::isnan(x) || std::isnan(y) || std::isinf(x)) return NAN;
    float rem = std::fmod(x, y);
    if (rem != 0.0f) {
        if ((rem < 0.0f) != (y < 0.0f)) {
            rem += y;
        }
    } else {
        rem = std::copysign(0.0f, y);
    }
    return rem;
}
static inline int64_t int64_remainder(int64_t x, int64_t y) {
    if (y == 0) {
        division_error_flag = 1;
        return 0;
    }
    if (x == INT64_MIN && y == -1) {
        return 0;
    }
    int64_t rem = x % y;
    if (rem != 0 && ((rem < 0) != (y < 0))) {
        rem += y;
    }
    return rem;
}
static inline int32_t int32_remainder(int32_t x, int32_t y) {
    if (y == 0) {
        division_error_flag = 1;
        return 0;
    }
    if (x == INT32_MIN && y == -1) {
        return 0;
    }
    int32_t rem = x % y;
    if (rem != 0 && ((rem < 0) != (y < 0))) {
        rem += y;
    }
    return rem;
}

static inline int64_t int64_fmod(int64_t x, int64_t y) {
    if (y == 0) {
        division_error_flag = 1;
        return 0;
    }
    if (x == INT64_MIN && y == -1) {
        return 0;
    }
    return x % y;
}
static inline int32_t int32_fmod(int32_t x, int32_t y) {
    if (y == 0) {
        division_error_flag = 1;
        return 0;
    }
    if (x == INT32_MIN && y == -1) {
        return 0;
    }
    return x % y;
}

static inline uint64_t gcd_u64(uint64_t u, uint64_t v) {
    while (v != 0) {
        uint64_t t = v;
        v = u % v;
        u = t;
    }
    return u;
}

template <typename T>
static inline T gcd_impl(T a, T b) {
    uint64_t u;
    uint64_t v;
    if constexpr (std::is_signed_v<T>) {
        u = (a < 0) ? -static_cast<uint64_t>(a) : static_cast<uint64_t>(a);
        v = (b < 0) ? -static_cast<uint64_t>(b) : static_cast<uint64_t>(b);
    } else {
        u = static_cast<uint64_t>(a);
        v = static_cast<uint64_t>(b);
    }
    return static_cast<T>(gcd_u64(u, v));
}

template <typename T>
static inline T lcm_impl(T a, T b) {
    if (a == 0 || b == 0) return 0;
    uint64_t u;
    uint64_t v;
    if constexpr (std::is_signed_v<T>) {
        u = (a < 0) ? -static_cast<uint64_t>(a) : static_cast<uint64_t>(a);
        v = (b < 0) ? -static_cast<uint64_t>(b) : static_cast<uint64_t>(b);
    } else {
        u = static_cast<uint64_t>(a);
        v = static_cast<uint64_t>(b);
    }
    uint64_t g = gcd_u64(u, v);
    return static_cast<T>((u / g) * v);
}

template <typename T>
static inline T heaviside_impl(T x, T h0) {
    if (std::isnan(x)) return x;
    if (x < (T)0) return (T)0;
    if (x > (T)0) return (T)1;
    return h0;
}

static inline cpx_t cpx_square(cpx_t z) {
    return (cpx_t){z.r * z.r - z.i * z.i, 2.0 * z.r * z.i};
}
static inline cpx_f_t cpx_square_f(cpx_f_t z) {
    return (cpx_f_t){z.r * z.r - z.i * z.i, 2.0f * z.r * z.i};
}

#define DEFINE_CONTIGUOUS_UNARY_IMPL(name, typeSrc, typeRes, expr) \
void name(const typeSrc *src, typeRes *res, int64_t size, const uint8_t *mask) { \
    v_unary_cast_impl(src, res, size, mask, [](typeSrc x) -> typeRes { return (expr); }); \
}

#define DEFINE_CONTIGUOUS_BINARY_IMPL(name, typeA, typeB, typeRes, expr) \
void name(const typeA *a, const typeB *b, typeRes *res, int64_t size, const uint8_t *mask) { \
    v_binary_impl(a, b, res, size, mask, [](typeA x, typeB y) -> typeRes { return (expr); }); \
}

/* 1. Contiguous (Vector) Implementations */

DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_double, double, double, x * x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_float, float, float, x * x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_int64, int64_t, int64_t, (int64_t)((uint64_t)x * (uint64_t)x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_int32, int32_t, int32_t, (int32_t)((uint32_t)x * (uint32_t)x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_complex128, cpx_t, cpx_t, cpx_square(x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_complex64, cpx_f_t, cpx_f_t, cpx_square_f(x))

// log2 and log10 real contiguous
DEFINE_CONTIGUOUS_UNARY_IMPL(v_log2_double, double, double, std::log2(x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_log2_float, float, float, std::log2(x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_log10_double, double, double, std::log10(x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_log10_float, float, float, std::log10(x))

// log2 and log10 complex contiguous
DEFINE_COMPLEX_UNARY_VEC(v_log2_complex128, cpx_t, cpx_log2)
DEFINE_COMPLEX_UNARY_VEC(v_log2_complex64, cpx_f_t, cpx_log2_f)
DEFINE_COMPLEX_UNARY_VEC(v_log10_complex128, cpx_t, cpx_log10)
DEFINE_COMPLEX_UNARY_VEC(v_log10_complex64, cpx_f_t, cpx_log10_f)

template <typename T>
static inline T int_reciprocal(T x) {
    if (x == 0) {
        division_error_flag = 1;
        return 0;
    }
    return 1 / x;
}
static inline int64_t int64_reciprocal(int64_t x) { return int_reciprocal(x); }
static inline int32_t int32_reciprocal(int32_t x) { return int_reciprocal(x); }
static inline int16_t int16_reciprocal(int16_t x) { return int_reciprocal(x); }
static inline uint8_t uint8_reciprocal(uint8_t x) { return int_reciprocal(x); }

// reciprocal contiguous
DEFINE_CONTIGUOUS_UNARY_IMPL(v_reciprocal_double, double, double, 1.0 / x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_reciprocal_float, float, float, 1.0f / x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_reciprocal_complex128, cpx_t, cpx_t, cpx_reciprocal(x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_reciprocal_complex64, cpx_f_t, cpx_f_t, cpx_reciprocal_f(x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_reciprocal_int64, int64_t, int64_t, int64_reciprocal(x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_reciprocal_int32, int32_t, int32_t, int32_reciprocal(x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_reciprocal_int16, int16_t, int16_t, int16_reciprocal(x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_reciprocal_uint8, uint8_t, uint8_t, uint8_reciprocal(x))

// positive contiguous
DEFINE_CONTIGUOUS_UNARY_IMPL(v_positive_double, double, double, x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_positive_float, float, float, x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_positive_complex128, cpx_t, cpx_t, x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_positive_complex64, cpx_f_t, cpx_f_t, x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_positive_int64, int64_t, int64_t, x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_positive_int32, int32_t, int32_t, x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_positive_int16, int16_t, int16_t, x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_positive_uint8, uint8_t, uint8_t, x)

DEFINE_CONTIGUOUS_BINARY_IMPL(v_pow_double, double, double, double, std::pow(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_pow_float, float, float, float, std::pow(x, y))

DEFINE_CONTIGUOUS_BINARY_IMPL(v_floordiv_double, double, double, double, double_floordiv(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_floordiv_float, float, float, float, float_floordiv(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_floordiv_int64, int64_t, int64_t, int64_t, int64_floordiv(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_floordiv_int32, int32_t, int32_t, int32_t, int32_floordiv(x, y))

DEFINE_CONTIGUOUS_BINARY_IMPL(v_remainder_double, double, double, double, double_remainder(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_remainder_float, float, float, float, float_remainder(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_remainder_int64, int64_t, int64_t, int64_t, int64_remainder(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_remainder_int32, int32_t, int32_t, int32_t, int32_remainder(x, y))

DEFINE_CONTIGUOUS_BINARY_IMPL(v_fmod_double, double, double, double, std::fmod(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_fmod_float, float, float, float, std::fmod(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_fmod_int64, int64_t, int64_t, int64_t, int64_fmod(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_fmod_int32, int32_t, int32_t, int32_t, int32_fmod(x, y))

DEFINE_CONTIGUOUS_BINARY_IMPL(v_gcd_int64, int64_t, int64_t, int64_t, gcd_impl<int64_t>(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_gcd_int32, int32_t, int32_t, int32_t, gcd_impl<int32_t>(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_lcm_int64, int64_t, int64_t, int64_t, lcm_impl<int64_t>(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_lcm_int32, int32_t, int32_t, int32_t, lcm_impl<int32_t>(x, y))

DEFINE_CONTIGUOUS_BINARY_IMPL(v_heaviside_double, double, double, double, heaviside_impl<double>(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_heaviside_float, float, float, float, heaviside_impl<float>(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_heaviside_int64, int64_t, int64_t, int64_t, (x < 0 ? 0 : (x > 0 ? 1 : y)))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_heaviside_int32, int32_t, int32_t, int32_t, (x < 0 ? 0 : (x > 0 ? 1 : y)))

DEFINE_CONTIGUOUS_UNARY_IMPL(v_isnan_double, double, uint8_t, std::isnan(x) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isnan_float, float, uint8_t, std::isnan(x) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isnan_complex128, cpx_t, uint8_t, (std::isnan(x.r) || std::isnan(x.i)) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isnan_complex64, cpx_f_t, uint8_t, (std::isnan(x.r) || std::isnan(x.i)) ? 1 : 0)

DEFINE_CONTIGUOUS_UNARY_IMPL(v_isinf_double, double, uint8_t, std::isinf(x) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isinf_float, float, uint8_t, std::isinf(x) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isinf_complex128, cpx_t, uint8_t, (std::isinf(x.r) || std::isinf(x.i)) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isinf_complex64, cpx_f_t, uint8_t, (std::isinf(x.r) || std::isinf(x.i)) ? 1 : 0)

DEFINE_CONTIGUOUS_UNARY_IMPL(v_isfinite_double, double, uint8_t, std::isfinite(x) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isfinite_float, float, uint8_t, std::isfinite(x) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isfinite_complex128, cpx_t, uint8_t, (std::isfinite(x.r) && std::isfinite(x.i)) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isfinite_complex64, cpx_f_t, uint8_t, (std::isfinite(x.r) && std::isfinite(x.i)) ? 1 : 0)

DEFINE_CONTIGUOUS_BINARY_IMPL(v_copysign_double, double, double, double, std::copysign(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_copysign_float, float, float, float, std::copysign(x, y))

/* 2. Strided Multidimensional Implementations */

#define DEFINE_STRIDED_UNARY_IMPL(name, typeSrc, typeRes, expr) \
void name(const typeSrc *src, const int64_t *stridesSrc, \
          typeRes *res, const int64_t *stridesRes, \
          const int64_t *shape, int rank, \
          const uint8_t *mask) { \
    strided_unary_cast_impl(src, stridesSrc, res, stridesRes, shape, rank, mask, [](typeSrc x) -> typeRes { return (expr); }); \
}

DEFINE_STRIDED_UNARY_IMPL(s_square_double, double, double, x * x)
DEFINE_STRIDED_UNARY_IMPL(s_square_float, float, float, x * x)
DEFINE_STRIDED_UNARY_IMPL(s_square_int64, int64_t, int64_t, (int64_t)((uint64_t)x * (uint64_t)x))
DEFINE_STRIDED_UNARY_IMPL(s_square_int32, int32_t, int32_t, (int32_t)((uint32_t)x * (uint32_t)x))
DEFINE_STRIDED_UNARY_IMPL(s_square_complex128, cpx_t, cpx_t, cpx_square(x))
DEFINE_STRIDED_UNARY_IMPL(s_square_complex64, cpx_f_t, cpx_f_t, cpx_square_f(x))

// log2 and log10 real strided
DEFINE_STRIDED_UNARY_IMPL(s_log2_double, double, double, std::log2(x))
DEFINE_STRIDED_UNARY_IMPL(s_log2_float, float, float, std::log2(x))
DEFINE_STRIDED_UNARY_IMPL(s_log10_double, double, double, std::log10(x))
DEFINE_STRIDED_UNARY_IMPL(s_log10_float, float, float, std::log10(x))

// log2 and log10 complex strided
DEFINE_STRIDED_UNARY_IMPL(s_log2_complex128, cpx_t, cpx_t, cpx_log2(x))
DEFINE_STRIDED_UNARY_IMPL(s_log2_complex64, cpx_f_t, cpx_f_t, cpx_log2_f(x))
DEFINE_STRIDED_UNARY_IMPL(s_log10_complex128, cpx_t, cpx_t, cpx_log10(x))
DEFINE_STRIDED_UNARY_IMPL(s_log10_complex64, cpx_f_t, cpx_f_t, cpx_log10_f(x))

// reciprocal strided
DEFINE_STRIDED_UNARY_IMPL(s_reciprocal_double, double, double, 1.0 / x)
DEFINE_STRIDED_UNARY_IMPL(s_reciprocal_float, float, float, 1.0f / x)
DEFINE_STRIDED_UNARY_IMPL(s_reciprocal_complex128, cpx_t, cpx_t, cpx_reciprocal(x))
DEFINE_STRIDED_UNARY_IMPL(s_reciprocal_complex64, cpx_f_t, cpx_f_t, cpx_reciprocal_f(x))
DEFINE_STRIDED_UNARY_IMPL(s_reciprocal_int64, int64_t, int64_t, int64_reciprocal(x))
DEFINE_STRIDED_UNARY_IMPL(s_reciprocal_int32, int32_t, int32_t, int32_reciprocal(x))
DEFINE_STRIDED_UNARY_IMPL(s_reciprocal_int16, int16_t, int16_t, int16_reciprocal(x))
DEFINE_STRIDED_UNARY_IMPL(s_reciprocal_uint8, uint8_t, uint8_t, uint8_reciprocal(x))

// positive strided
DEFINE_STRIDED_UNARY_IMPL(s_positive_double, double, double, x)
DEFINE_STRIDED_UNARY_IMPL(s_positive_float, float, float, x)
DEFINE_STRIDED_UNARY_IMPL(s_positive_complex128, cpx_t, cpx_t, x)
DEFINE_STRIDED_UNARY_IMPL(s_positive_complex64, cpx_f_t, cpx_f_t, x)
DEFINE_STRIDED_UNARY_IMPL(s_positive_int64, int64_t, int64_t, x)
DEFINE_STRIDED_UNARY_IMPL(s_positive_int32, int32_t, int32_t, x)
DEFINE_STRIDED_UNARY_IMPL(s_positive_int16, int16_t, int16_t, x)
DEFINE_STRIDED_UNARY_IMPL(s_positive_uint8, uint8_t, uint8_t, x)

#define DEFINE_STRIDED_BINARY_IMPL(name, typeA, typeB, typeRes, expr) \
void name(const typeA *a, const int64_t *stridesA, \
          const typeB *b, const int64_t *stridesB, \
          typeRes *res, const int64_t *stridesRes, \
          const int64_t *shape, int rank, \
          const uint8_t *mask) { \
    strided_binary_op_impl(a, stridesA, b, stridesB, res, stridesRes, shape, rank, mask, [](typeA x, typeB y) -> typeRes { return (expr); }); \
}

DEFINE_STRIDED_BINARY_IMPL(s_pow_double, double, double, double, std::pow(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_pow_float, float, float, float, std::pow(x, y))

DEFINE_STRIDED_BINARY_IMPL(s_floordiv_double, double, double, double, double_floordiv(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_floordiv_float, float, float, float, float_floordiv(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_floordiv_int64, int64_t, int64_t, int64_t, int64_floordiv(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_floordiv_int32, int32_t, int32_t, int32_t, int32_floordiv(x, y))

DEFINE_STRIDED_BINARY_IMPL(s_remainder_double, double, double, double, double_remainder(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_remainder_float, float, float, float, float_remainder(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_remainder_int64, int64_t, int64_t, int64_t, int64_remainder(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_remainder_int32, int32_t, int32_t, int32_t, int32_remainder(x, y))

DEFINE_STRIDED_BINARY_IMPL(s_fmod_double, double, double, double, std::fmod(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_fmod_float, float, float, float, std::fmod(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_fmod_int64, int64_t, int64_t, int64_t, int64_fmod(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_fmod_int32, int32_t, int32_t, int32_t, int32_fmod(x, y))

DEFINE_STRIDED_BINARY_IMPL(s_gcd_int64, int64_t, int64_t, int64_t, gcd_impl<int64_t>(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_gcd_int32, int32_t, int32_t, int32_t, gcd_impl<int32_t>(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_lcm_int64, int64_t, int64_t, int64_t, lcm_impl<int64_t>(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_lcm_int32, int32_t, int32_t, int32_t, lcm_impl<int32_t>(x, y))

DEFINE_STRIDED_BINARY_IMPL(s_heaviside_double, double, double, double, heaviside_impl<double>(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_heaviside_float, float, float, float, heaviside_impl<float>(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_heaviside_int64, int64_t, int64_t, int64_t, (x < 0 ? 0 : (x > 0 ? 1 : y)))
DEFINE_STRIDED_BINARY_IMPL(s_heaviside_int32, int32_t, int32_t, int32_t, (x < 0 ? 0 : (x > 0 ? 1 : y)))

DEFINE_STRIDED_BINARY_IMPL(s_copysign_double, double, double, double, std::copysign(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_copysign_float, float, float, float, std::copysign(x, y))

DEFINE_STRIDED_UNARY_IMPL(s_expm1_double, double, double, std::expm1(x))
DEFINE_STRIDED_UNARY_IMPL(s_expm1_float, float, float, std::expm1(x))

DEFINE_STRIDED_UNARY_IMPL(s_log1p_double, double, double, std::log1p(x))
DEFINE_STRIDED_UNARY_IMPL(s_log1p_float, float, float, std::log1p(x))

DEFINE_STRIDED_UNARY_IMPL(s_rint_double, double, double, std::rint(x))
DEFINE_STRIDED_UNARY_IMPL(s_rint_float, float, float, std::rint(x))

DEFINE_STRIDED_UNARY_IMPL(s_trunc_double, double, double, std::trunc(x))
DEFINE_STRIDED_UNARY_IMPL(s_trunc_float, float, float, std::trunc(x))

DEFINE_STRIDED_BINARY_IMPL(s_logaddexp_double, double, double, double, logaddexp_op<double>(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_logaddexp_float, float, float, float, logaddexp_op<float>(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_logaddexp2_double, double, double, double, logaddexp2_op<double>(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_logaddexp2_float, float, float, float, logaddexp2_op<float>(x, y))

DEFINE_STRIDED_UNARY_IMPL(s_isnan_double, double, uint8_t, std::isnan(x) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isnan_float, float, uint8_t, std::isnan(x) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isnan_complex128, cpx_t, uint8_t, (std::isnan(x.r) || std::isnan(x.i)) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isnan_complex64, cpx_f_t, uint8_t, (std::isnan(x.r) || std::isnan(x.i)) ? 1 : 0)

DEFINE_STRIDED_UNARY_IMPL(s_isinf_double, double, uint8_t, std::isinf(x) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isinf_float, float, uint8_t, std::isinf(x) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isinf_complex128, cpx_t, uint8_t, (std::isinf(x.r) || std::isinf(x.i)) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isinf_complex64, cpx_f_t, uint8_t, (std::isinf(x.r) || std::isinf(x.i)) ? 1 : 0)

DEFINE_STRIDED_UNARY_IMPL(s_isfinite_double, double, uint8_t, std::isfinite(x) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isfinite_float, float, uint8_t, std::isfinite(x) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isfinite_complex128, cpx_t, uint8_t, (std::isfinite(x.r) && std::isfinite(x.i)) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isfinite_complex64, cpx_f_t, uint8_t, (std::isfinite(x.r) && std::isfinite(x.i)) ? 1 : 0)

/* Logical and Casting-to-Boolean Implementations */
DEFINE_CONTIGUOUS_UNARY_IMPL(v_to_bool_double, double, uint8_t, (x != 0.0) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_to_bool_float, float, uint8_t, (x != 0.0f) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_to_bool_int64, int64_t, uint8_t, (x != 0) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_to_bool_int32, int32_t, uint8_t, (x != 0) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_to_bool_uint8, uint8_t, uint8_t, (x != 0) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_to_bool_int16, int16_t, uint8_t, (x != 0) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_to_bool_complex128, cpx_t, uint8_t, (x.r != 0.0 || x.i != 0.0) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_to_bool_complex64, cpx_f_t, uint8_t, (x.r != 0.0f || x.i != 0.0f) ? 1 : 0)

DEFINE_STRIDED_UNARY_IMPL(s_to_bool_double, double, uint8_t, (x != 0.0) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_to_bool_float, float, uint8_t, (x != 0.0f) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_to_bool_int64, int64_t, uint8_t, (x != 0) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_to_bool_int32, int32_t, uint8_t, (x != 0) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_to_bool_uint8, uint8_t, uint8_t, (x != 0) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_to_bool_int16, int16_t, uint8_t, (x != 0) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_to_bool_complex128, cpx_t, uint8_t, (x.r != 0.0 || x.i != 0.0) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_to_bool_complex64, cpx_f_t, uint8_t, (x.r != 0.0f || x.i != 0.0f) ? 1 : 0)

DEFINE_CONTIGUOUS_BINARY_IMPL(v_logical_and, uint8_t, uint8_t, uint8_t, (x && y) ? 1 : 0)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_logical_or, uint8_t, uint8_t, uint8_t, (x || y) ? 1 : 0)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_logical_xor, uint8_t, uint8_t, uint8_t, ((x != 0) != (y != 0)) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_logical_not, uint8_t, uint8_t, (!x) ? 1 : 0)

DEFINE_STRIDED_BINARY_IMPL(s_logical_and, uint8_t, uint8_t, uint8_t, (x && y) ? 1 : 0)
DEFINE_STRIDED_BINARY_IMPL(s_logical_or, uint8_t, uint8_t, uint8_t, (x || y) ? 1 : 0)
DEFINE_STRIDED_BINARY_IMPL(s_logical_xor, uint8_t, uint8_t, uint8_t, ((x != 0) != (y != 0)) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_logical_not, uint8_t, uint8_t, (!x) ? 1 : 0)

/* Safe shift helpers to prevent undefined C behavior */
template <typename T>
static inline T safe_left_shift_int(T x, T y) {
    using U = std::make_unsigned_t<T>;
    constexpr int bits = static_cast<int>(sizeof(T) * 8);
    if constexpr (std::is_signed_v<T>) {
        if (y < 0 || y >= bits) return 0;
    } else {
        if (y >= static_cast<T>(bits)) return 0;
    }
    return static_cast<T>(static_cast<uint64_t>(static_cast<U>(x)) << static_cast<unsigned int>(y));
}

template <typename T>
static inline T safe_right_shift_int(T x, T y) {
    constexpr int bits = static_cast<int>(sizeof(T) * 8);
    if constexpr (std::is_signed_v<T>) {
        if (y < 0) return 0;
        if (y >= bits) return (x < 0) ? static_cast<T>(-1) : static_cast<T>(0);
        return static_cast<T>(x >> static_cast<unsigned int>(y));
    } else {
        if (y >= static_cast<T>(bits)) return 0;
        return static_cast<T>(x >> static_cast<unsigned int>(y));
    }
}

static inline int32_t safe_left_shift_int32(int32_t x, int32_t y) {
    return safe_left_shift_int<int32_t>(x, y);
}
static inline int32_t safe_right_shift_int32(int32_t x, int32_t y) {
    return safe_right_shift_int<int32_t>(x, y);
}

static inline int64_t safe_left_shift_int64(int64_t x, int64_t y) {
    return safe_left_shift_int<int64_t>(x, y);
}
static inline int64_t safe_right_shift_int64(int64_t x, int64_t y) {
    return safe_right_shift_int<int64_t>(x, y);
}

static inline uint8_t safe_left_shift_uint8(uint8_t x, uint8_t y) {
    return safe_left_shift_int<uint8_t>(x, y);
}
static inline uint8_t safe_right_shift_uint8(uint8_t x, uint8_t y) {
    return safe_right_shift_int<uint8_t>(x, y);
}

static inline int16_t safe_left_shift_int16(int16_t x, int16_t y) {
    return safe_left_shift_int<int16_t>(x, y);
}
static inline int16_t safe_right_shift_int16(int16_t x, int16_t y) {
    return safe_right_shift_int<int16_t>(x, y);
}

/* Bitwise AND implementations */
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_and_int32, int32_t, int32_t, int32_t, x & y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_and_int64, int64_t, int64_t, int64_t, x & y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_and_uint8, uint8_t, uint8_t, uint8_t, x & y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_and_int16, int16_t, int16_t, int16_t, x & y)

DEFINE_STRIDED_BINARY_IMPL(s_bitwise_and_int32, int32_t, int32_t, int32_t, x & y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_and_int64, int64_t, int64_t, int64_t, x & y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_and_uint8, uint8_t, uint8_t, uint8_t, x & y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_and_int16, int16_t, int16_t, int16_t, x & y)

/* Bitwise OR implementations */
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_or_int32, int32_t, int32_t, int32_t, x | y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_or_int64, int64_t, int64_t, int64_t, x | y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_or_uint8, uint8_t, uint8_t, uint8_t, x | y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_or_int16, int16_t, int16_t, int16_t, x | y)

DEFINE_STRIDED_BINARY_IMPL(s_bitwise_or_int32, int32_t, int32_t, int32_t, x | y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_or_int64, int64_t, int64_t, int64_t, x | y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_or_uint8, uint8_t, uint8_t, uint8_t, x | y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_or_int16, int16_t, int16_t, int16_t, x | y)

/* Bitwise XOR implementations */
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_xor_int32, int32_t, int32_t, int32_t, x ^ y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_xor_int64, int64_t, int64_t, int64_t, x ^ y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_xor_uint8, uint8_t, uint8_t, uint8_t, x ^ y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_xor_int16, int16_t, int16_t, int16_t, x ^ y)

DEFINE_STRIDED_BINARY_IMPL(s_bitwise_xor_int32, int32_t, int32_t, int32_t, x ^ y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_xor_int64, int64_t, int64_t, int64_t, x ^ y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_xor_uint8, uint8_t, uint8_t, uint8_t, x ^ y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_xor_int16, int16_t, int16_t, int16_t, x ^ y)

/* Left Shift implementations */
DEFINE_CONTIGUOUS_BINARY_IMPL(v_left_shift_int32, int32_t, int32_t, int32_t, safe_left_shift_int32(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_left_shift_int64, int64_t, int64_t, int64_t, safe_left_shift_int64(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_left_shift_uint8, uint8_t, uint8_t, uint8_t, safe_left_shift_uint8(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_left_shift_int16, int16_t, int16_t, int16_t, safe_left_shift_int16(x, y))

DEFINE_STRIDED_BINARY_IMPL(s_left_shift_int32, int32_t, int32_t, int32_t, safe_left_shift_int32(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_left_shift_int64, int64_t, int64_t, int64_t, safe_left_shift_int64(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_left_shift_uint8, uint8_t, uint8_t, uint8_t, safe_left_shift_uint8(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_left_shift_int16, int16_t, int16_t, int16_t, safe_left_shift_int16(x, y))

/* Right Shift implementations */
DEFINE_CONTIGUOUS_BINARY_IMPL(v_right_shift_int32, int32_t, int32_t, int32_t, safe_right_shift_int32(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_right_shift_int64, int64_t, int64_t, int64_t, safe_right_shift_int64(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_right_shift_uint8, uint8_t, uint8_t, uint8_t, safe_right_shift_uint8(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_right_shift_int16, int16_t, int16_t, int16_t, safe_right_shift_int16(x, y))

DEFINE_STRIDED_BINARY_IMPL(s_right_shift_int32, int32_t, int32_t, int32_t, safe_right_shift_int32(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_right_shift_int64, int64_t, int64_t, int64_t, safe_right_shift_int64(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_right_shift_uint8, uint8_t, uint8_t, uint8_t, safe_right_shift_uint8(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_right_shift_int16, int16_t, int16_t, int16_t, safe_right_shift_int16(x, y))

/* Invert (Bitwise Negation) implementations */
DEFINE_CONTIGUOUS_UNARY_IMPL(v_invert_int32, int32_t, int32_t, ~x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_invert_int64, int64_t, int64_t, ~x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_invert_uint8, uint8_t, uint8_t, (uint8_t)(~x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_invert_int16, int16_t, int16_t, (int16_t)(~x))

DEFINE_STRIDED_UNARY_IMPL(s_invert_int32, int32_t, int32_t, ~x)
DEFINE_STRIDED_UNARY_IMPL(s_invert_int64, int64_t, int64_t, ~x)
DEFINE_STRIDED_UNARY_IMPL(s_invert_uint8, uint8_t, uint8_t, (uint8_t)(~x))
DEFINE_STRIDED_UNARY_IMPL(s_invert_int16, int16_t, int16_t, (int16_t)(~x))



/* ============================================================================
 * SECTION: Kronecker Product, Vector Outer Product, Cross Product & Norms
 * ============================================================================
 */

static inline cpx_t cpx_kron_mul(cpx_t c1, cpx_t c2) {
    cpx_t res;
    res.r = c1.r * c2.r - c1.i * c2.i;
    res.i = c1.r * c2.i + c1.i * c2.r;
    return res;
}

static inline cpx_f_t cpx_f_kron_mul(cpx_f_t c1, cpx_f_t c2) {
    cpx_f_t res;
    res.r = c1.r * c2.r - c1.i * c2.i;
    res.i = c1.r * c2.i + c1.i * c2.r;
    return res;
}

template <bool CONTIG_B, bool CONTIG_RES>
static inline void kron_row_double(
    const double a_val,
    const double * RESTRICT b_row, int64_t strideB_1,
    double * RESTRICT dest_ptr, int64_t strideRes_1,
    int64_t q)
{
    if constexpr (CONTIG_B && CONTIG_RES) {
        int64_t c = 0;
#if HAS_AVX2_REDUCTIONS
        __m256d va = _mm256_set1_pd(a_val);
        for (; c + 15 < q; c += 16) {
            __m256d vb0 = _mm256_loadu_pd(b_row + c);
            __m256d vb1 = _mm256_loadu_pd(b_row + c + 4);
            __m256d vb2 = _mm256_loadu_pd(b_row + c + 8);
            __m256d vb3 = _mm256_loadu_pd(b_row + c + 12);

            _mm256_storeu_pd(dest_ptr + c, _mm256_mul_pd(va, vb0));
            _mm256_storeu_pd(dest_ptr + c + 4, _mm256_mul_pd(va, vb1));
            _mm256_storeu_pd(dest_ptr + c + 8, _mm256_mul_pd(va, vb2));
            _mm256_storeu_pd(dest_ptr + c + 12, _mm256_mul_pd(va, vb3));
        }
        for (; c + 3 < q; c += 4) {
            __m256d vb = _mm256_loadu_pd(b_row + c);
            _mm256_storeu_pd(dest_ptr + c, _mm256_mul_pd(va, vb));
        }
#endif
        for (; c < q; c++) {
            dest_ptr[c] = a_val * b_row[c];
        }
    } else {
        for (int64_t c = 0; c < q; c++) {
            dest_ptr[c * strideRes_1] = a_val * b_row[c * strideB_1];
        }
    }
}

template <bool CONTIG_B, bool CONTIG_RES>
static inline void kron_row_float(
    const float a_val,
    const float * RESTRICT b_row, int64_t strideB_1,
    float * RESTRICT dest_ptr, int64_t strideRes_1,
    int64_t q)
{
    if constexpr (CONTIG_B && CONTIG_RES) {
        int64_t c = 0;
#if HAS_AVX2_REDUCTIONS
        __m256 va = _mm256_set1_ps(a_val);
        for (; c + 31 < q; c += 32) {
            __m256 vb0 = _mm256_loadu_ps(b_row + c);
            __m256 vb1 = _mm256_loadu_ps(b_row + c + 8);
            __m256 vb2 = _mm256_loadu_ps(b_row + c + 16);
            __m256 vb3 = _mm256_loadu_ps(b_row + c + 24);

            _mm256_storeu_ps(dest_ptr + c, _mm256_mul_ps(va, vb0));
            _mm256_storeu_ps(dest_ptr + c + 8, _mm256_mul_ps(va, vb1));
            _mm256_storeu_ps(dest_ptr + c + 16, _mm256_mul_ps(va, vb2));
            _mm256_storeu_ps(dest_ptr + c + 24, _mm256_mul_ps(va, vb3));
        }
        for (; c + 7 < q; c += 8) {
            __m256 vb = _mm256_loadu_ps(b_row + c);
            _mm256_storeu_ps(dest_ptr + c, _mm256_mul_ps(va, vb));
        }
#endif
        for (; c < q; c++) {
            dest_ptr[c] = a_val * b_row[c];
        }
    } else {
        for (int64_t c = 0; c < q; c++) {
            dest_ptr[c * strideRes_1] = a_val * b_row[c * strideB_1];
        }
    }
}

template <bool CONTIG_B, bool CONTIG_RES>
static inline void kron_row_int64(
    const int64_t a_val,
    const int64_t * RESTRICT b_row, int64_t strideB_1,
    int64_t * RESTRICT dest_ptr, int64_t strideRes_1,
    int64_t q)
{
    if constexpr (CONTIG_B && CONTIG_RES) {
        int64_t c = 0;
#pragma GCC ivdep
        for (; c < q; c++) {
            dest_ptr[c] = (int64_t)((uint64_t)a_val * (uint64_t)b_row[c]);
        }
    } else {
        for (int64_t c = 0; c < q; c++) {
            dest_ptr[c * strideRes_1] = (int64_t)((uint64_t)a_val * (uint64_t)b_row[c * strideB_1]);
        }
    }
}

template <bool CONTIG_B, bool CONTIG_RES>
static inline void kron_row_int32(
    const int32_t a_val,
    const int32_t * RESTRICT b_row, int64_t strideB_1,
    int32_t * RESTRICT dest_ptr, int64_t strideRes_1,
    int64_t q)
{
    if constexpr (CONTIG_B && CONTIG_RES) {
        int64_t c = 0;
#if HAS_AVX2_REDUCTIONS
        __m256i va = _mm256_set1_epi32(a_val);
        for (; c + 15 < q; c += 16) {
            __m256i vb0 = _mm256_loadu_si256((const __m256i *)(b_row + c));
            __m256i vb1 = _mm256_loadu_si256((const __m256i *)(b_row + c + 8));
            _mm256_storeu_si256((__m256i *)(dest_ptr + c), _mm256_mullo_epi32(va, vb0));
            _mm256_storeu_si256((__m256i *)(dest_ptr + c + 8), _mm256_mullo_epi32(va, vb1));
        }
        for (; c + 7 < q; c += 8) {
            __m256i vb = _mm256_loadu_si256((const __m256i *)(b_row + c));
            _mm256_storeu_si256((__m256i *)(dest_ptr + c), _mm256_mullo_epi32(va, vb));
        }
#endif
        for (; c < q; c++) {
            dest_ptr[c] = (int32_t)((uint32_t)a_val * (uint32_t)b_row[c]);
        }
    } else {
        for (int64_t c = 0; c < q; c++) {
            dest_ptr[c * strideRes_1] = (int32_t)((uint32_t)a_val * (uint32_t)b_row[c * strideB_1]);
        }
    }
}

template <bool CONTIG_B, bool CONTIG_RES>
static inline void kron_row_int16(
    const int16_t a_val,
    const int16_t * RESTRICT b_row, int64_t strideB_1,
    int16_t * RESTRICT dest_ptr, int64_t strideRes_1,
    int64_t q)
{
    if constexpr (CONTIG_B && CONTIG_RES) {
        int64_t c = 0;
#if HAS_AVX2_REDUCTIONS
        __m256i va = _mm256_set1_epi16(a_val);
        for (; c + 15 < q; c += 16) {
            __m256i vb = _mm256_loadu_si256((const __m256i *)(b_row + c));
            _mm256_storeu_si256((__m256i *)(dest_ptr + c), _mm256_mullo_epi16(va, vb));
        }
#endif
        for (; c < q; c++) {
            dest_ptr[c] = (int16_t)((uint16_t)a_val * (uint16_t)b_row[c]);
        }
    } else {
        for (int64_t c = 0; c < q; c++) {
            dest_ptr[c * strideRes_1] = (int16_t)((uint16_t)a_val * (uint16_t)b_row[c * strideB_1]);
        }
    }
}

template <bool CONTIG_B, bool CONTIG_RES>
static inline void kron_row_uint8(
    const uint8_t a_val,
    const uint8_t * RESTRICT b_row, int64_t strideB_1,
    uint8_t * RESTRICT dest_ptr, int64_t strideRes_1,
    int64_t q)
{
    if constexpr (CONTIG_B && CONTIG_RES) {
        int64_t c = 0;
#pragma GCC ivdep
        for (; c < q; c++) {
            dest_ptr[c] = (uint8_t)(a_val * b_row[c]);
        }
    } else {
        for (int64_t c = 0; c < q; c++) {
            dest_ptr[c * strideRes_1] = (uint8_t)(a_val * b_row[c * strideB_1]);
        }
    }
}

template <bool CONTIG_B, bool CONTIG_RES>
static inline void kron_row_complex128(
    const cpx_t a_val,
    const cpx_t * RESTRICT b_row, int64_t strideB_1,
    cpx_t * RESTRICT dest_ptr, int64_t strideRes_1,
    int64_t q)
{
    if constexpr (CONTIG_B && CONTIG_RES) {
        int64_t c = 0;
#if HAS_AVX2_REDUCTIONS
        const double *b_d = (const double *)b_row;
        double *res_d = (double *)dest_ptr;
        __m256d a_re = _mm256_set1_pd(a_val.r);
        __m256d a_im = _mm256_set1_pd(a_val.i);

        for (; c + 3 < q; c += 4) {
            __m256d vb0 = _mm256_loadu_pd(b_d + c * 2);
            __m256d vb1 = _mm256_loadu_pd(b_d + c * 2 + 4);

            __m256d b_swap0 = _mm256_permute_pd(vb0, 0b0101);
            __m256d b_swap1 = _mm256_permute_pd(vb1, 0b0101);

            __m256d prod0 = _mm256_mul_pd(a_im, b_swap0);
            __m256d prod1 = _mm256_mul_pd(a_im, b_swap1);

            __m256d r0 = _mm256_fmaddsub_pd(a_re, vb0, prod0);
            __m256d r1 = _mm256_fmaddsub_pd(a_re, vb1, prod1);

            _mm256_storeu_pd(res_d + c * 2, r0);
            _mm256_storeu_pd(res_d + c * 2 + 4, r1);
        }
        for (; c + 1 < q; c += 2) {
            __m256d vb0 = _mm256_loadu_pd(b_d + c * 2);
            __m256d b_swap0 = _mm256_permute_pd(vb0, 0b0101);
            __m256d prod0 = _mm256_mul_pd(a_im, b_swap0);
            __m256d r0 = _mm256_fmaddsub_pd(a_re, vb0, prod0);
            _mm256_storeu_pd(res_d + c * 2, r0);
        }
#endif
        for (; c < q; c++) {
            dest_ptr[c] = cpx_kron_mul(a_val, b_row[c]);
        }
    } else {
        for (int64_t c = 0; c < q; c++) {
            dest_ptr[c * strideRes_1] = cpx_kron_mul(a_val, b_row[c * strideB_1]);
        }
    }
}

template <bool CONTIG_B, bool CONTIG_RES>
static inline void kron_row_complex64(
    const cpx_f_t a_val,
    const cpx_f_t * RESTRICT b_row, int64_t strideB_1,
    cpx_f_t * RESTRICT dest_ptr, int64_t strideRes_1,
    int64_t q)
{
    if constexpr (CONTIG_B && CONTIG_RES) {
        int64_t c = 0;
#if HAS_AVX2_REDUCTIONS
        const float *b_f = (const float *)b_row;
        float *res_f = (float *)dest_ptr;
        __m256 a_re = _mm256_set1_ps(a_val.r);
        __m256 a_im = _mm256_set1_ps(a_val.i);

        for (; c + 7 < q; c += 8) {
            __m256 vb0 = _mm256_loadu_ps(b_f + c * 2);
            __m256 vb1 = _mm256_loadu_ps(b_f + c * 2 + 8);

            __m256 b_swap0 = _mm256_permute_ps(vb0, _MM_SHUFFLE(2, 3, 0, 1));
            __m256 b_swap1 = _mm256_permute_ps(vb1, _MM_SHUFFLE(2, 3, 0, 1));

            __m256 prod0 = _mm256_mul_ps(a_im, b_swap0);
            __m256 prod1 = _mm256_mul_ps(a_im, b_swap1);

            __m256 r0 = _mm256_fmaddsub_ps(a_re, vb0, prod0);
            __m256 r1 = _mm256_fmaddsub_ps(a_re, vb1, prod1);

            _mm256_storeu_ps(res_f + c * 2, r0);
            _mm256_storeu_ps(res_f + c * 2 + 8, r1);
        }
        for (; c + 3 < q; c += 4) {
            __m256 vb0 = _mm256_loadu_ps(b_f + c * 2);
            __m256 b_swap0 = _mm256_permute_ps(vb0, _MM_SHUFFLE(2, 3, 0, 1));
            __m256 prod0 = _mm256_mul_ps(a_im, b_swap0);
            __m256 r0 = _mm256_fmaddsub_ps(a_re, vb0, prod0);
            _mm256_storeu_ps(res_f + c * 2, r0);
        }
#endif
        for (; c < q; c++) {
            dest_ptr[c] = cpx_f_kron_mul(a_val, b_row[c]);
        }
    } else {
        for (int64_t c = 0; c < q; c++) {
            dest_ptr[c * strideRes_1] = cpx_f_kron_mul(a_val, b_row[c * strideB_1]);
        }
    }
}

template <bool CONTIG_B, bool CONTIG_RES>
static inline void kron_row_float16(
    const uint16_t a_val,
    const uint16_t * RESTRICT b_row, int64_t strideB_1,
    uint16_t * RESTRICT dest_ptr, int64_t strideRes_1,
    int64_t q)
{
    double a_d = decode_float16(a_val);
    if constexpr (CONTIG_B && CONTIG_RES) {
        for (int64_t c = 0; c < q; c++) {
            dest_ptr[c] = encode_float16(a_d * decode_float16(b_row[c]));
        }
    } else {
        for (int64_t c = 0; c < q; c++) {
            dest_ptr[c * strideRes_1] = encode_float16(a_d * decode_float16(b_row[c * strideB_1]));
        }
    }
}

template <bool CONTIG_B, bool CONTIG_RES>
static inline void kron_row_bfloat16(
    const uint16_t a_val,
    const uint16_t * RESTRICT b_row, int64_t strideB_1,
    uint16_t * RESTRICT dest_ptr, int64_t strideRes_1,
    int64_t q)
{
    double a_d = decode_bfloat16(a_val);
    if constexpr (CONTIG_B && CONTIG_RES) {
        for (int64_t c = 0; c < q; c++) {
            dest_ptr[c] = encode_bfloat16(a_d * decode_bfloat16(b_row[c]));
        }
    } else {
        for (int64_t c = 0; c < q; c++) {
            dest_ptr[c * strideRes_1] = encode_bfloat16(a_d * decode_bfloat16(b_row[c * strideB_1]));
        }
    }
}

template <bool CONTIG_B, bool CONTIG_RES>
static inline void kron_row_boolean(
    const uint8_t a_val,
    const uint8_t * RESTRICT b_row, int64_t strideB_1,
    uint8_t * RESTRICT dest_ptr, int64_t strideRes_1,
    int64_t q)
{
    if constexpr (CONTIG_B && CONTIG_RES) {
        if (!a_val) {
            memset(dest_ptr, 0, q);
        } else {
            for (int64_t c = 0; c < q; c++) {
                dest_ptr[c] = b_row[c] ? 1 : 0;
            }
        }
    } else {
        for (int64_t c = 0; c < q; c++) {
            dest_ptr[c * strideRes_1] = (uint8_t)((a_val != 0) && (b_row[c * strideB_1] != 0));
        }
    }
}

template <typename T, typename RowFuncContig, typename RowFuncStrided>
static void run_kron_2d(
    const T * RESTRICT a, int64_t strideA_0, int64_t strideA_1, int64_t m, int64_t n,
    const T * RESTRICT b, int64_t strideB_0, int64_t strideB_1, int64_t p, int64_t q,
    T * RESTRICT res, int64_t strideRes_0, int64_t strideRes_1,
    RowFuncContig contig_func, RowFuncStrided strided_func)
{
    bool is_contig = (strideB_1 == 1 && strideRes_1 == 1);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t rb = 0; rb < p; rb++) {
            int64_t row_c = i * p + rb;
            T * RESTRICT dest_row = res + row_c * strideRes_0;
            const T * RESTRICT b_row = b + rb * strideB_0;
            for (int64_t j = 0; j < n; j++) {
                T a_val = *(a + i * strideA_0 + j * strideA_1);
                T * RESTRICT dest_ptr = dest_row + j * q * strideRes_1;
                if (is_contig) {
                    contig_func(a_val, b_row, strideB_1, dest_ptr, strideRes_1, q);
                } else {
                    strided_func(a_val, b_row, strideB_1, dest_ptr, strideRes_1, q);
                }
            }
        }
    }
}

extern "C" void native_kron_2d(
    int dtype,
    const void *a, int64_t strideA_0, int64_t strideA_1, int64_t m, int64_t n,
    const void *b, int64_t strideB_0, int64_t strideB_1, int64_t p, int64_t q,
    void *res, int64_t strideRes_0, int64_t strideRes_1)
{
    if (a == nullptr || b == nullptr || res == nullptr || m <= 0 || n <= 0 || p <= 0 || q <= 0) return;
    switch (dtype) {
        case DTYPE_FLOAT64:
            run_kron_2d<double>((const double*)a, strideA_0, strideA_1, m, n,
                               (const double*)b, strideB_0, strideB_1, p, q,
                               (double*)res, strideRes_0, strideRes_1,
                               kron_row_double<true, true>, kron_row_double<false, false>);
            break;
        case DTYPE_FLOAT32:
            run_kron_2d<float>((const float*)a, strideA_0, strideA_1, m, n,
                              (const float*)b, strideB_0, strideB_1, p, q,
                              (float*)res, strideRes_0, strideRes_1,
                              kron_row_float<true, true>, kron_row_float<false, false>);
            break;
        case DTYPE_FLOAT16:
            run_kron_2d<uint16_t>((const uint16_t*)a, strideA_0, strideA_1, m, n,
                                 (const uint16_t*)b, strideB_0, strideB_1, p, q,
                                 (uint16_t*)res, strideRes_0, strideRes_1,
                                 kron_row_float16<true, true>, kron_row_float16<false, false>);
            break;
        case DTYPE_BFLOAT16:
            run_kron_2d<uint16_t>((const uint16_t*)a, strideA_0, strideA_1, m, n,
                                 (const uint16_t*)b, strideB_0, strideB_1, p, q,
                                 (uint16_t*)res, strideRes_0, strideRes_1,
                                 kron_row_bfloat16<true, true>, kron_row_bfloat16<false, false>);
            break;
        case DTYPE_INT64:
        case DTYPE_UINT64:
            run_kron_2d<int64_t>((const int64_t*)a, strideA_0, strideA_1, m, n,
                                (const int64_t*)b, strideB_0, strideB_1, p, q,
                                (int64_t*)res, strideRes_0, strideRes_1,
                                kron_row_int64<true, true>, kron_row_int64<false, false>);
            break;
        case DTYPE_INT32:
        case DTYPE_UINT32:
            run_kron_2d<int32_t>((const int32_t*)a, strideA_0, strideA_1, m, n,
                                (const int32_t*)b, strideB_0, strideB_1, p, q,
                                (int32_t*)res, strideRes_0, strideRes_1,
                                kron_row_int32<true, true>, kron_row_int32<false, false>);
            break;
        case DTYPE_INT8:
        case DTYPE_UINT8:
            run_kron_2d<uint8_t>((const uint8_t*)a, strideA_0, strideA_1, m, n,
                                (const uint8_t*)b, strideB_0, strideB_1, p, q,
                                (uint8_t*)res, strideRes_0, strideRes_1,
                                kron_row_uint8<true, true>, kron_row_uint8<false, false>);
            break;
        case DTYPE_INT16:
        case DTYPE_UINT16:
            run_kron_2d<int16_t>((const int16_t*)a, strideA_0, strideA_1, m, n,
                                (const int16_t*)b, strideB_0, strideB_1, p, q,
                                (int16_t*)res, strideRes_0, strideRes_1,
                                kron_row_int16<true, true>, kron_row_int16<false, false>);
            break;
        case DTYPE_COMPLEX128:
            run_kron_2d<cpx_t>((const cpx_t*)a, strideA_0, strideA_1, m, n,
                              (const cpx_t*)b, strideB_0, strideB_1, p, q,
                              (cpx_t*)res, strideRes_0, strideRes_1,
                              kron_row_complex128<true, true>, kron_row_complex128<false, false>);
            break;
        case DTYPE_COMPLEX64:
            run_kron_2d<cpx_f_t>((const cpx_f_t*)a, strideA_0, strideA_1, m, n,
                                (const cpx_f_t*)b, strideB_0, strideB_1, p, q,
                                (cpx_f_t*)res, strideRes_0, strideRes_1,
                                kron_row_complex64<true, true>, kron_row_complex64<false, false>);
            break;
        case DTYPE_BOOLEAN:
            run_kron_2d<uint8_t>((const uint8_t*)a, strideA_0, strideA_1, m, n,
                                (const uint8_t*)b, strideB_0, strideB_1, p, q,
                                (uint8_t*)res, strideRes_0, strideRes_1,
                                kron_row_boolean<true, true>, kron_row_boolean<false, false>);
            break;
        default:
            return;
    }
}

extern "C" void native_kron_nd(
    int dtype,
    const void *a, const int64_t *stridesA, const int64_t *shapeA,
    const void *b, const int64_t *stridesB, const int64_t *shapeB,
    void *res, const int64_t *stridesRes, const int64_t *shapeRes,
    int rank)
{
    if (a == nullptr || b == nullptr || res == nullptr || rank < 0) return;
    if (rank == 0) {
        native_kron_2d(dtype, a, 0, 0, 1, 1, b, 0, 0, 1, 1, res, 0, 0);
        return;
    }
    for (int64_t i = 0; i < rank; i++) {
        if (shapeRes[i] <= 0) return;
    }
    if (rank == 1) {
        native_kron_2d(dtype, a, 0, stridesA[0], 1, shapeA[0],
                       b, 0, stridesB[0], 1, shapeB[0],
                       res, 0, stridesRes[0]);
        return;
    }
    if (rank == 2) {
        native_kron_2d(dtype, a, stridesA[0], stridesA[1], shapeA[0], shapeA[1],
                       b, stridesB[0], stridesB[1], shapeB[0], shapeB[1],
                       res, stridesRes[0], stridesRes[1]);
        return;
    }

    int outer_rank = rank - 2;
    DECLARE_RANK_BUFFER(int64_t, coordA, outer_rank);
    DECLARE_RANK_BUFFER(int64_t, coordB, outer_rank);

    int64_t total_outer = 1;
    for (int64_t i = 0; i < outer_rank; i++) {
        total_outer *= (int64_t)shapeA[i] * (int64_t)shapeB[i];
    }

    int64_t m = shapeA[rank - 2];
    int64_t n = shapeA[rank - 1];
    int64_t p = shapeB[rank - 2];
    int64_t q = shapeB[rank - 1];

    int64_t strideA_0 = stridesA[rank - 2];
    int64_t strideA_1 = stridesA[rank - 1];
    int64_t strideB_0 = stridesB[rank - 2];
    int64_t strideB_1 = stridesB[rank - 1];
    int64_t strideRes_0 = stridesRes[rank - 2];
    int64_t strideRes_1 = stridesRes[rank - 1];

    int elem_size = 1;
    switch (dtype) {
        case DTYPE_FLOAT64:
        case DTYPE_INT64:
        case DTYPE_UINT64:
        case DTYPE_COMPLEX64:
            elem_size = 8;
            break;
        case DTYPE_FLOAT32:
        case DTYPE_INT32:
        case DTYPE_UINT32:
            elem_size = 4;
            break;
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
        case DTYPE_INT16:
        case DTYPE_UINT16:
            elem_size = 2;
            break;
        case DTYPE_COMPLEX128:
            elem_size = 16;
            break;
        case DTYPE_INT8:
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            elem_size = 1;
            break;
        default:
            return;
    }

    for (int64_t iter = 0; iter < total_outer; iter++) {
        int64_t offsetA = 0;
        int64_t offsetB = 0;
        int64_t offsetRes = 0;
        for (int d = 0; d < outer_rank; d++) {
            offsetA += (int64_t)coordA[d] * stridesA[d];
            offsetB += (int64_t)coordB[d] * stridesB[d];
            int64_t coordRes = coordA[d] * shapeB[d] + coordB[d];
            offsetRes += (int64_t)coordRes * stridesRes[d];
        }

        const char *a_ptr = (const char *)a + offsetA * elem_size;
        const char *b_ptr = (const char *)b + offsetB * elem_size;
        char *res_ptr = (char *)res + offsetRes * elem_size;

        native_kron_2d(dtype, a_ptr, strideA_0, strideA_1, m, n,
                       b_ptr, strideB_0, strideB_1, p, q,
                       res_ptr, strideRes_0, strideRes_1);

        for (int d = outer_rank - 1; d >= 0; d--) {
            coordB[d]++;
            if (coordB[d] < shapeB[d]) break;
            coordB[d] = 0;
            coordA[d]++;
            if (coordA[d] < shapeA[d]) break;
            coordA[d] = 0;
        }
    }
}

void s_kron_double(const double *a, const int64_t *stridesA, const int64_t *shapeA,
                   const double *b, const int64_t *stridesB, const int64_t *shapeB,
                   double *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank) {
    native_kron_nd(DTYPE_FLOAT64, a, stridesA, shapeA, b, stridesB, shapeB, res, stridesRes, shapeRes, rank);
}
void s_kron_float(const float *a, const int64_t *stridesA, const int64_t *shapeA,
                  const float *b, const int64_t *stridesB, const int64_t *shapeB,
                  float *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank) {
    native_kron_nd(DTYPE_FLOAT32, a, stridesA, shapeA, b, stridesB, shapeB, res, stridesRes, shapeRes, rank);
}
void s_kron_int64(const int64_t *a, const int64_t *stridesA, const int64_t *shapeA,
                  const int64_t *b, const int64_t *stridesB, const int64_t *shapeB,
                  int64_t *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank) {
    native_kron_nd(DTYPE_INT64, a, stridesA, shapeA, b, stridesB, shapeB, res, stridesRes, shapeRes, rank);
}
void s_kron_int32(const int32_t *a, const int64_t *stridesA, const int64_t *shapeA,
                  const int32_t *b, const int64_t *stridesB, const int64_t *shapeB,
                  int32_t *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank) {
    native_kron_nd(DTYPE_INT32, a, stridesA, shapeA, b, stridesB, shapeB, res, stridesRes, shapeRes, rank);
}
void s_kron_uint8(const uint8_t *a, const int64_t *stridesA, const int64_t *shapeA,
                  const uint8_t *b, const int64_t *stridesB, const int64_t *shapeB,
                  uint8_t *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank) {
    native_kron_nd(DTYPE_UINT8, a, stridesA, shapeA, b, stridesB, shapeB, res, stridesRes, shapeRes, rank);
}
void s_kron_int16(const int16_t *a, const int64_t *stridesA, const int64_t *shapeA,
                  const int16_t *b, const int64_t *stridesB, const int64_t *shapeB,
                  int16_t *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank) {
    native_kron_nd(DTYPE_INT16, a, stridesA, shapeA, b, stridesB, shapeB, res, stridesRes, shapeRes, rank);
}
void s_kron_complex128(const cpx_t *a, const int64_t *stridesA, const int64_t *shapeA,
                       const cpx_t *b, const int64_t *stridesB, const int64_t *shapeB,
                       cpx_t *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank) {
    native_kron_nd(DTYPE_COMPLEX128, a, stridesA, shapeA, b, stridesB, shapeB, res, stridesRes, shapeRes, rank);
}
void s_kron_complex64(const cpx_f_t *a, const int64_t *stridesA, const int64_t *shapeA,
                      const cpx_f_t *b, const int64_t *stridesB, const int64_t *shapeB,
                      cpx_f_t *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank) {
    native_kron_nd(DTYPE_COMPLEX64, a, stridesA, shapeA, b, stridesB, shapeB, res, stridesRes, shapeRes, rank);
}
void s_kron_boolean(const uint8_t *a, const int64_t *stridesA, const int64_t *shapeA,
                    const uint8_t *b, const int64_t *stridesB, const int64_t *shapeB,
                    uint8_t *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank) {
    native_kron_nd(DTYPE_BOOLEAN, a, stridesA, shapeA, b, stridesB, shapeB, res, stridesRes, shapeRes, rank);
}

/* Vector Outer Product */
#define NUM_MUL_OP(x, y) ((x) * (y))
#define INT64_MUL_OP(x, y) ((int64_t)((uint64_t)(x) * (uint64_t)(y)))
#define INT32_MUL_OP(x, y) ((int32_t)((uint32_t)(x) * (uint32_t)(y)))
#define INT16_MUL_OP(x, y) ((int16_t)((uint16_t)(x) * (uint16_t)(y)))
#define BOOL_AND_OP(x, y) (((x) != 0 && (y) != 0) ? 1 : 0)
#define DEFINE_OUTER_IMPL(name, type, op) \
void name(const type *a, int64_t strideA, int64_t sizeA, \
          const type *b, int64_t strideB, int64_t sizeB, \
          type *res, int64_t strideRowRes, int64_t strideColRes) { \
    if (a == nullptr || b == nullptr || res == nullptr || sizeA <= 0 || sizeB <= 0) return; \
    for (int64_t i = 0; i < sizeA; i++) { \
        type valA = a[i * strideA]; \
        for (int64_t j = 0; j < sizeB; j++) { \
            res[i * strideRowRes + j * strideColRes] = op(valA, b[j * strideB]); \
        } \
    } \
}

DEFINE_OUTER_IMPL(s_outer_double, double, NUM_MUL_OP)
DEFINE_OUTER_IMPL(s_outer_float, float, NUM_MUL_OP)
DEFINE_OUTER_IMPL(s_outer_int64, int64_t, INT64_MUL_OP)
DEFINE_OUTER_IMPL(s_outer_int32, int32_t, INT32_MUL_OP)
DEFINE_OUTER_IMPL(s_outer_uint8, uint8_t, NUM_MUL_OP)
DEFINE_OUTER_IMPL(s_outer_int16, int16_t, INT16_MUL_OP)
DEFINE_OUTER_IMPL(s_outer_boolean, uint8_t, BOOL_AND_OP)
DEFINE_OUTER_IMPL(s_outer_complex128, cpx_t, cpx_kron_mul)
DEFINE_OUTER_IMPL(s_outer_complex64, cpx_f_t, cpx_f_kron_mul)

/* Vector Cross Product */
#define DEFINE_CROSS_3D_IMPL(name, type, op_mul, op_sub) \
void name(const type *a, int64_t strideA, const type *b, int64_t strideB, type *res, int64_t strideRes) { \
    type a0 = a[0], a1 = a[strideA], a2 = a[2 * strideA]; \
    type b0 = b[0], b1 = b[strideB], b2 = b[2 * strideB]; \
    res[0] = op_sub(op_mul(a1, b2), op_mul(a2, b1)); \
    res[strideRes] = op_sub(op_mul(a2, b0), op_mul(a0, b2)); \
    res[2 * strideRes] = op_sub(op_mul(a0, b1), op_mul(a1, b0)); \
}

#define DEFINE_CROSS_2D_IMPL(name, type, op_mul, op_sub) \
void name(const type *a, int64_t strideA, const type *b, int64_t strideB, type *res) { \
    type a0 = a[0], a1 = a[strideA]; \
    type b0 = b[0], b1 = b[strideB]; \
    res[0] = op_sub(op_mul(a0, b1), op_mul(a1, b0)); \
}

#define NUM_SUB_OP(x, y) ((x) - (y))
#define INT64_SUB_OP(x, y) ((int64_t)((uint64_t)(x) - (uint64_t)(y)))
#define INT32_SUB_OP(x, y) ((int32_t)((uint32_t)(x) - (uint32_t)(y)))
#define INT16_SUB_OP(x, y) ((int16_t)((uint16_t)(x) - (uint16_t)(y)))
#define BOOL_SUB_OP(x, y) ((((x) != 0) != ((y) != 0)) ? 1 : 0)

DEFINE_CROSS_3D_IMPL(s_cross_3d_double, double, NUM_MUL_OP, NUM_SUB_OP)
DEFINE_CROSS_2D_IMPL(s_cross_2d_double, double, NUM_MUL_OP, NUM_SUB_OP)

DEFINE_CROSS_3D_IMPL(s_cross_3d_float, float, NUM_MUL_OP, NUM_SUB_OP)
DEFINE_CROSS_2D_IMPL(s_cross_2d_float, float, NUM_MUL_OP, NUM_SUB_OP)

DEFINE_CROSS_3D_IMPL(s_cross_3d_int64, int64_t, INT64_MUL_OP, INT64_SUB_OP)
DEFINE_CROSS_2D_IMPL(s_cross_2d_int64, int64_t, INT64_MUL_OP, INT64_SUB_OP)

DEFINE_CROSS_3D_IMPL(s_cross_3d_int32, int32_t, INT32_MUL_OP, INT32_SUB_OP)
DEFINE_CROSS_2D_IMPL(s_cross_2d_int32, int32_t, INT32_MUL_OP, INT32_SUB_OP)

DEFINE_CROSS_3D_IMPL(s_cross_3d_uint8, uint8_t, NUM_MUL_OP, NUM_SUB_OP)
DEFINE_CROSS_2D_IMPL(s_cross_2d_uint8, uint8_t, NUM_MUL_OP, NUM_SUB_OP)

DEFINE_CROSS_3D_IMPL(s_cross_3d_int16, int16_t, INT16_MUL_OP, INT16_SUB_OP)
DEFINE_CROSS_2D_IMPL(s_cross_2d_int16, int16_t, INT16_MUL_OP, INT16_SUB_OP)

DEFINE_CROSS_3D_IMPL(s_cross_3d_boolean, uint8_t, BOOL_AND_OP, BOOL_SUB_OP)
DEFINE_CROSS_2D_IMPL(s_cross_2d_boolean, uint8_t, BOOL_AND_OP, BOOL_SUB_OP)

DEFINE_CROSS_3D_IMPL(s_cross_3d_complex128, cpx_t, cpx_kron_mul, cpx_sub)
DEFINE_CROSS_2D_IMPL(s_cross_2d_complex128, cpx_t, cpx_kron_mul, cpx_sub)

DEFINE_CROSS_3D_IMPL(s_cross_3d_complex64, cpx_f_t, cpx_f_kron_mul, cpx_sub_f)
DEFINE_CROSS_2D_IMPL(s_cross_2d_complex64, cpx_f_t, cpx_f_kron_mul, cpx_sub_f)

/* Vector Norm Reductions */
#define DEFINE_NORM_REDUCTIONS(suffix, type) \
double r_norm_l1_##suffix(const type *src, int64_t stride, int64_t size) { \
    if (src == nullptr || size <= 0) return 0.0; \
    double sum = 0.0; \
    for (int64_t i = 0; i < size; i++) { \
        double val = (double)src[i * stride]; \
        sum += std::abs(val); \
    } \
    return sum; \
} \
double r_norm_l2_##suffix(const type *src, int64_t stride, int64_t size) { \
    if (src == nullptr || size <= 0) return 0.0; \
    double sum = 0.0; \
    for (int64_t i = 0; i < size; i++) { \
        double val = (double)src[i * stride]; \
        sum += val * val; \
    } \
    return sum; \
} \
double r_norm_lp_##suffix(const type *src, int64_t stride, int64_t size, double p) { \
    if (src == nullptr || size <= 0) return 0.0; \
    double sum = 0.0; \
    for (int64_t i = 0; i < size; i++) { \
        double val = (double)src[i * stride]; \
        double abs_val = std::abs(val); \
        if (p == 0.0) { \
            if ( std::isnan(abs_val)) return NAN; \
            if (abs_val != 0.0) sum += 1.0; \
        } else { \
            sum += std::pow(abs_val, p); \
        } \
    } \
    return sum; \
} \
double r_norm_inf_##suffix(const type *src, int64_t stride, int64_t size) { \
    if (src == nullptr || size <= 0) return 0.0; \
    double max_val = -1.0; \
    for (int64_t i = 0; i < size; i++) { \
        double val = (double)src[i * stride]; \
        double abs_val = std::abs(val); \
        if (std::isnan(abs_val)) return NAN; \
        if (abs_val > max_val || max_val < 0) max_val = abs_val; \
    } \
    return max_val; \
} \
double r_norm_neg_inf_##suffix(const type *src, int64_t stride, int64_t size) { \
    if (src == nullptr || size <= 0) return 0.0; \
    double min_val = -1.0; \
    for (int64_t i = 0; i < size; i++) { \
        double val = (double)src[i * stride]; \
        double abs_val = std::abs(val); \
        if (std::isnan(abs_val)) return NAN; \
        if (abs_val < min_val || min_val < 0) min_val = abs_val; \
    } \
    return min_val; \
}

DEFINE_NORM_REDUCTIONS(double, double)

#define DEFINE_NORM_REDUCTIONS_FLOAT(suffix, type) \
float r_norm_l1_##suffix(const type *src, int64_t stride, int64_t size) { \
    if (src == nullptr || size <= 0) return 0.0f; \
    float sum = 0.0f; \
    for (int64_t i = 0; i < size; i++) { \
        float val = (float)src[i * stride]; \
        sum += std::abs(val); \
    } \
    return sum; \
} \
float r_norm_l2_##suffix(const type *src, int64_t stride, int64_t size) { \
    if (src == nullptr || size <= 0) return 0.0f; \
    float sum = 0.0f; \
    for (int64_t i = 0; i < size; i++) { \
        float val = (float)src[i * stride]; \
        sum += val * val; \
    } \
    return sum; \
} \
float r_norm_lp_##suffix(const type *src, int64_t stride, int64_t size, float p) { \
    if (src == nullptr || size <= 0) return 0.0f; \
    float sum = 0.0f; \
    for (int64_t i = 0; i < size; i++) { \
        float val = (float)src[i * stride]; \
        float abs_val = std::abs(val); \
        if (p == 0.0f) { \
            if (std::isnan(abs_val)) return NAN; \
            if (abs_val != 0.0f) sum += 1.0f; \
        } else { \
            sum += std::pow(abs_val, p); \
        } \
    } \
    return sum; \
} \
float r_norm_inf_##suffix(const type *src, int64_t stride, int64_t size) { \
    if (src == nullptr || size <= 0) return 0.0f; \
    float max_val = -1.0f; \
    for (int64_t i = 0; i < size; i++) { \
        float val = (float)src[i * stride]; \
        float abs_val = std::abs(val); \
        if (std::isnan(abs_val)) return NAN; \
        if (abs_val > max_val || max_val < 0) max_val = abs_val; \
    } \
    return max_val; \
} \
float r_norm_neg_inf_##suffix(const type *src, int64_t stride, int64_t size) { \
    if (src == nullptr || size <= 0) return 0.0f; \
    float min_val = -1.0f; \
    for (int64_t i = 0; i < size; i++) { \
        float val = (float)src[i * stride]; \
        float abs_val = std::abs(val); \
        if (std::isnan(abs_val)) return NAN; \
        if (abs_val < min_val || min_val < 0) min_val = abs_val; \
    } \
    return min_val; \
}

DEFINE_NORM_REDUCTIONS_FLOAT(float, float)

/* Complex Norms */
#define DEFINE_COMPLEX_NORM_REDUCTIONS(suffix, type, float_type, sqrt_fn, pow_fn) \
float_type r_norm_l1_##suffix(const type *src, int64_t stride, int64_t size) { \
    if (src == nullptr || size <= 0) return (float_type)0.0; \
    float_type sum = (float_type)0.0; \
    for (int64_t i = 0; i < size; i++) { \
        float_type r = (float_type)src[i * stride].r; \
        float_type imag = (float_type)src[i * stride].i; \
        sum += std::hypot(r, imag); \
    } \
    return sum; \
} \
float_type r_norm_l2_##suffix(const type *src, int64_t stride, int64_t size) { \
    if (src == nullptr || size <= 0) return (float_type)0.0; \
    float_type sum = (float_type)0.0; \
    for (int64_t i = 0; i < size; i++) { \
        float_type r = (float_type)src[i * stride].r; \
        float_type imag = (float_type)src[i * stride].i; \
        sum += r * r + imag * imag; \
    } \
    return sum; \
} \
float_type r_norm_lp_##suffix(const type *src, int64_t stride, int64_t size, float_type p) { \
    if (src == nullptr || size <= 0) return (float_type)0.0; \
    float_type sum = (float_type)0.0; \
    for (int64_t i = 0; i < size; i++) { \
        float_type r = (float_type)src[i * stride].r; \
        float_type imag = (float_type)src[i * stride].i; \
        if (p == (float_type)0.0) { \
            if (std::isnan(r) || std::isnan(imag)) return NAN; \
            if (r != (float_type)0.0 || imag != (float_type)0.0) sum += (float_type)1.0; \
        } else { \
            sum += pow_fn(std::hypot(r, imag), p); \
        } \
    } \
    return sum; \
} \
float_type r_norm_inf_##suffix(const type *src, int64_t stride, int64_t size) { \
    if (src == nullptr || size <= 0) return (float_type)0.0; \
    float_type max_val = (float_type)-1.0; \
    for (int64_t i = 0; i < size; i++) { \
        float_type r = (float_type)src[i * stride].r; \
        float_type imag = (float_type)src[i * stride].i; \
        float_type val = std::hypot(r, imag); \
        if (std::isnan(val)) return NAN; \
        if (val > max_val || max_val < (float_type)0.0) max_val = val; \
    } \
    return max_val; \
} \
float_type r_norm_neg_inf_##suffix(const type *src, int64_t stride, int64_t size) { \
    if (src == nullptr || size <= 0) return (float_type)0.0; \
    float_type min_val = (float_type)-1.0; \
    for (int64_t i = 0; i < size; i++) { \
        float_type r = (float_type)src[i * stride].r; \
        float_type imag = (float_type)src[i * stride].i; \
        float_type val = std::hypot(r, imag); \
        if (std::isnan(val)) return NAN; \
        if (val < min_val || min_val < (float_type)0.0) min_val = val; \
    } \
    return min_val; \
}

DEFINE_COMPLEX_NORM_REDUCTIONS(complex128, cpx_t, double, std::sqrt, std::pow)
DEFINE_COMPLEX_NORM_REDUCTIONS(complex64, cpx_f_t, float, std::sqrt, std::pow)

/* Window Functions */
void v_hanning_double(double *res, int64_t M) {
    if (res == nullptr || M <= 0) return;
    if (M == 1) {
        res[0] = 1.0;
        return;
    }
    double pi2 = 2.0 * M_PI;
    for (int64_t n = 0; n < M; n++) {
        res[n] = 0.5 - 0.5 * std::cos(pi2 * n / (M - 1));
    }
}

void v_hanning_float(float *res, int64_t M) {
    if (res == nullptr || M <= 0) return;
    if (M == 1) {
        res[0] = 1.0f;
        return;
    }
    float pi2 = 2.0f * (float)M_PI;
    for (int64_t n = 0; n < M; n++) {
        res[n] = 0.5f - 0.5f * std::cos(pi2 * n / (M - 1));
    }
}

void v_hamming_double(double *res, int64_t M) {
    if (res == nullptr || M <= 0) return;
    if (M == 1) {
        res[0] = 1.0;
        return;
    }
    double pi2 = 2.0 * M_PI;
    for (int64_t n = 0; n < M; n++) {
        res[n] = 0.54 - 0.46 * std::cos(pi2 * n / (M - 1));
    }
}

void v_hamming_float(float *res, int64_t M) {
    if (res == nullptr || M <= 0) return;
    if (M == 1) {
        res[0] = 1.0f;
        return;
    }
    float pi2 = 2.0f * (float)M_PI;
    for (int64_t n = 0; n < M; n++) {
        res[n] = 0.54f - 0.46f * std::cos(pi2 * n / (M - 1));
    }
}

// ============================================================================
// 14. STRIDED TERNARY CLIP FUNCTIONS FOR ALL NUMERIC TYPES
// ============================================================================

template <typename T>
static inline T clip_scalar_val(T val, T mn, T mx) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(val) || std::isnan(mn) || std::isnan(mx)) {
            return std::numeric_limits<T>::quiet_NaN();
        }
    }
    if (val < mn) val = mn;
    if (val > mx) val = mx;
    return val;
}

#define IMPLEMENT_S_CLIP(TYPE_NAME, TYPE) \
void s_clip_##TYPE_NAME(const TYPE *a, const int64_t *stridesA, \
                        const TYPE *min_val, const int64_t *stridesMin, \
                        const TYPE *max_val, const int64_t *stridesMax, \
                        TYPE *res, const int64_t *stridesRes, \
                        const int64_t *shape, int rank, const uint8_t *mask) { \
    if (a == nullptr || min_val == nullptr || max_val == nullptr || res == nullptr || rank < 0) return; \
    if (rank > 0 && (stridesA == nullptr || stridesMin == nullptr || stridesMax == nullptr || stridesRes == nullptr || shape == nullptr)) return; \
    int64_t total_elements = 1; \
    for (int64_t i = 0; i < rank; i++) { \
        if (shape[i] <= 0) return; \
        total_elements *= shape[i]; \
    } \
    if (rank == 0) { \
        if (mask != nullptr && mask[0] == 0) return; \
        res[0] = clip_scalar_val<TYPE>(a[0], min_val[0], max_val[0]); \
        return; \
    } \
    bool overlap = strided_buffers_overlap_not_identical(a, stridesA, sizeof(TYPE), res, stridesRes, sizeof(TYPE), shape, rank) || \
                   strided_buffers_overlap_not_identical(min_val, stridesMin, sizeof(TYPE), res, stridesRes, sizeof(TYPE), shape, rank) || \
                   strided_buffers_overlap_not_identical(max_val, stridesMax, sizeof(TYPE), res, stridesRes, sizeof(TYPE), shape, rank); \
    if (overlap) { \
        NoThrowBuffer<TYPE> temp(total_elements); \
        if (!temp.ok()) { \
            ndarray_set_oom_flag(); \
            return; \
        } \
        DECLARE_RANK_BUFFER(int64_t, coord, rank); \
        int64_t offsetA = 0, offsetMin = 0, offsetMax = 0; \
        for (int64_t el = 0; el < total_elements; el++) { \
            if (mask == nullptr || mask[el] != 0) { \
                temp[el] = clip_scalar_val<TYPE>(a[offsetA], min_val[offsetMin], max_val[offsetMax]); \
            } \
            for (int d = rank - 1; d >= 0; d--) { \
                coord[d]++; \
                if (coord[d] < shape[d]) { \
                    offsetA   += stridesA[d]; \
                    offsetMin += stridesMin[d]; \
                    offsetMax += stridesMax[d]; \
                    break; \
                } \
                coord[d] = 0; \
                offsetA   -= (shape[d] - 1) * stridesA[d]; \
                offsetMin -= (shape[d] - 1) * stridesMin[d]; \
                offsetMax -= (shape[d] - 1) * stridesMax[d]; \
            } \
        } \
        DECLARE_RANK_BUFFER(int64_t, coordRes, rank); \
        int64_t offsetRes = 0; \
        for (int64_t el = 0; el < total_elements; el++) { \
            if (mask == nullptr || mask[el] != 0) { \
                res[offsetRes] = temp[el]; \
            } \
            for (int d = rank - 1; d >= 0; d--) { \
                coordRes[d]++; \
                if (coordRes[d] < shape[d]) { \
                    offsetRes += stridesRes[d]; \
                    break; \
                } \
                coordRes[d] = 0; \
                offsetRes -= (shape[d] - 1) * stridesRes[d]; \
            } \
        } \
        return; \
    } \
    int is_contiguous = 1; \
    int64_t expected_stride = 1; \
    for (int64_t i = rank - 1; i >= 0; i--) { \
        if (stridesA[i] != expected_stride || \
            stridesMin[i] != expected_stride || \
            stridesMax[i] != expected_stride || \
            stridesRes[i] != expected_stride) { \
            is_contiguous = 0; \
            break; \
        } \
        expected_stride *= shape[i]; \
    } \
    if (is_contiguous) { \
        for (int64_t i = 0; i < total_elements; i++) { \
            if (mask != nullptr && mask[i] == 0) continue; \
            res[i] = clip_scalar_val<TYPE>(a[i], min_val[i], max_val[i]); \
        } \
        return; \
    } \
    DECLARE_RANK_BUFFER(int64_t, coord, rank); \
    int64_t offsetA = 0, offsetMin = 0, offsetMax = 0, offsetRes = 0; \
    for (int64_t el = 0; el < total_elements; el++) { \
        if (mask == nullptr || mask[el] != 0) { \
            res[offsetRes] = clip_scalar_val<TYPE>(a[offsetA], min_val[offsetMin], max_val[offsetMax]); \
        } \
        for (int d = rank - 1; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetA   += stridesA[d]; \
                offsetMin += stridesMin[d]; \
                offsetMax += stridesMax[d]; \
                offsetRes += stridesRes[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetA   -= (shape[d] - 1) * stridesA[d]; \
            offsetMin -= (shape[d] - 1) * stridesMin[d]; \
            offsetMax -= (shape[d] - 1) * stridesMax[d]; \
            offsetRes -= (shape[d] - 1) * stridesRes[d]; \
        } \
    } \
}

IMPLEMENT_S_CLIP(double, double)
IMPLEMENT_S_CLIP(float, float)
IMPLEMENT_S_CLIP(int64, int64_t)
IMPLEMENT_S_CLIP(int32, int32_t)
IMPLEMENT_S_CLIP(uint8, uint8_t)
IMPLEMENT_S_CLIP(int16, int16_t)

// ============================================================================
// 15. CALCULUS SOLVERS: TRAPEZOIDAL INTEGRATION & N-DIMENSIONAL GRADIENT
// ============================================================================

void s_trapz_double(const double *y, const int64_t *stridesY,
                    const double *x, int64_t strideX, double dx,
                    double *res, const int64_t *stridesRes,
                    const int64_t *shape, int rank, int axis) {
    if (y == nullptr || res == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetRes = 0;
        int64_t offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesY[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesRes[targetD];
                }
            }
        }
        
        double sum = 0.0;
        for (int64_t i = 0; i < shape[axis] - 1; i++) {
            int64_t idxSrc = offsetSrc + i * stridesY[axis];
            int64_t idxSrcNext = offsetSrc + (i + 1) * stridesY[axis];
            double y_curr = y[idxSrc];
            double y_next = y[idxSrcNext];
            
            double h = dx;
            if (x != nullptr) {
                h = x[(i + 1) * strideX] - x[i * strideX];
            }
            sum += 0.5 * (y_curr + y_next) * h;
        }
        
        res[offsetRes] = sum;
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_trapz_float(const float *y, const int64_t *stridesY,
                   const float *x, int64_t strideX, float dx,
                   float *res, const int64_t *stridesRes,
                   const int64_t *shape, int rank, int axis) {
    if (y == nullptr || res == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetRes = 0;
        int64_t offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesY[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesRes[targetD];
                }
            }
        }
        
        float sum = 0.0f;
        for (int64_t i = 0; i < shape[axis] - 1; i++) {
            int64_t idxSrc = offsetSrc + i * stridesY[axis];
            int64_t idxSrcNext = offsetSrc + (i + 1) * stridesY[axis];
            float y_curr = y[idxSrc];
            float y_next = y[idxSrcNext];
            
            float h = dx;
            if (x != nullptr) {
                h = x[(i + 1) * strideX] - x[i * strideX];
            }
            sum += 0.5f * (y_curr + y_next) * h;
        }
        
        res[offsetRes] = sum;
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_trapz_complex128(const cpx_t *y, const int64_t *stridesY,
                        const double *x, int64_t strideX, double dx,
                        cpx_t *res, const int64_t *stridesRes,
                        const int64_t *shape, int rank, int axis) {
    if (y == nullptr || res == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetRes = 0;
        int64_t offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesY[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesRes[targetD];
                }
            }
        }
        
        cpx_t sum = {0.0, 0.0};
        for (int64_t i = 0; i < shape[axis] - 1; i++) {
            int64_t idxSrc = offsetSrc + i * stridesY[axis];
            int64_t idxSrcNext = offsetSrc + (i + 1) * stridesY[axis];
            cpx_t y_curr = y[idxSrc];
            cpx_t y_next = y[idxSrcNext];
            
            double h = dx;
            if (x != nullptr) {
                h = x[(i + 1) * strideX] - x[i * strideX];
            }
            sum.r += 0.5 * (y_curr.r + y_next.r) * h;
            sum.i += 0.5 * (y_curr.i + y_next.i) * h;
        }
        
        res[offsetRes] = sum;
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_trapz_complex64(const cpx_f_t *y, const int64_t *stridesY,
                   const float *x, int64_t strideX, float dx,
                   cpx_f_t *res, const int64_t *stridesRes,
                   const int64_t *shape, int rank, int axis) {
    if (y == nullptr || res == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }

    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetRes = 0;
        int64_t offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesY[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesRes[targetD];
                }
            }
        }

        cpx_f_t sum = {0.0f, 0.0f};
        for (int64_t i = 0; i < shape[axis] - 1; i++) {
            int64_t idxSrc = offsetSrc + i * stridesY[axis];
            int64_t idxSrcNext = offsetSrc + (i + 1) * stridesY[axis];
            cpx_f_t y_curr = y[idxSrc];
            cpx_f_t y_next = y[idxSrcNext];

            float h = dx;
            if (x != nullptr) {
                h = x[(i + 1) * strideX] - x[i * strideX];
            }
            sum.r += 0.5f * (y_curr.r + y_next.r) * h;
            sum.i += 0.5f * (y_curr.i + y_next.i) * h;
        }

        res[offsetRes] = sum;

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_trapz_complex128_all(const cpx_t *y, const int64_t *stridesY,
                            const cpx_t *x, int64_t strideX, cpx_t dx,
                            cpx_t *res, const int64_t *stridesRes,
                            const int64_t *shape, int rank, int axis) {
    if (y == nullptr || res == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }

    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetRes = 0;
        int64_t offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesY[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesRes[targetD];
                }
            }
        }

        cpx_t sum = {0.0, 0.0};
        for (int64_t i = 0; i < shape[axis] - 1; i++) {
            int64_t idxSrc = offsetSrc + i * stridesY[axis];
            int64_t idxSrcNext = offsetSrc + (i + 1) * stridesY[axis];
            cpx_t y_curr = y[idxSrc];
            cpx_t y_next = y[idxSrcNext];

            cpx_t h = dx;
            if (x != nullptr) {
                h.r = x[(i + 1) * strideX].r - x[i * strideX].r;
                h.i = x[(i + 1) * strideX].i - x[i * strideX].i;
            }

            double yr = (y_curr.r + y_next.r) * 0.5;
            double yi = (y_curr.i + y_next.i) * 0.5;

            // Complex multiplication: (yr + i*yi) * (h.r + i*h.i)
            sum.r += yr * h.r - yi * h.i;
            sum.i += yr * h.i + yi * h.r;
        }

        res[offsetRes] = sum;

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}
void s_gradient_double(const double *src, const int64_t *stridesSrc,
                       const double *x, int64_t strideX, double dx,
                       double *res, const int64_t *stridesRes,
                       const int64_t *shape, int rank, int axis, int edge_order) {
    if (src == nullptr || res == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    
    int64_t N = shape[axis];
    if (N <= 0) return;
    
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (shape[d] <= 0) return;
        if (d != axis) outer_size *= shape[d];
    }
    
    NoThrowBuffer<double> slice_buf(N);
    if (!slice_buf) {
        ndarray_set_oom_flag();
        return;
    }

    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetSrc = 0;
        int64_t offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetSrc += coord[d] * stridesSrc[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        for (int64_t i = 0; i < N; i++) {
            slice_buf[i] = src[offsetSrc + i * stridesSrc[axis]];
        }
        
        if (N == 1) {
            res[offsetRes] = 0.0;
        } else if (N == 2) {
            double h = dx;
            if (x != nullptr) {
                h = x[strideX] - x[0];
            }
            double diff = (slice_buf[1] - slice_buf[0]) / h;
            res[offsetRes] = diff;
            res[offsetRes + stridesRes[axis]] = diff;
        } else {
            // Left boundary (i = 0)
            if (edge_order == 1) {
                double h = dx;
                if (x != nullptr) {
                    h = x[strideX] - x[0];
                }
                double diff = (slice_buf[1] - slice_buf[0]) / h;
                res[offsetRes] = diff;
            } else {
                double h0 = dx;
                double h1 = dx;
                if (x != nullptr) {
                    h0 = x[strideX] - x[0];
                    h1 = x[2 * strideX] - x[strideX];
                }
                double f0 = slice_buf[0];
                double f1 = slice_buf[1];
                double f2 = slice_buf[2];
                
                double a = -(2.0 * h0 + h1) / (h0 * (h0 + h1));
                double b = (h0 + h1) / (h0 * h1);
                double c = -h0 / (h1 * (h0 + h1));
                
                res[offsetRes] = a * f0 + b * f1 + c * f2;
            }
            
            // Interior points
            for (int64_t i = 1; i < N - 1; i++) {
                int64_t idxRes = offsetRes + i * stridesRes[axis];
                
                double f_curr = slice_buf[i];
                double f_prev = slice_buf[i - 1];
                double f_next = slice_buf[i + 1];
                
                double h_s = dx;
                double h_d = dx;
                if (x != nullptr) {
                    h_s = x[i * strideX] - x[(i - 1) * strideX];
                    h_d = x[(i + 1) * strideX] - x[i * strideX];
                }
                
                res[idxRes] = (h_s * h_s * f_next + (h_d * h_d - h_s * h_s) * f_curr - h_d * h_d * f_prev) / (h_s * h_d * (h_s + h_d));
            }
            
            // Right boundary
            int64_t idxResEnd = offsetRes + (N - 1) * stridesRes[axis];
            if (edge_order == 1) {
                double h = dx;
                if (x != nullptr) {
                    h = x[(N - 1) * strideX] - x[(N - 2) * strideX];
                }
                res[idxResEnd] = (slice_buf[N - 1] - slice_buf[N - 2]) / h;
            } else {
                double h0 = dx;
                double h1 = dx;
                if (x != nullptr) {
                    h0 = x[(N - 2) * strideX] - x[(N - 3) * strideX];
                    h1 = x[(N - 1) * strideX] - x[(N - 2) * strideX];
                }
                double f0 = slice_buf[N - 3];
                double f1 = slice_buf[N - 2];
                double f2 = slice_buf[N - 1];
                
                double a = h1 / (h0 * (h0 + h1));
                double b = -(h0 + h1) / (h0 * h1);
                double c = (2.0 * h1 + h0) / (h1 * (h0 + h1));
                
                res[idxResEnd] = a * f0 + b * f1 + c * f2;
            }
        }
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_gradient_float(const float *src, const int64_t *stridesSrc,
                      const float *x, int64_t strideX, float dx,
                      float *res, const int64_t *stridesRes,
                      const int64_t *shape, int rank, int axis, int edge_order) {
    if (src == nullptr || res == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    
    int64_t N = shape[axis];
    if (N <= 0) return;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (shape[d] <= 0) return;
        if (d != axis) outer_size *= shape[d];
    }
    
    NoThrowBuffer<float> slice_buf(N);
    if (!slice_buf) {
        ndarray_set_oom_flag();
        return;
    }

    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetSrc = 0;
        int64_t offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetSrc += coord[d] * stridesSrc[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        for (int64_t i = 0; i < N; i++) {
            slice_buf[i] = src[offsetSrc + i * stridesSrc[axis]];
        }
        
        if (N == 1) {
            res[offsetRes] = 0.0f;
        } else if (N == 2) {
            float h = dx;
            if (x != nullptr) {
                h = x[strideX] - x[0];
            }
            float diff = (slice_buf[1] - slice_buf[0]) / h;
            res[offsetRes] = diff;
            res[offsetRes + stridesRes[axis]] = diff;
        } else {
            // Left boundary (i = 0)
            if (edge_order == 1) {
                float h = dx;
                if (x != nullptr) {
                    h = x[strideX] - x[0];
                }
                res[offsetRes] = (slice_buf[1] - slice_buf[0]) / h;
            } else {
                float h0 = dx;
                float h1 = dx;
                if (x != nullptr) {
                    h0 = x[strideX] - x[0];
                    h1 = x[2 * strideX] - x[strideX];
                }
                float f0 = slice_buf[0];
                float f1 = slice_buf[1];
                float f2 = slice_buf[2];
                
                float a = -(2.0f * h0 + h1) / (h0 * (h0 + h1));
                float b = (h0 + h1) / (h0 * h1);
                float c = -h0 / (h1 * (h0 + h1));
                
                res[offsetRes] = a * f0 + b * f1 + c * f2;
            }
            
            // Interior points
            for (int64_t i = 1; i < N - 1; i++) {
                int64_t idxRes = offsetRes + i * stridesRes[axis];
                
                float f_curr = slice_buf[i];
                float f_prev = slice_buf[i - 1];
                float f_next = slice_buf[i + 1];
                
                float h_s = dx;
                float h_d = dx;
                if (x != nullptr) {
                    h_s = x[i * strideX] - x[(i - 1) * strideX];
                    h_d = x[(i + 1) * strideX] - x[i * strideX];
                }
                
                res[idxRes] = (h_s * h_s * f_next + (h_d * h_d - h_s * h_s) * f_curr - h_d * h_d * f_prev) / (h_s * h_d * (h_s + h_d));
            }
            
            // Right boundary
            int64_t idxResEnd = offsetRes + (N - 1) * stridesRes[axis];
            if (edge_order == 1) {
                float h = dx;
                if (x != nullptr) {
                    h = x[(N - 1) * strideX] - x[(N - 2) * strideX];
                }
                res[idxResEnd] = (slice_buf[N - 1] - slice_buf[N - 2]) / h;
            } else {
                float h0 = dx;
                float h1 = dx;
                if (x != nullptr) {
                    h0 = x[(N - 2) * strideX] - x[(N - 3) * strideX];
                    h1 = x[(N - 1) * strideX] - x[(N - 2) * strideX];
                }
                float f0 = slice_buf[N - 3];
                float f1 = slice_buf[N - 2];
                float f2 = slice_buf[N - 1];
                
                float a = h1 / (h0 * (h0 + h1));
                float b = -(h0 + h1) / (h0 * h1);
                float c = (2.0f * h1 + h0) / (h1 * (h0 + h1));
                
                res[idxResEnd] = a * f0 + b * f1 + c * f2;
            }
        }
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_gradient_complex128(const cpx_t *src, const int64_t *stridesSrc,
                           const double *x, int64_t strideX, double dx,
                           cpx_t *res, const int64_t *stridesRes,
                           const int64_t *shape, int rank, int axis, int edge_order) {
    if (src == nullptr || res == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    
    int64_t N = shape[axis];
    if (N <= 0) return;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (shape[d] <= 0) return;
        if (d != axis) outer_size *= shape[d];
    }
    
    NoThrowBuffer<cpx_t> slice_buf(N);
    if (!slice_buf) {
        ndarray_set_oom_flag();
        return;
    }

    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetSrc = 0;
        int64_t offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetSrc += coord[d] * stridesSrc[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        for (int64_t i = 0; i < N; i++) {
            slice_buf[i] = src[offsetSrc + i * stridesSrc[axis]];
        }
        
        if (N == 1) {
            res[offsetRes] = (cpx_t){0.0, 0.0};
        } else if (N == 2) {
            double h = dx;
            if (x != nullptr) {
                h = x[strideX] - x[0];
            }
            cpx_t f0 = slice_buf[0];
            cpx_t f1 = slice_buf[1];
            cpx_t diff = (cpx_t){(f1.r - f0.r) / h, (f1.i - f0.i) / h};
            res[offsetRes] = diff;
            res[offsetRes + stridesRes[axis]] = diff;
        } else {
            // Left boundary (i = 0)
            if (edge_order == 1) {
                double h = dx;
                if (x != nullptr) {
                    h = x[strideX] - x[0];
                }
                cpx_t f0 = slice_buf[0];
                cpx_t f1 = slice_buf[1];
                res[offsetRes] = (cpx_t){(f1.r - f0.r) / h, (f1.i - f0.i) / h};
            } else {
                double h0 = dx;
                double h1 = dx;
                if (x != nullptr) {
                    h0 = x[strideX] - x[0];
                    h1 = x[2 * strideX] - x[strideX];
                }
                cpx_t f0 = slice_buf[0];
                cpx_t f1 = slice_buf[1];
                cpx_t f2 = slice_buf[2];
                
                double a = -(2.0 * h0 + h1) / (h0 * (h0 + h1));
                double b = (h0 + h1) / (h0 * h1);
                double c = -h0 / (h1 * (h0 + h1));
                
                res[offsetRes] = (cpx_t){
                    a * f0.r + b * f1.r + c * f2.r,
                    a * f0.i + b * f1.i + c * f2.i
                };
            }
            
            // Interior points
            for (int64_t i = 1; i < N - 1; i++) {
                int64_t idxRes = offsetRes + i * stridesRes[axis];
                
                cpx_t f_curr = slice_buf[i];
                cpx_t f_prev = slice_buf[i - 1];
                cpx_t f_next = slice_buf[i + 1];
                
                double h_s = dx;
                double h_d = dx;
                if (x != nullptr) {
                    h_s = x[i * strideX] - x[(i - 1) * strideX];
                    h_d = x[(i + 1) * strideX] - x[i * strideX];
                }
                
                double denom = h_s * h_d * (h_s + h_d);
                res[idxRes] = (cpx_t){
                    (h_s * h_s * f_next.r + (h_d * h_d - h_s * h_s) * f_curr.r - h_d * h_d * f_prev.r) / denom,
                    (h_s * h_s * f_next.i + (h_d * h_d - h_s * h_s) * f_curr.i - h_d * h_d * f_prev.i) / denom
                };
            }
            
            // Right boundary
            int64_t idxResEnd = offsetRes + (N - 1) * stridesRes[axis];
            if (edge_order == 1) {
                double h = dx;
                if (x != nullptr) {
                    h = x[(N - 1) * strideX] - x[(N - 2) * strideX];
                }
                cpx_t f0 = slice_buf[N - 2];
                cpx_t f1 = slice_buf[N - 1];
                res[idxResEnd] = (cpx_t){(f1.r - f0.r) / h, (f1.i - f0.i) / h};
            } else {
                double h0 = dx;
                double h1 = dx;
                if (x != nullptr) {
                    h0 = x[(N - 2) * strideX] - x[(N - 3) * strideX];
                    h1 = x[(N - 1) * strideX] - x[(N - 2) * strideX];
                }
                cpx_t f0 = slice_buf[N - 3];
                cpx_t f1 = slice_buf[N - 2];
                cpx_t f2 = slice_buf[N - 1];
                
                double a = h1 / (h0 * (h0 + h1));
                double b = -(h0 + h1) / (h0 * h1);
                double c = (2.0 * h1 + h0) / (h1 * (h0 + h1));
                
                res[idxResEnd] = (cpx_t){
                    a * f0.r + b * f1.r + c * f2.r,
                    a * f0.i + b * f1.i + c * f2.i
                };
            }
        }
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_gradient_complex64(const cpx_f_t *src, const int64_t *stridesSrc,
                          const float *x, int64_t strideX, float dx,
                          cpx_f_t *res, const int64_t *stridesRes,
                          const int64_t *shape, int rank, int axis, int edge_order) {
    if (src == nullptr || res == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    
    int64_t N = shape[axis];
    if (N <= 0) return;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (shape[d] <= 0) return;
        if (d != axis) outer_size *= shape[d];
    }
    
    NoThrowBuffer<cpx_f_t> slice_buf(N);
    if (!slice_buf) {
        ndarray_set_oom_flag();
        return;
    }

    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetSrc = 0;
        int64_t offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetSrc += coord[d] * stridesSrc[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        for (int64_t i = 0; i < N; i++) {
            slice_buf[i] = src[offsetSrc + i * stridesSrc[axis]];
        }
        
        if (N == 1) {
            res[offsetRes] = (cpx_f_t){0.0f, 0.0f};
        } else if (N == 2) {
            float h = dx;
            if (x != nullptr) {
                h = x[strideX] - x[0];
            }
            cpx_f_t f0 = slice_buf[0];
            cpx_f_t f1 = slice_buf[1];
            cpx_f_t diff = (cpx_f_t){(f1.r - f0.r) / h, (f1.i - f0.i) / h};
            res[offsetRes] = diff;
            res[offsetRes + stridesRes[axis]] = diff;
        } else {
            // Left boundary (i = 0)
            if (edge_order == 1) {
                float h = dx;
                if (x != nullptr) {
                    h = x[strideX] - x[0];
                }
                cpx_f_t f0 = slice_buf[0];
                cpx_f_t f1 = slice_buf[1];
                res[offsetRes] = (cpx_f_t){(f1.r - f0.r) / h, (f1.i - f0.i) / h};
            } else {
                float h0 = dx;
                float h1 = dx;
                if (x != nullptr) {
                    h0 = x[strideX] - x[0];
                    h1 = x[2 * strideX] - x[strideX];
                }
                cpx_f_t f0 = slice_buf[0];
                cpx_f_t f1 = slice_buf[1];
                cpx_f_t f2 = slice_buf[2];
                
                float a = -(2.0f * h0 + h1) / (h0 * (h0 + h1));
                float b = (h0 + h1) / (h0 * h1);
                float c = -h0 / (h1 * (h0 + h1));
                
                res[offsetRes] = (cpx_f_t){
                    a * f0.r + b * f1.r + c * f2.r,
                    a * f0.i + b * f1.i + c * f2.i
                };
            }
            
            // Interior points
            for (int64_t i = 1; i < N - 1; i++) {
                int64_t idxRes = offsetRes + i * stridesRes[axis];
                
                cpx_f_t f_curr = slice_buf[i];
                cpx_f_t f_prev = slice_buf[i - 1];
                cpx_f_t f_next = slice_buf[i + 1];
                
                float h_s = dx;
                float h_d = dx;
                if (x != nullptr) {
                    h_s = x[i * strideX] - x[(i - 1) * strideX];
                    h_d = x[(i + 1) * strideX] - x[i * strideX];
                }
                
                float denom = h_s * h_d * (h_s + h_d);
                res[idxRes] = (cpx_f_t){
                    (h_s * h_s * f_next.r + (h_d * h_d - h_s * h_s) * f_curr.r - h_d * h_d * f_prev.r) / denom,
                    (h_s * h_s * f_next.i + (h_d * h_d - h_s * h_s) * f_curr.i - h_d * h_d * f_prev.i) / denom
                };
            }
            
            // Right boundary
            int64_t idxResEnd = offsetRes + (N - 1) * stridesRes[axis];
            if (edge_order == 1) {
                float h = dx;
                if (x != nullptr) {
                    h = x[(N - 1) * strideX] - x[(N - 2) * strideX];
                }
                cpx_f_t f0 = slice_buf[N - 2];
                cpx_f_t f1 = slice_buf[N - 1];
                res[idxResEnd] = (cpx_f_t){(f1.r - f0.r) / h, (f1.i - f0.i) / h};
            } else {
                float h0 = dx;
                float h1 = dx;
                if (x != nullptr) {
                    h0 = x[(N - 2) * strideX] - x[(N - 3) * strideX];
                    h1 = x[(N - 1) * strideX] - x[(N - 2) * strideX];
                }
                cpx_f_t f0 = slice_buf[N - 3];
                cpx_f_t f1 = slice_buf[N - 2];
                cpx_f_t f2 = slice_buf[N - 1];
                
                float a = h1 / (h0 * (h0 + h1));
                float b = -(h0 + h1) / (h0 * h1);
                float c = (2.0f * h1 + h0) / (h1 * (h0 + h1));
                
                res[idxResEnd] = (cpx_f_t){
                    a * f0.r + b * f1.r + c * f2.r,
                    a * f0.i + b * f1.i + c * f2.i
                };
            }
        }
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}



static cpx_t c_mul(cpx_t a, cpx_t b) {
    cpx_t res;
    res.r = a.r * b.r - a.i * b.i;
    res.i = a.r * b.i + a.i * b.r;
    return res;
}

static cpx_t c_div(cpx_t n, cpx_t d) {
    cpx_t res;
    double denom = d.r * d.r + d.i * d.i;
    res.r = (n.r * d.r + n.i * d.i) / denom;
    res.i = (n.i * d.r - n.r * d.i) / denom;
    return res;
}

void s_gradient_complex128_all(const cpx_t *src, const int64_t *stridesSrc,
                               const cpx_t *x, int64_t strideX, cpx_t dx,
                               cpx_t *res, const int64_t *stridesRes,
                               const int64_t *shape, int rank, int axis, int edge_order) {
    if (src == nullptr || res == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    
    int64_t N = shape[axis];
    if (N <= 0) return;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (shape[d] <= 0) return;
        if (d != axis) outer_size *= shape[d];
    }
    
    NoThrowBuffer<cpx_t> slice_buf(N);
    if (!slice_buf) {
        ndarray_set_oom_flag();
        return;
    }

    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetSrc = 0;
        int64_t offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetSrc += coord[d] * stridesSrc[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        for (int64_t i = 0; i < N; i++) {
            slice_buf[i] = src[offsetSrc + i * stridesSrc[axis]];
        }
        
        if (N == 1) {
            res[offsetRes].r = 0.0;
            res[offsetRes].i = 0.0;
        } else if (N == 2) {
            cpx_t h = dx;
            if (x != nullptr) {
                h.r = x[strideX].r - x[0].r;
                h.i = x[strideX].i - x[0].i;
            }
            cpx_t diff;
            diff.r = slice_buf[1].r - slice_buf[0].r;
            diff.i = slice_buf[1].i - slice_buf[0].i;
            res[offsetRes] = c_div(diff, h);
            res[offsetRes + stridesRes[axis]] = res[offsetRes];
        } else {
            // Left boundary (i = 0)
            if (edge_order == 1) {
                cpx_t h = dx;
                if (x != nullptr) {
                    h.r = x[strideX].r - x[0].r;
                    h.i = x[strideX].i - x[0].i;
                }
                cpx_t num = {slice_buf[1].r - slice_buf[0].r,
                             slice_buf[1].i - slice_buf[0].i};
                res[offsetRes] = c_div(num, h);
            } else {
                cpx_t h0 = dx;
                cpx_t h1 = dx;
                if (x != nullptr) {
                    h0.r = x[strideX].r - x[0].r;
                    h0.i = x[strideX].i - x[0].i;
                    h1.r = x[2 * strideX].r - x[strideX].r;
                    h1.i = x[2 * strideX].i - x[strideX].i;
                }
                cpx_t f0 = slice_buf[0];
                cpx_t f1 = slice_buf[1];
                cpx_t f2 = slice_buf[2];
                
                cpx_t h0h1 = {h0.r + h1.r, h0.i + h1.i};
                cpx_t denom_a = c_mul(h0, h0h1);
                cpx_t num_a = {-(2.0 * h0.r + h1.r), -(2.0 * h0.i + h1.i)};
                cpx_t a = c_div(num_a, denom_a);
                
                cpx_t denom_b = c_mul(h0, h1);
                cpx_t b = c_div(h0h1, denom_b);
                
                cpx_t denom_c = c_mul(h1, h0h1);
                cpx_t num_c = {-h0.r, -h0.i};
                cpx_t c = c_div(num_c, denom_c);
                
                cpx_t term1 = c_mul(a, f0);
                cpx_t term2 = c_mul(b, f1);
                cpx_t term3 = c_mul(c, f2);
                
                res[offsetRes].r = term1.r + term2.r + term3.r;
                res[offsetRes].i = term1.i + term2.i + term3.i;
            }
            
            // Interior points
            for (int64_t i = 1; i < N - 1; i++) {
                int64_t idxRes = offsetRes + i * stridesRes[axis];
                
                cpx_t f_curr = slice_buf[i];
                cpx_t f_prev = slice_buf[i - 1];
                cpx_t f_next = slice_buf[i + 1];
                
                cpx_t h_s = dx;
                cpx_t h_d = dx;
                if (x != nullptr) {
                    h_s.r = x[i * strideX].r - x[(i - 1) * strideX].r;
                    h_s.i = x[i * strideX].i - x[(i - 1) * strideX].i;
                    h_d.r = x[(i + 1) * strideX].r - x[i * strideX].r;
                    h_d.i = x[(i + 1) * strideX].i - x[i * strideX].i;
                }
                
                cpx_t hs2 = c_mul(h_s, h_s);
                cpx_t hd2 = c_mul(h_d, h_d);
                cpx_t hd2_hs2 = {hd2.r - hs2.r, hd2.i - hs2.i};
                
                cpx_t term1 = c_mul(hs2, f_next);
                cpx_t term2 = c_mul(hd2_hs2, f_curr);
                cpx_t term3 = c_mul(hd2, f_prev);
                
                cpx_t num = {term1.r + term2.r - term3.r, term1.i + term2.i - term3.i};
                
                cpx_t hshd = c_mul(h_s, h_d);
                cpx_t hshdsum = {h_s.r + h_d.r, h_s.i + h_d.i};
                cpx_t denom = c_mul(hshd, hshdsum);
                
                res[idxRes] = c_div(num, denom);
            }
            
            // Right boundary
            int64_t idxResEnd = offsetRes + (N - 1) * stridesRes[axis];
            if (edge_order == 1) {
                cpx_t h = dx;
                if (x != nullptr) {
                    h.r = x[(N - 1) * strideX].r - x[(N - 2) * strideX].r;
                    h.i = x[(N - 1) * strideX].i - x[(N - 2) * strideX].i;
                }
                cpx_t num = {slice_buf[N - 1].r - slice_buf[N - 2].r,
                             slice_buf[N - 1].i - slice_buf[N - 2].i};
                res[idxResEnd] = c_div(num, h);
            } else {
                cpx_t h0 = dx;
                cpx_t h1 = dx;
                if (x != nullptr) {
                    h0.r = x[(N - 2) * strideX].r - x[(N - 3) * strideX].r;
                    h0.i = x[(N - 2) * strideX].i - x[(N - 3) * strideX].i;
                    h1.r = x[(N - 1) * strideX].r - x[(N - 2) * strideX].r;
                    h1.i = x[(N - 1) * strideX].i - x[(N - 2) * strideX].i;
                }
                cpx_t f0 = slice_buf[N - 3];
                cpx_t f1 = slice_buf[N - 2];
                cpx_t f2 = slice_buf[N - 1];
                
                cpx_t h0h1 = {h0.r + h1.r, h0.i + h1.i};
                cpx_t denom_a = c_mul(h0, h0h1);
                cpx_t a = c_div(h1, denom_a);
                cpx_t denom_b = c_mul(h0, h1);
                cpx_t num_b = {-h0h1.r, -h0h1.i};
                cpx_t b = c_div(num_b, denom_b);
                cpx_t denom_c = c_mul(h1, h0h1);
                cpx_t num_c = {2.0 * h1.r + h0.r, 2.0 * h1.i + h0.i};
                cpx_t c = c_div(num_c, denom_c);
                
                cpx_t term1 = c_mul(a, f0);
                cpx_t term2 = c_mul(b, f1);
                cpx_t term3 = c_mul(c, f2);
                
                res[idxResEnd].r = term1.r + term2.r + term3.r;
                res[idxResEnd].i = term1.i + term2.i + term3.i;
            }
        }
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_trapz_complex64_all(const cpx_f_t *y, const int64_t *stridesY,
                           const cpx_f_t *x, int64_t strideX, cpx_f_t dx,
                           cpx_f_t *res, const int64_t *stridesRes,
                           const int64_t *shape, int rank, int axis) {
    if (y == nullptr || res == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetRes = 0;
        int64_t offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesY[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesRes[targetD];
                }
            }
        }
        
        cpx_f_t sum = {0.0f, 0.0f};
        for (int64_t i = 0; i < shape[axis] - 1; i++) {
            int64_t idxSrc = offsetSrc + i * stridesY[axis];
            int64_t idxSrcNext = offsetSrc + (i + 1) * stridesY[axis];
            cpx_f_t y_curr = y[idxSrc];
            cpx_f_t y_next = y[idxSrcNext];
            
            cpx_f_t h = dx;
            if (x != nullptr) {
                h.r = x[(i + 1) * strideX].r - x[i * strideX].r;
                h.i = x[(i + 1) * strideX].i - x[i * strideX].i;
            }
            
            float yr = (y_curr.r + y_next.r) * 0.5f;
            float yi = (y_curr.i + y_next.i) * 0.5f;
            
            sum.r += yr * h.r - yi * h.i;
            sum.i += yr * h.i + yi * h.r;
        }
        
        res[offsetRes] = sum;
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

static cpx_f_t cf_mul(cpx_f_t a, cpx_f_t b) {
    cpx_f_t res;
    res.r = a.r * b.r - a.i * b.i;
    res.i = a.r * b.i + a.i * b.r;
    return res;
}

static cpx_f_t cf_div(cpx_f_t n, cpx_f_t d) {
    cpx_f_t res;
    float denom = d.r * d.r + d.i * d.i;
    res.r = (n.r * d.r + n.i * d.i) / denom;
    res.i = (n.i * d.r - n.r * d.i) / denom;
    return res;
}

void s_gradient_complex64_all(const cpx_f_t *src, const int64_t *stridesSrc,
                              const cpx_f_t *x, int64_t strideX, cpx_f_t dx,
                              cpx_f_t *res, const int64_t *stridesRes,
                              const int64_t *shape, int rank, int axis, int edge_order) {
    if (src == nullptr || res == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    
    int64_t N = shape[axis];
    if (N <= 0) return;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (shape[d] <= 0) return;
        if (d != axis) outer_size *= shape[d];
    }
    
    NoThrowBuffer<cpx_f_t> slice_buf(N);
    if (!slice_buf) {
        ndarray_set_oom_flag();
        return;
    }

    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetSrc = 0;
        int64_t offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetSrc += coord[d] * stridesSrc[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        for (int64_t i = 0; i < N; i++) {
            slice_buf[i] = src[offsetSrc + i * stridesSrc[axis]];
        }
        
        if (N == 1) {
            res[offsetRes].r = 0.0f;
            res[offsetRes].i = 0.0f;
        } else if (N == 2) {
            cpx_f_t h = dx;
            if (x != nullptr) {
                h.r = x[strideX].r - x[0].r;
                h.i = x[strideX].i - x[0].i;
            }
            cpx_f_t diff;
            diff.r = slice_buf[1].r - slice_buf[0].r;
            diff.i = slice_buf[1].i - slice_buf[0].i;
            res[offsetRes] = cf_div(diff, h);
            res[offsetRes + stridesRes[axis]] = res[offsetRes];
        } else {
            if (edge_order == 1) {
                cpx_f_t h = dx;
                if (x != nullptr) {
                    h.r = x[strideX].r - x[0].r;
                    h.i = x[strideX].i - x[0].i;
                }
                cpx_f_t num = {slice_buf[1].r - slice_buf[0].r,
                             slice_buf[1].i - slice_buf[0].i};
                res[offsetRes] = cf_div(num, h);
            } else {
                cpx_f_t h0 = dx;
                cpx_f_t h1 = dx;
                if (x != nullptr) {
                    h0.r = x[strideX].r - x[0].r;
                    h0.i = x[strideX].i - x[0].i;
                    h1.r = x[2 * strideX].r - x[strideX].r;
                    h1.i = x[2 * strideX].i - x[strideX].i;
                }
                cpx_f_t f0 = slice_buf[0];
                cpx_f_t f1 = slice_buf[1];
                cpx_f_t f2 = slice_buf[2];
                
                cpx_f_t h0h1 = {h0.r + h1.r, h0.i + h1.i};
                cpx_f_t denom_a = cf_mul(h0, h0h1);
                cpx_f_t num_a = {-(2.0f * h0.r + h1.r), -(2.0f * h0.i + h1.i)};
                cpx_f_t a = cf_div(num_a, denom_a);
                cpx_f_t denom_b = cf_mul(h0, h1);
                cpx_f_t b = cf_div(h0h1, denom_b);
                cpx_f_t denom_c = cf_mul(h1, h0h1);
                cpx_f_t num_c = {-h0.r, -h0.i};
                cpx_f_t c = cf_div(num_c, denom_c);
                
                cpx_f_t term1 = cf_mul(a, f0);
                cpx_f_t term2 = cf_mul(b, f1);
                cpx_f_t term3 = cf_mul(c, f2);
                res[offsetRes].r = term1.r + term2.r + term3.r;
                res[offsetRes].i = term1.i + term2.i + term3.i;
            }
            
            for (int64_t i = 1; i < N - 1; i++) {
                int64_t idxRes = offsetRes + i * stridesRes[axis];
                
                cpx_f_t f_curr = slice_buf[i];
                cpx_f_t f_prev = slice_buf[i - 1];
                cpx_f_t f_next = slice_buf[i + 1];
                
                cpx_f_t h_s = dx;
                cpx_f_t h_d = dx;
                if (x != nullptr) {
                    h_s.r = x[i * strideX].r - x[(i - 1) * strideX].r;
                    h_s.i = x[i * strideX].i - x[(i - 1) * strideX].i;
                    h_d.r = x[(i + 1) * strideX].r - x[i * strideX].r;
                    h_d.i = x[(i + 1) * strideX].i - x[i * strideX].i;
                }
                
                cpx_f_t hs2 = cf_mul(h_s, h_s);
                cpx_f_t hd2 = cf_mul(h_d, h_d);
                cpx_f_t hd2_hs2 = {hd2.r - hs2.r, hd2.i - hs2.i};
                cpx_f_t term1 = cf_mul(hs2, f_next);
                cpx_f_t term2 = cf_mul(hd2_hs2, f_curr);
                cpx_f_t term3 = cf_mul(hd2, f_prev);
                cpx_f_t num = {term1.r + term2.r - term3.r, term1.i + term2.i - term3.i};
                cpx_f_t hshd = cf_mul(h_s, h_d);
                cpx_f_t hshdsum = {h_s.r + h_d.r, h_s.i + h_d.i};
                cpx_f_t denom = cf_mul(hshd, hshdsum);
                res[idxRes] = cf_div(num, denom);
            }
            
            int64_t idxResEnd = offsetRes + (N - 1) * stridesRes[axis];
            if (edge_order == 1) {
                cpx_f_t h = dx;
                if (x != nullptr) {
                    h.r = x[(N - 1) * strideX].r - x[(N - 2) * strideX].r;
                    h.i = x[(N - 1) * strideX].i - x[(N - 2) * strideX].i;
                }
                cpx_f_t num = {slice_buf[N - 1].r - slice_buf[N - 2].r,
                             slice_buf[N - 1].i - slice_buf[N - 2].i};
                res[idxResEnd] = cf_div(num, h);
            } else {
                cpx_f_t h0 = dx;
                cpx_f_t h1 = dx;
                if (x != nullptr) {
                    h0.r = x[(N - 2) * strideX].r - x[(N - 3) * strideX].r;
                    h0.i = x[(N - 2) * strideX].i - x[(N - 3) * strideX].i;
                    h1.r = x[(N - 1) * strideX].r - x[(N - 2) * strideX].r;
                    h1.i = x[(N - 1) * strideX].i - x[(N - 2) * strideX].i;
                }
                cpx_f_t f0 = slice_buf[N - 3];
                cpx_f_t f1 = slice_buf[N - 2];
                cpx_f_t f2 = slice_buf[N - 1];
                cpx_f_t h0h1 = {h0.r + h1.r, h0.i + h1.i};
                cpx_f_t denom_a = cf_mul(h0, h0h1);
                cpx_f_t a = cf_div(h1, denom_a);
                cpx_f_t denom_b = cf_mul(h0, h1);
                cpx_f_t num_b = {-h0h1.r, -h0h1.i};
                cpx_f_t b = cf_div(num_b, denom_b);
                cpx_f_t denom_c = cf_mul(h1, h0h1);
                cpx_f_t num_c = {2.0f * h1.r + h0.r, 2.0f * h1.i + h0.i};
                cpx_f_t c = cf_div(num_c, denom_c);
                cpx_f_t term1 = cf_mul(a, f0);
                cpx_f_t term2 = cf_mul(b, f1);
                cpx_f_t term3 = cf_mul(c, f2);
                res[idxResEnd].r = term1.r + term2.r + term3.r;
                res[idxResEnd].i = term1.i + term2.i + term3.i;
            }
        }
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

/* ============================================================================
 * SECTION 10: linspace GRID INTRINSIC KERNELS
 * ============================================================================
 */

#define DEFINE_LINSPACE_GRID(name, type, val_expr, step_expr) \
void name(const type *start, const int64_t *stridesStart, \
          const type *stop, const int64_t *stridesStop, \
          type *res, const int64_t *stridesRes, \
          type *step, const int64_t *stridesStep, \
          const int64_t *shape, int rank, int axis, int64_t numSamples, int endpoint) { \
    if (start == nullptr || stop == nullptr || res == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return; \
    int64_t total_elements = 1; \
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i]; \
    double div = endpoint ? (double)(numSamples - 1) : (double)numSamples; \
    DECLARE_RANK_BUFFER(int64_t, coord, rank); \
    int64_t offsetStart = 0, offsetStop = 0, offsetRes = 0, offsetStep = 0; \
    for (int64_t el = 0; el < total_elements; el++) { \
        int64_t i = coord[axis]; \
        double t = (div <= 0.0) ? 0.0 : (double)i / div; \
        if (endpoint && numSamples > 1 && i == numSamples - 1) { \
            res[offsetRes] = stop[offsetStop]; \
        } else { \
            val_expr; \
        } \
        if (step != nullptr && coord[axis] == 0) { \
            step_expr; \
        } \
        for (int d = rank - 1; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetStart += stridesStart[d]; \
                offsetStop += stridesStop[d]; \
                offsetRes += stridesRes[d]; \
                if (step != nullptr) offsetStep += stridesStep[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetStart -= (shape[d] - 1) * stridesStart[d]; \
            offsetStop -= (shape[d] - 1) * stridesStop[d]; \
            offsetRes -= (shape[d] - 1) * stridesRes[d]; \
            if (step != nullptr) offsetStep -= (shape[d] - 1) * stridesStep[d]; \
        } \
    } \
}

DEFINE_LINSPACE_GRID(
    s_linspace_grid_double, double,
    { res[offsetRes] = start[offsetStart] + (stop[offsetStop] - start[offsetStart]) * t; },
    { step[offsetStep] = (div <= 0.0) ? 0.0 : (stop[offsetStop] - start[offsetStart]) / div; }
)

DEFINE_LINSPACE_GRID(
    s_linspace_grid_float, float,
    { res[offsetRes] = start[offsetStart] + (stop[offsetStop] - start[offsetStart]) * (float)t; },
    { step[offsetStep] = (div <= 0.0) ? 0.0f : (stop[offsetStop] - start[offsetStart]) / (float)div; }
)

DEFINE_LINSPACE_GRID(
    s_linspace_grid_complex128, cpx_t,
    {
        res[offsetRes].r = start[offsetStart].r + (stop[offsetStop].r - start[offsetStart].r) * t;
        res[offsetRes].i = start[offsetStart].i + (stop[offsetStop].i - start[offsetStart].i) * t;
    },
    {
        step[offsetStep].r = (div <= 0.0) ? 0.0 : (stop[offsetStop].r - start[offsetStart].r) / div;
        step[offsetStep].i = (div <= 0.0) ? 0.0 : (stop[offsetStop].i - start[offsetStart].i) / div;
    }
)

DEFINE_LINSPACE_GRID(
    s_linspace_grid_complex64, cpx_f_t,
    {
        res[offsetRes].r = start[offsetStart].r + (stop[offsetStop].r - start[offsetStart].r) * (float)t;
        res[offsetRes].i = start[offsetStart].i + (stop[offsetStop].i - start[offsetStart].i) * (float)t;
    },
    {
        step[offsetStep].r = (div <= 0.0) ? 0.0f : (stop[offsetStop].r - start[offsetStart].r) / (float)div;
        step[offsetStep].i = (div <= 0.0) ? 0.0f : (stop[offsetStop].i - start[offsetStart].i) / (float)div;
    }
)

DEFINE_LINSPACE_GRID(
    s_linspace_grid_int64, int64_t,
    { res[offsetRes] = saturating_float_to_int<int64_t>(std::floor((double)start[offsetStart] + ((double)stop[offsetStop] - (double)start[offsetStart]) * t)); },
    { step[offsetStep] = (div <= 0.0) ? 0 : saturating_float_to_int<int64_t>(std::floor(((double)stop[offsetStop] - (double)start[offsetStart]) / div)); }
)

DEFINE_LINSPACE_GRID(
    s_linspace_grid_int32, int32_t,
    { res[offsetRes] = saturating_float_to_int<int32_t>(std::floor((double)start[offsetStart] + ((double)stop[offsetStop] - (double)start[offsetStart]) * t)); },
    { step[offsetStep] = (div <= 0.0) ? 0 : saturating_float_to_int<int32_t>(std::floor(((double)stop[offsetStop] - (double)start[offsetStart]) / div)); }
)

DEFINE_LINSPACE_GRID(
    s_linspace_grid_int16, int16_t,
    { res[offsetRes] = saturating_float_to_int<int16_t>(std::floor((double)start[offsetStart] + ((double)stop[offsetStop] - (double)start[offsetStart]) * t)); },
    { step[offsetStep] = (div <= 0.0) ? 0 : saturating_float_to_int<int16_t>(std::floor(((double)stop[offsetStop] - (double)start[offsetStart]) / div)); }
)

DEFINE_LINSPACE_GRID(
    s_linspace_grid_uint8, uint8_t,
    { res[offsetRes] = saturating_float_to_int<uint8_t>(std::floor((double)start[offsetStart] + ((double)stop[offsetStop] - (double)start[offsetStart]) * t)); },
    { step[offsetStep] = (div <= 0.0) ? 0 : saturating_float_to_int<uint8_t>(std::floor(((double)stop[offsetStop] - (double)start[offsetStart]) / div)); }
)

/* ============================================================================
 * SECTION 12: DYNAMICALLY DISPATCHED MATRIX DETERMINANT INTRA-OPERATIONS
 * ============================================================================
 * Strided determinants offloading LU-factorization call dynamically to a
 * LAPACK function pointer passed from Dart FFI space. This makes the routine
 * 100% decoupled from compile-time OpenBLAS/LAPACK dependencies.
 */

#define DEFINE_DET_REAL(name, type) \
void name(const type *a, const int64_t *stridesA, \
          type *res, const int64_t *stridesRes, \
          const int64_t *shape, int rank, \
          type *aCopy, int *ipiv, \
          int (*lapack_getrf)(int, int, int, void *, int, int *)) { \
    if (a == nullptr || res == nullptr || shape == nullptr || lapack_getrf == nullptr || rank < 2) return; \
    int n = shape[rank - 1]; \
    if (n > 0 && (aCopy == nullptr || ipiv == nullptr)) return; \
    int64_t stack_elements = 1; \
    for (int64_t i = 0; i < rank - 2; i++) stack_elements *= shape[i]; \
    DECLARE_RANK_BUFFER(int64_t, coord, rank); \
    int64_t offsetA = 0, offsetRes = 0; \
    for (int64_t el = 0; el < stack_elements; el++) { \
        type detValue = 1.0; \
        if (n > 0) { \
            for (int64_t i = 0; i < n; i++) { \
                for (int64_t j = 0; j < n; j++) { \
                    aCopy[i * n + j] = a[offsetA + i * stridesA[rank - 2] + j * stridesA[rank - 1]]; \
                } \
            } \
            int info = lapack_getrf(101, n, n, aCopy, n, ipiv); \
            if (info < 0) { \
                detValue = NAN; \
            } else { \
                for (int64_t i = 0; i < n; i++) { \
                    detValue *= aCopy[i * n + i]; \
                } \
                if (detValue == 0.0) { \
                    detValue = 0.0; \
                } else if (!std::isnan(detValue)) { \
                    int swaps = 0; \
                    for (int64_t i = 0; i < n; i++) { \
                        if (ipiv[i] != i + 1) swaps++; \
                    } \
                    if (swaps % 2 != 0) detValue = -detValue; \
                } \
            } \
        } \
        res[offsetRes] = detValue; \
        for (int d = rank - 3; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetA += stridesA[d]; \
                offsetRes += stridesRes[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetA -= (shape[d] - 1) * stridesA[d]; \
            offsetRes -= (shape[d] - 1) * stridesRes[d]; \
        } \
    } \
}

#define DEFINE_DET_COMPLEX(name, type) \
void name(const type *a, const int64_t *stridesA, \
          type *res, const int64_t *stridesRes, \
          const int64_t *shape, int rank, \
          type *aCopy, int *ipiv, \
          int (*lapack_getrf)(int, int, int, void *, int, int *)) { \
    if (a == nullptr || res == nullptr || shape == nullptr || lapack_getrf == nullptr || rank < 2) return; \
    int n = shape[rank - 1]; \
    if (n > 0 && (aCopy == nullptr || ipiv == nullptr)) return; \
    int64_t stack_elements = 1; \
    for (int64_t i = 0; i < rank - 2; i++) stack_elements *= shape[i]; \
    DECLARE_RANK_BUFFER(int64_t, coord, rank); \
    int64_t offsetA = 0, offsetRes = 0; \
    for (int64_t el = 0; el < stack_elements; el++) { \
        type detValue = {1.0, 0.0}; \
        if (n > 0) { \
            for (int64_t i = 0; i < n; i++) { \
                for (int64_t j = 0; j < n; j++) { \
                    aCopy[i * n + j] = a[offsetA + i * stridesA[rank - 2] + j * stridesA[rank - 1]]; \
                } \
            } \
            int info = lapack_getrf(101, n, n, aCopy, n, ipiv); \
            if (info < 0) { \
                detValue.r = NAN; \
                detValue.i = NAN; \
            } else { \
                bool hasNonFinite = false; \
                for (int64_t i = 0; i < n; i++) { \
                    double r2 = aCopy[i * n + i].r, i2 = aCopy[i * n + i].i; \
                    if (!std::isfinite(r2) || !std::isfinite(i2)) { \
                        hasNonFinite = true; \
                    } \
                    double r1 = detValue.r, i1 = detValue.i; \
                    detValue.r = r1 * r2 - i1 * i2; \
                    detValue.i = r1 * i2 + i1 * r2; \
                } \
                if (info > 0 && !hasNonFinite) { \
                    detValue.r = 0.0; \
                    detValue.i = 0.0; \
                } else { \
                    int swaps = 0; \
                    for (int64_t i = 0; i < n; i++) { \
                        if (ipiv[i] != i + 1) swaps++; \
                    } \
                    if (swaps % 2 != 0) { \
                        detValue.r = -detValue.r; \
                        detValue.i = -detValue.i; \
                    } \
                } \
            } \
        } \
        res[offsetRes] = detValue; \
        for (int d = rank - 3; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetA += stridesA[d]; \
                offsetRes += stridesRes[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetA -= (shape[d] - 1) * stridesA[d]; \
            offsetRes -= (shape[d] - 1) * stridesRes[d]; \
        } \
    } \
}

DEFINE_DET_REAL(s_det_double, double)
DEFINE_DET_REAL(s_det_float, float)
DEFINE_DET_COMPLEX(s_det_complex_double, cpx_t)
DEFINE_DET_COMPLEX(s_det_complex_float, cpx_f_t)

/**
 * Assembles complex eigenvectors for double precision.
 * Reconstructs complex eigenvectors from the real Schur-like form returned by LAPACK dgeev.
 */
void assemble_eigenvectors_double(
    cpx_t *w,
    int64_t strideWLast,
    cpx_t *vr,
    int64_t strideVR1,
    int64_t strideVR2,
    const double *wr,
    const double *wi,
    const double *vrReal,
    int64_t n
) {
    if (w == nullptr || vr == nullptr || wr == nullptr || wi == nullptr || vrReal == nullptr || n <= 0) {
        return;
    }

    for (int64_t j = 0; j < n; j++) {
        w[j * strideWLast].r = wr[j];
        w[j * strideWLast].i = wi[j];
    }

    int64_t j = 0;
    while (j < n) {
        if (j + 1 < n && wi[j] > 0.0) {
            for (int64_t r = 0; r < n; r++) {
                double realPart = vrReal[r * n + j];
                double imagPart = vrReal[r * n + j + 1];
                vr[r * strideVR1 + j * strideVR2].r = realPart;
                vr[r * strideVR1 + j * strideVR2].i = imagPart;
                vr[r * strideVR1 + (j + 1) * strideVR2].r = realPart;
                vr[r * strideVR1 + (j + 1) * strideVR2].i = -imagPart;
            }
            j += 2;
        } else {
            double imagPart = std::isnan(wi[j]) ? wi[j] : 0.0;
            for (int64_t r = 0; r < n; r++) {
                vr[r * strideVR1 + j * strideVR2].r = vrReal[r * n + j];
                vr[r * strideVR1 + j * strideVR2].i = imagPart;
            }
            j++;
        }
    }
}

/**
 * Assembles complex eigenvectors for single precision.
 * Reconstructs complex eigenvectors from the real Schur-like form returned by LAPACK sgeev.
 */
void assemble_eigenvectors_float(
    cpx_f_t *w,
    int64_t strideWLast,
    cpx_f_t *vr,
    int64_t strideVR1,
    int64_t strideVR2,
    const float *wr,
    const float *wi,
    const float *vrReal,
    int64_t n
) {
    if (w == nullptr || vr == nullptr || wr == nullptr || wi == nullptr || vrReal == nullptr || n <= 0) {
        return;
    }

    for (int64_t j = 0; j < n; j++) {
        w[j * strideWLast].r = wr[j];
        w[j * strideWLast].i = wi[j];
    }

    int64_t j = 0;
    while (j < n) {
        if (j + 1 < n && wi[j] > 0.0f) {
            for (int64_t r = 0; r < n; r++) {
                float realPart = vrReal[r * n + j];
                float imagPart = vrReal[r * n + j + 1];
                vr[r * strideVR1 + j * strideVR2].r = realPart;
                vr[r * strideVR1 + j * strideVR2].i = imagPart;
                vr[r * strideVR1 + (j + 1) * strideVR2].r = realPart;
                vr[r * strideVR1 + (j + 1) * strideVR2].i = -imagPart;
            }
            j += 2;
        } else {
            float imagPart = std::isnan(wi[j]) ? wi[j] : 0.0f;
            for (int64_t r = 0; r < n; r++) {
                vr[r * strideVR1 + j * strideVR2].r = vrReal[r * n + j];
                vr[r * strideVR1 + j * strideVR2].i = imagPart;
            }
            j++;
        }
    }
}

/**
 * Macro to define optimized matrix multiplication for integer types.
 * Performs a standard 3-loop matmul supporting arbitrary strides for input and output matrices.
 */
#define DEFINE_MATMUL_INT(name, T) \
void name( \
    T *res, \
    int64_t strideResRow, \
    int64_t strideResCol, \
    const T *a, \
    int64_t strideARow, \
    int64_t strideACol, \
    const T *b, \
    int64_t strideBRow, \
    int64_t strideBCol, \
    int64_t m, \
    int64_t n, \
    int64_t k \
) { \
    if (res == nullptr || a == nullptr || b == nullptr || m <= 0 || n <= 0 || k <= 0) { \
        return; \
    } \
    using utype = typename std::conditional<sizeof(T) < 4, uint32_t, typename std::make_unsigned<T>::type>::type; \
    if (strideACol == 1 && strideBCol == 1 && strideResCol == 1) { \
        for (int64_t r = 0; r < m; r++) { \
            T *resRow = res + r * strideResRow; \
            std::memset(resRow, 0, static_cast<size_t>(n) * sizeof(T)); \
            const T *aRow = a + r * strideARow; \
            for (int64_t i = 0; i < k; i++) { \
                utype aVal = static_cast<utype>(aRow[i]); \
                const T *bRow = b + i * strideBRow; \
                for (int64_t c = 0; c < n; c++) { \
                    resRow[c] = static_cast<T>(static_cast<utype>(resRow[c]) + aVal * static_cast<utype>(bRow[c])); \
                } \
            } \
        } \
        return; \
    } \
    for (int64_t r = 0; r < m; r++) { \
        for (int64_t c = 0; c < n; c++) { \
            utype sum = 0; \
            for (int64_t i = 0; i < k; i++) { \
                sum += static_cast<utype>(a[r * strideARow + i * strideACol]) * static_cast<utype>(b[i * strideBRow + c * strideBCol]); \
            } \
            res[r * strideResRow + c * strideResCol] = static_cast<T>(sum); \
        } \
    } \
}

DEFINE_MATMUL_INT(matmul_int64, int64_t)
DEFINE_MATMUL_INT(matmul_int32, int32_t)
DEFINE_MATMUL_INT(matmul_int16, int16_t)
DEFINE_MATMUL_INT(matmul_uint8, uint8_t)

uint8_t v_any_less_than_zero_int32(const int32_t *arr, int64_t size) {
    if (arr == nullptr || size <= 0) return 0;
    for (int64_t i = 0; i < size; i++) {
        if (arr[i] < 0) return 1;
    }
    return 0;
}

uint8_t v_any_less_than_zero_int64(const int64_t *arr, int64_t size) {
    if (arr == nullptr || size <= 0) return 0;
    for (int64_t i = 0; i < size; i++) {
        if (arr[i] < 0) return 1;
    }
    return 0;
}

uint8_t v_any_equal_to_zero_int32(const int32_t *arr, int64_t size) {
    if (arr == nullptr || size <= 0) return 0;
    for (int64_t i = 0; i < size; i++) {
        if (arr[i] == 0) return 1;
    }
    return 0;
}

uint8_t v_any_equal_to_zero_int64(const int64_t *arr, int64_t size) {
    if (arr == nullptr || size <= 0) return 0;
    for (int64_t i = 0; i < size; i++) {
        if (arr[i] == 0) return 1;
    }
    return 0;
}



#define TO_DOUBLE(x) ((double)(x))
#define TO_COMPLEX(x) ((cpx_t){(double)(x), 0.0})
#define COMPLEX_TO_DOUBLE(x) ((x).r)
#define COMPLEXF_TO_DOUBLE(x) ((double)(x).r)
#define COMPLEXF_TO_COMPLEX(x) ((cpx_t){(double)(x).r, (double)(x).i})
#define BOOL_TO_DOUBLE(x) ((double)(x))
#define BOOL_TO_COMPLEX(x) ((cpx_t){(double)(x), 0.0})


template <typename DestType>
static void dispatch_cast_src(
    const void *src_ptr, const int64_t *stridesSrc, int dtypeSrc,
    void *dest_ptr,
    const int64_t *shape, int rank, int64_t total_elements
) {
    switch (dtypeSrc) {
        case DTYPE_FLOAT64: s_cast_generic_impl<double, DestType>(src_ptr, stridesSrc, dest_ptr, shape, rank, total_elements); break;
        case DTYPE_FLOAT32: s_cast_generic_impl<float, DestType>(src_ptr, stridesSrc, dest_ptr, shape, rank, total_elements); break;
        case DTYPE_FLOAT16: s_cast_generic_impl<float16_t, DestType>(src_ptr, stridesSrc, dest_ptr, shape, rank, total_elements); break;
        case DTYPE_BFLOAT16: s_cast_generic_impl<bfloat16_t, DestType>(src_ptr, stridesSrc, dest_ptr, shape, rank, total_elements); break;
        case DTYPE_INT32: s_cast_generic_impl<int32_t, DestType>(src_ptr, stridesSrc, dest_ptr, shape, rank, total_elements); break;
        case DTYPE_INT64: s_cast_generic_impl<int64_t, DestType>(src_ptr, stridesSrc, dest_ptr, shape, rank, total_elements); break;
        case DTYPE_UINT8: s_cast_generic_impl<uint8_t, DestType>(src_ptr, stridesSrc, dest_ptr, shape, rank, total_elements); break;
        case DTYPE_INT16: s_cast_generic_impl<int16_t, DestType>(src_ptr, stridesSrc, dest_ptr, shape, rank, total_elements); break;
        case DTYPE_COMPLEX128: s_cast_generic_impl<cpx_t, DestType>(src_ptr, stridesSrc, dest_ptr, shape, rank, total_elements); break;
        case DTYPE_COMPLEX64: s_cast_generic_impl<cpx_f_t, DestType>(src_ptr, stridesSrc, dest_ptr, shape, rank, total_elements); break;
        case DTYPE_BOOLEAN: s_cast_generic_impl<uint8_t, DestType>(src_ptr, stridesSrc, dest_ptr, shape, rank, total_elements); break;
        case DTYPE_INT8: s_cast_generic_impl<int8_t, DestType>(src_ptr, stridesSrc, dest_ptr, shape, rank, total_elements); break;
        case DTYPE_UINT64: s_cast_generic_impl<uint64_t, DestType>(src_ptr, stridesSrc, dest_ptr, shape, rank, total_elements); break;
        case DTYPE_UINT32: s_cast_generic_impl<uint32_t, DestType>(src_ptr, stridesSrc, dest_ptr, shape, rank, total_elements); break;
        case DTYPE_UINT16: s_cast_generic_impl<uint16_t, DestType>(src_ptr, stridesSrc, dest_ptr, shape, rank, total_elements); break;
        default: return;
    }
}

extern "C" void s_cast_generic(
    const void *src_ptr, const int64_t *stridesSrc, int dtypeSrc,
    void *dest_ptr, int dtypeDst,
    const int64_t *shape, int rank
) {
    if (src_ptr == nullptr || dest_ptr == nullptr || rank < 0) return;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];

    switch (dtypeDst) {
        case DTYPE_FLOAT64:
            dispatch_cast_src<double>(src_ptr, stridesSrc, dtypeSrc, dest_ptr, shape, rank, total_elements);
            break;
        case DTYPE_FLOAT32:
            dispatch_cast_src<float>(src_ptr, stridesSrc, dtypeSrc, dest_ptr, shape, rank, total_elements);
            break;
        case DTYPE_FLOAT16:
            dispatch_cast_src<float16_t>(src_ptr, stridesSrc, dtypeSrc, dest_ptr, shape, rank, total_elements);
            break;
        case DTYPE_BFLOAT16:
            dispatch_cast_src<bfloat16_t>(src_ptr, stridesSrc, dtypeSrc, dest_ptr, shape, rank, total_elements);
            break;
        case DTYPE_INT32:
            dispatch_cast_src<int32_t>(src_ptr, stridesSrc, dtypeSrc, dest_ptr, shape, rank, total_elements);
            break;
        case DTYPE_INT64:
            dispatch_cast_src<int64_t>(src_ptr, stridesSrc, dtypeSrc, dest_ptr, shape, rank, total_elements);
            break;
        case DTYPE_UINT8:
            dispatch_cast_src<uint8_t>(src_ptr, stridesSrc, dtypeSrc, dest_ptr, shape, rank, total_elements);
            break;
        case DTYPE_INT16:
            dispatch_cast_src<int16_t>(src_ptr, stridesSrc, dtypeSrc, dest_ptr, shape, rank, total_elements);
            break;
        case DTYPE_COMPLEX128:
            dispatch_cast_src<cpx_t>(src_ptr, stridesSrc, dtypeSrc, dest_ptr, shape, rank, total_elements);
            break;
        case DTYPE_COMPLEX64:
            dispatch_cast_src<cpx_f_t>(src_ptr, stridesSrc, dtypeSrc, dest_ptr, shape, rank, total_elements);
            break;
        case DTYPE_BOOLEAN:
            dispatch_cast_src<bool_byte_t>(src_ptr, stridesSrc, dtypeSrc, dest_ptr, shape, rank, total_elements);
            break;
        case DTYPE_INT8:
            dispatch_cast_src<int8_t>(src_ptr, stridesSrc, dtypeSrc, dest_ptr, shape, rank, total_elements);
            break;
        case DTYPE_UINT64:
            dispatch_cast_src<uint64_t>(src_ptr, stridesSrc, dtypeSrc, dest_ptr, shape, rank, total_elements);
            break;
        case DTYPE_UINT32:
            dispatch_cast_src<uint32_t>(src_ptr, stridesSrc, dtypeSrc, dest_ptr, shape, rank, total_elements);
            break;
        case DTYPE_UINT16:
            dispatch_cast_src<uint16_t>(src_ptr, stridesSrc, dtypeSrc, dest_ptr, shape, rank, total_elements);
            break;
        default:
            return;
    }
}


void v_extract_upper_triangular(
    const void *src_ptr,
    void *dest_ptr,
    int64_t k,
    int64_t n,
    int dtype
) {
    if (src_ptr == nullptr || dest_ptr == nullptr || k <= 0 || n <= 0) return;
    
    switch (dtype) {
        case DTYPE_FLOAT64: {
            const double *src = (const double *)src_ptr;
            double *dest = (double *)dest_ptr;
            for (int64_t i = 0; i < k; i++) {
                for (int64_t j = i; j < n; j++) {
                    dest[i * n + j] = src[i * n + j];
                }
            }
            break;
        }
        case DTYPE_FLOAT32: {
            const float *src = (const float *)src_ptr;
            float *dest = (float *)dest_ptr;
            for (int64_t i = 0; i < k; i++) {
                for (int64_t j = i; j < n; j++) {
                    dest[i * n + j] = src[i * n + j];
                }
            }
            break;
        }
        case DTYPE_COMPLEX128: {
            const cpx_t *src = (const cpx_t *)src_ptr;
            cpx_t *dest = (cpx_t *)dest_ptr;
            for (int64_t i = 0; i < k; i++) {
                for (int64_t j = i; j < n; j++) {
                    dest[i * n + j] = src[i * n + j];
                }
            }
            break;
        }
        case DTYPE_COMPLEX64: {
            const cpx_f_t *src = (const cpx_f_t *)src_ptr;
            cpx_f_t *dest = (cpx_f_t *)dest_ptr;
            for (int64_t i = 0; i < k; i++) {
                for (int64_t j = i; j < n; j++) {
                    dest[i * n + j] = src[i * n + j];
                }
            }
            break;
        }
        default:
            return;
    }
}

void v_zero_upper_triangular(
    void *ptr,
    int64_t n,
    int dtype
) {
    if (ptr == nullptr || n <= 0) return;
    
    switch (dtype) {
        case DTYPE_FLOAT64: {
            double *data = (double *)ptr;
            for (int64_t i = 0; i < n; i++) {
                for (int64_t j = i + 1; j < n; j++) {
                    data[i * n + j] = 0.0;
                }
            }
            break;
        }
        case DTYPE_FLOAT32: {
            float *data = (float *)ptr;
            for (int64_t i = 0; i < n; i++) {
                for (int64_t j = i + 1; j < n; j++) {
                    data[i * n + j] = 0.0f;
                }
            }
            break;
        }
        case DTYPE_COMPLEX128: {
            cpx_t *data = (cpx_t *)ptr;
            for (int64_t i = 0; i < n; i++) {
                for (int64_t j = i + 1; j < n; j++) {
                    data[i * n + j].r = 0.0;
                    data[i * n + j].i = 0.0;
                }
            }
            break;
        }
        case DTYPE_COMPLEX64: {
            cpx_f_t *data = (cpx_f_t *)ptr;
            for (int64_t i = 0; i < n; i++) {
                for (int64_t j = i + 1; j < n; j++) {
                    data[i * n + j].r = 0.0f;
                    data[i * n + j].i = 0.0f;
                }
            }
            break;
        }
        default:
            return;
    }
}

/* ============================================================================
 * SECTION 9: PADDING KERNELS IMPLEMENTATION
 * ============================================================================
 */

static inline int64_t reflect_map(int64_t i, int64_t N) {
    if (N <= 1) return 0;
    int64_t P = 2 * N - 2;
    uint64_t abs_i = (i < 0) ? (0ULL - static_cast<uint64_t>(i)) : static_cast<uint64_t>(i);
    int64_t i_mod = static_cast<int64_t>(abs_i % static_cast<uint64_t>(P));
    return i_mod < N ? i_mod : P - i_mod;
}

static inline int64_t symmetric_map(int64_t i, int64_t N) {
    if (N <= 0) return 0;
    int64_t P = 2 * N;
    int64_t i_mod;
    if (i < 0) {
        uint64_t neg_i_minus_1 = static_cast<uint64_t>(-(i + 1));
        i_mod = static_cast<int64_t>(neg_i_minus_1 % static_cast<uint64_t>(P));
    } else {
        i_mod = i % P;
    }
    return i_mod < N ? i_mod : P - 1 - i_mod;
}

static inline void insertion_sort_uint8(uint8_t *arr, int64_t n, int kind) {
    (void)kind;
    for (int64_t i = 1; i < n; i++) {
        uint8_t key = arr[i];
        int64_t j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

static inline int cmp_double(const void *a, const void *b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}
static inline int cmp_float(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}
static inline int cmp_int64(const void *a, const void *b) {
    int64_t ia = *(const int64_t*)a;
    int64_t ib = *(const int64_t*)b;
    return (ia > ib) - (ia < ib);
}
static inline int cmp_int32(const void *a, const void *b) {
    int32_t ia = *(const int32_t*)a;
    int32_t ib = *(const int32_t*)b;
    return (ia > ib) - (ia < ib);
}
static inline int cmp_uint8(const void *a, const void *b) {
    uint8_t ia = *(const uint8_t*)a;
    uint8_t ib = *(const uint8_t*)b;
    return (ia > ib) - (ia < ib);
}

static inline int cmp_cpx_lex_d(cpx_t a, cpx_t b) {
    if (a.r < b.r) return -1;
    if (a.r > b.r) return 1;
    if (a.i < b.i) return -1;
    if (a.i > b.i) return 1;
    return 0;
}

static inline int cmp_cpx_lex_f(cpx_f_t a, cpx_f_t b) {
    if (a.r < b.r) return -1;
    if (a.r > b.r) return 1;
    if (a.i < b.i) return -1;
    if (a.i > b.i) return 1;
    return 0;
}

static inline double interpolate_double(double start, double end, int64_t step, int64_t total_steps) {
    if (total_steps <= 0) return end;
    return start + (end - start) * (double)step / total_steps;
}
static inline float interpolate_float(float start, float end, int64_t step, int64_t total_steps) {
    if (total_steps <= 0) return end;
    return start + (end - start) * (float)step / total_steps;
}
static inline int64_t interpolate_int64(int64_t start, int64_t end, int64_t step, int64_t total_steps) {
    if (total_steps <= 0) return end;
    if (step <= 0 || start == end) return start;
    if (step >= total_steps) return end;
#if defined(__SIZEOF_INT128__)
    __int128 d = (__int128)total_steps;
    __int128 num = (__int128)start * (d - (__int128)step) + (__int128)end * (__int128)step;
    __int128 q = num / d;
    __int128 r = num % d;
    if (r < 0) q -= 1;
    return (int64_t)q;
#else
    double val = (double)start + ((double)end - (double)start) * ((double)step / (double)total_steps);
    return saturating_float_to_int<int64_t>(std::floor(val));
#endif
}
static inline uint64_t interpolate_uint64(uint64_t start, uint64_t end, int64_t step, int64_t total_steps) {
    if (total_steps <= 0) return end;
    if (step <= 0 || start == end) return start;
    if (step >= total_steps) return end;
#if defined(__SIZEOF_INT128__)
    unsigned __int128 d = (unsigned __int128)total_steps;
    unsigned __int128 num = (unsigned __int128)start * (d - (unsigned __int128)step) + (unsigned __int128)end * (unsigned __int128)step;
    unsigned __int128 q = num / d;
    return (uint64_t)q;
#else
    double val = (double)start + ((double)end - (double)start) * ((double)step / (double)total_steps);
    return saturating_float_to_int<uint64_t>(std::floor(val));
#endif
}
static inline int32_t interpolate_int32(int32_t start, int32_t end, int64_t step, int64_t total_steps) {
    if (total_steps <= 0) return end;
    double val = (double)start + ((double)end - (double)start) * (double)step / total_steps;
    return (int32_t)std::floor(val);
}
static inline int16_t interpolate_int16(int16_t start, int16_t end, int64_t step, int64_t total_steps) {
    if (total_steps <= 0) return end;
    double val = (double)start + ((double)end - (double)start) * (double)step / total_steps;
    return (int16_t)std::floor(val);
}
static inline uint8_t interpolate_uint8(uint8_t start, uint8_t end, int64_t step, int64_t total_steps) {
    if (total_steps <= 0) return end;
    double val = (double)start + ((double)end - (double)start) * (double)step / total_steps;
    return (uint8_t)std::floor(val);
}
static inline cpx_t interpolate_complex128(cpx_t start, cpx_t end, int64_t step, int64_t total_steps) {
    if (total_steps <= 0) return end;
    double r = start.r + (end.r - start.r) * (double)step / total_steps;
    double i = start.i + (end.i - start.i) * (double)step / total_steps;
    return (cpx_t){r, i};
}
static inline cpx_f_t interpolate_complex64(cpx_f_t start, cpx_f_t end, int64_t step, int64_t total_steps) {
    if (total_steps <= 0) return end;
    float r = start.r + (end.r - start.r) * (float)step / total_steps;
    float i = start.i + (end.i - start.i) * (float)step / total_steps;
    return (cpx_f_t){r, i};
}

static inline double round_half_to_even_d(double x) {
    if (!(std::abs(x) < 4503599627370496.0)) return x;
    double fl = std::floor(x);
    double diff = x - fl;
    if (diff > 0.5) return fl + 1.0;
    if (diff < 0.5) return fl;
    int64_t fl_int = (int64_t)fl;
    return ((fl_int & 1) != 0) ? (fl + 1.0) : fl;
}

// Helper macros for defining numeric statistics (double, float, int64, int32, uint8)
template <typename T>
static inline bool is_nan_check(T val) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::isnan(val);
    }
    return false;
}

template <typename PtrT, typename T>
static inline void invoke_sorter(void (*sorter)(PtrT, int64_t, int), T *buf, int64_t len, int kind) {
    sorter(reinterpret_cast<PtrT>(buf), len, kind);
}

static inline int64_t stats_mean_int64_impl(const int64_t *base, int64_t stride, int64_t len) {
#if defined(__SIZEOF_INT128__)
    __int128 sum = 0;
    for (int64_t i = 0; i < len; i++) {
        sum += (__int128)*(base + i * stride);
    }
    __int128 d = (__int128)len;
    __int128 q = sum / d;
    __int128 r = sum % d;
    if (r > 0 && (2 * r > d || (2 * r == d && (q & 1) != 0))) q += 1;
    else if (r < 0 && (-2 * r > d || (-2 * r == d && (q & 1) != 0))) q -= 1;
    return (int64_t)q;
#else
    int64_t q = 0;
    int64_t r = 0;
    for (int64_t i = 0; i < len; i++) {
        int64_t v = *(base + i * stride);
        q += v / len;
        r += v % len;
        q += r / len;
        r %= len;
    }
    if (r < 0 && q > 0) { q -= 1; r += len; }
    else if (r > 0 && q < 0) { q += 1; r -= len; }
    if (r > 0 && (r > len - r || (r == len - r && (q & 1) != 0))) q += 1;
    else if (r < 0 && (-r > len + r || (-r == len + r && (q & 1) != 0))) q -= 1;
    return q;
#endif
}

static inline int64_t stats_median_midpoint_int64_impl(int64_t a, int64_t b) {
#if defined(__SIZEOF_INT128__)
    __int128 sum = (__int128)a + (__int128)b;
    __int128 q = sum / 2;
    __int128 r = sum % 2;
    if (r != 0 && (q & 1) != 0) q += (r > 0 ? 1 : -1);
    return (int64_t)q;
#else
    int64_t q = (a / 2) + (b / 2);
    int64_t r = (a % 2) + (b % 2);
    q += r / 2;
    r %= 2;
    if (r < 0 && q > 0) { q -= 1; r += 2; }
    else if (r > 0 && q < 0) { q += 1; r -= 2; }
    if (r != 0 && (q & 1) != 0) q += (r > 0 ? 1 : -1);
    return q;
#endif
}

#define DEFINE_NUMERIC_STATS(TYPE, NAME_SUFFIX, SORTER, CAST_TYPE) static inline TYPE stats_min_##NAME_SUFFIX(const TYPE *base, int64_t stride, int64_t len) {     if (len <= 0) return (TYPE)0;     TYPE m = *base;     for (int64_t i = 1; i < len; i++) {         if (is_nan_check(m)) break;         TYPE v = *(base + i * stride);         if (is_nan_check(v)) { m = v; break; }         if (v < m) m = v;     }     return m; } static inline TYPE stats_max_##NAME_SUFFIX(const TYPE *base, int64_t stride, int64_t len) {     if (len <= 0) return (TYPE)0;     TYPE m = *base;     for (int64_t i = 1; i < len; i++) {         if (is_nan_check(m)) break;         TYPE v = *(base + i * stride);         if (is_nan_check(v)) { m = v; break; }         if (v > m) m = v;     }     return m; } static inline TYPE stats_mean_##NAME_SUFFIX(const TYPE *base, int64_t stride, int64_t len) {     if (len <= 0) return (TYPE)0;     if constexpr (std::is_same_v<TYPE, int64_t>) {         return (TYPE)stats_mean_int64_impl((const int64_t*)base, stride, len);     } else if constexpr (std::is_floating_point_v<TYPE>) {         double sum = 0;         for (int64_t i = 0; i < len; i++) {             sum += (double)*(base + i * stride);         }         return (TYPE)(sum / len);     } else {         double sum = 0;         for (int64_t i = 0; i < len; i++) {             sum += (double)*(base + i * stride);         }         return (TYPE)round_half_to_even_d(sum / len);     } } static inline TYPE stats_median_##NAME_SUFFIX(const TYPE *base, int64_t _stride, int64_t len) {     if (len <= 0) return (TYPE)0;     TYPE *buf = (TYPE*)malloc(len * sizeof(TYPE));     if (buf == nullptr) { ndarray_set_oom_flag(); return (TYPE)0; }     bool has_nan = false;     for (int64_t i = 0; i < len; i++) {         TYPE v = *(base + i * _stride);         if (is_nan_check(v)) has_nan = true;         buf[i] = v;     }     if (has_nan) {         free(buf);         if constexpr (std::is_floating_point_v<TYPE>) {             return (TYPE)NAN;         }         return (TYPE)0;     }     invoke_sorter(SORTER, (CAST_TYPE)buf, len, 0); /* 0 = quicksort */     TYPE res;     if (len % 2 == 1) {         res = buf[len / 2];     } else if constexpr (std::is_same_v<TYPE, int64_t>) {         res = (TYPE)stats_median_midpoint_int64_impl((int64_t)buf[len / 2 - 1], (int64_t)buf[len / 2]);     } else if constexpr (std::is_floating_point_v<TYPE>) {         res = (TYPE)(((double)buf[len / 2 - 1] + (double)buf[len / 2]) / 2.0);     } else {         res = (TYPE)round_half_to_even_d(((double)buf[len / 2 - 1] + (double)buf[len / 2]) / 2.0);     }     free(buf);     return res; }

DEFINE_NUMERIC_STATS(double, double, native_sort_double, double*)
DEFINE_NUMERIC_STATS(float, float, native_sort_float, float*)
DEFINE_NUMERIC_STATS(int64_t, int64, native_sort_int64, int64_t*)
DEFINE_NUMERIC_STATS(int32_t, int32, native_sort_int32, int*)
DEFINE_NUMERIC_STATS(int16_t, int16, native_sort_int16, int16_t*)
DEFINE_NUMERIC_STATS(uint8_t, uint8, insertion_sort_uint8, uint8_t*)

static inline uint64_t stats_min_uint64(const uint64_t *base, int64_t stride, int64_t len) {
    if (len <= 0) return 0;
    uint64_t m = *base;
    for (int64_t i = 1; i < len; i++) {
        uint64_t v = *(base + i * stride);
        if (v < m) m = v;
    }
    return m;
}
static inline uint64_t stats_max_uint64(const uint64_t *base, int64_t stride, int64_t len) {
    if (len <= 0) return 0;
    uint64_t m = *base;
    for (int64_t i = 1; i < len; i++) {
        uint64_t v = *(base + i * stride);
        if (v > m) m = v;
    }
    return m;
}
static inline uint64_t stats_mean_uint64(const uint64_t *base, int64_t stride, int64_t len) {
    if (len <= 0) return 0;
#if defined(__SIZEOF_INT128__)
    unsigned __int128 sum = 0;
    for (int64_t i = 0; i < len; i++) {
        sum += (unsigned __int128)*(base + i * stride);
    }
    unsigned __int128 d = (unsigned __int128)len;
    unsigned __int128 q = sum / d;
    unsigned __int128 r = sum % d;
    if (2 * r > d || (2 * r == d && (q & 1) != 0)) q += 1;
    return (uint64_t)q;
#else
    uint64_t d = (uint64_t)len;
    uint64_t q = 0;
    uint64_t r = 0;
    for (int64_t i = 0; i < len; i++) {
        uint64_t v = *(base + i * stride);
        q += v / d;
        uint64_t rem = v % d;
        if (r >= d - rem) {
            q += 1;
            r -= (d - rem);
        } else {
            r += rem;
        }
    }
    if (r > d - r || (r == d - r && (q & 1) != 0)) q += 1;
    return q;
#endif
}
static inline uint64_t stats_median_uint64(const uint64_t *base, int64_t _stride, int64_t len) {
    if (len <= 0) return 0;
    uint64_t *buf = (uint64_t*)malloc(len * sizeof(uint64_t));
    if (buf == nullptr) {
        ndarray_set_oom_flag();
        return 0;
    }
    for (int64_t i = 0; i < len; i++) {
        buf[i] = *(base + i * _stride);
    }
    std::sort(buf, buf + len);
    uint64_t res;
    if (len % 2 == 1) {
        res = buf[len / 2];
    } else {
#if defined(__SIZEOF_INT128__)
        unsigned __int128 sum = (unsigned __int128)buf[len / 2 - 1] + (unsigned __int128)buf[len / 2];
        unsigned __int128 q = sum / 2;
        unsigned __int128 r = sum % 2;
        if (r != 0 && (q & 1) != 0) q += 1;
        res = (uint64_t)q;
#else
        uint64_t a = buf[len / 2 - 1];
        uint64_t b = buf[len / 2];
        uint64_t q = (a >> 1) + (b >> 1) + (a & b & 1);
        uint64_t r = (a ^ b) & 1;
        if (r != 0 && (q & 1) != 0) q += 1;
        res = q;
#endif
    }
    free(buf);
    return res;
}


// Complex statistics helpers
static inline cpx_t stats_min_complex128(const cpx_t *base, int64_t stride, int64_t len) {
    if (len <= 0) return (cpx_t){0, 0};
    cpx_t m = *base;
    for (int64_t i = 1; i < len; i++) {
        cpx_t v = *(base + i * stride);
        if (cmp_cpx_lex_d(v, m) < 0) m = v;
    }
    return m;
}
static inline cpx_t stats_max_complex128(const cpx_t *base, int64_t stride, int64_t len) {
    if (len <= 0) return (cpx_t){0, 0};
    cpx_t m = *base;
    for (int64_t i = 1; i < len; i++) {
        cpx_t v = *(base + i * stride);
        if (cmp_cpx_lex_d(v, m) > 0) m = v;
    }
    return m;
}
static inline cpx_t stats_mean_complex128(const cpx_t *base, int64_t stride, int64_t len) {
    if (len <= 0) return (cpx_t){0, 0};
    double sum_r = 0;
    double sum_i = 0;
    for (int64_t i = 0; i < len; i++) {
        cpx_t v = *(base + i * stride);
        sum_r += v.r;
        sum_i += v.i;
    }
    return (cpx_t){sum_r / len, sum_i / len};
}
static inline cpx_t stats_median_complex128(const cpx_t *base, int64_t stride, int64_t len) {
    if (len <= 0) return (cpx_t){0, 0};
    cpx_t *buf = (cpx_t*)malloc(len * sizeof(cpx_t));
    if (buf == nullptr) {
        ndarray_set_oom_flag();
        return (cpx_t){0, 0};
    }
    for (int64_t i = 0; i < len; i++) {
        cpx_t v = *(base + i * stride);
        if (std::isnan(v.r) || std::isnan(v.i)) {
            free(buf);
            return (cpx_t){NAN, NAN};
        }
        buf[i] = v;
    }
    std::sort(buf, buf + len, [](const cpx_t &a, const cpx_t &b) {
        return cmp_cpx_lex_d(a, b) < 0;
    });
    cpx_t res;
    if (len % 2 == 1) {
        res = buf[len / 2];
    } else {
        res.r = (buf[len / 2 - 1].r + buf[len / 2].r) / 2.0;
        res.i = (buf[len / 2 - 1].i + buf[len / 2].i) / 2.0;
    }
    free(buf);
    return res;
}

static inline cpx_f_t stats_min_complex64(const cpx_f_t *base, int64_t stride, int64_t len) {
    if (len <= 0) return (cpx_f_t){0, 0};
    cpx_f_t m = *base;
    for (int64_t i = 1; i < len; i++) {
        cpx_f_t v = *(base + i * stride);
        if (cmp_cpx_lex_f(v, m) < 0) m = v;
    }
    return m;
}
static inline cpx_f_t stats_max_complex64(const cpx_f_t *base, int64_t stride, int64_t len) {
    if (len <= 0) return (cpx_f_t){0, 0};
    cpx_f_t m = *base;
    for (int64_t i = 1; i < len; i++) {
        cpx_f_t v = *(base + i * stride);
        if (cmp_cpx_lex_f(v, m) > 0) m = v;
    }
    return m;
}
static inline cpx_f_t stats_mean_complex64(const cpx_f_t *base, int64_t stride, int64_t len) {
    if (len <= 0) return (cpx_f_t){0, 0};
    float sum_r = 0;
    float sum_i = 0;
    for (int64_t i = 0; i < len; i++) {
        cpx_f_t v = *(base + i * stride);
        sum_r += v.r;
        sum_i += v.i;
    }
    return (cpx_f_t){sum_r / len, sum_i / len};
}
static inline cpx_f_t stats_median_complex64(const cpx_f_t *base, int64_t stride, int64_t len) {
    if (len <= 0) return (cpx_f_t){0, 0};
    cpx_f_t *buf = (cpx_f_t*)malloc(len * sizeof(cpx_f_t));
    if (buf == nullptr) {
        ndarray_set_oom_flag();
        return (cpx_f_t){0, 0};
    }
    for (int64_t i = 0; i < len; i++) {
        cpx_f_t v = *(base + i * stride);
        if (std::isnan(v.r) || std::isnan(v.i)) {
            free(buf);
            return (cpx_f_t){NAN, NAN};
        }
        buf[i] = v;
    }
    std::sort(buf, buf + len, [](const cpx_f_t &a, const cpx_f_t &b) {
        return cmp_cpx_lex_f(a, b) < 0;
    });
    cpx_f_t res;
    if (len % 2 == 1) {
        res = buf[len / 2];
    } else {
        res.r = (buf[len / 2 - 1].r + buf[len / 2].r) / 2.0f;
        res.i = (buf[len / 2 - 1].i + buf[len / 2].i) / 2.0f;
    }
    free(buf);
    return res;
}

#define DEFINE_PAD_AXIS(TYPE_NAME, T, STATS_MIN, STATS_MAX, STATS_MEAN, STATS_MEDIAN, INTERPOLATE) \
void pad_axis_##TYPE_NAME( \
    const T *src, const int64_t *shapeSrc, const int64_t *stridesSrc, \
    T *dest, const int64_t *shapeDest, \
    int rank, int axis, \
    int64_t padBefore, int64_t padAfter, \
    int mode, \
    T constantBefore, T constantAfter, \
    T endBefore, T endAfter, \
    int64_t statLengthBefore, int64_t statLengthAfter \
) { \
    if (dest == nullptr || shapeSrc == nullptr || shapeDest == nullptr || stridesSrc == nullptr || rank <= 0 || axis < 0 || axis >= rank) return; \
    int64_t N = shapeSrc[axis]; \
    if (N <= 0 && mode != 0) return; \
    if (src == nullptr && N > 0) return; \
    int64_t total_elements_dest = 1; \
    for (int64_t i = 0; i < rank; i++) total_elements_dest *= shapeDest[i]; \
    NoThrowBuffer<int64_t> coordDest_vec; \
    int64_t coordDest_stack[32] = {0}; \
    int64_t *coordDest = coordDest_stack; \
    if (rank > 32) { \
        coordDest_vec.assign(rank, 0); \
        if (!coordDest_vec.ok()) { \
            ndarray_set_oom_flag(); \
            return; \
        } \
        coordDest = coordDest_vec.data(); \
    } \
    int64_t offsetSrcSlice = 0; \
    T cached_stat_before = (T)0; \
    T cached_stat_after = (T)0; \
    int stats_cached = 0; \
    int new_slice = 1; \
    for (int64_t el = 0; el < total_elements_dest; el++) { \
        if (new_slice) { \
            stats_cached = 0; \
            new_slice = 0; \
        } \
        const T *src_vector_base = (src != nullptr && N > 0) ? (src + offsetSrcSlice) : nullptr; \
        int64_t src_stride = stridesSrc[axis]; \
        int64_t c = coordDest[axis]; \
        T val; \
        if (c < padBefore) { \
            switch(mode) { \
                case 0: /* constant */ \
                    val = constantBefore; \
                    break; \
                case 1: /* edge */ \
                    val = *src_vector_base; \
                    break; \
                case 2: /* reflect */ { \
                    int64_t src_idx = reflect_map(c - padBefore, N); \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 3: /* symmetric */ { \
                    int64_t src_idx = symmetric_map(c - padBefore, N); \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 4: /* wrap */ { \
                    int64_t src_idx = ((c - padBefore) % N + N) % N; \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 5: /* linear_ramp */ \
                    val = INTERPOLATE(endBefore, *src_vector_base, c, padBefore); \
                    break; \
                case 6: /* maximum */ \
                case 7: /* mean */ \
                case 8: /* median */ \
                case 9: /* minimum */ \
                    if (!stats_cached) { \
                        switch(mode) { \
                            case 6: \
                                cached_stat_after = STATS_MAX(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                cached_stat_before = STATS_MAX(src_vector_base, src_stride, statLengthBefore); \
                                break; \
                            case 7: \
                                cached_stat_after = STATS_MEAN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                cached_stat_before = STATS_MEAN(src_vector_base, src_stride, statLengthBefore); \
                                break; \
                            case 8: \
                                cached_stat_after = STATS_MEDIAN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                cached_stat_before = STATS_MEDIAN(src_vector_base, src_stride, statLengthBefore); \
                                break; \
                            case 9: \
                                cached_stat_after = STATS_MIN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                cached_stat_before = STATS_MIN(src_vector_base, src_stride, statLengthBefore); \
                                break; \
                        } \
                        stats_cached = 1; \
                    } \
                    val = cached_stat_before; \
                    break; \
                default: \
                    val = constantBefore; \
            } \
        } else if (c >= padBefore + N) { \
            switch(mode) { \
                case 0: /* constant */ \
                    val = constantAfter; \
                    break; \
                case 1: /* edge */ \
                    val = *(src_vector_base + (N - 1) * src_stride); \
                    break; \
                case 2: /* reflect */ { \
                    int64_t src_idx = reflect_map(c - padBefore, N); \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 3: /* symmetric */ { \
                    int64_t src_idx = symmetric_map(c - padBefore, N); \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 4: /* wrap */ { \
                    int64_t src_idx = (c - padBefore) % N; \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 5: /* linear_ramp */ { \
                    T edge_val = *(src_vector_base + (N - 1) * src_stride); \
                    int64_t diff = c - (padBefore + N - 1); \
                    val = INTERPOLATE(edge_val, endAfter, diff, padAfter); \
                    break; \
                } \
                case 6: /* maximum */ \
                case 7: /* mean */ \
                case 8: /* median */ \
                case 9: /* minimum */ \
                    if (!stats_cached) { \
                        switch(mode) { \
                            case 6: \
                                cached_stat_after = STATS_MAX(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                cached_stat_before = STATS_MAX(src_vector_base, src_stride, statLengthBefore); \
                                break; \
                            case 7: \
                                cached_stat_after = STATS_MEAN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                cached_stat_before = STATS_MEAN(src_vector_base, src_stride, statLengthBefore); \
                                break; \
                            case 8: \
                                cached_stat_after = STATS_MEDIAN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                cached_stat_before = STATS_MEDIAN(src_vector_base, src_stride, statLengthBefore); \
                                break; \
                            case 9: \
                                cached_stat_after = STATS_MIN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                cached_stat_before = STATS_MIN(src_vector_base, src_stride, statLengthBefore); \
                                break; \
                        } \
                        stats_cached = 1; \
                    } \
                    val = cached_stat_after; \
                    break; \
                default: \
                    val = constantAfter; \
            } \
        } else { \
            val = *(src_vector_base + (c - padBefore) * src_stride); \
        } \
        dest[el] = val; \
        for (int d = rank - 1; d >= 0; d--) { \
            coordDest[d]++; \
            if (coordDest[d] < shapeDest[d]) { \
                if (d != axis) { \
                    offsetSrcSlice += stridesSrc[d]; \
                    new_slice = 1; \
                } \
                break; \
            } \
            coordDest[d] = 0; \
            if (d != axis) { \
                offsetSrcSlice -= (shapeDest[d] - 1) * stridesSrc[d]; \
                new_slice = 1; \
            } \
        } \
    } \
}

// Instantiate DEFINE_PAD_AXIS for double, float, int64, uint64, int32, int16, uint8
DEFINE_PAD_AXIS(double, double, stats_min_double, stats_max_double, stats_mean_double, stats_median_double, interpolate_double)
DEFINE_PAD_AXIS(float, float, stats_min_float, stats_max_float, stats_mean_float, stats_median_float, interpolate_float)
DEFINE_PAD_AXIS(int64, int64_t, stats_min_int64, stats_max_int64, stats_mean_int64, stats_median_int64, interpolate_int64)
DEFINE_PAD_AXIS(uint64, uint64_t, stats_min_uint64, stats_max_uint64, stats_mean_uint64, stats_median_uint64, interpolate_uint64)
DEFINE_PAD_AXIS(int32, int32_t, stats_min_int32, stats_max_int32, stats_mean_int32, stats_median_int32, interpolate_int32)
DEFINE_PAD_AXIS(int16, int16_t, stats_min_int16, stats_max_int16, stats_mean_int16, stats_median_int16, interpolate_int16)
DEFINE_PAD_AXIS(uint8, uint8_t, stats_min_uint8, stats_max_uint8, stats_mean_uint8, stats_median_uint8, interpolate_uint8)

// Complex pad axis implementation (needs separate macro because cpx_t has zeroInit {0,0} and different stats/interpolation signatures)
#define DEFINE_PAD_AXIS_COMPLEX(TYPE_NAME, T, STATS_MIN, STATS_MAX, STATS_MEAN, STATS_MEDIAN, INTERPOLATE) \
void pad_axis_##TYPE_NAME( \
    const T *src, const int64_t *shapeSrc, const int64_t *stridesSrc, \
    T *dest, const int64_t *shapeDest, \
    int rank, int axis, \
    int64_t padBefore, int64_t padAfter, \
    int mode, \
    T constantBefore, T constantAfter, \
    T endBefore, T endAfter, \
    int64_t statLengthBefore, int64_t statLengthAfter \
) { \
    if (dest == nullptr || shapeSrc == nullptr || shapeDest == nullptr || stridesSrc == nullptr || rank <= 0 || axis < 0 || axis >= rank) return; \
    int64_t N = shapeSrc[axis]; \
    if (N <= 0 && mode != 0) return; \
    if (src == nullptr && N > 0) return; \
    int64_t total_elements_dest = 1; \
    for (int64_t i = 0; i < rank; i++) total_elements_dest *= shapeDest[i]; \
    NoThrowBuffer<int64_t> coordDest_vec; \
    int64_t coordDest_stack[32] = {0}; \
    int64_t *coordDest = coordDest_stack; \
    if (rank > 32) { \
        coordDest_vec.assign(rank, 0); \
        if (!coordDest_vec.ok()) { \
            ndarray_set_oom_flag(); \
            return; \
        } \
        coordDest = coordDest_vec.data(); \
    } \
    int64_t offsetSrcSlice = 0; \
    T cached_stat_before = {0, 0}; \
    T cached_stat_after = {0, 0}; \
    int stats_cached = 0; \
    int new_slice = 1; \
    for (int64_t el = 0; el < total_elements_dest; el++) { \
        if (new_slice) { \
            stats_cached = 0; \
            new_slice = 0; \
        } \
        const T *src_vector_base = (src != nullptr && N > 0) ? (src + offsetSrcSlice) : nullptr; \
        int64_t src_stride = stridesSrc[axis]; \
        int64_t c = coordDest[axis]; \
        T val; \
        if (c < padBefore) { \
            switch(mode) { \
                case 0: /* constant */ \
                    val = constantBefore; \
                    break; \
                case 1: /* edge */ \
                    val = *src_vector_base; \
                    break; \
                case 2: /* reflect */ { \
                    int64_t src_idx = reflect_map(c - padBefore, N); \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 3: /* symmetric */ { \
                    int64_t src_idx = symmetric_map(c - padBefore, N); \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 4: /* wrap */ { \
                    int64_t src_idx = ((c - padBefore) % N + N) % N; \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 5: /* linear_ramp */ \
                    val = INTERPOLATE(endBefore, *src_vector_base, c, padBefore); \
                    break; \
                case 6: /* maximum */ \
                case 7: /* mean */ \
                case 8: /* median */ \
                case 9: /* minimum */ \
                    if (!stats_cached) { \
                        switch(mode) { \
                            case 6: \
                                cached_stat_after = STATS_MAX(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                cached_stat_before = STATS_MAX(src_vector_base, src_stride, statLengthBefore); \
                                break; \
                            case 7: \
                                cached_stat_after = STATS_MEAN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                cached_stat_before = STATS_MEAN(src_vector_base, src_stride, statLengthBefore); \
                                break; \
                            case 8: \
                                cached_stat_after = STATS_MEDIAN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                cached_stat_before = STATS_MEDIAN(src_vector_base, src_stride, statLengthBefore); \
                                break; \
                            case 9: \
                                cached_stat_after = STATS_MIN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                cached_stat_before = STATS_MIN(src_vector_base, src_stride, statLengthBefore); \
                                break; \
                        } \
                        stats_cached = 1; \
                    } \
                    val = cached_stat_before; \
                    break; \
                default: \
                    val = constantBefore; \
            } \
        } else if (c >= padBefore + N) { \
            switch(mode) { \
                case 0: /* constant */ \
                    val = constantAfter; \
                    break; \
                case 1: /* edge */ \
                    val = *(src_vector_base + (N - 1) * src_stride); \
                    break; \
                case 2: /* reflect */ { \
                    int64_t src_idx = reflect_map(c - padBefore, N); \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 3: /* symmetric */ { \
                    int64_t src_idx = symmetric_map(c - padBefore, N); \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 4: /* wrap */ { \
                    int64_t src_idx = (c - padBefore) % N; \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 5: /* linear_ramp */ { \
                    T edge_val = *(src_vector_base + (N - 1) * src_stride); \
                    int64_t diff = c - (padBefore + N - 1); \
                    val = INTERPOLATE(edge_val, endAfter, diff, padAfter); \
                    break; \
                } \
                case 6: /* maximum */ \
                case 7: /* mean */ \
                case 8: /* median */ \
                case 9: /* minimum */ \
                    if (!stats_cached) { \
                        switch(mode) { \
                            case 6: \
                                cached_stat_after = STATS_MAX(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                cached_stat_before = STATS_MAX(src_vector_base, src_stride, statLengthBefore); \
                                break; \
                            case 7: \
                                cached_stat_after = STATS_MEAN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                cached_stat_before = STATS_MEAN(src_vector_base, src_stride, statLengthBefore); \
                                break; \
                            case 8: \
                                cached_stat_after = STATS_MEDIAN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                cached_stat_before = STATS_MEDIAN(src_vector_base, src_stride, statLengthBefore); \
                                break; \
                            case 9: \
                                cached_stat_after = STATS_MIN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                cached_stat_before = STATS_MIN(src_vector_base, src_stride, statLengthBefore); \
                                break; \
                        } \
                        stats_cached = 1; \
                    } \
                    val = cached_stat_after; \
                    break; \
                default: \
                    val = constantAfter; \
            } \
        } else { \
            val = *(src_vector_base + (c - padBefore) * src_stride); \
        } \
        dest[el] = val; \
        for (int d = rank - 1; d >= 0; d--) { \
            coordDest[d]++; \
            if (coordDest[d] < shapeDest[d]) { \
                if (d != axis) { \
                    offsetSrcSlice += stridesSrc[d]; \
                    new_slice = 1; \
                } \
                break; \
            } \
            coordDest[d] = 0; \
            if (d != axis) { \
                offsetSrcSlice -= (shapeDest[d] - 1) * stridesSrc[d]; \
                new_slice = 1; \
            } \
        } \
    } \
}

DEFINE_PAD_AXIS_COMPLEX(complex128, cpx_t, stats_min_complex128, stats_max_complex128, stats_mean_complex128, stats_median_complex128, interpolate_complex128)
DEFINE_PAD_AXIS_COMPLEX(complex64, cpx_f_t, stats_min_complex64, stats_max_complex64, stats_mean_complex64, stats_median_complex64, interpolate_complex64)

/* ============================================================================
 * SECTION 10: SET OPERATIONS KERNELS
 * ============================================================================
 */

static inline int set_cmp_double(double a, double b) {
    int nan_a = std::isnan(a);
    int nan_b = std::isnan(b);
    if (nan_a && nan_b) return 0;
    if (nan_a) return 1;
    if (nan_b) return -1;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int set_cmp_float(float a, float b) {
    int nan_a = std::isnan(a);
    int nan_b = std::isnan(b);
    if (nan_a && nan_b) return 0;
    if (nan_a) return 1;
    if (nan_b) return -1;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int set_cmp_int32(int32_t a, int32_t b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int set_cmp_int64(int64_t a, int64_t b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int set_cmp_uint8(uint8_t a, uint8_t b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int set_cmp_complex128(cpx_t a, cpx_t b) {
    int cmp_r = set_cmp_double(a.r, b.r);
    if (cmp_r != 0) return cmp_r;
    return set_cmp_double(a.i, b.i);
}

static inline int set_cmp_complex64(cpx_f_t a, cpx_f_t b) {
    int cmp_r = set_cmp_float(a.r, b.r);
    if (cmp_r != 0) return cmp_r;
    return set_cmp_float(a.i, b.i);
}


extern "C" int64_t ndarray_intersect1d(const void *ar1, int64_t size1, const void *ar2, int64_t size2, void *dest, int dtype) {
    if (ar1 == nullptr || ar2 == nullptr || dest == nullptr) return 0;
    switch (dtype) {
        case DTYPE_FLOAT64: return intersect1d_impl((const double *)ar1, size1, (const double *)ar2, size2, (double *)dest);
        case DTYPE_FLOAT32: return intersect1d_impl((const float *)ar1, size1, (const float *)ar2, size2, (float *)dest);
        case DTYPE_FLOAT16: return intersect1d_impl((const float16_t *)ar1, size1, (const float16_t *)ar2, size2, (float16_t *)dest);
        case DTYPE_BFLOAT16: return intersect1d_impl((const bfloat16_t *)ar1, size1, (const bfloat16_t *)ar2, size2, (bfloat16_t *)dest);
        case DTYPE_INT64: return intersect1d_impl((const int64_t *)ar1, size1, (const int64_t *)ar2, size2, (int64_t *)dest);
        case DTYPE_INT32: return intersect1d_impl((const int32_t *)ar1, size1, (const int32_t *)ar2, size2, (int32_t *)dest);
        case DTYPE_INT16: return intersect1d_impl((const int16_t *)ar1, size1, (const int16_t *)ar2, size2, (int16_t *)dest);
        case DTYPE_INT8: return intersect1d_impl((const int8_t *)ar1, size1, (const int8_t *)ar2, size2, (int8_t *)dest);
        case DTYPE_UINT64: return intersect1d_impl((const uint64_t *)ar1, size1, (const uint64_t *)ar2, size2, (uint64_t *)dest);
        case DTYPE_UINT32: return intersect1d_impl((const uint32_t *)ar1, size1, (const uint32_t *)ar2, size2, (uint32_t *)dest);
        case DTYPE_UINT16: return intersect1d_impl((const uint16_t *)ar1, size1, (const uint16_t *)ar2, size2, (uint16_t *)dest);
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN: return intersect1d_impl((const uint8_t *)ar1, size1, (const uint8_t *)ar2, size2, (uint8_t *)dest);
        case DTYPE_COMPLEX128: return intersect1d_impl((const cpx_t *)ar1, size1, (const cpx_t *)ar2, size2, (cpx_t *)dest);
        case DTYPE_COMPLEX64: return intersect1d_impl((const cpx_f_t *)ar1, size1, (const cpx_f_t *)ar2, size2, (cpx_f_t *)dest);
        default: return 0;
    }
}

extern "C" int64_t ndarray_setdiff1d(const void *ar1, int64_t size1, const void *ar2, int64_t size2, void *dest, int dtype) {
    if (ar1 == nullptr || ar2 == nullptr || dest == nullptr) return 0;
    switch (dtype) {
        case DTYPE_FLOAT64: return setdiff1d_impl((const double *)ar1, size1, (const double *)ar2, size2, (double *)dest);
        case DTYPE_FLOAT32: return setdiff1d_impl((const float *)ar1, size1, (const float *)ar2, size2, (float *)dest);
        case DTYPE_FLOAT16: return setdiff1d_impl((const float16_t *)ar1, size1, (const float16_t *)ar2, size2, (float16_t *)dest);
        case DTYPE_BFLOAT16: return setdiff1d_impl((const bfloat16_t *)ar1, size1, (const bfloat16_t *)ar2, size2, (bfloat16_t *)dest);
        case DTYPE_INT64: return setdiff1d_impl((const int64_t *)ar1, size1, (const int64_t *)ar2, size2, (int64_t *)dest);
        case DTYPE_INT32: return setdiff1d_impl((const int32_t *)ar1, size1, (const int32_t *)ar2, size2, (int32_t *)dest);
        case DTYPE_INT16: return setdiff1d_impl((const int16_t *)ar1, size1, (const int16_t *)ar2, size2, (int16_t *)dest);
        case DTYPE_INT8: return setdiff1d_impl((const int8_t *)ar1, size1, (const int8_t *)ar2, size2, (int8_t *)dest);
        case DTYPE_UINT64: return setdiff1d_impl((const uint64_t *)ar1, size1, (const uint64_t *)ar2, size2, (uint64_t *)dest);
        case DTYPE_UINT32: return setdiff1d_impl((const uint32_t *)ar1, size1, (const uint32_t *)ar2, size2, (uint32_t *)dest);
        case DTYPE_UINT16: return setdiff1d_impl((const uint16_t *)ar1, size1, (const uint16_t *)ar2, size2, (uint16_t *)dest);
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN: return setdiff1d_impl((const uint8_t *)ar1, size1, (const uint8_t *)ar2, size2, (uint8_t *)dest);
        case DTYPE_COMPLEX128: return setdiff1d_impl((const cpx_t *)ar1, size1, (const cpx_t *)ar2, size2, (cpx_t *)dest);
        case DTYPE_COMPLEX64: return setdiff1d_impl((const cpx_f_t *)ar1, size1, (const cpx_f_t *)ar2, size2, (cpx_f_t *)dest);
        default: return 0;
    }
}

extern "C" int64_t ndarray_setxor1d(const void *ar1, int64_t size1, const void *ar2, int64_t size2, void *dest, int dtype) {
    if (ar1 == nullptr || ar2 == nullptr || dest == nullptr) return 0;
    switch (dtype) {
        case DTYPE_FLOAT64: return setxor1d_impl((const double *)ar1, size1, (const double *)ar2, size2, (double *)dest);
        case DTYPE_FLOAT32: return setxor1d_impl((const float *)ar1, size1, (const float *)ar2, size2, (float *)dest);
        case DTYPE_FLOAT16: return setxor1d_impl((const float16_t *)ar1, size1, (const float16_t *)ar2, size2, (float16_t *)dest);
        case DTYPE_BFLOAT16: return setxor1d_impl((const bfloat16_t *)ar1, size1, (const bfloat16_t *)ar2, size2, (bfloat16_t *)dest);
        case DTYPE_INT64: return setxor1d_impl((const int64_t *)ar1, size1, (const int64_t *)ar2, size2, (int64_t *)dest);
        case DTYPE_INT32: return setxor1d_impl((const int32_t *)ar1, size1, (const int32_t *)ar2, size2, (int32_t *)dest);
        case DTYPE_INT16: return setxor1d_impl((const int16_t *)ar1, size1, (const int16_t *)ar2, size2, (int16_t *)dest);
        case DTYPE_INT8: return setxor1d_impl((const int8_t *)ar1, size1, (const int8_t *)ar2, size2, (int8_t *)dest);
        case DTYPE_UINT64: return setxor1d_impl((const uint64_t *)ar1, size1, (const uint64_t *)ar2, size2, (uint64_t *)dest);
        case DTYPE_UINT32: return setxor1d_impl((const uint32_t *)ar1, size1, (const uint32_t *)ar2, size2, (uint32_t *)dest);
        case DTYPE_UINT16: return setxor1d_impl((const uint16_t *)ar1, size1, (const uint16_t *)ar2, size2, (uint16_t *)dest);
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN: return setxor1d_impl((const uint8_t *)ar1, size1, (const uint8_t *)ar2, size2, (uint8_t *)dest);
        case DTYPE_COMPLEX128: return setxor1d_impl((const cpx_t *)ar1, size1, (const cpx_t *)ar2, size2, (cpx_t *)dest);
        case DTYPE_COMPLEX64: return setxor1d_impl((const cpx_f_t *)ar1, size1, (const cpx_f_t *)ar2, size2, (cpx_f_t *)dest);
        default: return 0;
    }
}

extern "C" int64_t ndarray_union1d(const void *ar1, int64_t size1, const void *ar2, int64_t size2, void *dest, int dtype) {
    if (ar1 == nullptr || ar2 == nullptr || dest == nullptr) return 0;
    switch (dtype) {
        case DTYPE_FLOAT64: return union1d_impl((const double *)ar1, size1, (const double *)ar2, size2, (double *)dest);
        case DTYPE_FLOAT32: return union1d_impl((const float *)ar1, size1, (const float *)ar2, size2, (float *)dest);
        case DTYPE_FLOAT16: return union1d_impl((const float16_t *)ar1, size1, (const float16_t *)ar2, size2, (float16_t *)dest);
        case DTYPE_BFLOAT16: return union1d_impl((const bfloat16_t *)ar1, size1, (const bfloat16_t *)ar2, size2, (bfloat16_t *)dest);
        case DTYPE_INT64: return union1d_impl((const int64_t *)ar1, size1, (const int64_t *)ar2, size2, (int64_t *)dest);
        case DTYPE_INT32: return union1d_impl((const int32_t *)ar1, size1, (const int32_t *)ar2, size2, (int32_t *)dest);
        case DTYPE_INT16: return union1d_impl((const int16_t *)ar1, size1, (const int16_t *)ar2, size2, (int16_t *)dest);
        case DTYPE_INT8: return union1d_impl((const int8_t *)ar1, size1, (const int8_t *)ar2, size2, (int8_t *)dest);
        case DTYPE_UINT64: return union1d_impl((const uint64_t *)ar1, size1, (const uint64_t *)ar2, size2, (uint64_t *)dest);
        case DTYPE_UINT32: return union1d_impl((const uint32_t *)ar1, size1, (const uint32_t *)ar2, size2, (uint32_t *)dest);
        case DTYPE_UINT16: return union1d_impl((const uint16_t *)ar1, size1, (const uint16_t *)ar2, size2, (uint16_t *)dest);
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN: return union1d_impl((const uint8_t *)ar1, size1, (const uint8_t *)ar2, size2, (uint8_t *)dest);
        case DTYPE_COMPLEX128: return union1d_impl((const cpx_t *)ar1, size1, (const cpx_t *)ar2, size2, (cpx_t *)dest);
        case DTYPE_COMPLEX64: return union1d_impl((const cpx_f_t *)ar1, size1, (const cpx_f_t *)ar2, size2, (cpx_f_t *)dest);
        default: return 0;
    }
}

extern "C" void ndarray_isin(const void *ar1, int64_t size1, const void *ar2, int64_t size2, uint8_t *dest, int dtype, int invert) {
    if (ar1 == nullptr || ar2 == nullptr || dest == nullptr) return;
    switch (dtype) {
        case DTYPE_FLOAT64: isin_impl((const double *)ar1, size1, (const double *)ar2, size2, dest, invert); break;
        case DTYPE_FLOAT32: isin_impl((const float *)ar1, size1, (const float *)ar2, size2, dest, invert); break;
        case DTYPE_FLOAT16: isin_impl((const float16_t *)ar1, size1, (const float16_t *)ar2, size2, dest, invert); break;
        case DTYPE_BFLOAT16: isin_impl((const bfloat16_t *)ar1, size1, (const bfloat16_t *)ar2, size2, dest, invert); break;
        case DTYPE_INT64: isin_impl((const int64_t *)ar1, size1, (const int64_t *)ar2, size2, dest, invert); break;
        case DTYPE_INT32: isin_impl((const int32_t *)ar1, size1, (const int32_t *)ar2, size2, dest, invert); break;
        case DTYPE_INT16: isin_impl((const int16_t *)ar1, size1, (const int16_t *)ar2, size2, dest, invert); break;
        case DTYPE_INT8: isin_impl((const int8_t *)ar1, size1, (const int8_t *)ar2, size2, dest, invert); break;
        case DTYPE_UINT64: isin_impl((const uint64_t *)ar1, size1, (const uint64_t *)ar2, size2, dest, invert); break;
        case DTYPE_UINT32: isin_impl((const uint32_t *)ar1, size1, (const uint32_t *)ar2, size2, dest, invert); break;
        case DTYPE_UINT16: isin_impl((const uint16_t *)ar1, size1, (const uint16_t *)ar2, size2, dest, invert); break;
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN: isin_impl((const uint8_t *)ar1, size1, (const uint8_t *)ar2, size2, dest, invert); break;
        case DTYPE_COMPLEX128: isin_impl((const cpx_t *)ar1, size1, (const cpx_t *)ar2, size2, dest, invert); break;
        case DTYPE_COMPLEX64: isin_impl((const cpx_f_t *)ar1, size1, (const cpx_f_t *)ar2, size2, dest, invert); break;
        default: return;
    }
}

// --- MEDIAN & QUANTILE IMPLEMENTATIONS ---

// Helper macro for median reduction to avoid code duplication.
// Allocates tmp_buf once per reduction call.
#define IMPLEMENT_MEDIAN_REDUCTION(NAME, TYPE, SORT_FUNC, CAST_TYPE) \
void NAME(const TYPE *src, const int64_t *stridesSrc, \
          TYPE *dest, const int64_t *stridesDest, \
          const int64_t *shape, int rank, int axis) { \
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return; \
    int64_t size_axis = shape[axis]; \
    if (size_axis <= 0) return; \
    TYPE *tmp_buf = (TYPE *)malloc(size_axis * sizeof(TYPE)); \
    if (tmp_buf == nullptr) { ndarray_set_oom_flag(); return; } \
    DECLARE_RANK_BUFFER(int64_t, coord, rank); \
    int64_t outer_size = 1; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) outer_size *= shape[d]; \
    } \
    for (int64_t o = 0; o < outer_size; o++) { \
        int64_t offsetRes = 0; \
        int64_t offsetSrc = 0; \
        for (int d = 0; d < rank; d++) { \
            if (d != axis) { \
                offsetSrc += coord[d] * stridesSrc[d]; \
                if (rank > 1) { \
                    int targetD = (d < axis) ? d : (d - 1); \
                    offsetRes += coord[d] * stridesDest[targetD]; \
                } \
            } \
        } \
        int64_t stride_axis = stridesSrc[axis]; \
        for (int64_t i = 0; i < size_axis; i++) { \
            tmp_buf[i] = src[offsetSrc + i * stride_axis]; \
        } \
        invoke_sorter(SORT_FUNC, (CAST_TYPE)tmp_buf, size_axis, 0); \
        TYPE median_val; \
        if (size_axis % 2 == 1) { \
            median_val = tmp_buf[size_axis / 2]; \
        } else { \
            median_val = (TYPE)(((double)tmp_buf[size_axis / 2 - 1] + (double)tmp_buf[size_axis / 2]) / 2.0); \
        } \
        dest[offsetRes] = median_val; \
        for (int d = rank - 1; d >= 0; d--) { \
            if (d == axis) continue; \
            coord[d]++; \
            if (coord[d] < shape[d]) break; \
            coord[d] = 0; \
        } \
    } \
    free(tmp_buf); \
}

// Special case for double and float where division by 2 should be floating point division
void s_median_double(const double *src, const int64_t *stridesSrc,
                     double *dest, const int64_t *stridesDest,
                     const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) return;
    double *tmp_buf = (double *)malloc(size_axis * sizeof(double));
    if (tmp_buf == nullptr) {
        ndarray_set_oom_flag();
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetRes = 0;
        int64_t offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        int64_t stride_axis = stridesSrc[axis];
        bool has_nan = false;
        for (int64_t i = 0; i < size_axis; i++) {
            double v = src[offsetSrc + i * stride_axis];
            if (std::isnan(v)) has_nan = true;
            tmp_buf[i] = v;
        }
        if (has_nan) {
            dest[offsetRes] = NAN;
        } else {
            native_sort_double(tmp_buf, size_axis, 0);
            double median_val;
            if (size_axis % 2 == 1) {
                median_val = tmp_buf[size_axis / 2];
            } else {
                median_val = (tmp_buf[size_axis / 2 - 1] + tmp_buf[size_axis / 2]) / 2.0;
            }
            dest[offsetRes] = median_val;
        }
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
    free(tmp_buf);
}

void s_median_float(const float *src, const int64_t *stridesSrc,
                    float *dest, const int64_t *stridesDest,
                    const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) return;
    float *tmp_buf = (float *)malloc(size_axis * sizeof(float));
    if (tmp_buf == nullptr) {
        ndarray_set_oom_flag();
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetRes = 0;
        int64_t offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        int64_t stride_axis = stridesSrc[axis];
        bool has_nan = false;
        for (int64_t i = 0; i < size_axis; i++) {
            float v = src[offsetSrc + i * stride_axis];
            if (std::isnan(v)) has_nan = true;
            tmp_buf[i] = v;
        }
        if (has_nan) {
            dest[offsetRes] = NAN;
        } else {
            native_sort_float(tmp_buf, size_axis, 0);
            float median_val;
            if (size_axis % 2 == 1) {
                median_val = tmp_buf[size_axis / 2];
            } else {
                median_val = (tmp_buf[size_axis / 2 - 1] + tmp_buf[size_axis / 2]) / 2.0f;
            }
            dest[offsetRes] = median_val;
        }
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
    free(tmp_buf);
}

IMPLEMENT_MEDIAN_REDUCTION(s_median_int64, int64_t, native_sort_int64, int64_t *)
IMPLEMENT_MEDIAN_REDUCTION(s_median_int32, int32_t, native_sort_int32, int *)
IMPLEMENT_MEDIAN_REDUCTION(s_median_int16, int16_t, native_sort_int16, int16_t *)
IMPLEMENT_MEDIAN_REDUCTION(s_median_uint8, uint8_t, insertion_sort_uint8, uint8_t *)

void s_median_complex128(const cpx_t *src, const int64_t *stridesSrc,
                         cpx_t *dest, const int64_t *stridesDest,
                         const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) return;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetRes = 0;
        int64_t offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        int64_t stride_axis = stridesSrc[axis];
        dest[offsetRes] = stats_median_complex128(src + offsetSrc, stride_axis, size_axis);

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_median_complex64(const cpx_f_t *src, const int64_t *stridesSrc,
                        cpx_f_t *dest, const int64_t *stridesDest,
                        const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) return;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetRes = 0;
        int64_t offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        int64_t stride_axis = stridesSrc[axis];
        dest[offsetRes] = stats_median_complex64(src + offsetSrc, stride_axis, size_axis);

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

// Global reductions for median (contiguous)
double r_median_double(const double *src, int64_t size) { return stats_median_double(src, 1, size); }
float r_median_float(const float *src, int64_t size) { return stats_median_float(src, 1, size); }
int64_t r_median_int64(const int64_t *src, int64_t size) { return stats_median_int64(src, 1, size); }
int32_t r_median_int32(const int32_t *src, int64_t size) { return stats_median_int32(src, 1, size); }
int16_t r_median_int16(const int16_t *src, int64_t size) { return stats_median_int16(src, 1, size); }
uint8_t r_median_uint8(const uint8_t *src, int64_t size) { return stats_median_uint8(src, 1, size); }
cpx_t r_median_complex128(const cpx_t *src, int64_t size) { return stats_median_complex128(src, 1, size); }
cpx_f_t r_median_complex64(const cpx_f_t *src, int64_t size) { return stats_median_complex64(src, 1, size); }

// Quantile helper definitions

typedef struct {
    int64_t idx_low;
    int64_t idx_high;
    double weight;
} QuantileInterpolationSpecs;

static QuantileInterpolationSpecs get_quantile_specs(int64_t N, double p, int method) {
    QuantileInterpolationSpecs specs;
    specs.idx_low = 0;
    specs.idx_high = 0;
    specs.weight = 0.0;

    if (N <= 0 || std::isnan(p)) {
        if (std::isnan(p)) specs.weight = NAN;
        return specs;
    }

    if (method < 0 || method > 12) {
        method = QUANTILE_LINEAR;
    }

    // Continuous methods (4-9)
    if (method >= QUANTILE_INTERPOLATED_INVERTED_CDF && method <= QUANTILE_NORMAL_UNBIASED) {
        double alpha = 0.0, beta = 0.0;
        double idx;
        if (method == QUANTILE_LINEAR) {
            idx = (double)(N - 1) * p;
        } else {
            switch (method) {
                case QUANTILE_INTERPOLATED_INVERTED_CDF: alpha = 0.0; beta = 1.0; break;
                case QUANTILE_HAZEN:                     alpha = 0.5; beta = 0.5; break;
                case QUANTILE_WEIBULL:                   alpha = 0.0; beta = 0.0; break;
                case QUANTILE_MEDIAN_UNBIASED:           alpha = 1.0/3.0; beta = 1.0/3.0; break;
                case QUANTILE_NORMAL_UNBIASED:           alpha = 3.0/8.0; beta = 3.0/8.0; break;
                default: break;
            }
            idx = (double)N * p + (alpha + p * (1.0 - alpha - beta)) - 1.0;
        }
        int64_t j = (int64_t)std::floor(idx);
        specs.idx_low = j;
        specs.idx_high = j + 1;
        specs.weight = idx - (double)j;
    }
    // Discontinuous methods (1-3)
    else if (method == QUANTILE_INVERTED_CDF) {
        double idx = p * N - 1.0;
        double prev = std::floor(idx);
        double gamma = idx - prev;
        int64_t res_idx = (gamma == 0.0) ? (int64_t)prev : (int64_t)prev + 1;
        specs.idx_low = res_idx;
        specs.idx_high = res_idx;
        specs.weight = 0.0;
    }
    else if (method == QUANTILE_AVERAGED_INVERTED_CDF) {
        double idx = p * N - 1.0;
        int64_t j = (int64_t)std::floor(idx);
        double gamma = idx - (double)j;
        specs.idx_low = j;
        specs.idx_high = j + 1;
        specs.weight = (gamma == 0.0) ? 0.5 : 1.0;
    }
    else if (method == QUANTILE_CLOSEST_OBSERVATION) {
        double idx = p * N - 1.5;
        double prev = std::floor(idx);
        double gamma = idx - prev;
        int64_t prev_int = (int64_t)prev;
        int is_odd = (prev_int % 2 != 0);
        int cond = (gamma == 0.0) && is_odd;
        int64_t res_idx = cond ? prev_int : prev_int + 1;
        specs.idx_low = res_idx;
        specs.idx_high = res_idx;
        specs.weight = 0.0;
    }
    // Backward compatibility methods (10-13)
    else if (method == QUANTILE_LOWER) {
        double idx = p * (N - 1);
        int64_t res_idx = (int64_t)std::floor(idx);
        specs.idx_low = res_idx;
        specs.idx_high = res_idx;
        specs.weight = 0.0;
    }
    else if (method == QUANTILE_HIGHER) {
        double idx = p * (N - 1);
        int64_t res_idx = (int64_t)std::ceil(idx);
        specs.idx_low = res_idx;
        specs.idx_high = res_idx;
        specs.weight = 0.0;
    }
    else if (method == QUANTILE_MIDPOINT) {
        double idx = p * (N - 1);
        specs.idx_low = (int64_t)std::floor(idx);
        specs.idx_high = (int64_t)std::ceil(idx);
        specs.weight = 0.5;
    }
    else if (method == QUANTILE_NEAREST) {
        double idx = p * (N - 1);
        int64_t res_idx = (int64_t)std::rint(idx);
        specs.idx_low = res_idx;
        specs.idx_high = res_idx;
        specs.weight = 0.0;
    }

    // Clip indices
    if (specs.idx_low < 0) specs.idx_low = 0;
    if (specs.idx_low >= N) specs.idx_low = N - 1;
    if (specs.idx_high < 0) specs.idx_high = 0;
    if (specs.idx_high >= N) specs.idx_high = N - 1;

    return specs;
}

#define DEFINE_NUMERIC_QUANTILE(TYPE, NAME_SUFFIX, SORTER, CAST_TYPE) \
static inline double stats_quantile_##NAME_SUFFIX(const TYPE *base, int64_t _stride, int64_t len, double q, int method) { \
    if (len <= 0) return 0.0; \
    if (std::isnan(q)) return NAN; \
    if (q < 0.0) q = 0.0; \
    if (q > 1.0) q = 1.0; \
    TYPE *buf = (TYPE*)malloc(len * sizeof(TYPE)); \
    if (buf == nullptr) { ndarray_set_oom_flag(); return 0.0; } \
    bool has_nan = false; \
    for (int64_t i = 0; i < len; i++) { \
        TYPE v = *(base + i * _stride); \
        if (is_nan_check(v)) has_nan = true; \
        buf[i] = v; \
    } \
    if (has_nan) { \
        free(buf); \
        if constexpr (std::is_floating_point_v<TYPE>) { \
            return NAN; \
        } \
        return 0.0; \
    } \
    invoke_sorter(SORTER, (CAST_TYPE)buf, len, 0); \
    QuantileInterpolationSpecs specs = get_quantile_specs(len, q, method); \
    double res = (1.0 - specs.weight) * (double)buf[specs.idx_low] + specs.weight * (double)buf[specs.idx_high]; \
    free(buf); \
    return res; \
}

DEFINE_NUMERIC_QUANTILE(double, double, native_sort_double, double*)
DEFINE_NUMERIC_QUANTILE(float, float, native_sort_float, float*)
DEFINE_NUMERIC_QUANTILE(int64_t, int64, native_sort_int64, int64_t*)
DEFINE_NUMERIC_QUANTILE(int32_t, int32, native_sort_int32, int*)
DEFINE_NUMERIC_QUANTILE(int16_t, int16, native_sort_int16, int16_t*)
DEFINE_NUMERIC_QUANTILE(uint8_t, uint8, insertion_sort_uint8, uint8_t*)

// Global reductions for quantile (contiguous)
double r_quantile_double(const double *src, int64_t size, double q, int method) { return stats_quantile_double(src, 1, size, q, method); }
double r_quantile_float(const float *src, int64_t size, double q, int method) { return stats_quantile_float(src, 1, size, q, method); }
double r_quantile_int64(const int64_t *src, int64_t size, double q, int method) { return stats_quantile_int64(src, 1, size, q, method); }
double r_quantile_int32(const int32_t *src, int64_t size, double q, int method) { return stats_quantile_int32(src, 1, size, q, method); }
double r_quantile_int16(const int16_t *src, int64_t size, double q, int method) { return stats_quantile_int16(src, 1, size, q, method); }
double r_quantile_uint8(const uint8_t *src, int64_t size, double q, int method) { return stats_quantile_uint8(src, 1, size, q, method); }

// Helper macro for quantile reduction.
// Allocates tmp_buf once per reduction call.
#define IMPLEMENT_QUANTILE_REDUCTION(NAME, TYPE, SORT_FUNC, CAST_TYPE) \
void NAME(const TYPE *src, const int64_t *stridesSrc, \
          double *dest, const int64_t *stridesDest, \
          const int64_t *shape, int rank, int axis, double q, int method) { \
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return; \
    int64_t size_axis = shape[axis]; \
    if (size_axis <= 0 || std::isnan(q) || q < 0.0 || q > 1.0) { \
        fill_reduction_dest<double>(dest, stridesDest, shape, rank, axis, NAN); \
        return; \
    } \
    TYPE *tmp_buf = (TYPE *)malloc(size_axis * sizeof(TYPE)); \
    if (tmp_buf == nullptr) { ndarray_set_oom_flag(); return; } \
    DECLARE_RANK_BUFFER(int64_t, coord, rank); \
    int64_t outer_size = 1; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) outer_size *= shape[d]; \
    } \
    QuantileInterpolationSpecs specs = get_quantile_specs(size_axis, q, method); \
    for (int64_t o = 0; o < outer_size; o++) { \
        int64_t offsetRes = 0; \
        int64_t offsetSrc = 0; \
        for (int d = 0; d < rank; d++) { \
            if (d != axis) { \
                offsetSrc += coord[d] * stridesSrc[d]; \
                if (rank > 1) { \
                    int targetD = (d < axis) ? d : (d - 1); \
                    offsetRes += coord[d] * stridesDest[targetD]; \
                } \
            } \
        } \
        int64_t stride_axis = stridesSrc[axis]; \
        bool has_nan = false; \
        for (int64_t i = 0; i < size_axis; i++) { \
            TYPE v = src[offsetSrc + i * stride_axis]; \
            if (is_nan_check(v)) has_nan = true; \
            tmp_buf[i] = v; \
        } \
        if (has_nan) { \
            if constexpr (std::is_floating_point_v<TYPE>) { \
                dest[offsetRes] = NAN; \
            } else { \
                dest[offsetRes] = 0.0; \
            } \
        } else { \
            invoke_sorter(SORT_FUNC, (CAST_TYPE)tmp_buf, size_axis, 0); \
            double val = (1.0 - specs.weight) * (double)tmp_buf[specs.idx_low] + specs.weight * (double)tmp_buf[specs.idx_high]; \
            dest[offsetRes] = val; \
        } \
        for (int d = rank - 1; d >= 0; d--) { \
            if (d == axis) continue; \
            coord[d]++; \
            if (coord[d] < shape[d]) break; \
            coord[d] = 0; \
        } \
    } \
    free(tmp_buf); \
}

IMPLEMENT_QUANTILE_REDUCTION(s_quantile_double, double, native_sort_double, double *)
IMPLEMENT_QUANTILE_REDUCTION(s_quantile_float, float, native_sort_float, float *)
IMPLEMENT_QUANTILE_REDUCTION(s_quantile_int64, int64_t, native_sort_int64, int64_t *)
IMPLEMENT_QUANTILE_REDUCTION(s_quantile_int32, int32_t, native_sort_int32, int *)
IMPLEMENT_QUANTILE_REDUCTION(s_quantile_int16, int16_t, native_sort_int16, int16_t *)
IMPLEMENT_QUANTILE_REDUCTION(s_quantile_uint8, uint8_t, insertion_sort_uint8, uint8_t *)

// Nanmedian implementations
double r_nanmedian_double(const double *src, int64_t size) {
    if (src == nullptr || size <= 0) return NAN;
    double *tmp_buf = (double *)malloc(size * sizeof(double));
    if (tmp_buf == nullptr) {
        ndarray_set_oom_flag();
        return NAN;
    }
    int64_t valid = 0;
    for (int64_t i = 0; i < size; i++) {
        double v = src[i];
        if (!std::isnan(v)) {
            tmp_buf[valid++] = v;
        }
    }
    if (valid == 0) {
        free(tmp_buf);
        return NAN;
    }
    native_sort_double(tmp_buf, valid, 0);
    double res;
    if (valid % 2 == 1) {
        res = tmp_buf[valid / 2];
    } else {
        res = (tmp_buf[valid / 2 - 1] + tmp_buf[valid / 2]) / 2.0;
    }
    free(tmp_buf);
    return res;
}

float r_nanmedian_float(const float *src, int64_t size) {
    if (src == nullptr || size <= 0) return NAN;
    float *tmp_buf = (float *)malloc(size * sizeof(float));
    if (tmp_buf == nullptr) {
        ndarray_set_oom_flag();
        return NAN;
    }
    int64_t valid = 0;
    for (int64_t i = 0; i < size; i++) {
        float v = src[i];
        if (!std::isnan(v)) {
            tmp_buf[valid++] = v;
        }
    }
    if (valid == 0) {
        free(tmp_buf);
        return NAN;
    }
    native_sort_float(tmp_buf, valid, 0);
    float res;
    if (valid % 2 == 1) {
        res = tmp_buf[valid / 2];
    } else {
        res = (tmp_buf[valid / 2 - 1] + tmp_buf[valid / 2]) / 2.0f;
    }
    free(tmp_buf);
    return res;
}

cpx_t r_nanmedian_complex128(const cpx_t *src, int64_t size) {
    if (src == nullptr || size <= 0) return (cpx_t){NAN, NAN};
    cpx_t *tmp_buf = (cpx_t *)malloc(size * sizeof(cpx_t));
    if (tmp_buf == nullptr) {
        ndarray_set_oom_flag();
        return (cpx_t){NAN, NAN};
    }
    int64_t valid = 0;
    for (int64_t i = 0; i < size; i++) {
        cpx_t v = src[i];
        if (!std::isnan(v.r) && !std::isnan(v.i)) {
            tmp_buf[valid++] = v;
        }
    }
    if (valid == 0) {
        free(tmp_buf);
        return (cpx_t){NAN, NAN};
    }
    std::sort(tmp_buf, tmp_buf + valid, [](const cpx_t &a, const cpx_t &b) {
        return cmp_cpx_lex_d(a, b) < 0;
    });
    cpx_t res;
    if (valid % 2 == 1) {
        res = tmp_buf[valid / 2];
    } else {
        res.r = (tmp_buf[valid / 2 - 1].r + tmp_buf[valid / 2].r) / 2.0;
        res.i = (tmp_buf[valid / 2 - 1].i + tmp_buf[valid / 2].i) / 2.0;
    }
    free(tmp_buf);
    return res;
}

cpx_f_t r_nanmedian_complex64(const cpx_f_t *src, int64_t size) {
    if (src == nullptr || size <= 0) return (cpx_f_t){NAN, NAN};
    cpx_f_t *tmp_buf = (cpx_f_t *)malloc(size * sizeof(cpx_f_t));
    if (tmp_buf == nullptr) {
        ndarray_set_oom_flag();
        return (cpx_f_t){NAN, NAN};
    }
    int64_t valid = 0;
    for (int64_t i = 0; i < size; i++) {
        cpx_f_t v = src[i];
        if (!std::isnan(v.r) && !std::isnan(v.i)) {
            tmp_buf[valid++] = v;
        }
    }
    if (valid == 0) {
        free(tmp_buf);
        return (cpx_f_t){NAN, NAN};
    }
    std::sort(tmp_buf, tmp_buf + valid, [](const cpx_f_t &a, const cpx_f_t &b) {
        return cmp_cpx_lex_f(a, b) < 0;
    });
    cpx_f_t res;
    if (valid % 2 == 1) {
        res = tmp_buf[valid / 2];
    } else {
        res.r = (tmp_buf[valid / 2 - 1].r + tmp_buf[valid / 2].r) / 2.0f;
        res.i = (tmp_buf[valid / 2 - 1].i + tmp_buf[valid / 2].i) / 2.0f;
    }
    free(tmp_buf);
    return res;
}

void s_nanmedian_double(const double *src, const int64_t *stridesSrc,
                        double *dest, const int64_t *stridesDest,
                        const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        fill_reduction_dest<double>(dest, stridesDest, shape, rank, axis, NAN);
        return;
    }
    double *tmp_buf = (double *)malloc(size_axis * sizeof(double));
    if (tmp_buf == nullptr) {
        ndarray_set_oom_flag();
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetRes = 0;
        int64_t offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        int64_t stride_axis = stridesSrc[axis];
        int64_t valid = 0;
        for (int64_t i = 0; i < size_axis; i++) {
            double v = src[offsetSrc + i * stride_axis];
            if (!std::isnan(v)) {
                tmp_buf[valid++] = v;
            }
        }
        if (valid == 0) {
            dest[offsetRes] = NAN;
        } else {
            native_sort_double(tmp_buf, valid, 0);
            double median_val;
            if (valid % 2 == 1) {
                median_val = tmp_buf[valid / 2];
            } else {
                median_val = (tmp_buf[valid / 2 - 1] + tmp_buf[valid / 2]) / 2.0;
            }
            dest[offsetRes] = median_val;
        }
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
    free(tmp_buf);
}

void s_nanmedian_float(const float *src, const int64_t *stridesSrc,
                       float *dest, const int64_t *stridesDest,
                       const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        fill_reduction_dest<float>(dest, stridesDest, shape, rank, axis, NAN);
        return;
    }
    float *tmp_buf = (float *)malloc(size_axis * sizeof(float));
    if (tmp_buf == nullptr) {
        ndarray_set_oom_flag();
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetRes = 0;
        int64_t offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        int64_t stride_axis = stridesSrc[axis];
        int64_t valid = 0;
        for (int64_t i = 0; i < size_axis; i++) {
            float v = src[offsetSrc + i * stride_axis];
            if (!std::isnan(v)) {
                tmp_buf[valid++] = v;
            }
        }
        if (valid == 0) {
            dest[offsetRes] = NAN;
        } else {
            native_sort_float(tmp_buf, valid, 0);
            float median_val;
            if (valid % 2 == 1) {
                median_val = tmp_buf[valid / 2];
            } else {
                median_val = (tmp_buf[valid / 2 - 1] + tmp_buf[valid / 2]) / 2.0f;
            }
            dest[offsetRes] = median_val;
        }
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
    free(tmp_buf);
}

void s_nanmedian_complex128(const cpx_t *src, const int64_t *stridesSrc,
                            cpx_t *dest, const int64_t *stridesDest,
                            const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        fill_reduction_dest<cpx_t>(dest, stridesDest, shape, rank, axis, cpx_t{NAN, NAN});
        return;
    }
    cpx_t *tmp_buf = (cpx_t *)malloc(size_axis * sizeof(cpx_t));
    if (tmp_buf == nullptr) {
        ndarray_set_oom_flag();
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetRes = 0;
        int64_t offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        int64_t stride_axis = stridesSrc[axis];
        int64_t valid = 0;
        for (int64_t i = 0; i < size_axis; i++) {
            cpx_t v = src[offsetSrc + i * stride_axis];
            if (!std::isnan(v.r) && !std::isnan(v.i)) {
                tmp_buf[valid++] = v;
            }
        }
        if (valid == 0) {
            dest[offsetRes] = (cpx_t){NAN, NAN};
        } else {
            std::sort(tmp_buf, tmp_buf + valid, [](const cpx_t &a, const cpx_t &b) {
                return cmp_cpx_lex_d(a, b) < 0;
            });
            cpx_t median_val;
            if (valid % 2 == 1) {
                median_val = tmp_buf[valid / 2];
            } else {
                median_val.r = (tmp_buf[valid / 2 - 1].r + tmp_buf[valid / 2].r) / 2.0;
                median_val.i = (tmp_buf[valid / 2 - 1].i + tmp_buf[valid / 2].i) / 2.0;
            }
            dest[offsetRes] = median_val;
        }
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
    free(tmp_buf);
}

void s_nanmedian_complex64(const cpx_f_t *src, const int64_t *stridesSrc,
                           cpx_f_t *dest, const int64_t *stridesDest,
                           const int64_t *shape, int rank, int axis) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) {
        fill_reduction_dest<cpx_f_t>(dest, stridesDest, shape, rank, axis, cpx_f_t{NAN, NAN});
        return;
    }
    cpx_f_t *tmp_buf = (cpx_f_t *)malloc(size_axis * sizeof(cpx_f_t));
    if (tmp_buf == nullptr) {
        ndarray_set_oom_flag();
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetRes = 0;
        int64_t offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        int64_t stride_axis = stridesSrc[axis];
        int64_t valid = 0;
        for (int64_t i = 0; i < size_axis; i++) {
            cpx_f_t v = src[offsetSrc + i * stride_axis];
            if (!std::isnan(v.r) && !std::isnan(v.i)) {
                tmp_buf[valid++] = v;
            }
        }
        if (valid == 0) {
            dest[offsetRes] = (cpx_f_t){NAN, NAN};
        } else {
            std::sort(tmp_buf, tmp_buf + valid, [](const cpx_f_t &a, const cpx_f_t &b) {
                return cmp_cpx_lex_f(a, b) < 0;
            });
            cpx_f_t median_val;
            if (valid % 2 == 1) {
                median_val = tmp_buf[valid / 2];
            } else {
                median_val.r = (tmp_buf[valid / 2 - 1].r + tmp_buf[valid / 2].r) / 2.0f;
                median_val.i = (tmp_buf[valid / 2 - 1].i + tmp_buf[valid / 2].i) / 2.0f;
            }
            dest[offsetRes] = median_val;
        }
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
    free(tmp_buf);
}

// Nanquantile implementations
double r_nanquantile_double(const double *src, int64_t size, double q, int method) {
    if (src == nullptr || size <= 0) return NAN;
    if (std::isnan(q) || q < 0.0 || q > 1.0) return NAN;
    double *tmp_buf = (double *)malloc(size * sizeof(double));
    if (tmp_buf == nullptr) {
        ndarray_set_oom_flag();
        return NAN;
    }
    int64_t valid = 0;
    for (int64_t i = 0; i < size; i++) {
        double v = src[i];
        if (!std::isnan(v)) {
            tmp_buf[valid++] = v;
        }
    }
    if (valid == 0) {
        free(tmp_buf);
        return NAN;
    }
    native_sort_double(tmp_buf, valid, 0);
    QuantileInterpolationSpecs specs = get_quantile_specs(valid, q, method);
    double res = (1.0 - specs.weight) * tmp_buf[specs.idx_low] + specs.weight * tmp_buf[specs.idx_high];
    free(tmp_buf);
    return res;
}

double r_nanquantile_float(const float *src, int64_t size, double q, int method) {
    if (src == nullptr || size <= 0) return NAN;
    if (std::isnan(q) || q < 0.0 || q > 1.0) return NAN;
    float *tmp_buf = (float *)malloc(size * sizeof(float));
    if (tmp_buf == nullptr) {
        ndarray_set_oom_flag();
        return NAN;
    }
    int64_t valid = 0;
    for (int64_t i = 0; i < size; i++) {
        float v = src[i];
        if (!std::isnan(v)) {
            tmp_buf[valid++] = v;
        }
    }
    if (valid == 0) {
        free(tmp_buf);
        return NAN;
    }
    native_sort_float(tmp_buf, valid, 0);
    QuantileInterpolationSpecs specs = get_quantile_specs(valid, q, method);
    double res = (1.0 - specs.weight) * (double)tmp_buf[specs.idx_low] + specs.weight * (double)tmp_buf[specs.idx_high];
    free(tmp_buf);
    return res;
}

void s_nanquantile_double(const double *src, const int64_t *stridesSrc,
                          double *dest, const int64_t *stridesDest,
                          const int64_t *shape, int rank, int axis, double q, int method) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0 || std::isnan(q) || q < 0.0 || q > 1.0) {
        fill_reduction_dest<double>(dest, stridesDest, shape, rank, axis, NAN);
        return;
    }
    double *tmp_buf = (double *)malloc(size_axis * sizeof(double));
    if (tmp_buf == nullptr) {
        ndarray_set_oom_flag();
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetRes = 0;
        int64_t offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        int64_t stride_axis = stridesSrc[axis];
        int64_t valid = 0;
        for (int64_t i = 0; i < size_axis; i++) {
            double v = src[offsetSrc + i * stride_axis];
            if (!std::isnan(v)) {
                tmp_buf[valid++] = v;
            }
        }
        if (valid == 0) {
            dest[offsetRes] = NAN;
        } else {
            native_sort_double(tmp_buf, valid, 0);
            QuantileInterpolationSpecs specs = get_quantile_specs(valid, q, method);
            double val = (1.0 - specs.weight) * tmp_buf[specs.idx_low] + specs.weight * tmp_buf[specs.idx_high];
            dest[offsetRes] = val;
        }
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
    free(tmp_buf);
}

void s_nanquantile_float(const float *src, const int64_t *stridesSrc,
                         double *dest, const int64_t *stridesDest,
                         const int64_t *shape, int rank, int axis, double q, int method) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0 || std::isnan(q) || q < 0.0 || q > 1.0) {
        fill_reduction_dest<double>(dest, stridesDest, shape, rank, axis, NAN);
        return;
    }
    float *tmp_buf = (float *)malloc(size_axis * sizeof(float));
    if (tmp_buf == nullptr) {
        ndarray_set_oom_flag();
        return;
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    for (int64_t o = 0; o < outer_size; o++) {
        int64_t offsetRes = 0;
        int64_t offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        int64_t stride_axis = stridesSrc[axis];
        int64_t valid = 0;
        for (int64_t i = 0; i < size_axis; i++) {
            float v = src[offsetSrc + i * stride_axis];
            if (!std::isnan(v)) {
                tmp_buf[valid++] = v;
            }
        }
        if (valid == 0) {
            dest[offsetRes] = NAN;
        } else {
            native_sort_float(tmp_buf, valid, 0);
            QuantileInterpolationSpecs specs = get_quantile_specs(valid, q, method);
            double val = (1.0 - specs.weight) * (double)tmp_buf[specs.idx_low] + specs.weight * (double)tmp_buf[specs.idx_high];
            dest[offsetRes] = val;
        }
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
    free(tmp_buf);
}
/* ============================================================================
 * SECTION 11: INTERPOLATION KERNELS
 * ============================================================================
 */

int is_strictly_increasing_double(const double *arr, int64_t size, int64_t stride) {
    if (size <= 1) return 1;
    for (int64_t i = 1; i < size; i++) {
        if (arr[i * stride] <= arr[(i - 1) * stride]) {
            return 0;
        }
    }
    return 1;
}

template <typename T>
static inline void fast_interp_vector(const T *x, int64_t x_size,
                                      const T *xp, int64_t xp_size,
                                      const T *fp, T *res,
                                      const T *left, const T *right) {
    if (x == nullptr || xp == nullptr || fp == nullptr || res == nullptr || x_size <= 0 || xp_size <= 0) {
        return;
    }
    if (xp_size == 1) {
        T val = fp[0];
        T lval = (left != nullptr) ? *left : val;
        T rval = (right != nullptr) ? *right : val;
        T xp0 = xp[0];
        for (int64_t i = 0; i < x_size; i++) {
            T xv = x[i];
            if (std::isnan(xv)) {
                res[i] = std::numeric_limits<T>::quiet_NaN();
            } else if (xv < xp0) {
                res[i] = lval;
            } else if (xv > xp0) {
                res[i] = rval;
            } else {
                res[i] = val;
            }
        }
        return;
    }

    T xp_min = xp[0];
    T xp_max = xp[xp_size - 1];
    T lval = (left != nullptr) ? *left : fp[0];
    T rval = (right != nullptr) ? *right : fp[xp_size - 1];

    int64_t j = 0;
    T x0 = xp[0];
    T x1 = xp[1];
    T y0 = fp[0];
    T y1 = fp[1];
    T dx = x1 - x0;
    T slope = (dx != 0) ? (y1 - y0) / dx : static_cast<T>(0);

    for (int64_t i = 0; i < x_size; i++) {
        T xv = x[i];
        if (std::isnan(xv)) {
            res[i] = std::numeric_limits<T>::quiet_NaN();
            continue;
        }
        if (xv < xp_min) {
            res[i] = lval;
            continue;
        }
        if (xv > xp_max) {
            res[i] = rval;
            continue;
        }
        if (xv == xp_max) {
            res[i] = fp[xp_size - 1];
            continue;
        }

        // Fast path for monotonic / localized query streams
        if (xv >= x0 && xv < x1) {
            // within current interval
        } else if (j + 1 < xp_size - 1 && xv >= x1 && xv < xp[j + 2]) {
            j++;
            x0 = x1;
            x1 = xp[j + 1];
            y0 = y1;
            y1 = fp[j + 1];
            dx = x1 - x0;
            slope = (dx != 0) ? (y1 - y0) / dx : static_cast<T>(0);
        } else if (j > 0 && xv >= xp[j - 1] && xv < x0) {
            j--;
            x1 = x0;
            x0 = xp[j];
            y1 = y0;
            y0 = fp[j];
            dx = x1 - x0;
            slope = (dx != 0) ? (y1 - y0) / dx : static_cast<T>(0);
        } else {
            // Binary search using std::upper_bound
            const T *search_start = (j + 2 < xp_size && xv >= xp[j + 2]) ? (xp + j + 2) : xp;
            const T *it = std::upper_bound(search_start, xp + xp_size, xv);
            j = static_cast<int64_t>(it - xp) - 1;
            if (j < 0) j = 0;
            if (j >= xp_size - 1) j = xp_size - 2;
            x0 = xp[j];
            x1 = xp[j + 1];
            y0 = fp[j];
            y1 = fp[j + 1];
            dx = x1 - x0;
            slope = (dx != 0) ? (y1 - y0) / dx : static_cast<T>(0);
        }

        if (xv == x0 || dx == 0) {
            res[i] = y0;
        } else if (std::isfinite(slope)) {
            res[i] = y0 + slope * (xv - x0);
        } else {
            T t = (xv - x0) / dx;
            res[i] = (static_cast<T>(1) - t) * y0 + t * y1;
        }
    }
}

template <typename T>
static inline void fast_interp_strided(const T *x, const int64_t *stridesX,
                                       const T *xp, int64_t strideXP, int64_t xp_size,
                                       const T *fp, int64_t strideFP,
                                       T *res, const int64_t *stridesRes,
                                       const int64_t *shape, int rank,
                                       const T *left, const T *right) {
    if (x == nullptr || xp == nullptr || fp == nullptr || res == nullptr || (rank > 0 && shape == nullptr) || rank < 0 || xp_size <= 0) {
        return;
    }

    if (xp_size == 1) {
        T val = fp[0];
        T lval = (left != nullptr) ? *left : val;
        T rval = (right != nullptr) ? *right : val;
        T xp0 = xp[0];
        int64_t total_elements = 1;
        for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetX = 0, offsetRes = 0;
        for (int64_t el = 0; el < total_elements; el++) {
            T xv = x[offsetX];
            if (std::isnan(xv)) {
                res[offsetRes] = std::numeric_limits<T>::quiet_NaN();
            } else if (xv < xp0) {
                res[offsetRes] = lval;
            } else if (xv > xp0) {
                res[offsetRes] = rval;
            } else {
                res[offsetRes] = val;
            }
            for (int d = rank - 1; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetX += stridesX[d];
                    offsetRes += stridesRes[d];
                    break;
                }
                coord[d] = 0;
                offsetX -= (shape[d] - 1) * stridesX[d];
                offsetRes -= (shape[d] - 1) * stridesRes[d];
            }
        }
        return;
    }

    T xp_min = xp[0];
    T xp_max = xp[(xp_size - 1) * strideXP];
    T lval = (left != nullptr) ? *left : fp[0];
    T rval = (right != nullptr) ? *right : fp[(xp_size - 1) * strideFP];

    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetX = 0, offsetRes = 0;
    int64_t j = 0;
    T x0 = xp[0];
    T x1 = xp[strideXP];
    T y0 = fp[0];
    T y1 = fp[strideFP];
    T dx = x1 - x0;
    T slope = (dx != 0) ? (y1 - y0) / dx : static_cast<T>(0);

    for (int64_t el = 0; el < total_elements; el++) {
        T xv = x[offsetX];
        if (std::isnan(xv)) {
            res[offsetRes] = std::numeric_limits<T>::quiet_NaN();
        } else if (xv < xp_min) {
            res[offsetRes] = lval;
        } else if (xv > xp_max) {
            res[offsetRes] = rval;
        } else if (xv == xp_max) {
            res[offsetRes] = fp[(xp_size - 1) * strideFP];
        } else {
            if (xv >= x0 && xv < x1) {
                // in current interval
            } else if (j + 1 < xp_size - 1 && xv >= x1 && xv < xp[(j + 2) * strideXP]) {
                j++;
                x0 = x1;
                x1 = xp[(j + 1) * strideXP];
                y0 = y1;
                y1 = fp[(j + 1) * strideFP];
                dx = x1 - x0;
                slope = (dx != 0) ? (y1 - y0) / dx : static_cast<T>(0);
            } else if (j > 0 && xv >= xp[(j - 1) * strideXP] && xv < x0) {
                j--;
                x1 = x0;
                x0 = xp[j * strideXP];
                y1 = y0;
                y0 = fp[j * strideFP];
                dx = x1 - x0;
                slope = (dx != 0) ? (y1 - y0) / dx : static_cast<T>(0);
            } else {
                int64_t low = (j + 2 < xp_size && xv >= xp[(j + 2) * strideXP]) ? (j + 2) : 0;
                int64_t high = xp_size - 1;
                while (low < high - 1) {
                    int64_t mid = low + (high - low) / 2;
                    if (xp[mid * strideXP] <= xv) {
                        low = mid;
                    } else {
                        high = mid;
                    }
                }
                j = low;
                x0 = xp[j * strideXP];
                x1 = xp[(j + 1) * strideXP];
                y0 = fp[j * strideFP];
                y1 = fp[(j + 1) * strideFP];
                dx = x1 - x0;
                slope = (dx != 0) ? (y1 - y0) / dx : static_cast<T>(0);
            }

            if (xv == x0 || dx == 0) {
                res[offsetRes] = y0;
            } else if (std::isfinite(slope)) {
                res[offsetRes] = y0 + slope * (xv - x0);
            } else {
                T t = (xv - x0) / dx;
                res[offsetRes] = (static_cast<T>(1) - t) * y0 + t * y1;
            }
        }

        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetX += stridesX[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetX -= (shape[d] - 1) * stridesX[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

void v_interp_double(const double *x, int64_t x_size,
                     const double *xp, int64_t xp_size,
                     const double *fp, double *res,
                     const double *left, const double *right) {
    fast_interp_vector<double>(x, x_size, xp, xp_size, fp, res, left, right);
}

void native_interp_double(const double *x, int64_t x_size,
                          const double *xp, int64_t xp_size,
                          const double *fp, double *res,
                          const double *left, const double *right) {
    fast_interp_vector<double>(x, x_size, xp, xp_size, fp, res, left, right);
}

void s_interp_double(const double *x, const int64_t *stridesX,
                     const double *xp, int64_t strideXP, int64_t xp_size,
                     const double *fp, int64_t strideFP,
                     double *res, const int64_t *stridesRes,
                     const int64_t *shape, int rank,
                     const double *left, const double *right) {
    fast_interp_strided<double>(x, stridesX, xp, strideXP, xp_size, fp, strideFP, res, stridesRes, shape, rank, left, right);
}

void v_interp_float(const float *x, int64_t x_size,
                    const float *xp, int64_t xp_size,
                    const float *fp, float *res,
                    const float *left, const float *right) {
    fast_interp_vector<float>(x, x_size, xp, xp_size, fp, res, left, right);
}

void native_interp_float(const float *x, int64_t x_size,
                         const float *xp, int64_t xp_size,
                         const float *fp, float *res,
                         const float *left, const float *right) {
    fast_interp_vector<float>(x, x_size, xp, xp_size, fp, res, left, right);
}

void s_interp_float(const float *x, const int64_t *stridesX,
                    const float *xp, int64_t strideXP, int64_t xp_size,
                    const float *fp, int64_t strideFP,
                    float *res, const int64_t *stridesRes,
                    const int64_t *shape, int rank,
                    const float *left, const float *right) {
    fast_interp_strided<float>(x, stridesX, xp, strideXP, xp_size, fp, strideFP, res, stridesRes, shape, rank, left, right);
}

// ============================================================================
// SECTION 12: COMPARISON & EQUALITY KERNELS
// ============================================================================

// C++ Operators for cpx_t and cpx_f_t to allow templated comparisons
inline bool operator==(const cpx_t& x, const cpx_t& y) { return x.r == y.r && x.i == y.i; }
inline bool operator!=(const cpx_t& x, const cpx_t& y) { return !(x == y); }
inline bool operator==(const cpx_f_t& x, const cpx_f_t& y) { return x.r == y.r && x.i == y.i; }
inline bool operator!=(const cpx_f_t& x, const cpx_f_t& y) { return !(x == y); }
inline bool operator==(const cpx_t& x, const cpx_f_t& y) { return x.r == y.r && x.i == y.i; }
inline bool operator==(const cpx_f_t& x, const cpx_t& y) { return x.r == y.r && x.i == y.i; }
inline bool operator!=(const cpx_t& x, const cpx_f_t& y) { return !(x == y); }
inline bool operator!=(const cpx_f_t& x, const cpx_t& y) { return !(x == y); }

template<typename T> inline bool operator==(const cpx_t& x, const T& y) { return x.r == y && x.i == 0; }
template<typename T> inline bool operator==(const T& x, const cpx_t& y) { return y == x; }
template<typename T> inline bool operator!=(const cpx_t& x, const T& y) { return !(x == y); }
template<typename T> inline bool operator!=(const T& x, const cpx_t& y) { return !(x == y); }

template<typename T> inline bool operator==(const cpx_f_t& x, const T& y) { return x.r == y && x.i == 0; }
template<typename T> inline bool operator==(const T& x, const cpx_f_t& y) { return y == x; }
template<typename T> inline bool operator!=(const cpx_f_t& x, const T& y) { return !(x == y); }
template<typename T> inline bool operator!=(const T& x, const cpx_f_t& y) { return !(x == y); }

template <typename T1, typename T2,
          bool IsInt1 = std::is_integral<T1>::value,
          bool IsInt2 = std::is_integral<T2>::value,
          bool IsSigned1 = std::is_signed<T1>::value,
          bool IsSigned2 = std::is_signed<T2>::value,
          bool IsCpx1 = std::is_same<T1, cpx_t>::value || std::is_same<T1, cpx_f_t>::value,
          bool IsCpx2 = std::is_same<T2, cpx_t>::value || std::is_same<T2, cpx_f_t>::value>
struct safe_cmp_helper {
    static inline bool eq(const T1& x, const T2& y) {
        return static_cast<double>(x) == static_cast<double>(y);
    }
    static inline bool lt(const T1& x, const T2& y) {
        return static_cast<double>(x) < static_cast<double>(y);
    }
};

template <typename T1, typename T2, bool IsInt1, bool IsInt2, bool IsSigned1, bool IsSigned2>
struct safe_cmp_helper<T1, T2, IsInt1, IsInt2, IsSigned1, IsSigned2, true, true> {
    static inline bool eq(const T1& x, const T2& y) {
        cpx_t cx = caster<cpx_t, T1>::cast(x);
        cpx_t cy = caster<cpx_t, T2>::cast(y);
        return cx.r == cy.r && cx.i == cy.i;
    }
    static inline bool lt(const T1& x, const T2& y) { return false; }
};

template <typename T1, typename T2, bool IsInt1, bool IsInt2, bool IsSigned1, bool IsSigned2>
struct safe_cmp_helper<T1, T2, IsInt1, IsInt2, IsSigned1, IsSigned2, true, false> {
    static inline bool eq(const T1& x, const T2& y) {
        cpx_t cx = caster<cpx_t, T1>::cast(x);
        return cx.r == static_cast<double>(y) && cx.i == 0.0;
    }
    static inline bool lt(const T1& x, const T2& y) { return false; }
};

template <typename T1, typename T2, bool IsInt1, bool IsInt2, bool IsSigned1, bool IsSigned2>
struct safe_cmp_helper<T1, T2, IsInt1, IsInt2, IsSigned1, IsSigned2, false, true> {
    static inline bool eq(const T1& x, const T2& y) {
        cpx_t cy = caster<cpx_t, T2>::cast(y);
        return static_cast<double>(x) == cy.r && cy.i == 0.0;
    }
    static inline bool lt(const T1& x, const T2& y) { return false; }
};

template <typename T1, typename T2, bool IsSigned>
struct safe_cmp_helper<T1, T2, true, true, IsSigned, IsSigned, false, false> {
    static inline bool eq(const T1& x, const T2& y) { return x == y; }
    static inline bool lt(const T1& x, const T2& y) { return x < y; }
};

template <typename T1, typename T2>
struct safe_cmp_helper<T1, T2, true, true, true, false, false, false> {
    static inline bool eq(const T1& x, const T2& y) {
        if (x < 0) return false;
        return static_cast<uint64_t>(x) == static_cast<uint64_t>(y);
    }
    static inline bool lt(const T1& x, const T2& y) {
        if (x < 0) return true;
        return static_cast<uint64_t>(x) < static_cast<uint64_t>(y);
    }
};

template <typename T1, typename T2>
struct safe_cmp_helper<T1, T2, true, true, false, true, false, false> {
    static inline bool eq(const T1& x, const T2& y) {
        if (y < 0) return false;
        return static_cast<uint64_t>(x) == static_cast<uint64_t>(y);
    }
    static inline bool lt(const T1& x, const T2& y) {
        if (y < 0) return false;
        return static_cast<uint64_t>(x) < static_cast<uint64_t>(y);
    }
};

struct OpEq {
    template<typename T1, typename T2>
    static inline bool apply(const T1& x, const T2& y) { return safe_cmp_helper<T1, T2>::eq(x, y); }
};
struct OpNe {
    template<typename T1, typename T2>
    static inline bool apply(const T1& x, const T2& y) { return !safe_cmp_helper<T1, T2>::eq(x, y); }
};
struct OpLt {
    template<typename T1, typename T2>
    static inline bool apply(const T1& x, const T2& y) { return safe_cmp_helper<T1, T2>::lt(x, y); }
};
struct OpLe {
    template<typename T1, typename T2>
    static inline bool apply(const T1& x, const T2& y) { return safe_cmp_helper<T1, T2>::lt(x, y) || safe_cmp_helper<T1, T2>::eq(x, y); }
};
struct OpGt {
    template<typename T1, typename T2>
    static inline bool apply(const T1& x, const T2& y) { return safe_cmp_helper<T2, T1>::lt(y, x); }
};
struct OpGe {
    template<typename T1, typename T2>
    static inline bool apply(const T1& x, const T2& y) { return safe_cmp_helper<T2, T1>::lt(y, x) || safe_cmp_helper<T1, T2>::eq(x, y); }
};

template <typename T1, typename T2, typename Op>
void s_compare_impl(const T1 *a, const int64_t *stridesA,
                    const T2 *b, const int64_t *stridesB,
                    uint8_t *res, const int64_t *stridesRes,
                    const int64_t *shape, int rank) {
    if (a == nullptr || b == nullptr || res == nullptr || rank < 0) return;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (total_elements <= 0) return;

    bool overlap_a = strided_buffers_overlap_not_identical(
        a, stridesA, sizeof(T1), res, stridesRes, sizeof(uint8_t), shape, rank);
    bool overlap_b = strided_buffers_overlap_not_identical(
        b, stridesB, sizeof(T2), res, stridesRes, sizeof(uint8_t), shape, rank);
    if (overlap_a || overlap_b) {
        NoThrowBuffer<uint8_t> temp(total_elements);
        if (!temp.ok()) {
            ndarray_set_oom_flag();
            return;
        }
        DECLARE_RANK_BUFFER(int64_t, temp_strides, rank);
        int64_t s = 1;
        for (int64_t i = rank - 1; i >= 0; i--) {
            temp_strides[i] = s;
            s *= shape[i];
        }
        s_compare_impl<T1, T2, Op>(a, stridesA, b, stridesB, temp.data(), temp_strides, shape, rank);
        DECLARE_RANK_BUFFER(int64_t, coord_out, rank);
        int64_t offsetRes = 0;
        for (int64_t el = 0; el < total_elements; el++) {
            res[offsetRes] = temp[el];
            for (int d = rank - 1; d >= 0; d--) {
                coord_out[d]++;
                if (coord_out[d] < shape[d]) {
                    offsetRes += stridesRes[d];
                    break;
                }
                coord_out[d] = 0;
                offsetRes -= (shape[d] - 1) * stridesRes[d];
            }
        }
        return;
    }

    int is_a_contig = 1, is_b_contig = 1, is_res_contig = 1;
    int64_t expected_stride = 1;
    for (int64_t i = rank - 1; i >= 0; i--) {
        if (stridesA[i] != expected_stride) is_a_contig = 0;
        if (stridesB[i] != expected_stride) is_b_contig = 0;
        if (stridesRes[i] != expected_stride) is_res_contig = 0;
        expected_stride *= shape[i];
    }
    int is_a_scal = 1, is_b_scal = 1;
    for (int64_t i = 0; i < rank; i++) {
        if (stridesA[i] != 0) is_a_scal = 0;
        if (stridesB[i] != 0) is_b_scal = 0;
    }
    if (is_res_contig) {
        if (is_a_contig && is_b_contig) {
            for (int64_t i = 0; i < total_elements; i++) {
                res[i] = (Op::apply(a[i], b[i])) ? 1 : 0;
            }
            return;
        }
        if (is_a_contig && is_b_scal) {
            T2 val = b[0];
            for (int64_t i = 0; i < total_elements; i++) {
                res[i] = (Op::apply(a[i], val)) ? 1 : 0;
            }
            return;
        }
        if (is_a_scal && is_b_contig) {
            T1 val = a[0];
            for (int64_t i = 0; i < total_elements; i++) {
                res[i] = (Op::apply(val, b[i])) ? 1 : 0;
            }
            return;
        }
    }
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetA = 0, offsetB = 0, offsetRes = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        res[offsetRes] = (Op::apply(a[offsetA], b[offsetB])) ? 1 : 0;
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetA += stridesA[d];
                offsetB += stridesB[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetA -= (shape[d] - 1) * stridesA[d];
            offsetB -= (shape[d] - 1) * stridesB[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

template <typename T1, typename Op>
void dispatch_compare_B_real(const T1 *a, const int64_t *stridesA,
                             int dtypeB, const void *b, const int64_t *stridesB,
                             uint8_t *res, const int64_t *stridesRes,
                             const int64_t *shape, int rank) {
    switch (dtypeB) {
        case DTYPE_FLOAT64: s_compare_impl<T1, double, Op>(a, stridesA, (const double*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_FLOAT32: s_compare_impl<T1, float, Op>(a, stridesA, (const float*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_FLOAT16: s_compare_impl<T1, float16_t, Op>(a, stridesA, (const float16_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_BFLOAT16: s_compare_impl<T1, bfloat16_t, Op>(a, stridesA, (const bfloat16_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_INT32: s_compare_impl<T1, int32_t, Op>(a, stridesA, (const int32_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_INT64: s_compare_impl<T1, int64_t, Op>(a, stridesA, (const int64_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_UINT8: s_compare_impl<T1, uint8_t, Op>(a, stridesA, (const uint8_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_INT16: s_compare_impl<T1, int16_t, Op>(a, stridesA, (const int16_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_INT8: s_compare_impl<T1, int8_t, Op>(a, stridesA, (const int8_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_UINT16: s_compare_impl<T1, uint16_t, Op>(a, stridesA, (const uint16_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_UINT32: s_compare_impl<T1, uint32_t, Op>(a, stridesA, (const uint32_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_UINT64: s_compare_impl<T1, uint64_t, Op>(a, stridesA, (const uint64_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_BOOLEAN: s_compare_impl<T1, uint8_t, Op>(a, stridesA, (const uint8_t*)b, stridesB, res, stridesRes, shape, rank); break;
        default: return;
    }
}

template <typename T1, typename Op>
void dispatch_compare_B_all(const T1 *a, const int64_t *stridesA,
                            int dtypeB, const void *b, const int64_t *stridesB,
                            uint8_t *res, const int64_t *stridesRes,
                            const int64_t *shape, int rank) {
    switch (dtypeB) {
        case DTYPE_FLOAT64: s_compare_impl<T1, double, Op>(a, stridesA, (const double*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_FLOAT32: s_compare_impl<T1, float, Op>(a, stridesA, (const float*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_FLOAT16: s_compare_impl<T1, float16_t, Op>(a, stridesA, (const float16_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_BFLOAT16: s_compare_impl<T1, bfloat16_t, Op>(a, stridesA, (const bfloat16_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_INT32: s_compare_impl<T1, int32_t, Op>(a, stridesA, (const int32_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_INT64: s_compare_impl<T1, int64_t, Op>(a, stridesA, (const int64_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_UINT8: s_compare_impl<T1, uint8_t, Op>(a, stridesA, (const uint8_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_INT16: s_compare_impl<T1, int16_t, Op>(a, stridesA, (const int16_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_INT8: s_compare_impl<T1, int8_t, Op>(a, stridesA, (const int8_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_UINT16: s_compare_impl<T1, uint16_t, Op>(a, stridesA, (const uint16_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_UINT32: s_compare_impl<T1, uint32_t, Op>(a, stridesA, (const uint32_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_UINT64: s_compare_impl<T1, uint64_t, Op>(a, stridesA, (const uint64_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_BOOLEAN: s_compare_impl<T1, uint8_t, Op>(a, stridesA, (const uint8_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_COMPLEX128: s_compare_impl<T1, cpx_t, Op>(a, stridesA, (const cpx_t*)b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_COMPLEX64: s_compare_impl<T1, cpx_f_t, Op>(a, stridesA, (const cpx_f_t*)b, stridesB, res, stridesRes, shape, rank); break;
        default: return;
    }
}

template <typename Op>
void dispatch_compare_A_real(int dtypeA, const void *a, const int64_t *stridesA,
                             int dtypeB, const void *b, const int64_t *stridesB,
                             uint8_t *res, const int64_t *stridesRes,
                             const int64_t *shape, int rank) {
    switch (dtypeA) {
        case DTYPE_FLOAT64: dispatch_compare_B_real<double, Op>((const double*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_FLOAT32: dispatch_compare_B_real<float, Op>((const float*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_FLOAT16: dispatch_compare_B_real<float16_t, Op>((const float16_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_BFLOAT16: dispatch_compare_B_real<bfloat16_t, Op>((const bfloat16_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_INT32: dispatch_compare_B_real<int32_t, Op>((const int32_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_INT64: dispatch_compare_B_real<int64_t, Op>((const int64_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_UINT8: dispatch_compare_B_real<uint8_t, Op>((const uint8_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_INT16: dispatch_compare_B_real<int16_t, Op>((const int16_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_INT8: dispatch_compare_B_real<int8_t, Op>((const int8_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_UINT16: dispatch_compare_B_real<uint16_t, Op>((const uint16_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_UINT32: dispatch_compare_B_real<uint32_t, Op>((const uint32_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_UINT64: dispatch_compare_B_real<uint64_t, Op>((const uint64_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_BOOLEAN: dispatch_compare_B_real<uint8_t, Op>((const uint8_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        default: return;
    }
}

template <typename Op>
void dispatch_compare_A_all(int dtypeA, const void *a, const int64_t *stridesA,
                            int dtypeB, const void *b, const int64_t *stridesB,
                            uint8_t *res, const int64_t *stridesRes,
                            const int64_t *shape, int rank) {
    switch (dtypeA) {
        case DTYPE_FLOAT64: dispatch_compare_B_all<double, Op>((const double*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_FLOAT32: dispatch_compare_B_all<float, Op>((const float*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_FLOAT16: dispatch_compare_B_all<float16_t, Op>((const float16_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_BFLOAT16: dispatch_compare_B_all<bfloat16_t, Op>((const bfloat16_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_INT32: dispatch_compare_B_all<int32_t, Op>((const int32_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_INT64: dispatch_compare_B_all<int64_t, Op>((const int64_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_UINT8: dispatch_compare_B_all<uint8_t, Op>((const uint8_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_INT16: dispatch_compare_B_all<int16_t, Op>((const int16_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_INT8: dispatch_compare_B_all<int8_t, Op>((const int8_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_UINT16: dispatch_compare_B_all<uint16_t, Op>((const uint16_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_UINT32: dispatch_compare_B_all<uint32_t, Op>((const uint32_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_UINT64: dispatch_compare_B_all<uint64_t, Op>((const uint64_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_BOOLEAN: dispatch_compare_B_all<uint8_t, Op>((const uint8_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_COMPLEX128: dispatch_compare_B_all<cpx_t, Op>((const cpx_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case DTYPE_COMPLEX64: dispatch_compare_B_all<cpx_f_t, Op>((const cpx_f_t*)a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        default: return;
    }
}

extern "C" void ndarray_compare(
    int op, int dtypeA, int dtypeB,
    const void *a, const int64_t *stridesA,
    const void *b, const int64_t *stridesB,
    uint8_t *res, const int64_t *stridesRes,
    const int64_t *shape, int rank
) {
    ComparisonOp cop = static_cast<ComparisonOp>(op);
    switch (cop) {
        case ComparisonOp::EQ: dispatch_compare_A_all<OpEq>(dtypeA, a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case ComparisonOp::NE: dispatch_compare_A_all<OpNe>(dtypeA, a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case ComparisonOp::LT: dispatch_compare_A_real<OpLt>(dtypeA, a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case ComparisonOp::LE: dispatch_compare_A_real<OpLe>(dtypeA, a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case ComparisonOp::GT: dispatch_compare_A_real<OpGt>(dtypeA, a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        case ComparisonOp::GE: dispatch_compare_A_real<OpGe>(dtypeA, a, stridesA, dtypeB, b, stridesB, res, stridesRes, shape, rank); break;
        default: return;
    }
}

#define EXPR_EQ(x, y) ((x) == (y))
#define EXPR_BOOL_EQ(x, y) (((x) != 0) == ((y) != 0))

// Structural Equality
#define DEFINE_EQUALS_FUNC(NAME, TYPE, EXPR) \
static int NAME(const TYPE *a, const int64_t *stridesA, \
         const TYPE *b, const int64_t *stridesB, \
         const int64_t *shape, int rank, [[maybe_unused]] const uint8_t *mask) { \
    if (a == nullptr || b == nullptr || rank < 0) return 0; \
    if (rank == 0) return EXPR(a[0], b[0]) ? 1 : 0; \
    int64_t total_elements = 1; \
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i]; \
    if (total_elements == 0) return 1; \
    int is_a_contig = 1, is_b_contig = 1; \
    int64_t expected_stride = 1; \
    for (int64_t i = rank - 1; i >= 0; i--) { \
        if (shape[i] > 1 && stridesA[i] != expected_stride) is_a_contig = 0; \
        if (shape[i] > 1 && stridesB[i] != expected_stride) is_b_contig = 0; \
        expected_stride *= shape[i]; \
    } \
    if (is_a_contig && is_b_contig) { \
        for (int64_t el = 0; el < total_elements; el++) { \
            if (!(EXPR(a[el], b[el]))) return 0; \
        } \
        return 1; \
    } \
    DECLARE_RANK_BUFFER_RET(int64_t, coord, rank, 0); \
    int64_t offsetA = 0, offsetB = 0; \
    for (int64_t el = 0; el < total_elements; el++) { \
        if (!(EXPR(a[offsetA], b[offsetB]))) return 0; \
        for (int d = rank - 1; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetA += stridesA[d]; \
                offsetB += stridesB[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetA -= (shape[d] - 1) * stridesA[d]; \
            offsetB -= (shape[d] - 1) * stridesB[d]; \
        } \
    } \
    return 1; \
}

DEFINE_EQUALS_FUNC(s_equals_double, double, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_float, float, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_float16, float16_t, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_bfloat16, bfloat16_t, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_int64, int64_t, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_int32, int32_t, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_int16, int16_t, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_int8, int8_t, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_uint64, uint64_t, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_uint32, uint32_t, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_uint16, uint16_t, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_uint8, uint8_t, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_complex128, cpx_t, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_complex64, cpx_f_t, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_boolean, uint8_t, EXPR_BOOL_EQ)

#undef DEFINE_EQUALS_FUNC

int ndarray_equals(
    int dtype,
    const void *a, const int64_t *stridesA,
    const void *b, const int64_t *stridesB,
    const int64_t *shape, int rank
) {
    switch (dtype) {
        case DTYPE_FLOAT64: return s_equals_double((const double*)a, stridesA, (const double*)b, stridesB, shape, rank, nullptr);
        case DTYPE_FLOAT32: return s_equals_float((const float*)a, stridesA, (const float*)b, stridesB, shape, rank, nullptr);
        case DTYPE_FLOAT16: return s_equals_float16((const float16_t*)a, stridesA, (const float16_t*)b, stridesB, shape, rank, nullptr);
        case DTYPE_BFLOAT16: return s_equals_bfloat16((const bfloat16_t*)a, stridesA, (const bfloat16_t*)b, stridesB, shape, rank, nullptr);
        case DTYPE_INT64: return s_equals_int64((const int64_t*)a, stridesA, (const int64_t*)b, stridesB, shape, rank, nullptr);
        case DTYPE_INT32: return s_equals_int32((const int32_t*)a, stridesA, (const int32_t*)b, stridesB, shape, rank, nullptr);
        case DTYPE_INT16: return s_equals_int16((const int16_t*)a, stridesA, (const int16_t*)b, stridesB, shape, rank, nullptr);
        case DTYPE_INT8: return s_equals_int8((const int8_t*)a, stridesA, (const int8_t*)b, stridesB, shape, rank, nullptr);
        case DTYPE_UINT64: return s_equals_uint64((const uint64_t*)a, stridesA, (const uint64_t*)b, stridesB, shape, rank, nullptr);
        case DTYPE_UINT32: return s_equals_uint32((const uint32_t*)a, stridesA, (const uint32_t*)b, stridesB, shape, rank, nullptr);
        case DTYPE_UINT16: return s_equals_uint16((const uint16_t*)a, stridesA, (const uint16_t*)b, stridesB, shape, rank, nullptr);
        case DTYPE_UINT8: return s_equals_uint8((const uint8_t*)a, stridesA, (const uint8_t*)b, stridesB, shape, rank, nullptr);
        case DTYPE_COMPLEX128: return s_equals_complex128((const cpx_t*)a, stridesA, (const cpx_t*)b, stridesB, shape, rank, nullptr);
        case DTYPE_COMPLEX64: return s_equals_complex64((const cpx_f_t*)a, stridesA, (const cpx_f_t*)b, stridesB, shape, rank, nullptr);
        case DTYPE_BOOLEAN: return s_equals_boolean((const uint8_t*)a, stridesA, (const uint8_t*)b, stridesB, shape, rank, nullptr);
        default: return 0;
    }
}

// Reduction Min/Max

#define DEFINE_R_MINMAX(NAME, TYPE, OP) \
TYPE r_##NAME##_##TYPE(const TYPE *src, int64_t size) { \
    if (src == nullptr || size <= 0) return (TYPE)0; \
    TYPE acc0 = src[0]; \
    if (is_nan_check(acc0)) return acc0; \
    TYPE acc1 = acc0, acc2 = acc0, acc3 = acc0; \
    int64_t i = 1; \
    for (; i + 3 < size; i += 4) { \
        TYPE v0 = src[i], v1 = src[i + 1], v2 = src[i + 2], v3 = src[i + 3]; \
        if (is_nan_check(v0)) return v0; \
        if (is_nan_check(v1)) return v1; \
        if (is_nan_check(v2)) return v2; \
        if (is_nan_check(v3)) return v3; \
        if (v0 OP acc0) acc0 = v0; \
        if (v1 OP acc1) acc1 = v1; \
        if (v2 OP acc2) acc2 = v2; \
        if (v3 OP acc3) acc3 = v3; \
    } \
    if (acc1 OP acc0) acc0 = acc1; \
    if (acc2 OP acc0) acc0 = acc2; \
    if (acc3 OP acc0) acc0 = acc3; \
    for (; i < size; i++) { \
        if (is_nan_check(src[i])) return src[i]; \
        if (src[i] OP acc0) acc0 = src[i]; \
    } \
    return acc0; \
}

DEFINE_R_MINMAX(min, double, <)
DEFINE_R_MINMAX(min, float, <)
DEFINE_R_MINMAX(min, int64_t, <)
DEFINE_R_MINMAX(min, int32_t, <)
DEFINE_R_MINMAX(min, uint8_t, <)
DEFINE_R_MINMAX(min, int16_t, <)

DEFINE_R_MINMAX(max, double, >)
DEFINE_R_MINMAX(max, float, >)
DEFINE_R_MINMAX(max, int64_t, >)
DEFINE_R_MINMAX(max, int32_t, >)
DEFINE_R_MINMAX(max, uint8_t, >)
DEFINE_R_MINMAX(max, int16_t, >)

#undef DEFINE_R_MINMAX

// Strided Reduction Min/Max

#define DEFINE_S_MINMAX(NAME, TYPE, OP) \
void s_##NAME##_##TYPE(const TYPE *src, const int64_t *stridesSrc, \
                       TYPE *dest, const int64_t *stridesDest, \
                       const int64_t *shape, int rank, int axis) { \
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return; \
    int64_t size_axis = shape[axis]; \
    if (size_axis <= 0) return; \
    DECLARE_RANK_BUFFER(int64_t, coord, rank); \
    int64_t outer_size = 1; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) outer_size *= shape[d]; \
    } \
    for (int64_t o = 0; o < outer_size; o++) { \
        int64_t offsetRes = 0; \
        int64_t offsetSrc = 0; \
        for (int d = 0; d < rank; d++) { \
            if (d != axis) { \
                offsetSrc += coord[d] * stridesSrc[d]; \
                if (rank > 1) { \
                    int targetD = (d < axis) ? d : (d - 1); \
                    offsetRes += coord[d] * stridesDest[targetD]; \
                } \
            } \
        } \
        TYPE val_acc = src[offsetSrc];         int64_t stride_axis = stridesSrc[axis];         for (int64_t i = 1; i < size_axis; i++) {             if (is_nan_check(val_acc)) break;             TYPE val = src[offsetSrc + i * stride_axis];             if (is_nan_check(val)) { val_acc = val; break; }             if (val OP val_acc) val_acc = val;         }         dest[offsetRes] = val_acc;         for (int d = rank - 1; d >= 0; d--) { \
            if (d == axis) continue; \
            coord[d]++; \
            if (coord[d] < shape[d]) break; \
            coord[d] = 0; \
        } \
    } \
}

DEFINE_S_MINMAX(min, double, <)
DEFINE_S_MINMAX(min, float, <)
DEFINE_S_MINMAX(min, int64_t, <)
DEFINE_S_MINMAX(min, int32_t, <)
DEFINE_S_MINMAX(min, uint8_t, <)
DEFINE_S_MINMAX(min, int16_t, <)

DEFINE_S_MINMAX(max, double, >)
DEFINE_S_MINMAX(max, float, >)
DEFINE_S_MINMAX(max, int64_t, >)
DEFINE_S_MINMAX(max, int32_t, >)
DEFINE_S_MINMAX(max, uint8_t, >)
DEFINE_S_MINMAX(max, int16_t, >)

#undef DEFINE_S_MINMAX

// Nanmin/Nanmax Reduction (float/double only)

#define DEFINE_R_NANMINMAX(NAME, TYPE, OP) \
TYPE r_##NAME##_##TYPE(const TYPE *src, int64_t size) { \
    if (src == nullptr || size <= 0) return (TYPE)NAN; \
    TYPE acc = (TYPE)NAN; \
    int64_t i = 0; \
    for (; i < size; i++) { \
        if (!std::isnan(src[i])) { \
            acc = src[i]; \
            i++; \
            break; \
        } \
    } \
    if (std::isnan(acc)) return (TYPE)NAN; \
    TYPE acc0 = acc, acc1 = acc, acc2 = acc, acc3 = acc; \
    for (; i + 3 < size; i += 4) { \
        TYPE v0 = src[i], v1 = src[i + 1], v2 = src[i + 2], v3 = src[i + 3]; \
        if (v0 OP acc0) acc0 = v0; \
        if (v1 OP acc1) acc1 = v1; \
        if (v2 OP acc2) acc2 = v2; \
        if (v3 OP acc3) acc3 = v3; \
    } \
    if (acc1 OP acc0) acc0 = acc1; \
    if (acc2 OP acc0) acc0 = acc2; \
    if (acc3 OP acc0) acc0 = acc3; \
    for (; i < size; i++) { \
        if (src[i] OP acc0) { \
            acc0 = src[i]; \
        } \
    } \
    return acc0; \
}

DEFINE_R_NANMINMAX(nanmin, double, <)
DEFINE_R_NANMINMAX(nanmin, float, <)
DEFINE_R_NANMINMAX(nanmax, double, >)
DEFINE_R_NANMINMAX(nanmax, float, >)

#undef DEFINE_R_NANMINMAX

// Strided Nanmin/Nanmax Reduction (float/double only)

#define DEFINE_S_NANMINMAX(NAME, TYPE, OP) \
void s_##NAME##_##TYPE(const TYPE *src, const int64_t *stridesSrc, \
                       TYPE *dest, const int64_t *stridesDest, \
                       const int64_t *shape, int rank, int axis) { \
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return; \
    int64_t size_axis = shape[axis]; \
    if (size_axis <= 0) return; \
    DECLARE_RANK_BUFFER(int64_t, coord, rank); \
    int64_t outer_size = 1; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) outer_size *= shape[d]; \
    } \
    for (int64_t o = 0; o < outer_size; o++) { \
        int64_t offsetRes = 0; \
        int64_t offsetSrc = 0; \
        for (int d = 0; d < rank; d++) { \
            if (d != axis) { \
                offsetSrc += coord[d] * stridesSrc[d]; \
                if (rank > 1) { \
                    int targetD = (d < axis) ? d : (d - 1); \
                    offsetRes += coord[d] * stridesDest[targetD]; \
                } \
            } \
        } \
        TYPE val_acc = (TYPE)NAN; \
        int64_t stride_axis = stridesSrc[axis]; \
        int64_t i = 0; \
        for (; i < size_axis; i++) { \
            TYPE val = src[offsetSrc + i * stride_axis]; \
            if (!std::isnan(val)) { \
                val_acc = val; \
                break; \
            } \
        } \
        if (!std::isnan(val_acc)) { \
            for (; i < size_axis; i++) { \
                TYPE val = src[offsetSrc + i * stride_axis]; \
                if (!std::isnan(val) && val OP val_acc) { \
                    val_acc = val; \
                } \
            } \
        } \
        dest[offsetRes] = val_acc; \
        for (int d = rank - 1; d >= 0; d--) { \
            if (d == axis) continue; \
            coord[d]++; \
            if (coord[d] < shape[d]) break; \
            coord[d] = 0; \
        } \
    } \
}

DEFINE_S_NANMINMAX(nanmin, double, <)
DEFINE_S_NANMINMAX(nanmin, float, <)
DEFINE_S_NANMINMAX(nanmax, double, >)
DEFINE_S_NANMINMAX(nanmax, float, >)

#undef DEFINE_S_NANMINMAX

// Find Index
#define DEFINE_FIND_INDEX_FUNC(NAME, TYPE) \
static int s_find_index_##NAME(const TYPE *a, const int64_t *stridesA, \
                        const int64_t *shape, int rank, \
                        int op, TYPE target, \
                        const int64_t *startCoords, const int64_t *directions, \
                        int64_t *matchCoords) { \
    if (a == nullptr || rank < 0) return 0; \
    int64_t total_elements = 1; \
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i]; \
    if (total_elements <= 0) return 0; \
    ComparisonOp cop = static_cast<ComparisonOp>(op); \
    if (rank == 0) { \
        TYPE val = a[0]; \
        int match = 0; \
        switch(cop) { \
            case ComparisonOp::EQ: match = (val == target); break; \
            case ComparisonOp::NE: match = (val != target); break; \
            case ComparisonOp::LT: match = (val < target); break; \
            case ComparisonOp::LE: match = (val <= target); break; \
            case ComparisonOp::GT: match = (val > target); break; \
            case ComparisonOp::GE: match = (val >= target); break; \
        } \
        return match ? 1 : 0; \
    } \
    DECLARE_RANK_BUFFER_RET(int64_t, coord, rank, 0); \
    for (int64_t i = 0; i < rank; i++) { \
        if (startCoords[i] < 0 || startCoords[i] >= shape[i]) return 0; \
        coord[i] = startCoords[i]; \
    } \
    int64_t offsetA = 0; \
    for (int64_t i = 0; i < rank; i++) { \
        offsetA += coord[i] * stridesA[i]; \
    } \
    while (1) { \
        TYPE val = a[offsetA]; \
        int match = 0; \
        switch(cop) { \
            case ComparisonOp::EQ: match = (val == target); break; \
            case ComparisonOp::NE: match = (val != target); break; \
            case ComparisonOp::LT: match = (val < target); break; \
            case ComparisonOp::LE: match = (val <= target); break; \
            case ComparisonOp::GT: match = (val > target); break; \
            case ComparisonOp::GE: match = (val >= target); break; \
        } \
        if (match) { \
            if (matchCoords != nullptr) { \
                for (int64_t i = 0; i < rank; i++) { \
                    matchCoords[i] = coord[i]; \
                } \
            } \
            return 1; \
        } \
        int finished = 0; \
        for (int d = rank - 1; d >= 0; d--) { \
            if (directions[d] >= 0) { \
                coord[d]++; \
                if (coord[d] < shape[d]) { \
                    offsetA += stridesA[d]; \
                    break; \
                } \
                if (d == 0) { \
                    finished = 1; \
                    break; \
                } \
                coord[d] = 0; \
                offsetA -= (shape[d] - 1) * stridesA[d]; \
            } else { \
                coord[d]--; \
                if (coord[d] >= 0) { \
                    offsetA -= stridesA[d]; \
                    break; \
                } \
                if (d == 0) { \
                    finished = 1; \
                    break; \
                } \
                coord[d] = shape[d] - 1; \
                offsetA += (shape[d] - 1) * stridesA[d]; \
            } \
        } \
        if (finished) break; \
    } \
    return 0; \
}

DEFINE_FIND_INDEX_FUNC(double, double)
DEFINE_FIND_INDEX_FUNC(float, float)
DEFINE_FIND_INDEX_FUNC(int32, int32_t)
DEFINE_FIND_INDEX_FUNC(int64, int64_t)
DEFINE_FIND_INDEX_FUNC(uint8, uint8_t)
DEFINE_FIND_INDEX_FUNC(int16, int16_t)
DEFINE_FIND_INDEX_FUNC(int8, int8_t)
DEFINE_FIND_INDEX_FUNC(uint16, uint16_t)
DEFINE_FIND_INDEX_FUNC(uint32, uint32_t)
DEFINE_FIND_INDEX_FUNC(uint64, uint64_t)

#undef DEFINE_FIND_INDEX_FUNC

#define DEFINE_FIND_INDEX_CONVERT_FUNC(NAME, SRC_TYPE, TARGET_TYPE, CONVERT_FN) \
static int s_find_index_##NAME(const SRC_TYPE *a, const int64_t *stridesA, \
                        const int64_t *shape, int rank, \
                        int op, TARGET_TYPE target, \
                        const int64_t *startCoords, const int64_t *directions, \
                        int64_t *matchCoords) { \
    if (a == nullptr || rank < 0) return 0; \
    int64_t total_elements = 1; \
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i]; \
    if (total_elements <= 0) return 0; \
    ComparisonOp cop = static_cast<ComparisonOp>(op); \
    if (rank == 0) { \
        TARGET_TYPE val = static_cast<TARGET_TYPE>(CONVERT_FN(a[0])); \
        int match = 0; \
        switch(cop) { \
            case ComparisonOp::EQ: match = (val == target); break; \
            case ComparisonOp::NE: match = (val != target); break; \
            case ComparisonOp::LT: match = (val < target); break; \
            case ComparisonOp::LE: match = (val <= target); break; \
            case ComparisonOp::GT: match = (val > target); break; \
            case ComparisonOp::GE: match = (val >= target); break; \
        } \
        return match ? 1 : 0; \
    } \
    DECLARE_RANK_BUFFER_RET(int64_t, coord, rank, 0); \
    for (int64_t i = 0; i < rank; i++) { \
        if (startCoords[i] < 0 || startCoords[i] >= shape[i]) return 0; \
        coord[i] = startCoords[i]; \
    } \
    int64_t offsetA = 0; \
    for (int64_t i = 0; i < rank; i++) { \
        offsetA += coord[i] * stridesA[i]; \
    } \
    while (1) { \
        TARGET_TYPE val = static_cast<TARGET_TYPE>(CONVERT_FN(a[offsetA])); \
        int match = 0; \
        switch(cop) { \
            case ComparisonOp::EQ: match = (val == target); break; \
            case ComparisonOp::NE: match = (val != target); break; \
            case ComparisonOp::LT: match = (val < target); break; \
            case ComparisonOp::LE: match = (val <= target); break; \
            case ComparisonOp::GT: match = (val > target); break; \
            case ComparisonOp::GE: match = (val >= target); break; \
        } \
        if (match) { \
            if (matchCoords != nullptr) { \
                for (int64_t i = 0; i < rank; i++) { \
                    matchCoords[i] = coord[i]; \
                } \
            } \
            return 1; \
        } \
        int finished = 0; \
        for (int d = rank - 1; d >= 0; d--) { \
            if (directions[d] >= 0) { \
                coord[d]++; \
                if (coord[d] < shape[d]) { \
                    offsetA += stridesA[d]; \
                    break; \
                } \
                if (d == 0) { \
                    finished = 1; \
                    break; \
                } \
                coord[d] = 0; \
                offsetA -= (shape[d] - 1) * stridesA[d]; \
            } else { \
                coord[d]--; \
                if (coord[d] >= 0) { \
                    offsetA -= stridesA[d]; \
                    break; \
                } \
                if (d == 0) { \
                    finished = 1; \
                    break; \
                } \
                coord[d] = shape[d] - 1; \
                offsetA += (shape[d] - 1) * stridesA[d]; \
            } \
        } \
        if (finished) break; \
    } \
    return 0; \
}

DEFINE_FIND_INDEX_CONVERT_FUNC(float16, uint16_t, float, decode_float16)
DEFINE_FIND_INDEX_CONVERT_FUNC(bfloat16, uint16_t, float, decode_bfloat16)

#undef DEFINE_FIND_INDEX_CONVERT_FUNC

static int s_find_index_complex128(const cpx_t *a, const int64_t *stridesA,
                             const int64_t *shape, int rank,
                             int op, cpx_t target,
                             const int64_t *startCoords, const int64_t *directions,
                             int64_t *matchCoords) {
    if (a == nullptr || rank < 0) return 0;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (total_elements <= 0) return 0;
    if (rank == 0) {
        cpx_t val = a[0];
        int match = 0;
        ComparisonOp cop = static_cast<ComparisonOp>(op);
        if (cop == ComparisonOp::EQ) {
            match = (val == target);
        } else if (cop == ComparisonOp::NE) {
            match = (val != target);
        }
        return match ? 1 : 0;
    }
    DECLARE_RANK_BUFFER_RET(int64_t, coord, rank, 0);
    for (int64_t i = 0; i < rank; i++) {
        if (startCoords[i] < 0 || startCoords[i] >= shape[i]) return 0;
        coord[i] = startCoords[i];
    }
    int64_t offsetA = 0;
    for (int64_t i = 0; i < rank; i++) {
        offsetA += coord[i] * stridesA[i];
    }
    while (1) {
        cpx_t val = a[offsetA];
        int match = 0;
        ComparisonOp cop = static_cast<ComparisonOp>(op);
        if (cop == ComparisonOp::EQ) {
            match = (val == target);
        } else if (cop == ComparisonOp::NE) {
            match = (val != target);
        }
        if (match) {
            if (matchCoords != nullptr) {
                for (int64_t i = 0; i < rank; i++) {
                    matchCoords[i] = coord[i];
                }
            }
            return 1;
        }
        int finished = 0;
        for (int d = rank - 1; d >= 0; d--) {
            if (directions[d] >= 0) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetA += stridesA[d];
                    break;
                }
                if (d == 0) {
                    finished = 1;
                    break;
                }
                coord[d] = 0;
                offsetA -= (shape[d] - 1) * stridesA[d];
            } else {
                coord[d]--;
                if (coord[d] >= 0) {
                    offsetA -= stridesA[d];
                    break;
                }
                if (d == 0) {
                    finished = 1;
                    break;
                }
                coord[d] = shape[d] - 1;
                offsetA += (shape[d] - 1) * stridesA[d];
            }
        }
        if (finished) break;
    }
    return 0;
}

static int s_find_index_complex64(const cpx_f_t *a, const int64_t *stridesA,
                            const int64_t *shape, int rank,
                            int op, cpx_f_t target,
                            const int64_t *startCoords, const int64_t *directions,
                            int64_t *matchCoords) {
    if (a == nullptr || rank < 0) return 0;
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) total_elements *= shape[i];
    if (total_elements <= 0) return 0;
    if (rank == 0) {
        cpx_f_t val = a[0];
        int match = 0;
        ComparisonOp cop = static_cast<ComparisonOp>(op);
        if (cop == ComparisonOp::EQ) {
            match = (val == target);
        } else if (cop == ComparisonOp::NE) {
            match = (val != target);
        }
        return match ? 1 : 0;
    }
    DECLARE_RANK_BUFFER_RET(int64_t, coord, rank, 0);
    for (int64_t i = 0; i < rank; i++) {
        if (startCoords[i] < 0 || startCoords[i] >= shape[i]) return 0;
        coord[i] = startCoords[i];
    }
    int64_t offsetA = 0;
    for (int64_t i = 0; i < rank; i++) {
        offsetA += coord[i] * stridesA[i];
    }
    while (1) {
        cpx_f_t val = a[offsetA];
        int match = 0;
        ComparisonOp cop = static_cast<ComparisonOp>(op);
        if (cop == ComparisonOp::EQ) {
            match = (val == target);
        } else if (cop == ComparisonOp::NE) {
            match = (val != target);
        }
        if (match) {
            if (matchCoords != nullptr) {
                for (int64_t i = 0; i < rank; i++) {
                    matchCoords[i] = coord[i];
                }
            }
            return 1;
        }
        int finished = 0;
        for (int d = rank - 1; d >= 0; d--) {
            if (directions[d] >= 0) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetA += stridesA[d];
                    break;
                }
                if (d == 0) {
                    finished = 1;
                    break;
                }
                coord[d] = 0;
                offsetA -= (shape[d] - 1) * stridesA[d];
            } else {
                coord[d]--;
                if (coord[d] >= 0) {
                    offsetA -= stridesA[d];
                    break;
                }
                if (d == 0) {
                    finished = 1;
                    break;
                }
                coord[d] = shape[d] - 1;
                offsetA += (shape[d] - 1) * stridesA[d];
            }
        }
        if (finished) break;
    }
    return 0;
}

int ndarray_find_index(
    int op, int dtype,
    const void *a, const int64_t *stridesA,
    const int64_t *shape, int rank,
    const void *target,
    const int64_t *startCoords,
    const int64_t *directions,
    int64_t *matchCoords
) {
    if (a == nullptr || target == nullptr) return 0;
    switch (dtype) {
        case DTYPE_FLOAT64:
            return s_find_index_double((const double*)a, stridesA, shape, rank, op, *(const double*)target, startCoords, directions, matchCoords);
        case DTYPE_FLOAT32:
            return s_find_index_float((const float*)a, stridesA, shape, rank, op, *(const float*)target, startCoords, directions, matchCoords);
        case DTYPE_INT32:
            return s_find_index_int32((const int32_t*)a, stridesA, shape, rank, op, *(const int32_t*)target, startCoords, directions, matchCoords);
        case DTYPE_INT64:
            return s_find_index_int64((const int64_t*)a, stridesA, shape, rank, op, *(const int64_t*)target, startCoords, directions, matchCoords);
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            return s_find_index_uint8((const uint8_t*)a, stridesA, shape, rank, op, *(const uint8_t*)target, startCoords, directions, matchCoords);
        case DTYPE_INT16:
            return s_find_index_int16((const int16_t*)a, stridesA, shape, rank, op, *(const int16_t*)target, startCoords, directions, matchCoords);
        case DTYPE_INT8:
            return s_find_index_int8((const int8_t*)a, stridesA, shape, rank, op, *(const int8_t*)target, startCoords, directions, matchCoords);
        case DTYPE_UINT16:
            return s_find_index_uint16((const uint16_t*)a, stridesA, shape, rank, op, *(const uint16_t*)target, startCoords, directions, matchCoords);
        case DTYPE_UINT32:
            return s_find_index_uint32((const uint32_t*)a, stridesA, shape, rank, op, *(const uint32_t*)target, startCoords, directions, matchCoords);
        case DTYPE_UINT64:
            return s_find_index_uint64((const uint64_t*)a, stridesA, shape, rank, op, *(const uint64_t*)target, startCoords, directions, matchCoords);
        case DTYPE_FLOAT16:
            return s_find_index_float16((const uint16_t*)a, stridesA, shape, rank, op, *(const float*)target, startCoords, directions, matchCoords);
        case DTYPE_BFLOAT16:
            return s_find_index_bfloat16((const uint16_t*)a, stridesA, shape, rank, op, *(const float*)target, startCoords, directions, matchCoords);
        case DTYPE_COMPLEX128:
            return s_find_index_complex128((const cpx_t*)a, stridesA, shape, rank, op, *(const cpx_t*)target, startCoords, directions, matchCoords);
        case DTYPE_COMPLEX64:
            return s_find_index_complex64((const cpx_f_t*)a, stridesA, shape, rank, op, *(const cpx_f_t*)target, startCoords, directions, matchCoords);
        default:
            return 0;
    }
}

#define COMPUTE_EUCLIDEAN(type, u, strideU, v, strideV, N, result) do { \
    double sum = 0.0; \
    for (int64_t i = 0; i < N; i++) { \
        double diff = (double)((const type*)u)[i * strideU] - (double)((const type*)v)[i * strideV]; \
        sum += diff * diff; \
    } \
    result = sqrt(sum); \
} while(0)

#define COMPUTE_COSINE(type, u, strideU, v, strideV, N, result) do { \
    double dot = 0.0; \
    double norm_u = 0.0; \
    double norm_v = 0.0; \
    for (int64_t i = 0; i < N; i++) { \
        double val_u = (double)((const type*)u)[i * strideU]; \
        double val_v = (double)((const type*)v)[i * strideV]; \
        dot += val_u * val_v; \
        norm_u += val_u * val_u; \
        norm_v += val_v * val_v; \
    } \
    if (norm_u == 0.0 || norm_v == 0.0) { \
        result = NAN; \
    } else { \
        result = 1.0 - (dot / (sqrt(norm_u) * sqrt(norm_v))); \
    } \
} while(0)

#define COMPUTE_HAMMING(type, u, strideU, v, strideV, N, result) do { \
    if (N == 0) { \
        result = NAN; \
    } else { \
        double diff_count = 0.0; \
        for (int64_t i = 0; i < N; i++) { \
            diff_count += (((const type*)u)[i * strideU] != ((const type*)v)[i * strideV]); \
        } \
        result = diff_count / (double)N; \
    } \
} while(0)

#define COMPUTE_CHEBYSHEV(type, u, strideU, v, strideV, N, result) do { \
    double max_diff = 0.0; \
    for (int64_t i = 0; i < N; i++) { \
        double diff = std::abs((double)((const type*)u)[i * strideU] - (double)((const type*)v)[i * strideV]); \
        if (std::isnan(diff)) { \
            max_diff = NAN; \
            break; \
        } \
        if (diff > max_diff) { \
            max_diff = diff; \
        } \
    } \
    result = max_diff; \
} while(0)

#define COMPUTE_DIST(type, u, strideU, v, strideV, N, metric, result) do { \
    switch(metric) { \
        case METRIC_EUCLIDEAN: COMPUTE_EUCLIDEAN(type, u, strideU, v, strideV, N, result); break; \
        case METRIC_COSINE: COMPUTE_COSINE(type, u, strideU, v, strideV, N, result); break; \
        case METRIC_HAMMING: COMPUTE_HAMMING(type, u, strideU, v, strideV, N, result); break; \
        case METRIC_CHEBYSHEV: COMPUTE_CHEBYSHEV(type, u, strideU, v, strideV, N, result); break; \
        default: result = NAN; break; \
    } \
} while(0)

#define DEFINE_PDIST(type) \
static void pdist_##type(const type *x, int64_t M, int64_t N, int64_t strideRowX, int64_t strideColX, int metric, double *out, int64_t strideOut) { \
    int64_t idx = 0; \
    for (int64_t i = 0; i < M; i++) { \
        const type *u = x + i * strideRowX; \
        for (int64_t j = i + 1; j < M; j++) { \
            const type *v = x + j * strideRowX; \
            double res; \
            COMPUTE_DIST(type, u, strideColX, v, strideColX, N, metric, res); \
            out[idx * strideOut] = res; \
            idx++; \
        } \
    } \
}

DEFINE_PDIST(double)
DEFINE_PDIST(float)
DEFINE_PDIST(float16_t)
DEFINE_PDIST(bfloat16_t)
DEFINE_PDIST(int64_t)
DEFINE_PDIST(int32_t)
DEFINE_PDIST(int16_t)
DEFINE_PDIST(int8_t)
DEFINE_PDIST(uint64_t)
DEFINE_PDIST(uint32_t)
DEFINE_PDIST(uint16_t)
DEFINE_PDIST(uint8_t)

#undef DEFINE_PDIST

void ndarray_pdist(
    int dtype,
    const void *x,
    int64_t M, int64_t N,
    int64_t strideRowX, int64_t strideColX,
    int metric,
    double *out,
    int64_t strideOut
) {
    if (x == nullptr || out == nullptr || M < 2 || N < 0) return;
    switch(dtype) {
        case DTYPE_FLOAT64:
            pdist_double((const double*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        case DTYPE_FLOAT32:
            pdist_float((const float*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        case DTYPE_FLOAT16:
            pdist_float16_t((const float16_t*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        case DTYPE_BFLOAT16:
            pdist_bfloat16_t((const bfloat16_t*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        case DTYPE_INT32:
            pdist_int32_t((const int32_t*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        case DTYPE_INT64:
            pdist_int64_t((const int64_t*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            pdist_uint8_t((const uint8_t*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        case DTYPE_INT16:
            pdist_int16_t((const int16_t*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        case DTYPE_INT8:
            pdist_int8_t((const int8_t*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        case DTYPE_UINT64:
            pdist_uint64_t((const uint64_t*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        case DTYPE_UINT32:
            pdist_uint32_t((const uint32_t*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        case DTYPE_UINT16:
            pdist_uint16_t((const uint16_t*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        default:
            return;
    }
}

#define DEFINE_CDIST(type) \
static void cdist_##type(const type *xa, const type *xb, int64_t M, int64_t K, int64_t N, \
                  int64_t strideRowXA, int64_t strideColXA, \
                  int64_t strideRowXB, int64_t strideColXB, \
                  int metric, \
                  double *out, int64_t strideRowOut, int64_t strideColOut) { \
    for (int64_t i = 0; i < M; i++) { \
        const type *u = xa + i * strideRowXA; \
        for (int64_t j = 0; j < K; j++) { \
            const type *v = xb + j * strideRowXB; \
            double res; \
            COMPUTE_DIST(type, u, strideColXA, v, strideColXB, N, metric, res); \
            out[i * strideRowOut + j * strideColOut] = res; \
        } \
    } \
}

DEFINE_CDIST(double)
DEFINE_CDIST(float)
DEFINE_CDIST(float16_t)
DEFINE_CDIST(bfloat16_t)
DEFINE_CDIST(int64_t)
DEFINE_CDIST(int32_t)
DEFINE_CDIST(int16_t)
DEFINE_CDIST(int8_t)
DEFINE_CDIST(uint64_t)
DEFINE_CDIST(uint32_t)
DEFINE_CDIST(uint16_t)
DEFINE_CDIST(uint8_t)

#undef DEFINE_CDIST

void ndarray_cdist(
    int dtype,
    const void *xa, const void *xb,
    int64_t M, int64_t K, int64_t N,
    int64_t strideRowXA, int64_t strideColXA,
    int64_t strideRowXB, int64_t strideColXB,
    int metric,
    double *out,
    int64_t strideRowOut, int64_t strideColOut
) {
    if (xa == nullptr || xb == nullptr || out == nullptr || M <= 0 || K <= 0 || N < 0) return;
    switch(dtype) {
        case DTYPE_FLOAT64:
            cdist_double((const double*)xa, (const double*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        case DTYPE_FLOAT32:
            cdist_float((const float*)xa, (const float*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        case DTYPE_FLOAT16:
            cdist_float16_t((const float16_t*)xa, (const float16_t*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        case DTYPE_BFLOAT16:
            cdist_bfloat16_t((const bfloat16_t*)xa, (const bfloat16_t*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        case DTYPE_INT32:
            cdist_int32_t((const int32_t*)xa, (const int32_t*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        case DTYPE_INT64:
            cdist_int64_t((const int64_t*)xa, (const int64_t*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            cdist_uint8_t((const uint8_t*)xa, (const uint8_t*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        case DTYPE_INT16:
            cdist_int16_t((const int16_t*)xa, (const int16_t*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        case DTYPE_INT8:
            cdist_int8_t((const int8_t*)xa, (const int8_t*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        case DTYPE_UINT64:
            cdist_uint64_t((const uint64_t*)xa, (const uint64_t*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        case DTYPE_UINT32:
            cdist_uint32_t((const uint32_t*)xa, (const uint32_t*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        case DTYPE_UINT16:
            cdist_uint16_t((const uint16_t*)xa, (const uint16_t*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        default:
            return;
    }
}

// Integer Absolute Value Implementations (Contiguous)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_abs_int64, int64_t, int64_t, (x < 0) ? (int64_t)(-(uint64_t)x) : x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_abs_int32, int32_t, int32_t, (x < 0) ? (int32_t)(-(uint32_t)x) : x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_abs_int16, int16_t, int16_t, (x < 0) ? (int16_t)(-(uint16_t)x) : x)

DEFINE_CONTIGUOUS_UNARY_IMPL(v_abs_uint8, uint8_t, uint8_t, x)

// Integer Absolute Value Implementations (Strided)
DEFINE_STRIDED_UNARY_IMPL(s_abs_int64, int64_t, int64_t, (x < 0) ? (int64_t)(-(uint64_t)x) : x)
DEFINE_STRIDED_UNARY_IMPL(s_abs_int32, int32_t, int32_t, (x < 0) ? (int32_t)(-(uint32_t)x) : x)
DEFINE_STRIDED_UNARY_IMPL(s_abs_int16, int16_t, int16_t, (x < 0) ? (int16_t)(-(uint16_t)x) : x)
DEFINE_STRIDED_UNARY_IMPL(s_abs_uint8, uint8_t, uint8_t, x)

// Integer & Extended DType Exponentiation Implementations (Contiguous)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_pow_int64, int64_t, int64_t, int64_t, ipow(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_pow_int32, int32_t, int32_t, int32_t, ipow(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_pow_int16, int16_t, int16_t, int16_t, ipow(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_pow_int8, int8_t, int8_t, int8_t, ipow(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_pow_uint64, uint64_t, uint64_t, uint64_t, ipow(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_pow_uint32, uint32_t, uint32_t, uint32_t, ipow(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_pow_uint16, uint16_t, uint16_t, uint16_t, ipow(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_pow_uint8, uint8_t, uint8_t, uint8_t, ipow(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_pow_float16, uint16_t, uint16_t, uint16_t, encode_float16(std::pow(decode_float16(x), decode_float16(y))))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_pow_bfloat16, uint16_t, uint16_t, uint16_t, encode_bfloat16(std::pow(decode_bfloat16(x), decode_bfloat16(y))))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_pow_boolean, uint8_t, uint8_t, uint8_t, (y == 0 || x != 0) ? (uint8_t)1 : (uint8_t)0)

// Integer & Extended DType Exponentiation Implementations (Strided)
DEFINE_STRIDED_BINARY_IMPL(s_pow_int64, int64_t, int64_t, int64_t, ipow(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_pow_int32, int32_t, int32_t, int32_t, ipow(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_pow_int16, int16_t, int16_t, int16_t, ipow(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_pow_int8, int8_t, int8_t, int8_t, ipow(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_pow_uint64, uint64_t, uint64_t, uint64_t, ipow(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_pow_uint32, uint32_t, uint32_t, uint32_t, ipow(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_pow_uint16, uint16_t, uint16_t, uint16_t, ipow(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_pow_uint8, uint8_t, uint8_t, uint8_t, ipow(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_pow_float16, uint16_t, uint16_t, uint16_t, encode_float16(std::pow(decode_float16(x), decode_float16(y))))
DEFINE_STRIDED_BINARY_IMPL(s_pow_bfloat16, uint16_t, uint16_t, uint16_t, encode_bfloat16(std::pow(decode_bfloat16(x), decode_bfloat16(y))))
DEFINE_STRIDED_BINARY_IMPL(s_pow_boolean, uint8_t, uint8_t, uint8_t, (y == 0 || x != 0) ? (uint8_t)1 : (uint8_t)0)

// Extended DType Square Implementations (Contiguous & Strided)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_int16, int16_t, int16_t, (int16_t)((uint16_t)x * (uint16_t)x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_int8, int8_t, int8_t, (int8_t)((uint8_t)x * (uint8_t)x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_uint64, uint64_t, uint64_t, x * x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_uint32, uint32_t, uint32_t, x * x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_uint16, uint16_t, uint16_t, (uint16_t)(x * x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_uint8, uint8_t, uint8_t, (uint8_t)(x * x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_float16, uint16_t, uint16_t, encode_float16(decode_float16(x) * decode_float16(x)))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_bfloat16, uint16_t, uint16_t, encode_bfloat16(decode_bfloat16(x) * decode_bfloat16(x)))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_boolean, uint8_t, uint8_t, (x != 0) ? (uint8_t)1 : (uint8_t)0)

DEFINE_STRIDED_UNARY_IMPL(s_square_int16, int16_t, int16_t, (int16_t)((uint16_t)x * (uint16_t)x))
DEFINE_STRIDED_UNARY_IMPL(s_square_int8, int8_t, int8_t, (int8_t)((uint8_t)x * (uint8_t)x))
DEFINE_STRIDED_UNARY_IMPL(s_square_uint64, uint64_t, uint64_t, x * x)
DEFINE_STRIDED_UNARY_IMPL(s_square_uint32, uint32_t, uint32_t, x * x)
DEFINE_STRIDED_UNARY_IMPL(s_square_uint16, uint16_t, uint16_t, (uint16_t)(x * x))
DEFINE_STRIDED_UNARY_IMPL(s_square_uint8, uint8_t, uint8_t, (uint8_t)(x * x))
DEFINE_STRIDED_UNARY_IMPL(s_square_float16, uint16_t, uint16_t, encode_float16(decode_float16(x) * decode_float16(x)))
DEFINE_STRIDED_UNARY_IMPL(s_square_bfloat16, uint16_t, uint16_t, encode_bfloat16(decode_bfloat16(x) * decode_bfloat16(x)))
DEFINE_STRIDED_UNARY_IMPL(s_square_boolean, uint8_t, uint8_t, (x != 0) ? (uint8_t)1 : (uint8_t)0)

// Bincount Implementations (Contiguous)
void v_bincount_int32(const int32_t *src, int64_t *res, int64_t size, int64_t res_size) {
    if (src == nullptr || res == nullptr || size <= 0 || res_size <= 0) return;
    for (int64_t i = 0; i < size; ++i) {
        int32_t val = src[i];
        if (val >= 0 && val < res_size) {
            res[val]++;
        }
    }
}

void v_bincount_int64(const int64_t *src, int64_t *res, int64_t size, int64_t res_size) {
    if (src == nullptr || res == nullptr || size <= 0 || res_size <= 0) return;
    for (int64_t i = 0; i < size; ++i) {
        int64_t val = src[i];
        if (val >= 0 && val < res_size) {
            res[val]++;
        }
    }
}

void v_bincount_weights_int32_double(const int32_t *src, const double *weights, double *res, int64_t size, int64_t res_size) {
    if (src == nullptr || weights == nullptr || res == nullptr || size <= 0 || res_size <= 0) return;
    for (int64_t i = 0; i < size; ++i) {
        int32_t val = src[i];
        if (val >= 0 && val < res_size) {
            res[val] += weights[i];
        }
    }
}

void v_bincount_weights_int64_double(const int64_t *src, const double *weights, double *res, int64_t size, int64_t res_size) {
    if (src == nullptr || weights == nullptr || res == nullptr || size <= 0 || res_size <= 0) return;
    for (int64_t i = 0; i < size; ++i) {
        int64_t val = src[i];
        if (val >= 0 && val < res_size) {
            res[val] += weights[i];
        }
    }
}

void v_bincount_weights_int32_float(const int32_t *src, const float *weights, float *res, int64_t size, int64_t res_size) {
    if (src == nullptr || weights == nullptr || res == nullptr || size <= 0 || res_size <= 0) return;
    for (int64_t i = 0; i < size; ++i) {
        int32_t val = src[i];
        if (val >= 0 && val < res_size) {
            res[val] += weights[i];
        }
    }
}

void v_bincount_weights_int64_float(const int64_t *src, const float *weights, float *res, int64_t size, int64_t res_size) {
    if (src == nullptr || weights == nullptr || res == nullptr || size <= 0 || res_size <= 0) return;
    for (int64_t i = 0; i < size; ++i) {
        int64_t val = src[i];
        if (val >= 0 && val < res_size) {
            res[val] += weights[i];
        }
    }
}

// Bincount Implementations (Strided)
void s_bincount_int32(const int32_t *src, int64_t strideSrc, int64_t *res, int64_t strideRes, int64_t size, int64_t res_size) {
    if (src == nullptr || res == nullptr || size <= 0 || res_size <= 0) return;
    for (int64_t i = 0; i < size; ++i) {
        int32_t val = src[i * strideSrc];
        if (val >= 0 && val < res_size) {
            res[val * strideRes]++;
        }
    }
}

void s_bincount_int64(const int64_t *src, int64_t strideSrc, int64_t *res, int64_t strideRes, int64_t size, int64_t res_size) {
    if (src == nullptr || res == nullptr || size <= 0 || res_size <= 0) return;
    for (int64_t i = 0; i < size; ++i) {
        int64_t val = src[i * strideSrc];
        if (val >= 0 && val < res_size) {
            res[val * strideRes]++;
        }
    }
}

void s_bincount_weights_int32_double(const int32_t *src, int64_t strideSrc, const double *weights, int64_t strideWeights, double *res, int64_t strideRes, int64_t size, int64_t res_size) {
    if (src == nullptr || weights == nullptr || res == nullptr || size <= 0 || res_size <= 0) return;
    for (int64_t i = 0; i < size; ++i) {
        int32_t val = src[i * strideSrc];
        if (val >= 0 && val < res_size) {
            res[val * strideRes] += weights[i * strideWeights];
        }
    }
}

void s_bincount_weights_int64_double(const int64_t *src, int64_t strideSrc, const double *weights, int64_t strideWeights, double *res, int64_t strideRes, int64_t size, int64_t res_size) {
    if (src == nullptr || weights == nullptr || res == nullptr || size <= 0 || res_size <= 0) return;
    for (int64_t i = 0; i < size; ++i) {
        int64_t val = src[i * strideSrc];
        if (val >= 0 && val < res_size) {
            res[val * strideRes] += weights[i * strideWeights];
        }
    }
}

void s_bincount_weights_int32_float(const int32_t *src, int64_t strideSrc, const float *weights, int64_t strideWeights, float *res, int64_t strideRes, int64_t size, int64_t res_size) {
    if (src == nullptr || weights == nullptr || res == nullptr || size <= 0 || res_size <= 0) return;
    for (int64_t i = 0; i < size; ++i) {
        int32_t val = src[i * strideSrc];
        if (val >= 0 && val < res_size) {
            res[val * strideRes] += weights[i * strideWeights];
        }
    }
}

void s_bincount_weights_int64_float(const int64_t *src, int64_t strideSrc, const float *weights, int64_t strideWeights, float *res, int64_t strideRes, int64_t size, int64_t res_size) {
    if (src == nullptr || weights == nullptr || res == nullptr || size <= 0 || res_size <= 0) return;
    for (int64_t i = 0; i < size; ++i) {
        int64_t val = src[i * strideSrc];
        if (val >= 0 && val < res_size) {
            res[val * strideRes] += weights[i * strideWeights];
        }
    }
}

#define DEFINE_SLOGDET_REAL(name, type) \
void name(const type *a, const int64_t *stridesA, \
          type *sign, const int64_t *stridesSign, \
          type *logdet, const int64_t *stridesLogdet, \
          const int64_t *shape, int rank, \
          type *aCopy, int *ipiv, \
          int (*lapack_getrf)(int, int, int, void *, int, int *)) { \
    if (a == nullptr || sign == nullptr || logdet == nullptr || shape == nullptr || lapack_getrf == nullptr || rank < 2) return; \
    int n = shape[rank - 1]; \
    if (n > 0 && (aCopy == nullptr || ipiv == nullptr)) return; \
    int64_t stack_elements = 1; \
    for (int64_t i = 0; i < rank - 2; i++) stack_elements *= shape[i]; \
    DECLARE_RANK_BUFFER(int64_t, coord, rank); \
    int64_t offsetA = 0, offsetSign = 0, offsetLogdet = 0; \
    for (int64_t el = 0; el < stack_elements; el++) { \
        type signValue = 1.0; \
        type logdetValue = 0.0; \
        if (n > 0) { \
            for (int64_t i = 0; i < n; i++) { \
                for (int64_t j = 0; j < n; j++) { \
                    aCopy[i * n + j] = a[offsetA + i * stridesA[rank - 2] + j * stridesA[rank - 1]]; \
                } \
            } \
            int info = lapack_getrf(101, n, n, aCopy, n, ipiv); \
            if (info < 0) { \
                signValue = std::numeric_limits<type>::quiet_NaN(); \
                logdetValue = std::numeric_limits<type>::quiet_NaN(); \
            } else { \
                for (int64_t i = 0; i < n; i++) { \
                    type val = aCopy[i * n + i]; \
                    if (std::isnan(val)) { \
                        signValue = std::numeric_limits<type>::quiet_NaN(); \
                        logdetValue = std::numeric_limits<type>::quiet_NaN(); \
                        break; \
                    } \
                    if (val == 0.0) { \
                        signValue = 0.0; \
                        logdetValue += -std::numeric_limits<type>::infinity(); \
                    } else if (val < 0.0) { \
                        signValue = -signValue; \
                        logdetValue += std::log(-val); \
                    } else { \
                        logdetValue += std::log(val); \
                    } \
                } \
                if (std::isnan(logdetValue)) { \
                    signValue = std::numeric_limits<type>::quiet_NaN(); \
                } else if (signValue != 0.0 && !std::isnan(signValue)) { \
                    int swaps = 0; \
                    for (int64_t i = 0; i < n; i++) { \
                        if (ipiv[i] != i + 1) swaps++; \
                    } \
                    if (swaps % 2 != 0) signValue = -signValue; \
                } \
            } \
        } \
        sign[offsetSign] = signValue; \
        logdet[offsetLogdet] = logdetValue; \
        for (int d = rank - 3; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetA += stridesA[d]; \
                offsetSign += stridesSign[d]; \
                offsetLogdet += stridesLogdet[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetA -= (shape[d] - 1) * stridesA[d]; \
            offsetSign -= (shape[d] - 1) * stridesSign[d]; \
            offsetLogdet -= (shape[d] - 1) * stridesLogdet[d]; \
        } \
    } \
}

#define DEFINE_SLOGDET_COMPLEX(name, cpx_type, real_type) \
void name(const cpx_type *a, const int64_t *stridesA, \
          cpx_type *sign, const int64_t *stridesSign, \
          real_type *logdet, const int64_t *stridesLogdet, \
          const int64_t *shape, int rank, \
          cpx_type *aCopy, int *ipiv, \
          int (*lapack_getrf)(int, int, int, void *, int, int *)) { \
    if (a == nullptr || sign == nullptr || logdet == nullptr || shape == nullptr || lapack_getrf == nullptr || rank < 2) return; \
    int n = shape[rank - 1]; \
    if (n > 0 && (aCopy == nullptr || ipiv == nullptr)) return; \
    int64_t stack_elements = 1; \
    for (int64_t i = 0; i < rank - 2; i++) stack_elements *= shape[i]; \
    DECLARE_RANK_BUFFER(int64_t, coord, rank); \
    int64_t offsetA = 0, offsetSign = 0, offsetLogdet = 0; \
    for (int64_t el = 0; el < stack_elements; el++) { \
        cpx_type signValue = {1.0, 0.0}; \
        real_type logdetValue = 0.0; \
        if (n > 0) { \
            for (int64_t i = 0; i < n; i++) { \
                for (int64_t j = 0; j < n; j++) { \
                    aCopy[i * n + j] = a[offsetA + i * stridesA[rank - 2] + j * stridesA[rank - 1]]; \
                } \
            } \
            int info = lapack_getrf(101, n, n, aCopy, n, ipiv); \
            if (info < 0) { \
                signValue.r = std::numeric_limits<real_type>::quiet_NaN(); \
                signValue.i = std::numeric_limits<real_type>::quiet_NaN(); \
                logdetValue = std::numeric_limits<real_type>::quiet_NaN(); \
            } else { \
                for (int64_t i = 0; i < n; i++) { \
                    real_type r = aCopy[i * n + i].r; \
                    real_type imag = aCopy[i * n + i].i; \
                    if (std::isnan(r) || std::isnan(imag)) { \
                        signValue.r = std::numeric_limits<real_type>::quiet_NaN(); \
                        signValue.i = std::numeric_limits<real_type>::quiet_NaN(); \
                        logdetValue = std::numeric_limits<real_type>::quiet_NaN(); \
                        break; \
                    } \
                    real_type abs_val = std::hypot(r, imag); \
                    if (abs_val == 0.0) { \
                        signValue.r = 0.0; \
                        signValue.i = 0.0; \
                        logdetValue += -std::numeric_limits<real_type>::infinity(); \
                    } else { \
                        logdetValue += std::log(abs_val); \
                        real_type sr = r / abs_val; \
                        real_type si = imag / abs_val; \
                        real_type old_sr = signValue.r; \
                        real_type old_si = signValue.i; \
                        signValue.r = old_sr * sr - old_si * si; \
                        signValue.i = old_sr * si + old_si * sr; \
                    } \
                } \
                if (std::isnan(logdetValue)) { \
                    signValue.r = std::numeric_limits<real_type>::quiet_NaN(); \
                    signValue.i = std::numeric_limits<real_type>::quiet_NaN(); \
                } else if ((signValue.r != 0.0 || signValue.i != 0.0) && !std::isnan(signValue.r) && !std::isnan(signValue.i)) { \
                    int swaps = 0; \
                    for (int64_t i = 0; i < n; i++) { \
                        if (ipiv[i] != i + 1) swaps++; \
                    } \
                    if (swaps % 2 != 0) { \
                        signValue.r = -signValue.r; \
                        signValue.i = -signValue.i; \
                    } \
                } \
            } \
        } \
        sign[offsetSign] = signValue; \
        logdet[offsetLogdet] = logdetValue; \
        for (int d = rank - 3; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetA += stridesA[d]; \
                offsetSign += stridesSign[d]; \
                offsetLogdet += stridesLogdet[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetA -= (shape[d] - 1) * stridesA[d]; \
            offsetSign -= (shape[d] - 1) * stridesSign[d]; \
            offsetLogdet -= (shape[d] - 1) * stridesLogdet[d]; \
        } \
    } \
}

DEFINE_SLOGDET_REAL(s_slogdet_double, double)
DEFINE_SLOGDET_REAL(s_slogdet_float, float)
DEFINE_SLOGDET_COMPLEX(s_slogdet_complex_double, cpx_t, double)
DEFINE_SLOGDET_COMPLEX(s_slogdet_complex_float, cpx_f_t, float)

// Eigenvalue assembly helpers
void assemble_eigenvalues_double(
    cpx_t *w,
    int64_t strideWLast,
    const double *wr,
    const double *wi,
    int64_t n
) {
    if (w == nullptr || wr == nullptr || wi == nullptr || n <= 0) {
        return;
    }
    for (int64_t j = 0; j < n; j++) {
        w[j * strideWLast].r = wr[j];
        w[j * strideWLast].i = wi[j];
    }
}

void assemble_eigenvalues_float(
    cpx_f_t *w,
    int64_t strideWLast,
    const float *wr,
    const float *wi,
    int64_t n
) {
    if (w == nullptr || wr == nullptr || wi == nullptr || n <= 0) {
        return;
    }
    for (int64_t j = 0; j < n; j++) {
        w[j * strideWLast].r = wr[j];
        w[j * strideWLast].i = wi[j];
    }
}

// ============================================================================
// SECTION: MODIFIED BESSEL FUNCTION OF THE FIRST KIND, ORDER 0 (i0)
// ============================================================================

const double NDARRAY_PI = 3.14159265358979323846;

template <typename T>
static inline T i0_real_impl(T x) {
    if (std::isinf(x)) return std::abs(x);
    T abs_x = std::abs(x);
    if (abs_x <= (T)3.75) {
        T t = x / (T)3.75;
        T u = t * t;
        return (T)1.0 + u * ((T)3.5156229 + u * ((T)3.0899424 + u * ((T)1.2067492 + u * ((T)0.2659732 + u * ((T)0.0360768 + u * (T)0.0045813)))));
    } else {
        T t = (T)3.75 / abs_x;
        T term = (T)0.39894228 + t * ((T)0.01328592 + t * ((T)0.00225319 + t * (-(T)0.00157565 + t * ((T)0.00916281 + t * (-(T)0.02057706 + t * ((T)0.02635537 + t * (-(T)0.01647633 + t * (T)0.00392377)))))));
        return (std::exp(abs_x) / std::sqrt(abs_x)) * term;
    }
}

template <typename T>
static inline std::complex<T> i0_complex_impl(std::complex<T> z) {
    T x = z.real();
    T y = z.imag();
    if (y == (T)0.0) {
        return std::complex<T>(i0_real_impl(x), (T)0.0);
    }
    
    T X = std::abs(x);
    T Y = std::abs(y);
    
    std::complex<T> z_q1(X, Y);
    T abs_z = std::abs(z_q1);
    std::complex<T> res;
    
    if (abs_z <= (T)15.0) {
        // Power series
        std::complex<T> sum = (T)1.0;
        std::complex<T> term = (T)1.0;
        std::complex<T> z2_4 = (z_q1 * z_q1) / (T)4.0;
        for (int64_t k = 1; k <= 100; ++k) {
            term = term * z2_4 / (T)(k * k);
            sum += term;
            if (std::abs(term) < (T)1e-15) {
                break;
            }
        }
        res = sum;
    } else {
        // Asymptotic expansion with both terms, valid in first quadrant
        std::complex<T> one(1.0, 0.0);
        std::complex<T> eight(8.0, 0.0);
        std::complex<T> w128(128.0, 0.0);
        std::complex<T> w1024(1024.0, 0.0);
        std::complex<T> w32768(32768.0, 0.0);
        
        std::complex<T> term1 = one / (eight * z_q1);
        std::complex<T> term2 = std::complex<T>(9.0, 0.0) / (w128 * z_q1 * z_q1);
        std::complex<T> term3 = std::complex<T>(75.0, 0.0) / (w1024 * z_q1 * z_q1 * z_q1);
        std::complex<T> term4 = std::complex<T>(1225.0, 0.0) / (w32768 * z_q1 * z_q1 * z_q1 * z_q1);
        
        std::complex<T> A_z = one + term1 + term2 + term3 + term4;
        std::complex<T> A_neg_z = one - term1 + term2 - term3 + term4;
        
        std::complex<T> pi(NDARRAY_PI, 0.0);
        std::complex<T> two_pi_z = std::complex<T>(2.0, 0.0) * pi * z_q1;
        std::complex<T> sqrt_two_pi_z = std::sqrt(two_pi_z);
        
        std::complex<T> term_exp = std::exp(z_q1) * A_z / sqrt_two_pi_z;
        std::complex<T> term_neg_exp = std::complex<T>(0.0, 1.0) * std::exp(-z_q1) * A_neg_z / sqrt_two_pi_z;
        
        res = term_exp + term_neg_exp;
    }
    
    if ((x < (T)0.0) ^ (y < (T)0.0)) {
        res = std::conj(res);
    }
    return res;
}

static inline cpx_f_t cpx_i0_f(cpx_f_t z) {
    std::complex<float> cz(z.r, z.i);
    auto cres = i0_complex_impl(cz);
    return {cres.real(), cres.imag()};
}

static inline cpx_t cpx_i0(cpx_t z) {
    std::complex<double> cz(z.r, z.i);
    auto cres = i0_complex_impl(cz);
    return {cres.real(), cres.imag()};
}

extern "C" {

DEFINE_CONTIGUOUS_UNARY_IMPL(v_i0_float, float, float, i0_real_impl<float>(x))
DEFINE_STRIDED_UNARY_IMPL(s_i0_float, float, float, i0_real_impl<float>(x))

DEFINE_CONTIGUOUS_UNARY_IMPL(v_i0_double, double, double, i0_real_impl<double>(x))
DEFINE_STRIDED_UNARY_IMPL(s_i0_double, double, double, i0_real_impl<double>(x))

DEFINE_CONTIGUOUS_UNARY_IMPL(v_i0_complex64, cpx_f_t, cpx_f_t, cpx_i0_f(x))
DEFINE_STRIDED_UNARY_IMPL(s_i0_complex64, cpx_f_t, cpx_f_t, cpx_i0_f(x))

DEFINE_CONTIGUOUS_UNARY_IMPL(v_i0_complex128, cpx_t, cpx_t, cpx_i0(x))
DEFINE_STRIDED_UNARY_IMPL(s_i0_complex128, cpx_t, cpx_t, cpx_i0(x))

DEFINE_CONTIGUOUS_UNARY_IMPL(v_gamma_float, float, float, std::tgamma(x))
DEFINE_STRIDED_UNARY_IMPL(s_gamma_float, float, float, std::tgamma(x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_gamma_double, double, double, std::tgamma(x))
DEFINE_STRIDED_UNARY_IMPL(s_gamma_double, double, double, std::tgamma(x))

DEFINE_CONTIGUOUS_UNARY_IMPL(v_erf_float, float, float, std::erf(x))
DEFINE_STRIDED_UNARY_IMPL(s_erf_float, float, float, std::erf(x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_erf_double, double, double, std::erf(x))
DEFINE_STRIDED_UNARY_IMPL(s_erf_double, double, double, std::erf(x))

}


/* Spatial N-D Correlation Implementation */

inline void add_prod_corr(double& sum, double a, double b) { sum += a * b; }
inline void add_prod_corr(float& sum, float a, float b) { sum += a * b; }
inline void add_prod_corr(int64_t& sum, int64_t a, int64_t b) {
    sum = (int64_t)((uint64_t)sum + (uint64_t)a * (uint64_t)b);
}
inline void add_prod_corr(uint64_t& sum, uint64_t a, uint64_t b) {
    sum += a * b;
}
inline void add_prod_corr(int32_t& sum, int32_t a, int32_t b) {
    sum = (int32_t)((uint32_t)sum + (uint32_t)a * (uint32_t)b);
}
inline void add_prod_corr(cpx_t& sum, cpx_t a, cpx_t b) {
    sum.r += a.r * b.r + a.i * b.i;
    sum.i += a.i * b.r - a.r * b.i;
}
inline void add_prod_corr(cpx_f_t& sum, cpx_f_t a, cpx_f_t b) {
    sum.r += a.r * b.r + a.i * b.i;
    sum.i += a.i * b.r - a.r * b.i;
}

inline void set_zero_corr(double& val) { val = 0.0; }
inline void set_zero_corr(float& val) { val = 0.0f; }
inline void set_zero_corr(int64_t& val) { val = 0; }
inline void set_zero_corr(uint64_t& val) { val = 0; }
inline void set_zero_corr(int32_t& val) { val = 0; }
inline void set_zero_corr(cpx_t& val) { val.r = 0.0; val.i = 0.0; }
inline void set_zero_corr(cpx_f_t& val) { val.r = 0.0f; val.i = 0.0f; }

template <typename T>
static void s_correlate_valid_generic(
    const T *src, const int64_t *stridesSrc,
    const T *kernel, const int64_t *stridesKernel,
    T *res, const int64_t *stridesRes,
    const int64_t *resShape, const int64_t *kernelShape, int rank
) {
    if (src == nullptr || kernel == nullptr || res == nullptr || rank <= 0 ||
        stridesSrc == nullptr || stridesKernel == nullptr || stridesRes == nullptr || resShape == nullptr || kernelShape == nullptr) {
        return;
    }
    if (rank == 1) {
        int64_t R = resShape[0];
        int64_t K = kernelShape[0];
        int64_t sSrc = stridesSrc[0];
        int64_t sK = stridesKernel[0];
        int64_t sRes = stridesRes[0];
        for (int64_t r = 0; r < R; ++r) {
            T sum;
            set_zero_corr(sum);
            const T* pSrc = src + r * sSrc;
            const T* pK = kernel;
            for (int64_t k = 0; k < K; ++k) {
                add_prod_corr(sum, *pSrc, *pK);
                pSrc += sSrc;
                pK += sK;
            }
            res[r * sRes] = sum;
        }
        return;
    }
    if (rank == 2) {
        int64_t R0 = resShape[0], R1 = resShape[1];
        int64_t K0 = kernelShape[0], K1 = kernelShape[1];
        int64_t sS0 = stridesSrc[0], sS1 = stridesSrc[1];
        int64_t sK0 = stridesKernel[0], sK1 = stridesKernel[1];
        int64_t sR0 = stridesRes[0], sR1 = stridesRes[1];
        for (int64_t r0 = 0; r0 < R0; ++r0) {
            for (int64_t r1 = 0; r1 < R1; ++r1) {
                T sum;
                set_zero_corr(sum);
                const T* pSrcRow = src + r0 * sS0 + r1 * sS1;
                const T* pKRow = kernel;
                for (int64_t k0 = 0; k0 < K0; ++k0) {
                    const T* pSrc = pSrcRow;
                    const T* pK = pKRow;
                    for (int64_t k1 = 0; k1 < K1; ++k1) {
                        add_prod_corr(sum, *pSrc, *pK);
                        pSrc += sS1;
                        pK += sK1;
                    }
                    pSrcRow += sS0;
                    pKRow += sK0;
                }
                res[r0 * sR0 + r1 * sR1] = sum;
            }
        }
        return;
    }

    auto recursive_kernel = [&](auto self, int dim, const T* pSrc, const T* pK, T& sum) -> void {
        if (dim == rank) {
            add_prod_corr(sum, *pSrc, *pK);
            return;
        }
        int64_t Kdim = kernelShape[dim];
        int64_t sS = stridesSrc[dim];
        int64_t sK = stridesKernel[dim];
        for (int64_t k = 0; k < Kdim; ++k) {
            self(self, dim + 1, pSrc + k * sS, pK + k * sK, sum);
        }
    };

    auto recursive_res = [&](auto self, int dim, const T* pSrcBase, T* pResCur) -> void {
        if (dim == rank) {
            T sum;
            set_zero_corr(sum);
            recursive_kernel(recursive_kernel, 0, pSrcBase, kernel, sum);
            *pResCur = sum;
            return;
        }
        int64_t Rdim = resShape[dim];
        int64_t sS = stridesSrc[dim];
        int64_t sR = stridesRes[dim];
        for (int64_t r = 0; r < Rdim; ++r) {
            self(self, dim + 1, pSrcBase + r * sS, pResCur + r * sR);
        }
    };

    recursive_res(recursive_res, 0, src, res);
}

template <typename T>
static void s_correlate_full_1d_generic(
    const T *src, int64_t N,
    const T *kernel, int64_t K,
    T *res
) {
    int64_t M = N + K - 1;
    for (int64_t i = 0; i < M; ++i) {
        int64_t s = i - K + 1;
        int64_t in1Start = std::max((int64_t)0, s);
        int64_t mStart = std::max((int64_t)0, -s);
        int64_t mEnd = std::min(K, N - s);
        int64_t L = mEnd - mStart;

        T sum;
        set_zero_corr(sum);
        if (L > 0) {
            const T* pSrc = src + in1Start;
            const T* pK = kernel + mStart;
            for (int64_t l = 0; l < L; ++l) {
                add_prod_corr(sum, pSrc[l], pK[l]);
            }
        }
        res[i] = sum;
    }
}

extern "C" {
void s_correlate_full_1d_double(const double *src, int64_t N, const double *kernel, int64_t K, double *res) {
    s_correlate_full_1d_generic<double>(src, N, kernel, K, res);
}
void s_correlate_full_1d_float(const float *src, int64_t N, const float *kernel, int64_t K, float *res) {
    s_correlate_full_1d_generic<float>(src, N, kernel, K, res);
}
void s_correlate_valid_double(const double *src, const int64_t *stridesSrc, const double *kernel, const int64_t *stridesKernel, double *res, const int64_t *stridesRes, const int64_t *resShape, const int64_t *kernelShape, int rank) {
    s_correlate_valid_generic<double>(src, stridesSrc, kernel, stridesKernel, res, stridesRes, resShape, kernelShape, rank);
}

void s_correlate_valid_float(const float *src, const int64_t *stridesSrc, const float *kernel, const int64_t *stridesKernel, float *res, const int64_t *stridesRes, const int64_t *resShape, const int64_t *kernelShape, int rank) {
    s_correlate_valid_generic<float>(src, stridesSrc, kernel, stridesKernel, res, stridesRes, resShape, kernelShape, rank);
}
void s_correlate_valid_complex128(const cpx_t *src, const int64_t *stridesSrc, const cpx_t *kernel, const int64_t *stridesKernel, cpx_t *res, const int64_t *stridesRes, const int64_t *resShape, const int64_t *kernelShape, int rank) {
    s_correlate_valid_generic<cpx_t>(src, stridesSrc, kernel, stridesKernel, res, stridesRes, resShape, kernelShape, rank);
}
void s_correlate_valid_complex64(const cpx_f_t *src, const int64_t *stridesSrc, const cpx_f_t *kernel, const int64_t *stridesKernel, cpx_f_t *res, const int64_t *stridesRes, const int64_t *resShape, const int64_t *kernelShape, int rank) {
    s_correlate_valid_generic<cpx_f_t>(src, stridesSrc, kernel, stridesKernel, res, stridesRes, resShape, kernelShape, rank);
}
void s_correlate_valid_int64(const int64_t *src, const int64_t *stridesSrc, const int64_t *kernel, const int64_t *stridesKernel, int64_t *res, const int64_t *stridesRes, const int64_t *resShape, const int64_t *kernelShape, int rank) {
    s_correlate_valid_generic<int64_t>(src, stridesSrc, kernel, stridesKernel, res, stridesRes, resShape, kernelShape, rank);
}
void s_correlate_valid_int32(const int32_t *src, const int64_t *stridesSrc, const int32_t *kernel, const int64_t *stridesKernel, int32_t *res, const int64_t *stridesRes, const int64_t *resShape, const int64_t *kernelShape, int rank) {
    s_correlate_valid_generic<int32_t>(src, stridesSrc, kernel, stridesKernel, res, stridesRes, resShape, kernelShape, rank);
}
void s_correlate_valid_uint64(const uint64_t *src, const int64_t *stridesSrc, const uint64_t *kernel, const int64_t *stridesKernel, uint64_t *res, const int64_t *stridesRes, const int64_t *resShape, const int64_t *kernelShape, int rank) {
    s_correlate_valid_generic<uint64_t>(src, stridesSrc, kernel, stridesKernel, res, stridesRes, resShape, kernelShape, rank);
}
}



/* ============================================================================
 * SECTION 12: GENERALIZED UFUNC REDUCTION, CUMULATIVE & SCATTER AT KERNELS
 * ============================================================================
 */

template <typename T, typename Op>
static inline T r_reduce_op_impl(const T *src, int64_t size, T default_empty, Op op) {
    if (src == nullptr || size <= 0) return default_empty;
    T acc = src[0];
    for (int64_t i = 1; i < size; i++) {
        acc = op(acc, src[i]);
    }
    return acc;
}

template <typename T, typename Op>
static inline void s_reduce_op_impl(
    const T *src, const int64_t *stridesSrc,
    T *dest, const int64_t *stridesDest,
    const int64_t *shape, int rank, int axis, Op op
) {
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return;
    int64_t size_axis = shape[axis];
    if (size_axis <= 0) return;
    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    int64_t offsetSrc = 0;
    int64_t offsetDest = 0;
    int64_t stride_axis = stridesSrc[axis];

    for (int64_t o = 0; o < outer_size; o++) {
        const T *axis_ptr = src + offsetSrc;
        T acc = axis_ptr[0];
        for (int64_t i = 1; i < size_axis; i++) {
            acc = op(acc, axis_ptr[i * stride_axis]);
        }
        dest[offsetDest] = acc;

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            int targetD = (d < axis) ? d : (d - 1);
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDest += stridesDest[targetD];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDest -= (shape[d] - 1) * stridesDest[targetD];
        }
    }
}

// Helpers for operators
#define OP_AND(x, y) ((x) & (y))
#define OP_OR(x, y)  ((x) | (y))
#define OP_XOR(x, y) ((x) ^ (y))
#define OP_LAND(x, y) ((x) && (y) ? 1 : 0)
#define OP_LOR(x, y)  ((x) || (y) ? 1 : 0)
#define OP_LXOR(x, y) (((x) != 0) != ((y) != 0) ? 1 : 0)

// Integer sums
int64_t r_sum_int64(const int64_t *src, int64_t size) { return r_reduce_op_impl(src, size, (int64_t)0, [](int64_t a, int64_t b) { return (int64_t)((uint64_t)a + (uint64_t)b); }); }
int32_t r_sum_int32(const int32_t *src, int64_t size) { return r_reduce_op_impl(src, size, (int32_t)0, [](int32_t a, int32_t b) { return (int32_t)((uint32_t)a + (uint32_t)b); }); }
uint8_t r_sum_uint8(const uint8_t *src, int64_t size) { return r_reduce_op_impl(src, size, (uint8_t)0, [](uint8_t a, uint8_t b) { return (uint8_t)(a + b); }); }
int16_t r_sum_int16(const int16_t *src, int64_t size) { return r_reduce_op_impl(src, size, (int16_t)0, [](int16_t a, int16_t b) { return (int16_t)((uint16_t)a + (uint16_t)b); }); }

void s_sum_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](int64_t a, int64_t b) { return (int64_t)((uint64_t)a + (uint64_t)b); });
}
void s_sum_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](int32_t a, int32_t b) { return (int32_t)((uint32_t)a + (uint32_t)b); });
}
void s_sum_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](uint8_t a, uint8_t b) { return (uint8_t)(a + b); });
}
void s_sum_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](int16_t a, int16_t b) { return (int16_t)((uint16_t)a + (uint16_t)b); });
}

// Integer and Complex Products
int64_t r_prod_int64(const int64_t *src, int64_t size) { return r_reduce_op_impl(src, size, (int64_t)1, [](int64_t a, int64_t b) { return (int64_t)((uint64_t)a * (uint64_t)b); }); }
int32_t r_prod_int32(const int32_t *src, int64_t size) { return r_reduce_op_impl(src, size, (int32_t)1, [](int32_t a, int32_t b) { return (int32_t)((uint32_t)a * (uint32_t)b); }); }
uint8_t r_prod_uint8(const uint8_t *src, int64_t size) { return r_reduce_op_impl(src, size, (uint8_t)1, [](uint8_t a, uint8_t b) { return (uint8_t)(a * b); }); }
int16_t r_prod_int16(const int16_t *src, int64_t size) { return r_reduce_op_impl(src, size, (int16_t)1, [](int16_t a, int16_t b) { return (int16_t)((uint16_t)a * (uint16_t)b); }); }
cpx_t r_prod_complex128(const cpx_t *src, int64_t size) { return r_reduce_op_impl(src, size, cpx_t{1.0, 0.0}, cpx_mul); }
cpx_f_t r_prod_complex64(const cpx_f_t *src, int64_t size) { return r_reduce_op_impl(src, size, cpx_f_t{1.0f, 0.0f}, cpx_mul_f); }

void s_prod_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](double a, double b) { return a * b; });
}
void s_prod_float(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](float a, float b) { return a * b; });
}
void s_prod_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](int64_t a, int64_t b) { return (int64_t)((uint64_t)a * (uint64_t)b); });
}
void s_prod_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](int32_t a, int32_t b) { return (int32_t)((uint32_t)a * (uint32_t)b); });
}
void s_prod_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](uint8_t a, uint8_t b) { return (uint8_t)(a * b); });
}
void s_prod_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](int16_t a, int16_t b) { return (int16_t)((uint16_t)a * (uint16_t)b); });
}
void s_prod_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, cpx_mul);
}
void s_prod_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, cpx_mul_f);
}

// Bitwise Reductions
int64_t r_bitwise_and_int64(const int64_t *src, int64_t size) { return r_reduce_op_impl(src, size, (int64_t)-1, [](int64_t a, int64_t b) { return a & b; }); }
int32_t r_bitwise_and_int32(const int32_t *src, int64_t size) { return r_reduce_op_impl(src, size, (int32_t)-1, [](int32_t a, int32_t b) { return a & b; }); }
uint8_t r_bitwise_and_uint8(const uint8_t *src, int64_t size) { return r_reduce_op_impl(src, size, (uint8_t)255, [](uint8_t a, uint8_t b) { return (uint8_t)(a & b); }); }
int16_t r_bitwise_and_int16(const int16_t *src, int64_t size) { return r_reduce_op_impl(src, size, (int16_t)-1, [](int16_t a, int16_t b) { return (int16_t)(a & b); }); }

int64_t r_bitwise_or_int64(const int64_t *src, int64_t size) { return r_reduce_op_impl(src, size, (int64_t)0, [](int64_t a, int64_t b) { return a | b; }); }
int32_t r_bitwise_or_int32(const int32_t *src, int64_t size) { return r_reduce_op_impl(src, size, (int32_t)0, [](int32_t a, int32_t b) { return a | b; }); }
uint8_t r_bitwise_or_uint8(const uint8_t *src, int64_t size) { return r_reduce_op_impl(src, size, (uint8_t)0, [](uint8_t a, uint8_t b) { return (uint8_t)(a | b); }); }
int16_t r_bitwise_or_int16(const int16_t *src, int64_t size) { return r_reduce_op_impl(src, size, (int16_t)0, [](int16_t a, int16_t b) { return (int16_t)(a | b); }); }

int64_t r_bitwise_xor_int64(const int64_t *src, int64_t size) { return r_reduce_op_impl(src, size, (int64_t)0, [](int64_t a, int64_t b) { return a ^ b; }); }
int32_t r_bitwise_xor_int32(const int32_t *src, int64_t size) { return r_reduce_op_impl(src, size, (int32_t)0, [](int32_t a, int32_t b) { return a ^ b; }); }
uint8_t r_bitwise_xor_uint8(const uint8_t *src, int64_t size) { return r_reduce_op_impl(src, size, (uint8_t)0, [](uint8_t a, uint8_t b) { return (uint8_t)(a ^ b); }); }
int16_t r_bitwise_xor_int16(const int16_t *src, int64_t size) { return r_reduce_op_impl(src, size, (int16_t)0, [](int16_t a, int16_t b) { return (int16_t)(a ^ b); }); }

void s_bitwise_and_red_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](int64_t a, int64_t b) { return a & b; });
}
void s_bitwise_and_red_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](int32_t a, int32_t b) { return a & b; });
}
void s_bitwise_and_red_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](uint8_t a, uint8_t b) { return (uint8_t)(a & b); });
}
void s_bitwise_and_red_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](int16_t a, int16_t b) { return (int16_t)(a & b); });
}

void s_bitwise_or_red_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](int64_t a, int64_t b) { return a | b; });
}
void s_bitwise_or_red_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](int32_t a, int32_t b) { return a | b; });
}
void s_bitwise_or_red_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](uint8_t a, uint8_t b) { return (uint8_t)(a | b); });
}
void s_bitwise_or_red_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](int16_t a, int16_t b) { return (int16_t)(a | b); });
}

void s_bitwise_xor_red_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](int64_t a, int64_t b) { return a ^ b; });
}
void s_bitwise_xor_red_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](int32_t a, int32_t b) { return a ^ b; });
}
void s_bitwise_xor_red_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](uint8_t a, uint8_t b) { return (uint8_t)(a ^ b); });
}
void s_bitwise_xor_red_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](int16_t a, int16_t b) { return (int16_t)(a ^ b); });
}

// Logical Reductions
uint8_t r_logical_and(const uint8_t *src, int64_t size) { return r_reduce_op_impl(src, size, (uint8_t)1, [](uint8_t a, uint8_t b) { return (uint8_t)(a && b ? 1 : 0); }); }
uint8_t r_logical_or(const uint8_t *src, int64_t size) { return r_reduce_op_impl(src, size, (uint8_t)0, [](uint8_t a, uint8_t b) { return (uint8_t)(a || b ? 1 : 0); }); }
uint8_t r_logical_xor(const uint8_t *src, int64_t size) { return r_reduce_op_impl(src, size, (uint8_t)0, [](uint8_t a, uint8_t b) { return (uint8_t)((a != 0) != (b != 0) ? 1 : 0); }); }

void s_logical_and_red(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](uint8_t a, uint8_t b) { return (uint8_t)(a && b ? 1 : 0); });
}
void s_logical_or_red(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](uint8_t a, uint8_t b) { return (uint8_t)(a || b ? 1 : 0); });
}
void s_logical_xor_red(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis) {
    s_reduce_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, [](uint8_t a, uint8_t b) { return (uint8_t)((a != 0) != (b != 0) ? 1 : 0); });
}

// Cumulative Bitwise & Logical Operations
DEFINE_STRIDED_CUM_OP(s_cumbitwise_and_int64, int64_t, OP_AND)
DEFINE_STRIDED_CUM_OP(s_cumbitwise_and_int32, int32_t, OP_AND)
DEFINE_STRIDED_CUM_OP(s_cumbitwise_and_uint8, uint8_t, OP_AND)
DEFINE_STRIDED_CUM_OP(s_cumbitwise_and_int16, int16_t, OP_AND)

DEFINE_STRIDED_CUM_OP(s_cumbitwise_or_int64, int64_t, OP_OR)
DEFINE_STRIDED_CUM_OP(s_cumbitwise_or_int32, int32_t, OP_OR)
DEFINE_STRIDED_CUM_OP(s_cumbitwise_or_uint8, uint8_t, OP_OR)
DEFINE_STRIDED_CUM_OP(s_cumbitwise_or_int16, int16_t, OP_OR)

DEFINE_STRIDED_CUM_OP(s_cumbitwise_xor_int64, int64_t, OP_XOR)
DEFINE_STRIDED_CUM_OP(s_cumbitwise_xor_int32, int32_t, OP_XOR)
DEFINE_STRIDED_CUM_OP(s_cumbitwise_xor_uint8, uint8_t, OP_XOR)
DEFINE_STRIDED_CUM_OP(s_cumbitwise_xor_int16, int16_t, OP_XOR)

DEFINE_STRIDED_CUM_OP(s_cumlogical_and, uint8_t, OP_LAND)
DEFINE_STRIDED_CUM_OP(s_cumlogical_or, uint8_t, OP_LOR)
DEFINE_STRIDED_CUM_OP(s_cumlogical_xor, uint8_t, OP_LXOR)

// Unbuffered Scatter Update Kernels (s_at_*)
template <typename T>
static inline T fdiv_int(T a, T b) {
    if (b == 0) {
        division_error_flag = 1;
        return 0;
    }
    if constexpr (std::is_signed_v<T>) {
        if (a == std::numeric_limits<T>::min() && b == static_cast<T>(-1)) {
            return std::numeric_limits<T>::min();
        }
    }
    return static_cast<T>(a / b);
}

template <typename T>
static inline T floordiv_int(T a, T b) {
    if (b == 0) {
        division_error_flag = 1;
        return 0;
    }
    if constexpr (std::is_signed_v<T>) {
        if (a == std::numeric_limits<T>::min() && b == static_cast<T>(-1)) {
            return std::numeric_limits<T>::min();
        }
        T res = static_cast<T>(a / b);
        T rem = static_cast<T>(a % b);
        if (rem != 0 && ((a < 0) ^ (b < 0))) {
            res--;
        }
        return res;
    } else {
        return static_cast<T>(a / b);
    }
}

template <typename T>
static inline T rem_int(T a, T b) {
    if (b == 0) {
        division_error_flag = 1;
        return 0;
    }
    if constexpr (std::is_signed_v<T>) {
        if (a == std::numeric_limits<T>::min() && b == static_cast<T>(-1)) return 0;
        T rem = static_cast<T>(a % b);
        if (rem != 0 && ((rem < 0) != (b < 0))) {
            rem = static_cast<T>(rem + b);
        }
        return rem;
    } else {
        return static_cast<T>(a % b);
    }
}

template <typename T>
static inline T fmod_int(T a, T b) {
    if (b == 0) {
        division_error_flag = 1;
        return 0;
    }
    if constexpr (std::is_signed_v<T>) {
        if (a == std::numeric_limits<T>::min() && b == static_cast<T>(-1)) return 0;
    }
    return static_cast<T>(a % b);
}

template <typename T>
static inline T apply_at_op(T a, T b, int opCode) {
    if constexpr (std::is_integral_v<T>) {
        switch (opCode) {
            case 0: return ndarray_wrapping_add(a, b);
            case 1: return ndarray_wrapping_mul(a, b);
            case 2:
            case 4: return (a < b) ? a : b;
            case 3:
            case 5: return (a > b) ? a : b;
            case 6: return saturating_float_to_int<T>(logaddexp_op((double)a, (double)b));
            case 7: return saturating_float_to_int<T>(logaddexp2_op((double)a, (double)b));
            case 8: return gcd_impl<T>(a, b);
            case 9: return lcm_impl<T>(a, b);
            case 10: return static_cast<T>(a & b);
            case 11: return static_cast<T>(a | b);
            case 12: return static_cast<T>(a ^ b);
            case 13: return (a != 0) && (b != 0) ? static_cast<T>(1) : static_cast<T>(0);
            case 14: return (a != 0) || (b != 0) ? static_cast<T>(1) : static_cast<T>(0);
            case 15: return ((a != 0) != (b != 0)) ? static_cast<T>(1) : static_cast<T>(0);
            case 16: return ndarray_wrapping_sub(a, b);
            case 17: return fdiv_int<T>(a, b);
            case 18: return floordiv_int<T>(a, b);
            case 19: return rem_int<T>(a, b);
            case 20: return fmod_int<T>(a, b);
            case 21:
            case 22: return ipow<T>(a, b);
            case 23: return saturating_float_to_int<T>(std::atan2((double)a, (double)b));
            case 24: return saturating_float_to_int<T>(std::hypot((double)a, (double)b));
            case 25: return saturating_float_to_int<T>(std::copysign((double)a, (double)b));
            case 26: return safe_left_shift_int<T>(a, b);
            case 27: return safe_right_shift_int<T>(a, b);
            case 28:
                if constexpr (std::is_signed_v<T>) {
                    return (a < 0) ? static_cast<T>(0) : (a > 0 ? static_cast<T>(1) : b);
                } else {
                    return (a > 0) ? static_cast<T>(1) : b;
                }
            default: return ndarray_wrapping_add(a, b);
        }
    } else {
        switch (opCode) {
            case 0: return a + b;
            case 1: return a * b;
            case 2: { // minimum
                double da = (double)a, db = (double)b;
                if (std::isnan(da) || std::isnan(db)) return static_cast<T>(std::numeric_limits<double>::quiet_NaN());
                if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                    if (da == 0.0 && db == 0.0) {
                        return (std::signbit(da) || std::signbit(db)) ? static_cast<T>(-0.0) : static_cast<T>(0.0);
                    }
                }
                return (a < b) ? a : b;
            }
            case 3: { // maximum
                double da = (double)a, db = (double)b;
                if (std::isnan(da) || std::isnan(db)) return static_cast<T>(std::numeric_limits<double>::quiet_NaN());
                if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                    if (da == 0.0 && db == 0.0) {
                        return (!std::signbit(da) || !std::signbit(db)) ? static_cast<T>(0.0) : static_cast<T>(-0.0);
                    }
                }
                return (a > b) ? a : b;
            }
            case 4: { // fmin
                double da = (double)a, db = (double)b;
                if (std::isnan(da) && std::isnan(db)) return a;
                if (std::isnan(da)) return b;
                if (std::isnan(db)) return a;
                if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                    if (da == 0.0 && db == 0.0) {
                        return (std::signbit(da) || std::signbit(db)) ? static_cast<T>(-0.0) : static_cast<T>(0.0);
                    }
                }
                return (a < b) ? a : b;
            }
            case 5: { // fmax
                double da = (double)a, db = (double)b;
                if (std::isnan(da) && std::isnan(db)) return a;
                if (std::isnan(da)) return b;
                if (std::isnan(db)) return a;
                if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                    if (da == 0.0 && db == 0.0) {
                        return (!std::signbit(da) || !std::signbit(db)) ? static_cast<T>(0.0) : static_cast<T>(-0.0);
                    }
                }
                return (a > b) ? a : b;
            }
            case 6: { double x = (double)a, y = (double)b; double mx = (x > y) ? x : y; return (T)(mx + std::log(std::exp(x - mx) + std::exp(y - mx))); }
            case 7: { double x = (double)a, y = (double)b; double mx = (x > y) ? x : y; return (T)(mx + std::log2(std::pow(2.0, x - mx) + std::pow(2.0, y - mx))); }
            case 16: return a - b;
            case 17: return a / b;
            case 18: return (T)std::floor((double)a / (double)b);
            case 19: return (T)std::fmod((double)a, (double)b);
            case 20: return (T)std::fmod((double)a, (double)b);
            case 21:
            case 22: return (T)std::pow((double)a, (double)b);
            case 23: return (T)std::atan2((double)a, (double)b);
            case 24: return (T)std::hypot((double)a, (double)b);
            case 25: return (T)std::copysign((double)a, (double)b);
            case 28: return ((double)a < 0.0) ? (T)0 : ((double)a == 0.0 ? b : (T)1);
            default: return a + b;
        }
    }
}

static inline uint8_t apply_at_op_boolean(uint8_t a, uint8_t b, int opCode) {
    uint8_t ba = (a != 0) ? 1 : 0;
    uint8_t bb = (b != 0) ? 1 : 0;
    switch (opCode) {
        case 0: return (ba || bb) ? 1 : 0;
        case 1: return (ba && bb) ? 1 : 0;
        case 2:
        case 4: return (ba < bb) ? ba : bb;
        case 3:
        case 5: return (ba > bb) ? ba : bb;
        case 10:
        case 13: return (ba && bb) ? 1 : 0;
        case 11:
        case 14: return (ba || bb) ? 1 : 0;
        case 12:
        case 15: return (ba != bb) ? 1 : 0;
        default: return apply_at_op<uint8_t>(ba, bb, opCode) != 0 ? 1 : 0;
    }
}

template <>
inline cpx_t apply_at_op<cpx_t>(cpx_t a, cpx_t b, int opCode) {
    switch (opCode) {
        case 0: return cpx_add(a, b);
        case 1: return cpx_mul(a, b);
        case 2: { // minimum
            bool a_has_nan = std::isnan(a.r) || std::isnan(a.i);
            bool b_has_nan = std::isnan(b.r) || std::isnan(b.i);
            if (a_has_nan) return a;
            if (b_has_nan) return b;
            bool a_less = (a.r < b.r) || (a.r == b.r && a.i < b.i);
            return a_less ? a : b;
        }
        case 3: { // maximum
            bool a_has_nan = std::isnan(a.r) || std::isnan(a.i);
            bool b_has_nan = std::isnan(b.r) || std::isnan(b.i);
            if (a_has_nan) return a;
            if (b_has_nan) return b;
            bool a_less = (a.r < b.r) || (a.r == b.r && a.i < b.i);
            return a_less ? b : a;
        }
        case 4: { // fmin
            bool a_has_nan = std::isnan(a.r) || std::isnan(a.i);
            bool b_has_nan = std::isnan(b.r) || std::isnan(b.i);
            if (a_has_nan && b_has_nan) return a;
            if (a_has_nan) return b;
            if (b_has_nan) return a;
            bool a_less = (a.r < b.r) || (a.r == b.r && a.i < b.i);
            return a_less ? a : b;
        }
        case 5: { // fmax
            bool a_has_nan = std::isnan(a.r) || std::isnan(a.i);
            bool b_has_nan = std::isnan(b.r) || std::isnan(b.i);
            if (a_has_nan && b_has_nan) return a;
            if (a_has_nan) return b;
            if (b_has_nan) return a;
            bool a_less = (a.r < b.r) || (a.r == b.r && a.i < b.i);
            return a_less ? b : a;
        }
        case 16: return cpx_sub(a, b);
        case 17: return cpx_div(a, b);
        default: return cpx_add(a, b);
    }
}

template <>
inline cpx_f_t apply_at_op<cpx_f_t>(cpx_f_t a, cpx_f_t b, int opCode) {
    switch (opCode) {
        case 0: return cpx_add_f(a, b);
        case 1: return cpx_mul_f(a, b);
        case 2: { // minimum
            bool a_has_nan = std::isnan(a.r) || std::isnan(a.i);
            bool b_has_nan = std::isnan(b.r) || std::isnan(b.i);
            if (a_has_nan) return a;
            if (b_has_nan) return b;
            bool a_less = (a.r < b.r) || (a.r == b.r && a.i < b.i);
            return a_less ? a : b;
        }
        case 3: { // maximum
            bool a_has_nan = std::isnan(a.r) || std::isnan(a.i);
            bool b_has_nan = std::isnan(b.r) || std::isnan(b.i);
            if (a_has_nan) return a;
            if (b_has_nan) return b;
            bool a_less = (a.r < b.r) || (a.r == b.r && a.i < b.i);
            return a_less ? b : a;
        }
        case 4: { // fmin
            bool a_has_nan = std::isnan(a.r) || std::isnan(a.i);
            bool b_has_nan = std::isnan(b.r) || std::isnan(b.i);
            if (a_has_nan && b_has_nan) return a;
            if (a_has_nan) return b;
            if (b_has_nan) return a;
            bool a_less = (a.r < b.r) || (a.r == b.r && a.i < b.i);
            return a_less ? a : b;
        }
        case 5: { // fmax
            bool a_has_nan = std::isnan(a.r) || std::isnan(a.i);
            bool b_has_nan = std::isnan(b.r) || std::isnan(b.i);
            if (a_has_nan && b_has_nan) return a;
            if (a_has_nan) return b;
            if (b_has_nan) return a;
            bool a_less = (a.r < b.r) || (a.r == b.r && a.i < b.i);
            return a_less ? b : a;
        }
        case 16: return (cpx_f_t){a.r - b.r, a.i - b.i};
        case 17: return cpx_div_f(a, b);
        default: return cpx_add_f(a, b);
    }
}

template <typename T, typename Op>
static void s_at_op_impl(T *a, const int64_t *stridesA, const int64_t *shapeA, int rankA,
                         const int64_t *indices, int64_t numIndices, int64_t strideIdx,
                         const T *b, const int64_t *stridesB, const int64_t *shapeB, int rankB,
                         Op op) {
    if (a == nullptr || indices == nullptr || b == nullptr || numIndices <= 0 || rankA <= 0) return;
    int64_t axis0_len = shapeA[0];
    int64_t sliceSize = 1;
    for (int d = 1; d < rankA; d++) sliceSize *= shapeA[d];

    for (int64_t i = 0; i < numIndices; i++) {
        int64_t idx = indices[i * strideIdx];
        if (idx < 0) idx += axis0_len;
        if (idx < 0 || idx >= axis0_len) continue;

        int64_t offsetA = idx * stridesA[0];
        int64_t offsetB = (rankB == rankA && shapeB != nullptr && shapeB[0] > 1) ? (i * stridesB[0]) : 0;

        if (sliceSize == 1 && rankA == 1) {
            a[offsetA] = op(a[offsetA], b[offsetB]);
        } else {
            NoThrowBuffer<int64_t> coord_vec;
            int64_t coord_stack[32] = {0};
            int64_t *coord = coord_stack;
            if (rankA > 32) {
                coord_vec.assign(rankA, 0);
                if (!coord_vec.ok()) {
                    ndarray_set_oom_flag();
                    return;
                }
                coord = coord_vec.data();
            }
            for (int64_t s = 0; s < sliceSize; s++) {
                int64_t offA = offsetA;
                int64_t offB = offsetB;
                for (int d = 1; d < rankA; d++) {
                    offA += coord[d - 1] * stridesA[d];
                    int dB = d - (rankA - rankB);
                    if (dB >= 0 && dB < rankB && (shapeB == nullptr || shapeB[dB] > 1)) {
                        offB += coord[d - 1] * stridesB[dB];
                    }
                }
                a[offA] = op(a[offA], b[offB]);

                for (int d = rankA - 2; d >= 0; d--) {
                    coord[d]++;
                    if (coord[d] < shapeA[d + 1]) break;
                    coord[d] = 0;
                }
            }
        }
    }
}

template <typename T>
static void s_at_impl(T *a, const int64_t *stridesA, const int64_t *shapeA, int rankA,
                      const int64_t *indices, int64_t numIndices, int64_t strideIdx,
                      const T *b, const int64_t *stridesB, const int64_t *shapeB, int rankB,
                      int opCode) {
    s_at_op_impl(a, stridesA, shapeA, rankA, indices, numIndices, strideIdx, b, stridesB, shapeB, rankB, [opCode](T x, T y) { return apply_at_op(x, y, opCode); });
}

void s_at_double(double *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const double *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode) {
    s_at_impl(a, stridesA, shapeA, rankA, indices, numIndices, strideIdx, b, stridesB, shapeB, rankB, opCode);
}
void s_at_float(float *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const float *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode) {
    s_at_impl(a, stridesA, shapeA, rankA, indices, numIndices, strideIdx, b, stridesB, shapeB, rankB, opCode);
}
void s_at_int64(int64_t *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const int64_t *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode) {
    s_at_impl(a, stridesA, shapeA, rankA, indices, numIndices, strideIdx, b, stridesB, shapeB, rankB, opCode);
}
void s_at_int32(int32_t *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const int32_t *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode) {
    s_at_impl(a, stridesA, shapeA, rankA, indices, numIndices, strideIdx, b, stridesB, shapeB, rankB, opCode);
}
void s_at_uint8(uint8_t *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const uint8_t *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode) {
    s_at_impl(a, stridesA, shapeA, rankA, indices, numIndices, strideIdx, b, stridesB, shapeB, rankB, opCode);
}
void s_at_int16(int16_t *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const int16_t *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode) {
    s_at_impl(a, stridesA, shapeA, rankA, indices, numIndices, strideIdx, b, stridesB, shapeB, rankB, opCode);
}
void s_at_complex128(cpx_t *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const cpx_t *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode) {
    s_at_impl(a, stridesA, shapeA, rankA, indices, numIndices, strideIdx, b, stridesB, shapeB, rankB, opCode);
}
void s_at_complex64(cpx_f_t *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const cpx_f_t *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode) {
    s_at_impl(a, stridesA, shapeA, rankA, indices, numIndices, strideIdx, b, stridesB, shapeB, rankB, opCode);
}
void s_at_boolean(uint8_t *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const uint8_t *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode) {
    s_at_op_impl(a, stridesA, shapeA, rankA, indices, numIndices, strideIdx, b, stridesB, shapeB, rankB, [opCode](uint8_t x, uint8_t y) { return apply_at_op_boolean(x, y, opCode); });
}



/* ============================================================================
 * SECTION 13: OPTIMIZED REDUCEAT SIMD & STRIDED KERNELS
 * ============================================================================
 */

template <typename T>
static inline T get_reduceat_identity(int opCode) {
    switch (opCode) {
        case 0: return (T)0;
        case 1: return (T)1;
        case 8: return (T)0;
        case 9: return (T)1;
        case 11: return (T)0;
        case 12: return (T)0;
        case 13: return (T)1;
        case 14: return (T)0;
        case 15: return (T)0;
        default: return (T)0;
    }
}

template <typename T, typename Op, bool IsScalar = std::is_arithmetic<T>::value>
struct v_reduceat_unroll_helper {
    static inline void run(const T *src, int64_t start, int64_t end, T &acc, [[maybe_unused]] int opCode, Op op) {
        int64_t len = end - start;
        int64_t j = start + 1;
        if (len > 16) {
            if (opCode == 0) { // add
                T id = get_reduceat_identity<T>(opCode);
                T acc0 = acc, acc1 = id, acc2 = id, acc3 = id;
                T acc4 = id, acc5 = id, acc6 = id, acc7 = id;
                int64_t rem = (end - j) % 8;
                int64_t vec_end = end - rem;
                for (; j < vec_end; j += 8) {
                    acc0 = apply_at_op(acc0, src[j], 0);
                    acc1 = apply_at_op(acc1, src[j + 1], 0);
                    acc2 = apply_at_op(acc2, src[j + 2], 0);
                    acc3 = apply_at_op(acc3, src[j + 3], 0);
                    acc4 = apply_at_op(acc4, src[j + 4], 0);
                    acc5 = apply_at_op(acc5, src[j + 5], 0);
                    acc6 = apply_at_op(acc6, src[j + 6], 0);
                    acc7 = apply_at_op(acc7, src[j + 7], 0);
                }
                T s1 = apply_at_op(acc0, acc1, 0);
                T s2 = apply_at_op(acc2, acc3, 0);
                T s3 = apply_at_op(acc4, acc5, 0);
                T s4 = apply_at_op(acc6, acc7, 0);
                acc = apply_at_op(apply_at_op(s1, s2, 0), apply_at_op(s3, s4, 0), 0);
            } else if (opCode == 1) { // multiply
                T id = get_reduceat_identity<T>(opCode);
                T acc0 = acc, acc1 = id, acc2 = id, acc3 = id;
                T acc4 = id, acc5 = id, acc6 = id, acc7 = id;
                int64_t rem = (end - j) % 8;
                int64_t vec_end = end - rem;
                for (; j < vec_end; j += 8) {
                    acc0 = apply_at_op(acc0, src[j], 1);
                    acc1 = apply_at_op(acc1, src[j + 1], 1);
                    acc2 = apply_at_op(acc2, src[j + 2], 1);
                    acc3 = apply_at_op(acc3, src[j + 3], 1);
                    acc4 = apply_at_op(acc4, src[j + 4], 1);
                    acc5 = apply_at_op(acc5, src[j + 5], 1);
                    acc6 = apply_at_op(acc6, src[j + 6], 1);
                    acc7 = apply_at_op(acc7, src[j + 7], 1);
                }
                T p1 = apply_at_op(acc0, acc1, 1);
                T p2 = apply_at_op(acc2, acc3, 1);
                T p3 = apply_at_op(acc4, acc5, 1);
                T p4 = apply_at_op(acc6, acc7, 1);
                acc = apply_at_op(apply_at_op(p1, p2, 1), apply_at_op(p3, p4, 1), 1);
            } else if (opCode == 2 || opCode == 3 || opCode == 4 || opCode == 5) { // minimum, maximum, fmin, fmax
                T acc0 = acc, acc1 = acc, acc2 = acc, acc3 = acc;
                T acc4 = acc, acc5 = acc, acc6 = acc, acc7 = acc;
                int64_t rem = (end - j) % 8;
                int64_t vec_end = end - rem;
                for (; j < vec_end; j += 8) {
                    acc0 = apply_at_op(acc0, src[j], opCode);
                    acc1 = apply_at_op(acc1, src[j + 1], opCode);
                    acc2 = apply_at_op(acc2, src[j + 2], opCode);
                    acc3 = apply_at_op(acc3, src[j + 3], opCode);
                    acc4 = apply_at_op(acc4, src[j + 4], opCode);
                    acc5 = apply_at_op(acc5, src[j + 5], opCode);
                    acc6 = apply_at_op(acc6, src[j + 6], opCode);
                    acc7 = apply_at_op(acc7, src[j + 7], opCode);
                }
                T m1 = apply_at_op(acc0, acc1, opCode);
                T m2 = apply_at_op(acc2, acc3, opCode);
                T m3 = apply_at_op(acc4, acc5, opCode);
                T m4 = apply_at_op(acc6, acc7, opCode);
                T m5 = apply_at_op(m1, m2, opCode);
                T m6 = apply_at_op(m3, m4, opCode);
                acc = apply_at_op(m5, m6, opCode);
            }
        }
        for (; j < end; j++) {
            acc = op(acc, src[j]);
        }
    }
};

template <typename T, typename Op>
struct v_reduceat_unroll_helper<T, Op, false> {
    static inline void run(const T *src, int64_t start, int64_t end, T &acc, [[maybe_unused]] int opCode, Op op) {
        for (int64_t j = start + 1; j < end; j++) {
            acc = op(acc, src[j]);
        }
    }
};

template <typename T, typename Op>
static inline void v_reduceat_op_impl(
    const T *src, int64_t size,
    const int64_t *indices, int64_t num_indices,
    T *dest, Op op, int opCode
) {
    if (src == nullptr || indices == nullptr || dest == nullptr || size <= 0 || num_indices <= 0) return;

    NoThrowBuffer<T> tmp_dest;
    T *orig_dest = dest;
    uintptr_t src_start = reinterpret_cast<uintptr_t>(src);
    uintptr_t src_end = reinterpret_cast<uintptr_t>(src + size);
    uintptr_t dst_start = reinterpret_cast<uintptr_t>(dest);
    uintptr_t dst_end = reinterpret_cast<uintptr_t>(dest + num_indices);
    if (src_start < dst_end && dst_start < src_end) {
        tmp_dest.allocate(num_indices);
        if (!tmp_dest.data()) return;
        dest = tmp_dest.data();
    }

    for (int64_t i = 0; i < num_indices; i++) {
        int64_t start = indices[i];
        if (start < 0) start += size;
        int64_t end = (i < num_indices - 1) ? indices[i + 1] : size;
        if (end < 0) end += size;

        if (start < 0) start = 0;
        if (start >= size) start = size - 1;
        end = std::min(end, size);

        if (start >= end) {
            dest[i] = src[start];
        } else {
            T acc = src[start];
            v_reduceat_unroll_helper<T, Op>::run(src, start, end, acc, opCode, op);
            dest[i] = acc;
        }
    }

    if (tmp_dest.data() != nullptr) {
        std::memcpy(orig_dest, tmp_dest.data(), num_indices * sizeof(T));
    }
}

template <typename T, typename Op>
static inline void s_reduceat_op_impl(
    const T *src, const int64_t *stridesSrc,
    T *dest, const int64_t *stridesDest,
    const int64_t *shape, int rank, int axis,
    const int64_t *indices, int64_t numIndices,
    Op op, [[maybe_unused]] int opCode
) {
    if (src == nullptr || dest == nullptr || shape == nullptr || indices == nullptr || rank <= 0 || axis < 0 || axis >= rank || numIndices <= 0) return;

    int64_t axis_len = shape[axis];
    if (axis_len <= 0) return;
    int64_t outer_size = 1;
    for (int d = 0; d < axis; d++) {
        if (shape[d] <= 0) return;
        outer_size *= shape[d];
    }
    int64_t inner_size = 1;
    for (int d = axis + 1; d < rank; d++) {
        if (shape[d] <= 0) return;
        inner_size *= shape[d];
    }

    DECLARE_RANK_BUFFER(int64_t, destShape, rank);
    for (int d = 0; d < rank; d++) {
        destShape[d] = (d == axis) ? numIndices : shape[d];
    }
    int64_t destTotal = outer_size * numIndices * inner_size;
    NoThrowBuffer<T> tmp_dest;
    T *orig_dest = dest;
    const int64_t *orig_strides_dest = stridesDest;
    DECLARE_RANK_BUFFER(int64_t, tmpStridesDest, rank);
    if (strided_buffers_any_overlap(src, stridesSrc, shape, rank, sizeof(T), dest, stridesDest, destShape, rank, sizeof(T))) {
        tmp_dest.allocate(destTotal);
        if (!tmp_dest.data()) return;
        dest = tmp_dest.data();
        int64_t st = 1;
        for (int d = rank - 1; d >= 0; d--) {
            tmpStridesDest[d] = st;
            st *= destShape[d];
        }
        stridesDest = tmpStridesDest;
    }

    bool is_inner_contiguous = true;
    int64_t cur_stride = 1;
    for (int d = rank - 1; d > axis; d--) {
        if (stridesSrc[d] != cur_stride || stridesDest[d] != cur_stride) {
            is_inner_contiguous = false;
            break;
        }
        cur_stride *= shape[d];
    }

    for (int64_t o = 0; o < outer_size; o++) {
        int64_t temp_o = o;
        int64_t src_outer_off = 0;
        int64_t dest_outer_off = 0;
        for (int d = axis - 1; d >= 0; d--) {
            int64_t idx = temp_o % shape[d];
            temp_o /= shape[d];
            src_outer_off += idx * stridesSrc[d];
            dest_outer_off += idx * stridesDest[d];
        }

        for (int64_t i = 0; i < numIndices; i++) {
            int64_t start = indices[i];
            if (start < 0) start += axis_len;
            int64_t end = (i < numIndices - 1) ? indices[i + 1] : axis_len;
            if (end < 0) end += axis_len;

            if (start < 0) start = 0;
            if (start >= axis_len) start = axis_len - 1;
            end = std::min<int64_t>(end, axis_len);

            int64_t dest_axis_off = i * stridesDest[axis];
            int64_t src_start_off = src_outer_off + start * stridesSrc[axis];
            int64_t dest_base = dest_outer_off + dest_axis_off;

            if (start >= end) {
                if (is_inner_contiguous) {
                    std::memcpy(dest + dest_base, src + src_start_off, inner_size * sizeof(T));
                } else {
                    for (int64_t in = 0; in < inner_size; in++) {
                        int64_t temp_in = in;
                        int64_t src_in_off = 0;
                        int64_t dest_in_off = 0;
                        for (int d = rank - 1; d > axis; d--) {
                            int64_t idx = temp_in % shape[d];
                            temp_in /= shape[d];
                            src_in_off += idx * stridesSrc[d];
                            dest_in_off += idx * stridesDest[d];
                        }
                        dest[dest_base + dest_in_off] = src[src_start_off + src_in_off];
                    }
                }
            } else {
                if (is_inner_contiguous) {
                    std::memcpy(dest + dest_base, src + src_start_off, inner_size * sizeof(T));

                    for (int64_t j = start + 1; j < end; j++) {
                        int64_t src_j_off = src_outer_off + j * stridesSrc[axis];
                                        for (int64_t in = 0; in < inner_size; in++) {
                            dest[dest_base + in] = op(dest[dest_base + in], src[src_j_off + in]);
                        }
                    }
                } else {
                    for (int64_t in = 0; in < inner_size; in++) {
                        int64_t temp_in = in;
                        int64_t src_in_off = 0;
                        int64_t dest_in_off = 0;
                        for (int d = rank - 1; d > axis; d--) {
                            int64_t idx = temp_in % shape[d];
                            temp_in /= shape[d];
                            src_in_off += idx * stridesSrc[d];
                            dest_in_off += idx * stridesDest[d];
                        }
                        dest[dest_base + dest_in_off] = src[src_start_off + src_in_off];
                    }

                    for (int64_t j = start + 1; j < end; j++) {
                        int64_t src_j_off = src_outer_off + j * stridesSrc[axis];
                        for (int64_t in = 0; in < inner_size; in++) {
                            int64_t temp_in = in;
                            int64_t src_in_off = 0;
                            int64_t dest_in_off = 0;
                            for (int d = rank - 1; d > axis; d--) {
                                int64_t idx = temp_in % shape[d];
                                temp_in /= shape[d];
                                src_in_off += idx * stridesSrc[d];
                                dest_in_off += idx * stridesDest[d];
                            }
                            int64_t d_idx = dest_base + dest_in_off;
                            dest[d_idx] = op(dest[d_idx], src[src_j_off + src_in_off]);
                        }
                    }
                }
            }
        }
    }

    if (tmp_dest.data() != nullptr) {
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offDest = 0;
        for (int64_t idx = 0; idx < destTotal; idx++) {
            orig_dest[offDest] = tmp_dest.data()[idx];
            for (int d = rank - 1; d >= 0; d--) {
                coord[d]++;
                offDest += orig_strides_dest[d];
                if (coord[d] < destShape[d]) break;
                offDest -= destShape[d] * orig_strides_dest[d];
                coord[d] = 0;
            }
        }
    }
}

template <typename T>
static void v_reduceat_generic(const T *src, int64_t size, const int64_t *indices, int64_t numIndices, T *dest, int opCode) {
    v_reduceat_op_impl(src, size, indices, numIndices, dest, [opCode](T a, T b) { return apply_at_op(a, b, opCode); }, opCode);
}

template <typename T>
static void s_reduceat_generic(const T *src, const int64_t *stridesSrc, T *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode) {
    s_reduceat_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, indices, numIndices, [opCode](T a, T b) { return apply_at_op(a, b, opCode); }, opCode);
}

void v_reduceat_double(const double *src, int64_t size, const int64_t *indices, int64_t numIndices, double *dest, int opCode) {
    v_reduceat_generic(src, size, indices, numIndices, dest, opCode);
}
void v_reduceat_float(const float *src, int64_t size, const int64_t *indices, int64_t numIndices, float *dest, int opCode) {
    v_reduceat_generic(src, size, indices, numIndices, dest, opCode);
}
void v_reduceat_int64(const int64_t *src, int64_t size, const int64_t *indices, int64_t numIndices, int64_t *dest, int opCode) {
    v_reduceat_generic(src, size, indices, numIndices, dest, opCode);
}
void v_reduceat_int32(const int32_t *src, int64_t size, const int64_t *indices, int64_t numIndices, int32_t *dest, int opCode) {
    v_reduceat_generic(src, size, indices, numIndices, dest, opCode);
}
void v_reduceat_uint8(const uint8_t *src, int64_t size, const int64_t *indices, int64_t numIndices, uint8_t *dest, int opCode) {
    v_reduceat_generic(src, size, indices, numIndices, dest, opCode);
}
void v_reduceat_int16(const int16_t *src, int64_t size, const int64_t *indices, int64_t numIndices, int16_t *dest, int opCode) {
    v_reduceat_generic(src, size, indices, numIndices, dest, opCode);
}
void v_reduceat_complex128(const cpx_t *src, int64_t size, const int64_t *indices, int64_t numIndices, cpx_t *dest, int opCode) {
    v_reduceat_generic(src, size, indices, numIndices, dest, opCode);
}
void v_reduceat_complex64(const cpx_f_t *src, int64_t size, const int64_t *indices, int64_t numIndices, cpx_f_t *dest, int opCode) {
    v_reduceat_generic(src, size, indices, numIndices, dest, opCode);
}
void v_reduceat_boolean(const uint8_t *src, int64_t size, const int64_t *indices, int64_t numIndices, uint8_t *dest, int opCode) {
    int unrollOp = (opCode == 0) ? 14 : ((opCode == 1) ? 13 : opCode);
    v_reduceat_op_impl(src, size, indices, numIndices, dest, [opCode](uint8_t a, uint8_t b) { return apply_at_op_boolean(a, b, opCode); }, unrollOp);
}

void s_reduceat_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode) {
    s_reduceat_generic(src, stridesSrc, dest, stridesDest, shape, rank, axis, indices, numIndices, opCode);
}
void s_reduceat_float(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode) {
    s_reduceat_generic(src, stridesSrc, dest, stridesDest, shape, rank, axis, indices, numIndices, opCode);
}
void s_reduceat_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode) {
    s_reduceat_generic(src, stridesSrc, dest, stridesDest, shape, rank, axis, indices, numIndices, opCode);
}
void s_reduceat_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode) {
    s_reduceat_generic(src, stridesSrc, dest, stridesDest, shape, rank, axis, indices, numIndices, opCode);
}
void s_reduceat_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode) {
    s_reduceat_generic(src, stridesSrc, dest, stridesDest, shape, rank, axis, indices, numIndices, opCode);
}
void s_reduceat_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode) {
    s_reduceat_generic(src, stridesSrc, dest, stridesDest, shape, rank, axis, indices, numIndices, opCode);
}
void s_reduceat_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode) {
    s_reduceat_generic(src, stridesSrc, dest, stridesDest, shape, rank, axis, indices, numIndices, opCode);
}
void s_reduceat_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode) {
    s_reduceat_generic(src, stridesSrc, dest, stridesDest, shape, rank, axis, indices, numIndices, opCode);
}
void s_reduceat_boolean(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode) {
    s_reduceat_op_impl(src, stridesSrc, dest, stridesDest, shape, rank, axis, indices, numIndices, [opCode](uint8_t a, uint8_t b) { return apply_at_op_boolean(a, b, opCode); }, opCode);
}

/* ============================================================================
 * SECTION 30: POLYNOMIAL & ORTHOGONAL SERIES EVALUATION (CLENSHAW RECURRENCE)
 * ============================================================================
 */

static inline cpx_t cpx_add_val(cpx_t a, cpx_t b) {
    return {a.r + b.r, a.i + b.i};
}
static inline cpx_t cpx_sub_val(cpx_t a, cpx_t b) {
    return {a.r - b.r, a.i - b.i};
}
static inline cpx_t cpx_mul_val(cpx_t a, cpx_t b) {
    return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}
static inline cpx_t cpx_mul_scalar(cpx_t a, double s) {
    return {a.r * s, a.i * s};
}
static inline bool cpx_isnan(cpx_t a) {
    return std::isnan(a.r) || std::isnan(a.i);
}

static inline cpx_f_t cpx_f_add_val(cpx_f_t a, cpx_f_t b) {
    return {a.r + b.r, a.i + b.i};
}
static inline cpx_f_t cpx_f_sub_val(cpx_f_t a, cpx_f_t b) {
    return {a.r - b.r, a.i - b.i};
}
static inline cpx_f_t cpx_f_mul_val(cpx_f_t a, cpx_f_t b) {
    return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}
static inline cpx_f_t cpx_f_mul_scalar(cpx_f_t a, float s) {
    return {a.r * s, a.i * s};
}
static inline bool cpx_f_isnan(cpx_f_t a) {
    return std::isnan(a.r) || std::isnan(a.i);
}

// ----------------------------------------------------------------------------
// Scalar Evaluators
// ----------------------------------------------------------------------------

static inline double eval_poly_double(const double *c, int64_t stride_c, int64_t n_c, double x) {
    if (n_c <= 0) return 0.0;
    if (n_c == 1) return std::isnan(x) ? NAN : c[0];
    double val = c[0];
    for (int64_t k = 1; k < n_c; k++) {
        val = val * x + c[k * stride_c];
    }
    return val;
}

static inline float eval_poly_float(const float *c, int64_t stride_c, int64_t n_c, float x) {
    if (n_c <= 0) return 0.0f;
    if (n_c == 1) return std::isnan(x) ? NAN : c[0];
    float val = c[0];
    for (int64_t k = 1; k < n_c; k++) {
        val = val * x + c[k * stride_c];
    }
    return val;
}

static inline cpx_t eval_poly_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, cpx_t x) {
    if (n_c <= 0) return {0.0, 0.0};
    if (n_c == 1) return cpx_isnan(x) ? (cpx_t){NAN, NAN} : c[0];
    cpx_t val = c[0];
    for (int64_t k = 1; k < n_c; k++) {
        val = cpx_add_val(cpx_mul_val(val, x), c[k * stride_c]);
    }
    return val;
}

static inline cpx_f_t eval_poly_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, cpx_f_t x) {
    if (n_c <= 0) return {0.0f, 0.0f};
    if (n_c == 1) return cpx_f_isnan(x) ? (cpx_f_t){NAN, NAN} : c[0];
    cpx_f_t val = c[0];
    for (int64_t k = 1; k < n_c; k++) {
        val = cpx_f_add_val(cpx_f_mul_val(val, x), c[k * stride_c]);
    }
    return val;
}

static inline double eval_cheb_double(const double *c, int64_t stride_c, int64_t n_c, double x) {
    if (n_c <= 0) return 0.0;
    if (n_c == 1) return std::isnan(x) ? NAN : c[0];
    if (n_c == 2) return c[0] + c[stride_c] * x;
    double x2 = 2.0 * x;
    double c0 = c[(n_c - 2) * stride_c];
    double c1 = c[(n_c - 1) * stride_c];
    for (int64_t k = 3; k <= n_c; k++) {
        double tmp = c0;
        c0 = c[(n_c - k) * stride_c] - c1;
        c1 = tmp + c1 * x2;
    }
    return c0 + c1 * x;
}

static inline float eval_cheb_float(const float *c, int64_t stride_c, int64_t n_c, float x) {
    if (n_c <= 0) return 0.0f;
    if (n_c == 1) return std::isnan(x) ? NAN : c[0];
    if (n_c == 2) return c[0] + c[stride_c] * x;
    float x2 = 2.0f * x;
    float c0 = c[(n_c - 2) * stride_c];
    float c1 = c[(n_c - 1) * stride_c];
    for (int64_t k = 3; k <= n_c; k++) {
        float tmp = c0;
        c0 = c[(n_c - k) * stride_c] - c1;
        c1 = tmp + c1 * x2;
    }
    return c0 + c1 * x;
}

static inline cpx_t eval_cheb_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, cpx_t x) {
    if (n_c <= 0) return {0.0, 0.0};
    if (n_c == 1) return cpx_isnan(x) ? (cpx_t){NAN, NAN} : c[0];
    if (n_c == 2) return cpx_add_val(c[0], cpx_mul_val(c[stride_c], x));
    cpx_t x2 = cpx_mul_scalar(x, 2.0);
    cpx_t c0 = c[(n_c - 2) * stride_c];
    cpx_t c1 = c[(n_c - 1) * stride_c];
    for (int64_t k = 3; k <= n_c; k++) {
        cpx_t tmp = c0;
        c0 = cpx_sub_val(c[(n_c - k) * stride_c], c1);
        c1 = cpx_add_val(tmp, cpx_mul_val(c1, x2));
    }
    return cpx_add_val(c0, cpx_mul_val(c1, x));
}

static inline cpx_f_t eval_cheb_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, cpx_f_t x) {
    if (n_c <= 0) return {0.0f, 0.0f};
    if (n_c == 1) return cpx_f_isnan(x) ? (cpx_f_t){NAN, NAN} : c[0];
    if (n_c == 2) return cpx_f_add_val(c[0], cpx_f_mul_val(c[stride_c], x));
    cpx_f_t x2 = cpx_f_mul_scalar(x, 2.0f);
    cpx_f_t c0 = c[(n_c - 2) * stride_c];
    cpx_f_t c1 = c[(n_c - 1) * stride_c];
    for (int64_t k = 3; k <= n_c; k++) {
        cpx_f_t tmp = c0;
        c0 = cpx_f_sub_val(c[(n_c - k) * stride_c], c1);
        c1 = cpx_f_add_val(tmp, cpx_f_mul_val(c1, x2));
    }
    return cpx_f_add_val(c0, cpx_f_mul_val(c1, x));
}

static inline double eval_leg_double(const double *c, int64_t stride_c, int64_t n_c, double x) {
    if (n_c <= 0) return 0.0;
    if (n_c == 1) return std::isnan(x) ? NAN : c[0];
    if (n_c == 2) return c[0] + c[stride_c] * x;
    double c0 = c[(n_c - 2) * stride_c];
    double c1 = c[(n_c - 1) * stride_c];
    int64_t nd = n_c;
    for (int64_t k = 3; k <= n_c; k++) {
        double tmp = c0;
        nd--;
        double f0 = (double)(nd - 1) / (double)nd;
        double f1 = (double)(2 * nd - 1) / (double)nd;
        c0 = c[(n_c - k) * stride_c] - c1 * f0;
        c1 = tmp + c1 * x * f1;
    }
    return c0 + c1 * x;
}

static inline float eval_leg_float(const float *c, int64_t stride_c, int64_t n_c, float x) {
    if (n_c <= 0) return 0.0f;
    if (n_c == 1) return std::isnan(x) ? NAN : c[0];
    if (n_c == 2) return c[0] + c[stride_c] * x;
    float c0 = c[(n_c - 2) * stride_c];
    float c1 = c[(n_c - 1) * stride_c];
    int64_t nd = n_c;
    for (int64_t k = 3; k <= n_c; k++) {
        float tmp = c0;
        nd--;
        float f0 = (float)(nd - 1) / (float)nd;
        float f1 = (float)(2 * nd - 1) / (float)nd;
        c0 = c[(n_c - k) * stride_c] - c1 * f0;
        c1 = tmp + c1 * x * f1;
    }
    return c0 + c1 * x;
}

static inline cpx_t eval_leg_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, cpx_t x) {
    if (n_c <= 0) return {0.0, 0.0};
    if (n_c == 1) return cpx_isnan(x) ? (cpx_t){NAN, NAN} : c[0];
    if (n_c == 2) return cpx_add_val(c[0], cpx_mul_val(c[stride_c], x));
    cpx_t c0 = c[(n_c - 2) * stride_c];
    cpx_t c1 = c[(n_c - 1) * stride_c];
    int64_t nd = n_c;
    for (int64_t k = 3; k <= n_c; k++) {
        cpx_t tmp = c0;
        nd--;
        double f0 = (double)(nd - 1) / (double)nd;
        double f1 = (double)(2 * nd - 1) / (double)nd;
        c0 = cpx_sub_val(c[(n_c - k) * stride_c], cpx_mul_scalar(c1, f0));
        c1 = cpx_add_val(tmp, cpx_mul_scalar(cpx_mul_val(c1, x), f1));
    }
    return cpx_add_val(c0, cpx_mul_val(c1, x));
}

static inline cpx_f_t eval_leg_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, cpx_f_t x) {
    if (n_c <= 0) return {0.0f, 0.0f};
    if (n_c == 1) return cpx_f_isnan(x) ? (cpx_f_t){NAN, NAN} : c[0];
    if (n_c == 2) return cpx_f_add_val(c[0], cpx_f_mul_val(c[stride_c], x));
    cpx_f_t c0 = c[(n_c - 2) * stride_c];
    cpx_f_t c1 = c[(n_c - 1) * stride_c];
    int64_t nd = n_c;
    for (int64_t k = 3; k <= n_c; k++) {
        cpx_f_t tmp = c0;
        nd--;
        float f0 = (float)(nd - 1) / (float)nd;
        float f1 = (float)(2 * nd - 1) / (float)nd;
        c0 = cpx_f_sub_val(c[(n_c - k) * stride_c], cpx_f_mul_scalar(c1, f0));
        c1 = cpx_f_add_val(tmp, cpx_f_mul_scalar(cpx_f_mul_val(c1, x), f1));
    }
    return cpx_f_add_val(c0, cpx_f_mul_val(c1, x));
}

static inline double eval_herm_double(const double *c, int64_t stride_c, int64_t n_c, double x) {
    if (n_c <= 0) return 0.0;
    if (n_c == 1) return std::isnan(x) ? NAN : c[0];
    double x2 = 2.0 * x;
    if (n_c == 2) return c[0] + c[stride_c] * x2;
    double c0 = c[(n_c - 2) * stride_c];
    double c1 = c[(n_c - 1) * stride_c];
    int64_t nd = n_c;
    for (int64_t k = 3; k <= n_c; k++) {
        double tmp = c0;
        nd--;
        double f0 = 2.0 * (double)(nd - 1);
        c0 = c[(n_c - k) * stride_c] - c1 * f0;
        c1 = tmp + c1 * x2;
    }
    return c0 + c1 * x2;
}

static inline float eval_herm_float(const float *c, int64_t stride_c, int64_t n_c, float x) {
    if (n_c <= 0) return 0.0f;
    if (n_c == 1) return std::isnan(x) ? NAN : c[0];
    float x2 = 2.0f * x;
    if (n_c == 2) return c[0] + c[stride_c] * x2;
    float c0 = c[(n_c - 2) * stride_c];
    float c1 = c[(n_c - 1) * stride_c];
    int64_t nd = n_c;
    for (int64_t k = 3; k <= n_c; k++) {
        float tmp = c0;
        nd--;
        float f0 = 2.0f * (float)(nd - 1);
        c0 = c[(n_c - k) * stride_c] - c1 * f0;
        c1 = tmp + c1 * x2;
    }
    return c0 + c1 * x2;
}

static inline cpx_t eval_herm_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, cpx_t x) {
    if (n_c <= 0) return {0.0, 0.0};
    if (n_c == 1) return cpx_isnan(x) ? (cpx_t){NAN, NAN} : c[0];
    cpx_t x2 = cpx_mul_scalar(x, 2.0);
    if (n_c == 2) return cpx_add_val(c[0], cpx_mul_val(c[stride_c], x2));
    cpx_t c0 = c[(n_c - 2) * stride_c];
    cpx_t c1 = c[(n_c - 1) * stride_c];
    int64_t nd = n_c;
    for (int64_t k = 3; k <= n_c; k++) {
        cpx_t tmp = c0;
        nd--;
        double f0 = 2.0 * (double)(nd - 1);
        c0 = cpx_sub_val(c[(n_c - k) * stride_c], cpx_mul_scalar(c1, f0));
        c1 = cpx_add_val(tmp, cpx_mul_val(c1, x2));
    }
    return cpx_add_val(c0, cpx_mul_val(c1, x2));
}

static inline cpx_f_t eval_herm_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, cpx_f_t x) {
    if (n_c <= 0) return {0.0f, 0.0f};
    if (n_c == 1) return cpx_f_isnan(x) ? (cpx_f_t){NAN, NAN} : c[0];
    cpx_f_t x2 = cpx_f_mul_scalar(x, 2.0f);
    if (n_c == 2) return cpx_f_add_val(c[0], cpx_f_mul_val(c[stride_c], x2));
    cpx_f_t c0 = c[(n_c - 2) * stride_c];
    cpx_f_t c1 = c[(n_c - 1) * stride_c];
    int64_t nd = n_c;
    for (int64_t k = 3; k <= n_c; k++) {
        cpx_f_t tmp = c0;
        nd--;
        float f0 = 2.0f * (float)(nd - 1);
        c0 = cpx_f_sub_val(c[(n_c - k) * stride_c], cpx_f_mul_scalar(c1, f0));
        c1 = cpx_f_add_val(tmp, cpx_f_mul_val(c1, x2));
    }
    return cpx_f_add_val(c0, cpx_f_mul_val(c1, x2));
}

static inline double eval_lag_double(const double *c, int64_t stride_c, int64_t n_c, double x) {
    if (n_c <= 0) return 0.0;
    if (n_c == 1) return std::isnan(x) ? NAN : c[0];
    if (n_c == 2) return c[0] + c[stride_c] * (1.0 - x);
    double c0 = c[(n_c - 2) * stride_c];
    double c1 = c[(n_c - 1) * stride_c];
    int64_t nd = n_c;
    for (int64_t k = 3; k <= n_c; k++) {
        double tmp = c0;
        nd--;
        double f0 = (double)(nd - 1) / (double)nd;
        c0 = c[(n_c - k) * stride_c] - c1 * f0;
        c1 = tmp + (c1 * ((2.0 * nd - 1.0) - x)) / (double)nd;
    }
    return c0 + c1 * (1.0 - x);
}

static inline float eval_lag_float(const float *c, int64_t stride_c, int64_t n_c, float x) {
    if (n_c <= 0) return 0.0f;
    if (n_c == 1) return std::isnan(x) ? NAN : c[0];
    if (n_c == 2) return c[0] + c[stride_c] * (1.0f - x);
    float c0 = c[(n_c - 2) * stride_c];
    float c1 = c[(n_c - 1) * stride_c];
    int64_t nd = n_c;
    for (int64_t k = 3; k <= n_c; k++) {
        float tmp = c0;
        nd--;
        float f0 = (float)(nd - 1) / (float)nd;
        c0 = c[(n_c - k) * stride_c] - c1 * f0;
        c1 = tmp + (c1 * ((2.0f * nd - 1.0f) - x)) / (float)nd;
    }
    return c0 + c1 * (1.0f - x);
}

static inline cpx_t eval_lag_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, cpx_t x) {
    if (n_c <= 0) return {0.0, 0.0};
    if (n_c == 1) return cpx_isnan(x) ? (cpx_t){NAN, NAN} : c[0];
    cpx_t omx = {1.0 - x.r, -x.i};
    if (n_c == 2) return cpx_add_val(c[0], cpx_mul_val(c[stride_c], omx));
    cpx_t c0 = c[(n_c - 2) * stride_c];
    cpx_t c1 = c[(n_c - 1) * stride_c];
    int64_t nd = n_c;
    for (int64_t k = 3; k <= n_c; k++) {
        cpx_t tmp = c0;
        nd--;
        double f0 = (double)(nd - 1) / (double)nd;
        cpx_t term = {(2.0 * nd - 1.0 - x.r) / (double)nd, -x.i / (double)nd};
        c0 = cpx_sub_val(c[(n_c - k) * stride_c], cpx_mul_scalar(c1, f0));
        c1 = cpx_add_val(tmp, cpx_mul_val(c1, term));
    }
    return cpx_add_val(c0, cpx_mul_val(c1, omx));
}

static inline cpx_f_t eval_lag_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, cpx_f_t x) {
    if (n_c <= 0) return {0.0f, 0.0f};
    if (n_c == 1) return cpx_f_isnan(x) ? (cpx_f_t){NAN, NAN} : c[0];
    cpx_f_t omx = {1.0f - x.r, -x.i};
    if (n_c == 2) return cpx_f_add_val(c[0], cpx_f_mul_val(c[stride_c], omx));
    cpx_f_t c0 = c[(n_c - 2) * stride_c];
    cpx_f_t c1 = c[(n_c - 1) * stride_c];
    int64_t nd = n_c;
    for (int64_t k = 3; k <= n_c; k++) {
        cpx_f_t tmp = c0;
        nd--;
        float f0 = (float)(nd - 1) / (float)nd;
        cpx_f_t term = {(2.0f * nd - 1.0f - x.r) / (float)nd, -x.i / (float)nd};
        c0 = cpx_f_sub_val(c[(n_c - k) * stride_c], cpx_f_mul_scalar(c1, f0));
        c1 = cpx_f_add_val(tmp, cpx_f_mul_val(c1, term));
    }
    return cpx_f_add_val(c0, cpx_f_mul_val(c1, omx));
}

// ----------------------------------------------------------------------------
// Flat Vector Kernels (AVX2 / SIMD + Loop Unrolling)
// ----------------------------------------------------------------------------

void v_polyval_double(const double *c, int64_t stride_c, int64_t n_c,
                      const double *x, double *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    if (n_c == 1) {
        double c0 = c[0];
        for (int64_t i = 0; i < size; i++) {
            res[i] = std::isnan(x[i]) ? NAN : c0;
        }
        return;
    }
    int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < size; i += 8) {
        __m256d vx0 = _mm256_loadu_pd(&x[i]);
        __m256d vx1 = _mm256_loadu_pd(&x[i + 4]);
        __m256d vval0 = _mm256_set1_pd(c[0]);
        __m256d vval1 = vval0;
        for (int64_t k = 1; k < n_c; k++) {
            __m256d vck = _mm256_set1_pd(c[k * stride_c]);
            vval0 = _mm256_fmadd_pd(vval0, vx0, vck);
            vval1 = _mm256_fmadd_pd(vval1, vx1, vck);
        }
        _mm256_storeu_pd(&res[i], vval0);
        _mm256_storeu_pd(&res[i + 4], vval1);
    }
#endif
    for (; i < size; i++) {
        double xi = x[i];
        double val = c[0];
        for (int64_t k = 1; k < n_c; k++) {
            val = val * xi + c[k * stride_c];
        }
        res[i] = val;
    }
}

void v_polyval_float(const float *c, int64_t stride_c, int64_t n_c,
                     const float *x, float *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    if (n_c == 1) {
        float c0 = c[0];
        for (int64_t i = 0; i < size; i++) {
            res[i] = std::isnan(x[i]) ? NAN : c0;
        }
        return;
    }
    int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    for (; i + 15 < size; i += 16) {
        __m256 vx0 = _mm256_loadu_ps(&x[i]);
        __m256 vx1 = _mm256_loadu_ps(&x[i + 8]);
        __m256 vval0 = _mm256_set1_ps(c[0]);
        __m256 vval1 = vval0;
        for (int64_t k = 1; k < n_c; k++) {
            __m256 vck = _mm256_set1_ps(c[k * stride_c]);
            vval0 = _mm256_fmadd_ps(vval0, vx0, vck);
            vval1 = _mm256_fmadd_ps(vval1, vx1, vck);
        }
        _mm256_storeu_ps(&res[i], vval0);
        _mm256_storeu_ps(&res[i + 8], vval1);
    }
#endif
    for (; i < size; i++) {
        float xi = x[i];
        float val = c[0];
        for (int64_t k = 1; k < n_c; k++) {
            val = val * xi + c[k * stride_c];
        }
        res[i] = val;
    }
}

void v_polyval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c,
                          const cpx_t *x, cpx_t *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = eval_poly_complex128(c, stride_c, n_c, x[i]);
    }
}

void v_polyval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c,
                         const cpx_f_t *x, cpx_f_t *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = eval_poly_complex64(c, stride_c, n_c, x[i]);
    }
}

void v_chebval_double(const double *c, int64_t stride_c, int64_t n_c,
                      const double *x, double *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    if (n_c == 1) {
        double c0 = c[0];
        for (int64_t i = 0; i < size; i++) {
            res[i] = std::isnan(x[i]) ? NAN : c0;
        }
        return;
    }
    if (n_c == 2) {
        double c0 = c[0];
        double c1 = c[stride_c];
        int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
        __m256d vc0 = _mm256_set1_pd(c0);
        __m256d vc1 = _mm256_set1_pd(c1);
        for (; i + 7 < size; i += 8) {
            __m256d vx0 = _mm256_loadu_pd(&x[i]);
            __m256d vx1 = _mm256_loadu_pd(&x[i + 4]);
            _mm256_storeu_pd(&res[i], _mm256_fmadd_pd(vc1, vx0, vc0));
            _mm256_storeu_pd(&res[i + 4], _mm256_fmadd_pd(vc1, vx1, vc0));
        }
#endif
        for (; i < size; i++) {
            res[i] = c0 + c1 * x[i];
        }
        return;
    }

    int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < size; i += 8) {
        __m256d vx0 = _mm256_loadu_pd(&x[i]);
        __m256d vx1 = _mm256_loadu_pd(&x[i + 4]);
        __m256d vx2_0 = _mm256_add_pd(vx0, vx0);
        __m256d vx2_1 = _mm256_add_pd(vx1, vx1);
        __m256d vc0_0 = _mm256_set1_pd(c[(n_c - 2) * stride_c]);
        __m256d vc0_1 = vc0_0;
        __m256d vc1_0 = _mm256_set1_pd(c[(n_c - 1) * stride_c]);
        __m256d vc1_1 = vc1_0;

        for (int64_t k = 3; k <= n_c; k++) {
            __m256d vtmp0 = vc0_0;
            __m256d vtmp1 = vc0_1;
            __m256d vck = _mm256_set1_pd(c[(n_c - k) * stride_c]);
            vc0_0 = _mm256_sub_pd(vck, vc1_0);
            vc0_1 = _mm256_sub_pd(vck, vc1_1);
            vc1_0 = _mm256_fmadd_pd(vc1_0, vx2_0, vtmp0);
            vc1_1 = _mm256_fmadd_pd(vc1_1, vx2_1, vtmp1);
        }
        _mm256_storeu_pd(&res[i], _mm256_fmadd_pd(vc1_0, vx0, vc0_0));
        _mm256_storeu_pd(&res[i + 4], _mm256_fmadd_pd(vc1_1, vx1, vc0_1));
    }
#endif
    for (; i < size; i++) {
        double xi = x[i];
        double x2 = 2.0 * xi;
        double c0 = c[(n_c - 2) * stride_c];
        double c1 = c[(n_c - 1) * stride_c];
        for (int64_t k = 3; k <= n_c; k++) {
            double tmp = c0;
            c0 = c[(n_c - k) * stride_c] - c1;
            c1 = tmp + c1 * x2;
        }
        res[i] = c0 + c1 * xi;
    }
}

void v_chebval_float(const float *c, int64_t stride_c, int64_t n_c,
                     const float *x, float *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    if (n_c == 1) {
        float c0 = c[0];
        for (int64_t i = 0; i < size; i++) {
            res[i] = std::isnan(x[i]) ? NAN : c0;
        }
        return;
    }
    if (n_c == 2) {
        float c0 = c[0];
        float c1 = c[stride_c];
        int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 vc0 = _mm256_set1_ps(c0);
        __m256 vc1 = _mm256_set1_ps(c1);
        for (; i + 15 < size; i += 16) {
            __m256 vx0 = _mm256_loadu_ps(&x[i]);
            __m256 vx1 = _mm256_loadu_ps(&x[i + 8]);
            _mm256_storeu_ps(&res[i], _mm256_fmadd_ps(vc1, vx0, vc0));
            _mm256_storeu_ps(&res[i + 8], _mm256_fmadd_ps(vc1, vx1, vc0));
        }
#endif
        for (; i < size; i++) {
            res[i] = c0 + c1 * x[i];
        }
        return;
    }

    int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    for (; i + 15 < size; i += 16) {
        __m256 vx0 = _mm256_loadu_ps(&x[i]);
        __m256 vx1 = _mm256_loadu_ps(&x[i + 8]);
        __m256 vx2_0 = _mm256_add_ps(vx0, vx0);
        __m256 vx2_1 = _mm256_add_ps(vx1, vx1);
        __m256 vc0_0 = _mm256_set1_ps(c[(n_c - 2) * stride_c]);
        __m256 vc0_1 = vc0_0;
        __m256 vc1_0 = _mm256_set1_ps(c[(n_c - 1) * stride_c]);
        __m256 vc1_1 = vc1_0;

        for (int64_t k = 3; k <= n_c; k++) {
            __m256 vtmp0 = vc0_0;
            __m256 vtmp1 = vc0_1;
            __m256 vck = _mm256_set1_ps(c[(n_c - k) * stride_c]);
            vc0_0 = _mm256_sub_ps(vck, vc1_0);
            vc0_1 = _mm256_sub_ps(vck, vc1_1);
            vc1_0 = _mm256_fmadd_ps(vc1_0, vx2_0, vtmp0);
            vc1_1 = _mm256_fmadd_ps(vc1_1, vx2_1, vtmp1);
        }
        _mm256_storeu_ps(&res[i], _mm256_fmadd_ps(vc1_0, vx0, vc0_0));
        _mm256_storeu_ps(&res[i + 8], _mm256_fmadd_ps(vc1_1, vx1, vc0_1));
    }
#endif
    for (; i < size; i++) {
        float xi = x[i];
        float x2 = 2.0f * xi;
        float c0 = c[(n_c - 2) * stride_c];
        float c1 = c[(n_c - 1) * stride_c];
        for (int64_t k = 3; k <= n_c; k++) {
            float tmp = c0;
            c0 = c[(n_c - k) * stride_c] - c1;
            c1 = tmp + c1 * x2;
        }
        res[i] = c0 + c1 * xi;
    }
}

void v_chebval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c,
                          const cpx_t *x, cpx_t *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = eval_cheb_complex128(c, stride_c, n_c, x[i]);
    }
}

void v_chebval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c,
                         const cpx_f_t *x, cpx_f_t *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = eval_cheb_complex64(c, stride_c, n_c, x[i]);
    }
}

void v_legval_double(const double *c, int64_t stride_c, int64_t n_c,
                     const double *x, double *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    if (n_c == 1) {
        double c0 = c[0];
        for (int64_t i = 0; i < size; i++) {
            res[i] = std::isnan(x[i]) ? NAN : c0;
        }
        return;
    }
    if (n_c == 2) {
        double c0 = c[0];
        double c1 = c[stride_c];
        int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
        __m256d vc0 = _mm256_set1_pd(c0);
        __m256d vc1 = _mm256_set1_pd(c1);
        for (; i + 7 < size; i += 8) {
            __m256d vx0 = _mm256_loadu_pd(&x[i]);
            __m256d vx1 = _mm256_loadu_pd(&x[i + 4]);
            _mm256_storeu_pd(&res[i], _mm256_fmadd_pd(vc1, vx0, vc0));
            _mm256_storeu_pd(&res[i + 4], _mm256_fmadd_pd(vc1, vx1, vc0));
        }
#endif
        for (; i < size; i++) {
            res[i] = c0 + c1 * x[i];
        }
        return;
    }

    int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < size; i += 8) {
        __m256d vx0 = _mm256_loadu_pd(&x[i]);
        __m256d vx1 = _mm256_loadu_pd(&x[i + 4]);
        __m256d vc0_0 = _mm256_set1_pd(c[(n_c - 2) * stride_c]);
        __m256d vc0_1 = vc0_0;
        __m256d vc1_0 = _mm256_set1_pd(c[(n_c - 1) * stride_c]);
        __m256d vc1_1 = vc1_0;

        int64_t nd = n_c;
        for (int64_t k = 3; k <= n_c; k++) {
            __m256d vtmp0 = vc0_0;
            __m256d vtmp1 = vc0_1;
            nd--;
            double f0 = (double)(nd - 1) / (double)nd;
            double f1 = (double)(2 * nd - 1) / (double)nd;
            __m256d vck = _mm256_set1_pd(c[(n_c - k) * stride_c]);
            __m256d vf0 = _mm256_set1_pd(f0);
            __m256d vf1 = _mm256_set1_pd(f1);

            vc0_0 = _mm256_fnmadd_pd(vc1_0, vf0, vck);
            vc0_1 = _mm256_fnmadd_pd(vc1_1, vf0, vck);
            __m256d v_scale0 = _mm256_mul_pd(vx0, vf1);
            __m256d v_scale1 = _mm256_mul_pd(vx1, vf1);
            vc1_0 = _mm256_fmadd_pd(vc1_0, v_scale0, vtmp0);
            vc1_1 = _mm256_fmadd_pd(vc1_1, v_scale1, vtmp1);
        }
        _mm256_storeu_pd(&res[i], _mm256_fmadd_pd(vc1_0, vx0, vc0_0));
        _mm256_storeu_pd(&res[i + 4], _mm256_fmadd_pd(vc1_1, vx1, vc0_1));
    }
#endif
    for (; i < size; i++) {
        double xi = x[i];
        double c0 = c[(n_c - 2) * stride_c];
        double c1 = c[(n_c - 1) * stride_c];
        int64_t nd = n_c;
        for (int64_t k = 3; k <= n_c; k++) {
            double tmp = c0;
            nd--;
            double f0 = (double)(nd - 1) / (double)nd;
            double f1 = (double)(2 * nd - 1) / (double)nd;
            c0 = c[(n_c - k) * stride_c] - c1 * f0;
            c1 = tmp + c1 * xi * f1;
        }
        res[i] = c0 + c1 * xi;
    }
}

void v_legval_float(const float *c, int64_t stride_c, int64_t n_c,
                    const float *x, float *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    if (n_c == 1) {
        float c0 = c[0];
        for (int64_t i = 0; i < size; i++) {
            res[i] = std::isnan(x[i]) ? NAN : c0;
        }
        return;
    }
    if (n_c == 2) {
        float c0 = c[0];
        float c1 = c[stride_c];
        int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 vc0 = _mm256_set1_ps(c0);
        __m256 vc1 = _mm256_set1_ps(c1);
        for (; i + 15 < size; i += 16) {
            __m256 vx0 = _mm256_loadu_ps(&x[i]);
            __m256 vx1 = _mm256_loadu_ps(&x[i + 8]);
            _mm256_storeu_ps(&res[i], _mm256_fmadd_ps(vc1, vx0, vc0));
            _mm256_storeu_ps(&res[i + 8], _mm256_fmadd_ps(vc1, vx1, vc0));
        }
#endif
        for (; i < size; i++) {
            res[i] = c0 + c1 * x[i];
        }
        return;
    }

    int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    for (; i + 15 < size; i += 16) {
        __m256 vx0 = _mm256_loadu_ps(&x[i]);
        __m256 vx1 = _mm256_loadu_ps(&x[i + 8]);
        __m256 vc0_0 = _mm256_set1_ps(c[(n_c - 2) * stride_c]);
        __m256 vc0_1 = vc0_0;
        __m256 vc1_0 = _mm256_set1_ps(c[(n_c - 1) * stride_c]);
        __m256 vc1_1 = vc1_0;

        int64_t nd = n_c;
        for (int64_t k = 3; k <= n_c; k++) {
            __m256 vtmp0 = vc0_0;
            __m256 vtmp1 = vc0_1;
            nd--;
            float f0 = (float)(nd - 1) / (float)nd;
            float f1 = (float)(2 * nd - 1) / (float)nd;
            __m256 vck = _mm256_set1_ps(c[(n_c - k) * stride_c]);
            __m256 vf0 = _mm256_set1_ps(f0);
            __m256 vf1 = _mm256_set1_ps(f1);

            vc0_0 = _mm256_fnmadd_ps(vc1_0, vf0, vck);
            vc0_1 = _mm256_fnmadd_ps(vc1_1, vf0, vck);
            __m256 v_scale0 = _mm256_mul_ps(vx0, vf1);
            __m256 v_scale1 = _mm256_mul_ps(vx1, vf1);
            vc1_0 = _mm256_fmadd_ps(vc1_0, v_scale0, vtmp0);
            vc1_1 = _mm256_fmadd_ps(vc1_1, v_scale1, vtmp1);
        }
        _mm256_storeu_ps(&res[i], _mm256_fmadd_ps(vc1_0, vx0, vc0_0));
        _mm256_storeu_ps(&res[i + 8], _mm256_fmadd_ps(vc1_1, vx1, vc0_1));
    }
#endif
    for (; i < size; i++) {
        float xi = x[i];
        float c0 = c[(n_c - 2) * stride_c];
        float c1 = c[(n_c - 1) * stride_c];
        int64_t nd = n_c;
        for (int64_t k = 3; k <= n_c; k++) {
            float tmp = c0;
            nd--;
            float f0 = (float)(nd - 1) / (float)nd;
            float f1 = (float)(2 * nd - 1) / (float)nd;
            c0 = c[(n_c - k) * stride_c] - c1 * f0;
            c1 = tmp + c1 * xi * f1;
        }
        res[i] = c0 + c1 * xi;
    }
}

void v_legval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c,
                         const cpx_t *x, cpx_t *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = eval_leg_complex128(c, stride_c, n_c, x[i]);
    }
}

void v_legval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c,
                        const cpx_f_t *x, cpx_f_t *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = eval_leg_complex64(c, stride_c, n_c, x[i]);
    }
}

void v_hermval_double(const double *c, int64_t stride_c, int64_t n_c,
                      const double *x, double *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    if (n_c == 1) {
        double c0 = c[0];
        for (int64_t i = 0; i < size; i++) {
            res[i] = std::isnan(x[i]) ? NAN : c0;
        }
        return;
    }
    if (n_c == 2) {
        double c0 = c[0];
        double c1 = c[stride_c];
        double c1_2 = 2.0 * c1;
        int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
        __m256d vc0 = _mm256_set1_pd(c0);
        __m256d vc1_2 = _mm256_set1_pd(c1_2);
        for (; i + 7 < size; i += 8) {
            __m256d vx0 = _mm256_loadu_pd(&x[i]);
            __m256d vx1 = _mm256_loadu_pd(&x[i + 4]);
            _mm256_storeu_pd(&res[i], _mm256_fmadd_pd(vc1_2, vx0, vc0));
            _mm256_storeu_pd(&res[i + 4], _mm256_fmadd_pd(vc1_2, vx1, vc0));
        }
#endif
        for (; i < size; i++) {
            res[i] = c0 + c1_2 * x[i];
        }
        return;
    }

    int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < size; i += 8) {
        __m256d vx0 = _mm256_loadu_pd(&x[i]);
        __m256d vx1 = _mm256_loadu_pd(&x[i + 4]);
        __m256d vx2_0 = _mm256_add_pd(vx0, vx0);
        __m256d vx2_1 = _mm256_add_pd(vx1, vx1);
        __m256d vc0_0 = _mm256_set1_pd(c[(n_c - 2) * stride_c]);
        __m256d vc0_1 = vc0_0;
        __m256d vc1_0 = _mm256_set1_pd(c[(n_c - 1) * stride_c]);
        __m256d vc1_1 = vc1_0;

        int64_t nd = n_c;
        for (int64_t k = 3; k <= n_c; k++) {
            __m256d vtmp0 = vc0_0;
            __m256d vtmp1 = vc0_1;
            nd--;
            double f0 = 2.0 * (double)(nd - 1);
            __m256d vck = _mm256_set1_pd(c[(n_c - k) * stride_c]);
            __m256d vf0 = _mm256_set1_pd(f0);

            vc0_0 = _mm256_fnmadd_pd(vc1_0, vf0, vck);
            vc0_1 = _mm256_fnmadd_pd(vc1_1, vf0, vck);
            vc1_0 = _mm256_fmadd_pd(vc1_0, vx2_0, vtmp0);
            vc1_1 = _mm256_fmadd_pd(vc1_1, vx2_1, vtmp1);
        }
        _mm256_storeu_pd(&res[i], _mm256_fmadd_pd(vc1_0, vx2_0, vc0_0));
        _mm256_storeu_pd(&res[i + 4], _mm256_fmadd_pd(vc1_1, vx2_1, vc0_1));
    }
#endif
    for (; i < size; i++) {
        double xi = x[i];
        double x2 = 2.0 * xi;
        double c0 = c[(n_c - 2) * stride_c];
        double c1 = c[(n_c - 1) * stride_c];
        int64_t nd = n_c;
        for (int64_t k = 3; k <= n_c; k++) {
            double tmp = c0;
            nd--;
            double f0 = 2.0 * (double)(nd - 1);
            c0 = c[(n_c - k) * stride_c] - c1 * f0;
            c1 = tmp + c1 * x2;
        }
        res[i] = c0 + c1 * x2;
    }
}

void v_hermval_float(const float *c, int64_t stride_c, int64_t n_c,
                     const float *x, float *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    if (n_c == 1) {
        float c0 = c[0];
        for (int64_t i = 0; i < size; i++) {
            res[i] = std::isnan(x[i]) ? NAN : c0;
        }
        return;
    }
    if (n_c == 2) {
        float c0 = c[0];
        float c1 = c[stride_c];
        float c1_2 = 2.0f * c1;
        int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 vc0 = _mm256_set1_ps(c0);
        __m256 vc1_2 = _mm256_set1_ps(c1_2);
        for (; i + 15 < size; i += 16) {
            __m256 vx0 = _mm256_loadu_ps(&x[i]);
            __m256 vx1 = _mm256_loadu_ps(&x[i + 8]);
            _mm256_storeu_ps(&res[i], _mm256_fmadd_ps(vc1_2, vx0, vc0));
            _mm256_storeu_ps(&res[i + 8], _mm256_fmadd_ps(vc1_2, vx1, vc0));
        }
#endif
        for (; i < size; i++) {
            res[i] = c0 + c1_2 * x[i];
        }
        return;
    }

    int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    for (; i + 15 < size; i += 16) {
        __m256 vx0 = _mm256_loadu_ps(&x[i]);
        __m256 vx1 = _mm256_loadu_ps(&x[i + 8]);
        __m256 vx2_0 = _mm256_add_ps(vx0, vx0);
        __m256 vx2_1 = _mm256_add_ps(vx1, vx1);
        __m256 vc0_0 = _mm256_set1_ps(c[(n_c - 2) * stride_c]);
        __m256 vc0_1 = vc0_0;
        __m256 vc1_0 = _mm256_set1_ps(c[(n_c - 1) * stride_c]);
        __m256 vc1_1 = vc1_0;

        int64_t nd = n_c;
        for (int64_t k = 3; k <= n_c; k++) {
            __m256 vtmp0 = vc0_0;
            __m256 vtmp1 = vc0_1;
            nd--;
            float f0 = 2.0f * (float)(nd - 1);
            __m256 vck = _mm256_set1_ps(c[(n_c - k) * stride_c]);
            __m256 vf0 = _mm256_set1_ps(f0);

            vc0_0 = _mm256_fnmadd_ps(vc1_0, vf0, vck);
            vc0_1 = _mm256_fnmadd_ps(vc1_1, vf0, vck);
            vc1_0 = _mm256_fmadd_ps(vc1_0, vx2_0, vtmp0);
            vc1_1 = _mm256_fmadd_ps(vc1_1, vx2_1, vtmp1);
        }
        _mm256_storeu_ps(&res[i], _mm256_fmadd_ps(vc1_0, vx2_0, vc0_0));
        _mm256_storeu_ps(&res[i + 8], _mm256_fmadd_ps(vc1_1, vx2_1, vc0_1));
    }
#endif
    for (; i < size; i++) {
        float xi = x[i];
        float x2 = 2.0f * xi;
        float c0 = c[(n_c - 2) * stride_c];
        float c1 = c[(n_c - 1) * stride_c];
        int64_t nd = n_c;
        for (int64_t k = 3; k <= n_c; k++) {
            float tmp = c0;
            nd--;
            float f0 = 2.0f * (float)(nd - 1);
            c0 = c[(n_c - k) * stride_c] - c1 * f0;
            c1 = tmp + c1 * x2;
        }
        res[i] = c0 + c1 * x2;
    }
}

void v_hermval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c,
                          const cpx_t *x, cpx_t *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = eval_herm_complex128(c, stride_c, n_c, x[i]);
    }
}

void v_hermval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c,
                         const cpx_f_t *x, cpx_f_t *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = eval_herm_complex64(c, stride_c, n_c, x[i]);
    }
}

void v_lagval_double(const double *c, int64_t stride_c, int64_t n_c,
                     const double *x, double *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    if (n_c == 1) {
        double c0 = c[0];
        for (int64_t i = 0; i < size; i++) {
            res[i] = std::isnan(x[i]) ? NAN : c0;
        }
        return;
    }
    if (n_c == 2) {
        double c0 = c[0];
        double c1 = c[stride_c];
        int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
        __m256d vc0 = _mm256_set1_pd(c0);
        __m256d vc1 = _mm256_set1_pd(c1);
        __m256d v1 = _mm256_set1_pd(1.0);
        for (; i + 7 < size; i += 8) {
            __m256d vx0 = _mm256_loadu_pd(&x[i]);
            __m256d vx1 = _mm256_loadu_pd(&x[i + 4]);
            __m256d vomx0 = _mm256_sub_pd(v1, vx0);
            __m256d vomx1 = _mm256_sub_pd(v1, vx1);
            _mm256_storeu_pd(&res[i], _mm256_fmadd_pd(vc1, vomx0, vc0));
            _mm256_storeu_pd(&res[i + 4], _mm256_fmadd_pd(vc1, vomx1, vc0));
        }
#endif
        for (; i < size; i++) {
            res[i] = c0 + c1 * (1.0 - x[i]);
        }
        return;
    }

    int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    __m256d v1 = _mm256_set1_pd(1.0);
    for (; i + 7 < size; i += 8) {
        __m256d vx0 = _mm256_loadu_pd(&x[i]);
        __m256d vx1 = _mm256_loadu_pd(&x[i + 4]);
        __m256d vc0_0 = _mm256_set1_pd(c[(n_c - 2) * stride_c]);
        __m256d vc0_1 = vc0_0;
        __m256d vc1_0 = _mm256_set1_pd(c[(n_c - 1) * stride_c]);
        __m256d vc1_1 = vc1_0;

        int64_t nd = n_c;
        for (int64_t k = 3; k <= n_c; k++) {
            __m256d vtmp0 = vc0_0;
            __m256d vtmp1 = vc0_1;
            nd--;
            double f0 = (double)(nd - 1) / (double)nd;
            double f1 = (double)(2 * nd - 1) / (double)nd;
            double inv_nd = 1.0 / (double)nd;
            __m256d vck = _mm256_set1_pd(c[(n_c - k) * stride_c]);
            __m256d vf0 = _mm256_set1_pd(f0);
            __m256d vf1 = _mm256_set1_pd(f1);
            __m256d vinv = _mm256_set1_pd(inv_nd);

            vc0_0 = _mm256_fnmadd_pd(vc1_0, vf0, vck);
            vc0_1 = _mm256_fnmadd_pd(vc1_1, vf0, vck);
            __m256d v_term0 = _mm256_fnmadd_pd(vx0, vinv, vf1);
            __m256d v_term1 = _mm256_fnmadd_pd(vx1, vinv, vf1);
            vc1_0 = _mm256_fmadd_pd(vc1_0, v_term0, vtmp0);
            vc1_1 = _mm256_fmadd_pd(vc1_1, v_term1, vtmp1);
        }
        __m256d vomx0 = _mm256_sub_pd(v1, vx0);
        __m256d vomx1 = _mm256_sub_pd(v1, vx1);
        _mm256_storeu_pd(&res[i], _mm256_fmadd_pd(vc1_0, vomx0, vc0_0));
        _mm256_storeu_pd(&res[i + 4], _mm256_fmadd_pd(vc1_1, vomx1, vc0_1));
    }
#endif
    for (; i < size; i++) {
        double xi = x[i];
        double c0 = c[(n_c - 2) * stride_c];
        double c1 = c[(n_c - 1) * stride_c];
        int64_t nd = n_c;
        for (int64_t k = 3; k <= n_c; k++) {
            double tmp = c0;
            nd--;
            double f0 = (double)(nd - 1) / (double)nd;
            c0 = c[(n_c - k) * stride_c] - c1 * f0;
            c1 = tmp + (c1 * ((2.0 * nd - 1.0) - xi)) / (double)nd;
        }
        res[i] = c0 + c1 * (1.0 - xi);
    }
}

void v_lagval_float(const float *c, int64_t stride_c, int64_t n_c,
                    const float *x, float *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    if (n_c == 1) {
        float c0 = c[0];
        for (int64_t i = 0; i < size; i++) {
            res[i] = std::isnan(x[i]) ? NAN : c0;
        }
        return;
    }
    if (n_c == 2) {
        float c0 = c[0];
        float c1 = c[stride_c];
        int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 vc0 = _mm256_set1_ps(c0);
        __m256 vc1 = _mm256_set1_ps(c1);
        __m256 v1 = _mm256_set1_ps(1.0f);
        for (; i + 15 < size; i += 16) {
            __m256 vx0 = _mm256_loadu_ps(&x[i]);
            __m256 vx1 = _mm256_loadu_ps(&x[i + 8]);
            __m256 vomx0 = _mm256_sub_ps(v1, vx0);
            __m256 vomx1 = _mm256_sub_ps(v1, vx1);
            _mm256_storeu_ps(&res[i], _mm256_fmadd_ps(vc1, vomx0, vc0));
            _mm256_storeu_ps(&res[i + 8], _mm256_fmadd_ps(vc1, vomx1, vc0));
        }
#endif
        for (; i < size; i++) {
            res[i] = c0 + c1 * (1.0f - x[i]);
        }
        return;
    }

    int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 v1 = _mm256_set1_ps(1.0f);
    for (; i + 15 < size; i += 16) {
        __m256 vx0 = _mm256_loadu_ps(&x[i]);
        __m256 vx1 = _mm256_loadu_ps(&x[i + 8]);
        __m256 vc0_0 = _mm256_set1_ps(c[(n_c - 2) * stride_c]);
        __m256 vc0_1 = vc0_0;
        __m256 vc1_0 = _mm256_set1_ps(c[(n_c - 1) * stride_c]);
        __m256 vc1_1 = vc1_0;

        int64_t nd = n_c;
        for (int64_t k = 3; k <= n_c; k++) {
            __m256 vtmp0 = vc0_0;
            __m256 vtmp1 = vc0_1;
            nd--;
            float f0 = (float)(nd - 1) / (float)nd;
            float f1 = (float)(2 * nd - 1) / (float)nd;
            float inv_nd = 1.0f / (float)nd;
            __m256 vck = _mm256_set1_ps(c[(n_c - k) * stride_c]);
            __m256 vf0 = _mm256_set1_ps(f0);
            __m256 vf1 = _mm256_set1_ps(f1);
            __m256 vinv = _mm256_set1_ps(inv_nd);

            vc0_0 = _mm256_fnmadd_ps(vc1_0, vf0, vck);
            vc0_1 = _mm256_fnmadd_ps(vc1_1, vf0, vck);
            __m256 v_term0 = _mm256_fnmadd_ps(vx0, vinv, vf1);
            __m256 v_term1 = _mm256_fnmadd_ps(vx1, vinv, vf1);
            vc1_0 = _mm256_fmadd_ps(vc1_0, v_term0, vtmp0);
            vc1_1 = _mm256_fmadd_ps(vc1_1, v_term1, vtmp1);
        }
        __m256 vomx0 = _mm256_sub_ps(v1, vx0);
        __m256 vomx1 = _mm256_sub_ps(v1, vx1);
        _mm256_storeu_ps(&res[i], _mm256_fmadd_ps(vc1_0, vomx0, vc0_0));
        _mm256_storeu_ps(&res[i + 8], _mm256_fmadd_ps(vc1_1, vomx1, vc0_1));
    }
#endif
    for (; i < size; i++) {
        float xi = x[i];
        float c0 = c[(n_c - 2) * stride_c];
        float c1 = c[(n_c - 1) * stride_c];
        int64_t nd = n_c;
        for (int64_t k = 3; k <= n_c; k++) {
            float tmp = c0;
            nd--;
            float f0 = (float)(nd - 1) / (float)nd;
            c0 = c[(n_c - k) * stride_c] - c1 * f0;
            c1 = tmp + (c1 * ((2.0f * nd - 1.0f) - xi)) / (float)nd;
        }
        res[i] = c0 + c1 * (1.0f - xi);
    }
}

void v_lagval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c,
                         const cpx_t *x, cpx_t *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = eval_lag_complex128(c, stride_c, n_c, x[i]);
    }
}

void v_lagval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c,
                        const cpx_f_t *x, cpx_f_t *res, int64_t size) {
    if (c == nullptr || x == nullptr || res == nullptr || size <= 0 || n_c <= 0) return;
    for (int64_t i = 0; i < size; i++) {
        res[i] = eval_lag_complex64(c, stride_c, n_c, x[i]);
    }
}

// ----------------------------------------------------------------------------
// Strided Multi-Dimensional Kernels
// ----------------------------------------------------------------------------

template <typename T, typename Func>
static void s_poly_generic_impl(const T *c, int64_t stride_c, int64_t n_c,
                                const T *x, const int64_t *stridesX,
                                T *res, const int64_t *stridesRes,
                                const int64_t *shape, int rank,
                                Func eval_fn) {
    if (c == nullptr || x == nullptr || res == nullptr || (rank > 0 && shape == nullptr) || rank < 0 || n_c <= 0) return;
    
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) {
        if (shape[i] <= 0) return;
        total_elements *= shape[i];
    }

    intptr_t min_off_c = (stride_c < 0) ? (intptr_t)(n_c - 1) * stride_c : 0;
    intptr_t max_off_c = (stride_c > 0) ? (intptr_t)(n_c - 1) * stride_c : 0;
    const uint8_t *min_c = reinterpret_cast<const uint8_t *>(c) + min_off_c * (intptr_t)sizeof(T);
    const uint8_t *max_c = reinterpret_cast<const uint8_t *>(c) + max_off_c * (intptr_t)sizeof(T) + (sizeof(T) - 1);

    intptr_t min_off_res = 0, max_off_res = 0;
    for (int d = 0; d < rank; d++) {
        intptr_t s = stridesRes[d];
        intptr_t extent = shape[d] - 1;
        if (s < 0) min_off_res += extent * s;
        else max_off_res += extent * s;
    }
    const uint8_t *min_res = reinterpret_cast<const uint8_t *>(res) + min_off_res * (intptr_t)sizeof(T);
    const uint8_t *max_res = reinterpret_cast<const uint8_t *>(res) + max_off_res * (intptr_t)sizeof(T) + (sizeof(T) - 1);

    bool overlap_c = (min_c <= max_res && min_res <= max_c);
    bool overlap_x = strided_buffers_overlap_not_identical(
        x, stridesX, sizeof(T),
        res, stridesRes, sizeof(T),
        shape, rank
    );
    if (overlap_c || overlap_x) {
        NoThrowBuffer<T> temp(total_elements);
        if (!temp.ok()) {
            ndarray_set_oom_flag();
            return;
        }
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetX = 0;
        for (int64_t el = 0; el < total_elements; el++) {
            temp[el] = eval_fn(c, stride_c, n_c, x[offsetX]);
            for (int d = rank - 1; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetX += stridesX[d];
                    break;
                }
                coord[d] = 0;
                offsetX -= (shape[d] - 1) * stridesX[d];
            }
        }
        DECLARE_RANK_BUFFER(int64_t, coord_out, rank);
        int64_t offsetRes = 0;
        for (int64_t el = 0; el < total_elements; el++) {
            res[offsetRes] = temp[el];
            for (int d = rank - 1; d >= 0; d--) {
                coord_out[d]++;
                if (coord_out[d] < shape[d]) {
                    offsetRes += stridesRes[d];
                    break;
                }
                coord_out[d] = 0;
                offsetRes -= (shape[d] - 1) * stridesRes[d];
            }
        }
        return;
    }

    DECLARE_RANK_BUFFER(int64_t, coord, rank);
    int64_t offsetX = 0, offsetRes = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        res[offsetRes] = eval_fn(c, stride_c, n_c, x[offsetX]);

        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetX += stridesX[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetX -= (shape[d] - 1) * stridesX[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

void s_polyval_double(const double *c, int64_t stride_c, int64_t n_c, const double *x, const int64_t *stridesX, double *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_poly_double);
}
void s_polyval_float(const float *c, int64_t stride_c, int64_t n_c, const float *x, const int64_t *stridesX, float *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_poly_float);
}
void s_polyval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, const cpx_t *x, const int64_t *stridesX, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_poly_complex128);
}
void s_polyval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, const cpx_f_t *x, const int64_t *stridesX, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_poly_complex64);
}

void s_chebval_double(const double *c, int64_t stride_c, int64_t n_c, const double *x, const int64_t *stridesX, double *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_cheb_double);
}
void s_chebval_float(const float *c, int64_t stride_c, int64_t n_c, const float *x, const int64_t *stridesX, float *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_cheb_float);
}
void s_chebval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, const cpx_t *x, const int64_t *stridesX, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_cheb_complex128);
}
void s_chebval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, const cpx_f_t *x, const int64_t *stridesX, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_cheb_complex64);
}

void s_legval_double(const double *c, int64_t stride_c, int64_t n_c, const double *x, const int64_t *stridesX, double *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_leg_double);
}
void s_legval_float(const float *c, int64_t stride_c, int64_t n_c, const float *x, const int64_t *stridesX, float *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_leg_float);
}
void s_legval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, const cpx_t *x, const int64_t *stridesX, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_leg_complex128);
}
void s_legval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, const cpx_f_t *x, const int64_t *stridesX, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_leg_complex64);
}

void s_hermval_double(const double *c, int64_t stride_c, int64_t n_c, const double *x, const int64_t *stridesX, double *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_herm_double);
}
void s_hermval_float(const float *c, int64_t stride_c, int64_t n_c, const float *x, const int64_t *stridesX, float *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_herm_float);
}
void s_hermval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, const cpx_t *x, const int64_t *stridesX, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_herm_complex128);
}
void s_hermval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, const cpx_f_t *x, const int64_t *stridesX, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_herm_complex64);
}

void s_lagval_double(const double *c, int64_t stride_c, int64_t n_c, const double *x, const int64_t *stridesX, double *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_lag_double);
}
void s_lagval_float(const float *c, int64_t stride_c, int64_t n_c, const float *x, const int64_t *stridesX, float *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_lag_float);
}
void s_lagval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, const cpx_t *x, const int64_t *stridesX, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_lag_complex128);
}
void s_lagval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, const cpx_f_t *x, const int64_t *stridesX, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank) {
    s_poly_generic_impl(c, stride_c, n_c, x, stridesX, res, stridesRes, shape, rank, eval_lag_complex64);
}

// ============================================================================
// Histogram Implementations (Uniform and Binary Search)
// ============================================================================

template <typename T, typename W, typename H>
static void v_histogram_uniform_kernel(
    const T *src,
    const W *weights,
    int64_t size,
    int64_t nbins,
    double min_val,
    double max_val,
    double norm,
    H *hist
) {
    double step = (max_val - min_val) / (double)nbins;
    for (int64_t i = 0; i < size; ++i) {
        double val = (double)src[i];
        if (std::isnan(val) || val < min_val || val > max_val) continue;
        int64_t bin;
        if (val == max_val) {
            bin = nbins - 1;
        } else {
            double scaled = std::floor((val - min_val) * norm);
            if (std::isnan(scaled) || scaled < 0.0) {
                bin = 0;
            } else if (scaled >= (double)nbins) {
                bin = nbins - 1;
            } else {
                bin = (int64_t)scaled;
            }
            if (bin < nbins - 1 && val >= min_val + (double)(bin + 1) * step) {
                bin++;
            } else if (bin > 0 && val < min_val + (double)bin * step) {
                bin--;
            }
        }
        if (weights != nullptr) {
            hist[bin] += (H)weights[i];
        } else {
            hist[bin] += 1;
        }
    }
}

template <typename T, typename W, typename H>
static void s_histogram_uniform_kernel(
    const T *src, int64_t strideSrc,
    const W *weights, int64_t strideWeights,
    int64_t size,
    int64_t nbins,
    double min_val,
    double max_val,
    double norm,
    H *hist, int64_t strideHist
) {
    double step = (max_val - min_val) / (double)nbins;
    for (int64_t i = 0; i < size; ++i) {
        double val = (double)src[i * strideSrc];
        if (std::isnan(val) || val < min_val || val > max_val) continue;
        int64_t bin;
        if (val == max_val) {
            bin = nbins - 1;
        } else {
            double scaled = std::floor((val - min_val) * norm);
            if (std::isnan(scaled) || scaled < 0.0) {
                bin = 0;
            } else if (scaled >= (double)nbins) {
                bin = nbins - 1;
            } else {
                bin = (int64_t)scaled;
            }
            if (bin < nbins - 1 && val >= min_val + (double)(bin + 1) * step) {
                bin++;
            } else if (bin > 0 && val < min_val + (double)bin * step) {
                bin--;
            }
        }
        if (weights != nullptr) {
            hist[bin * strideHist] += (H)weights[i * strideWeights];
        } else {
            hist[bin * strideHist] += 1;
        }
    }
}

template <typename T, typename W, typename H>
static void v_histogram_binsearch_kernel(
    const T *src,
    const W *weights,
    const double *bin_edges,
    int64_t num_edges,
    int64_t size,
    H *hist
) {
    int64_t nbins = num_edges - 1;
    double first_edge = bin_edges[0];
    double last_edge = bin_edges[num_edges - 1];
    for (int64_t i = 0; i < size; ++i) {
        double val = (double)src[i];
        if (std::isnan(val) || val < first_edge || val > last_edge) continue;
        int64_t bin;
        if (val == last_edge) {
            bin = nbins - 1;
        } else {
            const double *p = std::upper_bound(bin_edges, bin_edges + num_edges, val);
            bin = (int64_t)(p - bin_edges) - 1;
            if (bin < 0 || bin >= nbins) continue;
        }
        if (weights != nullptr) {
            hist[bin] += (H)weights[i];
        } else {
            hist[bin] += 1;
        }
    }
}

template <typename T, typename W, typename H>
static void s_histogram_binsearch_kernel(
    const T *src, int64_t strideSrc,
    const W *weights, int64_t strideWeights,
    const double *bin_edges,
    int64_t num_edges,
    int64_t size,
    H *hist, int64_t strideHist
) {
    int64_t nbins = num_edges - 1;
    double first_edge = bin_edges[0];
    double last_edge = bin_edges[num_edges - 1];
    for (int64_t i = 0; i < size; ++i) {
        double val = (double)src[i * strideSrc];
        if (std::isnan(val) || val < first_edge || val > last_edge) continue;
        int64_t bin;
        if (val == last_edge) {
            bin = nbins - 1;
        } else {
            const double *p = std::upper_bound(bin_edges, bin_edges + num_edges, val);
            bin = (int64_t)(p - bin_edges) - 1;
            if (bin < 0 || bin >= nbins) continue;
        }
        if (weights != nullptr) {
            hist[bin * strideHist] += (H)weights[i * strideWeights];
        } else {
            hist[bin * strideHist] += 1;
        }
    }
}

#define DISPATCH_HIST_H_TYPE_V(T, W, KERNEL_NAME, ...) \
    switch (dtypeHist) { \
        case DTYPE_INT64: \
            KERNEL_NAME<T, W, int64_t>(__VA_ARGS__, (int64_t *)hist); \
            break; \
        case DTYPE_FLOAT64: \
            KERNEL_NAME<T, W, double>(__VA_ARGS__, (double *)hist); \
            break; \
        case DTYPE_FLOAT32: \
            KERNEL_NAME<T, W, float>(__VA_ARGS__, (float *)hist); \
            break; \
        default: return; \
    }

#define DISPATCH_HIST_H_TYPE_S(T, W, KERNEL_NAME, ...) \
    switch (dtypeHist) { \
        case DTYPE_INT64: \
            KERNEL_NAME<T, W, int64_t>(__VA_ARGS__, (int64_t *)hist, strideHist); \
            break; \
        case DTYPE_FLOAT64: \
            KERNEL_NAME<T, W, double>(__VA_ARGS__, (double *)hist, strideHist); \
            break; \
        case DTYPE_FLOAT32: \
            KERNEL_NAME<T, W, float>(__VA_ARGS__, (float *)hist, strideHist); \
            break; \
        default: return; \
    }

#define DISPATCH_HIST_W_TYPE_V_UNIFORM(T) \
    if (weights == nullptr) { \
        DISPATCH_HIST_H_TYPE_V(T, double, v_histogram_uniform_kernel, (const T *)src, (const double *)nullptr, size, nbins, min_val, max_val, norm) \
    } else { \
        switch (dtypeWeights) { \
            case DTYPE_FLOAT64: \
                DISPATCH_HIST_H_TYPE_V(T, double, v_histogram_uniform_kernel, (const T *)src, (const double *)weights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_FLOAT32: \
                DISPATCH_HIST_H_TYPE_V(T, float, v_histogram_uniform_kernel, (const T *)src, (const float *)weights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_INT32: \
                DISPATCH_HIST_H_TYPE_V(T, int32_t, v_histogram_uniform_kernel, (const T *)src, (const int32_t *)weights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_INT64: \
                DISPATCH_HIST_H_TYPE_V(T, int64_t, v_histogram_uniform_kernel, (const T *)src, (const int64_t *)weights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_INT16: \
                DISPATCH_HIST_H_TYPE_V(T, int16_t, v_histogram_uniform_kernel, (const T *)src, (const int16_t *)weights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_INT8: \
                DISPATCH_HIST_H_TYPE_V(T, int8_t, v_histogram_uniform_kernel, (const T *)src, (const int8_t *)weights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_UINT64: \
                DISPATCH_HIST_H_TYPE_V(T, uint64_t, v_histogram_uniform_kernel, (const T *)src, (const uint64_t *)weights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_UINT32: \
                DISPATCH_HIST_H_TYPE_V(T, uint32_t, v_histogram_uniform_kernel, (const T *)src, (const uint32_t *)weights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_UINT16: \
                DISPATCH_HIST_H_TYPE_V(T, uint16_t, v_histogram_uniform_kernel, (const T *)src, (const uint16_t *)weights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_UINT8: \
            case DTYPE_BOOLEAN: \
                DISPATCH_HIST_H_TYPE_V(T, uint8_t, v_histogram_uniform_kernel, (const T *)src, (const uint8_t *)weights, size, nbins, min_val, max_val, norm) \
                break; \
            default: return; \
        } \
    }

#define DISPATCH_HIST_W_TYPE_S_UNIFORM(T) \
    if (weights == nullptr) { \
        DISPATCH_HIST_H_TYPE_S(T, double, s_histogram_uniform_kernel, (const T *)src, strideSrc, (const double *)nullptr, 0, size, nbins, min_val, max_val, norm) \
    } else { \
        switch (dtypeWeights) { \
            case DTYPE_FLOAT64: \
                DISPATCH_HIST_H_TYPE_S(T, double, s_histogram_uniform_kernel, (const T *)src, strideSrc, (const double *)weights, strideWeights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_FLOAT32: \
                DISPATCH_HIST_H_TYPE_S(T, float, s_histogram_uniform_kernel, (const T *)src, strideSrc, (const float *)weights, strideWeights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_INT32: \
                DISPATCH_HIST_H_TYPE_S(T, int32_t, s_histogram_uniform_kernel, (const T *)src, strideSrc, (const int32_t *)weights, strideWeights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_INT64: \
                DISPATCH_HIST_H_TYPE_S(T, int64_t, s_histogram_uniform_kernel, (const T *)src, strideSrc, (const int64_t *)weights, strideWeights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_INT16: \
                DISPATCH_HIST_H_TYPE_S(T, int16_t, s_histogram_uniform_kernel, (const T *)src, strideSrc, (const int16_t *)weights, strideWeights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_INT8: \
                DISPATCH_HIST_H_TYPE_S(T, int8_t, s_histogram_uniform_kernel, (const T *)src, strideSrc, (const int8_t *)weights, strideWeights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_UINT64: \
                DISPATCH_HIST_H_TYPE_S(T, uint64_t, s_histogram_uniform_kernel, (const T *)src, strideSrc, (const uint64_t *)weights, strideWeights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_UINT32: \
                DISPATCH_HIST_H_TYPE_S(T, uint32_t, s_histogram_uniform_kernel, (const T *)src, strideSrc, (const uint32_t *)weights, strideWeights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_UINT16: \
                DISPATCH_HIST_H_TYPE_S(T, uint16_t, s_histogram_uniform_kernel, (const T *)src, strideSrc, (const uint16_t *)weights, strideWeights, size, nbins, min_val, max_val, norm) \
                break; \
            case DTYPE_UINT8: \
            case DTYPE_BOOLEAN: \
                DISPATCH_HIST_H_TYPE_S(T, uint8_t, s_histogram_uniform_kernel, (const T *)src, strideSrc, (const uint8_t *)weights, strideWeights, size, nbins, min_val, max_val, norm) \
                break; \
            default: return; \
        } \
    }

#define DISPATCH_HIST_W_TYPE_V_BINSEARCH(T) \
    if (weights == nullptr) { \
        DISPATCH_HIST_H_TYPE_V(T, double, v_histogram_binsearch_kernel, (const T *)src, (const double *)nullptr, bin_edges, num_edges, size) \
    } else { \
        switch (dtypeWeights) { \
            case DTYPE_FLOAT64: \
                DISPATCH_HIST_H_TYPE_V(T, double, v_histogram_binsearch_kernel, (const T *)src, (const double *)weights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_FLOAT32: \
                DISPATCH_HIST_H_TYPE_V(T, float, v_histogram_binsearch_kernel, (const T *)src, (const float *)weights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_INT32: \
                DISPATCH_HIST_H_TYPE_V(T, int32_t, v_histogram_binsearch_kernel, (const T *)src, (const int32_t *)weights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_INT64: \
                DISPATCH_HIST_H_TYPE_V(T, int64_t, v_histogram_binsearch_kernel, (const T *)src, (const int64_t *)weights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_INT16: \
                DISPATCH_HIST_H_TYPE_V(T, int16_t, v_histogram_binsearch_kernel, (const T *)src, (const int16_t *)weights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_INT8: \
                DISPATCH_HIST_H_TYPE_V(T, int8_t, v_histogram_binsearch_kernel, (const T *)src, (const int8_t *)weights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_UINT64: \
                DISPATCH_HIST_H_TYPE_V(T, uint64_t, v_histogram_binsearch_kernel, (const T *)src, (const uint64_t *)weights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_UINT32: \
                DISPATCH_HIST_H_TYPE_V(T, uint32_t, v_histogram_binsearch_kernel, (const T *)src, (const uint32_t *)weights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_UINT16: \
                DISPATCH_HIST_H_TYPE_V(T, uint16_t, v_histogram_binsearch_kernel, (const T *)src, (const uint16_t *)weights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_UINT8: \
            case DTYPE_BOOLEAN: \
                DISPATCH_HIST_H_TYPE_V(T, uint8_t, v_histogram_binsearch_kernel, (const T *)src, (const uint8_t *)weights, bin_edges, num_edges, size) \
                break; \
            default: return; \
        } \
    }

#define DISPATCH_HIST_W_TYPE_S_BINSEARCH(T) \
    if (weights == nullptr) { \
        DISPATCH_HIST_H_TYPE_S(T, double, s_histogram_binsearch_kernel, (const T *)src, strideSrc, (const double *)nullptr, 0, bin_edges, num_edges, size) \
    } else { \
        switch (dtypeWeights) { \
            case DTYPE_FLOAT64: \
                DISPATCH_HIST_H_TYPE_S(T, double, s_histogram_binsearch_kernel, (const T *)src, strideSrc, (const double *)weights, strideWeights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_FLOAT32: \
                DISPATCH_HIST_H_TYPE_S(T, float, s_histogram_binsearch_kernel, (const T *)src, strideSrc, (const float *)weights, strideWeights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_INT32: \
                DISPATCH_HIST_H_TYPE_S(T, int32_t, s_histogram_binsearch_kernel, (const T *)src, strideSrc, (const int32_t *)weights, strideWeights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_INT64: \
                DISPATCH_HIST_H_TYPE_S(T, int64_t, s_histogram_binsearch_kernel, (const T *)src, strideSrc, (const int64_t *)weights, strideWeights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_INT16: \
                DISPATCH_HIST_H_TYPE_S(T, int16_t, s_histogram_binsearch_kernel, (const T *)src, strideSrc, (const int16_t *)weights, strideWeights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_INT8: \
                DISPATCH_HIST_H_TYPE_S(T, int8_t, s_histogram_binsearch_kernel, (const T *)src, strideSrc, (const int8_t *)weights, strideWeights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_UINT64: \
                DISPATCH_HIST_H_TYPE_S(T, uint64_t, s_histogram_binsearch_kernel, (const T *)src, strideSrc, (const uint64_t *)weights, strideWeights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_UINT32: \
                DISPATCH_HIST_H_TYPE_S(T, uint32_t, s_histogram_binsearch_kernel, (const T *)src, strideSrc, (const uint32_t *)weights, strideWeights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_UINT16: \
                DISPATCH_HIST_H_TYPE_S(T, uint16_t, s_histogram_binsearch_kernel, (const T *)src, strideSrc, (const uint16_t *)weights, strideWeights, bin_edges, num_edges, size) \
                break; \
            case DTYPE_UINT8: \
            case DTYPE_BOOLEAN: \
                DISPATCH_HIST_H_TYPE_S(T, uint8_t, s_histogram_binsearch_kernel, (const T *)src, strideSrc, (const uint8_t *)weights, strideWeights, bin_edges, num_edges, size) \
                break; \
            default: return; \
        } \
    }

extern "C" {
void v_histogram_uniform(
    const void *src, int dtypeSrc,
    const void *weights, int dtypeWeights,
    void *hist, int dtypeHist,
    int64_t size, int64_t nbins,
    double min_val, double max_val, double norm
) {
    if (src == nullptr || hist == nullptr || size <= 0 || nbins <= 0) return;
    switch (dtypeSrc) {
        case DTYPE_FLOAT64: DISPATCH_HIST_W_TYPE_V_UNIFORM(double); break;
        case DTYPE_FLOAT32: DISPATCH_HIST_W_TYPE_V_UNIFORM(float); break;
        case DTYPE_INT32: DISPATCH_HIST_W_TYPE_V_UNIFORM(int32_t); break;
        case DTYPE_INT64: DISPATCH_HIST_W_TYPE_V_UNIFORM(int64_t); break;
        case DTYPE_INT16: DISPATCH_HIST_W_TYPE_V_UNIFORM(int16_t); break;
        case DTYPE_INT8: DISPATCH_HIST_W_TYPE_V_UNIFORM(int8_t); break;
        case DTYPE_UINT64: DISPATCH_HIST_W_TYPE_V_UNIFORM(uint64_t); break;
        case DTYPE_UINT32: DISPATCH_HIST_W_TYPE_V_UNIFORM(uint32_t); break;
        case DTYPE_UINT16: DISPATCH_HIST_W_TYPE_V_UNIFORM(uint16_t); break;
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN: DISPATCH_HIST_W_TYPE_V_UNIFORM(uint8_t); break;
        default: return;
    }
}

void s_histogram_uniform(
    const void *src, int64_t strideSrc, int dtypeSrc,
    const void *weights, int64_t strideWeights, int dtypeWeights,
    void *hist, int64_t strideHist, int dtypeHist,
    int64_t size, int64_t nbins,
    double min_val, double max_val, double norm
) {
    if (src == nullptr || hist == nullptr || size <= 0 || nbins <= 0) return;
    switch (dtypeSrc) {
        case DTYPE_FLOAT64: DISPATCH_HIST_W_TYPE_S_UNIFORM(double); break;
        case DTYPE_FLOAT32: DISPATCH_HIST_W_TYPE_S_UNIFORM(float); break;
        case DTYPE_INT32: DISPATCH_HIST_W_TYPE_S_UNIFORM(int32_t); break;
        case DTYPE_INT64: DISPATCH_HIST_W_TYPE_S_UNIFORM(int64_t); break;
        case DTYPE_INT16: DISPATCH_HIST_W_TYPE_S_UNIFORM(int16_t); break;
        case DTYPE_INT8: DISPATCH_HIST_W_TYPE_S_UNIFORM(int8_t); break;
        case DTYPE_UINT64: DISPATCH_HIST_W_TYPE_S_UNIFORM(uint64_t); break;
        case DTYPE_UINT32: DISPATCH_HIST_W_TYPE_S_UNIFORM(uint32_t); break;
        case DTYPE_UINT16: DISPATCH_HIST_W_TYPE_S_UNIFORM(uint16_t); break;
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN: DISPATCH_HIST_W_TYPE_S_UNIFORM(uint8_t); break;
        default: return;
    }
}

void v_histogram_binsearch(
    const void *src, int dtypeSrc,
    const void *weights, int dtypeWeights,
    void *hist, int dtypeHist,
    const double *bin_edges, int64_t num_edges,
    int64_t size
) {
    if (src == nullptr || hist == nullptr || bin_edges == nullptr || size <= 0 || num_edges < 2) return;
    switch (dtypeSrc) {
        case DTYPE_FLOAT64: DISPATCH_HIST_W_TYPE_V_BINSEARCH(double); break;
        case DTYPE_FLOAT32: DISPATCH_HIST_W_TYPE_V_BINSEARCH(float); break;
        case DTYPE_INT32: DISPATCH_HIST_W_TYPE_V_BINSEARCH(int32_t); break;
        case DTYPE_INT64: DISPATCH_HIST_W_TYPE_V_BINSEARCH(int64_t); break;
        case DTYPE_INT16: DISPATCH_HIST_W_TYPE_V_BINSEARCH(int16_t); break;
        case DTYPE_INT8: DISPATCH_HIST_W_TYPE_V_BINSEARCH(int8_t); break;
        case DTYPE_UINT64: DISPATCH_HIST_W_TYPE_V_BINSEARCH(uint64_t); break;
        case DTYPE_UINT32: DISPATCH_HIST_W_TYPE_V_BINSEARCH(uint32_t); break;
        case DTYPE_UINT16: DISPATCH_HIST_W_TYPE_V_BINSEARCH(uint16_t); break;
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN: DISPATCH_HIST_W_TYPE_V_BINSEARCH(uint8_t); break;
        default: return;
    }
}

void s_histogram_binsearch(
    const void *src, int64_t strideSrc, int dtypeSrc,
    const void *weights, int64_t strideWeights, int dtypeWeights,
    void *hist, int64_t strideHist, int dtypeHist,
    const double *bin_edges, int64_t num_edges,
    int64_t size
) {
    if (src == nullptr || hist == nullptr || bin_edges == nullptr || size <= 0 || num_edges < 2) return;
    switch (dtypeSrc) {
        case DTYPE_FLOAT64: DISPATCH_HIST_W_TYPE_S_BINSEARCH(double); break;
        case DTYPE_FLOAT32: DISPATCH_HIST_W_TYPE_S_BINSEARCH(float); break;
        case DTYPE_INT32: DISPATCH_HIST_W_TYPE_S_BINSEARCH(int32_t); break;
        case DTYPE_INT64: DISPATCH_HIST_W_TYPE_S_BINSEARCH(int64_t); break;
        case DTYPE_INT16: DISPATCH_HIST_W_TYPE_S_BINSEARCH(int16_t); break;
        case DTYPE_INT8: DISPATCH_HIST_W_TYPE_S_BINSEARCH(int8_t); break;
        case DTYPE_UINT64: DISPATCH_HIST_W_TYPE_S_BINSEARCH(uint64_t); break;
        case DTYPE_UINT32: DISPATCH_HIST_W_TYPE_S_BINSEARCH(uint32_t); break;
        case DTYPE_UINT16: DISPATCH_HIST_W_TYPE_S_BINSEARCH(uint16_t); break;
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN: DISPATCH_HIST_W_TYPE_S_BINSEARCH(uint8_t); break;
        default: return;
    }
}
}

void v_vander_fit_double(const double *x, const double *y, const double *w,
                         double *v_mat, double *rhs,
                         int64_t m, int64_t deg) {
    if (x == nullptr || y == nullptr || v_mat == nullptr || rhs == nullptr || m <= 0 || deg < 0) {
        return;
    }
    int64_t n_cols = deg + 1;
    if (w != nullptr) {
        for (int64_t i = 0; i < m; i++) {
            double xi = x[i];
            double wi = w[i];
            rhs[i] = y[i] * wi;
            double *row = v_mat + i * n_cols;
            double cur = wi;
            row[deg] = cur;
            for (int64_t j = deg - 1; j >= 0; j--) {
                cur *= xi;
                row[j] = cur;
            }
        }
    } else {
        for (int64_t i = 0; i < m; i++) {
            double xi = x[i];
            rhs[i] = y[i];
            double *row = v_mat + i * n_cols;
            double cur = 1.0;
            row[deg] = 1.0;
            for (int64_t j = deg - 1; j >= 0; j--) {
                cur *= xi;
                row[j] = cur;
            }
        }
    }
}

void v_vander_fit_float(const float *x, const float *y, const float *w,
                        float *v_mat, float *rhs,
                        int64_t m, int64_t deg) {
    if (x == nullptr || y == nullptr || v_mat == nullptr || rhs == nullptr || m <= 0 || deg < 0) {
        return;
    }
    int64_t n_cols = deg + 1;
    if (w != nullptr) {
        for (int64_t i = 0; i < m; i++) {
            float xi = x[i];
            float wi = w[i];
            rhs[i] = y[i] * wi;
            float *row = v_mat + i * n_cols;
            float cur = wi;
            row[deg] = cur;
            for (int64_t j = deg - 1; j >= 0; j--) {
                cur *= xi;
                row[j] = cur;
            }
        }
    } else {
        for (int64_t i = 0; i < m; i++) {
            float xi = x[i];
            rhs[i] = y[i];
            float *row = v_mat + i * n_cols;
            float cur = 1.0f;
            row[deg] = 1.0f;
            for (int64_t j = deg - 1; j >= 0; j--) {
                cur *= xi;
                row[j] = cur;
            }
        }
    }
}

void v_vander_fit_complex128(const cpx_t *x, const cpx_t *y, const cpx_t *w,
                             cpx_t *v_mat, cpx_t *rhs,
                             int64_t m, int64_t deg) {
    if (x == nullptr || y == nullptr || v_mat == nullptr || rhs == nullptr || m <= 0 || deg < 0) {
        return;
    }
    int64_t n_cols = deg + 1;
    if (w != nullptr) {
        for (int64_t i = 0; i < m; i++) {
            cpx_t xi = x[i];
            cpx_t wi = w[i];
            rhs[i] = {y[i].r * wi.r - y[i].i * wi.i, y[i].r * wi.i + y[i].i * wi.r};
            cpx_t *row = v_mat + i * n_cols;
            cpx_t cur = wi;
            row[deg] = cur;
            for (int64_t j = deg - 1; j >= 0; j--) {
                cur = {cur.r * xi.r - cur.i * xi.i, cur.r * xi.i + cur.i * xi.r};
                row[j] = cur;
            }
        }
    } else {
        for (int64_t i = 0; i < m; i++) {
            cpx_t xi = x[i];
            rhs[i] = y[i];
            cpx_t *row = v_mat + i * n_cols;
            cpx_t cur = {1.0, 0.0};
            row[deg] = cur;
            for (int64_t j = deg - 1; j >= 0; j--) {
                cur = {cur.r * xi.r - cur.i * xi.i, cur.r * xi.i + cur.i * xi.r};
                row[j] = cur;
            }
        }
    }
}

void v_vander_fit_complex64(const cpx_f_t *x, const cpx_f_t *y, const cpx_f_t *w,
                            cpx_f_t *v_mat, cpx_f_t *rhs,
                            int64_t m, int64_t deg) {
    if (x == nullptr || y == nullptr || v_mat == nullptr || rhs == nullptr || m <= 0 || deg < 0) {
        return;
    }
    int64_t n_cols = deg + 1;
    if (w != nullptr) {
        for (int64_t i = 0; i < m; i++) {
            cpx_f_t xi = x[i];
            cpx_f_t wi = w[i];
            rhs[i] = {y[i].r * wi.r - y[i].i * wi.i, y[i].r * wi.i + y[i].i * wi.r};
            cpx_f_t *row = v_mat + i * n_cols;
            cpx_f_t cur = wi;
            row[deg] = cur;
            for (int64_t j = deg - 1; j >= 0; j--) {
                cur = {cur.r * xi.r - cur.i * xi.i, cur.r * xi.i + cur.i * xi.r};
                row[j] = cur;
            }
        }
    } else {
        for (int64_t i = 0; i < m; i++) {
            cpx_f_t xi = x[i];
            rhs[i] = y[i];
            cpx_f_t *row = v_mat + i * n_cols;
            cpx_f_t cur = {1.0f, 0.0f};
            row[deg] = cur;
            for (int64_t j = deg - 1; j >= 0; j--) {
                cur = {cur.r * xi.r - cur.i * xi.i, cur.r * xi.i + cur.i * xi.r};
                row[j] = cur;
            }
        }
    }
}

void s_vander_fit_double(const double *x, int64_t strideX,
                         const double *y, int64_t strideY,
                         const double *w, int64_t strideW,
                         double *v_mat, double *rhs,
                         int64_t m, int64_t deg) {
    if (x == nullptr || y == nullptr || v_mat == nullptr || rhs == nullptr || m <= 0 || deg < 0) {
        return;
    }
    int64_t n_cols = deg + 1;
    if (w != nullptr) {
        for (int64_t i = 0; i < m; i++) {
            double xi = x[i * strideX];
            double wi = w[i * strideW];
            rhs[i] = y[i * strideY] * wi;
            double *row = v_mat + i * n_cols;
            double cur = wi;
            row[deg] = cur;
            for (int64_t j = deg - 1; j >= 0; j--) {
                cur *= xi;
                row[j] = cur;
            }
        }
    } else {
        for (int64_t i = 0; i < m; i++) {
            double xi = x[i * strideX];
            rhs[i] = y[i * strideY];
            double *row = v_mat + i * n_cols;
            double cur = 1.0;
            row[deg] = 1.0;
            for (int64_t j = deg - 1; j >= 0; j--) {
                cur *= xi;
                row[j] = cur;
            }
        }
    }
}

void s_vander_fit_float(const float *x, int64_t strideX,
                        const float *y, int64_t strideY,
                        const float *w, int64_t strideW,
                        float *v_mat, float *rhs,
                        int64_t m, int64_t deg) {
    if (x == nullptr || y == nullptr || v_mat == nullptr || rhs == nullptr || m <= 0 || deg < 0) {
        return;
    }
    int64_t n_cols = deg + 1;
    if (w != nullptr) {
        for (int64_t i = 0; i < m; i++) {
            float xi = x[i * strideX];
            float wi = w[i * strideW];
            rhs[i] = y[i * strideY] * wi;
            float *row = v_mat + i * n_cols;
            float cur = wi;
            row[deg] = cur;
            for (int64_t j = deg - 1; j >= 0; j--) {
                cur *= xi;
                row[j] = cur;
            }
        }
    } else {
        for (int64_t i = 0; i < m; i++) {
            float xi = x[i * strideX];
            rhs[i] = y[i * strideY];
            float *row = v_mat + i * n_cols;
            float cur = 1.0f;
            row[deg] = 1.0f;
            for (int64_t j = deg - 1; j >= 0; j--) {
                cur *= xi;
                row[j] = cur;
            }
        }
    }
}

void s_vander_fit_complex128(const cpx_t *x, int64_t strideX,
                             const cpx_t *y, int64_t strideY,
                             const cpx_t *w, int64_t strideW,
                             cpx_t *v_mat, cpx_t *rhs,
                             int64_t m, int64_t deg) {
    if (x == nullptr || y == nullptr || v_mat == nullptr || rhs == nullptr || m <= 0 || deg < 0) {
        return;
    }
    int64_t n_cols = deg + 1;
    if (w != nullptr) {
        for (int64_t i = 0; i < m; i++) {
            cpx_t xi = x[i * strideX];
            cpx_t wi = w[i * strideW];
            rhs[i] = {y[i * strideY].r * wi.r - y[i * strideY].i * wi.i, y[i * strideY].r * wi.i + y[i * strideY].i * wi.r};
            cpx_t *row = v_mat + i * n_cols;
            cpx_t cur = wi;
            row[deg] = cur;
            for (int64_t j = deg - 1; j >= 0; j--) {
                cur = {cur.r * xi.r - cur.i * xi.i, cur.r * xi.i + cur.i * xi.r};
                row[j] = cur;
            }
        }
    } else {
        for (int64_t i = 0; i < m; i++) {
            cpx_t xi = x[i * strideX];
            rhs[i] = y[i * strideY];
            cpx_t *row = v_mat + i * n_cols;
            cpx_t cur = {1.0, 0.0};
            row[deg] = cur;
            for (int64_t j = deg - 1; j >= 0; j--) {
                cur = {cur.r * xi.r - cur.i * xi.i, cur.r * xi.i + cur.i * xi.r};
                row[j] = cur;
            }
        }
    }
}

void s_vander_fit_complex64(const cpx_f_t *x, int64_t strideX,
                            const cpx_f_t *y, int64_t strideY,
                            const cpx_f_t *w, int64_t strideW,
                            cpx_f_t *v_mat, cpx_f_t *rhs,
                            int64_t m, int64_t deg) {
    if (x == nullptr || y == nullptr || v_mat == nullptr || rhs == nullptr || m <= 0 || deg < 0) {
        return;
    }
    int64_t n_cols = deg + 1;
    if (w != nullptr) {
        for (int64_t i = 0; i < m; i++) {
            cpx_f_t xi = x[i * strideX];
            cpx_f_t wi = w[i * strideW];
            rhs[i] = {y[i * strideY].r * wi.r - y[i * strideY].i * wi.i, y[i * strideY].r * wi.i + y[i * strideY].i * wi.r};
            cpx_f_t *row = v_mat + i * n_cols;
            cpx_f_t cur = wi;
            row[deg] = cur;
            for (int64_t j = deg - 1; j >= 0; j--) {
                cur = {cur.r * xi.r - cur.i * xi.i, cur.r * xi.i + cur.i * xi.r};
                row[j] = cur;
            }
        }
    } else {
        for (int64_t i = 0; i < m; i++) {
            cpx_f_t xi = x[i * strideX];
            rhs[i] = y[i * strideY];
            cpx_f_t *row = v_mat + i * n_cols;
            cpx_f_t cur = {1.0f, 0.0f};
            row[deg] = cur;
            for (int64_t j = deg - 1; j >= 0; j--) {
                cur = {cur.r * xi.r - cur.i * xi.i, cur.r * xi.i + cur.i * xi.r};
                row[j] = cur;
            }
        }
    }
}

/* ============================================================================
 * SECTION: RANDOM SHUFFLE & CHOICE NATIVE ACCELERATORS IMPLEMENTATION
 * ============================================================================
 */

typedef struct {
    uint64_t a;
    uint64_t b;
} Item16;

template <typename T>
static inline void shuffle_1d_contiguous_impl(T* RESTRICT data, int64_t size, uint64_t s[4]) {
    for (int64_t i = size - 1; i > 0; i--) {
        int64_t j = (int64_t)(xoshiro256_next(s) % (uint64_t)(i + 1));
        if (i != j) {
            T temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
}

template <typename T>
static inline void shuffle_1d_strided_impl(T* RESTRICT data, int64_t size, int64_t stride, uint64_t s[4]) {
    for (int64_t i = size - 1; i > 0; i--) {
        int64_t j = (int64_t)(xoshiro256_next(s) % (uint64_t)(i + 1));
        if (i != j) {
            T temp = data[i * stride];
            data[i * stride] = data[j * stride];
            data[j * stride] = temp;
        }
    }
}

template <typename T>
static inline void choice_uniform_impl(
    const T* RESTRICT src, int64_t src_stride,
    T* RESTRICT dest, int64_t dest_stride,
    int64_t src_size, int64_t sample_count,
    uint64_t s[4]
) {
    if (src_stride == 1 && dest_stride == 1) {
        for (int64_t i = 0; i < sample_count; i++) {
            uint64_t idx = xoshiro256_next(s) % (uint64_t)src_size;
            dest[i] = src[idx];
        }
    } else {
        for (int64_t i = 0; i < sample_count; i++) {
            uint64_t idx = xoshiro256_next(s) % (uint64_t)src_size;
            dest[i * dest_stride] = src[idx * src_stride];
        }
    }
}

template <typename T>
static inline void choice_weighted_impl(
    const T* RESTRICT src, int64_t src_stride,
    T* RESTRICT dest, int64_t dest_stride,
    const double* RESTRICT cdf,
    int64_t src_size, int64_t sample_count,
    uint64_t s[4]
) {
    if (src_stride == 1 && dest_stride == 1) {
        for (int64_t i = 0; i < sample_count; i++) {
            double u = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
            const double* it = std::upper_bound(cdf, cdf + src_size, u);
            int64_t idx = (int64_t)(it - cdf);
            if (idx >= src_size) idx = src_size - 1;
            dest[i] = src[idx];
        }
    } else {
        for (int64_t i = 0; i < sample_count; i++) {
            double u = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
            const double* it = std::upper_bound(cdf, cdf + src_size, u);
            int64_t idx = (int64_t)(it - cdf);
            if (idx >= src_size) idx = src_size - 1;
            dest[i * dest_stride] = src[idx * src_stride];
        }
    }
}

template <typename T>
static inline void choice_without_replacement_impl(
    const T* RESTRICT src, int64_t src_stride,
    T* RESTRICT dest, int64_t dest_stride,
    int64_t src_size, int64_t sample_count,
    uint64_t s[4]
) {
    if (sample_count == src_size && src_stride == 1 && dest_stride == 1) {
        memcpy(dest, src, src_size * sizeof(T));
        shuffle_1d_contiguous_impl<T>(dest, src_size, s);
        return;
    }

    int64_t* indices = (int64_t*)malloc(src_size * sizeof(int64_t));
    if (!indices) {
        ndarray_set_oom_flag();
        return;
    }
    for (int64_t i = 0; i < src_size; i++) {
        indices[i] = i;
    }

    if (src_stride == 1 && dest_stride == 1) {
        for (int64_t i = 0; i < sample_count; i++) {
            int64_t j = i + (int64_t)(xoshiro256_next(s) % (uint64_t)(src_size - i));
            int64_t chosen = indices[j];
            indices[j] = indices[i];
            dest[i] = src[chosen];
        }
    } else {
        for (int64_t i = 0; i < sample_count; i++) {
            int64_t j = i + (int64_t)(xoshiro256_next(s) % (uint64_t)(src_size - i));
            int64_t chosen = indices[j];
            indices[j] = indices[i];
            dest[i * dest_stride] = src[chosen * src_stride];
        }
    }

    free(indices);
}

template <typename T>
static inline void choice_weighted_without_replacement_impl(
    const T* RESTRICT src, int64_t src_stride,
    T* RESTRICT dest, int64_t dest_stride,
    const double* RESTRICT probs,
    int64_t src_size, int64_t sample_count,
    uint64_t s[4]
) {
    double* temp_probs = (double*)malloc(src_size * sizeof(double));
    uint8_t* drawn = (uint8_t*)calloc(src_size, sizeof(uint8_t));
    if (!temp_probs || !drawn) {
        if (temp_probs) free(temp_probs);
        if (drawn) free(drawn);
        ndarray_set_oom_flag();
        return;
    }

    for (int64_t i = 0; i < src_size; i++) {
        temp_probs[i] = probs[i];
    }

    for (int64_t draw = 0; draw < sample_count; draw++) {
        double sumP = 0.0;
        for (int64_t i = 0; i < src_size; i++) {
            if (!drawn[i]) {
                sumP += temp_probs[i];
            }
        }

        int64_t index = src_size - 1;
        if (sumP > 0.0) {
            double u = ((double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0)) * sumP;
            double cum = 0.0;
            for (int64_t j = 0; j < src_size; j++) {
                if (!drawn[j]) {
                    cum += temp_probs[j];
                    if (cum > u || (cum == u && temp_probs[j] > 0.0)) {
                        index = j;
                        break;
                    }
                }
            }
        } else {
            int64_t remaining = src_size - draw;
            int64_t pick = (remaining > 0) ? (int64_t)(xoshiro256_next(s) % (uint64_t)remaining) : 0;
            int64_t count = 0;
            for (int64_t j = 0; j < src_size; j++) {
                if (!drawn[j]) {
                    if (count == pick) {
                        index = j;
                        break;
                    }
                    count++;
                }
            }
        }

        drawn[index] = 1;
        if (src_stride == 1 && dest_stride == 1) {
            dest[draw] = src[index];
        } else {
            dest[draw * dest_stride] = src[index * src_stride];
        }
    }

    free(temp_probs);
    free(drawn);
}

extern "C" {

void native_shuffle_1d(
    void *data,
    int64_t size,
    int64_t stride,
    int item_size,
    uint64_t seed
) {
    if (data == nullptr || size <= 1 || item_size <= 0) return;
    uint64_t s[4];
    xoshiro256_seed(seed, s);

    if (stride == 1) {
        switch (item_size) {
            case 1:
                shuffle_1d_contiguous_impl<uint8_t>((uint8_t*)data, size, s);
                break;
            case 2:
                shuffle_1d_contiguous_impl<uint16_t>((uint16_t*)data, size, s);
                break;
            case 4:
                shuffle_1d_contiguous_impl<uint32_t>((uint32_t*)data, size, s);
                break;
            case 8:
                shuffle_1d_contiguous_impl<uint64_t>((uint64_t*)data, size, s);
                break;
            case 16:
                shuffle_1d_contiguous_impl<Item16>((Item16*)data, size, s);
                break;
            default: {
                char stack_buf[256];
                char* temp = (item_size <= (int)sizeof(stack_buf)) ? stack_buf : (char*)malloc(item_size);
                if (!temp) {
                    ndarray_set_oom_flag();
                    return;
                }
                char* ptr = (char*)data;
                for (int64_t i = size - 1; i > 0; i--) {
                    int64_t j = (int64_t)(xoshiro256_next(s) % (uint64_t)(i + 1));
                    if (i != j) {
                        memcpy(temp, ptr + i * item_size, item_size);
                        memcpy(ptr + i * item_size, ptr + j * item_size, item_size);
                        memcpy(ptr + j * item_size, temp, item_size);
                    }
                }
                if (temp != stack_buf) free(temp);
                break;
            }
        }
    } else {
        switch (item_size) {
            case 1:
                shuffle_1d_strided_impl<uint8_t>((uint8_t*)data, size, stride, s);
                break;
            case 2:
                shuffle_1d_strided_impl<uint16_t>((uint16_t*)data, size, stride, s);
                break;
            case 4:
                shuffle_1d_strided_impl<uint32_t>((uint32_t*)data, size, stride, s);
                break;
            case 8:
                shuffle_1d_strided_impl<uint64_t>((uint64_t*)data, size, stride, s);
                break;
            case 16:
                shuffle_1d_strided_impl<Item16>((Item16*)data, size, stride, s);
                break;
            default: {
                char stack_buf[256];
                char* temp = (item_size <= (int)sizeof(stack_buf)) ? stack_buf : (char*)malloc(item_size);
                if (!temp) {
                    ndarray_set_oom_flag();
                    return;
                }
                char* ptr = (char*)data;
                for (int64_t i = size - 1; i > 0; i--) {
                    int64_t j = (int64_t)(xoshiro256_next(s) % (uint64_t)(i + 1));
                    if (i != j) {
                        memcpy(temp, ptr + i * stride * item_size, item_size);
                        memcpy(ptr + i * stride * item_size, ptr + j * stride * item_size, item_size);
                        memcpy(ptr + j * stride * item_size, temp, item_size);
                    }
                }
                if (temp != stack_buf) free(temp);
                break;
            }
        }
    }
}

static void copy_slice_strided_recursive(
    const char* src, int64_t src_offset, const int64_t* shape, const int64_t* strides_src,
    char* dest, int64_t dest_offset, const int64_t* strides_dest,
    int dim, int rank, int item_size
) {
    int64_t count = shape[dim];
    int64_t step_src = strides_src[dim] * item_size;
    int64_t step_dest = strides_dest[dim] * item_size;
    if (dim == rank - 1) {
        for (int64_t i = 0; i < count; i++) {
            memcpy(dest + dest_offset + i * step_dest, src + src_offset + i * step_src, item_size);
        }
    } else {
        for (int64_t i = 0; i < count; i++) {
            copy_slice_strided_recursive(
                src, src_offset + i * step_src, shape, strides_src,
                dest, dest_offset + i * step_dest, strides_dest,
                dim + 1, rank, item_size
            );
        }
    }
}

void native_shuffle_nd(
    void *data,
    const int64_t *shape,
    const int64_t *strides,
    int rank,
    int item_size,
    uint64_t seed
) {
    if (data == nullptr || shape == nullptr || strides == nullptr || rank <= 0 || item_size <= 0) return;
    int64_t d0 = shape[0];
    if (d0 <= 1) return;

    if (rank == 1) {
        native_shuffle_1d(data, shape[0], strides[0], item_size, seed);
        return;
    }

    uint64_t s[4];
    xoshiro256_seed(seed, s);

    int64_t slice_elements = 1;
    for (int64_t i = 1; i < rank; i++) {
        slice_elements *= shape[i];
    }
    if (slice_elements <= 0) return;

    bool slice_is_contiguous = true;
    int64_t expected = 1;
    for (int64_t i = rank - 1; i >= 1; i--) {
        if (strides[i] != expected) {
            slice_is_contiguous = false;
            break;
        }
        expected *= shape[i];
    }

    int64_t slice_bytes = slice_elements * item_size;
    char stack_buf[4096];
    char* temp = (slice_bytes <= (int64_t)sizeof(stack_buf)) ? stack_buf : (char*)malloc(slice_bytes);
    if (!temp) {
        ndarray_set_oom_flag();
        return;
    }

    char* base = (char*)data;
    if (slice_is_contiguous) {
        int64_t step_bytes = strides[0] * item_size;
        for (int64_t i = d0 - 1; i > 0; i--) {
            int64_t j = (int64_t)(xoshiro256_next(s) % (uint64_t)(i + 1));
            if (i != j) {
                char* pi = base + i * step_bytes;
                char* pj = base + j * step_bytes;
                memcpy(temp, pi, slice_bytes);
                memcpy(pi, pj, slice_bytes);
                memcpy(pj, temp, slice_bytes);
            }
        }
    } else {
        DECLARE_RANK_BUFFER(int64_t, temp_strides, rank);
        temp_strides[rank - 1] = 1;
        for (int64_t i = rank - 2; i >= 1; i--) {
            temp_strides[i] = temp_strides[i + 1] * shape[i + 1];
        }
        int64_t step_i_bytes = strides[0] * item_size;
        for (int64_t i = d0 - 1; i > 0; i--) {
            int64_t j = (int64_t)(xoshiro256_next(s) % (uint64_t)(i + 1));
            if (i != j) {
                int64_t offsetI = i * step_i_bytes;
                int64_t offsetJ = j * step_i_bytes;
                copy_slice_strided_recursive(base, offsetI, shape, strides, temp, 0, temp_strides, 1, rank, item_size);
                copy_slice_strided_recursive(base, offsetJ, shape, strides, base, offsetI, strides, 1, rank, item_size);
                copy_slice_strided_recursive(temp, 0, shape, temp_strides, base, offsetJ, strides, 1, rank, item_size);
            }
        }
    }

    if (temp != stack_buf) {
        free(temp);
    }
}

void native_choice_uniform(
    const void *src,
    int64_t src_stride,
    void *dest,
    int64_t dest_stride,
    int64_t src_size,
    int64_t sample_count,
    int item_size,
    uint64_t seed
) {
    if (src == nullptr || dest == nullptr || src_size <= 0 || sample_count <= 0 || item_size <= 0) return;
    uint64_t s[4];
    xoshiro256_seed(seed, s);

    switch (item_size) {
        case 1:
            choice_uniform_impl<uint8_t>((const uint8_t*)src, src_stride, (uint8_t*)dest, dest_stride, src_size, sample_count, s);
            break;
        case 2:
            choice_uniform_impl<uint16_t>((const uint16_t*)src, src_stride, (uint16_t*)dest, dest_stride, src_size, sample_count, s);
            break;
        case 4:
            choice_uniform_impl<uint32_t>((const uint32_t*)src, src_stride, (uint32_t*)dest, dest_stride, src_size, sample_count, s);
            break;
        case 8:
            choice_uniform_impl<uint64_t>((const uint64_t*)src, src_stride, (uint64_t*)dest, dest_stride, src_size, sample_count, s);
            break;
        case 16:
            choice_uniform_impl<Item16>((const Item16*)src, src_stride, (Item16*)dest, dest_stride, src_size, sample_count, s);
            break;
        default: {
            const char* s_src = (const char*)src;
            char* d_dest = (char*)dest;
            for (int64_t i = 0; i < sample_count; i++) {
                uint64_t idx = xoshiro256_next(s) % (uint64_t)src_size;
                memcpy(d_dest + i * dest_stride * item_size, s_src + idx * src_stride * item_size, item_size);
            }
            break;
        }
    }
}

void native_choice_weighted(
    const void *src,
    int64_t src_stride,
    void *dest,
    int64_t dest_stride,
    const double *cdf,
    int64_t src_size,
    int64_t sample_count,
    int item_size,
    uint64_t seed
) {
    if (src == nullptr || dest == nullptr || cdf == nullptr || src_size <= 0 || sample_count <= 0 || item_size <= 0) return;
    uint64_t s[4];
    xoshiro256_seed(seed, s);

    switch (item_size) {
        case 1:
            choice_weighted_impl<uint8_t>((const uint8_t*)src, src_stride, (uint8_t*)dest, dest_stride, cdf, src_size, sample_count, s);
            break;
        case 2:
            choice_weighted_impl<uint16_t>((const uint16_t*)src, src_stride, (uint16_t*)dest, dest_stride, cdf, src_size, sample_count, s);
            break;
        case 4:
            choice_weighted_impl<uint32_t>((const uint32_t*)src, src_stride, (uint32_t*)dest, dest_stride, cdf, src_size, sample_count, s);
            break;
        case 8:
            choice_weighted_impl<uint64_t>((const uint64_t*)src, src_stride, (uint64_t*)dest, dest_stride, cdf, src_size, sample_count, s);
            break;
        case 16:
            choice_weighted_impl<Item16>((const Item16*)src, src_stride, (Item16*)dest, dest_stride, cdf, src_size, sample_count, s);
            break;
        default: {
            const char* s_src = (const char*)src;
            char* d_dest = (char*)dest;
            for (int64_t i = 0; i < sample_count; i++) {
                double u = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
                const double* it = std::upper_bound(cdf, cdf + src_size, u);
                int64_t idx = (int64_t)(it - cdf);
                if (idx >= src_size) idx = src_size - 1;
                memcpy(d_dest + i * dest_stride * item_size, s_src + idx * src_stride * item_size, item_size);
            }
            break;
        }
    }
}

void native_choice_without_replacement(
    const void *src,
    int64_t src_stride,
    void *dest,
    int64_t dest_stride,
    int64_t src_size,
    int64_t sample_count,
    int item_size,
    uint64_t seed
) {
    if (src == nullptr || dest == nullptr || src_size <= 0 || sample_count <= 0 || item_size <= 0 || sample_count > src_size) return;
    uint64_t s[4];
    xoshiro256_seed(seed, s);

    switch (item_size) {
        case 1:
            choice_without_replacement_impl<uint8_t>((const uint8_t*)src, src_stride, (uint8_t*)dest, dest_stride, src_size, sample_count, s);
            break;
        case 2:
            choice_without_replacement_impl<uint16_t>((const uint16_t*)src, src_stride, (uint16_t*)dest, dest_stride, src_size, sample_count, s);
            break;
        case 4:
            choice_without_replacement_impl<uint32_t>((const uint32_t*)src, src_stride, (uint32_t*)dest, dest_stride, src_size, sample_count, s);
            break;
        case 8:
            choice_without_replacement_impl<uint64_t>((const uint64_t*)src, src_stride, (uint64_t*)dest, dest_stride, src_size, sample_count, s);
            break;
        case 16:
            choice_without_replacement_impl<Item16>((const Item16*)src, src_stride, (Item16*)dest, dest_stride, src_size, sample_count, s);
            break;
        default: {
            int64_t* indices = (int64_t*)malloc(src_size * sizeof(int64_t));
            if (!indices) {
                ndarray_set_oom_flag();
                return;
            }
            for (int64_t i = 0; i < src_size; i++) indices[i] = i;
            const char* s_src = (const char*)src;
            char* d_dest = (char*)dest;
            for (int64_t i = 0; i < sample_count; i++) {
                int64_t j = i + (int64_t)(xoshiro256_next(s) % (uint64_t)(src_size - i));
                int64_t chosen = indices[j];
                indices[j] = indices[i];
                memcpy(d_dest + i * dest_stride * item_size, s_src + chosen * src_stride * item_size, item_size);
            }
            free(indices);
            break;
        }
    }
}

void native_choice_weighted_without_replacement(
    const void *src,
    int64_t src_stride,
    void *dest,
    int64_t dest_stride,
    const double *probs,
    int64_t src_size,
    int64_t sample_count,
    int item_size,
    uint64_t seed
) {
    if (src == nullptr || dest == nullptr || probs == nullptr || src_size <= 0 || sample_count <= 0 || item_size <= 0 || sample_count > src_size) return;
    uint64_t s[4];
    xoshiro256_seed(seed, s);

    switch (item_size) {
        case 1:
            choice_weighted_without_replacement_impl<uint8_t>((const uint8_t*)src, src_stride, (uint8_t*)dest, dest_stride, probs, src_size, sample_count, s);
            break;
        case 2:
            choice_weighted_without_replacement_impl<uint16_t>((const uint16_t*)src, src_stride, (uint16_t*)dest, dest_stride, probs, src_size, sample_count, s);
            break;
        case 4:
            choice_weighted_without_replacement_impl<uint32_t>((const uint32_t*)src, src_stride, (uint32_t*)dest, dest_stride, probs, src_size, sample_count, s);
            break;
        case 8:
            choice_weighted_without_replacement_impl<uint64_t>((const uint64_t*)src, src_stride, (uint64_t*)dest, dest_stride, probs, src_size, sample_count, s);
            break;
        case 16:
            choice_weighted_without_replacement_impl<Item16>((const Item16*)src, src_stride, (Item16*)dest, dest_stride, probs, src_size, sample_count, s);
            break;
        default: {
            double* temp_probs = (double*)malloc(src_size * sizeof(double));
            uint8_t* drawn = (uint8_t*)calloc(src_size, sizeof(uint8_t));
            if (!temp_probs || !drawn) {
                if (temp_probs) free(temp_probs);
                if (drawn) free(drawn);
                ndarray_set_oom_flag();
                return;
            }
            for (int64_t i = 0; i < src_size; i++) temp_probs[i] = probs[i];
            const char* s_src = (const char*)src;
            char* d_dest = (char*)dest;

            for (int64_t draw = 0; draw < sample_count; draw++) {
                double sumP = 0.0;
                for (int64_t i = 0; i < src_size; i++) {
                    if (!drawn[i]) sumP += temp_probs[i];
                }
                int64_t index = src_size - 1;
                if (sumP > 0.0) {
                    double u = ((double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0)) * sumP;
                    double cum = 0.0;
                    for (int64_t j = 0; j < src_size; j++) {
                        if (!drawn[j]) {
                            cum += temp_probs[j];
                            if (cum > u || (cum == u && temp_probs[j] > 0.0)) {
                                index = j;
                                break;
                            }
                        }
                    }
                } else {
                    int64_t remaining = src_size - draw;
                    int64_t pick = (remaining > 0) ? (int64_t)(xoshiro256_next(s) % (uint64_t)remaining) : 0;
                    int64_t count = 0;
                    for (int64_t j = 0; j < src_size; j++) {
                        if (!drawn[j]) {
                            if (count == pick) {
                                index = j;
                                break;
                            }
                            count++;
                        }
                    }
                }
                drawn[index] = 1;
                memcpy(d_dest + draw * dest_stride * item_size, s_src + index * src_stride * item_size, item_size);
            }
            free(temp_probs);
            free(drawn);
            break;
        }
    }
}

void native_secure_shuffle_1d(
    void *data,
    int64_t size,
    int64_t stride,
    int item_size
) {
    if (data == nullptr || size <= 1 || item_size <= 0) return;
    SecureRng rng;
    char stack_buf[256];
    char* temp = (item_size <= (int)sizeof(stack_buf)) ? stack_buf : (char*)malloc(item_size);
    if (!temp) {
        ndarray_set_oom_flag();
        return;
    }
    char* ptr = (char*)data;
    for (int64_t i = size - 1; i > 0; i--) {
        uint64_t r = rng.next_uint64();
        int64_t j = (int64_t)(r % (uint64_t)(i + 1));
        if (i != j) {
            memcpy(temp, ptr + i * stride * item_size, item_size);
            memcpy(ptr + i * stride * item_size, ptr + j * stride * item_size, item_size);
            memcpy(ptr + j * stride * item_size, temp, item_size);
        }
    }
    if (temp != stack_buf) free(temp);
}

void native_secure_shuffle_nd(
    void *data,
    const int64_t *shape,
    const int64_t *strides,
    int rank,
    int item_size
) {
    if (data == nullptr || shape == nullptr || strides == nullptr || rank <= 0 || item_size <= 0) return;
    int64_t d0 = shape[0];
    if (d0 <= 1) return;
    if (rank == 1) {
        native_secure_shuffle_1d(data, shape[0], strides[0], item_size);
        return;
    }

    int64_t slice_elements = 1;
    for (int64_t i = 1; i < rank; i++) {
        slice_elements *= shape[i];
    }
    if (slice_elements <= 0) return;

    bool slice_is_contiguous = true;
    int64_t expected = 1;
    for (int64_t i = rank - 1; i >= 1; i--) {
        if (strides[i] != expected) {
            slice_is_contiguous = false;
            break;
        }
        expected *= shape[i];
    }

    int64_t slice_bytes = slice_elements * item_size;
    char stack_buf[4096];
    char* temp = (slice_bytes <= (int64_t)sizeof(stack_buf)) ? stack_buf : (char*)malloc(slice_bytes);
    if (!temp) {
        ndarray_set_oom_flag();
        return;
    }

    SecureRng rng;
    char* base = (char*)data;
    if (slice_is_contiguous) {
        int64_t step_bytes = strides[0] * item_size;
        for (int64_t i = d0 - 1; i > 0; i--) {
            uint64_t r = rng.next_uint64();
            int64_t j = (int64_t)(r % (uint64_t)(i + 1));
            if (i != j) {
                char* pi = base + i * step_bytes;
                char* pj = base + j * step_bytes;
                memcpy(temp, pi, slice_bytes);
                memcpy(pi, pj, slice_bytes);
                memcpy(pj, temp, slice_bytes);
            }
        }
    } else {
        DECLARE_RANK_BUFFER(int64_t, temp_strides, rank);
        temp_strides[rank - 1] = 1;
        for (int64_t i = rank - 2; i >= 1; i--) {
            temp_strides[i] = temp_strides[i + 1] * shape[i + 1];
        }
        int64_t step_i_bytes = strides[0] * item_size;
        for (int64_t i = d0 - 1; i > 0; i--) {
            uint64_t r = rng.next_uint64();
            int64_t j = (int64_t)(r % (uint64_t)(i + 1));
            if (i != j) {
                int64_t offsetI = i * step_i_bytes;
                int64_t offsetJ = j * step_i_bytes;
                copy_slice_strided_recursive(base, offsetI, shape, strides, temp, 0, temp_strides, 1, rank, item_size);
                copy_slice_strided_recursive(base, offsetJ, shape, strides, base, offsetI, strides, 1, rank, item_size);
                copy_slice_strided_recursive(temp, 0, shape, temp_strides, base, offsetJ, strides, 1, rank, item_size);
            }
        }
    }

    if (temp != stack_buf) {
        free(temp);
    }
}

void native_secure_choice_uniform(
    const void *src,
    int64_t src_stride,
    void *dest,
    int64_t dest_stride,
    int64_t src_size,
    int64_t sample_count,
    int item_size
) {
    if (src == nullptr || dest == nullptr || src_size <= 0 || sample_count <= 0 || item_size <= 0) return;
    SecureRng rng;
    const char* s_src = (const char*)src;
    char* d_dest = (char*)dest;
    for (int64_t i = 0; i < sample_count; i++) {
        uint64_t r = rng.next_uint64();
        uint64_t idx = r % (uint64_t)src_size;
        memcpy(d_dest + i * dest_stride * item_size, s_src + idx * src_stride * item_size, item_size);
    }
}

void native_secure_choice_weighted(
    const void *src,
    int64_t src_stride,
    void *dest,
    int64_t dest_stride,
    const double *cdf,
    int64_t src_size,
    int64_t sample_count,
    int item_size
) {
    if (src == nullptr || dest == nullptr || cdf == nullptr || src_size <= 0 || sample_count <= 0 || item_size <= 0) return;
    SecureRng rng;
    const char* s_src = (const char*)src;
    char* d_dest = (char*)dest;
    for (int64_t i = 0; i < sample_count; i++) {
        double u = rng.next_double();
        const double* it = std::upper_bound(cdf, cdf + src_size, u);
        int64_t idx = (int64_t)(it - cdf);
        if (idx >= src_size) idx = src_size - 1;
        memcpy(d_dest + i * dest_stride * item_size, s_src + idx * src_stride * item_size, item_size);
    }
}

void native_secure_choice_without_replacement(
    const void *src,
    int64_t src_stride,
    void *dest,
    int64_t dest_stride,
    int64_t src_size,
    int64_t sample_count,
    int item_size
) {
    if (src == nullptr || dest == nullptr || src_size <= 0 || sample_count <= 0 || item_size <= 0 || sample_count > src_size) return;
    int64_t* indices = (int64_t*)malloc(src_size * sizeof(int64_t));
    if (!indices) {
        ndarray_set_oom_flag();
        return;
    }
    for (int64_t i = 0; i < src_size; i++) indices[i] = i;
    SecureRng rng;
    const char* s_src = (const char*)src;
    char* d_dest = (char*)dest;
    for (int64_t i = 0; i < sample_count; i++) {
        uint64_t r = rng.next_uint64();
        int64_t j = i + (int64_t)(r % (uint64_t)(src_size - i));
        int64_t chosen = indices[j];
        indices[j] = indices[i];
        memcpy(d_dest + i * dest_stride * item_size, s_src + chosen * src_stride * item_size, item_size);
    }
    free(indices);
}

void native_secure_choice_weighted_without_replacement(
    const void *src,
    int64_t src_stride,
    void *dest,
    int64_t dest_stride,
    const double *probs,
    int64_t src_size,
    int64_t sample_count,
    int item_size
) {
    if (src == nullptr || dest == nullptr || probs == nullptr || src_size <= 0 || sample_count <= 0 || item_size <= 0 || sample_count > src_size) return;
    double* temp_probs = (double*)malloc(src_size * sizeof(double));
    uint8_t* drawn = (uint8_t*)calloc(src_size, sizeof(uint8_t));
    if (!temp_probs || !drawn) {
        if (temp_probs) free(temp_probs);
        if (drawn) free(drawn);
        ndarray_set_oom_flag();
        return;
    }
    for (int64_t i = 0; i < src_size; i++) temp_probs[i] = probs[i];
    SecureRng rng;
    const char* s_src = (const char*)src;
    char* d_dest = (char*)dest;

    for (int64_t draw = 0; draw < sample_count; draw++) {
        double sumP = 0.0;
        for (int64_t i = 0; i < src_size; i++) {
            if (!drawn[i]) sumP += temp_probs[i];
        }
        int64_t index = src_size - 1;
        if (sumP > 0.0) {
            double u = rng.next_double() * sumP;
            double cum = 0.0;
            for (int64_t j = 0; j < src_size; j++) {
                if (!drawn[j]) {
                    cum += temp_probs[j];
                    if (cum > u || (cum == u && temp_probs[j] > 0.0)) {
                        index = j;
                        break;
                    }
                }
            }
        } else {
            int64_t remaining = src_size - draw;
            uint64_t r = rng.next_uint64();
            int64_t pick = (remaining > 0) ? (int64_t)(r % (uint64_t)remaining) : 0;
            int64_t count = 0;
            for (int64_t j = 0; j < src_size; j++) {
                if (!drawn[j]) {
                    if (count == pick) {
                        index = j;
                        break;
                    }
                    count++;
                }
            }
        }
        drawn[index] = 1;
        memcpy(d_dest + draw * dest_stride * item_size, s_src + index * src_stride * item_size, item_size);
    }
    free(temp_probs);
    free(drawn);
}

double r_nansum_double(const double *src, int64_t size) {
    if (src == nullptr || size <= 0) return 0.0;
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    int64_t i = 0;
    int64_t limit = size - 3;
    for (; i < limit; i += 4) {
        double v0 = src[i];
        double v1 = src[i + 1];
        double v2 = src[i + 2];
        double v3 = src[i + 3];
        if (!std::isnan(v0)) sum0 += v0;
        if (!std::isnan(v1)) sum1 += v1;
        if (!std::isnan(v2)) sum2 += v2;
        if (!std::isnan(v3)) sum3 += v3;
    }
    for (; i < size; i++) {
        double v = src[i];
        if (!std::isnan(v)) sum0 += v;
    }
    return (sum0 + sum1) + (sum2 + sum3);
}

double r_nansum_float(const float *src, int64_t size) {
    if (src == nullptr || size <= 0) return 0.0;
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    int64_t i = 0;
    int64_t limit = size - 3;
    for (; i < limit; i += 4) {
        float v0 = src[i];
        float v1 = src[i + 1];
        float v2 = src[i + 2];
        float v3 = src[i + 3];
        if (!std::isnan(v0)) sum0 += (double)v0;
        if (!std::isnan(v1)) sum1 += (double)v1;
        if (!std::isnan(v2)) sum2 += (double)v2;
        if (!std::isnan(v3)) sum3 += (double)v3;
    }
    for (; i < size; i++) {
        float v = src[i];
        if (!std::isnan(v)) sum0 += (double)v;
    }
    return (sum0 + sum1) + (sum2 + sum3);
}

double r_nanmean_double(const double *src, int64_t size) {
    if (src == nullptr || size <= 0) return NAN;
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    int64_t count = 0;
    int64_t i = 0;
    int64_t limit = size - 3;
    for (; i < limit; i += 4) {
        double v0 = src[i];
        double v1 = src[i + 1];
        double v2 = src[i + 2];
        double v3 = src[i + 3];
        if (!std::isnan(v0)) { sum0 += v0; count++; }
        if (!std::isnan(v1)) { sum1 += v1; count++; }
        if (!std::isnan(v2)) { sum2 += v2; count++; }
        if (!std::isnan(v3)) { sum3 += v3; count++; }
    }
    for (; i < size; i++) {
        double v = src[i];
        if (!std::isnan(v)) { sum0 += v; count++; }
    }
    return count == 0 ? NAN : ((sum0 + sum1) + (sum2 + sum3)) / (double)count;
}

double r_nanmean_float(const float *src, int64_t size) {
    if (src == nullptr || size <= 0) return NAN;
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    int64_t count = 0;
    int64_t i = 0;
    int64_t limit = size - 3;
    for (; i < limit; i += 4) {
        float v0 = src[i];
        float v1 = src[i + 1];
        float v2 = src[i + 2];
        float v3 = src[i + 3];
        if (!std::isnan(v0)) { sum0 += (double)v0; count++; }
        if (!std::isnan(v1)) { sum1 += (double)v1; count++; }
        if (!std::isnan(v2)) { sum2 += (double)v2; count++; }
        if (!std::isnan(v3)) { sum3 += (double)v3; count++; }
    }
    for (; i < size; i++) {
        float v = src[i];
        if (!std::isnan(v)) { sum0 += (double)v; count++; }
    }
    return count == 0 ? NAN : ((sum0 + sum1) + (sum2 + sum3)) / (double)count;
}

double r_nanvar_double(const double *src, int64_t size) {
    if (src == nullptr || size <= 0) return NAN;
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    int64_t count = 0;
    int64_t i = 0;
    int64_t limit = size - 3;
    for (; i < limit; i += 4) {
        double v0 = src[i];
        double v1 = src[i + 1];
        double v2 = src[i + 2];
        double v3 = src[i + 3];
        if (!std::isnan(v0)) { sum0 += v0; count++; }
        if (!std::isnan(v1)) { sum1 += v1; count++; }
        if (!std::isnan(v2)) { sum2 += v2; count++; }
        if (!std::isnan(v3)) { sum3 += v3; count++; }
    }
    for (; i < size; i++) {
        double v = src[i];
        if (!std::isnan(v)) { sum0 += v; count++; }
    }
    if (count == 0) return NAN;
    double meanVal = ((sum0 + sum1) + (sum2 + sum3)) / (double)count;
    double sq0 = 0.0, sq1 = 0.0, sq2 = 0.0, sq3 = 0.0;
    i = 0;
    for (; i < limit; i += 4) {
        double v0 = src[i];
        double v1 = src[i + 1];
        double v2 = src[i + 2];
        double v3 = src[i + 3];
        if (!std::isnan(v0)) { double d0 = v0 - meanVal; sq0 += d0 * d0; }
        if (!std::isnan(v1)) { double d1 = v1 - meanVal; sq1 += d1 * d1; }
        if (!std::isnan(v2)) { double d2 = v2 - meanVal; sq2 += d2 * d2; }
        if (!std::isnan(v3)) { double d3 = v3 - meanVal; sq3 += d3 * d3; }
    }
    for (; i < size; i++) {
        double v = src[i];
        if (!std::isnan(v)) { double d = v - meanVal; sq0 += d * d; }
    }
    return ((sq0 + sq1) + (sq2 + sq3)) / (double)count;
}

double r_nanvar_float(const float *src, int64_t size) {
    if (src == nullptr || size <= 0) return NAN;
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    int64_t count = 0;
    int64_t i = 0;
    int64_t limit = size - 3;
    for (; i < limit; i += 4) {
        float v0 = src[i];
        float v1 = src[i + 1];
        float v2 = src[i + 2];
        float v3 = src[i + 3];
        if (!std::isnan(v0)) { sum0 += (double)v0; count++; }
        if (!std::isnan(v1)) { sum1 += (double)v1; count++; }
        if (!std::isnan(v2)) { sum2 += (double)v2; count++; }
        if (!std::isnan(v3)) { sum3 += (double)v3; count++; }
    }
    for (; i < size; i++) {
        float v = src[i];
        if (!std::isnan(v)) { sum0 += (double)v; count++; }
    }
    if (count == 0) return NAN;
    double meanVal = ((sum0 + sum1) + (sum2 + sum3)) / (double)count;
    double sq0 = 0.0, sq1 = 0.0, sq2 = 0.0, sq3 = 0.0;
    i = 0;
    for (; i < limit; i += 4) {
        float v0 = src[i];
        float v1 = src[i + 1];
        float v2 = src[i + 2];
        float v3 = src[i + 3];
        if (!std::isnan(v0)) { double d0 = (double)v0 - meanVal; sq0 += d0 * d0; }
        if (!std::isnan(v1)) { double d1 = (double)v1 - meanVal; sq1 += d1 * d1; }
        if (!std::isnan(v2)) { double d2 = (double)v2 - meanVal; sq2 += d2 * d2; }
        if (!std::isnan(v3)) { double d3 = (double)v3 - meanVal; sq3 += d3 * d3; }
    }
    for (; i < size; i++) {
        float v = src[i];
        if (!std::isnan(v)) { double d = (double)v - meanVal; sq0 += d * d; }
    }
    return ((sq0 + sq1) + (sq2 + sq3)) / (double)count;
}

#define DEFINE_S_NANSUM(TYPE, DEST_TYPE) \
void s_nansum_##TYPE(const TYPE *src, const int64_t *stridesSrc, \
                     DEST_TYPE *dest, const int64_t *stridesDest, \
                     const int64_t *shape, int rank, int axis) { \
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return; \
    int64_t size_axis = shape[axis]; \
    DECLARE_RANK_BUFFER(int64_t, coord, rank); \
    int64_t outer_size = 1; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) outer_size *= shape[d]; \
    } \
    int64_t stride_axis = stridesSrc[axis]; \
    for (int64_t o = 0; o < outer_size; o++) { \
        int64_t offsetRes = 0; \
        int64_t offsetSrc = 0; \
        for (int d = 0; d < rank; d++) { \
            if (d != axis) { \
                offsetSrc += coord[d] * stridesSrc[d]; \
                if (rank > 1) { \
                    int targetD = (d < axis) ? d : (d - 1); \
                    offsetRes += coord[d] * stridesDest[targetD]; \
                } \
            } \
        } \
        double sum_acc = 0.0; \
        for (int64_t i = 0; i < size_axis; i++) { \
            TYPE val = src[offsetSrc + i * stride_axis]; \
            if (!std::isnan(val)) sum_acc += (double)val; \
        } \
        dest[offsetRes] = (DEST_TYPE)sum_acc; \
        for (int d = rank - 1; d >= 0; d--) { \
            if (d == axis) continue; \
            coord[d]++; \
            if (coord[d] < shape[d]) break; \
            coord[d] = 0; \
        } \
    } \
}

DEFINE_S_NANSUM(double, double)
DEFINE_S_NANSUM(float, float)
#undef DEFINE_S_NANSUM

#define DEFINE_S_NANMEAN(TYPE) \
void s_nanmean_##TYPE(const TYPE *src, const int64_t *stridesSrc, \
                      double *dest, const int64_t *stridesDest, \
                      const int64_t *shape, int rank, int axis) { \
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return; \
    int64_t size_axis = shape[axis]; \
    DECLARE_RANK_BUFFER(int64_t, coord, rank); \
    int64_t outer_size = 1; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) outer_size *= shape[d]; \
    } \
    int64_t stride_axis = stridesSrc[axis]; \
    for (int64_t o = 0; o < outer_size; o++) { \
        int64_t offsetRes = 0; \
        int64_t offsetSrc = 0; \
        for (int d = 0; d < rank; d++) { \
            if (d != axis) { \
                offsetSrc += coord[d] * stridesSrc[d]; \
                if (rank > 1) { \
                    int targetD = (d < axis) ? d : (d - 1); \
                    offsetRes += coord[d] * stridesDest[targetD]; \
                } \
            } \
        } \
        double sum_acc = 0.0; \
        int64_t count = 0; \
        for (int64_t i = 0; i < size_axis; i++) { \
            TYPE val = src[offsetSrc + i * stride_axis]; \
            if (!std::isnan(val)) { sum_acc += (double)val; count++; } \
        } \
        dest[offsetRes] = count == 0 ? NAN : (sum_acc / (double)count); \
        for (int d = rank - 1; d >= 0; d--) { \
            if (d == axis) continue; \
            coord[d]++; \
            if (coord[d] < shape[d]) break; \
            coord[d] = 0; \
        } \
    } \
}

DEFINE_S_NANMEAN(double)
DEFINE_S_NANMEAN(float)
#undef DEFINE_S_NANMEAN

#define DEFINE_S_NANVAR(TYPE) \
void s_nanvar_##TYPE(const TYPE *src, const int64_t *stridesSrc, \
                     double *dest, const int64_t *stridesDest, \
                     const int64_t *shape, int rank, int axis) { \
    if (src == nullptr || dest == nullptr || shape == nullptr || rank <= 0 || axis < 0 || axis >= rank) return; \
    int64_t size_axis = shape[axis]; \
    DECLARE_RANK_BUFFER(int64_t, coord, rank); \
    int64_t outer_size = 1; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) outer_size *= shape[d]; \
    } \
    int64_t stride_axis = stridesSrc[axis]; \
    for (int64_t o = 0; o < outer_size; o++) { \
        int64_t offsetRes = 0; \
        int64_t offsetSrc = 0; \
        for (int d = 0; d < rank; d++) { \
            if (d != axis) { \
                offsetSrc += coord[d] * stridesSrc[d]; \
                if (rank > 1) { \
                    int targetD = (d < axis) ? d : (d - 1); \
                    offsetRes += coord[d] * stridesDest[targetD]; \
                } \
            } \
        } \
        double sum_acc = 0.0; \
        int64_t count = 0; \
        for (int64_t i = 0; i < size_axis; i++) { \
            TYPE val = src[offsetSrc + i * stride_axis]; \
            if (!std::isnan(val)) { sum_acc += (double)val; count++; } \
        } \
        if (count == 0) { \
            dest[offsetRes] = NAN; \
        } else { \
            double mean_val = sum_acc / (double)count; \
            double sq_acc = 0.0; \
            for (int64_t i = 0; i < size_axis; i++) { \
                TYPE val = src[offsetSrc + i * stride_axis]; \
                if (!std::isnan(val)) { \
                    double diff = (double)val - mean_val; \
                    sq_acc += diff * diff; \
                } \
            } \
            dest[offsetRes] = sq_acc / (double)count; \
        } \
        for (int d = rank - 1; d >= 0; d--) { \
            if (d == axis) continue; \
            coord[d]++; \
            if (coord[d] < shape[d]) break; \
            coord[d] = 0; \
        } \
    } \
}

DEFINE_S_NANVAR(double)
DEFINE_S_NANVAR(float)
#undef DEFINE_S_NANVAR

} // extern "C"

template <typename T>
static void copy_strided_impl(
    const T *src,
    const int64_t *stridesSrc,
    T *dest,
    const int64_t *stridesDest,
    const int64_t *shape,
    int rank
) {
    if (rank == 0) {
        dest[0] = src[0];
        return;
    }
    int64_t total = 1;
    for (int d = 0; d < rank; d++) {
        if (shape[d] <= 0) return;
        total *= shape[d];
    }
    NoThrowBuffer<T> tmp_dest;
    T *orig_dest = dest;
    const int64_t *orig_strides_dest = stridesDest;
    DECLARE_RANK_BUFFER(int64_t, tmpStridesDest, rank);
    if (strided_buffers_overlap_not_identical(src, stridesSrc, sizeof(T), dest, stridesDest, sizeof(T), shape, rank)) {
        tmp_dest.allocate(total);
        if (!tmp_dest.data()) return;
        dest = tmp_dest.data();
        int64_t st = 1;
        for (int d = rank - 1; d >= 0; d--) {
            tmpStridesDest[d] = st;
            st *= shape[d];
        }
        stridesDest = tmpStridesDest;
    }
    if (rank == 1) {
        int64_t sSrc = stridesSrc[0];
        int64_t sDest = stridesDest[0];
        int64_t n = shape[0];
        for (int64_t i = 0; i < n; i++) {
            dest[i * sDest] = src[i * sSrc];
        }
    } else if (rank == 2) {
        int64_t n0 = shape[0], n1 = shape[1];
        int64_t sSrc0 = stridesSrc[0], sSrc1 = stridesSrc[1];
        int64_t sDest0 = stridesDest[0], sDest1 = stridesDest[1];
        for (int64_t i = 0; i < n0; i++) {
            const T *rowSrc = src + i * sSrc0;
            T *rowDest = dest + i * sDest0;
            for (int64_t j = 0; j < n1; j++) {
                rowDest[j * sDest1] = rowSrc[j * sSrc1];
            }
        }
    } else {
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetSrc = 0;
        int64_t offsetDest = 0;
        for (int64_t idx = 0; idx < total; idx++) {
            dest[offsetDest] = src[offsetSrc];
            for (int d = rank - 1; d >= 0; d--) {
                coord[d]++;
                offsetSrc += stridesSrc[d];
                offsetDest += stridesDest[d];
                if (coord[d] < shape[d]) break;
                offsetSrc -= shape[d] * stridesSrc[d];
                offsetDest -= shape[d] * stridesDest[d];
                coord[d] = 0;
            }
        }
    }
    if (tmp_dest.data() != nullptr) {
        DECLARE_RANK_BUFFER(int64_t, coord, rank);
        int64_t offsetDest = 0;
        for (int64_t idx = 0; idx < total; idx++) {
            orig_dest[offsetDest] = tmp_dest.data()[idx];
            for (int d = rank - 1; d >= 0; d--) {
                coord[d]++;
                offsetDest += orig_strides_dest[d];
                if (coord[d] < shape[d]) break;
                offsetDest -= shape[d] * orig_strides_dest[d];
                coord[d] = 0;
            }
        }
    }
}

extern "C" {

void native_copy_strided(
    const void *src,
    const int64_t *stridesSrc,
    void *dest,
    const int64_t *stridesDest,
    const int64_t *shape,
    int rank,
    int itemSize
) {
    if (src == nullptr || dest == nullptr || rank < 0) return;
    switch (itemSize) {
        case 1:
            copy_strided_impl<uint8_t>((const uint8_t*)src, stridesSrc, (uint8_t*)dest, stridesDest, shape, rank);
            break;
        case 2:
            copy_strided_impl<uint16_t>((const uint16_t*)src, stridesSrc, (uint16_t*)dest, stridesDest, shape, rank);
            break;
        case 4:
            copy_strided_impl<uint32_t>((const uint32_t*)src, stridesSrc, (uint32_t*)dest, stridesDest, shape, rank);
            break;
        case 8:
            copy_strided_impl<uint64_t>((const uint64_t*)src, stridesSrc, (uint64_t*)dest, stridesDest, shape, rank);
            break;
        case 16:
            copy_strided_impl<Item16>((const Item16*)src, stridesSrc, (Item16*)dest, stridesDest, shape, rank);
            break;
        default:
            return;
    }
}

}

// ============================================================================
// SECTION: VECTORIZED & STRIDED BINARY MIN/MAX/FMIN/FMAX (P1-2)
// ============================================================================

template <typename T>
static inline void vectorized_minmax_impl(
    int op_code,
    const T* RESTRICT a,
    const T* RESTRICT b,
    T* RESTRICT out,
    int64_t n
) {
    if constexpr (!std::is_floating_point_v<T>) {
        if (op_code == 2 || op_code == 4) {
            for (int64_t i = 0; i < n; i++) {
                out[i] = a[i] < b[i] ? a[i] : b[i];
            }
        } else {
            for (int64_t i = 0; i < n; i++) {
                out[i] = a[i] > b[i] ? a[i] : b[i];
            }
        }
    } else {
        for (int64_t i = 0; i < n; i++) {
            out[i] = apply_at_op(a[i], b[i], op_code);
        }
    }
}

template <typename T>
static void strided_binary_minmax_impl(
    int op_code,
    int ndim,
    const int64_t* shape,
    const void* a_data,
    const int64_t* a_strides,
    const void* b_data,
    const int64_t* b_strides,
    void* out_data,
    const int64_t* out_strides
) {
    const T* a = static_cast<const T*>(a_data);
    const T* b = static_cast<const T*>(b_data);
    T* out = static_cast<T*>(out_data);

    if (ndim <= 0) {
        out[0] = apply_at_op(a[0], b[0], op_code);
        return;
    }

    int64_t total_elements = 1;
    for (int d = 0; d < ndim; d++) {
        if (shape[d] <= 0) return;
        total_elements *= shape[d];
    }

    NoThrowBuffer<T> tmp_out;
    T* orig_out = out;
    const int64_t* orig_out_strides = out_strides;
    DECLARE_RANK_BUFFER(int64_t, tmpOutStrides, ndim);
    if (strided_buffers_overlap_not_identical(a, a_strides, sizeof(T), out, out_strides, sizeof(T), shape, ndim) ||
        strided_buffers_overlap_not_identical(b, b_strides, sizeof(T), out, out_strides, sizeof(T), shape, ndim)) {
        tmp_out.allocate(total_elements);
        if (!tmp_out.data()) return;
        out = tmp_out.data();
        int64_t st = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            tmpOutStrides[d] = st;
            st *= shape[d];
        }
        out_strides = tmpOutStrides;
    }

    if (ndim == 1) {
        int64_t sa = a_strides[0];
        int64_t sb = b_strides[0];
        int64_t so = out_strides[0];
        int64_t len = shape[0];
        for (int64_t i = 0; i < len; i++) {
            out[i * so] = apply_at_op(a[i * sa], b[i * sb], op_code);
        }
    } else {
        DECLARE_RANK_BUFFER(int64_t, coord, ndim);
        int64_t offA = 0, offB = 0, offOut = 0;
        for (int64_t el = 0; el < total_elements; el++) {
            out[offOut] = apply_at_op(a[offA], b[offB], op_code);
            for (int d = ndim - 1; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offA += a_strides[d];
                    offB += b_strides[d];
                    offOut += out_strides[d];
                    break;
                }
                coord[d] = 0;
                offA -= (shape[d] - 1) * a_strides[d];
                offB -= (shape[d] - 1) * b_strides[d];
                offOut -= (shape[d] - 1) * out_strides[d];
            }
        }
    }

    if (tmp_out.data() != nullptr) {
        DECLARE_RANK_BUFFER(int64_t, coord, ndim);
        int64_t offOut = 0;
        for (int64_t el = 0; el < total_elements; el++) {
            orig_out[offOut] = tmp_out.data()[el];
            for (int d = ndim - 1; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offOut += orig_out_strides[d];
                    break;
                }
                coord[d] = 0;
                offOut -= (shape[d] - 1) * orig_out_strides[d];
            }
        }
    }
}

extern "C" {

VECTORIZED_TARGETS void v_binary_minmax(int op_code, int dtype, const void* a, const void* b, void* out, int64_t n) {
    if (a == nullptr || b == nullptr || out == nullptr || n <= 0) return;
    size_t elem_size = 0;
    switch (dtype) {
        case 0: case 4: case 8: elem_size = 8; break;
        case 1: case 5: case 9: elem_size = 4; break;
        case 2: case 3: case 6: case 10: elem_size = 2; break;
        case 7: case 11: case 14: elem_size = 1; break;
        case 12: elem_size = 16; break;
        case 13: elem_size = 8; break;
        default: return;
    }
    size_t total_bytes = static_cast<size_t>(n) * elem_size;
    NoThrowBuffer<uint8_t> tmp_buf;
    void* target_out = out;
    if (elem_size > 0 &&
        (contiguous_buffers_overlap_not_identical(a, total_bytes, out, total_bytes) ||
         contiguous_buffers_overlap_not_identical(b, total_bytes, out, total_bytes))) {
        if (!tmp_buf.resize(total_bytes)) {
            ndarray_set_oom_flag();
            return;
        }
        target_out = tmp_buf.data();
    }
    switch (dtype) {
        case 0: // float64
            vectorized_minmax_impl<double>(op_code, static_cast<const double*>(a), static_cast<const double*>(b), static_cast<double*>(target_out), n);
            break;
        case 1: // float32
            vectorized_minmax_impl<float>(op_code, static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(target_out), n);
            break;
        case 2: { // float16
            const float16_t* fa = static_cast<const float16_t*>(a);
            const float16_t* fb = static_cast<const float16_t*>(b);
            float16_t* fo = static_cast<float16_t*>(target_out);
            for (int64_t i = 0; i < n; i++) fo[i] = apply_at_op(fa[i], fb[i], op_code);
            break;
        }
        case 3: { // bfloat16
            const bfloat16_t* ba = static_cast<const bfloat16_t*>(a);
            const bfloat16_t* bb = static_cast<const bfloat16_t*>(b);
            bfloat16_t* bo = static_cast<bfloat16_t*>(target_out);
            for (int64_t i = 0; i < n; i++) bo[i] = apply_at_op(ba[i], bb[i], op_code);
            break;
        }
        case 4: // int64
            vectorized_minmax_impl<int64_t>(op_code, static_cast<const int64_t*>(a), static_cast<const int64_t*>(b), static_cast<int64_t*>(target_out), n);
            break;
        case 5: // int32
            vectorized_minmax_impl<int32_t>(op_code, static_cast<const int32_t*>(a), static_cast<const int32_t*>(b), static_cast<int32_t*>(target_out), n);
            break;
        case 6: // int16
            vectorized_minmax_impl<int16_t>(op_code, static_cast<const int16_t*>(a), static_cast<const int16_t*>(b), static_cast<int16_t*>(target_out), n);
            break;
        case 7: // int8
            vectorized_minmax_impl<int8_t>(op_code, static_cast<const int8_t*>(a), static_cast<const int8_t*>(b), static_cast<int8_t*>(target_out), n);
            break;
        case 8: // uint64
            vectorized_minmax_impl<uint64_t>(op_code, static_cast<const uint64_t*>(a), static_cast<const uint64_t*>(b), static_cast<uint64_t*>(target_out), n);
            break;
        case 9: // uint32
            vectorized_minmax_impl<uint32_t>(op_code, static_cast<const uint32_t*>(a), static_cast<const uint32_t*>(b), static_cast<uint32_t*>(target_out), n);
            break;
        case 10: // uint16
            vectorized_minmax_impl<uint16_t>(op_code, static_cast<const uint16_t*>(a), static_cast<const uint16_t*>(b), static_cast<uint16_t*>(target_out), n);
            break;
        case 11: // uint8
            vectorized_minmax_impl<uint8_t>(op_code, static_cast<const uint8_t*>(a), static_cast<const uint8_t*>(b), static_cast<uint8_t*>(target_out), n);
            break;
        case 12: { // complex128
            const cpx_t* ca = static_cast<const cpx_t*>(a);
            const cpx_t* cb = static_cast<const cpx_t*>(b);
            cpx_t* co = static_cast<cpx_t*>(target_out);
            for (int64_t i = 0; i < n; i++) co[i] = apply_at_op(ca[i], cb[i], op_code);
            break;
        }
        case 13: { // complex64
            const cpx_f_t* ca = static_cast<const cpx_f_t*>(a);
            const cpx_f_t* cb = static_cast<const cpx_f_t*>(b);
            cpx_f_t* co = static_cast<cpx_f_t*>(target_out);
            for (int64_t i = 0; i < n; i++) co[i] = apply_at_op(ca[i], cb[i], op_code);
            break;
        }
        case 14: { // boolean
            const uint8_t* ba = static_cast<const uint8_t*>(a);
            const uint8_t* bb = static_cast<const uint8_t*>(b);
            uint8_t* bo = static_cast<uint8_t*>(target_out);
            for (int64_t i = 0; i < n; i++) bo[i] = apply_at_op_boolean(ba[i], bb[i], op_code);
            break;
        }
        default:
            return;
    }
    if (target_out != out) {
        std::memcpy(out, target_out, total_bytes);
    }
}

void s_binary_minmax(
    int op_code,
    int dtype,
    int ndim,
    const int64_t* shape,
    const void* a_data,
    const int64_t* a_strides,
    const void* b_data,
    const int64_t* b_strides,
    void* out_data,
    const int64_t* out_strides
) {
    if (a_data == nullptr || b_data == nullptr || out_data == nullptr || (ndim > 0 && (shape == nullptr || a_strides == nullptr || b_strides == nullptr || out_strides == nullptr))) return;
    switch (dtype) {
        case 0:
            strided_binary_minmax_impl<double>(op_code, ndim, shape, a_data, a_strides, b_data, b_strides, out_data, out_strides);
            break;
        case 1:
            strided_binary_minmax_impl<float>(op_code, ndim, shape, a_data, a_strides, b_data, b_strides, out_data, out_strides);
            break;
        case 2:
            strided_binary_minmax_impl<float16_t>(op_code, ndim, shape, a_data, a_strides, b_data, b_strides, out_data, out_strides);
            break;
        case 3:
            strided_binary_minmax_impl<bfloat16_t>(op_code, ndim, shape, a_data, a_strides, b_data, b_strides, out_data, out_strides);
            break;
        case 4:
            strided_binary_minmax_impl<int64_t>(op_code, ndim, shape, a_data, a_strides, b_data, b_strides, out_data, out_strides);
            break;
        case 5:
            strided_binary_minmax_impl<int32_t>(op_code, ndim, shape, a_data, a_strides, b_data, b_strides, out_data, out_strides);
            break;
        case 6:
            strided_binary_minmax_impl<int16_t>(op_code, ndim, shape, a_data, a_strides, b_data, b_strides, out_data, out_strides);
            break;
        case 7:
            strided_binary_minmax_impl<int8_t>(op_code, ndim, shape, a_data, a_strides, b_data, b_strides, out_data, out_strides);
            break;
        case 8:
            strided_binary_minmax_impl<uint64_t>(op_code, ndim, shape, a_data, a_strides, b_data, b_strides, out_data, out_strides);
            break;
        case 9:
            strided_binary_minmax_impl<uint32_t>(op_code, ndim, shape, a_data, a_strides, b_data, b_strides, out_data, out_strides);
            break;
        case 10:
            strided_binary_minmax_impl<uint16_t>(op_code, ndim, shape, a_data, a_strides, b_data, b_strides, out_data, out_strides);
            break;
        case 11:
            strided_binary_minmax_impl<uint8_t>(op_code, ndim, shape, a_data, a_strides, b_data, b_strides, out_data, out_strides);
            break;
        case 12:
            strided_binary_minmax_impl<cpx_t>(op_code, ndim, shape, a_data, a_strides, b_data, b_strides, out_data, out_strides);
            break;
        case 13:
            strided_binary_minmax_impl<cpx_f_t>(op_code, ndim, shape, a_data, a_strides, b_data, b_strides, out_data, out_strides);
            break;
        case 14: {
            const uint8_t* a = static_cast<const uint8_t*>(a_data);
            const uint8_t* b = static_cast<const uint8_t*>(b_data);
            uint8_t* out = static_cast<uint8_t*>(out_data);
            if (ndim <= 0) {
                out[0] = apply_at_op_boolean(a[0], b[0], op_code);
                return;
            }
            int64_t total_elements = 1;
            for (int d = 0; d < ndim; d++) {
                if (shape[d] <= 0) return;
                total_elements *= shape[d];
            }
            NoThrowBuffer<uint8_t> tmp_out;
            uint8_t* orig_out = out;
            const int64_t* orig_out_strides = out_strides;
            DECLARE_RANK_BUFFER(int64_t, tmpOutStrides, ndim);
            if (strided_buffers_overlap_not_identical(a, a_strides, sizeof(uint8_t), out, out_strides, sizeof(uint8_t), shape, ndim) ||
                strided_buffers_overlap_not_identical(b, b_strides, sizeof(uint8_t), out, out_strides, sizeof(uint8_t), shape, ndim)) {
                tmp_out.allocate(total_elements);
                if (!tmp_out.data()) return;
                out = tmp_out.data();
                int64_t st = 1;
                for (int d = ndim - 1; d >= 0; d--) {
                    tmpOutStrides[d] = st;
                    st *= shape[d];
                }
                out_strides = tmpOutStrides;
            }
            if (ndim == 1) {
                int64_t sa = a_strides[0];
                int64_t sb = b_strides[0];
                int64_t so = out_strides[0];
                int64_t len = shape[0];
                for (int64_t i = 0; i < len; i++) {
                    out[i * so] = apply_at_op_boolean(a[i * sa], b[i * sb], op_code);
                }
            } else {
                DECLARE_RANK_BUFFER(int64_t, coord, ndim);
                int64_t offA = 0, offB = 0, offOut = 0;
                for (int64_t el = 0; el < total_elements; el++) {
                    out[offOut] = apply_at_op_boolean(a[offA], b[offB], op_code);
                    for (int d = ndim - 1; d >= 0; d--) {
                        coord[d]++;
                        if (coord[d] < shape[d]) {
                            offA += a_strides[d];
                            offB += b_strides[d];
                            offOut += out_strides[d];
                            break;
                        }
                        coord[d] = 0;
                        offA -= (shape[d] - 1) * a_strides[d];
                        offB -= (shape[d] - 1) * b_strides[d];
                        offOut -= (shape[d] - 1) * out_strides[d];
                    }
                }
            }
            if (tmp_out.data() != nullptr) {
                DECLARE_RANK_BUFFER(int64_t, coord, ndim);
                int64_t offOut = 0;
                for (int64_t el = 0; el < total_elements; el++) {
                    orig_out[offOut] = tmp_out.data()[el];
                    for (int d = ndim - 1; d >= 0; d--) {
                        coord[d]++;
                        if (coord[d] < shape[d]) {
                            offOut += orig_out_strides[d];
                            break;
                        }
                        coord[d] = 0;
                        offOut -= (shape[d] - 1) * orig_out_strides[d];
                    }
                }
            }
            break;
        }
        default:
            return;
    }
}

}


