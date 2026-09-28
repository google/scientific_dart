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
 * ndarray CUSTOM INDEXING KERNELS (custom_indexing.cpp)
 * ============================================================================
 * High-performance C++ implementation of take_along_axis and put_along_axis
 * for arbitrary rank arrays, strided and contiguous layouts, with fast paths
 * for 1D/2D arrays.
 * ============================================================================
 */

#include "custom_indexing.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <type_traits>

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
            return false;
        }
        ptr_ = static_cast<T *>(std::calloc(n, sizeof(T)));
        if (!ptr_) {
            ok_ = false;
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

#if defined(_MSC_VER)
#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT restrict
#endif

typedef struct {
    double real;
    double imag;
} complex128_t;

typedef struct {
    float real;
    float imag;
} complex64_t;

// ============================================================================
// 1. TAKE_ALONG_AXIS KERNEL TEMPLATE
// ============================================================================

template <typename T, typename IndexT>
static int take_along_axis_impl(
    const T *RESTRICT src,
    const int64_t *RESTRICT arr_shape,
    const int64_t *RESTRICT arr_strides,
    const IndexT *RESTRICT idx_ptr,
    const int64_t *RESTRICT idx_shape,
    const int64_t *RESTRICT idx_strides,
    T *RESTRICT dest,
    const int64_t *RESTRICT out_shape,
    const int64_t *RESTRICT out_strides,
    int64_t rank,
    int64_t axis,
    int64_t *RESTRICT out_error_idx
) {
    if (rank <= 0) return 0;
    const int64_t axis_size = arr_shape[axis];
    if (axis_size <= 0) return 0;

    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) {
        total_elements *= out_shape[i];
    }
    if (total_elements == 0) return 0;

    // --- Fast Path: 1D ---
    if (rank == 1) {
        const int64_t arr_s = arr_strides[0];
        const int64_t idx_s = idx_strides[0];
        const int64_t out_s = out_strides[0];

        if (arr_s == 1 && idx_s == 1 && out_s == 1) {
            for (int64_t i = 0; i < total_elements; i++) {
                IndexT raw_idx = idx_ptr[i];
                int64_t idx;
                if (std::is_signed<IndexT>::value) {
                    idx = (int64_t)raw_idx;
                    if (idx < 0) idx += axis_size;
                    if (idx < 0 || idx >= axis_size) {
                        *out_error_idx = (int64_t)raw_idx;
                        return -1;
                    }
                } else {
                    if ((uint64_t)raw_idx >= (uint64_t)axis_size) {
                        *out_error_idx = (int64_t)raw_idx;
                        return -1;
                    }
                    idx = (int64_t)raw_idx;
                }
                dest[i] = src[idx];
            }
            return 0;
        } else {
            for (int64_t i = 0; i < total_elements; i++) {
                IndexT raw_idx = idx_ptr[i * idx_s];
                int64_t idx;
                if (std::is_signed<IndexT>::value) {
                    idx = (int64_t)raw_idx;
                    if (idx < 0) idx += axis_size;
                    if (idx < 0 || idx >= axis_size) {
                        *out_error_idx = (int64_t)raw_idx;
                        return -1;
                    }
                } else {
                    if ((uint64_t)raw_idx >= (uint64_t)axis_size) {
                        *out_error_idx = (int64_t)raw_idx;
                        return -1;
                    }
                    idx = (int64_t)raw_idx;
                }
                dest[i * out_s] = src[idx * arr_s];
            }
            return 0;
        }
    }

    // --- Fast Path: 2D Contiguous (Row-Major) ---
    if (rank == 2) {
        const int64_t M_out = out_shape[0];
        const int64_t N_out = out_shape[1];

        bool out_c = (out_strides[0] == N_out && out_strides[1] == 1);
        bool arr_c = (arr_strides[0] == arr_shape[1] && arr_strides[1] == 1);

        // Case 2A: axis == 0 (gather across rows)
        if (axis == 0 && out_c && arr_c) {
            if (arr_shape[1] == N_out && (idx_shape[1] == N_out || idx_shape[1] == 1)) {
                const int64_t arr_row_stride = arr_strides[0];
                const int64_t idx_s0 = idx_strides[0];
                const int64_t idx_s1 = (idx_shape[1] == 1 ? 0 : idx_strides[1]);

                for (int64_t i = 0; i < M_out; i++) {
                    const IndexT *row_idx = idx_ptr + i * idx_s0;
                    T *row_dest = dest + i * N_out;
                    for (int64_t j = 0; j < N_out; j++) {
                        IndexT raw_idx = row_idx[j * idx_s1];
                        int64_t idx;
                        if (std::is_signed<IndexT>::value) {
                            idx = (int64_t)raw_idx;
                            if (idx < 0) idx += axis_size;
                            if (idx < 0 || idx >= axis_size) {
                                *out_error_idx = (int64_t)raw_idx;
                                return -1;
                            }
                        } else {
                            if ((uint64_t)raw_idx >= (uint64_t)axis_size) {
                                *out_error_idx = (int64_t)raw_idx;
                                return -1;
                            }
                            idx = (int64_t)raw_idx;
                        }
                        row_dest[j] = src[idx * arr_row_stride + j];
                    }
                }
                return 0;
            }
        }

        // Case 2B: axis == 1 (gather across cols)
        if (axis == 1 && out_c && arr_c) {
            if (arr_shape[0] == M_out && (idx_shape[0] == M_out || idx_shape[0] == 1)) {
                const int64_t arr_row_stride = arr_strides[0];
                const int64_t idx_s0 = (idx_shape[0] == 1 ? 0 : idx_strides[0]);
                const int64_t idx_s1 = idx_strides[1];

                for (int64_t i = 0; i < M_out; i++) {
                    const T *row_src = src + i * arr_row_stride;
                    const IndexT *row_idx = idx_ptr + i * idx_s0;
                    T *row_dest = dest + i * N_out;
                    for (int64_t j = 0; j < N_out; j++) {
                        IndexT raw_idx = row_idx[j * idx_s1];
                        int64_t idx;
                        if (std::is_signed<IndexT>::value) {
                            idx = (int64_t)raw_idx;
                            if (idx < 0) idx += axis_size;
                            if (idx < 0 || idx >= axis_size) {
                                *out_error_idx = (int64_t)raw_idx;
                                return -1;
                            }
                        } else {
                            if ((uint64_t)raw_idx >= (uint64_t)axis_size) {
                                *out_error_idx = (int64_t)raw_idx;
                                return -1;
                            }
                            idx = (int64_t)raw_idx;
                        }
                        row_dest[j] = row_src[idx];
                    }
                }
                return 0;
            }
        }
    }

    // --- General Multi-Dimensional Strided Path ---
    int64_t eff_idx_strides[32];
    int64_t eff_arr_base_strides[32];
    NoThrowBuffer<int64_t> eff_idx_vec, eff_arr_vec;
    int64_t *p_eff_idx = eff_idx_strides;
    int64_t *p_eff_arr = eff_arr_base_strides;
    if (rank > 32) {
        if (!eff_idx_vec.resize(rank) || !eff_arr_vec.resize(rank)) {
            return -4;
        }
        p_eff_idx = eff_idx_vec.data();
        p_eff_arr = eff_arr_vec.data();
    }

    for (int64_t d = 0; d < rank; d++) {
        p_eff_idx[d] = (idx_shape[d] == 1 ? 0 : idx_strides[d]);
        if (d == axis) {
            p_eff_arr[d] = 0;
        } else {
            p_eff_arr[d] = (arr_shape[d] == 1 ? 0 : arr_strides[d]);
        }
    }

    const int64_t axis_stride = arr_strides[axis];

    int64_t coord_stack[32] = {0};
    NoThrowBuffer<int64_t> coord_vec;
    int64_t *coord = coord_stack;
    if (rank > 32) {
        if (!coord_vec.assign(rank, 0)) {
            return -4;
        }
        coord = coord_vec.data();
    }

    int64_t offsetOut = 0;
    int64_t offsetIdx = 0;
    int64_t baseOffsetArr = 0;

    for (int64_t el = 0; el < total_elements; el++) {
        IndexT raw_idx = idx_ptr[offsetIdx];
        int64_t idx;
        if (std::is_signed<IndexT>::value) {
            idx = (int64_t)raw_idx;
            if (idx < 0) idx += axis_size;
            if (idx < 0 || idx >= axis_size) {
                *out_error_idx = (int64_t)raw_idx;
                return -1;
            }
        } else {
            if ((uint64_t)raw_idx >= (uint64_t)axis_size) {
                *out_error_idx = (int64_t)raw_idx;
                return -1;
            }
            idx = (int64_t)raw_idx;
        }
        int64_t offsetArr = baseOffsetArr + idx * axis_stride;
        dest[offsetOut] = src[offsetArr];

        for (int64_t d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < out_shape[d]) {
                offsetOut += out_strides[d];
                offsetIdx += p_eff_idx[d];
                baseOffsetArr += p_eff_arr[d];
                break;
            }
            coord[d] = 0;
            offsetOut -= (out_shape[d] - 1) * out_strides[d];
            offsetIdx -= (out_shape[d] - 1) * p_eff_idx[d];
            baseOffsetArr -= (out_shape[d] - 1) * p_eff_arr[d];
        }
    }

    return 0;
}

template <typename IndexT>
static int dispatch_take_along_axis_by_dtype(
    int dtype,
    const void *src,
    const int64_t *arr_shape,
    const int64_t *arr_strides,
    const IndexT *idx_ptr,
    const int64_t *idx_shape,
    const int64_t *idx_strides,
    void *dest,
    const int64_t *out_shape,
    const int64_t *out_strides,
    int64_t rank,
    int64_t axis,
    int64_t *out_error_idx
) {
    switch (dtype) {
        case DTYPE_FLOAT64:
            return take_along_axis_impl<double, IndexT>(
                (const double *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (double *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        case DTYPE_FLOAT32:
            return take_along_axis_impl<float, IndexT>(
                (const float *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (float *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        case DTYPE_INT64:
            return take_along_axis_impl<int64_t, IndexT>(
                (const int64_t *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (int64_t *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        case DTYPE_INT32:
            return take_along_axis_impl<int32_t, IndexT>(
                (const int32_t *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (int32_t *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        case DTYPE_INT16:
        case DTYPE_UINT16:
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
            return take_along_axis_impl<uint16_t, IndexT>(
                (const uint16_t *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (uint16_t *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        case DTYPE_INT8:
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            return take_along_axis_impl<uint8_t, IndexT>(
                (const uint8_t *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (uint8_t *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        case DTYPE_UINT64:
            return take_along_axis_impl<uint64_t, IndexT>(
                (const uint64_t *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (uint64_t *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        case DTYPE_UINT32:
            return take_along_axis_impl<uint32_t, IndexT>(
                (const uint32_t *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (uint32_t *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        case DTYPE_COMPLEX128:
            return take_along_axis_impl<complex128_t, IndexT>(
                (const complex128_t *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (complex128_t *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        case DTYPE_COMPLEX64:
            return take_along_axis_impl<complex64_t, IndexT>(
                (const complex64_t *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (complex64_t *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        default:
            return -2;
    }
}

extern "C" int native_take_along_axis(
    int dtype,
    int index_dtype,
    const void *src,
    const int64_t *arr_shape,
    const int64_t *arr_strides,
    const void *indices,
    const int64_t *idx_shape,
    const int64_t *idx_strides,
    void *dest,
    const int64_t *out_shape,
    const int64_t *out_strides,
    int64_t rank,
    int64_t axis,
    int64_t *out_error_idx
) {
    if (src == nullptr || indices == nullptr || dest == nullptr ||
        arr_shape == nullptr || arr_strides == nullptr ||
        idx_shape == nullptr || idx_strides == nullptr ||
        out_shape == nullptr || out_strides == nullptr ||
        out_error_idx == nullptr) {
        return -3;
    }

    switch (index_dtype) {
        case DTYPE_INT64:
            return dispatch_take_along_axis_by_dtype<int64_t>(
                dtype, src, arr_shape, arr_strides,
                (const int64_t *)indices, idx_shape, idx_strides,
                dest, out_shape, out_strides, rank, axis, out_error_idx);
        case DTYPE_INT32:
            return dispatch_take_along_axis_by_dtype<int32_t>(
                dtype, src, arr_shape, arr_strides,
                (const int32_t *)indices, idx_shape, idx_strides,
                dest, out_shape, out_strides, rank, axis, out_error_idx);
        case DTYPE_INT16:
            return dispatch_take_along_axis_by_dtype<int16_t>(
                dtype, src, arr_shape, arr_strides,
                (const int16_t *)indices, idx_shape, idx_strides,
                dest, out_shape, out_strides, rank, axis, out_error_idx);
        case DTYPE_INT8:
            return dispatch_take_along_axis_by_dtype<int8_t>(
                dtype, src, arr_shape, arr_strides,
                (const int8_t *)indices, idx_shape, idx_strides,
                dest, out_shape, out_strides, rank, axis, out_error_idx);
        case DTYPE_UINT64:
            return dispatch_take_along_axis_by_dtype<uint64_t>(
                dtype, src, arr_shape, arr_strides,
                (const uint64_t *)indices, idx_shape, idx_strides,
                dest, out_shape, out_strides, rank, axis, out_error_idx);
        case DTYPE_UINT32:
            return dispatch_take_along_axis_by_dtype<uint32_t>(
                dtype, src, arr_shape, arr_strides,
                (const uint32_t *)indices, idx_shape, idx_strides,
                dest, out_shape, out_strides, rank, axis, out_error_idx);
        case DTYPE_UINT16:
            return dispatch_take_along_axis_by_dtype<uint16_t>(
                dtype, src, arr_shape, arr_strides,
                (const uint16_t *)indices, idx_shape, idx_strides,
                dest, out_shape, out_strides, rank, axis, out_error_idx);
        case DTYPE_UINT8:
            return dispatch_take_along_axis_by_dtype<uint8_t>(
                dtype, src, arr_shape, arr_strides,
                (const uint8_t *)indices, idx_shape, idx_strides,
                dest, out_shape, out_strides, rank, axis, out_error_idx);
        default:
            return -2;
    }
}

// ============================================================================
// 2. PUT_ALONG_AXIS KERNEL TEMPLATE
// ============================================================================

template <typename T, typename IndexT>
static int put_along_axis_impl(
    T *RESTRICT target,
    const int64_t *RESTRICT target_shape,
    const int64_t *RESTRICT target_strides,
    const IndexT *RESTRICT idx_ptr,
    const int64_t *RESTRICT idx_shape,
    const int64_t *RESTRICT idx_strides,
    const T *RESTRICT val_ptr,
    const int64_t *RESTRICT val_shape,
    const int64_t *RESTRICT val_strides,
    int64_t rank,
    int64_t axis,
    int64_t *RESTRICT out_error_idx
) {
    if (rank <= 0) return 0;
    const int64_t axis_size = target_shape[axis];
    if (axis_size <= 0) return 0;

    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) {
        total_elements *= idx_shape[i];
    }
    if (total_elements == 0) return 0;

    // --- Fast Path: 1D ---
    if (rank == 1) {
        const int64_t tgt_s = target_strides[0];
        const int64_t idx_s = idx_strides[0];
        const int64_t val_s = (val_shape[0] == 1 ? 0 : val_strides[0]);

        for (int64_t i = 0; i < total_elements; i++) {
            IndexT raw_idx = idx_ptr[i * idx_s];
            int64_t idx;
            if (std::is_signed<IndexT>::value) {
                idx = (int64_t)raw_idx;
                if (idx < 0) idx += axis_size;
                if (idx < 0 || idx >= axis_size) {
                    *out_error_idx = (int64_t)raw_idx;
                    return -1;
                }
            } else {
                if ((uint64_t)raw_idx >= (uint64_t)axis_size) {
                    *out_error_idx = (int64_t)raw_idx;
                    return -1;
                }
                idx = (int64_t)raw_idx;
            }
            target[idx * tgt_s] = val_ptr[i * val_s];
        }
        return 0;
    }

    // --- Fast Path: 2D Contiguous (Row-Major) ---
    if (rank == 2) {
        const int64_t M_idx = idx_shape[0];
        const int64_t N_idx = idx_shape[1];

        bool tgt_c = (target_strides[0] == target_shape[1] && target_strides[1] == 1);

        if (axis == 0 && tgt_c) {
            if (target_shape[1] == N_idx) {
                const int64_t tgt_row_stride = target_strides[0];
                const int64_t idx_s0 = idx_strides[0];
                const int64_t idx_s1 = idx_strides[1];
                const int64_t val_s0 = (val_shape[0] == 1 ? 0 : val_strides[0]);
                const int64_t val_s1 = (val_shape[1] == 1 ? 0 : val_strides[1]);

                for (int64_t i = 0; i < M_idx; i++) {
                    const IndexT *row_idx = idx_ptr + i * idx_s0;
                    const T *row_val = val_ptr + i * val_s0;
                    for (int64_t j = 0; j < N_idx; j++) {
                        IndexT raw_idx = row_idx[j * idx_s1];
                        int64_t idx;
                        if (std::is_signed<IndexT>::value) {
                            idx = (int64_t)raw_idx;
                            if (idx < 0) idx += axis_size;
                            if (idx < 0 || idx >= axis_size) {
                                *out_error_idx = (int64_t)raw_idx;
                                return -1;
                            }
                        } else {
                            if ((uint64_t)raw_idx >= (uint64_t)axis_size) {
                                *out_error_idx = (int64_t)raw_idx;
                                return -1;
                            }
                            idx = (int64_t)raw_idx;
                        }
                        target[idx * tgt_row_stride + j] = row_val[j * val_s1];
                    }
                }
                return 0;
            }
        }

        if (axis == 1 && tgt_c) {
            if (target_shape[0] == M_idx) {
                const int64_t tgt_row_stride = target_strides[0];
                const int64_t idx_s0 = idx_strides[0];
                const int64_t idx_s1 = idx_strides[1];
                const int64_t val_s0 = (val_shape[0] == 1 ? 0 : val_strides[0]);
                const int64_t val_s1 = (val_shape[1] == 1 ? 0 : val_strides[1]);

                for (int64_t i = 0; i < M_idx; i++) {
                    T *row_tgt = target + i * tgt_row_stride;
                    const IndexT *row_idx = idx_ptr + i * idx_s0;
                    const T *row_val = val_ptr + i * val_s0;
                    for (int64_t j = 0; j < N_idx; j++) {
                        IndexT raw_idx = row_idx[j * idx_s1];
                        int64_t idx;
                        if (std::is_signed<IndexT>::value) {
                            idx = (int64_t)raw_idx;
                            if (idx < 0) idx += axis_size;
                            if (idx < 0 || idx >= axis_size) {
                                *out_error_idx = (int64_t)raw_idx;
                                return -1;
                            }
                        } else {
                            if ((uint64_t)raw_idx >= (uint64_t)axis_size) {
                                *out_error_idx = (int64_t)raw_idx;
                                return -1;
                            }
                            idx = (int64_t)raw_idx;
                        }
                        row_tgt[idx] = row_val[j * val_s1];
                    }
                }
                return 0;
            }
        }
    }

    // --- General Multi-Dimensional Strided Path ---
    int64_t eff_val_strides[32];
    int64_t eff_tgt_base_strides[32];
    NoThrowBuffer<int64_t> eff_val_vec, eff_tgt_vec;
    int64_t *p_eff_val = eff_val_strides;
    int64_t *p_eff_tgt = eff_tgt_base_strides;
    if (rank > 32) {
        if (!eff_val_vec.resize(rank) || !eff_tgt_vec.resize(rank)) {
            return -4;
        }
        p_eff_val = eff_val_vec.data();
        p_eff_tgt = eff_tgt_vec.data();
    }

    for (int64_t d = 0; d < rank; d++) {
        p_eff_val[d] = (val_shape[d] == 1 ? 0 : val_strides[d]);
        if (d == axis) {
            p_eff_tgt[d] = 0;
        } else {
            p_eff_tgt[d] = (target_shape[d] == 1 ? 0 : target_strides[d]);
        }
    }

    const int64_t axis_stride = target_strides[axis];

    int64_t coord_stack[32] = {0};
    NoThrowBuffer<int64_t> coord_vec;
    int64_t *coord = coord_stack;
    if (rank > 32) {
        if (!coord_vec.assign(rank, 0)) {
            return -4;
        }
        coord = coord_vec.data();
    }

    int64_t offsetIdx = 0;
    int64_t offsetVal = 0;
    int64_t baseOffsetTarget = 0;

    for (int64_t el = 0; el < total_elements; el++) {
        IndexT raw_idx = idx_ptr[offsetIdx];
        int64_t idx;
        if (std::is_signed<IndexT>::value) {
            idx = (int64_t)raw_idx;
            if (idx < 0) idx += axis_size;
            if (idx < 0 || idx >= axis_size) {
                *out_error_idx = (int64_t)raw_idx;
                return -1;
            }
        } else {
            if ((uint64_t)raw_idx >= (uint64_t)axis_size) {
                *out_error_idx = (int64_t)raw_idx;
                return -1;
            }
            idx = (int64_t)raw_idx;
        }
        int64_t offsetTarget = baseOffsetTarget + idx * axis_stride;
        target[offsetTarget] = val_ptr[offsetVal];

        for (int64_t d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < idx_shape[d]) {
                offsetIdx += idx_strides[d];
                offsetVal += p_eff_val[d];
                baseOffsetTarget += p_eff_tgt[d];
                break;
            }
            coord[d] = 0;
            offsetIdx -= (idx_shape[d] - 1) * idx_strides[d];
            offsetVal -= (idx_shape[d] - 1) * p_eff_val[d];
            baseOffsetTarget -= (idx_shape[d] - 1) * p_eff_tgt[d];
        }
    }

    return 0;
}

template <typename IndexT>
static int dispatch_put_along_axis_by_dtype(
    int dtype,
    void *target,
    const int64_t *target_shape,
    const int64_t *target_strides,
    const IndexT *idx_ptr,
    const int64_t *idx_shape,
    const int64_t *idx_strides,
    const void *values,
    const int64_t *val_shape,
    const int64_t *val_strides,
    int64_t rank,
    int64_t axis,
    int64_t *out_error_idx
) {
    switch (dtype) {
        case DTYPE_FLOAT64:
            return put_along_axis_impl<double, IndexT>(
                (double *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const double *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        case DTYPE_FLOAT32:
            return put_along_axis_impl<float, IndexT>(
                (float *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const float *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        case DTYPE_INT64:
            return put_along_axis_impl<int64_t, IndexT>(
                (int64_t *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const int64_t *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        case DTYPE_INT32:
            return put_along_axis_impl<int32_t, IndexT>(
                (int32_t *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const int32_t *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        case DTYPE_INT16:
        case DTYPE_UINT16:
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
            return put_along_axis_impl<uint16_t, IndexT>(
                (uint16_t *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const uint16_t *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        case DTYPE_INT8:
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            return put_along_axis_impl<uint8_t, IndexT>(
                (uint8_t *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const uint8_t *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        case DTYPE_UINT64:
            return put_along_axis_impl<uint64_t, IndexT>(
                (uint64_t *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const uint64_t *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        case DTYPE_UINT32:
            return put_along_axis_impl<uint32_t, IndexT>(
                (uint32_t *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const uint32_t *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        case DTYPE_COMPLEX128:
            return put_along_axis_impl<complex128_t, IndexT>(
                (complex128_t *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const complex128_t *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        case DTYPE_COMPLEX64:
            return put_along_axis_impl<complex64_t, IndexT>(
                (complex64_t *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const complex64_t *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        default:
            return -2;
    }
}

extern "C" int native_put_along_axis(
    int dtype,
    int index_dtype,
    void *target,
    const int64_t *target_shape,
    const int64_t *target_strides,
    const void *indices,
    const int64_t *idx_shape,
    const int64_t *idx_strides,
    const void *values,
    const int64_t *val_shape,
    const int64_t *val_strides,
    int64_t rank,
    int64_t axis,
    int64_t *out_error_idx
) {
    if (target == nullptr || indices == nullptr || values == nullptr ||
        target_shape == nullptr || target_strides == nullptr ||
        idx_shape == nullptr || idx_strides == nullptr ||
        val_shape == nullptr || val_strides == nullptr ||
        out_error_idx == nullptr) {
        return -3;
    }

    switch (index_dtype) {
        case DTYPE_INT64:
            return dispatch_put_along_axis_by_dtype<int64_t>(
                dtype, target, target_shape, target_strides,
                (const int64_t *)indices, idx_shape, idx_strides,
                values, val_shape, val_strides, rank, axis, out_error_idx);
        case DTYPE_INT32:
            return dispatch_put_along_axis_by_dtype<int32_t>(
                dtype, target, target_shape, target_strides,
                (const int32_t *)indices, idx_shape, idx_strides,
                values, val_shape, val_strides, rank, axis, out_error_idx);
        case DTYPE_INT16:
            return dispatch_put_along_axis_by_dtype<int16_t>(
                dtype, target, target_shape, target_strides,
                (const int16_t *)indices, idx_shape, idx_strides,
                values, val_shape, val_strides, rank, axis, out_error_idx);
        case DTYPE_INT8:
            return dispatch_put_along_axis_by_dtype<int8_t>(
                dtype, target, target_shape, target_strides,
                (const int8_t *)indices, idx_shape, idx_strides,
                values, val_shape, val_strides, rank, axis, out_error_idx);
        case DTYPE_UINT64:
            return dispatch_put_along_axis_by_dtype<uint64_t>(
                dtype, target, target_shape, target_strides,
                (const uint64_t *)indices, idx_shape, idx_strides,
                values, val_shape, val_strides, rank, axis, out_error_idx);
        case DTYPE_UINT32:
            return dispatch_put_along_axis_by_dtype<uint32_t>(
                dtype, target, target_shape, target_strides,
                (const uint32_t *)indices, idx_shape, idx_strides,
                values, val_shape, val_strides, rank, axis, out_error_idx);
        case DTYPE_UINT16:
            return dispatch_put_along_axis_by_dtype<uint16_t>(
                dtype, target, target_shape, target_strides,
                (const uint16_t *)indices, idx_shape, idx_strides,
                values, val_shape, val_strides, rank, axis, out_error_idx);
        case DTYPE_UINT8:
            return dispatch_put_along_axis_by_dtype<uint8_t>(
                dtype, target, target_shape, target_strides,
                (const uint8_t *)indices, idx_shape, idx_strides,
                values, val_shape, val_strides, rank, axis, out_error_idx);
        default:
            return -2;
    }
}

// ============================================================================
// 3. TILE / BLOCK REPLICATION KERNELS
// ============================================================================

static void fill_tile_contiguous_nd(
    uint8_t *RESTRICT dest,
    const uint8_t *RESTRICT src,
    const int64_t *RESTRICT src_shape,
    const int64_t *RESTRICT reps,
    const size_t *RESTRICT src_block_bytes,
    const size_t *RESTRICT dest_block_bytes,
    int64_t dim,
    int64_t rank,
    size_t itemsize
) {
    if (dim == rank - 1) {
        size_t s_dim = (size_t)src_shape[dim];
        size_t r_dim = (size_t)reps[dim];
        size_t chunk_bytes = s_dim * itemsize;
        size_t total_bytes = r_dim * chunk_bytes;

        memcpy(dest, src, chunk_bytes);
        size_t copied = chunk_bytes;
        while (copied < total_bytes) {
            size_t to_copy = (copied <= total_bytes - copied) ? copied : (total_bytes - copied);
            memcpy(dest + copied, dest, to_copy);
            copied += to_copy;
        }
        return;
    }

    int64_t count = src_shape[dim];
    size_t child_dest_stride = dest_block_bytes[dim + 1];
    size_t child_src_stride = src_block_bytes[dim + 1];

    for (int64_t i = 0; i < count; i++) {
        fill_tile_contiguous_nd(
            dest + i * child_dest_stride,
            src + i * child_src_stride,
            src_shape,
            reps,
            src_block_bytes,
            dest_block_bytes,
            dim + 1,
            rank,
            itemsize
        );
    }

    size_t block_bytes = (size_t)count * child_dest_stride;
    size_t total_dim_bytes = (size_t)reps[dim] * block_bytes;
    size_t copied_block = block_bytes;
    while (copied_block < total_dim_bytes) {
        size_t to_copy = (copied_block <= total_dim_bytes - copied_block) ? copied_block : (total_dim_bytes - copied_block);
        memcpy(dest + copied_block, dest, to_copy);
        copied_block += to_copy;
    }
}

extern "C" int native_tile_contiguous(
    int dtype,
    const void *src,
    const int64_t *src_shape,
    const int64_t *reps,
    void *dest,
    const int64_t *out_shape,
    int64_t rank
) {
    if (src == nullptr || dest == nullptr || src_shape == nullptr || reps == nullptr || out_shape == nullptr) {
        return -3;
    }

    size_t itemsize = 0;
    switch (dtype) {
        case DTYPE_FLOAT64:
        case DTYPE_INT64:
        case DTYPE_UINT64:
        case DTYPE_COMPLEX64:
            itemsize = 8;
            break;
        case DTYPE_FLOAT32:
        case DTYPE_INT32:
        case DTYPE_UINT32:
            itemsize = 4;
            break;
        case DTYPE_INT16:
        case DTYPE_UINT16:
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
            itemsize = 2;
            break;
        case DTYPE_INT8:
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            itemsize = 1;
            break;
        case DTYPE_COMPLEX128:
            itemsize = 16;
            break;
        default:
            return -2;
    }

    if (rank <= 0) {
        memcpy(dest, src, itemsize);
        return 0;
    }

    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) {
        if (src_shape[i] <= 0 || reps[i] <= 0 || out_shape[i] <= 0) return 0;
        total_elements *= out_shape[i];
    }
    if (total_elements == 0) return 0;

    // --- Fast Path: 1D ---
    if (rank == 1) {
        size_t chunk_bytes = (size_t)src_shape[0] * itemsize;
        size_t total_bytes = (size_t)reps[0] * chunk_bytes;
        uint8_t *d = (uint8_t *)dest;
        memcpy(d, src, chunk_bytes);
        size_t copied = chunk_bytes;
        while (copied < total_bytes) {
            size_t to_copy = (copied <= total_bytes - copied) ? copied : (total_bytes - copied);
            memcpy(d + copied, d, to_copy);
            copied += to_copy;
        }
        return 0;
    }

    // --- Fast Path: 2D ---
    if (rank == 2) {
        int64_t M = src_shape[0];
        int64_t N = src_shape[1];
        int64_t R_rows = reps[0];
        int64_t R_cols = reps[1];
        size_t row_src_bytes = (size_t)N * itemsize;
        size_t row_dest_bytes = (size_t)N * R_cols * itemsize;
        const uint8_t *s = (const uint8_t *)src;
        uint8_t *d = (uint8_t *)dest;

        for (int64_t i = 0; i < M; i++) {
            uint8_t *dest_row = d + i * row_dest_bytes;
            const uint8_t *src_row = s + i * row_src_bytes;
            memcpy(dest_row, src_row, row_src_bytes);
            size_t copied = row_src_bytes;
            while (copied < row_dest_bytes) {
                size_t to_copy = (copied <= row_dest_bytes - copied) ? copied : (row_dest_bytes - copied);
                memcpy(dest_row + copied, dest_row, to_copy);
                copied += to_copy;
            }
        }

        size_t block_bytes = (size_t)M * row_dest_bytes;
        size_t total_dest_bytes = (size_t)R_rows * block_bytes;
        size_t copied_block = block_bytes;
        while (copied_block < total_dest_bytes) {
            size_t to_copy = (copied_block <= total_dest_bytes - copied_block) ? copied_block : (total_dest_bytes - copied_block);
            memcpy(d + copied_block, d, to_copy);
            copied_block += to_copy;
        }
        return 0;
    }

    // --- N-D General Path ---
    size_t src_block_bytes[32];
    size_t dest_block_bytes[32];
    NoThrowBuffer<size_t> src_block_vec, dest_block_vec;
    size_t *p_src_block = src_block_bytes;
    size_t *p_dest_block = dest_block_bytes;
    if (rank > 32) {
        if (!src_block_vec.resize(rank) || !dest_block_vec.resize(rank)) {
            return -4;
        }
        p_src_block = src_block_vec.data();
        p_dest_block = dest_block_vec.data();
    }

    p_src_block[rank - 1] = (size_t)src_shape[rank - 1] * itemsize;
    p_dest_block[rank - 1] = (size_t)out_shape[rank - 1] * itemsize;
    for (int64_t d = rank - 2; d >= 0; d--) {
        p_src_block[d] = p_src_block[d + 1] * (size_t)src_shape[d];
        p_dest_block[d] = p_dest_block[d + 1] * (size_t)out_shape[d];
    }

    fill_tile_contiguous_nd(
        (uint8_t *)dest,
        (const uint8_t *)src,
        src_shape,
        reps,
        p_src_block,
        p_dest_block,
        0,
        rank,
        itemsize
    );

    return 0;
}

template <typename T>
static int tile_strided_impl(
    const T *RESTRICT src,
    const int64_t *RESTRICT src_shape,
    const int64_t *RESTRICT src_strides,
    const int64_t *RESTRICT reps,
    T *RESTRICT dest,
    const int64_t *RESTRICT out_shape,
    const int64_t *RESTRICT out_strides,
    int64_t rank
) {
    if (rank <= 0) {
        dest[0] = src[0];
        return 0;
    }
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) {
        if (src_shape[i] <= 0 || reps[i] <= 0 || out_shape[i] <= 0) return 0;
        total_elements *= out_shape[i];
    }
    if (total_elements == 0) return 0;

    int64_t coord_stack[32] = {0};
    NoThrowBuffer<int64_t> coord_vec;
    int64_t *coord = coord_stack;
    if (rank > 32) {
        if (!coord_vec.assign(rank, 0)) {
            return -4;
        }
        coord = coord_vec.data();
    }

    int64_t offsetOut = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        int64_t offsetSrc = 0;
        for (int64_t d = 0; d < rank; d++) {
            int64_t src_coord = coord[d] % src_shape[d];
            offsetSrc += src_coord * src_strides[d];
        }
        dest[offsetOut] = src[offsetSrc];

        for (int64_t d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < out_shape[d]) {
                offsetOut += out_strides[d];
                break;
            }
            coord[d] = 0;
            offsetOut -= (out_shape[d] - 1) * out_strides[d];
        }
    }
    return 0;
}

extern "C" int native_tile_strided(
    int dtype,
    const void *src,
    const int64_t *src_shape,
    const int64_t *src_strides,
    const int64_t *reps,
    void *dest,
    const int64_t *out_shape,
    const int64_t *out_strides,
    int64_t rank
) {
    if (src == nullptr || dest == nullptr || src_shape == nullptr ||
        src_strides == nullptr || reps == nullptr || out_shape == nullptr ||
        out_strides == nullptr) {
        return -3;
    }
    switch (dtype) {
        case DTYPE_FLOAT64:
            return tile_strided_impl<double>(
                (const double *)src, src_shape, src_strides, reps,
                (double *)dest, out_shape, out_strides, rank);
        case DTYPE_FLOAT32:
            return tile_strided_impl<float>(
                (const float *)src, src_shape, src_strides, reps,
                (float *)dest, out_shape, out_strides, rank);
        case DTYPE_INT64:
            return tile_strided_impl<int64_t>(
                (const int64_t *)src, src_shape, src_strides, reps,
                (int64_t *)dest, out_shape, out_strides, rank);
        case DTYPE_INT32:
            return tile_strided_impl<int32_t>(
                (const int32_t *)src, src_shape, src_strides, reps,
                (int32_t *)dest, out_shape, out_strides, rank);
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            return tile_strided_impl<uint8_t>(
                (const uint8_t *)src, src_shape, src_strides, reps,
                (uint8_t *)dest, out_shape, out_strides, rank);
        case DTYPE_INT16:
            return tile_strided_impl<int16_t>(
                (const int16_t *)src, src_shape, src_strides, reps,
                (int16_t *)dest, out_shape, out_strides, rank);
        case DTYPE_UINT64:
            return tile_strided_impl<uint64_t>(
                (const uint64_t *)src, src_shape, src_strides, reps,
                (uint64_t *)dest, out_shape, out_strides, rank);
        case DTYPE_UINT32:
            return tile_strided_impl<uint32_t>(
                (const uint32_t *)src, src_shape, src_strides, reps,
                (uint32_t *)dest, out_shape, out_strides, rank);
        case DTYPE_UINT16:
            return tile_strided_impl<uint16_t>(
                (const uint16_t *)src, src_shape, src_strides, reps,
                (uint16_t *)dest, out_shape, out_strides, rank);
        case DTYPE_INT8:
            return tile_strided_impl<int8_t>(
                (const int8_t *)src, src_shape, src_strides, reps,
                (int8_t *)dest, out_shape, out_strides, rank);
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
            return tile_strided_impl<uint16_t>(
                (const uint16_t *)src, src_shape, src_strides, reps,
                (uint16_t *)dest, out_shape, out_strides, rank);
        case DTYPE_COMPLEX128:
            return tile_strided_impl<complex128_t>(
                (const complex128_t *)src, src_shape, src_strides, reps,
                (complex128_t *)dest, out_shape, out_strides, rank);
        case DTYPE_COMPLEX64:
            return tile_strided_impl<complex64_t>(
                (const complex64_t *)src, src_shape, src_strides, reps,
                (complex64_t *)dest, out_shape, out_strides, rank);
        default:
            return -2;
    }
}

// ============================================================================
// 4. MULTIDIMENSIONAL ARRAY PADDING KERNELS
// ============================================================================

template <typename T>
static inline void fill_typed(T *RESTRICT dest, int64_t count, T val) {
    if (count <= 0) return;
    static const T zero_val = {};
    if (memcmp(&val, &zero_val, sizeof(T)) == 0) {
        memset(dest, 0, (size_t)count * sizeof(T));
        return;
    }
    dest[0] = val;
    int64_t copied = 1;
    while (copied < count) {
        int64_t to_copy = (copied <= count - copied) ? copied : (count - copied);
        memcpy(dest + copied, dest, (size_t)to_copy * sizeof(T));
        copied += to_copy;
    }
}

static inline int64_t pad_reflect_map(int64_t i, int64_t N) {
    if (N <= 1) return 0;
    int64_t P = 2 * N - 2;
    int64_t i_mod = (i < 0 ? -i : i) % P;
    return i_mod < N ? i_mod : P - i_mod;
}

static inline int64_t pad_symmetric_map(int64_t i, int64_t N) {
    if (N <= 0) return 0;
    int64_t P = 2 * N;
    int64_t i_mod;
    if (i < 0) {
        i_mod = (-i - 1) % P;
    } else {
        i_mod = i % P;
    }
    return i_mod < N ? i_mod : P - 1 - i_mod;
}

static inline int64_t pad_wrap_map(int64_t i, int64_t N) {
    if (N <= 0) return 0;
    int64_t rem = i % N;
    return rem < 0 ? rem + N : rem;
}

static inline int64_t pad_map_index(int64_t i, int64_t N, int mode) {
    switch (mode) {
        case 1: /* edge */
            if (i < 0) return 0;
            if (i >= N) return N - 1;
            return i;
        case 2: /* reflect */
            return pad_reflect_map(i, N);
        case 3: /* symmetric */
            return pad_symmetric_map(i, N);
        case 4: /* wrap */
            return pad_wrap_map(i, N);
        default:
            return 0;
    }
}

template <typename T>
static int pad_1d_impl(
    const T *RESTRICT src,
    int64_t src_len,
    int64_t src_stride,
    T *RESTRICT dest,
    int64_t pad_before,
    int64_t pad_after,
    int mode,
    T const_before,
    T const_after,
    int is_uniform_constant
) {
    int64_t dest_len = pad_before + src_len + pad_after;
    if (dest_len <= 0) return 0;

    T *dest_interior = dest + pad_before;

    // Fast path: constant mode
    if (mode == 0) {
        if (is_uniform_constant) {
            fill_typed<T>(dest, dest_len, const_before);
        } else {
            fill_typed<T>(dest, pad_before, const_before);
            fill_typed<T>(dest + pad_before + src_len, pad_after, const_after);
        }
        if (src_len > 0) {
            if (src_stride == 1) {
                memcpy(dest_interior, src, (size_t)src_len * sizeof(T));
            } else {
                for (int64_t i = 0; i < src_len; i++) {
                    dest_interior[i] = src[i * src_stride];
                }
            }
        }
        return 0;
    }

    // Step 1: Copy interior
    if (src_len > 0) {
        if (src_stride == 1) {
            memcpy(dest_interior, src, (size_t)src_len * sizeof(T));
        } else {
            for (int64_t i = 0; i < src_len; i++) {
                dest_interior[i] = src[i * src_stride];
            }
        }
    }

    // Step 2: Fill left border
    if (pad_before > 0 && src_len > 0) {
        if (mode == 1) { // edge
            fill_typed<T>(dest, pad_before, dest_interior[0]);
        } else {
            for (int64_t c = 0; c < pad_before; c++) {
                int64_t src_idx = pad_map_index(c - pad_before, src_len, mode);
                dest[c] = dest_interior[src_idx];
            }
        }
    }

    // Step 3: Fill right border
    if (pad_after > 0 && src_len > 0) {
        T *dest_right = dest_interior + src_len;
        if (mode == 1) { // edge
            fill_typed<T>(dest_right, pad_after, dest_interior[src_len - 1]);
        } else {
            for (int64_t c = 0; c < pad_after; c++) {
                int64_t src_idx = pad_map_index(src_len + c, src_len, mode);
                dest_right[c] = dest_interior[src_idx];
            }
        }
    }

    return 0;
}

template <typename T>
static int pad_2d_impl(
    const T *RESTRICT src,
    int64_t src_rows,
    int64_t src_cols,
    int64_t src_stride_rows,
    int64_t src_stride_cols,
    T *RESTRICT dest,
    int64_t pad_top,
    int64_t pad_bottom,
    int64_t pad_left,
    int64_t pad_right,
    int mode,
    const T *const_before, // [top, left]
    const T *const_after,  // [bottom, right]
    int is_uniform_constant
) {
    int64_t dest_rows = pad_top + src_rows + pad_bottom;
    int64_t dest_cols = pad_left + src_cols + pad_right;
    if (dest_rows <= 0 || dest_cols <= 0) return 0;

    int64_t total_dest_elements = dest_rows * dest_cols;

    // Fast path: constant mode
    if (mode == 0) {
        T cb_left = const_before ? const_before[1] : T{};
        T ca_right = const_after ? const_after[1] : T{};
        T cb_top = const_before ? const_before[0] : T{};
        T ca_bottom = const_after ? const_after[0] : T{};
        size_t row_bytes = (size_t)dest_cols * sizeof(T);

        // Fill top rows
        if (pad_top > 0) {
            fill_typed<T>(dest, dest_cols, cb_top);
            for (int64_t i = 1; i < pad_top; i++) {
                memcpy(dest + i * dest_cols, dest, row_bytes);
            }
        }

        // Fill interior rows with left & right padding
        bool src_contiguous = (src_stride_rows == src_cols && src_stride_cols == 1);
        for (int64_t r = 0; r < src_rows; r++) {
            T *dest_row = dest + (pad_top + r) * dest_cols;
            if (pad_left > 0) {
                fill_typed<T>(dest_row, pad_left, cb_left);
            }
            if (src_cols > 0) {
                T *dest_interior = dest_row + pad_left;
                if (src_contiguous) {
                    const T *src_row = src + r * src_cols;
                    memcpy(dest_interior, src_row, (size_t)src_cols * sizeof(T));
                } else {
                    const T *src_row = src + r * src_stride_rows;
                    for (int64_t c = 0; c < src_cols; c++) {
                        dest_interior[c] = src_row[c * src_stride_cols];
                    }
                }
            }
            if (pad_right > 0) {
                fill_typed<T>(dest_row + pad_left + src_cols, pad_right, ca_right);
            }
        }

        // Fill bottom rows
        if (pad_bottom > 0) {
            T *first_bottom = dest + (pad_top + src_rows) * dest_cols;
            fill_typed<T>(first_bottom, dest_cols, ca_bottom);
            for (int64_t i = pad_top + src_rows + 1; i < dest_rows; i++) {
                memcpy(dest + i * dest_cols, first_bottom, row_bytes);
            }
        }

        // If non-uniform constant, overwrite left and right columns of top and bottom rows
        if (!is_uniform_constant) {
            if (pad_left > 0) {
                for (int64_t i = 0; i < pad_top; i++) {
                    fill_typed<T>(dest + i * dest_cols, pad_left, cb_left);
                }
                for (int64_t i = pad_top + src_rows; i < dest_rows; i++) {
                    fill_typed<T>(dest + i * dest_cols, pad_left, cb_left);
                }
            }
            if (pad_right > 0) {
                for (int64_t i = 0; i < pad_top; i++) {
                    fill_typed<T>(dest + i * dest_cols + pad_left + src_cols, pad_right, ca_right);
                }
                for (int64_t i = pad_top + src_rows; i < dest_rows; i++) {
                    fill_typed<T>(dest + i * dest_cols + pad_left + src_cols, pad_right, ca_right);
                }
            }
        }
        return 0;
    }

    // Precompute horizontal index maps if needed
    int64_t map_left_stack[256];
    int64_t map_right_stack[256];
    NoThrowBuffer<int64_t> map_left_vec, map_right_vec;
    int64_t *p_map_left = map_left_stack;
    int64_t *p_map_right = map_right_stack;
    if (pad_left > 256) {
        if (!map_left_vec.resize(pad_left)) return -4;
        p_map_left = map_left_vec.data();
    }
    if (pad_right > 256) {
        if (!map_right_vec.resize(pad_right)) return -4;
        p_map_right = map_right_vec.data();
    }

    if (src_cols > 0 && (mode == 2 || mode == 3 || mode == 4)) {
        for (int64_t c = 0; c < pad_left; c++) {
            p_map_left[c] = pad_map_index(c - pad_left, src_cols, mode);
        }
        for (int64_t c = 0; c < pad_right; c++) {
            p_map_right[c] = pad_map_index(src_cols + c, src_cols, mode);
        }
    }

    T cb_left = const_before ? const_before[1] : T{};
    T ca_right = const_after ? const_after[1] : T{};
    T cb_top = const_before ? const_before[0] : T{};
    T ca_bottom = const_after ? const_after[0] : T{};

    bool src_row_contiguous = (src_stride_cols == 1);

    // Phase 1: Fill rows [pad_top .. pad_top + src_rows - 1]
    for (int64_t r = 0; r < src_rows; r++) {
        T *row_base = dest + (pad_top + r) * dest_cols;
        T *row_interior = row_base + pad_left;
        const T *src_row = src + r * src_stride_rows;

        // Copy interior
        if (src_cols > 0) {
            if (src_row_contiguous) {
                memcpy(row_interior, src_row, (size_t)src_cols * sizeof(T));
            } else {
                for (int64_t c = 0; c < src_cols; c++) {
                    row_interior[c] = src_row[c * src_stride_cols];
                }
            }
        }

        // Left padding for row
        if (pad_left > 0) {
            if (mode == 0) { // constant
                fill_typed<T>(row_base, pad_left, cb_left);
            } else if (src_cols > 0) {
                if (mode == 1) { // edge
                    fill_typed<T>(row_base, pad_left, row_interior[0]);
                } else { // reflect, symmetric, wrap
                    for (int64_t c = 0; c < pad_left; c++) {
                        row_base[c] = row_interior[p_map_left[c]];
                    }
                }
            }
        }

        // Right padding for row
        if (pad_right > 0) {
            T *row_right = row_interior + src_cols;
            if (mode == 0) { // constant
                fill_typed<T>(row_right, pad_right, ca_right);
            } else if (src_cols > 0) {
                if (mode == 1) { // edge
                    fill_typed<T>(row_right, pad_right, row_interior[src_cols - 1]);
                } else { // reflect, symmetric, wrap
                    for (int64_t c = 0; c < pad_right; c++) {
                        row_right[c] = row_interior[p_map_right[c]];
                    }
                }
            }
        }
    }

    // Phase 2: Fill top rows [0 .. pad_top - 1]
    size_t row_bytes = (size_t)dest_cols * sizeof(T);
    if (pad_top > 0) {
        if (mode == 0) { // constant
            for (int64_t i = 0; i < pad_top; i++) {
                fill_typed<T>(dest + i * dest_cols, dest_cols, cb_top);
            }
        } else if (src_rows > 0) {
            if (mode == 1) { // edge
                const T *src_full_row = dest + pad_top * dest_cols;
                for (int64_t i = 0; i < pad_top; i++) {
                    memcpy(dest + i * dest_cols, src_full_row, row_bytes);
                }
            } else { // reflect, symmetric, wrap
                for (int64_t i = 0; i < pad_top; i++) {
                    int64_t src_r = pad_map_index(i - pad_top, src_rows, mode);
                    const T *src_full_row = dest + (pad_top + src_r) * dest_cols;
                    memcpy(dest + i * dest_cols, src_full_row, row_bytes);
                }
            }
        }
    }

    // Phase 3: Fill bottom rows [pad_top + src_rows .. dest_rows - 1]
    if (pad_bottom > 0) {
        if (mode == 0) { // constant
            for (int64_t i = pad_top + src_rows; i < dest_rows; i++) {
                fill_typed<T>(dest + i * dest_cols, dest_cols, ca_bottom);
            }
        } else if (src_rows > 0) {
            if (mode == 1) { // edge
                const T *src_full_row = dest + (pad_top + src_rows - 1) * dest_cols;
                for (int64_t i = pad_top + src_rows; i < dest_rows; i++) {
                    memcpy(dest + i * dest_cols, src_full_row, row_bytes);
                }
            } else { // reflect, symmetric, wrap
                for (int64_t i = pad_top + src_rows; i < dest_rows; i++) {
                    int64_t src_r = pad_map_index(i - pad_top, src_rows, mode);
                    const T *src_full_row = dest + (pad_top + src_r) * dest_cols;
                    memcpy(dest + i * dest_cols, src_full_row, row_bytes);
                }
            }
        }
    }

    return 0;
}

template <typename T>
static int pad_nd_impl(
    const T *RESTRICT src,
    const int64_t *RESTRICT src_shape,
    const int64_t *RESTRICT src_strides,
    T *RESTRICT dest,
    const int64_t *RESTRICT dest_shape,
    const int64_t *RESTRICT pad_before,
    const int64_t *RESTRICT pad_after,
    int64_t rank,
    int mode,
    const T *const_before,
    const T *const_after,
    int is_uniform_constant
) {
    if (rank <= 0) return 0;
    if (rank == 1) {
        return pad_1d_impl<T>(
            src, src_shape[0], src_strides[0],
            dest, pad_before[0], pad_after[0],
            mode,
            const_before ? const_before[0] : T{},
            const_after ? const_after[0] : T{},
            is_uniform_constant
        );
    }
    if (rank == 2) {
        return pad_2d_impl<T>(
            src, src_shape[0], src_shape[1],
            src_strides[0], src_strides[1],
            dest,
            pad_before[0], pad_after[0],
            pad_before[1], pad_after[1],
            mode,
            const_before, const_after,
            is_uniform_constant
        );
    }

    int64_t total_dest_elements = 1;
    for (int64_t d = 0; d < rank; d++) {
        if (dest_shape[d] <= 0) return 0;
        total_dest_elements *= dest_shape[d];
    }

    // Fast path: uniform constant mode
    if (mode == 0 && is_uniform_constant) {
        T cb_val = const_before ? const_before[0] : T{};
        fill_typed<T>(dest, total_dest_elements, cb_val);
    }

    // Step 0: Copy interior src to dest
    int64_t inner_len = src_shape[rank - 1];
    int64_t inner_src_stride = src_strides[rank - 1];
    int64_t inner_pad_before = pad_before[rank - 1];
    int64_t inner_pad_after = pad_after[rank - 1];

    int64_t dest_strides[32];
    NoThrowBuffer<int64_t> dest_strides_vec;
    int64_t *p_dest_strides = dest_strides;
    if (rank > 32) {
        if (!dest_strides_vec.resize(rank)) return -4;
        p_dest_strides = dest_strides_vec.data();
    }
    p_dest_strides[rank - 1] = 1;
    for (int64_t d = rank - 2; d >= 0; d--) {
        p_dest_strides[d] = p_dest_strides[d + 1] * dest_shape[d + 1];
    }

    int64_t outer_elements = 1;
    for (int64_t d = 0; d < rank - 1; d++) {
        outer_elements *= src_shape[d];
    }

    int64_t coord[32] = {0};
    NoThrowBuffer<int64_t> coord_vec;
    int64_t *p_coord = coord;
    if (rank > 32) {
        if (!coord_vec.assign(rank, 0)) return -4;
        p_coord = coord_vec.data();
    }

    // Precompute inner left and right index maps
    int64_t map_left_stack[256];
    int64_t map_right_stack[256];
    NoThrowBuffer<int64_t> map_left_vec, map_right_vec;
    int64_t *p_map_left = map_left_stack;
    int64_t *p_map_right = map_right_stack;
    if (inner_pad_before > 256) {
        if (!map_left_vec.resize(inner_pad_before)) return -4;
        p_map_left = map_left_vec.data();
    }
    if (inner_pad_after > 256) {
        if (!map_right_vec.resize(inner_pad_after)) return -4;
        p_map_right = map_right_vec.data();
    }

    if (inner_len > 0 && (mode == 2 || mode == 3 || mode == 4)) {
        for (int64_t c = 0; c < inner_pad_before; c++) {
            p_map_left[c] = pad_map_index(c - inner_pad_before, inner_len, mode);
        }
        for (int64_t c = 0; c < inner_pad_after; c++) {
            p_map_right[c] = pad_map_index(inner_len + c, inner_len, mode);
        }
    }

    T cb_inner = const_before ? const_before[rank - 1] : T{};
    T ca_inner = const_after ? const_after[rank - 1] : T{};

    int64_t src_offset = 0;
    int64_t dest_base_offset = 0;
    for (int64_t d = 0; d < rank - 1; d++) {
        dest_base_offset += pad_before[d] * p_dest_strides[d];
    }

    if (outer_elements > 0) {
        // Iterate over all outer slices of src
        for (int64_t el = 0; el < outer_elements; el++) {
            T *dest_row_base = dest + dest_base_offset;
            T *dest_row_interior = dest_row_base + inner_pad_before;
            const T *src_row = src + src_offset;

            // Copy interior
            if (inner_len > 0) {
                if (inner_src_stride == 1) {
                    memcpy(dest_row_interior, src_row, (size_t)inner_len * sizeof(T));
                } else {
                    for (int64_t c = 0; c < inner_len; c++) {
                        dest_row_interior[c] = src_row[c * inner_src_stride];
                    }
                }
            }

            // For non-constant modes, fill inner left and right borders
            if (mode != 0) {
                if (inner_pad_before > 0 && inner_len > 0) {
                    if (mode == 1) {
                        fill_typed<T>(dest_row_base, inner_pad_before, dest_row_interior[0]);
                    } else {
                        for (int64_t c = 0; c < inner_pad_before; c++) {
                            dest_row_base[c] = dest_row_interior[p_map_left[c]];
                        }
                    }
                }
                if (inner_pad_after > 0 && inner_len > 0) {
                    T *dest_row_right = dest_row_interior + inner_len;
                    if (mode == 1) {
                        fill_typed<T>(dest_row_right, inner_pad_after, dest_row_interior[inner_len - 1]);
                    } else {
                        for (int64_t c = 0; c < inner_pad_after; c++) {
                            dest_row_right[c] = dest_row_interior[p_map_right[c]];
                        }
                    }
                }
            }

            // Advance outer coordinate
            for (int64_t d = rank - 2; d >= 0; d--) {
                p_coord[d]++;
                if (p_coord[d] < src_shape[d]) {
                    src_offset += src_strides[d];
                    dest_base_offset += p_dest_strides[d];
                    break;
                }
                p_coord[d] = 0;
                src_offset -= (src_shape[d] - 1) * src_strides[d];
                dest_base_offset -= (src_shape[d] - 1) * p_dest_strides[d];
            }
        }
    }

    if (mode == 0 && is_uniform_constant) {
        return 0;
    }

    if (mode == 0) {
        // Non-uniform constant mode: pad axes in forward order 0 .. rank - 1
        // so later axes overwrite earlier axes at corner/edge intersections,
        // matching sequential axis-by-axis padding.
        for (int64_t dim = 0; dim < rank; dim++) {
            int64_t b_dim = pad_before[dim];
            int64_t a_dim = pad_after[dim];
            if (b_dim == 0 && a_dim == 0) {
                continue;
            }

            int64_t s_dim = src_shape[dim];
            int64_t d_dim = dest_shape[dim];
            int64_t block_elements = p_dest_strides[dim];
            int64_t slice_stride = d_dim * block_elements;
            T cb_dim = const_before ? const_before[dim] : T{};
            T ca_dim = const_after ? const_after[dim] : T{};

            int64_t outer_dest_elements = 1;
            for (int64_t d = 0; d < dim; d++) {
                outer_dest_elements *= dest_shape[d];
            }

            for (int64_t el = 0; el < outer_dest_elements; el++) {
                T *base = dest + el * slice_stride;
                if (b_dim > 0) {
                    fill_typed<T>(base, b_dim * block_elements, cb_dim);
                }
                if (a_dim > 0) {
                    fill_typed<T>(base + (b_dim + s_dim) * block_elements, a_dim * block_elements, ca_dim);
                }
            }
        }
        return 0;
    }

    // Step 2: Pad outer axes from rank - 2 down to 0
    for (int64_t dim = rank - 2; dim >= 0; dim--) {
        int64_t b_dim = pad_before[dim];
        int64_t a_dim = pad_after[dim];
        int64_t s_dim = src_shape[dim];
        int64_t d_dim = dest_shape[dim];
        int64_t block_elements = p_dest_strides[dim];
        size_t block_bytes = (size_t)block_elements * sizeof(T);
        T cb_dim = const_before ? const_before[dim] : T{};
        T ca_dim = const_after ? const_after[dim] : T{};

        if (b_dim == 0 && a_dim == 0) {
            continue;
        }

        int64_t outer_dim_elements = 1;
        for (int64_t d = 0; d < dim; d++) {
            outer_dim_elements *= src_shape[d];
        }

        int64_t coord_dim[32] = {0};
        NoThrowBuffer<int64_t> coord_dim_vec;
        int64_t *p_coord_dim = coord_dim;
        if (dim > 32) {
            if (!coord_dim_vec.assign(dim, 0)) return -4;
            p_coord_dim = coord_dim_vec.data();
        }

        int64_t dest_dim_base = 0;
        for (int64_t d = 0; d < dim; d++) {
            dest_dim_base += pad_before[d] * p_dest_strides[d];
        }

        for (int64_t el = 0; el < outer_dim_elements; el++) {
            T *base = dest + dest_dim_base;

            // Fill before padding
            if (b_dim > 0) {
                if (mode == 0) {
                    for (int64_t j = 0; j < b_dim; j++) {
                        fill_typed<T>(base + j * block_elements, block_elements, cb_dim);
                    }
                } else if (s_dim > 0) {
                    if (mode == 1) { // edge
                        const T *src_block = base + b_dim * block_elements;
                        for (int64_t j = 0; j < b_dim; j++) {
                            memcpy(base + j * block_elements, src_block, block_bytes);
                        }
                    } else { // reflect, symmetric, wrap
                        for (int64_t j = 0; j < b_dim; j++) {
                            int64_t src_j = pad_map_index(j - b_dim, s_dim, mode);
                            const T *src_block = base + (b_dim + src_j) * block_elements;
                            memcpy(base + j * block_elements, src_block, block_bytes);
                        }
                    }
                }
            }

            // Fill after padding
            if (a_dim > 0) {
                if (mode == 0) {
                    for (int64_t j = b_dim + s_dim; j < d_dim; j++) {
                        fill_typed<T>(base + j * block_elements, block_elements, ca_dim);
                    }
                } else if (s_dim > 0) {
                    if (mode == 1) { // edge
                        const T *src_block = base + (b_dim + s_dim - 1) * block_elements;
                        for (int64_t j = b_dim + s_dim; j < d_dim; j++) {
                            memcpy(base + j * block_elements, src_block, block_bytes);
                        }
                    } else { // reflect, symmetric, wrap
                        for (int64_t j = b_dim + s_dim; j < d_dim; j++) {
                            int64_t src_j = pad_map_index(j - b_dim, s_dim, mode);
                            const T *src_block = base + (b_dim + src_j) * block_elements;
                            memcpy(base + j * block_elements, src_block, block_bytes);
                        }
                    }
                }
            }

            // Advance outer coordinate
            for (int64_t d = dim - 1; d >= 0; d--) {
                p_coord_dim[d]++;
                if (p_coord_dim[d] < src_shape[d]) {
                    dest_dim_base += p_dest_strides[d];
                    break;
                }
                p_coord_dim[d] = 0;
                dest_dim_base -= (src_shape[d] - 1) * p_dest_strides[d];
            }
        }
    }

    return 0;
}

extern "C" int native_pad_2d(
    int dtype,
    const void *src,
    int64_t src_rows,
    int64_t src_cols,
    int64_t src_stride_rows,
    int64_t src_stride_cols,
    void *dest,
    int64_t pad_top,
    int64_t pad_bottom,
    int64_t pad_left,
    int64_t pad_right,
    int mode,
    const void *const_before,
    const void *const_after,
    int is_uniform_constant
) {
    if (src == nullptr || dest == nullptr) return -3;
    switch (dtype) {
        case DTYPE_FLOAT64:
            return pad_2d_impl<double>(
                (const double *)src, src_rows, src_cols, src_stride_rows, src_stride_cols,
                (double *)dest, pad_top, pad_bottom, pad_left, pad_right, mode,
                (const double *)const_before, (const double *)const_after, is_uniform_constant);
        case DTYPE_FLOAT32:
            return pad_2d_impl<float>(
                (const float *)src, src_rows, src_cols, src_stride_rows, src_stride_cols,
                (float *)dest, pad_top, pad_bottom, pad_left, pad_right, mode,
                (const float *)const_before, (const float *)const_after, is_uniform_constant);
        case DTYPE_INT64:
            return pad_2d_impl<int64_t>(
                (const int64_t *)src, src_rows, src_cols, src_stride_rows, src_stride_cols,
                (int64_t *)dest, pad_top, pad_bottom, pad_left, pad_right, mode,
                (const int64_t *)const_before, (const int64_t *)const_after, is_uniform_constant);
        case DTYPE_INT32:
            return pad_2d_impl<int32_t>(
                (const int32_t *)src, src_rows, src_cols, src_stride_rows, src_stride_cols,
                (int32_t *)dest, pad_top, pad_bottom, pad_left, pad_right, mode,
                (const int32_t *)const_before, (const int32_t *)const_after, is_uniform_constant);
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            return pad_2d_impl<uint8_t>(
                (const uint8_t *)src, src_rows, src_cols, src_stride_rows, src_stride_cols,
                (uint8_t *)dest, pad_top, pad_bottom, pad_left, pad_right, mode,
                (const uint8_t *)const_before, (const uint8_t *)const_after, is_uniform_constant);
        case DTYPE_INT16:
            return pad_2d_impl<int16_t>(
                (const int16_t *)src, src_rows, src_cols, src_stride_rows, src_stride_cols,
                (int16_t *)dest, pad_top, pad_bottom, pad_left, pad_right, mode,
                (const int16_t *)const_before, (const int16_t *)const_after, is_uniform_constant);
        case DTYPE_UINT64:
            return pad_2d_impl<uint64_t>(
                (const uint64_t *)src, src_rows, src_cols, src_stride_rows, src_stride_cols,
                (uint64_t *)dest, pad_top, pad_bottom, pad_left, pad_right, mode,
                (const uint64_t *)const_before, (const uint64_t *)const_after, is_uniform_constant);
        case DTYPE_UINT32:
            return pad_2d_impl<uint32_t>(
                (const uint32_t *)src, src_rows, src_cols, src_stride_rows, src_stride_cols,
                (uint32_t *)dest, pad_top, pad_bottom, pad_left, pad_right, mode,
                (const uint32_t *)const_before, (const uint32_t *)const_after, is_uniform_constant);
        case DTYPE_UINT16:
            return pad_2d_impl<uint16_t>(
                (const uint16_t *)src, src_rows, src_cols, src_stride_rows, src_stride_cols,
                (uint16_t *)dest, pad_top, pad_bottom, pad_left, pad_right, mode,
                (const uint16_t *)const_before, (const uint16_t *)const_after, is_uniform_constant);
        case DTYPE_INT8:
            return pad_2d_impl<int8_t>(
                (const int8_t *)src, src_rows, src_cols, src_stride_rows, src_stride_cols,
                (int8_t *)dest, pad_top, pad_bottom, pad_left, pad_right, mode,
                (const int8_t *)const_before, (const int8_t *)const_after, is_uniform_constant);
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
            return pad_2d_impl<uint16_t>(
                (const uint16_t *)src, src_rows, src_cols, src_stride_rows, src_stride_cols,
                (uint16_t *)dest, pad_top, pad_bottom, pad_left, pad_right, mode,
                (const uint16_t *)const_before, (const uint16_t *)const_after, is_uniform_constant);
        case DTYPE_COMPLEX128:
            return pad_2d_impl<complex128_t>(
                (const complex128_t *)src, src_rows, src_cols, src_stride_rows, src_stride_cols,
                (complex128_t *)dest, pad_top, pad_bottom, pad_left, pad_right, mode,
                (const complex128_t *)const_before, (const complex128_t *)const_after, is_uniform_constant);
        case DTYPE_COMPLEX64:
            return pad_2d_impl<complex64_t>(
                (const complex64_t *)src, src_rows, src_cols, src_stride_rows, src_stride_cols,
                (complex64_t *)dest, pad_top, pad_bottom, pad_left, pad_right, mode,
                (const complex64_t *)const_before, (const complex64_t *)const_after, is_uniform_constant);
        default:
            return -2;
    }
}

extern "C" int native_pad_nd(
    int dtype,
    const void *src,
    const int64_t *src_shape,
    const int64_t *src_strides,
    void *dest,
    const int64_t *dest_shape,
    const int64_t *dest_strides,
    const int64_t *pad_before,
    const int64_t *pad_after,
    int64_t rank,
    int mode,
    const void *const_before,
    const void *const_after,
    int is_uniform_constant
) {
    if (src == nullptr || dest == nullptr || src_shape == nullptr || src_strides == nullptr ||
        dest_shape == nullptr || pad_before == nullptr || pad_after == nullptr) {
        return -3;
    }
    (void)dest_strides;
    switch (dtype) {
        case DTYPE_FLOAT64:
            return pad_nd_impl<double>(
                (const double *)src, src_shape, src_strides,
                (double *)dest, dest_shape, pad_before, pad_after, rank, mode,
                (const double *)const_before, (const double *)const_after, is_uniform_constant);
        case DTYPE_FLOAT32:
            return pad_nd_impl<float>(
                (const float *)src, src_shape, src_strides,
                (float *)dest, dest_shape, pad_before, pad_after, rank, mode,
                (const float *)const_before, (const float *)const_after, is_uniform_constant);
        case DTYPE_INT64:
            return pad_nd_impl<int64_t>(
                (const int64_t *)src, src_shape, src_strides,
                (int64_t *)dest, dest_shape, pad_before, pad_after, rank, mode,
                (const int64_t *)const_before, (const int64_t *)const_after, is_uniform_constant);
        case DTYPE_INT32:
            return pad_nd_impl<int32_t>(
                (const int32_t *)src, src_shape, src_strides,
                (int32_t *)dest, dest_shape, pad_before, pad_after, rank, mode,
                (const int32_t *)const_before, (const int32_t *)const_after, is_uniform_constant);
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            return pad_nd_impl<uint8_t>(
                (const uint8_t *)src, src_shape, src_strides,
                (uint8_t *)dest, dest_shape, pad_before, pad_after, rank, mode,
                (const uint8_t *)const_before, (const uint8_t *)const_after, is_uniform_constant);
        case DTYPE_INT16:
            return pad_nd_impl<int16_t>(
                (const int16_t *)src, src_shape, src_strides,
                (int16_t *)dest, dest_shape, pad_before, pad_after, rank, mode,
                (const int16_t *)const_before, (const int16_t *)const_after, is_uniform_constant);
        case DTYPE_UINT64:
            return pad_nd_impl<uint64_t>(
                (const uint64_t *)src, src_shape, src_strides,
                (uint64_t *)dest, dest_shape, pad_before, pad_after, rank, mode,
                (const uint64_t *)const_before, (const uint64_t *)const_after, is_uniform_constant);
        case DTYPE_UINT32:
            return pad_nd_impl<uint32_t>(
                (const uint32_t *)src, src_shape, src_strides,
                (uint32_t *)dest, dest_shape, pad_before, pad_after, rank, mode,
                (const uint32_t *)const_before, (const uint32_t *)const_after, is_uniform_constant);
        case DTYPE_UINT16:
            return pad_nd_impl<uint16_t>(
                (const uint16_t *)src, src_shape, src_strides,
                (uint16_t *)dest, dest_shape, pad_before, pad_after, rank, mode,
                (const uint16_t *)const_before, (const uint16_t *)const_after, is_uniform_constant);
        case DTYPE_INT8:
            return pad_nd_impl<int8_t>(
                (const int8_t *)src, src_shape, src_strides,
                (int8_t *)dest, dest_shape, pad_before, pad_after, rank, mode,
                (const int8_t *)const_before, (const int8_t *)const_after, is_uniform_constant);
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
            return pad_nd_impl<uint16_t>(
                (const uint16_t *)src, src_shape, src_strides,
                (uint16_t *)dest, dest_shape, pad_before, pad_after, rank, mode,
                (const uint16_t *)const_before, (const uint16_t *)const_after, is_uniform_constant);
        case DTYPE_COMPLEX128:
            return pad_nd_impl<complex128_t>(
                (const complex128_t *)src, src_shape, src_strides,
                (complex128_t *)dest, dest_shape, pad_before, pad_after, rank, mode,
                (const complex128_t *)const_before, (const complex128_t *)const_after, is_uniform_constant);
        case DTYPE_COMPLEX64:
            return pad_nd_impl<complex64_t>(
                (const complex64_t *)src, src_shape, src_strides,
                (complex64_t *)dest, dest_shape, pad_before, pad_after, rank, mode,
                (const complex64_t *)const_before, (const complex64_t *)const_after, is_uniform_constant);
        default:
            return -2;
    }
}


// ============================================================================
// 4. ROLL KERNELS
// ============================================================================
static inline size_t get_roll_dtype_itemsize(int dtype) {
    switch (dtype) {
        case DTYPE_FLOAT64:
        case DTYPE_INT64:
        case DTYPE_UINT64:
        case DTYPE_COMPLEX64:
            return 8;
        case DTYPE_FLOAT32:
        case DTYPE_INT32:
        case DTYPE_UINT32:
            return 4;
        case DTYPE_INT16:
        case DTYPE_UINT16:
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
            return 2;
        case DTYPE_INT8:
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            return 1;
        case DTYPE_COMPLEX128:
            return 16;
        default:
            abort();
    }
}

static inline bool is_c_contiguous(const int64_t *shape, const int64_t *strides, int64_t rank) {
    if (strides == nullptr) return true;
    int64_t expected_stride = 1;
    for (int64_t d = rank - 1; d >= 0; d--) {
        if (shape[d] > 1 && strides[d] != expected_stride) {
            return false;
        }
        expected_stride *= shape[d];
    }
    return true;
}

extern "C" int native_roll_1d(
    int dtype,
    const void *src,
    int64_t size,
    int64_t shift,
    void *dest
) {
    if (src == nullptr || dest == nullptr) {
        return -3;
    }
    if (size <= 0) return 0;

    size_t itemsize = get_roll_dtype_itemsize(dtype);
    if (itemsize == 0) return -2;

    int64_t s = shift % size;
    if (s < 0) s += size;

    if (s == 0) {
        if (src != dest) {
            memmove(dest, src, (size_t)size * itemsize);
        }
        return 0;
    }

    size_t total_bytes = (size_t)size * itemsize;
    size_t b1 = (size_t)s * itemsize;
    size_t b2 = (size_t)(size - s) * itemsize;
    const uint8_t *s_ptr = (const uint8_t *)src;
    uint8_t *d_ptr = (uint8_t *)dest;

    bool overlaps = d_ptr < s_ptr + total_bytes && s_ptr < d_ptr + total_bytes;
    if (!overlaps) {
        memcpy(d_ptr, s_ptr + b2, b1);
        memcpy(d_ptr + b1, s_ptr, b2);
    } else {
        NoThrowBuffer<uint8_t> tmp(total_bytes);
        if (!tmp.ok()) return -4;
        memcpy(tmp.data(), s_ptr + b2, b1);
        memcpy(tmp.data() + b1, s_ptr, b2);
        memcpy(d_ptr, tmp.data(), total_bytes);
    }
    return 0;
}

template <typename T>
static int roll_strided_impl(
    const T *src,
    const int64_t *RESTRICT shape,
    const int64_t *RESTRICT src_strides,
    int64_t rank,
    int64_t shift,
    int64_t axis,
    T *dest,
    const int64_t *RESTRICT dest_strides
) {
    if (rank <= 0) {
        if (src != dest) {
            dest[0] = src[0];
        }
        return 0;
    }

    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) {
        if (shape[i] <= 0) return 0;
        total_elements *= shape[i];
    }

    int64_t axis_size = shape[axis];
    int64_t s = shift % axis_size;
    if (s < 0) s += axis_size;

    NoThrowBuffer<T> temp(total_elements);
    NoThrowBuffer<int64_t> coords(rank, 0);
    if (!temp.ok() || !coords.ok()) return -4;

    for (int64_t idx = 0; idx < total_elements; idx++) {
        int64_t src_offset = 0;
        for (int64_t d = 0; d < rank; d++) {
            src_offset += coords[d] * src_strides[d];
        }
        temp[idx] = src[src_offset];

        for (int64_t d = rank - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < shape[d]) {
                break;
            }
            coords[d] = 0;
        }
    }

    std::fill(coords.begin(), coords.end(), 0);
    for (int64_t idx = 0; idx < total_elements; idx++) {
        int64_t dest_offset = 0;
        for (int64_t d = 0; d < rank; d++) {
            if (d == axis) {
                int64_t dest_coord_axis = (coords[d] + s) % axis_size;
                dest_offset += dest_coord_axis * dest_strides[d];
            } else {
                dest_offset += coords[d] * dest_strides[d];
            }
        }
        dest[dest_offset] = temp[idx];

        for (int64_t d = rank - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < shape[d]) {
                break;
            }
            coords[d] = 0;
        }
    }
    return 0;
}

static int dispatch_roll_strided_by_dtype(
    int dtype,
    const void *src,
    const int64_t *shape,
    const int64_t *src_strides,
    int64_t rank,
    int64_t shift,
    int64_t axis,
    void *dest,
    const int64_t *dest_strides
) {
    switch (dtype) {
        case DTYPE_FLOAT64:
            return roll_strided_impl<double>(
                (const double *)src, shape, src_strides, rank, shift, axis,
                (double *)dest, dest_strides);
        case DTYPE_FLOAT32:
            return roll_strided_impl<float>(
                (const float *)src, shape, src_strides, rank, shift, axis,
                (float *)dest, dest_strides);
        case DTYPE_INT64:
            return roll_strided_impl<int64_t>(
                (const int64_t *)src, shape, src_strides, rank, shift, axis,
                (int64_t *)dest, dest_strides);
        case DTYPE_INT32:
            return roll_strided_impl<int32_t>(
                (const int32_t *)src, shape, src_strides, rank, shift, axis,
                (int32_t *)dest, dest_strides);
        case DTYPE_UINT64:
            return roll_strided_impl<uint64_t>(
                (const uint64_t *)src, shape, src_strides, rank, shift, axis,
                (uint64_t *)dest, dest_strides);
        case DTYPE_UINT32:
            return roll_strided_impl<uint32_t>(
                (const uint32_t *)src, shape, src_strides, rank, shift, axis,
                (uint32_t *)dest, dest_strides);
        case DTYPE_UINT16:
            return roll_strided_impl<uint16_t>(
                (const uint16_t *)src, shape, src_strides, rank, shift, axis,
                (uint16_t *)dest, dest_strides);
        case DTYPE_INT16:
            return roll_strided_impl<int16_t>(
                (const int16_t *)src, shape, src_strides, rank, shift, axis,
                (int16_t *)dest, dest_strides);
        case DTYPE_INT8:
            return roll_strided_impl<int8_t>(
                (const int8_t *)src, shape, src_strides, rank, shift, axis,
                (int8_t *)dest, dest_strides);
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            return roll_strided_impl<uint8_t>(
                (const uint8_t *)src, shape, src_strides, rank, shift, axis,
                (uint8_t *)dest, dest_strides);
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
            return roll_strided_impl<uint16_t>(
                (const uint16_t *)src, shape, src_strides, rank, shift, axis,
                (uint16_t *)dest, dest_strides);
        case DTYPE_COMPLEX128:
            return roll_strided_impl<complex128_t>(
                (const complex128_t *)src, shape, src_strides, rank, shift, axis,
                (complex128_t *)dest, dest_strides);
        case DTYPE_COMPLEX64:
            return roll_strided_impl<complex64_t>(
                (const complex64_t *)src, shape, src_strides, rank, shift, axis,
                (complex64_t *)dest, dest_strides);
        default:
            return -2;
    }
}

static int roll_contiguous_impl(
    size_t itemsize,
    const void *src,
    const int64_t *shape,
    int64_t rank,
    int64_t shift,
    int64_t axis,
    void *dest
) {
    if (rank <= 0) {
        if (src != dest) {
            memmove(dest, src, itemsize);
        }
        return 0;
    }
    const int64_t axis_size = shape[axis];
    if (axis_size <= 0) return 0;

    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) {
        if (shape[i] <= 0) return 0;
        total_elements *= shape[i];
    }
    if (total_elements == 0) return 0;

    size_t total_bytes = (size_t)total_elements * itemsize;
    int64_t s = shift % axis_size;
    if (s < 0) s += axis_size;

    if (s == 0) {
        if (src != dest) {
            memmove(dest, src, total_bytes);
        }
        return 0;
    }

    const uint8_t *s_ptr = (const uint8_t *)src;
    uint8_t *d_ptr = (uint8_t *)dest;
    bool overlaps = d_ptr < s_ptr + total_bytes && s_ptr < d_ptr + total_bytes;

    if (rank == 1) {
        size_t b1 = (size_t)s * itemsize;
        size_t b2 = (size_t)(axis_size - s) * itemsize;
        if (!overlaps) {
            memcpy(d_ptr, s_ptr + b2, b1);
            memcpy(d_ptr + b1, s_ptr, b2);
        } else {
            NoThrowBuffer<uint8_t> tmp(total_bytes);
            if (!tmp.ok()) return -4;
            memcpy(tmp.data(), s_ptr + b2, b1);
            memcpy(tmp.data() + b1, s_ptr, b2);
            memcpy(d_ptr, tmp.data(), total_bytes);
        }
        return 0;
    }

    int64_t outer_count = 1;
    for (int64_t i = 0; i < axis; i++) {
        outer_count *= shape[i];
    }

    int64_t inner_count = 1;
    for (int64_t i = axis + 1; i < rank; i++) {
        inner_count *= shape[i];
    }

    size_t chunk_bytes = (size_t)inner_count * itemsize;
    size_t outer_stride_bytes = (size_t)axis_size * chunk_bytes;
    size_t b1 = (size_t)s * chunk_bytes;
    size_t b2 = (size_t)(axis_size - s) * chunk_bytes;

    if (!overlaps) {
        for (int64_t o = 0; o < outer_count; o++) {
            const uint8_t *src_outer = s_ptr + o * outer_stride_bytes;
            uint8_t *dest_outer = d_ptr + o * outer_stride_bytes;
            memcpy(dest_outer, src_outer + b2, b1);
            memcpy(dest_outer + b1, src_outer, b2);
        }
    } else {
        NoThrowBuffer<uint8_t> tmp(total_bytes);
        if (!tmp.ok()) return -4;
        for (int64_t o = 0; o < outer_count; o++) {
            const uint8_t *src_outer = s_ptr + o * outer_stride_bytes;
            uint8_t *tmp_outer = tmp.data() + o * outer_stride_bytes;
            memcpy(tmp_outer, src_outer + b2, b1);
            memcpy(tmp_outer + b1, src_outer, b2);
        }
        memcpy(d_ptr, tmp.data(), total_bytes);
    }

    return 0;
}

extern "C" int native_roll_nd(
    int dtype,
    const void *src,
    const int64_t *shape,
    const int64_t *src_strides,
    int64_t rank,
    int64_t shift,
    int64_t axis,
    void *dest,
    const int64_t *dest_strides
) {
    if (src == nullptr || dest == nullptr || shape == nullptr) {
        return -3;
    }
    if (rank > 0 && (axis < 0 || axis >= rank)) {
        return -1;
    }

    size_t itemsize = get_roll_dtype_itemsize(dtype);
    if (itemsize == 0) return -2;

    bool src_contig = is_c_contiguous(shape, src_strides, rank);
    bool dest_contig = is_c_contiguous(shape, dest_strides, rank);

    if (src_contig && dest_contig) {
        return roll_contiguous_impl(itemsize, src, shape, rank, shift, axis, dest);
    } else {
        if (src_strides == nullptr || dest_strides == nullptr) {
            return -3;
        }
        return dispatch_roll_strided_by_dtype(
            dtype, src, shape, src_strides, rank, shift, axis, dest, dest_strides);
    }
}
