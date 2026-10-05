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

#pragma once

#include <stddef.h>
#include <stdint.h>

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

#ifdef __cplusplus
extern "C" {
#endif

int native_take_along_axis(
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
);

int native_put_along_axis(
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
);

int native_tile_contiguous(
    int dtype,
    const void *src,
    const int64_t *src_shape,
    const int64_t *reps,
    void *dest,
    const int64_t *out_shape,
    int64_t rank
);

int native_tile_strided(
    int dtype,
    const void *src,
    const int64_t *src_shape,
    const int64_t *src_strides,
    const int64_t *reps,
    void *dest,
    const int64_t *out_shape,
    const int64_t *out_strides,
    int64_t rank
);

int native_pad_2d(
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
);

int native_pad_nd(
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
);

int native_roll_1d(
    int dtype,
    const void *src,
    int64_t size,
    int64_t shift,
    void *dest
);

int native_roll_nd(
    int dtype,
    const void *src,
    const int64_t *shape,
    const int64_t *src_strides,
    int64_t rank,
    int64_t shift,
    int64_t axis,
    void *dest,
    const int64_t *dest_strides
);

int native_unravel_index(
    int index_dtype,
    const void *indices_ptr,
    const int64_t *indices_shape,
    const int64_t *indices_strides,
    int64_t indices_rank,
    int64_t indices_size,
    const int64_t *dims,
    int64_t ndims,
    int order,
    int64_t **out_ptrs,
    const int64_t *out_strides_flat,
    int is_contiguous,
    int64_t *out_error_idx
);

int native_ravel_multi_index(
    const int64_t *const *coords_ptrs,
    const int64_t *coords_strides_flat,
    const int64_t *target_shape,
    int64_t target_rank,
    int64_t total_size,
    const int64_t *dims,
    const int *modes,
    int64_t ndims,
    int order,
    int64_t *out_ptr,
    const int64_t *out_strides,
    int is_contiguous,
    int64_t *out_error_val
);

int native_indices_int64(
    const int64_t *dims,
    int64_t ndims,
    int64_t slice_size,
    int64_t *out_ptr
);

int native_tril_indices(
    int64_t n,
    int64_t m,
    int64_t k,
    int64_t *out_row,
    int64_t *out_col
);

int native_triu_indices(
    int64_t n,
    int64_t m,
    int64_t k,
    int64_t *out_row,
    int64_t *out_col
);

#ifdef __cplusplus
}
#endif

