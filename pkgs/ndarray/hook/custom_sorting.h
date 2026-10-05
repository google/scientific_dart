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

#ifdef __cplusplus
extern "C" {
#endif

void ndarray_set_oom_flag(void);
int ndarray_consume_oom_flag(void);

/* Test support: counts calls in a process-global atomic (safe to use as a
 * `NativeFinalizer` callback, which may run on a GC helper thread) and
 * returns-and-resets the count. */
void ndarray_test_finalizer_hit(void *token);
int ndarray_test_consume_finalizer_hits(void);

int64_t unpack_mask_c(const uint8_t *mask_ptr, int64_t size, int64_t stride, int64_t *out_indices);
int64_t native_count_mask(const uint8_t *mask, int64_t size);
void native_apply_mask(int dtype, const void *src, const uint8_t *mask, void *dest, int64_t size);

// ----------------------------------------------------------------------------
// Public Sorters with Kind Parameter
// kind: 0 = quicksort, 1 = mergesort/stable, 2 = heapsort
// ----------------------------------------------------------------------------
void native_sort_double(double *array, int64_t size, int kind);
void native_sort_float(float *array, int64_t size, int kind);
void native_sort_int64(long long *array, int64_t size, int kind);
void native_sort_int32(int *array, int64_t size, int kind);
void native_sort_int16(int16_t *array, int64_t size, int kind);
void native_sort_int8(int8_t *array, int64_t size, int kind);
void native_sort_uint64(uint64_t *array, int64_t size, int kind);
void native_sort_uint32(uint32_t *array, int64_t size, int kind);
void native_sort_uint16(uint16_t *array, int64_t size, int kind);
void native_sort_uint8(uint8_t *array, int64_t size, int kind);
void native_sort_float16(uint16_t *array, int64_t size, int kind);
void native_sort_bfloat16(uint16_t *array, int64_t size, int kind);
void native_sort_complex128(double *array, int64_t size, int kind);
void native_sort_complex64(float *array, int64_t size, int kind);

// ----------------------------------------------------------------------------
// Public Argsort Sorters with Kind Parameter
// ----------------------------------------------------------------------------
void native_argsort_double(const double *data, int64_t *indices, int64_t size, int kind);
void native_argsort_float(const float *data, int64_t *indices, int64_t size, int kind);
void native_argsort_int64(const long long *data, int64_t *indices, int64_t size, int kind);
void native_argsort_int32(const int *data, int64_t *indices, int64_t size, int kind);
void native_argsort_int16(const int16_t *data, int64_t *indices, int64_t size, int kind);
void native_argsort_int8(const int8_t *data, int64_t *indices, int64_t size, int kind);
void native_argsort_uint64(const uint64_t *data, int64_t *indices, int64_t size, int kind);
void native_argsort_uint32(const uint32_t *data, int64_t *indices, int64_t size, int kind);
void native_argsort_uint16(const uint16_t *data, int64_t *indices, int64_t size, int kind);
void native_argsort_uint8(const uint8_t *data, int64_t *indices, int64_t size, int kind);
void native_argsort_float16(const uint16_t *data, int64_t *indices, int64_t size, int kind);
void native_argsort_bfloat16(const uint16_t *data, int64_t *indices, int64_t size, int kind);

// ----------------------------------------------------------------------------
// Public Partition Sorters
// ----------------------------------------------------------------------------
void native_partition_double(double *array, int64_t size, const int64_t *k_list, int64_t k_size);
void native_partition_float(float *array, int64_t size, const int64_t *k_list, int64_t k_size);
void native_partition_int64(long long *array, int64_t size, const int64_t *k_list, int64_t k_size);
void native_partition_int32(int *array, int64_t size, const int64_t *k_list, int64_t k_size);
void native_partition_int16(int16_t *array, int64_t size, const int64_t *k_list, int64_t k_size);
void native_partition_int8(int8_t *array, int64_t size, const int64_t *k_list, int64_t k_size);
void native_partition_uint64(uint64_t *array, int64_t size, const int64_t *k_list, int64_t k_size);
void native_partition_uint32(uint32_t *array, int64_t size, const int64_t *k_list, int64_t k_size);
void native_partition_uint16(uint16_t *array, int64_t size, const int64_t *k_list, int64_t k_size);
void native_partition_uint8(uint8_t *array, int64_t size, const int64_t *k_list, int64_t k_size);
void native_partition_float16(uint16_t *array, int64_t size, const int64_t *k_list, int64_t k_size);
void native_partition_bfloat16(uint16_t *array, int64_t size, const int64_t *k_list, int64_t k_size);
void native_partition_complex128(double *array, int64_t size, const int64_t *k_list, int64_t k_size);
void native_partition_complex64(float *array, int64_t size, const int64_t *k_list, int64_t k_size);

// ----------------------------------------------------------------------------
// Public Argpartition Sorters
// ----------------------------------------------------------------------------
void native_argpartition_double(const double *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size);
void native_argpartition_float(const float *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size);
void native_argpartition_int64(const long long *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size);
void native_argpartition_int32(const int *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size);
void native_argpartition_int16(const int16_t *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size);
void native_argpartition_int8(const int8_t *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size);
void native_argpartition_uint64(const uint64_t *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size);
void native_argpartition_uint32(const uint32_t *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size);
void native_argpartition_uint16(const uint16_t *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size);
void native_argpartition_uint8(const uint8_t *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size);
void native_argpartition_float16(const uint16_t *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size);
void native_argpartition_bfloat16(const uint16_t *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size);
void native_argpartition_complex128(const double *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size);
void native_argpartition_complex64(const float *data, int64_t *indices, int64_t size, const int64_t *k_list, int64_t k_size);

// ----------------------------------------------------------------------------
// Public Searchsorted (Binary Search) functions
// ----------------------------------------------------------------------------
void native_searchsorted_double(const double *array, int64_t size, const double *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter);
void native_searchsorted_float(const float *array, int64_t size, const float *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter);
void native_searchsorted_int64(const long long *array, int64_t size, const long long *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter);
void native_searchsorted_int32(const int *array, int64_t size, const int *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter);
void native_searchsorted_int16(const int16_t *array, int64_t size, const int16_t *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter);
void native_searchsorted_int8(const int8_t *array, int64_t size, const int8_t *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter);
void native_searchsorted_uint64(const uint64_t *array, int64_t size, const uint64_t *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter);
void native_searchsorted_uint32(const uint32_t *array, int64_t size, const uint32_t *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter);
void native_searchsorted_uint16(const uint16_t *array, int64_t size, const uint16_t *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter);
void native_searchsorted_uint8(const uint8_t *array, int64_t size, const uint8_t *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter);
void native_searchsorted_float16(const uint16_t *array, int64_t size, const uint16_t *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter);
void native_searchsorted_bfloat16(const uint16_t *array, int64_t size, const uint16_t *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter);
void native_searchsorted_complex128(const double *array, int64_t size, const double *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter);
void native_searchsorted_complex64(const float *array, int64_t size, const float *values, int64_t *out_indices, int64_t num_values, int side_left, const int64_t *sorter);

// ----------------------------------------------------------------------------
// Utility operations
// ----------------------------------------------------------------------------
int custom_memcmp(const void *s1, const void *s2, size_t n);
void native_zero_memory(void *ptr, size_t bytes);
void custom_memcpy(void *dest, const void *src, size_t n);
void native_collect_nonzero_coords(const unsigned char *cond, int64_t total_size, const int64_t *shape, const int64_t *strides, int rank, int64_t **out_coords);
void native_collect_nonzero_coords_grouped(const unsigned char *cond, int64_t total_size, const int64_t *shape, const int64_t *strides, int rank, int64_t *out_coords);
void native_to_bool_mask_double(const void *src, int64_t size, const int64_t *shape, const int64_t *strides, int rank, int is_contiguous, unsigned char *dest);
void native_to_bool_mask_float(const void *src, int64_t size, const int64_t *shape, const int64_t *strides, int rank, int is_contiguous, unsigned char *dest);
void native_to_bool_mask_int64(const void *src, int64_t size, const int64_t *shape, const int64_t *strides, int rank, int is_contiguous, unsigned char *dest);
void native_to_bool_mask_int32(const void *src, int64_t size, const int64_t *shape, const int64_t *strides, int rank, int is_contiguous, unsigned char *dest);
void native_to_bool_mask_complex128(const void *src, int64_t size, const int64_t *shape, const int64_t *strides, int rank, int is_contiguous, unsigned char *dest);
void native_to_bool_mask_complex64(const void *src, int64_t size, const int64_t *shape, const int64_t *strides, int rank, int is_contiguous, unsigned char *dest);
void native_to_bool_mask_uint8(const void *src, int64_t size, const int64_t *shape, const int64_t *strides, int rank, int is_contiguous, unsigned char *dest);
void native_to_bool_mask_int16(const void *src, int64_t size, const int64_t *shape, const int64_t *strides, int rank, int is_contiguous, unsigned char *dest);
void native_argminmax_double(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_max, int is_contiguous);
void native_argminmax_float(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_max, int is_contiguous);
void native_argminmax_int64(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_max, int is_contiguous);
void native_argminmax_int32(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_max, int is_contiguous);
void native_argminmax_uint8(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_max, int is_contiguous);
void native_argminmax_int16(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_max, int is_contiguous);
void native_count_nonzero_double(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_contiguous);
void native_count_nonzero_float(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_contiguous);
void native_count_nonzero_int64(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_contiguous);
void native_count_nonzero_int32(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_contiguous);
void native_count_nonzero_uint8(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_contiguous);
void native_count_nonzero_int16(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_contiguous);
void native_count_nonzero_complex128(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_contiguous);
void native_count_nonzero_complex64(const void *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int is_contiguous);

int64_t ndarray_unique(const void *src, void *dest, int64_t size, int dtype, int64_t *out_index, int64_t *out_inverse, int64_t *out_counts);


#ifdef __cplusplus
}
#endif

