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
#include <algorithm>
#include <cmath>
#include <limits>
#define VQSORT_ENABLED 1
#include "hwy/contrib/sort/vqsort.h"
#include "hwy/highway.h"
#include <vector>

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

static inline int compare_indices_double_timsort(int idx_a, int idx_b) {
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

static inline int compare_indices_float_timsort(int idx_a, int idx_b) {
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

static inline int compare_indices_int64_timsort(int idx_a, int idx_b) {
    long long val_a = global_int64_data[idx_a];
    long long val_b = global_int64_data[idx_b];
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

static inline int compare_indices_int32_timsort(int idx_a, int idx_b) {
    int val_a = global_int32_data[idx_a];
    int val_b = global_int32_data[idx_b];
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

static inline int compare_indices_int16_timsort(int idx_a, int idx_b) {
    int16_t val_a = global_int16_data[idx_a];
    int16_t val_b = global_int16_data[idx_b];
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

static inline int compare_indices_uint8_timsort(int idx_a, int idx_b) {
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
#define SORT_TYPE int
#define SORT_CMP(x, y) compare_indices_double_timsort(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_indices_float
#define SORT_TYPE int
#define SORT_CMP(x, y) compare_indices_float_timsort(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_indices_int64
#define SORT_TYPE int
#define SORT_CMP(x, y) compare_indices_int64_timsort(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_indices_int32
#define SORT_TYPE int
#define SORT_CMP(x, y) compare_indices_int32_timsort(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_indices_int16
#define SORT_TYPE int
#define SORT_CMP(x, y) compare_indices_int16_timsort(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_indices_uint8
#define SORT_TYPE int
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
static void insertion_sort(T *arr, int left, int right, Compare cmp) {
    for (int i = left + 1; i <= right; i++) {
        T key = arr[i];
        int j = i - 1;
        while (j >= left && cmp(key, arr[j]) < 0) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

template <typename T, typename Compare>
static void quicksort_rec(T *arr, int left, int right, Compare cmp) {
    if (right - left <= 10) {
        insertion_sort(arr, left, right, cmp);
        return;
    }
    int mid = left + (right - left) / 2;
    if (cmp(arr[mid], arr[left]) < 0) { T t = arr[left]; arr[left] = arr[mid]; arr[mid] = t; }
    if (cmp(arr[right], arr[left]) < 0) { T t = arr[left]; arr[left] = arr[right]; arr[right] = t; }
    if (cmp(arr[right], arr[mid]) < 0) { T t = arr[mid]; arr[mid] = arr[right]; arr[right] = t; }
    T pivot = arr[mid];
    T t1 = arr[mid]; arr[mid] = arr[right - 1]; arr[right - 1] = t1;
    int i = left;
    int j = right - 1;
    while (1) {
        while (cmp(arr[++i], pivot) < 0);
        while (cmp(pivot, arr[--j]) < 0);
        if (i >= j) break;
        T t2 = arr[i]; arr[i] = arr[j]; arr[j] = t2;
    }
    T t3 = arr[i]; arr[i] = arr[right - 1]; arr[right - 1] = t3;
    quicksort_rec(arr, left, i - 1, cmp);
    quicksort_rec(arr, i + 1, right, cmp);
}

template <typename T, typename Compare>
static void quicksort(T *arr, int size, Compare cmp) {
    if (arr == nullptr || size <= 1) return;
    quicksort_rec(arr, 0, size - 1, cmp);
}

template <typename T, typename Compare>
static void heapify(T *arr, int n, int i, Compare cmp) {
    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;
    if (l < n && cmp(arr[l], arr[largest]) > 0) largest = l;
    if (r < n && cmp(arr[r], arr[largest]) > 0) largest = r;
    if (largest != i) {
        T tmp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = tmp;
        heapify(arr, n, largest, cmp);
    }
}

template <typename T, typename Compare>
static void heapsort(T *arr, int size, Compare cmp) {
    if (arr == nullptr || size <= 1) return;
    for (int i = size / 2 - 1; i >= 0; i--)
        heapify(arr, size, i, cmp);
    for (int i = size - 1; i > 0; i--) {
        T tmp = arr[0];
        arr[0] = arr[i];
        arr[i] = tmp;
        heapify(arr, i, 0, cmp);
    }
}

template <typename T, typename Compare>
static void quickselect(T *arr, int left, int right, int k, Compare cmp) {
    while (left < right) {
        if (right - left <= 10) {
            insertion_sort(arr, left, right, cmp);
            return;
        }
        int pivot_idx = left + (right - left) / 2;
        T pivot = arr[pivot_idx];
        arr[pivot_idx] = arr[right];
        arr[right] = pivot;
        int i = left;
        for (int j = left; j < right; j++) {
            if (cmp(arr[j], pivot) < 0) {
                T tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
                i++;
            }
        }
        arr[right] = arr[i];
        arr[i] = pivot;
        if (i == k) {
            return;
        } else if (i < k) {
            left = i + 1;
        } else {
            right = i - 1;
        }
    }
}

template <typename T, typename Compare>
static void quickselect_multi(T *arr, int left, int right, const int *k_list, int k_start, int k_end, Compare cmp) {
    if (k_start > k_end || left >= right) return;
    int mid_k_idx = k_start + (k_end - k_start) / 2;
    int k = k_list[mid_k_idx];
    quickselect(arr, left, right, k, cmp);
    quickselect_multi(arr, left, k - 1, k_list, k_start, mid_k_idx - 1, cmp);
    quickselect_multi(arr, k + 1, right, k_list, mid_k_idx + 1, k_end, cmp);
}

template <typename T, typename Compare>
static void partition(T *arr, int size, const int *k_list, int k_size, Compare cmp) {
    if (arr == nullptr || size <= 1 || k_list == nullptr || k_size <= 0) return;
    quickselect_multi(arr, 0, size - 1, k_list, 0, k_size - 1, cmp);
}

template <typename T, typename Compare>
static void arg_insertion_sort(const T *arr, int *indices, int left, int right, Compare cmp) {
    for (int i = left + 1; i <= right; i++) {
        int key = indices[i];
        int j = i - 1;
        while (j >= left && cmp(arr[key], arr[indices[j]]) < 0) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key;
    }
}

template <typename T, typename Compare>
static void arg_quickselect(const T *arr, int *indices, int left, int right, int k, Compare cmp) {
    while (left < right) {
        if (right - left <= 10) {
            arg_insertion_sort(arr, indices, left, right, cmp);
            return;
        }
        int pivot_idx = left + (right - left) / 2;
        int pivot_val = indices[pivot_idx];
        indices[pivot_idx] = indices[right];
        indices[right] = pivot_val;
        int i = left;
        for (int j = left; j < right; j++) {
            if (cmp(arr[indices[j]], arr[pivot_val]) < 0) {
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
                i++;
            }
        }
        indices[right] = indices[i];
        indices[i] = pivot_val;
        if (i == k) {
            return;
        } else if (i < k) {
            left = i + 1;
        } else {
            right = i - 1;
        }
    }
}

template <typename T, typename Compare>
static void arg_quickselect_multi(const T *arr, int *indices, int left, int right, const int *k_list, int k_start, int k_end, Compare cmp) {
    if (k_start > k_end || left >= right) return;
    int mid_k_idx = k_start + (k_end - k_start) / 2;
    int k = k_list[mid_k_idx];
    arg_quickselect(arr, indices, left, right, k, cmp);
    arg_quickselect_multi(arr, indices, left, k - 1, k_list, k_start, mid_k_idx - 1, cmp);
    arg_quickselect_multi(arr, indices, k + 1, right, k_list, mid_k_idx + 1, k_end, cmp);
}

template <typename T, typename Compare>
static void argpartition(const T *arr, int *indices, int size, const int *k_list, int k_size, Compare cmp) {
    if (arr == nullptr || indices == nullptr || size <= 0 || k_list == nullptr || k_size <= 0) return;
    for (int i = 0; i < size; i++) indices[i] = i;
    if (size <= 1) return;
    arg_quickselect_multi(arr, indices, 0, size - 1, k_list, 0, k_size - 1, cmp);
}

template <typename T, typename Compare>
static void ind_quicksort_rec(const T *arr, int *indices, int left, int right, Compare cmp) {
    if (right - left <= 10) {
        arg_insertion_sort(arr, indices, left, right, cmp);
        return;
    }
    int mid = left + (right - left) / 2;
    if (cmp(arr[indices[mid]], arr[indices[left]]) < 0) { int t = indices[left]; indices[left] = indices[mid]; indices[mid] = t; }
    if (cmp(arr[indices[right]], arr[indices[left]]) < 0) { int t = indices[left]; indices[left] = indices[right]; indices[right] = t; }
    if (cmp(arr[indices[right]], arr[indices[mid]]) < 0) { int t = indices[mid]; indices[mid] = indices[right]; indices[right] = t; }
    int pivot = indices[mid];
    int t1 = indices[mid]; indices[mid] = indices[right - 1]; indices[right - 1] = t1;
    int i = left;
    int j = right - 1;
    while (1) {
        while (cmp(arr[indices[++i]], arr[pivot]) < 0);
        while (cmp(arr[pivot], arr[indices[--j]]) < 0);
        if (i >= j) break;
        int t2 = indices[i]; indices[i] = indices[j]; indices[j] = t2;
    }
    int t3 = indices[i]; indices[i] = indices[right - 1]; indices[right - 1] = t3;
    ind_quicksort_rec(arr, indices, left, i - 1, cmp);
    ind_quicksort_rec(arr, indices, i + 1, right, cmp);
}

template <typename T, typename Compare>
static void ind_quicksort(const T *arr, int *indices, int size, Compare cmp) {
    if (arr == nullptr || indices == nullptr || size <= 0) return;
    for (int i = 0; i < size; i++) indices[i] = i;
    if (size <= 1) return;
    ind_quicksort_rec(arr, indices, 0, size - 1, cmp);
}

template <typename T, typename Compare>
static void ind_heapify(const T *arr, int *indices, int n, int i, Compare cmp) {
    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;
    if (l < n && cmp(arr[indices[l]], arr[indices[largest]]) > 0) largest = l;
    if (r < n && cmp(arr[indices[r]], arr[indices[largest]]) > 0) largest = r;
    if (largest != i) {
        int tmp = indices[i];
        indices[i] = indices[largest];
        indices[largest] = tmp;
        ind_heapify(arr, indices, n, largest, cmp);
    }
}

template <typename T, typename Compare>
static void ind_heapsort(const T *arr, int *indices, int size, Compare cmp) {
    if (arr == nullptr || indices == nullptr || size <= 0) return;
    for (int i = 0; i < size; i++) indices[i] = i;
    if (size <= 1) return;
    for (int i = size / 2 - 1; i >= 0; i--)
        ind_heapify(arr, indices, size, i, cmp);
    for (int i = size - 1; i > 0; i--) {
        int tmp = indices[0];
        indices[0] = indices[i];
        indices[i] = tmp;
        ind_heapify(arr, indices, i, 0, cmp);
    }
}

template <typename T, typename Compare>
static void searchsorted(const T *arr, int size, const T *values, int *out_indices, int num_values, int side_left, const int *sorter, Compare cmp) {
    if (arr == nullptr || values == nullptr || out_indices == nullptr || num_values <= 0) return;
    for (int v_idx = 0; v_idx < num_values; v_idx++) {
        T val = values[v_idx];
        int low = 0;
        int high = size;
        while (low < high) {
            int mid = low + (high - low) / 2;
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
    int size,
    const int *shape,
    const int *strides,
    int rank,
    int is_contiguous,
    unsigned char *dest
) {
    if (src == nullptr || dest == nullptr || size <= 0) return;
    if (is_contiguous) {
        for (int i = 0; i < size; i++) {
            dest[i] = is_nonzero(src[i]) ? 1 : 0;
        }
        return;
    }
    if (shape == nullptr || strides == nullptr || rank <= 0) return;
    std::vector<int> coord_vec;
    int coord_stack[32] = {0};
    int *coord = coord_stack;
    if (rank > 32) {
        coord_vec.assign(rank, 0);
        coord = coord_vec.data();
    }
    int offset = 0;
    for (int i = 0; i < size; i++) {
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

template <typename T, typename Compare>
static void argminmax(
    const T *src,
    const int *stridesSrc,
    int *dest,
    const int *stridesDest,
    const int *shape,
    int rank,
    int axis,
    int is_max,
    int is_contiguous,
    Compare cmp
) {
    if (src == nullptr || dest == nullptr || shape == nullptr || stridesSrc == nullptr || stridesDest == nullptr || rank <= 0) return;
    if (is_contiguous && axis == -1) {
        int best_idx = 0;
        T best_val = src[0];
        for (int i = 1; i < shape[0]; i++) {
            T val = src[i];
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
        dest[0] = best_idx;
        return;
    }
    int dest_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) dest_size *= shape[d];
    }
    std::vector<int> coord_dest_vec, strides_dest_vec, shape_dest_vec;
    int coord_dest_stack[32] = {0};
    int strides_dest_stack[32] = {0};
    int shape_dest_stack[32] = {0};
    int *coord_dest = coord_dest_stack;
    int *strides_dest_clean = strides_dest_stack;
    int *shape_dest_clean = shape_dest_stack;
    if (rank > 32) {
        coord_dest_vec.assign(rank, 0);
        strides_dest_vec.assign(rank, 0);
        shape_dest_vec.assign(rank, 0);
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
    for (int el = 0; el < dest_size; el++) {
        int dest_offset = 0;
        for (int d = 0; d < rank_dest; d++) {
            dest_offset += coord_dest[d] * strides_dest_clean[d];
        }
        int best_idx = 0;
        int base_src_offset = 0;
        int rank_dest_idx = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                base_src_offset += coord_dest[rank_dest_idx] * stridesSrc[d];
                rank_dest_idx++;
            }
        }
        T best_val = src[base_src_offset];
        for (int i = 1; i < shape[axis]; i++) {
            int src_offset = base_src_offset + i * stridesSrc[axis];
            T val = src[src_offset];
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
    const int *stridesSrc,
    int *dest,
    const int *stridesDest,
    const int *shape,
    int rank,
    int axis,
    int is_contiguous
) {
    if (src == nullptr || dest == nullptr || shape == nullptr || stridesSrc == nullptr || stridesDest == nullptr || rank <= 0) return;
    if (is_contiguous && axis == -1) {
        int count = 0;
        for (int i = 0; i < shape[0]; i++) {
            if (is_nonzero(src[i])) count++;
        }
        dest[0] = count;
        return;
    }
    int dest_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) dest_size *= shape[d];
    }
    std::vector<int> coord_dest_vec, strides_dest_vec, shape_dest_vec;
    int coord_dest_stack[32] = {0};
    int strides_dest_stack[32] = {0};
    int shape_dest_stack[32] = {0};
    int *coord_dest = coord_dest_stack;
    int *strides_dest_clean = strides_dest_stack;
    int *shape_dest_clean = shape_dest_stack;
    if (rank > 32) {
        coord_dest_vec.assign(rank, 0);
        strides_dest_vec.assign(rank, 0);
        shape_dest_vec.assign(rank, 0);
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
    for (int el = 0; el < dest_size; el++) {
        int dest_offset = 0;
        for (int d = 0; d < rank_dest; d++) {
            dest_offset += coord_dest[d] * strides_dest_clean[d];
        }
        int count = 0;
        int base_src_offset = 0;
        int rank_dest_idx = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                base_src_offset += coord_dest[rank_dest_idx] * stridesSrc[d];
                rank_dest_idx++;
            }
        }
        for (int i = 0; i < shape[axis]; i++) {
            int src_offset = base_src_offset + i * stridesSrc[axis];
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

extern "C" void native_sort_double(double *array, int size, int kind) {
    if (array == nullptr || size <= 1) return;

    // Segregate NaNs to the end of the array stably
    double *non_nan_end = std::stable_partition(array, array + size, [](double x) {
        return !std::isnan(x);
    });
    int non_nan_size = non_nan_end - array;

    if (non_nan_size <= 1) return;

    if (kind == 0) {
        hwy::VQSort(array, non_nan_size, hwy::SortAscending());
    } else if (kind == 2) {
        std::make_heap(array, array + non_nan_size);
        std::sort_heap(array, array + non_nan_size);
    } else {
        tim_fast_double_tim_sort(array, non_nan_size);
    }
}

extern "C" void native_sort_float(float *array, int size, int kind) {
    if (array == nullptr || size <= 1) return;

    // Segregate NaNs to the end of the array stably
    float *non_nan_end = std::stable_partition(array, array + size, [](float x) {
        return !std::isnan(x);
    });
    int non_nan_size = non_nan_end - array;

    if (non_nan_size <= 1) return;

    if (kind == 0) {
        hwy::VQSort(array, non_nan_size, hwy::SortAscending());
    } else if (kind == 2) {
        std::make_heap(array, array + non_nan_size);
        std::sort_heap(array, array + non_nan_size);
    } else {
        tim_fast_float_tim_sort(array, non_nan_size);
    }
}

extern "C" void native_sort_int64(long long *array, int size, int kind) {
    if (array == nullptr || size <= 1) return;
    if (kind == 0) {
        hwy::VQSort((int64_t *)array, size, hwy::SortAscending());
    } else if (kind == 2) {
        heapsort(array, size, compare_int64_inline);
    } else {
        tim_int64_tim_sort(array, size);
    }
}

extern "C" void native_sort_int32(int *array, int size, int kind) {
    if (array == nullptr || size <= 1) return;
    if (kind == 0) {
        hwy::VQSort((int32_t *)array, size, hwy::SortAscending());
    } else if (kind == 2) {
        heapsort(array, size, compare_int32_inline);
    } else {
        tim_int32_tim_sort(array, size);
    }
}

extern "C" void native_sort_int16(int16_t *array, int size, int kind) {
    if (array == nullptr || size <= 1) return;
    if (kind == 0) {
        hwy::VQSort(array, size, hwy::SortAscending());
    } else if (kind == 2) {
        heapsort(array, size, compare_int16_inline);
    } else {
        tim_int16_tim_sort(array, size);
    }
}

extern "C" void native_sort_uint8(uint8_t *array, int size, int kind) {
    if (array == nullptr || size <= 1) return;
    if (kind == 0) {
        if (size > 32) {
            int counts[256] = {0};
            for (int i = 0; i < size; i++) {
                counts[array[i]]++;
            }
            int idx = 0;
            for (int val = 0; val < 256; val++) {
                int c = counts[val];
                if (c > 0) {
                    memset(array + idx, val, c);
                    idx += c;
                }
            }
        } else {
            std::sort(array, array + size);
        }
    } else if (kind == 2) {
        heapsort(array, size, compare_uint8_inline);
    } else {
        tim_uint8_tim_sort(array, size);
    }
}

extern "C" void native_sort_complex128(double *array, int size, int kind) {
    if (array == nullptr || size <= 1) return;
    if (kind == 0) {
        quicksort((complex128_t *)array, size, compare_complex128_inline);
    } else if (kind == 2) {
        heapsort((complex128_t *)array, size, compare_complex128_inline);
    } else {
        tim_complex128_tim_sort((complex128_t *)array, size);
    }
}

extern "C" void native_sort_complex64(float *array, int size, int kind) {
    if (array == nullptr || size <= 1) return;
    if (kind == 0) {
        quicksort((complex64_t *)array, size, compare_complex64_inline);
    } else if (kind == 2) {
        heapsort((complex64_t *)array, size, compare_complex64_inline);
    } else {
        tim_complex64_tim_sort((complex64_t *)array, size);
    }
}

// ----------------------------------------------------------------------------
// Public Argsort Sorters with Kind Parameter
// ----------------------------------------------------------------------------

extern "C" void native_argsort_double(const double *data, int *indices, int size, int kind) {
    if (data == nullptr || indices == nullptr || size <= 0) return;
    if (size == 1) { indices[0] = 0; return; }
    if (kind == 0) {
        ind_quicksort(data, indices, size, compare_double_inline);
    } else if (kind == 2) {
        ind_heapsort(data, indices, size, compare_double_inline);
    } else {
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }
        global_double_data = data;
        tim_indices_double_tim_sort(indices, size);
        global_double_data = nullptr;
    }
}

extern "C" void native_argsort_float(const float *data, int *indices, int size, int kind) {
    if (data == nullptr || indices == nullptr || size <= 0) return;
    if (size == 1) { indices[0] = 0; return; }
    if (kind == 0) {
        ind_quicksort(data, indices, size, compare_float_inline);
    } else if (kind == 2) {
        ind_heapsort(data, indices, size, compare_float_inline);
    } else {
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }
        global_float_data = data;
        tim_indices_float_tim_sort(indices, size);
        global_float_data = nullptr;
    }
}

extern "C" void native_argsort_int64(const long long *data, int *indices, int size, int kind) {
    if (data == nullptr || indices == nullptr || size <= 0) return;
    if (size == 1) { indices[0] = 0; return; }
    if (kind == 0) {
        ind_quicksort(data, indices, size, compare_int64_inline);
    } else if (kind == 2) {
        ind_heapsort(data, indices, size, compare_int64_inline);
    } else {
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }
        global_int64_data = data;
        tim_indices_int64_tim_sort(indices, size);
        global_int64_data = nullptr;
    }
}

extern "C" void native_argsort_int32(const int *data, int *indices, int size, int kind) {
    if (data == nullptr || indices == nullptr || size <= 0) return;
    if (size == 1) { indices[0] = 0; return; }
    if (kind == 0) {
        ind_quicksort(data, indices, size, compare_int32_inline);
    } else if (kind == 2) {
        ind_heapsort(data, indices, size, compare_int32_inline);
    } else {
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }
        global_int32_data = data;
        tim_indices_int32_tim_sort(indices, size);
        global_int32_data = nullptr;
    }
}

extern "C" void native_argsort_int16(const int16_t *data, int *indices, int size, int kind) {
    if (data == nullptr || indices == nullptr || size <= 0) return;
    if (size == 1) { indices[0] = 0; return; }
    if (kind == 0) {
        ind_quicksort(data, indices, size, compare_int16_inline);
    } else if (kind == 2) {
        ind_heapsort(data, indices, size, compare_int16_inline);
    } else {
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }
        global_int16_data = data;
        tim_indices_int16_tim_sort(indices, size);
        global_int16_data = nullptr;
    }
}

extern "C" void native_argsort_uint8(const uint8_t *data, int *indices, int size, int kind) {
    if (data == nullptr || indices == nullptr || size <= 0) return;
    if (size == 1) { indices[0] = 0; return; }
    if (kind == 0) {
        ind_quicksort(data, indices, size, compare_uint8_inline);
    } else if (kind == 2) {
        ind_heapsort(data, indices, size, compare_uint8_inline);
    } else {
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }
        global_uint8_data = data;
        tim_indices_uint8_tim_sort(indices, size);
        global_uint8_data = nullptr;
    }
}

// ----------------------------------------------------------------------------
// Public Partition Sorters
// ----------------------------------------------------------------------------

extern "C" void native_partition_double(double *array, int size, const int *k_list, int k_size) {
    partition(array, size, k_list, k_size, compare_double_inline);
}

extern "C" void native_partition_float(float *array, int size, const int *k_list, int k_size) {
    partition(array, size, k_list, k_size, compare_float_inline);
}

extern "C" void native_partition_int64(long long *array, int size, const int *k_list, int k_size) {
    partition(array, size, k_list, k_size, compare_int64_inline);
}

extern "C" void native_partition_int32(int *array, int size, const int *k_list, int k_size) {
    partition(array, size, k_list, k_size, compare_int32_inline);
}

extern "C" void native_partition_int16(int16_t *array, int size, const int *k_list, int k_size) {
    partition(array, size, k_list, k_size, compare_int16_inline);
}

extern "C" void native_partition_uint8(uint8_t *array, int size, const int *k_list, int k_size) {
    partition(array, size, k_list, k_size, compare_uint8_inline);
}

extern "C" void native_partition_complex128(double *array, int size, const int *k_list, int k_size) {
    partition((complex128_t *)array, size, k_list, k_size, compare_complex128_inline);
}

extern "C" void native_partition_complex64(float *array, int size, const int *k_list, int k_size) {
    partition((complex64_t *)array, size, k_list, k_size, compare_complex64_inline);
}

// ----------------------------------------------------------------------------
// Public Argpartition Sorters
// ----------------------------------------------------------------------------

extern "C" void native_argpartition_double(const double *data, int *indices, int size, const int *k_list, int k_size) {
    argpartition(data, indices, size, k_list, k_size, compare_double_inline);
}

extern "C" void native_argpartition_float(const float *data, int *indices, int size, const int *k_list, int k_size) {
    argpartition(data, indices, size, k_list, k_size, compare_float_inline);
}

extern "C" void native_argpartition_int64(const long long *data, int *indices, int size, const int *k_list, int k_size) {
    argpartition(data, indices, size, k_list, k_size, compare_int64_inline);
}

extern "C" void native_argpartition_int32(const int *data, int *indices, int size, const int *k_list, int k_size) {
    argpartition(data, indices, size, k_list, k_size, compare_int32_inline);
}

extern "C" void native_argpartition_int16(const int16_t *data, int *indices, int size, const int *k_list, int k_size) {
    argpartition(data, indices, size, k_list, k_size, compare_int16_inline);
}

extern "C" void native_argpartition_uint8(const uint8_t *data, int *indices, int size, const int *k_list, int k_size) {
    argpartition(data, indices, size, k_list, k_size, compare_uint8_inline);
}

extern "C" void native_argpartition_complex128(const double *data, int *indices, int size, const int *k_list, int k_size) {
    argpartition((const complex128_t *)data, indices, size, k_list, k_size, compare_complex128_inline);
}

extern "C" void native_argpartition_complex64(const float *data, int *indices, int size, const int *k_list, int k_size) {
    argpartition((const complex64_t *)data, indices, size, k_list, k_size, compare_complex64_inline);
}

// ----------------------------------------------------------------------------
// Public Searchsorted (Binary Search) functions
// ----------------------------------------------------------------------------

extern "C" void native_searchsorted_double(const double *array, int size, const double *values, int *out_indices, int num_values, int side_left, const int *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, compare_double_inline);
}

extern "C" void native_searchsorted_float(const float *array, int size, const float *values, int *out_indices, int num_values, int side_left, const int *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, compare_float_inline);
}

extern "C" void native_searchsorted_int64(const long long *array, int size, const long long *values, int *out_indices, int num_values, int side_left, const int *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, compare_int64_inline);
}

extern "C" void native_searchsorted_int32(const int *array, int size, const int *values, int *out_indices, int num_values, int side_left, const int *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, compare_int32_inline);
}

extern "C" void native_searchsorted_int16(const int16_t *array, int size, const int16_t *values, int *out_indices, int num_values, int side_left, const int *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, compare_int16_inline);
}

extern "C" void native_searchsorted_uint8(const uint8_t *array, int size, const uint8_t *values, int *out_indices, int num_values, int side_left, const int *sorter) {
    searchsorted(array, size, values, out_indices, num_values, side_left, sorter, compare_uint8_inline);
}

extern "C" void native_searchsorted_complex128(const double *array, int size, const double *values, int *out_indices, int num_values, int side_left, const int *sorter) {
    searchsorted((const complex128_t *)array, size, (const complex128_t *)values, out_indices, num_values, side_left, sorter, compare_complex128_inline);
}

extern "C" void native_searchsorted_complex64(const float *array, int size, const float *values, int *out_indices, int num_values, int side_left, const int *sorter) {
    searchsorted((const complex64_t *)array, size, (const complex64_t *)values, out_indices, num_values, side_left, sorter, compare_complex64_inline);
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
    memcpy(dest, src, n);
}

extern "C" void native_collect_nonzero_coords(
    const unsigned char *cond,
    int total_size,
    const int *shape,
    const int *strides,
    int rank,
    int **out_coords
) {
    if (cond == nullptr || shape == nullptr || strides == nullptr || out_coords == nullptr || total_size <= 0 || rank <= 0) return;
    std::vector<int> coord_vec;
    int coord_stack[32] = {0};
    int *coord = coord_stack;
    if (rank > 32) {
        coord_vec.assign(rank, 0);
        coord = coord_vec.data();
    }
    int offset = 0;
    int write_idx = 0;

    for (int el = 0; el < total_size; el++) {
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
    int total_size,
    const int *shape,
    const int *strides,
    int rank,
    int *out_coords
) {
    if (cond == nullptr || shape == nullptr || strides == nullptr || out_coords == nullptr || total_size <= 0 || rank <= 0) return;
    std::vector<int> coord_vec;
    int coord_stack[32] = {0};
    int *coord = coord_stack;
    if (rank > 32) {
        coord_vec.assign(rank, 0);
        coord = coord_vec.data();
    }
    int offset = 0;
    int write_idx = 0;

    for (int el = 0; el < total_size; el++) {
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

extern "C" void native_to_bool_mask_double(const void *src, int size, const int *shape, const int *strides, int rank, int is_contiguous, unsigned char *dest) {
    to_bool_mask((const double *)src, size, shape, strides, rank, is_contiguous, dest);
}
extern "C" void native_to_bool_mask_float(const void *src, int size, const int *shape, const int *strides, int rank, int is_contiguous, unsigned char *dest) {
    to_bool_mask((const float *)src, size, shape, strides, rank, is_contiguous, dest);
}
extern "C" void native_to_bool_mask_int64(const void *src, int size, const int *shape, const int *strides, int rank, int is_contiguous, unsigned char *dest) {
    to_bool_mask((const long long *)src, size, shape, strides, rank, is_contiguous, dest);
}
extern "C" void native_to_bool_mask_int32(const void *src, int size, const int *shape, const int *strides, int rank, int is_contiguous, unsigned char *dest) {
    to_bool_mask((const int *)src, size, shape, strides, rank, is_contiguous, dest);
}
extern "C" void native_to_bool_mask_complex128(const void *src, int size, const int *shape, const int *strides, int rank, int is_contiguous, unsigned char *dest) {
    to_bool_mask((const complex128_t *)src, size, shape, strides, rank, is_contiguous, dest);
}
extern "C" void native_to_bool_mask_complex64(const void *src, int size, const int *shape, const int *strides, int rank, int is_contiguous, unsigned char *dest) {
    to_bool_mask((const complex64_t *)src, size, shape, strides, rank, is_contiguous, dest);
}
extern "C" void native_to_bool_mask_uint8(const void *src, int size, const int *shape, const int *strides, int rank, int is_contiguous, unsigned char *dest) {
    to_bool_mask((const unsigned char *)src, size, shape, strides, rank, is_contiguous, dest);
}
extern "C" void native_to_bool_mask_int16(const void *src, int size, const int *shape, const int *strides, int rank, int is_contiguous, unsigned char *dest) {
    to_bool_mask((const short *)src, size, shape, strides, rank, is_contiguous, dest);
}

extern "C" void native_argminmax_double(const void *src, const int *stridesSrc, int *dest, const int *stridesDest, const int *shape, int rank, int axis, int is_max, int is_contiguous) {
    argminmax((const double *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_max, is_contiguous, standard_compare<double>);
}
extern "C" void native_argminmax_float(const void *src, const int *stridesSrc, int *dest, const int *stridesDest, const int *shape, int rank, int axis, int is_max, int is_contiguous) {
    argminmax((const float *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_max, is_contiguous, standard_compare<float>);
}
extern "C" void native_argminmax_int64(const void *src, const int *stridesSrc, int *dest, const int *stridesDest, const int *shape, int rank, int axis, int is_max, int is_contiguous) {
    argminmax((const long long *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_max, is_contiguous, standard_compare<long long>);
}
extern "C" void native_argminmax_int32(const void *src, const int *stridesSrc, int *dest, const int *stridesDest, const int *shape, int rank, int axis, int is_max, int is_contiguous) {
    argminmax((const int *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_max, is_contiguous, standard_compare<int>);
}
extern "C" void native_argminmax_uint8(const void *src, const int *stridesSrc, int *dest, const int *stridesDest, const int *shape, int rank, int axis, int is_max, int is_contiguous) {
    argminmax((const unsigned char *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_max, is_contiguous, standard_compare<unsigned char>);
}
extern "C" void native_argminmax_int16(const void *src, const int *stridesSrc, int *dest, const int *stridesDest, const int *shape, int rank, int axis, int is_max, int is_contiguous) {
    argminmax((const short *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_max, is_contiguous, standard_compare<short>);
}

extern "C" void native_count_nonzero_double(const void *src, const int *stridesSrc, int *dest, const int *stridesDest, const int *shape, int rank, int axis, int is_contiguous) {
    count_nonzero((const double *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_contiguous);
}
extern "C" void native_count_nonzero_float(const void *src, const int *stridesSrc, int *dest, const int *stridesDest, const int *shape, int rank, int axis, int is_contiguous) {
    count_nonzero((const float *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_contiguous);
}
extern "C" void native_count_nonzero_int64(const void *src, const int *stridesSrc, int *dest, const int *stridesDest, const int *shape, int rank, int axis, int is_contiguous) {
    count_nonzero((const long long *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_contiguous);
}
extern "C" void native_count_nonzero_int32(const void *src, const int *stridesSrc, int *dest, const int *stridesDest, const int *shape, int rank, int axis, int is_contiguous) {
    count_nonzero((const int *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_contiguous);
}
extern "C" void native_count_nonzero_uint8(const void *src, const int *stridesSrc, int *dest, const int *stridesDest, const int *shape, int rank, int axis, int is_contiguous) {
    count_nonzero((const unsigned char *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_contiguous);
}
extern "C" void native_count_nonzero_int16(const void *src, const int *stridesSrc, int *dest, const int *stridesDest, const int *shape, int rank, int axis, int is_contiguous) {
    count_nonzero((const short *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_contiguous);
}

extern "C" void native_count_nonzero_complex128(const void *src, const int *stridesSrc, int *dest, const int *stridesDest, const int *shape, int rank, int axis, int is_contiguous) {
    count_nonzero((const complex128_t *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_contiguous);
}

extern "C" void native_count_nonzero_complex64(const void *src, const int *stridesSrc, int *dest, const int *stridesDest, const int *shape, int rank, int axis, int is_contiguous) {
    count_nonzero((const complex64_t *)src, stridesSrc, dest, stridesDest, shape, rank, axis, is_contiguous);
}

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

int UnpackMaskImpl(const uint8_t *mask_ptr, int size, int *out_indices) {
    const hn::ScalableTag<int32_t> d;
    using Rebind8 = hn::Rebind<uint8_t, decltype(d)>;
    const Rebind8 d8;
    const int L = hn::Lanes(d);

    int count = 0;
    int j = 0;
    int limit = size - L;
    auto v_base = hn::Iota(d, 0);

    for (; j <= limit; j += L) {
        auto mask_bytes = hn::LoadU(d8, mask_ptr + j);
        auto mask_i32 = hn::PromoteTo(d, mask_bytes);
        auto mask = (mask_i32 != hn::Zero(d));
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
int unpack_mask_c(
    const uint8_t *mask_ptr,
    int size,
    int stride,
    int *out_indices
) {
    if (mask_ptr == nullptr || out_indices == nullptr || size <= 0) return 0;

    if (stride == 1) {
        return hwy::HWY_NAMESPACE::UnpackMaskImpl(mask_ptr, size, out_indices);
    } else {
        int count = 0;
        for (int j = 0; j < size; j++) {
            if (mask_ptr[j * stride] != 0) {
                out_indices[count++] = j;
            }
        }
        return count;
    }
}
}

extern "C" {
int native_count_mask(const uint8_t *mask, int size) {
    if (mask == nullptr || size <= 0) return 0;
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (mask[i] != 0) count++;
    }
    return count;
}

void native_apply_mask(
    int dtype,
    const void *src,
    const uint8_t *mask,
    void *dest,
    int size
) {
    if (src == nullptr || mask == nullptr || dest == nullptr || size <= 0) return;
    
    switch (dtype) {
        case DTYPE_FLOAT32: { // float32
            const float *s = (const float *)src;
            float *d = (float *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_FLOAT64: { // float64
            const double *s = (const double *)src;
            double *d = (double *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_COMPLEX64: { // complex64
            struct C64 { float real, imag; };
            const C64 *s = (const C64 *)src;
            C64 *d = (C64 *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_COMPLEX128: { // complex128
            struct C128 { double real, imag; };
            const C128 *s = (const C128 *)src;
            C128 *d = (C128 *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_INT32: { // int32
            const int32_t *s = (const int32_t *)src;
            int32_t *d = (int32_t *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_INT64: { // int64
            const int64_t *s = (const int64_t *)src;
            int64_t *d = (int64_t *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_UINT8: { // uint8
            const uint8_t *s = (const uint8_t *)src;
            uint8_t *d = (uint8_t *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_INT16: { // int16
            const int16_t *s = (const int16_t *)src;
            int16_t *d = (int16_t *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_UINT64: { // uint64
            const uint64_t *s = (const uint64_t *)src;
            uint64_t *d = (uint64_t *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_UINT32: { // uint32
            const uint32_t *s = (const uint32_t *)src;
            uint32_t *d = (uint32_t *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_UINT16: { // uint16
            const uint16_t *s = (const uint16_t *)src;
            uint16_t *d = (uint16_t *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_INT8: { // int8
            const int8_t *s = (const int8_t *)src;
            int8_t *d = (int8_t *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16: {
            const uint16_t *s = (const uint16_t *)src;
            uint16_t *d = (uint16_t *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case DTYPE_BOOLEAN: { // boolean
            const uint8_t *s = (const uint8_t *)src;
            uint8_t *d = (uint8_t *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
    }
}
}


#include <algorithm>
#include <vector>
#include <cmath>

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

static int unique_double_fast(const double *src, double *dest, int size) {
    if (size <= 0) return 0;
    memcpy(dest, src, size * sizeof(double));
    double *non_nan_end = std::partition(dest, dest + size, [](double x) {
        return !std::isnan(x);
    });
    int non_nan_size = non_nan_end - dest;
    if (non_nan_size > 1) {
        hwy::VQSort(dest, non_nan_size, hwy::SortAscending());
    }
    int write_idx = 0;
    if (non_nan_size > 0) {
        for (int read_idx = 1; read_idx < non_nan_size; read_idx++) {
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

static int unique_float_fast(const float *src, float *dest, int size) {
    if (size <= 0) return 0;
    memcpy(dest, src, size * sizeof(float));
    float *non_nan_end = std::partition(dest, dest + size, [](float x) {
        return !std::isnan(x);
    });
    int non_nan_size = non_nan_end - dest;
    if (non_nan_size > 1) {
        hwy::VQSort(dest, non_nan_size, hwy::SortAscending());
    }
    int write_idx = 0;
    if (non_nan_size > 0) {
        for (int read_idx = 1; read_idx < non_nan_size; read_idx++) {
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

static int unique_int32_fast(const int32_t *src, int32_t *dest, int size) {
    if (size <= 0) return 0;
    memcpy(dest, src, size * sizeof(int32_t));
    if (size > 1) {
        hwy::VQSort(dest, size, hwy::SortAscending());
    }
    int write_idx = 0;
    for (int read_idx = 1; read_idx < size; read_idx++) {
        if (dest[read_idx] != dest[write_idx]) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

static int unique_int64_fast(const int64_t *src, int64_t *dest, int size) {
    if (size <= 0) return 0;
    memcpy(dest, src, size * sizeof(int64_t));
    if (size > 1) {
        hwy::VQSort((int64_t *)dest, size, hwy::SortAscending());
    }
    int write_idx = 0;
    for (int read_idx = 1; read_idx < size; read_idx++) {
        if (dest[read_idx] != dest[write_idx]) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

static int unique_int16_fast(const int16_t *src, int16_t *dest, int size) {
    if (size <= 0) return 0;
    memcpy(dest, src, size * sizeof(int16_t));
    if (size > 1) {
        hwy::VQSort(dest, size, hwy::SortAscending());
    }
    int write_idx = 0;
    for (int read_idx = 1; read_idx < size; read_idx++) {
        if (dest[read_idx] != dest[write_idx]) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

static int unique_uint8_fast(const uint8_t *src, uint8_t *dest, int size) {
    if (size <= 0) return 0;
    bool present[256] = {false};
    for (int i = 0; i < size; i++) {
        present[src[i]] = true;
    }
    int count = 0;
    for (int v = 0; v < 256; v++) {
        if (present[v]) {
            dest[count++] = (uint8_t)v;
        }
    }
    return count;
}

static int unique_complex128_fast(const complex128_t *src, complex128_t *dest, int size) {
    if (size <= 0) return 0;
    memcpy(dest, src, size * sizeof(complex128_t));
    std::sort(dest, dest + size, comp_complex_impl<complex128_t>);
    int write_idx = 0;
    for (int read_idx = 1; read_idx < size; read_idx++) {
        if (!eq_complex_impl(dest[read_idx], dest[write_idx])) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

static int unique_complex64_fast(const complex64_t *src, complex64_t *dest, int size) {
    if (size <= 0) return 0;
    memcpy(dest, src, size * sizeof(complex64_t));
    std::sort(dest, dest + size, comp_complex_impl<complex64_t>);
    int write_idx = 0;
    for (int read_idx = 1; read_idx < size; read_idx++) {
        if (!eq_complex_impl(dest[read_idx], dest[write_idx])) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

template<typename T, typename Comp, typename Eq>
static int unique_template(const T *src, T *dest, int size,
                           int64_t *out_index, int64_t *out_inverse, int64_t *out_counts,
                           Comp comp, Eq eq) {
    if (size <= 0) return 0;
    
    std::vector<int> idx(size);
    for (int i = 0; i < size; i++) idx[i] = i;
    
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        return comp(src[a], src[b]);
    });
    
    int write_idx = 0;
    dest[0] = src[idx[0]];
    if (out_index) out_index[0] = idx[0];
    if (out_inverse) out_inverse[idx[0]] = 0;
    
    int64_t current_count = 1;
    
    for (int read_idx = 1; read_idx < size; read_idx++) {
        if (!eq(src[idx[read_idx]], src[idx[read_idx - 1]])) {
            if (out_counts) out_counts[write_idx] = current_count;
            write_idx++;
            dest[write_idx] = src[idx[read_idx]];
            if (out_index) out_index[write_idx] = idx[read_idx];
            if (out_inverse) out_inverse[idx[read_idx]] = write_idx;
            current_count = 1;
        } else {
            if (out_inverse) out_inverse[idx[read_idx]] = write_idx;
            current_count++;
        }
    }
    if (out_counts) out_counts[write_idx] = current_count;
    
    return write_idx + 1;
}

template<typename T>
static int unique_scalar_fast(const T *src, T *dest, int size) {
    if (size <= 0) return 0;
    memcpy(dest, src, size * sizeof(T));
    std::sort(dest, dest + size);
    int write_idx = 0;
    for (int read_idx = 1; read_idx < size; read_idx++) {
        if (dest[read_idx] != dest[write_idx]) {
            write_idx++;
            dest[write_idx] = dest[read_idx];
        }
    }
    return write_idx + 1;
}

extern "C" {
int ndarray_unique(const void *src, void *dest, int size, int dtype,
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
                return unique_scalar_fast<int64_t>((const int64_t *)src, (int64_t *)dest, size);
            case DTYPE_INT32:
                return unique_scalar_fast<int32_t>((const int32_t *)src, (int32_t *)dest, size);
            case DTYPE_INT16:
                return unique_scalar_fast<int16_t>((const int16_t *)src, (int16_t *)dest, size);
            case DTYPE_INT8:
                return unique_scalar_fast<int8_t>((const int8_t *)src, (int8_t *)dest, size);
            case DTYPE_UINT64:
                return unique_scalar_fast<uint64_t>((const uint64_t *)src, (uint64_t *)dest, size);
            case DTYPE_UINT32:
                return unique_scalar_fast<uint32_t>((const uint32_t *)src, (uint32_t *)dest, size);
            case DTYPE_UINT16:
                return unique_scalar_fast<uint16_t>((const uint16_t *)src, (uint16_t *)dest, size);
            case DTYPE_UINT8:
            case DTYPE_BOOLEAN:
                return unique_scalar_fast<uint8_t>((const uint8_t *)src, (uint8_t *)dest, size);
            case DTYPE_FLOAT16:
            case DTYPE_BFLOAT16:
                return unique_scalar_fast<uint16_t>((const uint16_t *)src, (uint16_t *)dest, size);
            case DTYPE_COMPLEX128:
                return unique_complex128_fast((const complex128_t *)src, (complex128_t *)dest, size);
            case DTYPE_COMPLEX64:
                return unique_complex64_fast((const complex64_t *)src, (complex64_t *)dest, size);
            default:
                return 0;
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
        case DTYPE_BFLOAT16:
            return unique_template<uint16_t>(
                (const uint16_t *)src, (uint16_t *)dest, size,
                out_index, out_inverse, out_counts,
                std::less<uint16_t>(), std::equal_to<uint16_t>()
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
            return 0;
    }
}
}
