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


#include <stdint.h>
#include <stddef.h>
#include "custom_sorting.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    QUANTILE_INVERTED_CDF = 0,
    QUANTILE_AVERAGED_INVERTED_CDF,
    QUANTILE_CLOSEST_OBSERVATION,
    QUANTILE_INTERPOLATED_INVERTED_CDF,
    QUANTILE_HAZEN,
    QUANTILE_WEIBULL,
    QUANTILE_LINEAR,
    QUANTILE_MEDIAN_UNBIASED,
    QUANTILE_NORMAL_UNBIASED,
    QUANTILE_LOWER,
    QUANTILE_HIGHER,
    QUANTILE_MIDPOINT,
    QUANTILE_NEAREST
} QuantileMethod;

#define FOR_EACH_BINARY_OP(X) \
    X(OP_ADD, "add") \
    X(OP_MULTIPLY, "multiply") \
    X(OP_MINIMUM, "minimum") \
    X(OP_MAXIMUM, "maximum") \
    X(OP_FMIN, "fmin") \
    X(OP_FMAX, "fmax") \
    X(OP_LOGADDEXP, "logaddexp") \
    X(OP_LOGADDEXP2, "logaddexp2") \
    X(OP_GCD, "gcd") \
    X(OP_LCM, "lcm") \
    X(OP_BITWISE_AND, "bitwiseAnd") \
    X(OP_BITWISE_OR, "bitwiseOr") \
    X(OP_BITWISE_XOR, "bitwiseXor") \
    X(OP_LOGICAL_AND, "logicalAnd") \
    X(OP_LOGICAL_OR, "logicalOr") \
    X(OP_LOGICAL_XOR, "logicalXor") \
    X(OP_SUBTRACT, "subtract") \
    X(OP_DIVIDE, "divide") \
    X(OP_FLOOR_DIVIDE, "floorDivide") \
    X(OP_REMAINDER, "remainder") \
    X(OP_FMOD, "fmod") \
    X(OP_POWER, "power") \
    X(OP_FLOAT_POWER, "floatPower") \
    X(OP_ARCTAN2, "arctan2") \
    X(OP_HYPOT, "hypot") \
    X(OP_COPYSIGN, "copysign") \
    X(OP_LEFT_SHIFT, "leftShift") \
    X(OP_RIGHT_SHIFT, "rightShift") \
    X(OP_HEAVISIDE, "heaviside") \
    X(OP_EQUAL, "equal") \
    X(OP_NOT_EQUAL, "notEqual") \
    X(OP_GREATER, "greater") \
    X(OP_GREATER_EQUAL, "greaterEqual") \
    X(OP_LESS, "less") \
    X(OP_LESS_EQUAL, "lessEqual")

#define MAKE_BINARY_OP_ENUM(code, name) code,
typedef enum {
    FOR_EACH_BINARY_OP(MAKE_BINARY_OP_ENUM)
    OP_COUNT
} BinaryOpCode;
#undef MAKE_BINARY_OP_ENUM

const char* get_binary_op_name(int index);
int get_binary_op_count(void);
int get_binary_op_enum_val(int index);


/* ============================================================================
 * SECTION 1: CORE COMPLEX TYPE DEFINITIONS
 * ============================================================================
 * Binary-compatible complex number structures matching kiss_fft_cpx
 * and mapped directly to ndarray's custom Complex structures.
 */

typedef struct {
    double r; /** Real component */
    double i; /** Imaginary component */
} cpx_t;

typedef struct {
    float r;  /** Real component */
    float i;  /** Imaginary component */
} cpx_f_t;

/* ============================================================================
 * SECTION 2: FLAT CONTIGUOUS VECTOR MATHEMATICS (ufuncs)
 * ============================================================================
 * Optimized flat vector sweeps operating on sequential same-shape memory arrays.
 * Bypasses coordinate translation and stride checking entirely for peak performance.
 */

void v_add_double(const double *a, const double *b, double *res, int64_t size, const uint8_t *mask);
void v_sub_double(const double *a, const double *b, double *res, int64_t size, const uint8_t *mask);
void v_mul_double(const double *a, const double *b, double *res, int64_t size, const uint8_t *mask);
void v_div_double(const double *a, const double *b, double *res, int64_t size, const uint8_t *mask);

void v_sin_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_sinc_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_cos_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_exp_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_log_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_expm1_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_log1p_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_logaddexp_double(const double *a, const double *b, double *res, int64_t size, const uint8_t *mask);
void v_logaddexp2_double(const double *a, const double *b, double *res, int64_t size, const uint8_t *mask);

double r_sum_double(const double *src, int64_t size);
double r_prod_double(const double *src, int64_t size);
double r_mean_double(const double *src, int64_t size);
double r_var_double(const double *src, int64_t size, int ddof);
double r_std_double(const double *src, int64_t size, int ddof);
void s_sum_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_mean_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_var_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof);
void s_std_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof);

void v_add_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int64_t size, const uint8_t *mask);
void v_sub_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int64_t size, const uint8_t *mask);
void v_mul_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int64_t size, const uint8_t *mask);
void v_div_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int64_t size, const uint8_t *mask);
void v_add_complex64(const cpx_f_t *a, const cpx_f_t *b, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_sub_complex64(const cpx_f_t *a, const cpx_f_t *b, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_mul_complex64(const cpx_f_t *a, const cpx_f_t *b, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_div_complex64(const cpx_f_t *a, const cpx_f_t *b, cpx_f_t *res, int64_t size, const uint8_t *mask);

void v_add_float(const float *a, const float *b, float *res, int64_t size, const uint8_t *mask);
void v_sub_float(const float *a, const float *b, float *res, int64_t size, const uint8_t *mask);
void v_mul_float(const float *a, const float *b, float *res, int64_t size, const uint8_t *mask);
void v_div_float(const float *a, const float *b, float *res, int64_t size, const uint8_t *mask);

void v_sin_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_sinc_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_cos_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_exp_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_log_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_expm1_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_log1p_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_logaddexp_float(const float *a, const float *b, float *res, int64_t size, const uint8_t *mask);
void v_logaddexp2_float(const float *a, const float *b, float *res, int64_t size, const uint8_t *mask);

void v_log2_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_log2_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_log2_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_log2_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);

void v_log10_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_log10_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_log10_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_log10_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);

float r_sum_float(const float *src, int64_t size);
float r_prod_float(const float *src, int64_t size);
float r_mean_float(const float *src, int64_t size);
double r_mean_float_to_double(const float *src, int64_t size);
float r_var_float(const float *src, int64_t size, int ddof);
double r_var_float_to_double(const float *src, int64_t size, int ddof);
float r_std_float(const float *src, int64_t size, int ddof);
double r_std_float_to_double(const float *src, int64_t size, int ddof);
void s_sum_float(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_mean_float(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_mean_float_to_double(const float *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_var_float(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof);
void s_var_float_to_double(const float *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof);
void s_std_float(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof);
void s_std_float_to_double(const float *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof);

void v_sqrt_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_tan_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_abs_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_ceil_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_floor_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_round_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_rint_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_trunc_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_clip_double(const double *src, double *res, double min_val, double max_val, int64_t size, const uint8_t *mask);

void v_sqrt_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_tan_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_abs_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_ceil_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_floor_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_round_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_rint_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_trunc_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_clip_float(const float *src, float *res, float min_val, float max_val, int64_t size, const uint8_t *mask);

void v_asin_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_asin_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_acos_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_acos_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_atan_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_atan_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_atan2_double(const double *y, const double *x, double *res, int64_t size, const uint8_t *mask);
void v_atan2_float(const float *y, const float *x, float *res, int64_t size, const uint8_t *mask);

void v_sinh_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_sinh_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_cosh_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_cosh_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_tanh_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_tanh_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_asinh_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_asinh_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_acosh_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_acosh_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_atanh_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_atanh_float(const float *src, float *res, int64_t size, const uint8_t *mask);

void v_sin_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_sin_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_sinc_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_sinc_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_cos_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_cos_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_tan_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_tan_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);

void v_asin_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_asin_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_acos_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_acos_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_atan_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_atan_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);

void v_atanh_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_atanh_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);

void v_sinh_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_sinh_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_cosh_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_cosh_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_tanh_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_tanh_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_asinh_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_asinh_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_exp_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_exp_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_log_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_log_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_sqrt_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_sqrt_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_acosh_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_acosh_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_abs_complex128(const cpx_t *src, double *res, int64_t size, const uint8_t *mask);
void v_abs_complex64(const cpx_f_t *src, float *res, int64_t size, const uint8_t *mask);
void v_expm1_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_expm1_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_log1p_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_log1p_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);

void v_hypot_complex128(const cpx_t *x1, const cpx_t *x2, double *res, int64_t size, const uint8_t *mask);
void v_hypot_complex64(const cpx_f_t *x1, const cpx_f_t *x2, float *res, int64_t size, const uint8_t *mask);

void v_pow_complex128(const cpx_t *x1, const cpx_t *x2, cpx_t *res, int64_t size, const uint8_t *mask);
void v_pow_complex64(const cpx_f_t *x1, const cpx_f_t *x2, cpx_f_t *res, int64_t size, const uint8_t *mask);


void v_conj_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_conj_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_angle_complex128(const cpx_t *src, double *res, int64_t size, const uint8_t *mask);
void v_angle_complex64(const cpx_f_t *src, float *res, int64_t size, const uint8_t *mask);

void v_square_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_square_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_square_int64(const int64_t *src, int64_t *res, int64_t size, const uint8_t *mask);
void v_square_int32(const int32_t *src, int32_t *res, int64_t size, const uint8_t *mask);
void v_square_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_square_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);

void v_reciprocal_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_reciprocal_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_reciprocal_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_reciprocal_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_reciprocal_int64(const int64_t *src, int64_t *res, int64_t size, const uint8_t *mask);
void v_reciprocal_int32(const int32_t *src, int32_t *res, int64_t size, const uint8_t *mask);
void v_reciprocal_int16(const int16_t *src, int16_t *res, int64_t size, const uint8_t *mask);
void v_reciprocal_uint8(const uint8_t *src, uint8_t *res, int64_t size, const uint8_t *mask);

void v_positive_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void v_positive_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void v_positive_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void v_positive_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void v_positive_int64(const int64_t *src, int64_t *res, int64_t size, const uint8_t *mask);
void v_positive_int32(const int32_t *src, int32_t *res, int64_t size, const uint8_t *mask);
void v_positive_int16(const int16_t *src, int16_t *res, int64_t size, const uint8_t *mask);
void v_positive_uint8(const uint8_t *src, uint8_t *res, int64_t size, const uint8_t *mask);

void v_pow_double(const double *x1, const double *x2, double *res, int64_t size, const uint8_t *mask);
void v_pow_float(const float *x1, const float *x2, float *res, int64_t size, const uint8_t *mask);

void v_floordiv_double(const double *x1, const double *x2, double *res, int64_t size, const uint8_t *mask);
void v_floordiv_float(const float *x1, const float *x2, float *res, int64_t size, const uint8_t *mask);
void v_floordiv_int64(const int64_t *x1, const int64_t *x2, int64_t *res, int64_t size, const uint8_t *mask);
void v_floordiv_int32(const int32_t *x1, const int32_t *x2, int32_t *res, int64_t size, const uint8_t *mask);

void v_remainder_double(const double *x1, const double *x2, double *res, int64_t size, const uint8_t *mask);
void v_remainder_float(const float *x1, const float *x2, float *res, int64_t size, const uint8_t *mask);
void v_remainder_int64(const int64_t *x1, const int64_t *x2, int64_t *res, int64_t size, const uint8_t *mask);
void v_remainder_int32(const int32_t *x1, const int32_t *x2, int32_t *res, int64_t size, const uint8_t *mask);

void v_fmod_double(const double *x1, const double *x2, double *res, int64_t size, const uint8_t *mask);
void v_fmod_float(const float *x1, const float *x2, float *res, int64_t size, const uint8_t *mask);
void v_fmod_int64(const int64_t *x1, const int64_t *x2, int64_t *res, int64_t size, const uint8_t *mask);
void v_fmod_int32(const int32_t *x1, const int32_t *x2, int32_t *res, int64_t size, const uint8_t *mask);

void v_gcd_int64(const int64_t *x1, const int64_t *x2, int64_t *res, int64_t size, const uint8_t *mask);
void v_gcd_int32(const int32_t *x1, const int32_t *x2, int32_t *res, int64_t size, const uint8_t *mask);
void v_lcm_int64(const int64_t *x1, const int64_t *x2, int64_t *res, int64_t size, const uint8_t *mask);
void v_lcm_int32(const int32_t *x1, const int32_t *x2, int32_t *res, int64_t size, const uint8_t *mask);

void v_heaviside_double(const double *x1, const double *x2, double *res, int64_t size, const uint8_t *mask);
void v_heaviside_float(const float *x1, const float *x2, float *res, int64_t size, const uint8_t *mask);
void v_heaviside_int64(const int64_t *x1, const int64_t *x2, int64_t *res, int64_t size, const uint8_t *mask);
void v_heaviside_int32(const int32_t *x1, const int32_t *x2, int32_t *res, int64_t size, const uint8_t *mask);

void v_isnan_double(const double *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_isnan_float(const float *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_isnan_complex128(const cpx_t *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_isnan_complex64(const cpx_f_t *src, uint8_t *res, int64_t size, const uint8_t *mask);

void v_isinf_double(const double *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_isinf_float(const float *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_isinf_complex128(const cpx_t *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_isinf_complex64(const cpx_f_t *src, uint8_t *res, int64_t size, const uint8_t *mask);

void v_isfinite_double(const double *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_isfinite_float(const float *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_isfinite_complex128(const cpx_t *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_isfinite_complex64(const cpx_f_t *src, uint8_t *res, int64_t size, const uint8_t *mask);

void v_copysign_double(const double *x1, const double *x2, double *res, int64_t size, const uint8_t *mask);
void v_copysign_float(const float *x1, const float *x2, float *res, int64_t size, const uint8_t *mask);

/* Vectorized Bitwise Universal Functions */
void v_bitwise_and_int32(const int32_t *a, const int32_t *b, int32_t *res, int64_t size, const uint8_t *mask);
void v_bitwise_and_int64(const int64_t *a, const int64_t *b, int64_t *res, int64_t size, const uint8_t *mask);
void v_bitwise_and_uint8(const uint8_t *a, const uint8_t *b, uint8_t *res, int64_t size, const uint8_t *mask);
void v_bitwise_and_int16(const int16_t *a, const int16_t *b, int16_t *res, int64_t size, const uint8_t *mask);

void v_bitwise_or_int32(const int32_t *a, const int32_t *b, int32_t *res, int64_t size, const uint8_t *mask);
void v_bitwise_or_int64(const int64_t *a, const int64_t *b, int64_t *res, int64_t size, const uint8_t *mask);
void v_bitwise_or_uint8(const uint8_t *a, const uint8_t *b, uint8_t *res, int64_t size, const uint8_t *mask);
void v_bitwise_or_int16(const int16_t *a, const int16_t *b, int16_t *res, int64_t size, const uint8_t *mask);

void v_bitwise_xor_int32(const int32_t *a, const int32_t *b, int32_t *res, int64_t size, const uint8_t *mask);
void v_bitwise_xor_int64(const int64_t *a, const int64_t *b, int64_t *res, int64_t size, const uint8_t *mask);
void v_bitwise_xor_uint8(const uint8_t *a, const uint8_t *b, uint8_t *res, int64_t size, const uint8_t *mask);
void v_bitwise_xor_int16(const int16_t *a, const int16_t *b, int16_t *res, int64_t size, const uint8_t *mask);

void v_left_shift_int32(const int32_t *a, const int32_t *b, int32_t *res, int64_t size, const uint8_t *mask);
void v_left_shift_int64(const int64_t *a, const int64_t *b, int64_t *res, int64_t size, const uint8_t *mask);
void v_left_shift_uint8(const uint8_t *a, const uint8_t *b, uint8_t *res, int64_t size, const uint8_t *mask);
void v_left_shift_int16(const int16_t *a, const int16_t *b, int16_t *res, int64_t size, const uint8_t *mask);

void v_right_shift_int32(const int32_t *a, const int32_t *b, int32_t *res, int64_t size, const uint8_t *mask);
void v_right_shift_int64(const int64_t *a, const int64_t *b, int64_t *res, int64_t size, const uint8_t *mask);
void v_right_shift_uint8(const uint8_t *a, const uint8_t *b, uint8_t *res, int64_t size, const uint8_t *mask);
void v_right_shift_int16(const int16_t *a, const int16_t *b, int16_t *res, int64_t size, const uint8_t *mask);

void v_invert_int32(const int32_t *src, int32_t *res, int64_t size, const uint8_t *mask);
void v_invert_int64(const int64_t *src, int64_t *res, int64_t size, const uint8_t *mask);
void v_invert_uint8(const uint8_t *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_invert_int16(const int16_t *src, int16_t *res, int64_t size, const uint8_t *mask);


/* ============================================================================
 * SECTION 3: STRIDED MULTI-DIMENSIONAL ODOMETER WALKS (ufuncs)
 * ============================================================================
 * High-speed ND strided broadcasting and view loops executing on unmanaged C.
 * Resolves sliced views, transposes, and dimension broadcasting recursively.
 */

void s_add_double(const double *a, const int64_t *stridesA,
                  const double *b, const int64_t *stridesB,
                  double *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank);
void s_sub_double(const double *a, const int64_t *stridesA,
                  const double *b, const int64_t *stridesB,
                  double *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank);
void s_mul_double(const double *a, const int64_t *stridesA,
                  const double *b, const int64_t *stridesB,
                  double *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank);
void s_div_double(const double *a, const int64_t *stridesA,
                  const double *b, const int64_t *stridesB,
                  double *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank);

void s_add_complex(const cpx_t *a, const int64_t *stridesA,
                  const cpx_t *b, const int64_t *stridesB,
                  cpx_t *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank);
void s_sub_complex(const cpx_t *a, const int64_t *stridesA,
                  const cpx_t *b, const int64_t *stridesB,
                  cpx_t *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank);
void s_mul_complex(const cpx_t *a, const int64_t *stridesA,
                  const cpx_t *b, const int64_t *stridesB,
                  cpx_t *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank);
void s_div_complex(const cpx_t *a, const int64_t *stridesA,
                  const cpx_t *b, const int64_t *stridesB,
                  cpx_t *res, const int64_t *stridesRes,
                  const int64_t *shape, int rank);

void s_where_double(const unsigned char *cond, const int64_t *stridesCond,
                    const double *x, const int64_t *stridesX,
                    const double *y, const int64_t *stridesY,
                    double *res, const int64_t *stridesRes,
                    const int64_t *shape, int rank);

void s_where_float(const unsigned char *cond, const int64_t *stridesCond,
                   const float *x, const int64_t *stridesX,
                   const float *y, const int64_t *stridesY,
                   float *res, const int64_t *stridesRes,
                   const int64_t *shape, int rank);

void s_where_int64(const unsigned char *cond, const int64_t *stridesCond,
                   const int64_t *x, const int64_t *stridesX,
                   const int64_t *y, const int64_t *stridesY,
                   int64_t *res, const int64_t *stridesRes,
                   const int64_t *shape, int rank);

void s_where_int32(const unsigned char *cond, const int64_t *stridesCond,
                   const int32_t *x, const int64_t *stridesX,
                   const int32_t *y, const int64_t *stridesY,
                   int32_t *res, const int64_t *stridesRes,
                   const int64_t *shape, int rank);

void s_where_uint8(const unsigned char *cond, const int64_t *stridesCond,
                   const uint8_t *x, const int64_t *stridesX,
                   const uint8_t *y, const int64_t *stridesY,
                   uint8_t *res, const int64_t *stridesRes,
                   const int64_t *shape, int rank);

void s_where_int16(const unsigned char *cond, const int64_t *stridesCond,
                   const int16_t *x, const int64_t *stridesX,
                   const int16_t *y, const int64_t *stridesY,
                   int16_t *res, const int64_t *stridesRes,
                   const int64_t *shape, int rank);

void s_where_complex128(const unsigned char *cond, const int64_t *stridesCond,
                        const cpx_t *x, const int64_t *stridesX,
                        const cpx_t *y, const int64_t *stridesY,
                        cpx_t *res, const int64_t *stridesRes,
                        const int64_t *shape, int rank);

void s_where_complex64(const unsigned char *cond, const int64_t *stridesCond,
                       const cpx_f_t *x, const int64_t *stridesX,
                       const cpx_f_t *y, const int64_t *stridesY,
                       cpx_f_t *res, const int64_t *stridesRes,
                       const int64_t *shape, int rank);

void s_sin_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_sin_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_sinc_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_sinc_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_cos_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_cos_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_tan_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_tan_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_asin_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_asin_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_acos_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_acos_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_atan_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_atan_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_atanh_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_atanh_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_sinh_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_sinh_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_cosh_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_cosh_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_tanh_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_tanh_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_asinh_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_asinh_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_exp_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_exp_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_log_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_log_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_sqrt_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_sqrt_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_acosh_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_acosh_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_abs_complex128(const cpx_t *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_abs_complex64(const cpx_f_t *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_expm1_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_expm1_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_log1p_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_log1p_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_hypot_complex128(const cpx_t *x1, const int64_t *stridesX1, const cpx_t *x2, const int64_t *stridesX2, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_hypot_complex64(const cpx_f_t *x1, const int64_t *stridesX1, const cpx_f_t *x2, const int64_t *stridesX2, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_pow_complex128(const cpx_t *x1, const int64_t *stridesX1, const cpx_t *x2, const int64_t *stridesX2, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_pow_complex64(const cpx_f_t *x1, const int64_t *stridesX1, const cpx_f_t *x2, const int64_t *stridesX2, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_conj_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_conj_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_angle_complex128(const cpx_t *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_angle_complex64(const cpx_f_t *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_cumsum_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumsum_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumsum_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumsum_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumsum_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumsum_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumsum_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumsum_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);

void s_cumprod_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumprod_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumprod_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumprod_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumprod_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumprod_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumprod_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumprod_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);

void s_cummin_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cummin_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cummin_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cummin_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cummin_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cummin_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);

void s_cummax_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cummax_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cummax_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cummax_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cummax_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cummax_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);

void s_diff_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_diff_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_diff_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_diff_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_diff_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_diff_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_unwrap_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis, double discont);
void s_unwrap_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis, float discont);

void s_sin_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_sin_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_sinc_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_sinc_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_cos_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_cos_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_asin_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_asin_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_acos_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_acos_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_atan_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_atan_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_atan2_double(const double *y, const int64_t *stridesY, const double *x, const int64_t *stridesX, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_atan2_float(const float *y, const int64_t *stridesY, const float *x, const int64_t *stridesX, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_tan_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_tan_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_exp_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_exp_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_log_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_log_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_expm1_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_expm1_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_log1p_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_log1p_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_rint_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_rint_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_trunc_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_trunc_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_logaddexp_double(const double *a, const int64_t *stridesA, const double *b, const int64_t *stridesB, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_logaddexp_float(const float *a, const int64_t *stridesA, const float *b, const int64_t *stridesB, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_logaddexp2_double(const double *a, const int64_t *stridesA, const double *b, const int64_t *stridesB, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_logaddexp2_float(const float *a, const int64_t *stridesA, const float *b, const int64_t *stridesB, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_log2_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_log2_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_log2_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_log2_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_log10_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_log10_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_log10_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_log10_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_sinh_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_sinh_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_cosh_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_cosh_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_tanh_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_tanh_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_asinh_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_asinh_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_acosh_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_acosh_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_atanh_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_atanh_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_square_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_square_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_square_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_square_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_square_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_square_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_reciprocal_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_reciprocal_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_reciprocal_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_reciprocal_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_reciprocal_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_reciprocal_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_reciprocal_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_reciprocal_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_positive_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_positive_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_positive_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_positive_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_positive_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_positive_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_positive_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_positive_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_pow_double(const double *x1, const int64_t *stridesX1, const double *x2, const int64_t *stridesX2, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_pow_float(const float *x1, const int64_t *stridesX1, const float *x2, const int64_t *stridesX2, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_floordiv_double(const double *x1, const int64_t *stridesX1, const double *x2, const int64_t *stridesX2, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_floordiv_float(const float *x1, const int64_t *stridesX1, const float *x2, const int64_t *stridesX2, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_floordiv_int64(const int64_t *x1, const int64_t *stridesX1, const int64_t *x2, const int64_t *stridesX2, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_floordiv_int32(const int32_t *x1, const int64_t *stridesX1, const int32_t *x2, const int64_t *stridesX2, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_remainder_double(const double *x1, const int64_t *stridesX1, const double *x2, const int64_t *stridesX2, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_remainder_float(const float *x1, const int64_t *stridesX1, const float *x2, const int64_t *stridesX2, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_remainder_int64(const int64_t *x1, const int64_t *stridesX1, const int64_t *x2, const int64_t *stridesX2, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_remainder_int32(const int32_t *x1, const int64_t *stridesX1, const int32_t *x2, const int64_t *stridesX2, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_fmod_double(const double *x1, const int64_t *stridesX1, const double *x2, const int64_t *stridesX2, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_fmod_float(const float *x1, const int64_t *stridesX1, const float *x2, const int64_t *stridesX2, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_fmod_int64(const int64_t *x1, const int64_t *stridesX1, const int64_t *x2, const int64_t *stridesX2, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_fmod_int32(const int32_t *x1, const int64_t *stridesX1, const int32_t *x2, const int64_t *stridesX2, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_gcd_int64(const int64_t *x1, const int64_t *stridesX1, const int64_t *x2, const int64_t *stridesX2, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_gcd_int32(const int32_t *x1, const int64_t *stridesX1, const int32_t *x2, const int64_t *stridesX2, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_lcm_int64(const int64_t *x1, const int64_t *stridesX1, const int64_t *x2, const int64_t *stridesX2, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_lcm_int32(const int32_t *x1, const int64_t *stridesX1, const int32_t *x2, const int64_t *stridesX2, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_heaviside_double(const double *x1, const int64_t *stridesX1, const double *x2, const int64_t *stridesX2, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_heaviside_float(const float *x1, const int64_t *stridesX1, const float *x2, const int64_t *stridesX2, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_heaviside_int64(const int64_t *x1, const int64_t *stridesX1, const int64_t *x2, const int64_t *stridesX2, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_heaviside_int32(const int32_t *x1, const int64_t *stridesX1, const int32_t *x2, const int64_t *stridesX2, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_isnan_double(const double *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_isnan_float(const float *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_isnan_complex128(const cpx_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_isnan_complex64(const cpx_f_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_isinf_double(const double *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_isinf_float(const float *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_isinf_complex128(const cpx_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_isinf_complex64(const cpx_f_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_isfinite_double(const double *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_isfinite_float(const float *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_isfinite_complex128(const cpx_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_isfinite_complex64(const cpx_f_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_copysign_double(const double *x1, const int64_t *stridesX1, const double *x2, const int64_t *stridesX2, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_copysign_float(const float *x1, const int64_t *stridesX1, const float *x2, const int64_t *stridesX2, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

/* Strided Vectorized Bitwise Universal Functions */
void s_bitwise_and_int32(const int32_t *a, const int64_t *stridesA, const int32_t *b, const int64_t *stridesB, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_bitwise_and_int64(const int64_t *a, const int64_t *stridesA, const int64_t *b, const int64_t *stridesB, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_bitwise_and_uint8(const uint8_t *a, const int64_t *stridesA, const uint8_t *b, const int64_t *stridesB, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_bitwise_and_int16(const int16_t *a, const int64_t *stridesA, const int16_t *b, const int64_t *stridesB, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_bitwise_or_int32(const int32_t *a, const int64_t *stridesA, const int32_t *b, const int64_t *stridesB, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_bitwise_or_int64(const int64_t *a, const int64_t *stridesA, const int64_t *b, const int64_t *stridesB, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_bitwise_or_uint8(const uint8_t *a, const int64_t *stridesA, const uint8_t *b, const int64_t *stridesB, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_bitwise_or_int16(const int16_t *a, const int64_t *stridesA, const int16_t *b, const int64_t *stridesB, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_bitwise_xor_int32(const int32_t *a, const int64_t *stridesA, const int32_t *b, const int64_t *stridesB, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_bitwise_xor_int64(const int64_t *a, const int64_t *stridesA, const int64_t *b, const int64_t *stridesB, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_bitwise_xor_uint8(const uint8_t *a, const int64_t *stridesA, const uint8_t *b, const int64_t *stridesB, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_bitwise_xor_int16(const int16_t *a, const int64_t *stridesA, const int16_t *b, const int64_t *stridesB, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_left_shift_int32(const int32_t *a, const int64_t *stridesA, const int32_t *b, const int64_t *stridesB, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_left_shift_int64(const int64_t *a, const int64_t *stridesA, const int64_t *b, const int64_t *stridesB, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_left_shift_uint8(const uint8_t *a, const int64_t *stridesA, const uint8_t *b, const int64_t *stridesB, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_left_shift_int16(const int16_t *a, const int64_t *stridesA, const int16_t *b, const int64_t *stridesB, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_right_shift_int32(const int32_t *a, const int64_t *stridesA, const int32_t *b, const int64_t *stridesB, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_right_shift_int64(const int64_t *a, const int64_t *stridesA, const int64_t *b, const int64_t *stridesB, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_right_shift_uint8(const uint8_t *a, const int64_t *stridesA, const uint8_t *b, const int64_t *stridesB, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_right_shift_int16(const int16_t *a, const int64_t *stridesA, const int16_t *b, const int64_t *stridesB, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void s_invert_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_invert_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_invert_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_invert_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);


/* ============================================================================
 * SECTION 4: FLAT FLATTENING & STRIDED COPYING KERNELS
 * ============================================================================
 * Used inside ravel() and flatten() for fast layout conversions.
 */

void s_flatten_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *shape, int rank);
void s_flatten_float(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *shape, int rank);
void s_flatten_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *shape, int rank);
void s_flatten_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *shape, int rank);
void s_flatten_complex128(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *shape, int rank);
void s_flatten_complex64(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *shape, int rank);
void s_flatten_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *shape, int rank);
void s_flatten_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *shape, int rank);


/* ============================================================================
 * SECTION 5: HASHING KERNELS
 * ============================================================================
 * High-speed hashing algorithms for quick comparison validation.
 */

uint32_t s_hash_double(const double *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous);
uint32_t s_hash_float(const float *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous);
uint32_t s_hash_int64(const int64_t *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous);
uint32_t s_hash_int32(const int32_t *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous);
uint32_t s_hash_int16(const int16_t *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous);
uint32_t s_hash_uint8(const uint8_t *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous);
uint32_t s_hash_complex128(const double *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous);
uint32_t s_hash_complex64(const float *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous);
uint32_t s_hash_boolean(const uint8_t *a, const int64_t *strides, const int64_t *shape, int rank, int is_contiguous);

/* ============================================================================
 * SECTION 6: RANDOM GENERATION KERNELS
 * ============================================================================
 * High-performance random statistical simulators.
 */

void v_normal_double(double *res, int64_t size, double loc, double scale, unsigned long long seed);
void v_normal_float(float *res, int64_t size, float loc, float scale, unsigned long long seed);
void v_uniform_double(double *res, int64_t size, unsigned long long seed);
void v_uniform_float(float *res, int64_t size, unsigned long long seed);

void v_randint_int64(int64_t *res, int64_t size, int64_t low, int64_t high, unsigned long long seed);
void v_randint_int32(int32_t *res, int64_t size, int32_t low, int32_t high, unsigned long long seed);
void v_randint_int16(int16_t *res, int64_t size, int low, int high, unsigned long long seed);
void v_randint_uint8(uint8_t *res, int64_t size, int low, int high, unsigned long long seed);

void v_fill_double(double *res, double value, int64_t size);
void v_fill_float(float *res, float value, int64_t size);
void v_fill_int64(int64_t *res, int64_t value, int64_t size);
void v_fill_int32(int32_t *res, int32_t value, int64_t size);
void v_fill_int16(int16_t *res, int16_t value, int64_t size);
void v_fill_uint8(uint8_t *res, uint8_t value, int64_t size);
void v_fill_complex128(cpx_t *res, double valR, double valI, int64_t size);
void v_fill_complex64(cpx_f_t *res, float valR, float valI, int64_t size);
void v_fill_boolean(uint8_t *res, uint8_t value, int64_t size);

void s_fill_double(double *res, const int64_t *strides, const int64_t *shape, int rank, double value);
void s_fill_float(float *res, const int64_t *strides, const int64_t *shape, int rank, float value);
void s_fill_int64(int64_t *res, const int64_t *strides, const int64_t *shape, int rank, int64_t value);
void s_fill_int32(int32_t *res, const int64_t *strides, const int64_t *shape, int rank, int32_t value);
void s_fill_int16(int16_t *res, const int64_t *strides, const int64_t *shape, int rank, int16_t value);
void s_fill_uint8(uint8_t *res, const int64_t *strides, const int64_t *shape, int rank, uint8_t value);
void s_fill_complex128(cpx_t *res, const int64_t *strides, const int64_t *shape, int rank, double valR, double valI);
void s_fill_complex64(cpx_f_t *res, const int64_t *strides, const int64_t *shape, int rank, float valR, float valI);
void s_fill_boolean(uint8_t *res, const int64_t *strides, const int64_t *shape, int rank, uint8_t value);

void v_linspace_double(double *res, double start, double step, int64_t size);
void v_linspace_float(float *res, float start, float step, int64_t size);
void v_linspace_complex128(cpx_t *res, double startR, double startI, double stepR, double stepI, int64_t size);
void v_linspace_complex64(cpx_f_t *res, float startR, float startI, float stepR, float stepI, int64_t size);
void v_linspace_int64(int64_t *res, double start, double step, int64_t size);
void v_linspace_int32(int32_t *res, double start, double step, int64_t size);
void v_linspace_int16(int16_t *res, double start, double step, int64_t size);
void v_linspace_uint8(uint8_t *res, double start, double step, int64_t size);
void v_logspace_double(double *res, double start, double step, double base, int64_t size);
void v_logspace_float(float *res, float start, float step, float base, int64_t size);
void v_logspace_complex128(cpx_t *res, double startR, double startI, double stepR, double stepI, double baseR, double baseI, int64_t size);
void v_logspace_complex64(cpx_f_t *res, float startR, float startI, float stepR, float stepI, float baseR, float baseI, int64_t size);
void v_geomspace_double(double *res, double logStart, double step, double sign, int64_t size);
void v_geomspace_float(float *res, float logStart, float step, float sign, int64_t size);
void v_geomspace_complex128(cpx_t *res, double logStartR, double logStartI, double stepR, double stepI, int64_t size);
void v_geomspace_complex64(cpx_f_t *res, float logStartR, float logStartI, float stepR, float stepI, int64_t size);

void v_secure_uniform_double(double *res, int64_t size);
void v_secure_uniform_float(float *res, int64_t size);
void v_secure_randint_int64(int64_t *res, int64_t size, int64_t low, int64_t high);
void v_secure_randint_int32(int32_t *res, int64_t size, int32_t low, int32_t high);
void v_secure_randint_int16(int16_t *res, int64_t size, int low, int high);
void v_secure_randint_uint8(uint8_t *res, int64_t size, int low, int high);

void v_secure_normal_double(double *res, int64_t size, double loc, double scale);
void v_secure_normal_float(float *res, int64_t size, float loc, float scale);
void v_poisson_int64(int64_t *res, int64_t size, double lam, unsigned long long seed);
void v_poisson_int32(int32_t *res, int64_t size, double lam, unsigned long long seed);
void v_binomial_int64(int64_t *res, int64_t size, int n, double p, unsigned long long seed);
void v_binomial_int32(int32_t *res, int64_t size, int n, double p, unsigned long long seed);

/* ============================================================================
 * SECTION 7: TRIANGULAR MATRIX EXTRACTORS
 * ============================================================================
 */

void v_tril_double(const double *src, double *res, int64_t batch_count, int64_t rows, int64_t cols, int64_t k);
void v_tril_float(const float *src, float *res, int64_t batch_count, int64_t rows, int64_t cols, int64_t k);
void v_triu_double(const double *src, double *res, int64_t batch_count, int64_t rows, int64_t cols, int64_t k);
void v_triu_float(const float *src, float *res, int64_t batch_count, int64_t rows, int64_t cols, int64_t k);

/* ============================================================================
 * SECTION 8: AUTO-GENERATED CROSS-TYPE BINARY MATH FFI HELPERS
 * ============================================================================
 */

#define GENERATE_COMMUTATIVE_COMBINATIONS(OP, MACRO) \
  MACRO(OP, double, double, double, double, double, double) \
  MACRO(OP, double, float, double, double, float, double) \
  MACRO(OP, double, int64, double, double, int64_t, double) \
  MACRO(OP, double, int32, double, double, int32_t, double) \
  MACRO(OP, double, uint8, double, double, uint8_t, double) \
  MACRO(OP, double, int16, double, double, int16_t, double) \
  MACRO(OP, double, cpx, cpx, double, cpx_t, cpx_t) \
  MACRO(OP, double, cpx64, cpx, double, cpx_f_t, cpx_t) \
  MACRO(OP, float, float, float, float, float, float) \
  MACRO(OP, float, int64, float, float, int64_t, float) \
  MACRO(OP, float, int32, float, float, int32_t, float) \
  MACRO(OP, float, uint8, float, float, uint8_t, float) \
  MACRO(OP, float, int16, float, float, int16_t, float) \
  MACRO(OP, float, cpx, cpx, float, cpx_t, cpx_t) \
  MACRO(OP, float, cpx64, cpx64, float, cpx_f_t, cpx_f_t) \
  MACRO(OP, int64, int64, int64, int64_t, int64_t, int64_t) \
  MACRO(OP, int64, int32, int64, int64_t, int32_t, int64_t) \
  MACRO(OP, int64, uint8, int64, int64_t, uint8_t, int64_t) \
  MACRO(OP, int64, int16, int64, int64_t, int16_t, int64_t) \
  MACRO(OP, int64, cpx, cpx, int64_t, cpx_t, cpx_t) \
  MACRO(OP, int64, cpx64, cpx64, int64_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, int32, int32, int32, int32_t, int32_t, int32_t) \
  MACRO(OP, int32, uint8, int32, int32_t, uint8_t, int32_t) \
  MACRO(OP, int32, int16, int32, int32_t, int16_t, int32_t) \
  MACRO(OP, int32, cpx, cpx, int32_t, cpx_t, cpx_t) \
  MACRO(OP, int32, cpx64, cpx64, int32_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, uint8, uint8, uint8, uint8_t, uint8_t, uint8_t) \
  MACRO(OP, uint8, int16, int16, uint8_t, int16_t, int16_t) \
  MACRO(OP, uint8, cpx, cpx, uint8_t, cpx_t, cpx_t) \
  MACRO(OP, uint8, cpx64, cpx64, uint8_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, int16, int16, int16, int16_t, int16_t, int16_t) \
  MACRO(OP, int16, cpx, cpx, int16_t, cpx_t, cpx_t) \
  MACRO(OP, int16, cpx64, cpx64, int16_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, cpx, cpx, cpx, cpx_t, cpx_t, cpx_t) \
  MACRO(OP, cpx, cpx64, cpx, cpx_t, cpx_f_t, cpx_t) \
  MACRO(OP, cpx64, cpx64, cpx64, cpx_f_t, cpx_f_t, cpx_f_t)

#define GENERATE_OP_COMBINATIONS(OP, MACRO) \
  MACRO(OP, double, double, double, double, double, double) \
  MACRO(OP, double, float, double, double, float, double) \
  MACRO(OP, double, int64, double, double, int64_t, double) \
  MACRO(OP, double, int32, double, double, int32_t, double) \
  MACRO(OP, double, uint8, double, double, uint8_t, double) \
  MACRO(OP, double, int16, double, double, int16_t, double) \
  MACRO(OP, double, cpx, cpx, double, cpx_t, cpx_t) \
  MACRO(OP, float, double, double, float, double, double) \
  MACRO(OP, float, float, float, float, float, float) \
  MACRO(OP, float, int64, float, float, int64_t, float) \
  MACRO(OP, float, int32, float, float, int32_t, float) \
  MACRO(OP, float, uint8, float, float, uint8_t, float) \
  MACRO(OP, float, int16, float, float, int16_t, float) \
  MACRO(OP, float, cpx, cpx, float, cpx_t, cpx_t) \
  MACRO(OP, int64, double, double, int64_t, double, double) \
  MACRO(OP, int64, float, float, int64_t, float, float) \
  MACRO(OP, int64, int64, int64, int64_t, int64_t, int64_t) \
  MACRO(OP, int64, int32, int64, int64_t, int32_t, int64_t) \
  MACRO(OP, int64, uint8, int64, int64_t, uint8_t, int64_t) \
  MACRO(OP, int64, int16, int64, int64_t, int16_t, int64_t) \
  MACRO(OP, int64, cpx, cpx, int64_t, cpx_t, cpx_t) \
  MACRO(OP, int32, double, double, int32_t, double, double) \
  MACRO(OP, int32, float, float, int32_t, float, float) \
  MACRO(OP, int32, int64, int64, int32_t, int64_t, int64_t) \
  MACRO(OP, int32, int32, int32, int32_t, int32_t, int32_t) \
  MACRO(OP, int32, uint8, int32, int32_t, uint8_t, int32_t) \
  MACRO(OP, int32, int16, int32, int32_t, int16_t, int32_t) \
  MACRO(OP, int32, cpx, cpx, int32_t, cpx_t, cpx_t) \
  MACRO(OP, uint8, double, double, uint8_t, double, double) \
  MACRO(OP, uint8, float, float, uint8_t, float, float) \
  MACRO(OP, uint8, int64, int64, uint8_t, int64_t, int64_t) \
  MACRO(OP, uint8, int32, int32, uint8_t, int32_t, int32_t) \
  MACRO(OP, uint8, uint8, uint8, uint8_t, uint8_t, uint8_t) \
  MACRO(OP, uint8, int16, int16, uint8_t, int16_t, int16_t) \
  MACRO(OP, uint8, cpx, cpx, uint8_t, cpx_t, cpx_t) \
  MACRO(OP, int16, double, double, int16_t, double, double) \
  MACRO(OP, int16, float, float, int16_t, float, float) \
  MACRO(OP, int16, int64, int64, int16_t, int64_t, int64_t) \
  MACRO(OP, int16, int32, int32, int16_t, int32_t, int32_t) \
  MACRO(OP, int16, uint8, int16, int16_t, uint8_t, int16_t) \
  MACRO(OP, int16, int16, int16, int16_t, int16_t, int16_t) \
  MACRO(OP, int16, cpx, cpx, int16_t, cpx_t, cpx_t) \
  MACRO(OP, cpx, double, cpx, cpx_t, double, cpx_t) \
  MACRO(OP, cpx, float, cpx, cpx_t, float, cpx_t) \
  MACRO(OP, cpx, int64, cpx, cpx_t, int64_t, cpx_t) \
  MACRO(OP, cpx, int32, cpx, cpx_t, int32_t, cpx_t) \
  MACRO(OP, cpx, uint8, cpx, cpx_t, uint8_t, cpx_t) \
  MACRO(OP, cpx, int16, cpx, cpx_t, int16_t, cpx_t) \
  MACRO(OP, cpx, cpx, cpx, cpx_t, cpx_t, cpx_t) \
  MACRO(OP, cpx64, double, cpx, cpx_f_t, double, cpx_t) \
  MACRO(OP, cpx64, float, cpx64, cpx_f_t, float, cpx_f_t) \
  MACRO(OP, cpx64, int64, cpx64, cpx_f_t, int64_t, cpx_f_t) \
  MACRO(OP, cpx64, int32, cpx64, cpx_f_t, int32_t, cpx_f_t) \
  MACRO(OP, cpx64, uint8, cpx64, cpx_f_t, uint8_t, cpx_f_t) \
  MACRO(OP, cpx64, int16, cpx64, cpx_f_t, int16_t, cpx_f_t) \
  MACRO(OP, cpx64, cpx, cpx, cpx_f_t, cpx_t, cpx_t) \
  MACRO(OP, double, cpx64, cpx, double, cpx_f_t, cpx_t) \
  MACRO(OP, float, cpx64, cpx64, float, cpx_f_t, cpx_f_t) \
  MACRO(OP, int64, cpx64, cpx64, int64_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, int32, cpx64, cpx64, int32_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, uint8, cpx64, cpx64, uint8_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, int16, cpx64, cpx64, int16_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, cpx, cpx64, cpx, cpx_t, cpx_f_t, cpx_t) \
  MACRO(OP, cpx64, cpx64, cpx64, cpx_f_t, cpx_f_t, cpx_f_t)

#define GENERATE_DIV_COMBINATIONS(OP, MACRO) \
  MACRO(OP, double, double, double, double, double, double) \
  MACRO(OP, double, float, double, double, float, double) \
  MACRO(OP, double, int64, double, double, int64_t, double) \
  MACRO(OP, double, int32, double, double, int32_t, double) \
  MACRO(OP, double, uint8, double, double, uint8_t, double) \
  MACRO(OP, double, int16, double, double, int16_t, double) \
  MACRO(OP, double, cpx, cpx, double, cpx_t, cpx_t) \
  MACRO(OP, float, double, double, float, double, double) \
  MACRO(OP, float, float, float, float, float, float) \
  MACRO(OP, float, int64, float, float, int64_t, float) \
  MACRO(OP, float, int32, float, float, int32_t, float) \
  MACRO(OP, float, uint8, float, float, uint8_t, float) \
  MACRO(OP, float, int16, float, float, int16_t, float) \
  MACRO(OP, float, cpx, cpx, float, cpx_t, cpx_t) \
  MACRO(OP, int64, double, double, int64_t, double, double) \
  MACRO(OP, int64, float, float, int64_t, float, float) \
  MACRO(OP, int64, int64, double, int64_t, int64_t, double) \
  MACRO(OP, int64, int32, double, int64_t, int32_t, double) \
  MACRO(OP, int64, uint8, double, int64_t, uint8_t, double) \
  MACRO(OP, int64, int16, double, int64_t, int16_t, double) \
  MACRO(OP, int64, cpx, cpx, int64_t, cpx_t, cpx_t) \
  MACRO(OP, int32, double, double, int32_t, double, double) \
  MACRO(OP, int32, float, float, int32_t, float, float) \
  MACRO(OP, int32, int64, double, int32_t, int64_t, double) \
  MACRO(OP, int32, int32, double, int32_t, int32_t, double) \
  MACRO(OP, int32, uint8, double, int32_t, uint8_t, double) \
  MACRO(OP, int32, int16, double, int32_t, int16_t, double) \
  MACRO(OP, int32, cpx, cpx, int32_t, cpx_t, cpx_t) \
  MACRO(OP, uint8, double, double, uint8_t, double, double) \
  MACRO(OP, uint8, float, float, uint8_t, float, float) \
  MACRO(OP, uint8, int64, double, uint8_t, int64_t, double) \
  MACRO(OP, uint8, int32, double, uint8_t, int32_t, double) \
  MACRO(OP, uint8, uint8, double, uint8_t, uint8_t, double) \
  MACRO(OP, uint8, int16, double, uint8_t, int16_t, double) \
  MACRO(OP, uint8, cpx, cpx, uint8_t, cpx_t, cpx_t) \
  MACRO(OP, int16, double, double, int16_t, double, double) \
  MACRO(OP, int16, float, float, int16_t, float, float) \
  MACRO(OP, int16, int64, double, int16_t, int64_t, double) \
  MACRO(OP, int16, int32, double, int16_t, int32_t, double) \
  MACRO(OP, int16, uint8, double, int16_t, uint8_t, double) \
  MACRO(OP, int16, int16, double, int16_t, int16_t, double) \
  MACRO(OP, int16, cpx, cpx, int16_t, cpx_t, cpx_t) \
  MACRO(OP, cpx, double, cpx, cpx_t, double, cpx_t) \
  MACRO(OP, cpx, float, cpx, cpx_t, float, cpx_t) \
  MACRO(OP, cpx, int64, cpx, cpx_t, int64_t, cpx_t) \
  MACRO(OP, cpx, int32, cpx, cpx_t, int32_t, cpx_t) \
  MACRO(OP, cpx, uint8, cpx, cpx_t, uint8_t, cpx_t) \
  MACRO(OP, cpx, int16, cpx, cpx_t, int16_t, cpx_t) \
  MACRO(OP, cpx, cpx, cpx, cpx_t, cpx_t, cpx_t) \
  MACRO(OP, cpx64, double, cpx, cpx_f_t, double, cpx_t) \
  MACRO(OP, cpx64, float, cpx64, cpx_f_t, float, cpx_f_t) \
  MACRO(OP, cpx64, int64, cpx64, cpx_f_t, int64_t, cpx_f_t) \
  MACRO(OP, cpx64, int32, cpx64, cpx_f_t, int32_t, cpx_f_t) \
  MACRO(OP, cpx64, uint8, cpx64, cpx_f_t, uint8_t, cpx_f_t) \
  MACRO(OP, cpx64, int16, cpx64, cpx_f_t, int16_t, cpx_f_t) \
  MACRO(OP, cpx64, cpx, cpx, cpx_f_t, cpx_t, cpx_t) \
  MACRO(OP, double, cpx64, cpx, double, cpx_f_t, cpx_t) \
  MACRO(OP, float, cpx64, cpx64, float, cpx_f_t, cpx_f_t) \
  MACRO(OP, int64, cpx64, cpx64, int64_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, int32, cpx64, cpx64, int32_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, uint8, cpx64, cpx64, uint8_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, int16, cpx64, cpx64, int16_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, cpx, cpx64, cpx, cpx_t, cpx_f_t, cpx_t) \
  MACRO(OP, cpx64, cpx64, cpx64, cpx_f_t, cpx_f_t, cpx_f_t)

#define DECLARE_FFI_HELPER(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr) \
  void v_##OP##_##Ta_tok##_##Tb_tok##_##Tr_tok(const Ta *a, const Tb *b, Tr *res, int64_t size, const uint8_t *mask); \
  void s_##OP##_##Ta_tok##_##Tb_tok##_##Tr_tok(const Ta *a, const int64_t *stridesA, \
                                               const Tb *b, const int64_t *stridesB, \
                                               Tr *res, const int64_t *stridesRes, \
                                               const int64_t *shape, int rank, \
                                               const uint8_t *mask);

GENERATE_COMMUTATIVE_COMBINATIONS(add, DECLARE_FFI_HELPER)
GENERATE_OP_COMBINATIONS(sub, DECLARE_FFI_HELPER)
GENERATE_COMMUTATIVE_COMBINATIONS(mul, DECLARE_FFI_HELPER)
GENERATE_DIV_COMBINATIONS(div, DECLARE_FFI_HELPER)

/* ============================================================================
 * SECTION 9: BUFFERED CONVERTER KERNELS (CASTING)
 * ============================================================================
 */
void cast_uint8_to_double(const uint8_t *src, double *dst, int64_t size);
void cast_int16_to_double(const int16_t *src, double *dst, int64_t size);
void cast_double_to_uint8(const double *src, uint8_t *dst, int64_t size);
void cast_double_to_int16(const double *src, int16_t *dst, int64_t size);

void s_cast_uint8_to_double(const uint8_t *src, const int64_t *stridesSrc, double *dst, const int64_t *stridesDst, const int64_t *shape, int rank);
void s_cast_int16_to_double(const int16_t *src, const int64_t *stridesSrc, double *dst, const int64_t *stridesDst, const int64_t *shape, int rank);
void s_cast_double_to_uint8(const double *src, const int64_t *stridesSrc, uint8_t *dst, const int64_t *stridesDst, const int64_t *shape, int rank);
void s_cast_double_to_int16(const double *src, const int64_t *stridesSrc, int16_t *dst, const int64_t *stridesDst, const int64_t *shape, int rank);

/* Logical & Casting-to-Boolean operations */
void v_to_bool_double(const double *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_to_bool_float(const float *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_to_bool_int64(const int64_t *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_to_bool_int32(const int32_t *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_to_bool_uint8(const uint8_t *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_to_bool_int16(const int16_t *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_to_bool_complex128(const cpx_t *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_to_bool_complex64(const cpx_f_t *src, uint8_t *res, int64_t size, const uint8_t *mask);

void s_to_bool_double(const double *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_to_bool_float(const float *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_to_bool_int64(const int64_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_to_bool_int32(const int32_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_to_bool_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_to_bool_int16(const int16_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_to_bool_complex128(const cpx_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_to_bool_complex64(const cpx_f_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void v_logical_and(const uint8_t *a, const uint8_t *b, uint8_t *res, int64_t size, const uint8_t *mask);
void v_logical_or(const uint8_t *a, const uint8_t *b, uint8_t *res, int64_t size, const uint8_t *mask);
void v_logical_xor(const uint8_t *a, const uint8_t *b, uint8_t *res, int64_t size, const uint8_t *mask);
void v_logical_not(const uint8_t *src, uint8_t *res, int64_t size, const uint8_t *mask);

void s_logical_and(const uint8_t *a, const int64_t *stridesA, const uint8_t *b, const int64_t *stridesB, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_logical_or(const uint8_t *a, const int64_t *stridesA, const uint8_t *b, const int64_t *stridesB, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_logical_xor(const uint8_t *a, const int64_t *stridesA, const uint8_t *b, const int64_t *stridesB, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_logical_not(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

/* Optimized native advanced indexing recursive copy kernel */
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
);


/* Kronecker Product */
void native_kron_2d(
    int dtype,
    const void *a, int64_t strideA_0, int64_t strideA_1, int64_t m, int64_t n,
    const void *b, int64_t strideB_0, int64_t strideB_1, int64_t p, int64_t q,
    void *res, int64_t strideRes_0, int64_t strideRes_1
);
void native_kron_nd(
    int dtype,
    const void *a, const int64_t *stridesA, const int64_t *shapeA,
    const void *b, const int64_t *stridesB, const int64_t *shapeB,
    void *res, const int64_t *stridesRes, const int64_t *shapeRes,
    int rank
);
void s_kron_double(const double *a, const int64_t *stridesA, const int64_t *shapeA,
                   const double *b, const int64_t *stridesB, const int64_t *shapeB,
                   double *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank);
void s_kron_float(const float *a, const int64_t *stridesA, const int64_t *shapeA,
                  const float *b, const int64_t *stridesB, const int64_t *shapeB,
                  float *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank);
void s_kron_int64(const int64_t *a, const int64_t *stridesA, const int64_t *shapeA,
                  const int64_t *b, const int64_t *stridesB, const int64_t *shapeB,
                  int64_t *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank);
void s_kron_int32(const int32_t *a, const int64_t *stridesA, const int64_t *shapeA,
                  const int32_t *b, const int64_t *stridesB, const int64_t *shapeB,
                  int32_t *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank);
void s_kron_uint8(const uint8_t *a, const int64_t *stridesA, const int64_t *shapeA,
                  const uint8_t *b, const int64_t *stridesB, const int64_t *shapeB,
                  uint8_t *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank);
void s_kron_int16(const int16_t *a, const int64_t *stridesA, const int64_t *shapeA,
                  const int16_t *b, const int64_t *stridesB, const int64_t *shapeB,
                  int16_t *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank);
void s_kron_complex128(const cpx_t *a, const int64_t *stridesA, const int64_t *shapeA,
                       const cpx_t *b, const int64_t *stridesB, const int64_t *shapeB,
                       cpx_t *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank);
void s_kron_complex64(const cpx_f_t *a, const int64_t *stridesA, const int64_t *shapeA,
                      const cpx_f_t *b, const int64_t *stridesB, const int64_t *shapeB,
                      cpx_f_t *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank);
void s_kron_boolean(const uint8_t *a, const int64_t *stridesA, const int64_t *shapeA,
                    const uint8_t *b, const int64_t *stridesB, const int64_t *shapeB,
                    uint8_t *res, const int64_t *stridesRes, const int64_t *shapeRes, int rank);

/* Vector Outer Product */
void s_outer_double(const double *a, int64_t strideA, int64_t sizeA,
                    const double *b, int64_t strideB, int64_t sizeB,
                    double *res, int64_t strideRowRes, int64_t strideColRes);
void s_outer_float(const float *a, int64_t strideA, int64_t sizeA,
                   const float *b, int64_t strideB, int64_t sizeB,
                   float *res, int64_t strideRowRes, int64_t strideColRes);
void s_outer_int64(const int64_t *a, int64_t strideA, int64_t sizeA,
                   const int64_t *b, int64_t strideB, int64_t sizeB,
                   int64_t *res, int64_t strideRowRes, int64_t strideColRes);
void s_outer_int32(const int32_t *a, int64_t strideA, int64_t sizeA,
                   const int32_t *b, int64_t strideB, int64_t sizeB,
                   int32_t *res, int64_t strideRowRes, int64_t strideColRes);
void s_outer_uint8(const uint8_t *a, int64_t strideA, int64_t sizeA,
                   const uint8_t *b, int64_t strideB, int64_t sizeB,
                   uint8_t *res, int64_t strideRowRes, int64_t strideColRes);
void s_outer_int16(const int16_t *a, int64_t strideA, int64_t sizeA,
                   const int16_t *b, int64_t strideB, int64_t sizeB,
                   int16_t *res, int64_t strideRowRes, int64_t strideColRes);
void s_outer_complex128(const cpx_t *a, int64_t strideA, int64_t sizeA,
                        const cpx_t *b, int64_t strideB, int64_t sizeB,
                        cpx_t *res, int64_t strideRowRes, int64_t strideColRes);
void s_outer_complex64(const cpx_f_t *a, int64_t strideA, int64_t sizeA,
                       const cpx_f_t *b, int64_t strideB, int64_t sizeB,
                       cpx_f_t *res, int64_t strideRowRes, int64_t strideColRes);
void s_outer_boolean(const uint8_t *a, int64_t strideA, int64_t sizeA,
                     const uint8_t *b, int64_t strideB, int64_t sizeB,
                     uint8_t *res, int64_t strideRowRes, int64_t strideColRes);

/* Vector Cross Product */
void s_cross_3d_double(const double *a, int64_t strideA, const double *b, int64_t strideB, double *res, int64_t strideRes);
void s_cross_2d_double(const double *a, int64_t strideA, const double *b, int64_t strideB, double *res);
void s_cross_3d_float(const float *a, int64_t strideA, const float *b, int64_t strideB, float *res, int64_t strideRes);
void s_cross_2d_float(const float *a, int64_t strideA, const float *b, int64_t strideB, float *res);
void s_cross_3d_int64(const int64_t *a, int64_t strideA, const int64_t *b, int64_t strideB, int64_t *res, int64_t strideRes);
void s_cross_2d_int64(const int64_t *a, int64_t strideA, const int64_t *b, int64_t strideB, int64_t *res);
void s_cross_3d_int32(const int32_t *a, int64_t strideA, const int32_t *b, int64_t strideB, int32_t *res, int64_t strideRes);
void s_cross_2d_int32(const int32_t *a, int64_t strideA, const int32_t *b, int64_t strideB, int32_t *res);
void s_cross_3d_uint8(const uint8_t *a, int64_t strideA, const uint8_t *b, int64_t strideB, uint8_t *res, int64_t strideRes);
void s_cross_2d_uint8(const uint8_t *a, int64_t strideA, const uint8_t *b, int64_t strideB, uint8_t *res);
void s_cross_3d_int16(const int16_t *a, int64_t strideA, const int16_t *b, int64_t strideB, int16_t *res, int64_t strideRes);
void s_cross_2d_int16(const int16_t *a, int64_t strideA, const int16_t *b, int64_t strideB, int16_t *res);
void s_cross_3d_complex128(const cpx_t *a, int64_t strideA, const cpx_t *b, int64_t strideB, cpx_t *res, int64_t strideRes);
void s_cross_2d_complex128(const cpx_t *a, int64_t strideA, const cpx_t *b, int64_t strideB, cpx_t *res);
void s_cross_3d_complex64(const cpx_f_t *a, int64_t strideA, const cpx_f_t *b, int64_t strideB, cpx_f_t *res, int64_t strideRes);
void s_cross_2d_complex64(const cpx_f_t *a, int64_t strideA, const cpx_f_t *b, int64_t strideB, cpx_f_t *res);
void s_cross_3d_boolean(const uint8_t *a, int64_t strideA, const uint8_t *b, int64_t strideB, uint8_t *res, int64_t strideRes);
void s_cross_2d_boolean(const uint8_t *a, int64_t strideA, const uint8_t *b, int64_t strideB, uint8_t *res);

/* Vector Norm Reductions */
double r_norm_l1_double(const double *src, int64_t stride, int64_t size);
double r_norm_l2_double(const double *src, int64_t stride, int64_t size);
double r_norm_lp_double(const double *src, int64_t stride, int64_t size, double p);
double r_norm_inf_double(const double *src, int64_t stride, int64_t size);
double r_norm_neg_inf_double(const double *src, int64_t stride, int64_t size);

float r_norm_l1_float(const float *src, int64_t stride, int64_t size);
float r_norm_l2_float(const float *src, int64_t stride, int64_t size);
float r_norm_lp_float(const float *src, int64_t stride, int64_t size, float p);
float r_norm_inf_float(const float *src, int64_t stride, int64_t size);
float r_norm_neg_inf_float(const float *src, int64_t stride, int64_t size);

double r_norm_l1_complex128(const cpx_t *src, int64_t stride, int64_t size);
double r_norm_l2_complex128(const cpx_t *src, int64_t stride, int64_t size);
double r_norm_lp_complex128(const cpx_t *src, int64_t stride, int64_t size, double p);
double r_norm_inf_complex128(const cpx_t *src, int64_t stride, int64_t size);
double r_norm_neg_inf_complex128(const cpx_t *src, int64_t stride, int64_t size);

float r_norm_l1_complex64(const cpx_f_t *src, int64_t stride, int64_t size);
float r_norm_l2_complex64(const cpx_f_t *src, int64_t stride, int64_t size);
float r_norm_lp_complex64(const cpx_f_t *src, int64_t stride, int64_t size, float p);
float r_norm_inf_complex64(const cpx_f_t *src, int64_t stride, int64_t size);
float r_norm_neg_inf_complex64(const cpx_f_t *src, int64_t stride, int64_t size);

/* Window Functions */
void v_hanning_double(double *res, int64_t M);
void v_hanning_float(float *res, int64_t M);
void v_hamming_double(double *res, int64_t M);
void v_hamming_float(float *res, int64_t M);

/* Strided Ternary Clip Functions */
void s_clip_double(const double *a, const int64_t *stridesA, const double *min_val, const int64_t *stridesMin, const double *max_val, const int64_t *stridesMax, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_clip_float(const float *a, const int64_t *stridesA, const float *min_val, const int64_t *stridesMin, const float *max_val, const int64_t *stridesMax, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_clip_int64(const int64_t *a, const int64_t *stridesA, const int64_t *min_val, const int64_t *stridesMin, const int64_t *max_val, const int64_t *stridesMax, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_clip_int32(const int32_t *a, const int64_t *stridesA, const int32_t *min_val, const int64_t *stridesMin, const int32_t *max_val, const int64_t *stridesMax, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_clip_uint8(const uint8_t *a, const int64_t *stridesA, const uint8_t *min_val, const int64_t *stridesMin, const uint8_t *max_val, const int64_t *stridesMax, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_clip_int16(const int16_t *a, const int64_t *stridesA, const int16_t *min_val, const int64_t *stridesMin, const int16_t *max_val, const int64_t *stridesMax, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

/* Calculus Solvers: Trapz and Gradient */
void s_trapz_double(const double *y, const int64_t *stridesY, const double *x, int64_t strideX, double dx, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_trapz_float(const float *y, const int64_t *stridesY, const float *x, int64_t strideX, float dx, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_trapz_complex128(const cpx_t *y, const int64_t *stridesY, const double *x, int64_t strideX, double dx, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_trapz_complex64(const cpx_f_t *y, const int64_t *stridesY, const float *x, int64_t strideX, float dx, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_trapz_complex128_all(const cpx_t *y, const int64_t *stridesY, const cpx_t *x, int64_t strideX, const cpx_t *dx, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_trapz_complex64_all(const cpx_f_t *y, const int64_t *stridesY, const cpx_f_t *x, int64_t strideX, const cpx_f_t *dx, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);

void s_gradient_double(const double *src, const int64_t *stridesSrc, const double *x, int64_t strideX, double dx, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis, int edge_order);
void s_gradient_float(const float *src, const int64_t *stridesSrc, const float *x, int64_t strideX, float dx, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis, int edge_order);
void s_gradient_complex128(const cpx_t *src, const int64_t *stridesSrc, const double *x, int64_t strideX, double dx, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis, int edge_order);
void s_gradient_complex64(const cpx_f_t *src, const int64_t *stridesSrc, const float *x, int64_t strideX, float dx, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis, int edge_order);
void s_gradient_complex128_all(const cpx_t *src, const int64_t *stridesSrc, const cpx_t *x, int64_t strideX, const cpx_t *dx, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis, int edge_order);
void s_gradient_complex64_all(const cpx_f_t *src, const int64_t *stridesSrc, const cpx_f_t *x, int64_t strideX, const cpx_f_t *dx, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis, int edge_order);

/* linspace Grid Intrinsic Kernels */
void s_linspace_grid_double(const double *start, const int64_t *stridesStart, const double *stop, const int64_t *stridesStop, double *res, const int64_t *stridesRes, double *step, const int64_t *stridesStep, const int64_t *shape, int rank, int axis, int64_t numSamples, int endpoint);
void s_linspace_grid_float(const float *start, const int64_t *stridesStart, const float *stop, const int64_t *stridesStop, float *res, const int64_t *stridesRes, float *step, const int64_t *stridesStep, const int64_t *shape, int rank, int axis, int64_t numSamples, int endpoint);
void s_linspace_grid_complex128(const cpx_t *start, const int64_t *stridesStart, const cpx_t *stop, const int64_t *stridesStop, cpx_t *res, const int64_t *stridesRes, cpx_t *step, const int64_t *stridesStep, const int64_t *shape, int rank, int axis, int64_t numSamples, int endpoint);
void s_linspace_grid_complex64(const cpx_f_t *start, const int64_t *stridesStart, const cpx_f_t *stop, const int64_t *stridesStop, cpx_f_t *res, const int64_t *stridesRes, cpx_f_t *step, const int64_t *stridesStep, const int64_t *shape, int rank, int axis, int64_t numSamples, int endpoint);
void s_linspace_grid_int64(const int64_t *start, const int64_t *stridesStart, const int64_t *stop, const int64_t *stridesStop, int64_t *res, const int64_t *stridesRes, int64_t *step, const int64_t *stridesStep, const int64_t *shape, int rank, int axis, int64_t numSamples, int endpoint);
void s_linspace_grid_int32(const int32_t *start, const int64_t *stridesStart, const int32_t *stop, const int64_t *stridesStop, int32_t *res, const int64_t *stridesRes, int32_t *step, const int64_t *stridesStep, const int64_t *shape, int rank, int axis, int64_t numSamples, int endpoint);
void s_linspace_grid_int16(const int16_t *start, const int64_t *stridesStart, const int16_t *stop, const int64_t *stridesStop, int16_t *res, const int64_t *stridesRes, int16_t *step, const int64_t *stridesStep, const int64_t *shape, int rank, int axis, int64_t numSamples, int endpoint);
void s_linspace_grid_uint8(const uint8_t *start, const int64_t *stridesStart, const uint8_t *stop, const int64_t *stridesStop, uint8_t *res, const int64_t *stridesRes, uint8_t *step, const int64_t *stridesStep, const int64_t *shape, int rank, int axis, int64_t numSamples, int endpoint);

int get_and_reset_division_error(void);

/* Decoupled LAPACK-Dispatched Determinants */
void s_det_double(const double *a, const int64_t *stridesA, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, double *aCopy, int *ipiv, int (*lapack_getrf)(int, int, int, void *, int, int *));
void s_det_float(const float *a, const int64_t *stridesA, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, float *aCopy, int *ipiv, int (*lapack_getrf)(int, int, int, void *, int, int *));
void s_det_complex_double(const cpx_t *a, const int64_t *stridesA, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, cpx_t *aCopy, int *ipiv, int (*lapack_getrf)(int, int, int, void *, int, int *));
void s_det_complex_float(const cpx_f_t *a, const int64_t *stridesA, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, cpx_f_t *aCopy, int *ipiv, int (*lapack_getrf)(int, int, int, void *, int, int *));

/* Optimized FFI Eigenvector assembly intrinsics */

/**
 * Assembles complex eigenvectors for double precision.
 * Combines real and imaginary parts from LAPACK dgeev output into complex128 format,
 * handling conjugate pairs for complex eigenvalues.
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
);

/**
 * Assembles complex eigenvectors for single precision.
 * Combines real and imaginary parts from LAPACK sgeev output into complex64 format,
 * handling conjugate pairs for complex eigenvalues.
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
);

/* Optimized integer matrix multiplication intrinsics */

/**
 * Performs matrix multiplication for 64-bit signed integers.
 * Computes res = a * b using a strided row-major implementation.
 */
void matmul_int64(
    int64_t *res,
    int64_t strideResRow,
    int64_t strideResCol,
    const int64_t *a,
    int64_t strideARow,
    int64_t strideACol,
    const int64_t *b,
    int64_t strideBRow,
    int64_t strideBCol,
    int64_t m,
    int64_t n,
    int64_t k
);

/**
 * Performs matrix multiplication for 32-bit signed integers.
 * Computes res = a * b using a strided row-major implementation.
 */
void matmul_int32(
    int32_t *res,
    int64_t strideResRow,
    int64_t strideResCol,
    const int32_t *a,
    int64_t strideARow,
    int64_t strideACol,
    const int32_t *b,
    int64_t strideBRow,
    int64_t strideBCol,
    int64_t m,
    int64_t n,
    int64_t k
);

/**
 * Performs matrix multiplication for 16-bit signed integers.
 * Computes res = a * b using a strided row-major implementation.
 */
void matmul_int16(
    int16_t *res,
    int64_t strideResRow,
    int64_t strideResCol,
    const int16_t *a,
    int64_t strideARow,
    int64_t strideACol,
    const int16_t *b,
    int64_t strideBRow,
    int64_t strideBCol,
    int64_t m,
    int64_t n,
    int64_t k
);

/**
 * Performs matrix multiplication for 8-bit unsigned integers.
 * Computes res = a * b using a strided row-major implementation.
 */
void matmul_uint8(
    uint8_t *res,
    int64_t strideResRow,
    int64_t strideResCol,
    const uint8_t *a,
    int64_t strideARow,
    int64_t strideACol,
    const uint8_t *b,
    int64_t strideBRow,
    int64_t strideBCol,
    int64_t m,
    int64_t n,
    int64_t k
);

uint8_t v_any_less_than_zero_int32(const int32_t *arr, int64_t size);
uint8_t v_any_less_than_zero_int64(const int64_t *arr, int64_t size);
uint8_t v_any_equal_to_zero_int32(const int32_t *arr, int64_t size);
uint8_t v_any_equal_to_zero_int64(const int64_t *arr, int64_t size);

void s_cast_generic(
    const void *src_ptr, const int64_t *stridesSrc, int dtypeSrc,
    void *dest_ptr, int dtypeDst,
    const int64_t *shape, int rank
);

void v_extract_upper_triangular(
    const void *src_ptr,
    void *dest_ptr,
    int64_t k,
    int64_t n,
    int dtype
);

void v_zero_upper_triangular(
    void *ptr,
    int64_t n,
    int dtype
);

/* ============================================================================
 * SECTION 9: PADDING KERNELS
 * ============================================================================
 */

void pad_axis_double(
    const double *src, const int64_t *shapeSrc, const int64_t *stridesSrc,
    double *dest, const int64_t *shapeDest,
    int rank, int axis,
    int64_t padBefore, int64_t padAfter,
    int mode,
    double constantBefore, double constantAfter,
    double endBefore, double endAfter,
    int64_t statLengthBefore, int64_t statLengthAfter
);

void pad_axis_float(
    const float *src, const int64_t *shapeSrc, const int64_t *stridesSrc,
    float *dest, const int64_t *shapeDest,
    int rank, int axis,
    int64_t padBefore, int64_t padAfter,
    int mode,
    float constantBefore, float constantAfter,
    float endBefore, float endAfter,
    int64_t statLengthBefore, int64_t statLengthAfter
);

void pad_axis_int64(
    const int64_t *src, const int64_t *shapeSrc, const int64_t *stridesSrc,
    int64_t *dest, const int64_t *shapeDest,
    int rank, int axis,
    int64_t padBefore, int64_t padAfter,
    int mode,
    int64_t constantBefore, int64_t constantAfter,
    int64_t endBefore, int64_t endAfter,
    int64_t statLengthBefore, int64_t statLengthAfter
);

void pad_axis_uint64(
    const uint64_t *src, const int64_t *shapeSrc, const int64_t *stridesSrc,
    uint64_t *dest, const int64_t *shapeDest,
    int rank, int axis,
    int64_t padBefore, int64_t padAfter,
    int mode,
    uint64_t constantBefore, uint64_t constantAfter,
    uint64_t endBefore, uint64_t endAfter,
    int64_t statLengthBefore, int64_t statLengthAfter
);

void pad_axis_int32(
    const int32_t *src, const int64_t *shapeSrc, const int64_t *stridesSrc,
    int32_t *dest, const int64_t *shapeDest,
    int rank, int axis,
    int64_t padBefore, int64_t padAfter,
    int mode,
    int32_t constantBefore, int32_t constantAfter,
    int32_t endBefore, int32_t endAfter,
    int64_t statLengthBefore, int64_t statLengthAfter
);

void pad_axis_int16(
    const int16_t *src, const int64_t *shapeSrc, const int64_t *stridesSrc,
    int16_t *dest, const int64_t *shapeDest,
    int rank, int axis,
    int64_t padBefore, int64_t padAfter,
    int mode,
    int16_t constantBefore, int16_t constantAfter,
    int16_t endBefore, int16_t endAfter,
    int64_t statLengthBefore, int64_t statLengthAfter
);

void pad_axis_uint8(
    const uint8_t *src, const int64_t *shapeSrc, const int64_t *stridesSrc,
    uint8_t *dest, const int64_t *shapeDest,
    int rank, int axis,
    int64_t padBefore, int64_t padAfter,
    int mode,
    uint8_t constantBefore, uint8_t constantAfter,
    uint8_t endBefore, uint8_t endAfter,
    int64_t statLengthBefore, int64_t statLengthAfter
);

void pad_axis_complex128(
    const cpx_t *src, const int64_t *shapeSrc, const int64_t *stridesSrc,
    cpx_t *dest, const int64_t *shapeDest,
    int rank, int axis,
    int64_t padBefore, int64_t padAfter,
    int mode,
    const cpx_t *constantBefore, const cpx_t *constantAfter,
    const cpx_t *endBefore, const cpx_t *endAfter,
    int64_t statLengthBefore, int64_t statLengthAfter
);

void pad_axis_complex64(
    const cpx_f_t *src, const int64_t *shapeSrc, const int64_t *stridesSrc,
    cpx_f_t *dest, const int64_t *shapeDest,
    int rank, int axis,
    int64_t padBefore, int64_t padAfter,
    int mode,
    const cpx_f_t *constantBefore, const cpx_f_t *constantAfter,
    const cpx_f_t *endBefore, const cpx_f_t *endAfter,
    int64_t statLengthBefore, int64_t statLengthAfter
);

/* ============================================================================
 * SECTION 10: SET OPERATIONS KERNELS
 * ============================================================================
 */

int64_t ndarray_unique(const void *src, void *dest, int64_t size, int dtype, int64_t *out_index, int64_t *out_inverse, int64_t *out_counts);
int64_t ndarray_intersect1d(const void *ar1, int64_t size1, const void *ar2, int64_t size2, void *dest, int dtype);
int64_t ndarray_setdiff1d(const void *ar1, int64_t size1, const void *ar2, int64_t size2, void *dest, int dtype);
int64_t ndarray_setxor1d(const void *ar1, int64_t size1, const void *ar2, int64_t size2, void *dest, int dtype);
int64_t ndarray_union1d(const void *ar1, int64_t size1, const void *ar2, int64_t size2, void *dest, int dtype);
void ndarray_isin(const void *ar1, int64_t size1, const void *ar2, int64_t size2, uint8_t *dest, int dtype, int invert);
/* Median global reductions (contiguous) */
double r_median_double(const double *src, int64_t size);
float r_median_float(const float *src, int64_t size);
int64_t r_median_int64(const int64_t *src, int64_t size);
int32_t r_median_int32(const int32_t *src, int64_t size);
int16_t r_median_int16(const int16_t *src, int64_t size);
uint8_t r_median_uint8(const uint8_t *src, int64_t size);
void r_median_complex128(const cpx_t *src, int64_t size, cpx_t *out);
void r_median_complex64(const cpx_f_t *src, int64_t size, cpx_f_t *out);

/* Median axis reductions (strided) */
void s_median_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_median_float(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_median_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_median_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_median_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_median_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_median_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_median_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);

/* Quantile global reductions (contiguous) */
double r_quantile_double(const double *src, int64_t size, double q, int method);
double r_quantile_float(const float *src, int64_t size, double q, int method);
double r_quantile_int64(const int64_t *src, int64_t size, double q, int method);
double r_quantile_int32(const int32_t *src, int64_t size, double q, int method);
double r_quantile_int16(const int16_t *src, int64_t size, double q, int method);
double r_quantile_uint8(const uint8_t *src, int64_t size, double q, int method);

/* Quantile axis reductions (strided) */
void s_quantile_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, double q, int method);
void s_quantile_float(const float *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, double q, int method);
void s_quantile_int64(const int64_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, double q, int method);
void s_quantile_int32(const int32_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, double q, int method);
void s_quantile_int16(const int16_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, double q, int method);
void s_quantile_uint8(const uint8_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, double q, int method);

/* ============================================================================
 * SECTION 11: INTERPOLATION KERNELS
 * ============================================================================
 */

int is_strictly_increasing_double(const double *arr, int64_t size, int64_t stride);

void v_interp_double(const double *x, int64_t x_size,
                     const double *xp, int64_t xp_size,
                     const double *fp, double *res,
                     const double *left, const double *right);

void s_interp_double(const double *x, const int64_t *stridesX,
                     const double *xp, int64_t strideXP, int64_t xp_size,
                     const double *fp, int64_t strideFP,
                     double *res, const int64_t *stridesRes,
                     const int64_t *shape, int rank,
                     const double *left, const double *right);

void v_interp_float(const float *x, int64_t x_size,
                    const float *xp, int64_t xp_size,
                    const float *fp, float *res,
                    const float *left, const float *right);

void s_interp_float(const float *x, const int64_t *stridesX,
                    const float *xp, int64_t strideXP, int64_t xp_size,
                    const float *fp, int64_t strideFP,
                    float *res, const int64_t *stridesRes,
                    const int64_t *shape, int rank,
                    const float *left, const float *right);

void native_interp_double(const double *x, int64_t x_size,
                          const double *xp, int64_t xp_size,
                          const double *fp, double *res,
                          const double *left, const double *right);

void native_interp_float(const float *x, int64_t x_size,
                         const float *xp, int64_t xp_size,
                         const float *fp, float *res,
                         const float *left, const float *right);

#define CMP_OP_EQ 0
#define CMP_OP_NE 1
#define CMP_OP_LT 2
#define CMP_OP_LE 3
#define CMP_OP_GT 4
#define CMP_OP_GE 5

#ifdef __cplusplus
enum class ComparisonOp : int {
    EQ = CMP_OP_EQ,
    NE = CMP_OP_NE,
    LT = CMP_OP_LT,
    LE = CMP_OP_LE,
    GT = CMP_OP_GT,
    GE = CMP_OP_GE
};
#endif

int ndarray_equals(
    int dtype,
    const void *a, const int64_t *stridesA,
    const void *b, const int64_t *stridesB,
    const int64_t *shape, int rank
);

void ndarray_compare(
    int op, int dtypeA, int dtypeB,
    const void *a, const int64_t *stridesA,
    const void *b, const int64_t *stridesB,
    uint8_t *res, const int64_t *stridesRes,
    const int64_t *shape, int rank
);

// Reduction Min/Max
double r_min_double(const double *src, int64_t size);
float r_min_float(const float *src, int64_t size);
int64_t r_min_int64_t(const int64_t *src, int64_t size);
int32_t r_min_int32_t(const int32_t *src, int64_t size);
uint8_t r_min_uint8_t(const uint8_t *src, int64_t size);
int16_t r_min_int16_t(const int16_t *src, int64_t size);

double r_max_double(const double *src, int64_t size);
float r_max_float(const float *src, int64_t size);
int64_t r_max_int64_t(const int64_t *src, int64_t size);
int32_t r_max_int32_t(const int32_t *src, int64_t size);
uint8_t r_max_uint8_t(const uint8_t *src, int64_t size);
int16_t r_max_int16_t(const int16_t *src, int64_t size);

// Strided Reduction Min/Max
void s_min_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_min_float(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_min_int64_t(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_min_int32_t(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_min_uint8_t(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_min_int16_t(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);

void s_max_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_max_float(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_max_int64_t(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_max_int32_t(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_max_uint8_t(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_max_int16_t(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);

// Nanmin/Nanmax Reduction (float/double only)
double r_nanmin_double(const double *src, int64_t size);
double r_nanmax_double(const double *src, int64_t size);
float r_nanmin_float(const float *src, int64_t size);
float r_nanmax_float(const float *src, int64_t size);

void s_nanmin_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_nanmax_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_nanmin_float(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_nanmax_float(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);

double r_nansum_double(const double *src, int64_t size);
double r_nansum_float(const float *src, int64_t size);
double r_nanmean_double(const double *src, int64_t size);
double r_nanmean_float(const float *src, int64_t size);
double r_nanvar_double(const double *src, int64_t size);
double r_nanvar_float(const float *src, int64_t size);

void s_nansum_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_nansum_float(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_nanmean_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_nanmean_float(const float *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_nanvar_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_nanvar_float(const float *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);

void native_copy_strided(
    const void *src,
    const int64_t *stridesSrc,
    void *dest,
    const int64_t *stridesDest,
    const int64_t *shape,
    int rank,
    int itemSize
);
int ndarray_find_index(
    int op, int dtype,
    const void *a, const int64_t *stridesA,
    const int64_t *shape, int rank,
    const void *target,
    const int64_t *startCoords,
    const int64_t *directions,
    int64_t *matchCoords
);

// Pairwise distance functions
typedef enum {
    METRIC_EUCLIDEAN = 0,
    METRIC_COSINE,
    METRIC_HAMMING,
    METRIC_CHEBYSHEV
} DistanceMetric;

void ndarray_pdist(
    int dtype,
    const void *x,
    int64_t M, int64_t N,
    int64_t strideRowX, int64_t strideColX,
    int metric,
    double *out,
    int64_t strideOut
);

void ndarray_cdist(
    int dtype,
    const void *xa, const void *xb,
    int64_t M, int64_t K, int64_t N,
    int64_t strideRowXA, int64_t strideColXA,
    int64_t strideRowXB, int64_t strideColXB,
    int metric,
    double *out,
    int64_t strideRowOut, int64_t strideColOut
);

// Integer Absolute Value and Exponentiation ufuncs
void v_abs_int64(const int64_t *src, int64_t *res, int64_t size, const uint8_t *mask);
void v_abs_int32(const int32_t *src, int32_t *res, int64_t size, const uint8_t *mask);
void v_abs_int16(const int16_t *src, int16_t *res, int64_t size, const uint8_t *mask);
void v_abs_uint8(const uint8_t *src, uint8_t *res, int64_t size, const uint8_t *mask);

void s_abs_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_abs_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_abs_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_abs_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void v_pow_int64(const int64_t *x1, const int64_t *x2, int64_t *res, int64_t size, const uint8_t *mask);
void v_pow_int32(const int32_t *x1, const int32_t *x2, int32_t *res, int64_t size, const uint8_t *mask);
void v_pow_int16(const int16_t *x1, const int16_t *x2, int16_t *res, int64_t size, const uint8_t *mask);
void v_pow_int8(const int8_t *x1, const int8_t *x2, int8_t *res, int64_t size, const uint8_t *mask);
void v_pow_uint64(const uint64_t *x1, const uint64_t *x2, uint64_t *res, int64_t size, const uint8_t *mask);
void v_pow_uint32(const uint32_t *x1, const uint32_t *x2, uint32_t *res, int64_t size, const uint8_t *mask);
void v_pow_uint16(const uint16_t *x1, const uint16_t *x2, uint16_t *res, int64_t size, const uint8_t *mask);
void v_pow_uint8(const uint8_t *x1, const uint8_t *x2, uint8_t *res, int64_t size, const uint8_t *mask);
void v_pow_float16(const uint16_t *x1, const uint16_t *x2, uint16_t *res, int64_t size, const uint8_t *mask);
void v_pow_bfloat16(const uint16_t *x1, const uint16_t *x2, uint16_t *res, int64_t size, const uint8_t *mask);
void v_pow_boolean(const uint8_t *x1, const uint8_t *x2, uint8_t *res, int64_t size, const uint8_t *mask);

void s_pow_int64(const int64_t *x1, const int64_t *stridesX1, const int64_t *x2, const int64_t *stridesX2, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_pow_int32(const int32_t *x1, const int64_t *stridesX1, const int32_t *x2, const int64_t *stridesX2, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_pow_int16(const int16_t *x1, const int64_t *stridesX1, const int16_t *x2, const int64_t *stridesX2, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_pow_int8(const int8_t *x1, const int64_t *stridesX1, const int8_t *x2, const int64_t *stridesX2, int8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_pow_uint64(const uint64_t *x1, const int64_t *stridesX1, const uint64_t *x2, const int64_t *stridesX2, uint64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_pow_uint32(const uint32_t *x1, const int64_t *stridesX1, const uint32_t *x2, const int64_t *stridesX2, uint32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_pow_uint16(const uint16_t *x1, const int64_t *stridesX1, const uint16_t *x2, const int64_t *stridesX2, uint16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_pow_uint8(const uint8_t *x1, const int64_t *stridesX1, const uint8_t *x2, const int64_t *stridesX2, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_pow_float16(const uint16_t *x1, const int64_t *stridesX1, const uint16_t *x2, const int64_t *stridesX2, uint16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_pow_bfloat16(const uint16_t *x1, const int64_t *stridesX1, const uint16_t *x2, const int64_t *stridesX2, uint16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_pow_boolean(const uint8_t *x1, const int64_t *stridesX1, const uint8_t *x2, const int64_t *stridesX2, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void v_square_int16(const int16_t *src, int16_t *res, int64_t size, const uint8_t *mask);
void v_square_int8(const int8_t *src, int8_t *res, int64_t size, const uint8_t *mask);
void v_square_uint64(const uint64_t *src, uint64_t *res, int64_t size, const uint8_t *mask);
void v_square_uint32(const uint32_t *src, uint32_t *res, int64_t size, const uint8_t *mask);
void v_square_uint16(const uint16_t *src, uint16_t *res, int64_t size, const uint8_t *mask);
void v_square_uint8(const uint8_t *src, uint8_t *res, int64_t size, const uint8_t *mask);
void v_square_float16(const uint16_t *src, uint16_t *res, int64_t size, const uint8_t *mask);
void v_square_bfloat16(const uint16_t *src, uint16_t *res, int64_t size, const uint8_t *mask);
void v_square_boolean(const uint8_t *src, uint8_t *res, int64_t size, const uint8_t *mask);

void s_square_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_square_int8(const int8_t *src, const int64_t *stridesSrc, int8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_square_uint64(const uint64_t *src, const int64_t *stridesSrc, uint64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_square_uint32(const uint32_t *src, const int64_t *stridesSrc, uint32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_square_uint16(const uint16_t *src, const int64_t *stridesSrc, uint16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_square_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_square_float16(const uint16_t *src, const int64_t *stridesSrc, uint16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_square_bfloat16(const uint16_t *src, const int64_t *stridesSrc, uint16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void s_square_boolean(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

/* Bincount operations */
void v_bincount_int32(const int32_t *src, int64_t *res, int64_t size, int64_t res_size);
void v_bincount_int64(const int64_t *src, int64_t *res, int64_t size, int64_t res_size);
void v_bincount_weights_int32_double(const int32_t *src, const double *weights, double *res, int64_t size, int64_t res_size);
void v_bincount_weights_int64_double(const int64_t *src, const double *weights, double *res, int64_t size, int64_t res_size);
void v_bincount_weights_int32_float(const int32_t *src, const float *weights, float *res, int64_t size, int64_t res_size);
void v_bincount_weights_int64_float(const int64_t *src, const float *weights, float *res, int64_t size, int64_t res_size);

void s_bincount_int32(const int32_t *src, int64_t strideSrc, int64_t *res, int64_t strideRes, int64_t size, int64_t res_size);
void s_bincount_int64(const int64_t *src, int64_t strideSrc, int64_t *res, int64_t strideRes, int64_t size, int64_t res_size);
void s_bincount_weights_int32_double(const int32_t *src, int64_t strideSrc, const double *weights, int64_t strideWeights, double *res, int64_t strideRes, int64_t size, int64_t res_size);
void s_bincount_weights_int64_double(const int64_t *src, int64_t strideSrc, const double *weights, int64_t strideWeights, double *res, int64_t strideRes, int64_t size, int64_t res_size);
void s_bincount_weights_int32_float(const int32_t *src, int64_t strideSrc, const float *weights, int64_t strideWeights, float *res, int64_t strideRes, int64_t size, int64_t res_size);
void s_bincount_weights_int64_float(const int64_t *src, int64_t strideSrc, const float *weights, int64_t strideWeights, float *res, int64_t strideRes, int64_t size, int64_t res_size);

void s_slogdet_double(const double *a, const int64_t *stridesA, double *sign, const int64_t *stridesSign, double *logdet, const int64_t *stridesLogdet, const int64_t *shape, int rank, double *aCopy, int *ipiv, int (*lapack_getrf)(int, int, int, void *, int, int *));
void s_slogdet_float(const float *a, const int64_t *stridesA, float *sign, const int64_t *stridesSign, float *logdet, const int64_t *stridesLogdet, const int64_t *shape, int rank, float *aCopy, int *ipiv, int (*lapack_getrf)(int, int, int, void *, int, int *));
void s_slogdet_complex_double(const cpx_t *a, const int64_t *stridesA, cpx_t *sign, const int64_t *stridesSign, double *logdet, const int64_t *stridesLogdet, const int64_t *shape, int rank, cpx_t *aCopy, int *ipiv, int (*lapack_getrf)(int, int, int, void *, int, int *));
void s_slogdet_complex_float(const cpx_f_t *a, const int64_t *stridesA, cpx_f_t *sign, const int64_t *stridesSign, float *logdet, const int64_t *stridesLogdet, const int64_t *shape, int rank, cpx_f_t *aCopy, int *ipiv, int (*lapack_getrf)(int, int, int, void *, int, int *));

void assemble_eigenvalues_double(cpx_t *w, int64_t strideWLast, const double *wr, const double *wi, int64_t n);
void assemble_eigenvalues_float(cpx_f_t *w, int64_t strideWLast, const float *wr, const float *wi, int64_t n);

/* Bessel I0 operations */
void v_i0_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void s_i0_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void v_i0_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void s_i0_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void v_i0_complex64(const cpx_f_t *src, cpx_f_t *res, int64_t size, const uint8_t *mask);
void s_i0_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void v_i0_complex128(const cpx_t *src, cpx_t *res, int64_t size, const uint8_t *mask);
void s_i0_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void v_gamma_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void s_gamma_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void v_gamma_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void s_gamma_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);

void v_erf_float(const float *src, float *res, int64_t size, const uint8_t *mask);
void s_erf_float(const float *src, const int64_t *stridesSrc, float *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);
void v_erf_double(const double *src, double *res, int64_t size, const uint8_t *mask);
void s_erf_double(const double *src, const int64_t *stridesSrc, double *res, const int64_t *stridesRes, const int64_t *shape, int rank, const uint8_t *mask);


/* Spatial N-D Correlation */
void s_correlate_valid_double(const double *src, const int64_t *stridesSrc, const double *kernel, const int64_t *stridesKernel, double *res, const int64_t *stridesRes, const int64_t *resShape, const int64_t *kernelShape, int rank);
void s_correlate_valid_float(const float *src, const int64_t *stridesSrc, const float *kernel, const int64_t *stridesKernel, float *res, const int64_t *stridesRes, const int64_t *resShape, const int64_t *kernelShape, int rank);
void s_correlate_valid_complex128(const cpx_t *src, const int64_t *stridesSrc, const cpx_t *kernel, const int64_t *stridesKernel, cpx_t *res, const int64_t *stridesRes, const int64_t *resShape, const int64_t *kernelShape, int rank);
void s_correlate_valid_complex64(const cpx_f_t *src, const int64_t *stridesSrc, const cpx_f_t *kernel, const int64_t *stridesKernel, cpx_f_t *res, const int64_t *stridesRes, const int64_t *resShape, const int64_t *kernelShape, int rank);
void s_correlate_valid_int64(const int64_t *src, const int64_t *stridesSrc, const int64_t *kernel, const int64_t *stridesKernel, int64_t *res, const int64_t *stridesRes, const int64_t *resShape, const int64_t *kernelShape, int rank);
void s_correlate_valid_int32(const int32_t *src, const int64_t *stridesSrc, const int32_t *kernel, const int64_t *stridesKernel, int32_t *res, const int64_t *stridesRes, const int64_t *resShape, const int64_t *kernelShape, int rank);


/* Generalized Ufunc Reduction, Cumulative & Scatter AT Kernels */
double r_prod_double(const double *src, int64_t size);
float r_prod_float(const float *src, int64_t size);
int64_t r_sum_int64(const int64_t *src, int64_t size);
int32_t r_sum_int32(const int32_t *src, int64_t size);
uint8_t r_sum_uint8(const uint8_t *src, int64_t size);
int16_t r_sum_int16(const int16_t *src, int64_t size);
void r_sum_complex128(const cpx_t *src, int64_t size, cpx_t *out);
void r_sum_complex64(const cpx_f_t *src, int64_t size, cpx_f_t *out);

double r_mean_int64_to_double(const int64_t *src, int64_t size);
double r_mean_int32_to_double(const int32_t *src, int64_t size);
double r_mean_uint8_to_double(const uint8_t *src, int64_t size);
double r_mean_int16_to_double(const int16_t *src, int64_t size);
void r_mean_complex128(const cpx_t *src, int64_t size, cpx_t *out);
void r_mean_complex64(const cpx_f_t *src, int64_t size, cpx_f_t *out);
void r_mean_complex64_to_complex128(const cpx_f_t *src, int64_t size, cpx_t *out);

double r_var_int64_to_double(const int64_t *src, int64_t size, int ddof);
double r_var_int32_to_double(const int32_t *src, int64_t size, int ddof);
double r_var_uint8_to_double(const uint8_t *src, int64_t size, int ddof);
double r_var_int16_to_double(const int16_t *src, int64_t size, int ddof);

double r_std_int64_to_double(const int64_t *src, int64_t size, int ddof);
double r_std_int32_to_double(const int32_t *src, int64_t size, int ddof);
double r_std_uint8_to_double(const uint8_t *src, int64_t size, int ddof);
double r_std_int16_to_double(const int16_t *src, int64_t size, int ddof);

int64_t r_prod_int64(const int64_t *src, int64_t size);
int32_t r_prod_int32(const int32_t *src, int64_t size);
uint8_t r_prod_uint8(const uint8_t *src, int64_t size);
int16_t r_prod_int16(const int16_t *src, int64_t size);
void r_prod_complex128(const cpx_t *src, int64_t size, cpx_t *out);
void r_prod_complex64(const cpx_f_t *src, int64_t size, cpx_f_t *out);

void s_prod_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_prod_float(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_prod_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_prod_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_prod_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_prod_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_prod_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_prod_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);

void s_sum_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_sum_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_sum_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_sum_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_sum_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_sum_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);

void s_mean_int64_to_double(const int64_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_mean_int32_to_double(const int32_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_mean_uint8_to_double(const uint8_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_mean_int16_to_double(const int16_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_mean_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_mean_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_mean_complex64_to_complex128(const cpx_f_t *src, const int64_t *stridesSrc, cpx_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);

void s_var_int64_to_double(const int64_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof);
void s_var_int32_to_double(const int32_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof);
void s_var_uint8_to_double(const uint8_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof);
void s_var_int16_to_double(const int16_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof);

void s_std_int64_to_double(const int64_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof);
void s_std_int32_to_double(const int32_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof);
void s_std_uint8_to_double(const uint8_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof);
void s_std_int16_to_double(const int16_t *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, int ddof);

int64_t r_bitwise_and_int64(const int64_t *src, int64_t size);
int32_t r_bitwise_and_int32(const int32_t *src, int64_t size);
uint8_t r_bitwise_and_uint8(const uint8_t *src, int64_t size);
int16_t r_bitwise_and_int16(const int16_t *src, int64_t size);
int64_t r_bitwise_or_int64(const int64_t *src, int64_t size);
int32_t r_bitwise_or_int32(const int32_t *src, int64_t size);
uint8_t r_bitwise_or_uint8(const uint8_t *src, int64_t size);
int16_t r_bitwise_or_int16(const int16_t *src, int64_t size);
int64_t r_bitwise_xor_int64(const int64_t *src, int64_t size);
int32_t r_bitwise_xor_int32(const int32_t *src, int64_t size);
uint8_t r_bitwise_xor_uint8(const uint8_t *src, int64_t size);
int16_t r_bitwise_xor_int16(const int16_t *src, int64_t size);

void s_bitwise_and_red_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_bitwise_and_red_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_bitwise_and_red_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_bitwise_and_red_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);

void s_bitwise_or_red_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_bitwise_or_red_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_bitwise_or_red_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_bitwise_or_red_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);

void s_bitwise_xor_red_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_bitwise_xor_red_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_bitwise_xor_red_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_bitwise_xor_red_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);

uint8_t r_logical_and(const uint8_t *src, int64_t size);
uint8_t r_logical_or(const uint8_t *src, int64_t size);
uint8_t r_logical_xor(const uint8_t *src, int64_t size);

void s_logical_and_red(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_logical_or_red(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);
void s_logical_xor_red(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis);

void s_cumbitwise_and_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumbitwise_and_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumbitwise_and_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumbitwise_and_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);

void s_cumbitwise_or_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumbitwise_or_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumbitwise_or_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumbitwise_or_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);

void s_cumbitwise_xor_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumbitwise_xor_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumbitwise_xor_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumbitwise_xor_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);

void s_cumlogical_and(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumlogical_or(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);
void s_cumlogical_xor(const uint8_t *src, const int64_t *stridesSrc, uint8_t *res, const int64_t *stridesRes, const int64_t *shape, int rank, int axis);

void s_at_double(double *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const double *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode);
void s_at_float(float *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const float *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode);
void s_at_int64(int64_t *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const int64_t *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode);
void s_at_int32(int32_t *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const int32_t *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode);
void s_at_uint8(uint8_t *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const uint8_t *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode);
void s_at_int16(int16_t *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const int16_t *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode);
void s_at_complex128(cpx_t *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const cpx_t *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode);
void s_at_complex64(cpx_f_t *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const cpx_f_t *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode);
void s_at_boolean(uint8_t *a, const int64_t *stridesA, const int64_t *shapeA, int rankA, const int64_t *indices, int64_t numIndices, int64_t strideIdx, const uint8_t *b, const int64_t *stridesB, const int64_t *shapeB, int rankB, int opCode);


void v_reduceat_double(const double *src, int64_t size, const int64_t *indices, int64_t numIndices, double *dest, int opCode);
void v_reduceat_float(const float *src, int64_t size, const int64_t *indices, int64_t numIndices, float *dest, int opCode);
void v_reduceat_int64(const int64_t *src, int64_t size, const int64_t *indices, int64_t numIndices, int64_t *dest, int opCode);
void v_reduceat_int32(const int32_t *src, int64_t size, const int64_t *indices, int64_t numIndices, int32_t *dest, int opCode);
void v_reduceat_uint8(const uint8_t *src, int64_t size, const int64_t *indices, int64_t numIndices, uint8_t *dest, int opCode);
void v_reduceat_int16(const int16_t *src, int64_t size, const int64_t *indices, int64_t numIndices, int16_t *dest, int opCode);
void v_reduceat_complex128(const cpx_t *src, int64_t size, const int64_t *indices, int64_t numIndices, cpx_t *dest, int opCode);
void v_reduceat_complex64(const cpx_f_t *src, int64_t size, const int64_t *indices, int64_t numIndices, cpx_f_t *dest, int opCode);
void v_reduceat_boolean(const uint8_t *src, int64_t size, const int64_t *indices, int64_t numIndices, uint8_t *dest, int opCode);

void s_reduceat_double(const double *src, const int64_t *stridesSrc, double *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode);
void s_reduceat_float(const float *src, const int64_t *stridesSrc, float *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode);
void s_reduceat_int64(const int64_t *src, const int64_t *stridesSrc, int64_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode);
void s_reduceat_int32(const int32_t *src, const int64_t *stridesSrc, int32_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode);
void s_reduceat_uint8(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode);
void s_reduceat_int16(const int16_t *src, const int64_t *stridesSrc, int16_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode);
void s_reduceat_complex128(const cpx_t *src, const int64_t *stridesSrc, cpx_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode);
void s_reduceat_complex64(const cpx_f_t *src, const int64_t *stridesSrc, cpx_f_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode);
void s_reduceat_boolean(const uint8_t *src, const int64_t *stridesSrc, uint8_t *dest, const int64_t *stridesDest, const int64_t *shape, int rank, int axis, const int64_t *indices, int64_t numIndices, int opCode);

void v_polyval_double(const double *c, int64_t stride_c, int64_t n_c, const double *x, double *res, int64_t size);
void v_polyval_float(const float *c, int64_t stride_c, int64_t n_c, const float *x, float *res, int64_t size);
void v_polyval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, const cpx_t *x, cpx_t *res, int64_t size);
void v_polyval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, const cpx_f_t *x, cpx_f_t *res, int64_t size);

void s_polyval_double(const double *c, int64_t stride_c, int64_t n_c, const double *x, const int64_t *stridesX, double *res, const int64_t *stridesRes, const int64_t *shape, int rank);
void s_polyval_float(const float *c, int64_t stride_c, int64_t n_c, const float *x, const int64_t *stridesX, float *res, const int64_t *stridesRes, const int64_t *shape, int rank);
void s_polyval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, const cpx_t *x, const int64_t *stridesX, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank);
void s_polyval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, const cpx_f_t *x, const int64_t *stridesX, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank);

void v_chebval_double(const double *c, int64_t stride_c, int64_t n_c, const double *x, double *res, int64_t size);
void v_chebval_float(const float *c, int64_t stride_c, int64_t n_c, const float *x, float *res, int64_t size);
void v_chebval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, const cpx_t *x, cpx_t *res, int64_t size);
void v_chebval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, const cpx_f_t *x, cpx_f_t *res, int64_t size);

void s_chebval_double(const double *c, int64_t stride_c, int64_t n_c, const double *x, const int64_t *stridesX, double *res, const int64_t *stridesRes, const int64_t *shape, int rank);
void s_chebval_float(const float *c, int64_t stride_c, int64_t n_c, const float *x, const int64_t *stridesX, float *res, const int64_t *stridesRes, const int64_t *shape, int rank);
void s_chebval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, const cpx_t *x, const int64_t *stridesX, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank);
void s_chebval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, const cpx_f_t *x, const int64_t *stridesX, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank);

void v_legval_double(const double *c, int64_t stride_c, int64_t n_c, const double *x, double *res, int64_t size);
void v_legval_float(const float *c, int64_t stride_c, int64_t n_c, const float *x, float *res, int64_t size);
void v_legval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, const cpx_t *x, cpx_t *res, int64_t size);
void v_legval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, const cpx_f_t *x, cpx_f_t *res, int64_t size);

void s_legval_double(const double *c, int64_t stride_c, int64_t n_c, const double *x, const int64_t *stridesX, double *res, const int64_t *stridesRes, const int64_t *shape, int rank);
void s_legval_float(const float *c, int64_t stride_c, int64_t n_c, const float *x, const int64_t *stridesX, float *res, const int64_t *stridesRes, const int64_t *shape, int rank);
void s_legval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, const cpx_t *x, const int64_t *stridesX, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank);
void s_legval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, const cpx_f_t *x, const int64_t *stridesX, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank);

void v_hermval_double(const double *c, int64_t stride_c, int64_t n_c, const double *x, double *res, int64_t size);
void v_hermval_float(const float *c, int64_t stride_c, int64_t n_c, const float *x, float *res, int64_t size);
void v_hermval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, const cpx_t *x, cpx_t *res, int64_t size);
void v_hermval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, const cpx_f_t *x, cpx_f_t *res, int64_t size);

void s_hermval_double(const double *c, int64_t stride_c, int64_t n_c, const double *x, const int64_t *stridesX, double *res, const int64_t *stridesRes, const int64_t *shape, int rank);
void s_hermval_float(const float *c, int64_t stride_c, int64_t n_c, const float *x, const int64_t *stridesX, float *res, const int64_t *stridesRes, const int64_t *shape, int rank);
void s_hermval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, const cpx_t *x, const int64_t *stridesX, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank);
void s_hermval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, const cpx_f_t *x, const int64_t *stridesX, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank);

void v_lagval_double(const double *c, int64_t stride_c, int64_t n_c, const double *x, double *res, int64_t size);
void v_lagval_float(const float *c, int64_t stride_c, int64_t n_c, const float *x, float *res, int64_t size);
void v_lagval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, const cpx_t *x, cpx_t *res, int64_t size);
void v_lagval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, const cpx_f_t *x, cpx_f_t *res, int64_t size);

void s_lagval_double(const double *c, int64_t stride_c, int64_t n_c, const double *x, const int64_t *stridesX, double *res, const int64_t *stridesRes, const int64_t *shape, int rank);
void s_lagval_float(const float *c, int64_t stride_c, int64_t n_c, const float *x, const int64_t *stridesX, float *res, const int64_t *stridesRes, const int64_t *shape, int rank);
void s_lagval_complex128(const cpx_t *c, int64_t stride_c, int64_t n_c, const cpx_t *x, const int64_t *stridesX, cpx_t *res, const int64_t *stridesRes, const int64_t *shape, int rank);
void s_lagval_complex64(const cpx_f_t *c, int64_t stride_c, int64_t n_c, const cpx_f_t *x, const int64_t *stridesX, cpx_f_t *res, const int64_t *stridesRes, const int64_t *shape, int rank);

void v_histogram_uniform(
    const void *src, int dtypeSrc,
    const void *weights, int dtypeWeights,
    void *hist, int dtypeHist,
    int64_t size, int64_t nbins,
    double min_val, double max_val, double norm
);

void s_histogram_uniform(
    const void *src, int64_t strideSrc, int dtypeSrc,
    const void *weights, int64_t strideWeights, int dtypeWeights,
    void *hist, int64_t strideHist, int dtypeHist,
    int64_t size, int64_t nbins,
    double min_val, double max_val, double norm
);

void v_histogram_binsearch(
    const void *src, int dtypeSrc,
    const void *weights, int dtypeWeights,
    void *hist, int dtypeHist,
    const double *bin_edges, int64_t num_edges,
    int64_t size
);

void s_histogram_binsearch(
    const void *src, int64_t strideSrc, int dtypeSrc,
    const void *weights, int64_t strideWeights, int dtypeWeights,
    void *hist, int64_t strideHist, int dtypeHist,
    const double *bin_edges, int64_t num_edges,
    int64_t size
);

void v_vander_fit_double(const double *x, const double *y, const double *w,
                         double *v_mat, double *rhs,
                         int64_t m, int64_t deg);
void v_vander_fit_float(const float *x, const float *y, const float *w,
                        float *v_mat, float *rhs,
                        int64_t m, int64_t deg);
void v_vander_fit_complex128(const cpx_t *x, const cpx_t *y, const cpx_t *w,
                             cpx_t *v_mat, cpx_t *rhs,
                             int64_t m, int64_t deg);
void v_vander_fit_complex64(const cpx_f_t *x, const cpx_f_t *y, const cpx_f_t *w,
                            cpx_f_t *v_mat, cpx_f_t *rhs,
                            int64_t m, int64_t deg);

void s_vander_fit_double(const double *x, int64_t strideX,
                         const double *y, int64_t strideY,
                         const double *w, int64_t strideW,
                         double *v_mat, double *rhs,
                         int64_t m, int64_t deg);
void s_vander_fit_float(const float *x, int64_t strideX,
                        const float *y, int64_t strideY,
                        const float *w, int64_t strideW,
                        float *v_mat, float *rhs,
                        int64_t m, int64_t deg);
void s_vander_fit_complex128(const cpx_t *x, int64_t strideX,
                             const cpx_t *y, int64_t strideY,
                             const cpx_t *w, int64_t strideW,
                             cpx_t *v_mat, cpx_t *rhs,
                             int64_t m, int64_t deg);
void s_vander_fit_complex64(const cpx_f_t *x, int64_t strideX,
                            const cpx_f_t *y, int64_t strideY,
                            const cpx_f_t *w, int64_t strideW,
                            cpx_f_t *v_mat, cpx_f_t *rhs,
                            int64_t m, int64_t deg);

/* ============================================================================
 * SECTION: RANDOM SHUFFLE & CHOICE NATIVE ACCELERATORS
 * ============================================================================
 */

void native_shuffle_1d(
    void *data,
    int64_t size,
    int64_t stride,
    int item_size,
    unsigned long long seed
);

void native_shuffle_nd(
    void *data,
    const int64_t *shape,
    const int64_t *strides,
    int rank,
    int item_size,
    unsigned long long seed
);

void native_choice_uniform(
    const void *src,
    int64_t src_stride,
    void *dest,
    int64_t dest_stride,
    int64_t src_size,
    int64_t sample_count,
    int item_size,
    unsigned long long seed
);

void native_choice_weighted(
    const void *src,
    int64_t src_stride,
    void *dest,
    int64_t dest_stride,
    const double *cdf,
    int64_t src_size,
    int64_t sample_count,
    int item_size,
    unsigned long long seed
);

void native_choice_without_replacement(
    const void *src,
    int64_t src_stride,
    void *dest,
    int64_t dest_stride,
    int64_t src_size,
    int64_t sample_count,
    int item_size,
    unsigned long long seed
);

void native_choice_weighted_without_replacement(
    const void *src,
    int64_t src_stride,
    void *dest,
    int64_t dest_stride,
    const double *probs,
    int64_t src_size,
    int64_t sample_count,
    int item_size,
    unsigned long long seed
);

void v_binary_minmax(int op_code, int dtype, const void* a, const void* b, void* out, int64_t n);

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
);

#ifdef __cplusplus
}
#endif

