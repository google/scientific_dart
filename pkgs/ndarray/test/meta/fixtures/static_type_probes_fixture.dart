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

// Positive static-type probe fixture for `static_type_probes_test.dart`.
// Every `t_*` top-level variable has its inferred static type checked by
// `package:analyzer` against its `// expect: <Type>` annotation.
// ignore_for_file: non_constant_identifier_names, type_annotate_public_apis

import 'package:ndarray/ndarray.dart';

late final NDArray<Float64> f64;
late final NDArray<Float32> f32;
late final NDArray<Float16> f16;
late final NDArray<BFloat16> bf16;
late final NDArray<Int64> i64;
late final NDArray<Int32> i32;
late final NDArray<Int16> i16;
late final NDArray<Int8> i8;
late final NDArray<Uint64> u64;
late final NDArray<Uint32> u32;
late final NDArray<Uint16> u16;
late final NDArray<Uint8> u8;
late final NDArray<Complex128> c128;
late final NDArray<Complex64> c64;
late final NDArray<Boolean> b;
late final NDArray<DTypeTag> dyn;
late final NDArray<AnySpec> anySpec;

// ---------------------------------------------------------------------------
// Section A — acceptance-criteria probes (ORIGINAL_REQUEST.md §Acceptance)
// ---------------------------------------------------------------------------

// expect: NDArray<Float64>
final t_sin_i32 = sin(i32);

// expect: NDArray<Float64>
final t_sin_f16 = sin(f16);

// expect: NDArray<Complex64>
final t_sin_c64 = sin(c64);

// expect: NDArray<Float64>
final t_sin_b = sin(b);

// expect: NDArray<Float16>
final t_opdiv_f16 = f16 / f16;

// expect: NDArray<Float64>
final t_opdiv_i32 = i32 / i32;

// expect: NDArray<Float64>
final t_divide_b = divide(b, b);

// expect: NDArray<Complex64>
final t_opdiv_c64 = c64 / c64;

// expect: NDArray<Uint64>
final t_sum_u8 = sum(u8);

// expect: NDArray<Int64>
final t_sum_b = sum(b);

// expect: NDArray<Float16>
final t_sum_f16 = sum(f16);

// expect: NDArray<Float64>
final t_abs_c128 = abs(c128);

// expect: NDArray<Float32>
final t_real_c64 = real(c64);

// expect: List<double>
final t_tolist_f16 = f16.toList();

// expect: List<int>
final t_tolist_i32 = i32.toList();

// expect: List<bool>
final t_tolist_b = b.toList();

// expect: List<Complex>
final t_tolist_c64 = c64.toList();

// expect: List<NDArray<IntegerDType>>
final t_lub_i64_i32 = [i64, i32];

// expect: List<int>
final t_lub_first_tolist = [i64, i32].first.toList();

// expect: NDArray<Float64>
final t_sin_lub_i64_i32 = sin([i64, i32].first);

// expect: NDArray<DTypeTag>
final t_sin_anySpec = sin(anySpec);

// expect: NDArray<DTypeTag>
final t_opdiv_anySpec = anySpec / anySpec;

// expect: NDArray<DTypeTag>
final t_sum_anySpec = sum(anySpec);

// expect: NDArray<Float16>
final t_add_f16 = add(f16, f16);

// expect: NDArray<Int32>
final t_bitwiseAnd_i32 = bitwiseAnd(i32, i32);

// expect: NDArray<Int32>
final t_opshl_i32 = i32 << i32;

// expect: NDArray<Boolean>
final t_opand_b = b & b;

// ---------------------------------------------------------------------------
// Section A' — additional positive probes (LUBs, escape hatches, markers, ops)
// ---------------------------------------------------------------------------

// expect: List<NDArray<BitwiseDType>>
final t_lub_i32_b = [i32, b];

// expect: List<NDArray<DTypeTag>>
final t_lub_f64_f32 = [f64, f32];

// expect: List<NDArray<IntegerDType>>
final t_lub_i32_u8 = [i32, u8];

// expect: List<NDArray<DTypeTag>>
final t_lub_c64_f32 = [c64, f32];

// expect: List<dynamic>
final t_tolist_dyn = dyn.toList();

// expect: NDArray<DTypeTag>
final t_opdiv_dyn = dyn / dyn;

// expect: List<dynamic>
final t_tolist_anySpec = anySpec.toList();

// expect: List<double>
final t_data_f32 = f32.data;

// expect: int
final t_scalar_u64 = u64.scalar;

// expect: dynamic
final t_scalar_dyn = dyn.scalar;

// expect: NDArray<DTypeTag>
final t_add_dyn = add(dyn, dyn);

// expect: NDArray<DTypeTag>
final t_add_mixed_f64_f32 = add(f64, f32);

// expect: NDArray<IntegerDType>
final t_add_mixed_i64_i32 = add(i64, i32);

// expect: NDArray<Float64>
final t_opadd_mixed = f64 + f32;

// expect: NDArray<Float32>
final t_sin_f32 = sin(f32);

// expect: NDArray<Float64>
final t_sin_bf16 = sin(bf16);

// expect: NDArray<Complex128>
final t_sin_c128 = sin(c128);

// expect: NDArray<Float16>
final t_divide_f16 = divide(f16, f16);

// expect: NDArray<Float64>
final t_divide_i32 = divide(i32, i32);

// expect: NDArray<BFloat16>
final t_opdiv_bf16 = bf16 / bf16;

// expect: NDArray<Int64>
final t_sum_i8 = sum(i8);

// expect: NDArray<Uint64>
final t_sum_u32 = sum(u32);

// expect: NDArray<BFloat16>
final t_sum_bf16 = sum(bf16);

// expect: NDArray<Complex64>
final t_sum_c64 = sum(c64);

// expect: NDArray<Float32>
final t_abs_c64 = abs(c64);

// expect: NDArray<Int32>
final t_abs_i32 = abs(i32);

// expect: NDArray<Float16>
final t_abs_f16 = abs(f16);

// expect: NDArray<Float64>
final t_real_c128 = real(c128);

// expect: NDArray<Float32>
final t_real_f32 = real(f32);

// expect: NDArray<Float32>
final t_imag_c64 = imag(c64);

// expect: NDArray<Float64>
final t_mean_i32 = mean(i32);

// expect: NDArray<Complex128>
final t_mean_c64 = mean(c64);

// expect: NDArray<Float64>
final t_mean_f16 = mean(f16);

// expect: NDArray<Float32>
final t_rint_f32 = rint(f32);

// expect: NDArray<Float32>
final t_rint_c64 = rint(c64);

// expect: NDArray<Complex64>
final t_fft_f32 = fft(f32);

// expect: NDArray<Complex128>
final t_fft_i32 = fft(i32);

// expect: NDArray<Complex64>
final t_fft_c64 = fft(c64);

// expect: NDArray<Boolean>
final t_bitwiseAnd_b = bitwiseAnd(b, b);

// expect: NDArray<Uint8>
final t_leftShift_u8 = leftShift(u8, u8);

// expect: NDArray<Int32>
final t_opand_i32 = i32 & i32;

// expect: NDArray<Uint16>
final t_opor_u16 = u16 | u16;

// expect: NDArray<Boolean>
final t_opinv_b = ~b;

// expect: NDArray<Int64>
final t_opshr_i64 = i64 >> i64;

// expect: NDArray<DTypeTag>
final t_asAnySpec_sin = sin(f64.asAnySpec);

// expect: ({NDArray<Float32> eigenvalues, NDArray<Complex64> eigenvectors})
final t_eigh_c64 = eigh(c64);

// expect: ({NDArray<Float32> eigenvalues, NDArray<Float32> eigenvectors})
final t_eigh_f32 = eigh(f32);

// expect: ({NDArray<Float32> logabsdet, NDArray<Complex64> sign})
final t_slogdet_c64 = slogdet(c64);

// expect: ({NDArray<Float32> s, NDArray<Float32> u, NDArray<Float32> vh})
final t_svd_f32 = svd(f32);

// expect: NDArray<IntegerDType>
final t_bitwiseAnd_lub_i32_u8 = bitwiseAnd([i32, u8].first, [i32, u8].first);

// expect: List<dynamic>
final t_tolist_lub_i32_b = [i32, b].first.toList();

// ---------------------------------------------------------------------------
// Generic-context helpers bounded by IntegerDType and BitwiseDType (R2)
// ---------------------------------------------------------------------------

List<int> genericIntsOf<T extends IntegerDType>(NDArray<T> a) => a.toList();

NDArray<Float64> genericSinOfInt<T extends IntegerDType>(NDArray<T> a) =>
    sin(a);

NDArray<Float64> genericSinOfBitwise<T extends BitwiseDType>(NDArray<T> a) =>
    sin(a);

NDArray<Float64> genericDivOfInt<T extends IntegerDType>(NDArray<T> a) => a / a;

NDArray<Float64> genericDivOfBitwise<T extends BitwiseDType>(NDArray<T> a) =>
    a / a;

// expect: List<int>
final t_generic_ints_i32 = genericIntsOf(i32);

// expect: NDArray<Float64>
final t_generic_sin_int_i32 = genericSinOfInt(i32);

// expect: NDArray<Float64>
final t_generic_sin_bitwise_b = genericSinOfBitwise(b);

// expect: NDArray<Float64>
final t_generic_div_int_i32 = genericDivOfInt(i32);

// expect: NDArray<Float64>
final t_generic_div_bitwise_b = genericDivOfBitwise(b);
