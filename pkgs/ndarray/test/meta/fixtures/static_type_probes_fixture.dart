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

// expect: NDArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_add_anySpec = add(anySpec, anySpec);

// expect: NDArray<DTypeTag>
final t_divide_anySpec = divide(anySpec, anySpec);

// expect: NDArray<DTypeSpec<BitwiseDType, dynamic, Float64, Complex128, Float64, DTypeTag, Float64, Float64>>
final t_bitwiseAnd_asBitwise = bitwiseAnd(
  f64.asBitwiseDType,
  f32.asBitwiseDType,
);

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

// expect: NDArray<DTypeSpec<IntegerDType, int, Float64, Complex128, Float64, DTypeTag, Float64, Float64>>
final t_leftShift_asInteger = leftShift(i32.asIntegerDType, u8.asIntegerDType);

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

// ---------------------------------------------------------------------------
// Section B — same-dtype binary & multi-array operations (SelfOf bounds)
// ---------------------------------------------------------------------------

// Dtype-preserving binaries (`T extends SelfOf<DTypeTag>`):

// expect: NDArray<Float32>
final t_matmul_f32 = matmul(f32, f32);

// expect: NDArray<Complex64>
final t_solve_c64 = solve(c64, c64);

// expect: NDArray<Int64>
final t_gcd_i64 = gcd(i64, i64);

// expect: NDArray<Boolean>
final t_equal_i64 = equal(i64, i64);

// expect: NDArray<Int64>
final t_searchsorted_f64 = searchsorted(f64, f64);

// Projecting binaries (`T extends SelfOf<XOf<R>>` still infers `R`):

// expect: NDArray<Float32>
final t_atan2_f32 = atan2(f32, f32);

// expect: NDArray<Float64>
final t_atan2_i32 = atan2(i32, i32);

// expect: NDArray<Complex64>
final t_hypot_c64 = hypot(c64, c64);

// expect: NDArray<Float64>
final t_floatPower_f32 = floatPower(f32, f32);

// expect: NDArray<Float64>
final t_polyval_i32 = polyval(i32, i32);

// expect: NDArray<Float64>
final t_cov_f32 = cov(f32, y: f32);

// Multi-array operations (`List<NDArray<T>>`):

// expect: NDArray<Float64>
final t_concat_f64 = concatenate([f64, f64]);

// expect: NDArray<Int32>
final t_stack_i32 = stack([i32, i32]);

// expect: NDArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_concat_anySpec = concatenate([f64.asAnySpec, f32.asAnySpec]);

// AnySpec absorbs a concrete operand (dtype validated at run time):

// expect: NDArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_add_anySpec_f32 = add(anySpec, f32);

// Unary bitwise / shift operators on the checked escape hatches:

// expect: NDArray<DTypeSpec<BitwiseDType, dynamic, Float64, Complex128, Float64, DTypeTag, Float64, Float64>>
final t_opinv_asBitwise = ~dyn.asBitwiseDType;

// expect: NDArray<DTypeSpec<IntegerDType, int, Float64, Complex128, Float64, DTypeTag, Float64, Float64>>
final t_opshl_asInteger = dyn.asIntegerDType << 1;

// Float32 and Int32 share exactly one depth-2 interface, so their LUB is a
// projection interface rather than DTypeTag. SelfOf still rejects
// `floatPower(f32, i32)` (see negative probes) because no projection
// interface implements SelfOf.

// expect: List<NDArray<DoublePrecisionOf<Float64>>>
final t_lub_f32_i32 = [f32, i32];

// Generic helpers must bound `T` by `SelfOf<...>` to call same-dtype
// operations; `T extends DTypeTag` / `T extends IntegerDType` cannot (see
// negative probes).
NDArray<T> genericAddSelfOf<T extends SelfOf<DTypeTag>>(
  NDArray<T> a,
  NDArray<T> b,
) => add(a, b);

NDArray<T> genericGcdSelfOf<T extends SelfOf<RealOf<IntegerDType>>>(
  NDArray<T> a,
  NDArray<T> b,
) => gcd(a, b);

NDArray<R> genericAtan2SelfOf<
  T extends SelfOf<InexactOf<R>>,
  R extends DTypeTag
>(NDArray<T> a, NDArray<T> b) => atan2(a, b);

// expect: NDArray<Int32>
final t_generic_add_selfof_i32 = genericAddSelfOf(i32, i32);

// expect: NDArray<Uint8>
final t_generic_gcd_selfof_u8 = genericGcdSelfOf(u8, u8);

// expect: NDArray<Float64>
final t_generic_atan2_selfof_i16 = genericAtan2SelfOf(i16, i16);

// expect: NDArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_generic_add_selfof_anySpec = genericAddSelfOf(anySpec, anySpec);

// ---------------------------------------------------------------------------
// Section C — run-time-typed results, schur projections, marker operators,
// and composable row bounds for generic code
// ---------------------------------------------------------------------------

// Results whose dtype is only known at run time default to AnySpec and
// therefore compose with every other operation:

// expect: NDArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_multi_dot = multi_dot([f64, f64]);

// expect: NDArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_add_multi_dot = add(multi_dot([f64, f64]), f64);

// expect: NDArray<DTypeTag>
final t_sin_multi_dot = sin(multi_dot([f64, f64]));

// expect: NDArray<Float32>
final t_multi_dot_out = multi_dot([f32, f32], out: f32);

// expect: NDArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_lstsq_x = lstsq(f64, f64).x;

// expect: NDArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_binaryUfunc = binaryUfunc(i32, i32, op: BinaryOp.add);

// schur projects InexactOf, complexSchur projects ComplexOf:

// expect: ({NDArray<Float64> t, NDArray<Float64> z})
final t_schur_f64 = schur(f64);

// expect: ({NDArray<Float32> t, NDArray<Float32> z})
final t_schur_f32 = schur(f32);

// expect: ({NDArray<Float64> t, NDArray<Float64> z})
final t_schur_i32 = schur(i32);

// expect: ({NDArray<Complex64> t, NDArray<Complex64> z})
final t_schur_c64 = schur(c64);

// expect: ({NDArray<Complex128> t, NDArray<Complex128> z})
final t_complexSchur_f64 = complexSchur(f64);

// expect: ({NDArray<Complex64> t, NDArray<Complex64> z})
final t_complexSchur_f32 = complexSchur(f32);

// expect: NDArray<Float64>
final t_matmul_schur = matmul(schur(f64).z, schur(f64).t);

// Marker-typed (LUB) arrays keep the operator forms of the bitwise family:

// expect: NDArray<IntegerDType>
final t_opinv_lub = ~[i32, u8].first;

// expect: NDArray<IntegerDType>
final t_opand_lub = [i32, u8].first & [i32, u8].first;

// expect: NDArray<IntegerDType>
final t_opshl_lub = [i32, u8].first << 1;

// expect: NDArray<BitwiseDType>
final t_opinv_lub_bitwise = ~[i32, b].first;

// Row-typedef bounds compose same-dtype binary and projecting unary
// operations on one type parameter (unlike `SelfOf<XOf<R>>`):
NDArray<R> genericSinOfSum<T extends InexactSpec<R>, R extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b,
) {
  final s = add(a, b);
  return sin(s);
}

NDArray<R> genericDivideRow<T extends DivideSpec<R>, R extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b,
) => divide(a, b);

// In row-bounded generic code the operator forms resolve to the non-generic
// `NDArrayBitwiseSpec` / `NDArrayShiftSpec` extensions and widen to the row;
// the function forms keep `T`.
NDArray<T> genericGcdOfInverted<T extends AnyIntegerSpec>(
  NDArray<T> a,
  NDArray<T> b,
) => gcd(invert(a), bitwiseAnd(b, b));

NDArray<T> genericAddAfterSin<T extends AnySpec>(NDArray<T> a, NDArray<T> b) {
  final s = sin(a);
  final q = a / b;
  print([s, q]);
  return add(a, b);
}

// expect: NDArray<Float64>
final t_generic_sin_of_sum_i32 = genericSinOfSum(i32, i32);

// expect: NDArray<Float32>
final t_generic_sin_of_sum_f32 = genericSinOfSum(f32, f32);

// expect: NDArray<Float64>
final t_generic_divide_i16 = genericDivideRow(i16, i16);

// expect: NDArray<Uint8>
final t_generic_gcd_inverted_u8 = genericGcdOfInverted(u8, u8);

// expect: NDArray<Int32>
final t_generic_add_after_sin_i32 = genericAddAfterSin(i32, i32);

// expect: NDArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_generic_add_after_sin_any = genericAddAfterSin(anySpec, anySpec);

// The operator forms on the escape-hatch rows resolve to the non-generic
// `NDArrayBitwiseSpec`, so `~` on an `AnyIntegerSpec` array widens to
// `AnyBitwiseSpec`; the function form keeps the integer row.

// expect: NDArray<DTypeSpec<BitwiseDType, dynamic, Float64, Complex128, Float64, DTypeTag, Float64, Float64>>
final t_opinv_asInteger = ~dyn.asIntegerDType;

// expect: NDArray<DTypeSpec<IntegerDType, int, Float64, Complex128, Float64, DTypeTag, Float64, Float64>>
final t_invert_asInteger = invert(dyn.asIntegerDType);

// ---------------------------------------------------------------------------
// Section D — `out:` buffers typed by dependent `Out extends T` parameters
// ---------------------------------------------------------------------------

// A matching `out:` keeps the exact result type:

// expect: NDArray<Float64>
final t_sort_out = sort(f64, out: f64);

// expect: NDArray<Float32>
final t_sin_out = sin(f32, out: f32);

// expect: NDArray<Float64>
final t_sin_i32_out = sin(i32, out: f64);

// expect: NDArray<Int64>
final t_sumAs_out = sumAs(i32, DType.int64, out: i64);

// expect: NDArray<Int64>
final t_argsortAs_out = argsortAs(f64, DType.int64, out: i64);

// expect: ({NDArray<Float64> t, NDArray<Float64> z})
final t_schur_out = schur(f64, outT: f64, outZ: f64);

// expect: NDArray<Float64>
final t_divide_out = divide(i32, i32, out: f64);

// `NDArray<DTypeTag>` inputs and `T extends DTypeTag` generic callers keep
// working (the dependent parameter only constrains `out` relative to `T`):

// expect: NDArray<DTypeTag>
final t_sort_dyn = sort(dyn);

// expect: NDArray<DTypeTag>
final t_sort_dyn_out = sort(dyn, out: f32);

NDArray<T> genericSortInto<T extends DTypeTag>(NDArray<T> a, NDArray<T> buf) =>
    sort(a, out: buf);

// expect: NDArray<Int16>
final t_generic_sort_into = genericSortInto(i16, i16);

// Creation-style functions infer their dtype from `dtype:` or `out:` alone:

// expect: NDArray<Float32>
final t_uniform_dtype = uniform([2], dtype: DType.float32);

// expect: NDArray<Float32>
final t_uniform_out = uniform([2], out: f32);

// expect: NDArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_uniform_default = uniform([2]);

// expect: NDArray<Float64>
final t_hanning_out = hanning(5, out: f64);

// expect: NDArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_hanning_dyn_dtype = hanning(5, dtype: DType.float64.asAnySpec);
