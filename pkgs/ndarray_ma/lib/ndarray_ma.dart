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

/// Masked arrays for the `ndarray` package.
///
/// This library provides [MaskedArray] and top-level operations for arrays
/// with missing or invalid data, using the same static dtype projection
/// (`*Of<R>`), same-dtype (`SelfOf<R>`), capability marker ([BitwiseDType],
/// [IntegerDType]), and destination-buffer (`Out extends R`) bounds as
/// `package:ndarray` `0.3.0`.
library;

import 'package:ndarray/ndarray.dart';
import 'src/masked_array.dart';

export 'package:ndarray/ndarray.dart'
    show
        AccumulatorOf,
        AnyBitwiseSpec,
        AnyIntegerSpec,
        AnySpec,
        BFloat16,
        BitwiseDType,
        Boolean,
        Complex,
        Complex128,
        Complex64,
        ComplexOf,
        DType,
        DTypeSpec,
        DTypeTag,
        DivideOf,
        DivideSpec,
        DoublePrecisionOf,
        DoublePrecisionSpec,
        ElementOf,
        Float16,
        Float32,
        Float64,
        Index,
        InexactOf,
        InexactSpec,
        Int16,
        Int32,
        Int64,
        Int8,
        IntegerDType,
        NDArray,
        NDArrayBaseDivide,
        NDArrayBaseElements,
        NDArrayBitwise,
        NDArrayBitwiseSpec,
        NDArrayDivide,
        NDArrayElements,
        NDArrayShift,
        NDArrayShiftSpec,
        RealFloatOf,
        RealOf,
        Selector,
        SelfOf,
        Slice,
        Uint16,
        Uint32,
        Uint64,
        Uint8;
export 'src/masked_array.dart'
    hide maAtan2, maConcatenate, maDivide, maHypot, maWhere;

// ==========================================
// Top-Level Factories & Inspection
// ==========================================

/// Creates a [MaskedArray] automatically masking `NaN` and infinite values in
/// [data].
///
/// Elements in [data] that are `NaN` or `Infinity` (positive or negative) have
/// their corresponding mask elements set to `true`.
MaskedArray<T> maskedInvalid<T extends DTypeTag>(
  NDArray<T> data, {
  Object? fillValue,
}) => MaskedArray.maskedInvalid(data, fillValue: fillValue);

/// Creates a [MaskedArray] automatically masking elements in [data] equal to
/// [value].
MaskedArray<T> maskedEqual<T extends DTypeTag>(
  NDArray<T> data,
  Object value, {
  Object? fillValue,
}) => MaskedArray.maskedEqual(data, value, fillValue: fillValue);

/// Creates a [MaskedArray] automatically masking elements in [data] not equal
/// to [value].
MaskedArray<T> maskedNotEqual<T extends DTypeTag>(
  NDArray<T> data,
  Object value, {
  Object? fillValue,
}) => MaskedArray.maskedNotEqual(data, value, fillValue: fillValue);

/// Creates a [MaskedArray] automatically masking elements in [data] greater
/// than [value].
MaskedArray<T> maskedGreater<T extends DTypeTag>(
  NDArray<T> data,
  Object value, {
  Object? fillValue,
}) => MaskedArray.maskedGreater(data, value, fillValue: fillValue);

/// Creates a [MaskedArray] automatically masking elements in [data] greater
/// than or equal to [value].
MaskedArray<T> maskedGreaterEqual<T extends DTypeTag>(
  NDArray<T> data,
  Object value, {
  Object? fillValue,
}) => MaskedArray.maskedGreaterEqual(data, value, fillValue: fillValue);

/// Creates a [MaskedArray] automatically masking elements in [data] less than
/// [value].
MaskedArray<T> maskedLess<T extends DTypeTag>(
  NDArray<T> data,
  Object value, {
  Object? fillValue,
}) => MaskedArray.maskedLess(data, value, fillValue: fillValue);

/// Creates a [MaskedArray] automatically masking elements in [data] less than
/// or equal to [value].
MaskedArray<T> maskedLessEqual<T extends DTypeTag>(
  NDArray<T> data,
  Object value, {
  Object? fillValue,
}) => MaskedArray.maskedLessEqual(data, value, fillValue: fillValue);

/// Creates a [MaskedArray] masking elements in [data] inside the closed
/// interval `[v1, v2]` (or `[v2, v1]` if `v2 < v1`).
MaskedArray<T> maskedInside<T extends DTypeTag>(
  NDArray<T> data,
  num v1,
  num v2, {
  Object? fillValue,
}) => MaskedArray.maskedInside(data, v1, v2, fillValue: fillValue);

/// Creates a [MaskedArray] masking elements in [data] outside the closed
/// interval `[v1, v2]` (or `[v2, v1]` if `v2 < v1`).
MaskedArray<T> maskedOutside<T extends DTypeTag>(
  NDArray<T> data,
  num v1,
  num v2, {
  Object? fillValue,
}) => MaskedArray.maskedOutside(data, v1, v2, fillValue: fillValue);

/// Creates a [MaskedArray] masking elements in [data] where [condition] is
/// `true`.
MaskedArray<T> maskedWhere<T extends DTypeTag>(
  NDArray<Boolean> condition,
  NDArray<T> data, {
  Object? fillValue,
}) => MaskedArray.maskedWhere(condition, data, fillValue: fillValue);

/// Returns the underlying [NDArray] data of [a].
NDArray<T> getdata<T extends DTypeTag>(MaskedArray<T> a) => a.data;

/// Returns the boolean mask [NDArray] of [a].
NDArray<Boolean> getmaskarray(MaskedArray<DTypeTag> a) => a.mask;

/// Returns a standard [NDArray] with masked elements of [a] replaced by
/// [fillValue] (or `a.fillValue`), writing into [out] if provided.
NDArray<T> filled<T extends DTypeTag, Out extends T>(
  MaskedArray<T> a, {
  Object? fillValue,
  NDArray<Out>? out,
}) => a.filled(fillValue: fillValue, out: out);

/// Returns a 1D standard [NDArray] containing all unmasked elements of [a].
NDArray<T> compressed<T extends DTypeTag>(MaskedArray<T> a) => a.compressed();

// ==========================================
// Top-Level Reductions
// ==========================================

/// Returns the sum of [a] elements along the given [axis], ignoring masked
/// elements.
///
/// If [axis] is null, returns a 0-dimensional [MaskedArray] containing the sum
/// of all elements. Masked elements are treated as `0` during the sum. The
/// output mask is `true` only if all elements along the reduction axis are
/// masked.
MaskedArray<R>
sum<T extends AccumulatorOf<R>, R extends DTypeTag, Out extends R>(
  MaskedArray<T> a, {
  int? axis,
  MaskedArray<Out>? out,
}) => a.sum<Out>(axis: axis, out: out);

/// Returns the product of [a] elements along the given [axis], ignoring masked
/// elements.
///
/// If [axis] is null, returns a 0-dimensional [MaskedArray] containing the
/// product of all elements. Masked elements are treated as `1` during the
/// product. The output mask is `true` only if all elements along the reduction
/// axis are masked.
MaskedArray<R>
prod<T extends AccumulatorOf<R>, R extends DTypeTag, Out extends R>(
  MaskedArray<T> a, {
  int? axis,
  MaskedArray<Out>? out,
}) => a.prod<Out>(axis: axis, out: out);

/// Returns the minimum of [a] elements along the given [axis], ignoring masked
/// elements.
///
/// If [axis] is null, returns a 0-dimensional [MaskedArray] containing the
/// minimum of all elements. The output mask is `true` only if all elements
/// along the reduction axis are masked.
MaskedArray<T> min<T extends DTypeTag, Out extends T>(
  MaskedArray<T> a, {
  int? axis,
  MaskedArray<Out>? out,
}) => a.min(axis: axis, out: out);

/// Returns the maximum of [a] elements along the given [axis], ignoring masked
/// elements.
///
/// If [axis] is null, returns a 0-dimensional [MaskedArray] containing the
/// maximum of all elements. The output mask is `true` only if all elements
/// along the reduction axis are masked.
MaskedArray<T> max<T extends DTypeTag, Out extends T>(
  MaskedArray<T> a, {
  int? axis,
  MaskedArray<Out>? out,
}) => a.max(axis: axis, out: out);

/// Returns the mean of [a] elements along the given [axis], ignoring masked
/// elements.
///
/// Calculated as `sum(a, axis) / count(a, axis)`. Returns a [MaskedArray] with
/// [DType.float64] (or [DType.complex128] if input is complex). The output
/// mask is `true` only if all elements along the reduction axis are masked.
MaskedArray<D>
mean<T extends DoublePrecisionOf<D>, D extends DTypeTag, Out extends D>(
  MaskedArray<T> a, {
  int? axis,
  MaskedArray<Out>? out,
}) => a.mean<Out>(axis: axis, out: out);

/// Returns the variance of [a] elements along the given [axis], ignoring
/// masked elements.
///
/// Calculated as `mean(|x - mean|^2)` in [DType.float64]. The output mask is
/// `true` only if all elements along the reduction axis are masked.
MaskedArray<Float64> variance(
  MaskedArray<DTypeTag> a, {
  int? axis,
  MaskedArray<Float64>? out,
}) => a.variance(axis: axis, out: out);

/// Returns the standard deviation of [a] elements along the given [axis],
/// ignoring masked elements.
///
/// Calculated as `sqrt(variance(a, axis))` in [DType.float64]. The output mask
/// is `true` only if all elements along the reduction axis are masked.
MaskedArray<Float64> std(
  MaskedArray<DTypeTag> a, {
  int? axis,
  MaskedArray<Float64>? out,
}) => a.std(axis: axis, out: out);

/// Returns the count of unmasked (valid) elements in [a] along the given
/// [axis] as an [NDArray] of [DType.int64].
NDArray<Int64> count(
  MaskedArray<DTypeTag> a, {
  int? axis,
  NDArray<Int64>? out,
}) => a.count(axis: axis, out: out);

// ==========================================
// Top-Level Same-DType Binary & Multi-Array Operations
// ==========================================
//
// Every top-level function that binds two or more input MaskedArrays to one
// type parameter bounds it by `SelfOf<...>`, so mixed-dtype calls such as
// `add(m64, m32)` are rejected at compile time while same-dtype calls infer
// `T` without type arguments. Scalars and unmasked NDArrays are accepted by
// the instance methods and operators (`a.add(b)`, `a + b`); run-time-typed
// arrays widen with `MaskedArray.asAnySpec`, `MaskedArray.asBitwiseDType`, or
// `MaskedArray.asIntegerDType`.

/// Performs element-wise addition of [a] and [b], propagating masks.
///
/// Both operands must have the same dtype; use `a.add(other)` or `a + other`
/// when combining a [MaskedArray] with an [NDArray] or scalar.
MaskedArray<T> add<T extends SelfOf<DTypeTag>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<T>? out,
}) => a.add(b, out: out);

/// Performs element-wise subtraction of [b] from [a], propagating masks.
///
/// Both operands must have the same dtype; use `a.subtract(other)` or
/// `a - other` when combining a [MaskedArray] with an [NDArray] or scalar.
MaskedArray<T> subtract<T extends SelfOf<DTypeTag>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<T>? out,
}) => a.subtract(b, out: out);

/// Performs element-wise multiplication of [a] and [b], propagating masks.
///
/// Both operands must have the same dtype; use `a.multiply(other)` or
/// `a * other` when combining a [MaskedArray] with an [NDArray] or scalar.
MaskedArray<T> multiply<T extends SelfOf<DTypeTag>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<T>? out,
}) => a.multiply(b, out: out);

/// Performs element-wise floor division of [a] and [b], propagating masks and
/// masking zero divisors.
///
/// Both operands must have the same dtype; use `a.floorDivide(other)` or
/// `a ~/ other` when combining a [MaskedArray] with an [NDArray] or scalar.
MaskedArray<T> floorDivide<T extends SelfOf<DTypeTag>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<T>? out,
}) => a.floorDivide(b, out: out);

/// Performs element-wise remainder of [a] and [b], propagating masks and
/// masking zero divisors.
///
/// Both operands must have the same dtype; use `a.remainder(other)` or
/// `a % other` when combining a [MaskedArray] with an [NDArray] or scalar.
MaskedArray<T> remainder<T extends SelfOf<DTypeTag>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<T>? out,
}) => a.remainder(b, out: out);

/// Performs element-wise true division of [a] and [b], propagating masks and
/// masking zero divisors.
///
/// Both operands must have the same dtype; integer and boolean inputs promote
/// to [Float64] ([DivideOf]). Use `a.divide(other)` or `a / other` when
/// combining a [MaskedArray] with an [NDArray] or scalar.
MaskedArray<M>
divide<T extends SelfOf<DivideOf<M>>, M extends DTypeTag, Out extends M>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<Out>? out,
}) => maDivide<T, M>(a, b, out: out);

/// Performs element-wise minimum of [a] and [b], propagating masks.
MaskedArray<T> minimum<T extends SelfOf<DTypeTag>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<T>? out,
}) => a.minimum(b, out: out);

/// Performs element-wise maximum of [a] and [b], propagating masks.
MaskedArray<T> maximum<T extends SelfOf<DTypeTag>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<T>? out,
}) => a.maximum(b, out: out);

/// Performs element-wise exponentiation ($a^b$), propagating masks.
MaskedArray<T> power<T extends SelfOf<DTypeTag>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<T>? out,
}) => a.power(b, out: out);

/// Computes the element-wise four-quadrant inverse tangent of [y] and [x],
/// propagating masks.
MaskedArray<R>
atan2<T extends SelfOf<InexactOf<R>>, R extends DTypeTag, Out extends R>(
  MaskedArray<T> y,
  MaskedArray<T> x, {
  MaskedArray<Out>? out,
}) => maAtan2<T, R>(y, x, out: out);

/// Computes the element-wise Euclidean hypotenuse of [x1] and [x2],
/// propagating masks.
MaskedArray<R>
hypot<T extends SelfOf<InexactOf<R>>, R extends DTypeTag, Out extends R>(
  MaskedArray<T> x1,
  MaskedArray<T> x2, {
  MaskedArray<Out>? out,
}) => maHypot<T, R>(x1, x2, out: out);

/// Performs element-wise equality comparison (`a == b`), propagating masks.
///
/// Both operands must have the same dtype; use `a.eq(other)` when comparing
/// against an [NDArray] or scalar.
MaskedArray<Boolean> equal<T extends SelfOf<DTypeTag>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<Boolean>? out,
}) => a.eq(b, out: out);

/// Performs element-wise inequality comparison (`a != b`), propagating masks.
///
/// Both operands must have the same dtype; use `a.ne(other)` when comparing
/// against an [NDArray] or scalar.
MaskedArray<Boolean> notEqual<T extends SelfOf<DTypeTag>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<Boolean>? out,
}) => a.ne(b, out: out);

/// Performs element-wise greater-than comparison (`a > b`), propagating masks.
///
/// Both operands must have the same dtype; use `a > other` or `a.gt(other)`
/// when comparing against an [NDArray] or scalar.
MaskedArray<Boolean> greater<T extends SelfOf<DTypeTag>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<Boolean>? out,
}) => a.gt(b, out: out);

/// Performs element-wise greater-than-or-equal comparison (`a >= b`),
/// propagating masks.
MaskedArray<Boolean> greaterEqual<T extends SelfOf<DTypeTag>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<Boolean>? out,
}) => a.ge(b, out: out);

/// Performs element-wise less-than comparison (`a < b`), propagating masks.
MaskedArray<Boolean> less<T extends SelfOf<DTypeTag>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<Boolean>? out,
}) => a.lt(b, out: out);

/// Performs element-wise less-than-or-equal comparison (`a <= b`),
/// propagating masks.
MaskedArray<Boolean> lessEqual<T extends SelfOf<DTypeTag>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<Boolean>? out,
}) => a.le(b, out: out);

/// Performs element-wise bitwise AND (`&`), propagating masks.
///
/// Both operands must have the same integer or boolean dtype.
MaskedArray<T> bitwiseAnd<T extends SelfOf<RealOf<BitwiseDType>>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<T>? out,
}) =>
    a.asBitwiseDType.bitwiseAnd(b, out: out?.asBitwiseDType) as MaskedArray<T>;

/// Performs element-wise bitwise OR (`|`), propagating masks.
///
/// Both operands must have the same integer or boolean dtype.
MaskedArray<T> bitwiseOr<T extends SelfOf<RealOf<BitwiseDType>>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<T>? out,
}) => a.asBitwiseDType.bitwiseOr(b, out: out?.asBitwiseDType) as MaskedArray<T>;

/// Performs element-wise bitwise XOR (`^`), propagating masks.
///
/// Both operands must have the same integer or boolean dtype.
MaskedArray<T> bitwiseXor<T extends SelfOf<RealOf<BitwiseDType>>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<T>? out,
}) =>
    a.asBitwiseDType.bitwiseXor(b, out: out?.asBitwiseDType) as MaskedArray<T>;

/// Performs element-wise bitwise NOT / inversion (`~a`), preserving the mask.
///
/// The input [a] must have an integer or boolean dtype.
MaskedArray<T> invert<T extends RealOf<BitwiseDType>, Out extends T>(
  MaskedArray<T> a, {
  MaskedArray<Out>? out,
}) => a.asBitwiseDType.invert(out: out?.asBitwiseDType) as MaskedArray<T>;

/// Performs element-wise bitwise left shift (`<<`), propagating masks.
///
/// Both operands must have the same integer dtype.
MaskedArray<T> leftShift<T extends SelfOf<RealOf<IntegerDType>>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<T>? out,
}) => a.asIntegerDType.leftShift(b, out: out?.asIntegerDType) as MaskedArray<T>;

/// Performs element-wise bitwise right shift (`>>`), propagating masks.
///
/// Both operands must have the same integer dtype.
MaskedArray<T> rightShift<T extends SelfOf<RealOf<IntegerDType>>>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<T>? out,
}) =>
    a.asIntegerDType.rightShift(b, out: out?.asIntegerDType) as MaskedArray<T>;

/// Selects elements from [x] where [condition] is `true` and from [y]
/// elsewhere, propagating masks from [condition], [x], and [y].
///
/// Both [x] and [y] must have the same dtype.
MaskedArray<T> where<T extends SelfOf<DTypeTag>>(
  Object condition,
  MaskedArray<T> x,
  MaskedArray<T> y, {
  MaskedArray<T>? out,
}) => maWhere<T, T>(condition, x, y, out: out);

/// Concatenates [arrays] along [axis], propagating their masks.
///
/// All elements of [arrays] must have the same dtype.
MaskedArray<T> concatenate<T extends SelfOf<DTypeTag>>(
  List<MaskedArray<T>> arrays, {
  int axis = 0,
  MaskedArray<T>? out,
}) => maConcatenate<T, T>(arrays, axis: axis, out: out);

// ==========================================
// Top-Level Unary Math Operations
// ==========================================

/// Element-wise negation (`-a`), preserving the mask.
MaskedArray<T> negative<T extends DTypeTag, Out extends T>(
  MaskedArray<T> a, {
  MaskedArray<Out>? out,
}) => a.negative(out: out);

/// Element-wise absolute value (or complex magnitude), preserving the mask.
MaskedArray<R> abs<T extends RealOf<R>, R extends DTypeTag, Out extends R>(
  MaskedArray<T> a, {
  MaskedArray<Out>? out,
}) => a.abs<Out>(out: out);

/// Element-wise square root, preserving the mask.
MaskedArray<R> sqrt<T extends InexactOf<R>, R extends DTypeTag, Out extends R>(
  MaskedArray<T> a, {
  MaskedArray<Out>? out,
}) => a.sqrt<Out>(out: out);

/// Element-wise exponential ($e^a$), preserving the mask.
MaskedArray<R> exp<T extends InexactOf<R>, R extends DTypeTag, Out extends R>(
  MaskedArray<T> a, {
  MaskedArray<Out>? out,
}) => a.exp<Out>(out: out);

/// Element-wise natural logarithm ($\ln a$), preserving the mask.
MaskedArray<R> log<T extends InexactOf<R>, R extends DTypeTag, Out extends R>(
  MaskedArray<T> a, {
  MaskedArray<Out>? out,
}) => a.log<Out>(out: out);

/// Element-wise sine, preserving the mask.
MaskedArray<R> sin<T extends InexactOf<R>, R extends DTypeTag, Out extends R>(
  MaskedArray<T> a, {
  MaskedArray<Out>? out,
}) => a.sin<Out>(out: out);

/// Element-wise cosine, preserving the mask.
MaskedArray<R> cos<T extends InexactOf<R>, R extends DTypeTag, Out extends R>(
  MaskedArray<T> a, {
  MaskedArray<Out>? out,
}) => a.cos<Out>(out: out);

/// Element-wise tangent, preserving the mask.
MaskedArray<R> tan<T extends InexactOf<R>, R extends DTypeTag, Out extends R>(
  MaskedArray<T> a, {
  MaskedArray<Out>? out,
}) => a.tan<Out>(out: out);

/// Clips (limits) the unmasked values in [a] to `[aMin, aMax]`, preserving
/// the mask.
MaskedArray<T> clip<T extends DTypeTag, Out extends T>(
  MaskedArray<T> a, {
  num? aMin,
  num? aMax,
  MaskedArray<Out>? out,
}) => a.clip(aMin: aMin, aMax: aMax, out: out);
