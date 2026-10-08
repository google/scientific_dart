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
/// This library provides [MaskedArray] and utility functions to work with
/// arrays that have missing or invalid data.
library;

import 'package:ndarray/ndarray.dart';
import 'src/masked_array.dart';

export 'src/masked_array.dart';

// ==========================================
// Top-Level Factories
// ==========================================

/// Creates a [MaskedArray] automatically masking NaN and infinite values in [data].
///
/// Elements in [data] that are `NaN` or `Infinity` (positive or negative) will
/// have their corresponding mask elements set to `true`.
///
/// Preconditions:
/// - [data] must be of a numeric type (Float32, Float64, Complex64, Complex128)
///   to have NaN/Inf values. For other types, this is equivalent to creating
///   a MaskedArray with an all-false mask.
///
/// Example:
/// ```dart
/// final data = NDArray.fromList([1.0, double.nan, 3.0], [3], DType.float64);
/// final marr = maskedInvalid(data);
/// print(marr.mask.toList()); // [false, true, false]
/// ```
MaskedArray<T> maskedInvalid<T extends DTypeTag>(
  NDArray<T> data, {
  Object? fillValue,
}) => MaskedArray.maskedInvalid(data, fillValue: fillValue);

/// Creates a [MaskedArray] automatically masking elements in [data] equal to [value].
///
/// Preconditions:
/// - [value] must be compatible with the [data]'s [DType].
///
/// Example:
/// ```dart
/// final data = NDArray.fromList([1, 2, 3, 2], [4], DType.int32);
/// final marr = maskedEqual(data, 2);
/// print(marr.mask.toList()); // [false, true, false, true]
/// ```
MaskedArray<T> maskedEqual<T extends DTypeTag>(
  NDArray<T> data,
  Object value, {
  Object? fillValue,
}) => MaskedArray.maskedEqual(data, value, fillValue: fillValue);

/// Creates a [MaskedArray] automatically masking elements in [data] greater than [value].
///
/// Preconditions:
/// - [value] must be compatible with the [data]'s [DType].
/// - [data] DType must support comparison operators.
MaskedArray<T> maskedGreater<T extends DTypeTag>(
  NDArray<T> data,
  Object value, {
  Object? fillValue,
}) => MaskedArray.maskedGreater(data, value, fillValue: fillValue);

/// Creates a [MaskedArray] automatically masking elements in [data] greater than or equal to [value].
///
/// Preconditions:
/// - [value] must be compatible with the [data]'s [DType].
/// - [data] DType must support comparison operators.
MaskedArray<T> maskedGreaterEqual<T extends DTypeTag>(
  NDArray<T> data,
  Object value, {
  Object? fillValue,
}) => MaskedArray.maskedGreaterEqual(data, value, fillValue: fillValue);

/// Creates a [MaskedArray] automatically masking elements in [data] less than [value].
///
/// Preconditions:
/// - [value] must be compatible with the [data]'s [DType].
/// - [data] DType must support comparison operators.
MaskedArray<T> maskedLess<T extends DTypeTag>(
  NDArray<T> data,
  Object value, {
  Object? fillValue,
}) => MaskedArray.maskedLess(data, value, fillValue: fillValue);

/// Creates a [MaskedArray] automatically masking elements in [data] less than or equal to [value].
///
/// Preconditions:
/// - [value] must be compatible with the [data]'s [DType].
/// - [data] DType must support comparison operators.
MaskedArray<T> maskedLessEqual<T extends DTypeTag>(
  NDArray<T> data,
  Object value, {
  Object? fillValue,
}) => MaskedArray.maskedLessEqual(data, value, fillValue: fillValue);

// ==========================================
// Top-Level Reductions
// ==========================================

/// Returns the sum of [a] elements along the given [axis], ignoring masked elements.
///
/// If [axis] is null, returns a 0-dimensional [MaskedArray] containing the sum of all elements.
/// Masked elements are treated as `0` during the sum.
/// The output mask is `true` only if all elements along the reduction axis are masked.
MaskedArray<R> sum<
  T extends DTypeSpec<
    DTypeTag,
    Object?,
    DTypeTag,
    DTypeTag,
    DTypeTag,
    R,
    DTypeTag,
    DTypeTag
  >,
  R extends DTypeTag
>(MaskedArray<T> a, {int? axis}) => a.sum(axis: axis);

/// Returns the product of [a] elements along the given [axis], ignoring masked elements.
///
/// If [axis] is null, returns a 0-dimensional [MaskedArray] containing the product of all elements.
/// Masked elements are treated as `1` during the product.
/// The output mask is `true` only if all elements along the reduction axis are masked.
MaskedArray<R> prod<
  T extends DTypeSpec<
    DTypeTag,
    Object?,
    DTypeTag,
    DTypeTag,
    DTypeTag,
    R,
    DTypeTag,
    DTypeTag
  >,
  R extends DTypeTag
>(MaskedArray<T> a, {int? axis}) => a.prod(axis: axis);

/// Returns the minimum of [a] elements along the given [axis], ignoring masked elements.
///
/// If [axis] is null, returns a 0-dimensional [MaskedArray] containing the minimum of all elements.
/// Masked elements are treated as the maximum value for the [DType] during the reduction.
/// The output mask is `true` only if all elements along the reduction axis are masked.
MaskedArray<T> min<T extends DTypeTag>(MaskedArray<T> a, {int? axis}) =>
    a.min(axis: axis);

/// Returns the maximum of [a] elements along the given [axis], ignoring masked elements.
///
/// If [axis] is null, returns a 0-dimensional [MaskedArray] containing the maximum of all elements.
/// Masked elements are treated as the minimum value for the [DType] during the reduction.
/// The output mask is `true` only if all elements along the reduction axis are masked.
MaskedArray<T> max<T extends DTypeTag>(MaskedArray<T> a, {int? axis}) =>
    a.max(axis: axis);

/// Returns the mean of [a] elements along the given [axis], ignoring masked elements.
///
/// Calculated as `sum(a, axis) / count(a, axis)`.
/// Returns a new [MaskedArray] with [DType.float64] (or [DType.complex128] if input is complex).
/// The output mask is `true` only if all elements along the reduction axis are masked.
MaskedArray<D> mean<
  T extends DTypeSpec<
    DTypeTag,
    Object?,
    DTypeTag,
    DTypeTag,
    DTypeTag,
    DTypeTag,
    D,
    DTypeTag
  >,
  D extends DTypeTag
>(MaskedArray<T> a, {int? axis}) => a.mean(axis: axis);

/// Returns the variance of [a] elements along the given [axis], ignoring masked elements.
///
/// Calculated as `mean(|x - mean|^2)`.
/// Returns a new [MaskedArray] with [DType.float64].
/// The output mask is `true` only if all elements along the reduction axis are masked.
MaskedArray<Float64> variance(MaskedArray a, {int? axis}) =>
    a.variance(axis: axis);

/// Returns the standard deviation of [a] elements along the given [axis], ignoring masked elements.
///
/// Calculated as `sqrt(variance(a, axis))`.
/// Returns a new [MaskedArray] with [DType.float64].
/// The output mask is `true` only if all elements along the reduction axis are masked.
MaskedArray<Float64> std(MaskedArray a, {int? axis}) => a.std(axis: axis);

/// Returns the count of unmasked (valid) elements in [a] along the given [axis].
///
/// Returns a standard [NDArray<Int64>] containing the counts.
NDArray<Int64> count(MaskedArray a, {int? axis}) => a.count(axis: axis);

// ==========================================
// Top-Level Arithmetic
// ==========================================

/// Performs element-wise addition of [a] and [b], propagating masks.
MaskedArray<T> add<T extends DTypeTag>(MaskedArray<T> a, Object? b) => a.add(b);

/// Performs element-wise subtraction of [a] and [b], propagating masks.
MaskedArray<T> subtract<T extends DTypeTag>(MaskedArray<T> a, Object? b) =>
    a.subtract(b);

/// Performs element-wise multiplication of [a] and [b], propagating masks.
MaskedArray<T> multiply<T extends DTypeTag>(MaskedArray<T> a, Object? b) =>
    a.multiply(b);

/// Performs element-wise floor division of [a] and [b], propagating masks.
MaskedArray<T> floorDivide<T extends DTypeTag>(MaskedArray<T> a, Object? b) =>
    a.floorDivide(b);

/// Performs element-wise remainder of [a] and [b], propagating masks.
MaskedArray<T> remainder<T extends DTypeTag>(MaskedArray<T> a, Object? b) =>
    a.remainder(b);

/// Performs element-wise true division of [a] and [b], propagating masks.
///
/// Elements where the divisor [b] is zero are automatically masked in the result.
MaskedArray<M> divide<
  T extends DTypeSpec<
    DTypeTag,
    Object?,
    DTypeTag,
    DTypeTag,
    DTypeTag,
    DTypeTag,
    DTypeTag,
    M
  >,
  M extends DTypeTag
>(MaskedArray<T> a, Object? b) => a.divide(b);
