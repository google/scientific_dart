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

library;

import 'package:ndarray/ndarray.dart';
import 'package:ndarray/ndarray.dart' as ndops;

part 'utils.dart';
part 'ops/arithmetic.dart';
part 'ops/reductions.dart';
part 'ops/views.dart';

/// An array with associated boolean mask to represent missing or invalid data.
///
/// A [MaskedArray] packages a standard [NDArray] of dtype tag [T] with a
/// boolean [NDArray] mask of the same shape. Elements where the mask is `true`
/// are considered invalid or missing, and are automatically bypassed in
/// arithmetic operations and reductions.
///
/// Operations whose result dtype depends deterministically on [T] (such as
/// [MaskedArrayAccumulatingReductions.sum], [MaskedArrayDoublePrecisionReductions.mean],
/// [MaskedArrayDivide.divide], and the typed [MaskedArrayElements.scalar]
/// accessor) are provided as extensions bounded by [DTypeSpec], so that their
/// static return types match the runtime dtype for every concrete tag.
/// Receivers typed as `MaskedArray<DTypeTag>` fall back to the untyped
/// `MaskedArrayBase*` extensions.
final class MaskedArray<T extends DTypeTag> {
  /// The underlying data array containing all values (both valid and masked).
  final NDArray<T> data;

  /// The boolean mask array of the same shape as [data].
  ///
  /// A value of `true` indicates that the corresponding element in [data] is
  /// invalid or missing (masked).
  final NDArray<Boolean> mask;

  /// The value used to fill masked elements when converting to a standard [NDArray].
  final Object fillValue;

  /// Creates a [MaskedArray] view wrapping [data] and [mask].
  ///
  /// Preconditions:
  /// - The shape of [data] and [mask] must be identical.
  ///
  /// Throws:
  /// - [ArgumentError] if [data] and [mask] shapes do not match.
  MaskedArray(this.data, this.mask, {Object? fillValue})
    : fillValue = fillValue ?? _defaultFillValue(data.dtype) {
    if (!data.hasSameShape(mask)) {
      throw ArgumentError.value(
        mask,
        'mask',
        'Must have the same shape as data (${data.shape}), '
            'but has shape ${mask.shape}.',
      );
    }
  }

  /// Creates a [MaskedArray] wrapping [data] with every element unmasked.
  ///
  /// The mask is a newly allocated all-`false` array of the same shape as
  /// [data].
  factory MaskedArray.unmasked(NDArray<T> data, {Object? fillValue}) =>
      MaskedArray(
        data,
        NDArray<Boolean>.zeros(data.shape, DType.boolean),
        fillValue: fillValue,
      );

  /// The shape (dimensions) of the array.
  List<int> get shape => data.shape;

  /// The rank (number of dimensions) of the array.
  int get rank => data.rank;

  /// The total number of elements in the array.
  int get size => data.size;

  /// The data type of the elements.
  DType<T> get dtype => data.dtype;

  /// Creates a [MaskedArray] with all elements unmasked and initialized to zero.
  factory MaskedArray.zeros(
    List<int> shape,
    DType<T> dtype, {
    Object? fillValue,
  }) {
    final data = NDArray<T>.zeros(shape, dtype);
    final mask = NDArray<Boolean>.zeros(shape, DType.boolean);
    return MaskedArray(data, mask, fillValue: fillValue);
  }

  /// Creates a [MaskedArray] with all elements unmasked and initialized to one.
  factory MaskedArray.ones(
    List<int> shape,
    DType<T> dtype, {
    Object? fillValue,
  }) {
    final data = NDArray<T>.ones(shape, dtype);
    final mask = NDArray<Boolean>.zeros(shape, DType.boolean);
    return MaskedArray(data, mask, fillValue: fillValue);
  }

  /// Creates a [MaskedArray] automatically masking NaN and infinite values in [data].
  ///
  /// Elements in [data] that are `NaN` or `Infinity` will be masked (`mask` set to `true`).
  /// This is only relevant for numeric float/complex types.
  factory MaskedArray.maskedInvalid(NDArray<T> data, {Object? fillValue}) {
    return NDArray.scope(() {
      final nanMask = ndops.isnan(data);
      final infMask = ndops.isinf(data);
      final mask = ndops.logicalOr(nanMask, infMask);
      return MaskedArray(
        data,
        mask.detachToParentScope(),
        fillValue: fillValue,
      );
    });
  }

  /// Creates a [MaskedArray] automatically masking elements in [data] equal to [value].
  factory MaskedArray.maskedEqual(
    NDArray<T> data,
    Object value, {
    Object? fillValue,
  }) {
    return NDArray.scope(() {
      final valArray = _wrapScalar<T>(value, data.dtype);
      final mask = ndops.equal(data, valArray);
      return MaskedArray(
        data,
        mask.detachToParentScope(),
        fillValue: fillValue,
      );
    });
  }

  /// Creates a [MaskedArray] automatically masking elements in [data] greater than [value].
  factory MaskedArray.maskedGreater(
    NDArray<T> data,
    Object value, {
    Object? fillValue,
  }) {
    return NDArray.scope(() {
      final mask = data > value;
      return MaskedArray(
        data,
        mask.detachToParentScope(),
        fillValue: fillValue,
      );
    });
  }

  /// Creates a [MaskedArray] automatically masking elements in [data] greater than or equal to [value].
  factory MaskedArray.maskedGreaterEqual(
    NDArray<T> data,
    Object value, {
    Object? fillValue,
  }) {
    return NDArray.scope(() {
      final mask = data >= value;
      return MaskedArray(
        data,
        mask.detachToParentScope(),
        fillValue: fillValue,
      );
    });
  }

  /// Creates a [MaskedArray] automatically masking elements in [data] less than [value].
  factory MaskedArray.maskedLess(
    NDArray<T> data,
    Object value, {
    Object? fillValue,
  }) {
    return NDArray.scope(() {
      final mask = data < value;
      return MaskedArray(
        data,
        mask.detachToParentScope(),
        fillValue: fillValue,
      );
    });
  }

  /// Creates a [MaskedArray] automatically masking elements in [data] less than or equal to [value].
  factory MaskedArray.maskedLessEqual(
    NDArray<T> data,
    Object value, {
    Object? fillValue,
  }) {
    return NDArray.scope(() {
      final mask = data <= value;
      return MaskedArray(
        data,
        mask.detachToParentScope(),
        fillValue: fillValue,
      );
    });
  }

  /// The single value of a 0-dimensional [MaskedArray], untyped.
  ///
  /// Returns `null` if the element is masked. Prefer the typed
  /// [MaskedArrayElements.scalar] getter, which recovers the Dart element type
  /// from [T]; this accessor is the dtype-agnostic fallback.
  ///
  /// It is an error if this array is not 0-dimensional.
  Object? get scalarRaw {
    if (rank != 0) {
      throw StateError('scalar getter is only valid for 0-dimensional arrays');
    }
    return mask.scalar ? null : (data as NDArray<DTypeTag>).scalar;
  }

  /// The element at the given multi-dimensional [coords], untyped.
  ///
  /// Returns `null` if the element is masked. Prefer the typed
  /// [MaskedArrayElements.getCell]; this accessor is the dtype-agnostic
  /// fallback.
  ///
  /// It is an error if [coords] has a different length than [rank], or if
  /// any coordinate is out of range.
  Object? getCellUntyped(List<int> coords) =>
      mask.getCell(coords) ? null : (data as NDArray<DTypeTag>).getCell(coords);

  /// Element access and slicing.
  ///
  /// If [spec] represents coordinates (e.g., [int] for 1D, [List<int>] for ND):
  /// - Returns the scalar value [T] at that coordinate, or `null` if it is masked.
  ///
  /// If [spec] represents a slice (e.g., [Slice], [List<Slice>], [List<Selector>]):
  /// - Returns a new [MaskedArray] view representing the sliced portion of [data] and [mask].
  ///
  /// Throws:
  /// - [ArgumentError] if [spec] is not a valid coordinate or selector.
  dynamic operator [](dynamic spec) {
    final isCoords =
        (spec is int && rank == 1) ||
        (spec is List<int>) ||
        (spec is List && spec.every((e) => e is int));

    if (isCoords) {
      final List<int> coords;
      if (spec is int) {
        coords = [spec];
      } else if (spec is List<int>) {
        coords = spec;
      } else {
        coords = (spec as List).cast<int>();
      }
      return getCellUntyped(coords);
    } else {
      // Slicing
      final List<Selector> selectors;
      if (spec is Slice) {
        selectors = [spec];
      } else if (spec is List<Slice>) {
        selectors = spec;
      } else if (spec is List<Selector>) {
        selectors = spec;
      } else if (spec is List) {
        selectors = spec.map((e) {
          if (e is int) return Index(e);
          if (e is Selector) return e;
          throw ArgumentError.value(
            spec,
            'spec',
            'Must contain only int or Selector entries, but found $e.',
          );
        }).toList();
      } else if (spec is int) {
        selectors = [Index(spec)];
      } else {
        throw ArgumentError.value(
          spec,
          'spec',
          'Must be an int, a List<int>, a Slice, or a List of Selectors.',
        );
      }

      return MaskedArray(
        data.slice(selectors),
        mask.slice(selectors),
        fillValue: fillValue,
      );
    }
  }

  /// Element and slice assignment.
  ///
  /// If [spec] represents coordinates:
  /// - If [value] is `null`, masks the element.
  /// - If [value] is non-null [T], sets the value in [data] and unmasks it.
  ///
  /// If [spec] represents a slice:
  /// - If [value] is `null`, masks all elements in the slice.
  /// - If [value] is [MaskedArray], copies values and masks from it (broadcasting if needed).
  /// - If [value] is [NDArray], copies values from it and unmasks all elements in the slice.
  /// - If [value] is scalar [T], fills the slice with [value] and unmasks it.
  ///
  /// Throws:
  /// - [ArgumentError] if [spec] is invalid or [value] type is unsupported.
  void operator []=(dynamic spec, dynamic value) {
    final isCoords =
        (spec is int && rank == 1) ||
        (spec is List<int>) ||
        (spec is List && spec.every((e) => e is int));

    if (isCoords) {
      final List<int> coords;
      if (spec is int) {
        coords = [spec];
      } else if (spec is List<int>) {
        coords = spec;
      } else {
        coords = (spec as List).cast<int>();
      }

      if (value == null) {
        mask.setCell(coords, true);
      } else {
        data.setCell(coords, value as Object);
        mask.setCell(coords, false);
      }
    } else {
      // Slicing
      final List<Selector> selectors;
      if (spec is Slice) {
        selectors = [spec];
      } else if (spec is List<Slice>) {
        selectors = spec;
      } else if (spec is List<Selector>) {
        selectors = spec;
      } else if (spec is List) {
        selectors = spec.map((e) {
          if (e is int) return Index(e);
          if (e is Selector) return e;
          throw ArgumentError.value(
            spec,
            'spec',
            'Must contain only int or Selector entries, but found $e.',
          );
        }).toList();
      } else if (spec is int) {
        selectors = [Index(spec)];
      } else {
        throw ArgumentError.value(
          spec,
          'spec',
          'Must be an int, a List<int>, a Slice, or a List of Selectors.',
        );
      }

      final dataView = data.slice(selectors);
      final maskView = mask.slice(selectors);

      if (value == null) {
        maskView.fill(true);
      } else if (value is MaskedArray<T>) {
        NDArray.scope(() {
          final broadcastedData = ndops.broadcastTo(value.data, dataView.shape);
          final broadcastedMask = ndops.broadcastTo(value.mask, maskView.shape);
          broadcastedData.copy(out: dataView);
          broadcastedMask.copy(out: maskView);
        });
      } else if (value is NDArray<T>) {
        NDArray.scope(() {
          final broadcastedData = ndops.broadcastTo(value, dataView.shape);
          broadcastedData.copy(out: dataView);
          maskView.fill(false);
        });
      } else if (value is num || value is Complex || value is bool) {
        dataView.fill(value as Object);
        maskView.fill(false);
      } else {
        throw ArgumentError.value(
          value,
          'value',
          'Must be null, a MaskedArray<$T>, an NDArray<$T>, or a scalar.',
        );
      }
    }
  }

  /// Detaches this array's components (data and mask) from the current automatic disposal scope.
  MaskedArray<T> detachFromScope() {
    data.detachFromScope();
    mask.detachFromScope();
    return this;
  }

  /// Detaches this array's components from the current scope and promotes them to the parent scope.
  MaskedArray<T> detachToParentScope() {
    data.detachToParentScope();
    mask.detachToParentScope();
    return this;
  }

  // ==========================================
  // Conversions
  // ==========================================

  /// Returns a copy of this array with [data] converted to [dtype].
  ///
  /// The mask is copied, and [fillValue] is coerced to the new dtype (falling
  /// back to the default fill value of [dtype] when it cannot be represented).
  ///
  /// **Performance considerations:**
  /// - Time complexity: $O(n)$; allocates a new data array and a new mask.
  MaskedArray<R> astype<R extends DTypeTag>(DType<R> dtype) {
    return NDArray.scope(() {
      final converted = data.astype(dtype);
      final maskCopy = mask.copy();
      return dispatchCreateMaskedArray(
            converted.detachToParentScope(),
            maskCopy.detachToParentScope(),
            fillValue: fillValue,
          )
          as MaskedArray<R>;
    });
  }

  // ==========================================
  // Arithmetic (dtype-preserving)
  // ==========================================
  //
  // True division (`divide`, `/`) follows NumPy's `true_divide` rule and is
  // therefore provided by [MaskedArrayDivide] / [MaskedArrayBaseDivide].

  /// Element-wise addition, propagating masks.
  ///
  /// A scalar [other] is converted to this array's dtype. A [MaskedArray] or
  /// [NDArray] operand must have the same dtype as this array; cast explicitly
  /// with [astype] before combining arrays of different dtypes. The result
  /// mask is the logical OR of both operand masks.
  ///
  /// It is an error if [other] is an array of a different dtype, a scalar
  /// that cannot be represented in this array's dtype, or not broadcastable
  /// against this array.
  MaskedArray<T> add(Object? other) => _maAdd<T>(this, other);

  /// Element-wise subtraction, propagating masks.
  ///
  /// See [add] for the operand and error contract.
  MaskedArray<T> subtract(Object? other) => _maSubtract<T>(this, other);

  /// Element-wise multiplication, propagating masks.
  ///
  /// See [add] for the operand and error contract.
  MaskedArray<T> multiply(Object? other) => _maMultiply<T>(this, other);

  /// Element-wise floor division, propagating masks and masking zero divisors.
  ///
  /// Positions where the divisor is zero (or masked) are excluded from the
  /// computation and masked in the result. See [add] for the operand and error
  /// contract.
  MaskedArray<T> floorDivide(Object? other) => _maFloorDivide<T>(this, other);

  /// Element-wise remainder, propagating masks and masking zero divisors.
  ///
  /// Positions where the divisor is zero (or masked) are excluded from the
  /// computation and masked in the result. See [add] for the operand and error
  /// contract.
  MaskedArray<T> remainder(Object? other) => _maRemainder<T>(this, other);

  /// Element-wise addition (`this + other`); see [add].
  MaskedArray<T> operator +(Object? other) => _maAdd<T>(this, other);

  /// Element-wise subtraction (`this - other`); see [subtract].
  MaskedArray<T> operator -(Object? other) => _maSubtract<T>(this, other);

  /// Element-wise multiplication (`this * other`); see [multiply].
  MaskedArray<T> operator *(Object? other) => _maMultiply<T>(this, other);

  /// Element-wise floor division (`this ~/ other`); see [floorDivide].
  MaskedArray<T> operator ~/(Object? other) => _maFloorDivide<T>(this, other);

  /// Element-wise remainder (`this % other`); see [remainder].
  MaskedArray<T> operator %(Object? other) => _maRemainder<T>(this, other);

  /// Element-wise negation (`-this`), preserving the mask.
  MaskedArray<T> operator -() => mapUnary((d) => ndops.negative<T>(d));

  // ==========================================
  // Comparisons
  // ==========================================

  /// Element-wise `this < other`, returning a boolean [MaskedArray] whose mask
  /// is the logical OR of both operand masks.
  ///
  /// See [add] for the operand and error contract.
  MaskedArray<Boolean> operator <(Object? other) =>
      _maCompare<T>(this, other, '<', (a, b) => a < b);

  /// Element-wise `this <= other`; see [operator <].
  MaskedArray<Boolean> operator <=(Object? other) =>
      _maCompare<T>(this, other, '<=', (a, b) => a <= b);

  /// Element-wise `this > other`; see [operator <].
  MaskedArray<Boolean> operator >(Object? other) =>
      _maCompare<T>(this, other, '>', (a, b) => a > b);

  /// Element-wise `this >= other`; see [operator <].
  MaskedArray<Boolean> operator >=(Object? other) =>
      _maCompare<T>(this, other, '>=', (a, b) => a >= b);

  // ==========================================
  // Reductions
  // ==========================================
  //
  // `sum`, `prod` (accumulator dtype) and `mean` (double-precision dtype) are
  // provided by the DTypeSpec-projected extensions below.

  /// Returns the minimum of elements along the given [axis], ignoring masked elements.
  ///
  /// Throws an [UnsupportedError] for complex and boolean dtypes.
  MaskedArray<T> min({int? axis}) => _maMin<T>(this, axis: axis);

  /// Returns the maximum of elements along the given [axis], ignoring masked elements.
  ///
  /// Throws an [UnsupportedError] for complex and boolean dtypes.
  MaskedArray<T> max({int? axis}) => _maMax<T>(this, axis: axis);

  /// Returns the count of unmasked (valid) elements along the given [axis]
  /// as an [NDArray] of [DType.int64].
  NDArray<Int64> count({int? axis}) => _maCount(this, axis: axis);

  /// Returns the variance of elements along the given [axis], ignoring masked
  /// elements, as a [MaskedArray] of [DType.float64].
  ///
  /// Computed as the mean of the squared magnitude of the deviation from the
  /// mean, so complex inputs also produce a real [DType.float64] result.
  MaskedArray<Float64> variance({int? axis}) => _maVariance(this, axis: axis);

  /// Returns the standard deviation of elements along the given [axis],
  /// ignoring masked elements, as a [MaskedArray] of [DType.float64].
  MaskedArray<Float64> std({int? axis}) => _maStd(this, axis: axis);

  // ==========================================
  // Views & shape manipulation
  // ==========================================

  /// Returns a new [MaskedArray] view with reshaped data and mask.
  MaskedArray<T> reshape(List<int> newShape) => _maReshape<T>(this, newShape);

  /// Returns a transposed [MaskedArray] view.
  MaskedArray<T> transpose([List<int>? axes]) => _maTranspose<T>(this, axes);

  /// Returns a [MaskedArray] view with an expanded dimension inserted at [axis].
  MaskedArray<T> expandDims(int axis) => _maExpandDims<T>(this, axis);

  /// Returns a 1D standard [NDArray] containing all active (unmasked) elements.
  ///
  /// The array is flattened in the process.
  NDArray<T> compressed() => _maCompressed<T>(this);

  /// Returns a standard copy of [NDArray] with masked elements replaced by [fillValue] (or [this.fillValue]).
  NDArray<T> filled({Object? fillValue}) =>
      _maFilled<T>(this, fillValue: fillValue);

  /// Maps a unary ufunc over the data, preserving the mask.
  MaskedArray<R> mapUnary<R extends DTypeTag>(
    NDArray<R> Function(NDArray<T>) ufunc,
  ) => _maMapUnary<T, R>(this, ufunc);
}

/// Typed element access for a [MaskedArray].
///
/// The element type [E] is recovered from the dtype tag [T] through its
/// [DTypeSpec] bound, so `MaskedArray<Float64>.scalar` has static type
/// `double?` and `MaskedArray<Int32>.scalar` has static type `int?`, where
/// `null` means the element is masked.
extension MaskedArrayElements<
  T extends DTypeSpec<
    DTypeTag,
    E,
    DTypeTag,
    DTypeTag,
    DTypeTag,
    DTypeTag,
    DTypeTag,
    DTypeTag
  >,
  E
>
    on MaskedArray<T> {
  /// The single value of a 0-dimensional [MaskedArray], or `null` if masked.
  ///
  /// It is an error if the array is not 0-dimensional.
  E? get scalar => scalarRaw as E?;

  /// The element at the given multi-dimensional [coords], or `null` if masked.
  ///
  /// It is an error if [coords] has a different length than [MaskedArray.rank],
  /// or if any coordinate is out of range.
  E? getCell(List<int> coords) => getCellUntyped(coords) as E?;
}

/// Fallback element access when the type argument is widened to [DTypeTag].
extension MaskedArrayBaseElements on MaskedArray<DTypeTag> {
  /// The single value of a 0-dimensional [MaskedArray], or `null` if masked.
  dynamic get scalar => scalarRaw;

  /// The element at the given multi-dimensional [coords], or `null` if masked.
  dynamic getCell(List<int> coords) => getCellUntyped(coords);
}

/// True division for a [MaskedArray], inferring the concrete result tag [M]
/// from the [DTypeSpec.DivideTag] slot of [T] (`Float64` for integer and
/// boolean arrays; [T] itself for floating-point and complex arrays).
extension MaskedArrayDivide<
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
>
    on MaskedArray<T> {
  /// Element-wise true division, propagating masks and masking zero divisors.
  ///
  /// Positions where the divisor is zero (or masked) are excluded from the
  /// computation and masked in the result. See [MaskedArray.add] for the
  /// operand and error contract.
  MaskedArray<M> divide(Object? other) => _maDivide<T, M>(this, other);

  /// Element-wise true division (`this / other`); see [divide].
  MaskedArray<M> operator /(Object? other) => _maDivide<T, M>(this, other);
}

/// Fallback true division when the type argument is widened to [DTypeTag].
extension MaskedArrayBaseDivide on MaskedArray<DTypeTag> {
  /// Element-wise true division, propagating masks and masking zero divisors.
  MaskedArray<DTypeTag> divide(Object? other) =>
      _maDivide<DTypeTag, DTypeTag>(this, other);

  /// Element-wise true division (`this / other`); see [divide].
  MaskedArray<DTypeTag> operator /(Object? other) =>
      _maDivide<DTypeTag, DTypeTag>(this, other);
}

/// Sum and product reductions for a [MaskedArray], inferring the concrete
/// accumulator tag [R] from the [DTypeSpec.AccumulatorTag] slot of [T]
/// (`Int64` for boolean and signed integer arrays, `Uint64` for unsigned
/// integer arrays, and [T] itself otherwise), matching `package:ndarray`'s
/// `sum` and `prod`.
extension MaskedArrayAccumulatingReductions<
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
>
    on MaskedArray<T> {
  /// Returns the sum of elements along the given [axis], ignoring masked elements.
  ///
  /// Masked elements contribute `0`. The result is masked only where every
  /// element along the reduction is masked.
  MaskedArray<R> sum({int? axis}) => _maSum<T, R>(this, axis: axis);

  /// Returns the product of elements along the given [axis], ignoring masked elements.
  ///
  /// Masked elements contribute `1`. The result is masked only where every
  /// element along the reduction is masked.
  MaskedArray<R> prod({int? axis}) => _maProd<T, R>(this, axis: axis);
}

/// Mean reduction for a [MaskedArray], inferring the concrete result tag [D]
/// from the [DTypeSpec.DoublePrecisionTag] slot of [T] (`Complex128` for
/// complex arrays and `Float64` otherwise), matching `package:ndarray`'s
/// `mean`.
extension MaskedArrayDoublePrecisionReductions<
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
>
    on MaskedArray<T> {
  /// Returns the mean of elements along the given [axis], ignoring masked elements.
  ///
  /// Computed as `sum / count` over the unmasked elements. The result is
  /// masked only where every element along the reduction is masked.
  MaskedArray<D> mean({int? axis}) => _maMean<T, D>(this, axis: axis);
}

/// Fallback sum, product, and mean reductions when the type argument is
/// widened to [DTypeTag].
extension MaskedArrayBaseReductions on MaskedArray<DTypeTag> {
  /// Returns the sum of elements along the given [axis], ignoring masked elements.
  MaskedArray<DTypeTag> sum({int? axis}) =>
      _maSum<DTypeTag, DTypeTag>(this, axis: axis);

  /// Returns the product of elements along the given [axis], ignoring masked elements.
  MaskedArray<DTypeTag> prod({int? axis}) =>
      _maProd<DTypeTag, DTypeTag>(this, axis: axis);

  /// Returns the mean of elements along the given [axis], ignoring masked elements.
  MaskedArray<DTypeTag> mean({int? axis}) =>
      _maMean<DTypeTag, DTypeTag>(this, axis: axis);
}
