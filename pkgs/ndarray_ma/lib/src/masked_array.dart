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

/// An array with an associated boolean mask representing missing or invalid
/// data.
///
/// A [MaskedArray] pairs a standard [NDArray] of dtype tag [T] with a boolean
/// [NDArray] mask of the same shape. Elements where [mask] is `true` are
/// considered invalid or missing, and are bypassed in arithmetic operations
/// and reductions.
///
/// Operations whose result dtype depends deterministically on [T] (such as
/// [MaskedArrayAccumulatingReductions.sum],
/// [MaskedArrayDoublePrecisionReductions.mean], [MaskedArrayDivide.divide],
/// [MaskedArrayRealReductions.abs], [MaskedArrayInexactMath.sin], and the
/// typed [MaskedArrayElements.scalar] accessor) are provided as extensions
/// bounded by the single-projection interfaces ([AccumulatorOf],
/// [DoublePrecisionOf], [DivideOf], [RealOf], [InexactOf], [ElementOf]), so
/// that their static return types match the runtime dtype for every concrete
/// tag. Capability-restricted bitwise and shift operators are provided by
/// [MaskedArrayBitwise] and [MaskedArrayShift]. Receivers typed as
/// `MaskedArray<DTypeTag>` fall back to the `MaskedArrayBase*` extensions, or
/// can be widened explicitly with [asAnySpec], [asBitwiseDType], or
/// [asIntegerDType].
final class MaskedArray<T extends DTypeTag> {
  /// The underlying data array containing all values (both valid and masked).
  final NDArray<T> data;

  /// The boolean mask array of the same shape as [data].
  ///
  /// A value of `true` indicates that the corresponding element in [data] is
  /// invalid or missing (masked).
  final NDArray<Boolean> mask;

  /// The value used to fill masked elements when converting to a standard
  /// [NDArray] via [filled].
  final Object fillValue;

  /// Creates a [MaskedArray] wrapping [data] and [mask].
  ///
  /// The shape of [data] and [mask] must be identical.
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

  /// Whether either [data] or [mask] has been disposed.
  bool get isDisposed => data.isDisposed || mask.isDisposed;

  /// Disposes both the underlying [data] and [mask] arrays.
  void dispose() {
    data.dispose();
    mask.dispose();
  }

  /// Creates a deep copy of this [MaskedArray], or copies its [data] and
  /// [mask] into [out] when provided.
  MaskedArray<T> copy({MaskedArray<T>? out}) {
    return NDArray.scope(() {
      final dataCopy = data.copy();
      final maskCopy = mask.copy();
      return _writeOrCreateMaskedArray<T>(
        dataCopy,
        maskCopy,
        fillValue: fillValue,
        out: out,
      );
    });
  }

  /// Views this array with the wildcard [AnySpec] tag.
  ///
  /// This bypasses static dtype constraints and is intended for helpers and
  /// tests that dispatch or validate dtypes dynamically: a
  /// `MaskedArray<AnySpec>` satisfies every `SelfOf<...>` and projection
  /// bound, with the dtype validated at run time instead.
  MaskedArray<AnySpec> get asAnySpec => this as MaskedArray<AnySpec>;

  /// Views this array with the [AnyBitwiseSpec] tag.
  ///
  /// This bypasses static dtype constraints for helpers that dispatch bitwise
  /// operations dynamically; the result satisfies the
  /// `SelfOf<RealOf<BitwiseDType>>` bound of [bitwiseAnd], [bitwiseOr],
  /// [bitwiseXor], and [invert] and carries the operators of
  /// [MaskedArrayBitwiseSpec].
  ///
  /// The [dtype] must be an integer or boolean data type.
  MaskedArray<AnyBitwiseSpec> get asBitwiseDType {
    if (this is MaskedArray<BitwiseDType>) {
      return this as MaskedArray<AnyBitwiseSpec>;
    }
    throw ArgumentError.value(
      dtype,
      'dtype',
      'Must be integer or boolean data type for bitwise operations.',
    );
  }

  /// Views this array with the [AnyIntegerSpec] tag.
  ///
  /// This bypasses static dtype constraints for helpers that dispatch integer
  /// shift operations dynamically; the result satisfies the
  /// `SelfOf<RealOf<IntegerDType>>` bound of [leftShift] and [rightShift] and
  /// carries the operators of [MaskedArrayShiftSpec].
  ///
  /// The [dtype] must be a signed or unsigned integer data type.
  MaskedArray<AnyIntegerSpec> get asIntegerDType {
    if (this is MaskedArray<IntegerDType>) {
      return this as MaskedArray<AnyIntegerSpec>;
    }
    throw ArgumentError.value(dtype, 'dtype', 'Must be integer data type.');
  }

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

  /// Creates a [MaskedArray] automatically masking `NaN` and infinite values
  /// in [data].
  ///
  /// Elements in [data] that are `NaN` or `Infinity` will be masked (`mask`
  /// set to `true`).
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

  /// Creates a [MaskedArray] automatically masking elements in [data] equal to
  /// [value].
  factory MaskedArray.maskedEqual(
    NDArray<T> data,
    Object value, {
    Object? fillValue,
  }) {
    return NDArray.scope(() {
      final valArray = _wrapScalar<T>(value, data.dtype);
      final mask = ndops.equal(data.asAnySpec, valArray.asAnySpec);
      return MaskedArray(
        data,
        mask.detachToParentScope(),
        fillValue: fillValue,
      );
    });
  }

  /// Creates a [MaskedArray] automatically masking elements in [data] not
  /// equal to [value].
  factory MaskedArray.maskedNotEqual(
    NDArray<T> data,
    Object value, {
    Object? fillValue,
  }) {
    return NDArray.scope(() {
      final valArray = _wrapScalar<T>(value, data.dtype);
      final mask = ndops.notEqual(data.asAnySpec, valArray.asAnySpec);
      return MaskedArray(
        data,
        mask.detachToParentScope(),
        fillValue: fillValue,
      );
    });
  }

  /// Creates a [MaskedArray] automatically masking elements in [data] greater
  /// than [value].
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

  /// Creates a [MaskedArray] automatically masking elements in [data] greater
  /// than or equal to [value].
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

  /// Creates a [MaskedArray] automatically masking elements in [data] less
  /// than [value].
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

  /// Creates a [MaskedArray] automatically masking elements in [data] less
  /// than or equal to [value].
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

  /// Creates a [MaskedArray] masking elements in [data] inside the closed
  /// interval `[v1, v2]` (or `[v2, v1]` if `v2 < v1`).
  factory MaskedArray.maskedInside(
    NDArray<T> data,
    num v1,
    num v2, {
    Object? fillValue,
  }) {
    final low = v1 <= v2 ? v1 : v2;
    final high = v1 <= v2 ? v2 : v1;
    return NDArray.scope(() {
      final geLow = data >= low;
      final leHigh = data <= high;
      final mask = ndops.logicalAnd(geLow, leHigh);
      return MaskedArray(
        data,
        mask.detachToParentScope(),
        fillValue: fillValue,
      );
    });
  }

  /// Creates a [MaskedArray] masking elements in [data] outside the closed
  /// interval `[v1, v2]` (or `[v2, v1]` if `v2 < v1`).
  factory MaskedArray.maskedOutside(
    NDArray<T> data,
    num v1,
    num v2, {
    Object? fillValue,
  }) {
    final low = v1 <= v2 ? v1 : v2;
    final high = v1 <= v2 ? v2 : v1;
    return NDArray.scope(() {
      final ltLow = data < low;
      final gtHigh = data > high;
      final mask = ndops.logicalOr(ltLow, gtHigh);
      return MaskedArray(
        data,
        mask.detachToParentScope(),
        fillValue: fillValue,
      );
    });
  }

  /// Creates a [MaskedArray] masking elements in [data] where [condition] is
  /// `true`.
  factory MaskedArray.maskedWhere(
    NDArray<Boolean> condition,
    NDArray<T> data, {
    Object? fillValue,
  }) {
    return MaskedArray(data, condition.copy(), fillValue: fillValue);
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

  /// Returns a flat list of the elements of this array, with `null` at masked
  /// positions.
  List<Object?> toListUntyped() {
    final flatData = (data as NDArray<DTypeTag>).toList();
    final flatMask = mask.toList();
    return List<Object?>.generate(
      flatData.length,
      (i) => flatMask[i] ? null : flatData[i],
    );
  }

  /// Element access and slicing.
  ///
  /// If [spec] represents coordinates ([int] for 1D, [List<int>] for ND),
  /// returns the scalar value at that coordinate, or `null` if it is masked.
  ///
  /// If [spec] represents a slice ([Slice], [List<Slice>], [List<Selector>]),
  /// returns a new [MaskedArray] view representing the sliced portion of
  /// [data] and [mask].
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
  /// - If [value] is non-null, sets the value in [data] and unmasks it.
  ///
  /// If [spec] represents a slice:
  /// - If [value] is `null`, masks all elements in the slice.
  /// - If [value] is [MaskedArray], copies values and masks from it
  ///   (broadcasting if needed).
  /// - If [value] is [NDArray], copies values from it and unmasks all elements
  ///   in the slice.
  /// - If [value] is a scalar, fills the slice with [value] and unmasks it.
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

  /// Structural equality check comparing shape, dtype, masks, and unmasked
  /// data elements.
  bool equals(MaskedArray<DTypeTag> other) {
    if (dtype != other.dtype || !data.hasSameShape(other.data)) {
      return false;
    }
    if (!mask.equals(other.mask)) {
      return false;
    }
    return NDArray.scope(() {
      final zero = _wrapScalar<T>(_zeroValue(dtype), dtype);
      final aClean = ndops.where<T>(mask, zero, data) as NDArray<T>;
      final bClean =
          ndops.where<T>(other.mask, zero, other.data as NDArray<T>)
              as NDArray<T>;
      return aClean.equals(bClean);
    });
  }

  /// Detaches this array's components ([data] and [mask]) from the current
  /// automatic disposal scope.
  MaskedArray<T> detachFromScope() {
    data.detachFromScope();
    mask.detachFromScope();
    return this;
  }

  /// Detaches this array's components from the current scope and promotes them
  /// to the parent scope.
  MaskedArray<T> detachToParentScope() {
    data.detachToParentScope();
    mask.detachToParentScope();
    return this;
  }

  // ==========================================
  // Conversions
  // ==========================================

  /// Returns a copy of this array with [data] converted to [dtype] (or writes
  /// into [out] if provided).
  ///
  /// The mask is copied, and [fillValue] is coerced to the new dtype (falling
  /// back to the default fill value of [dtype] when it cannot be represented).
  MaskedArray<R> astype<R extends DTypeTag, Out extends R>(
    DType<R> dtype, {
    MaskedArray<Out>? out,
  }) {
    return NDArray.scope(() {
      final converted = data.astype(dtype);
      final maskCopy = mask.copy();
      return _writeOrCreateMaskedArray<R>(
        converted,
        maskCopy,
        fillValue: fillValue,
        out: out,
      );
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
  MaskedArray<T> add(Object? other, {MaskedArray<T>? out}) =>
      _maAdd<T>(this, other, out: out);

  /// Element-wise subtraction, propagating masks.
  ///
  /// See [add] for the operand and error contract.
  MaskedArray<T> subtract(Object? other, {MaskedArray<T>? out}) =>
      _maSubtract<T>(this, other, out: out);

  /// Element-wise multiplication, propagating masks.
  ///
  /// See [add] for the operand and error contract.
  MaskedArray<T> multiply(Object? other, {MaskedArray<T>? out}) =>
      _maMultiply<T>(this, other, out: out);

  /// Element-wise floor division, propagating masks and masking zero divisors.
  ///
  /// Positions where the divisor is zero (or masked) are excluded from the
  /// computation and masked in the result. See [add] for the operand and error
  /// contract.
  MaskedArray<T> floorDivide(Object? other, {MaskedArray<T>? out}) =>
      _maFloorDivide<T>(this, other, out: out);

  /// Element-wise remainder, propagating masks and masking zero divisors.
  ///
  /// Positions where the divisor is zero (or masked) are excluded from the
  /// computation and masked in the result. See [add] for the operand and error
  /// contract.
  MaskedArray<T> remainder(Object? other, {MaskedArray<T>? out}) =>
      _maRemainder<T>(this, other, out: out);

  /// Element-wise minimum of `this` and [other], propagating masks.
  MaskedArray<T> minimum(Object? other, {MaskedArray<T>? out}) =>
      _maMinimum<T>(this, other, out: out);

  /// Element-wise maximum of `this` and [other], propagating masks.
  MaskedArray<T> maximum(Object? other, {MaskedArray<T>? out}) =>
      _maMaximum<T>(this, other, out: out);

  /// Element-wise exponentiation (`this ^ other`), propagating masks.
  MaskedArray<T> power(Object? other, {MaskedArray<T>? out}) =>
      _maPower<T>(this, other, out: out);

  /// Element-wise negation (`-this`), preserving the mask.
  MaskedArray<T> negative({MaskedArray<T>? out}) =>
      mapUnary<T, T>((d) => ndops.negative(d), out: out);

  /// Clips (limits) the unmasked values in this array to `[aMin, aMax]`,
  /// preserving the mask.
  MaskedArray<T> clip({num? aMin, num? aMax, MaskedArray<T>? out}) =>
      _maClip<T>(this, aMin: aMin, aMax: aMax, out: out);

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
  MaskedArray<T> operator -() => negative();

  // ==========================================
  // Comparisons
  // ==========================================

  /// Element-wise equality (`this == other`), returning a boolean
  /// [MaskedArray] whose mask is the logical OR of both operand masks.
  MaskedArray<Boolean> eq(Object? other, {MaskedArray<Boolean>? out}) =>
      _maCompare<T>(this, other, '==', (a, b) => a.eq(b), out: out);

  /// Element-wise inequality (`this != other`), returning a boolean
  /// [MaskedArray] whose mask is the logical OR of both operand masks.
  MaskedArray<Boolean> ne(Object? other, {MaskedArray<Boolean>? out}) =>
      _maCompare<T>(this, other, '!=', (a, b) => a.ne(b), out: out);

  /// Element-wise `this < other`, returning a boolean [MaskedArray] whose mask
  /// is the logical OR of both operand masks.
  MaskedArray<Boolean> lt(Object? other, {MaskedArray<Boolean>? out}) =>
      _maCompare<T>(this, other, '<', (a, b) => a < b, out: out);

  /// Element-wise `this <= other`; see [lt].
  MaskedArray<Boolean> le(Object? other, {MaskedArray<Boolean>? out}) =>
      _maCompare<T>(this, other, '<=', (a, b) => a <= b, out: out);

  /// Element-wise `this > other`; see [lt].
  MaskedArray<Boolean> gt(Object? other, {MaskedArray<Boolean>? out}) =>
      _maCompare<T>(this, other, '>', (a, b) => a > b, out: out);

  /// Element-wise `this >= other`; see [lt].
  MaskedArray<Boolean> ge(Object? other, {MaskedArray<Boolean>? out}) =>
      _maCompare<T>(this, other, '>=', (a, b) => a >= b, out: out);

  /// Element-wise `this < other`, returning a boolean [MaskedArray] whose mask
  /// is the logical OR of both operand masks.
  ///
  /// See [add] for the operand and error contract.
  MaskedArray<Boolean> operator <(Object? other) => lt(other);

  /// Element-wise `this <= other`; see [operator <].
  MaskedArray<Boolean> operator <=(Object? other) => le(other);

  /// Element-wise `this > other`; see [operator <].
  MaskedArray<Boolean> operator >(Object? other) => gt(other);

  /// Element-wise `this >= other`; see [operator <].
  MaskedArray<Boolean> operator >=(Object? other) => ge(other);

  // ==========================================
  // Reductions
  // ==========================================
  //
  // `sum`, `prod` ([AccumulatorOf]) and `mean` ([DoublePrecisionOf]) are
  // provided by the projection-bounded extensions below.

  /// Returns the minimum of elements along the given [axis], ignoring masked
  /// elements.
  ///
  /// It is an error to call [min] on complex or boolean arrays.
  MaskedArray<T> min({int? axis, MaskedArray<T>? out}) =>
      _maMin<T>(this, axis: axis, out: out);

  /// Returns the maximum of elements along the given [axis], ignoring masked
  /// elements.
  ///
  /// It is an error to call [max] on complex or boolean arrays.
  MaskedArray<T> max({int? axis, MaskedArray<T>? out}) =>
      _maMax<T>(this, axis: axis, out: out);

  /// Returns the count of unmasked (valid) elements along the given [axis]
  /// as an [NDArray] of [DType.int64].
  NDArray<Int64> count({int? axis, NDArray<Int64>? out}) =>
      _maCount(this, axis: axis, out: out);

  /// Returns the variance of elements along the given [axis], ignoring masked
  /// elements, as a [MaskedArray] of [DType.float64].
  ///
  /// Computed as the mean of the squared magnitude of the deviation from the
  /// mean, so complex inputs also produce a real [DType.float64] result.
  MaskedArray<Float64> variance({int? axis, MaskedArray<Float64>? out}) =>
      _maVariance(this, axis: axis, out: out);

  /// Returns the standard deviation of elements along the given [axis],
  /// ignoring masked elements, as a [MaskedArray] of [DType.float64].
  MaskedArray<Float64> std({int? axis, MaskedArray<Float64>? out}) =>
      _maStd(this, axis: axis, out: out);

  // ==========================================
  // Views & shape manipulation
  // ==========================================

  /// Returns a new [MaskedArray] view with reshaped data and mask.
  MaskedArray<T> reshape(List<int> newShape) => _maReshape<T>(this, newShape);

  /// Returns a transposed [MaskedArray] view.
  MaskedArray<T> transpose([List<int>? axes]) => _maTranspose<T>(this, axes);

  /// A transposed view of this [MaskedArray] with reversed axes.
  MaskedArray<T> get transposed => _maTranspose<T>(this);

  /// Returns a [MaskedArray] view with an expanded dimension inserted at [axis].
  MaskedArray<T> expandDims(int axis) => _maExpandDims<T>(this, axis);

  /// Returns a [MaskedArray] view with size-1 dimensions removed.
  MaskedArray<T> squeeze([int? axis]) => _maSqueeze<T>(this, axis);

  /// Returns a flattened 1D copy of this [MaskedArray].
  MaskedArray<T> flatten({MaskedArray<T>? out}) =>
      _maFlatten<T>(this, out: out);

  /// Returns a 1D standard [NDArray] containing all active (unmasked) elements.
  ///
  /// The array is flattened in the process.
  NDArray<T> compressed() => _maCompressed<T>(this);

  /// Returns a standard copy of [NDArray] with masked elements replaced by
  /// [fillValue] (or [this.fillValue]), writing into [out] if provided.
  NDArray<T> filled({Object? fillValue, NDArray<T>? out}) =>
      _maFilled<T, T>(this, fillValue: fillValue, out: out);

  /// Maps a unary [ufunc] over [data], preserving the mask.
  MaskedArray<R> mapUnary<R extends DTypeTag, Out extends R>(
    NDArray<R> Function(NDArray<T>) ufunc, {
    MaskedArray<Out>? out,
  }) => _maMapUnary<T, R>(this, ufunc, out: out);
}

/// Typed element access for a [MaskedArray].
///
/// The element type [E] is recovered from the dtype tag [T] through its
/// [ElementOf] bound, so `MaskedArray<Float64>.scalar` has static type
/// `double?` and `MaskedArray<Int32>.scalar` has static type `int?`, where
/// `null` means the element is masked.
extension MaskedArrayElements<T extends ElementOf<E>, E> on MaskedArray<T> {
  /// The single value of a 0-dimensional [MaskedArray], or `null` if masked.
  ///
  /// It is an error if the array is not 0-dimensional.
  E? get scalar => scalarRaw as E?;

  /// The element at the given multi-dimensional [coords], or `null` if masked.
  ///
  /// It is an error if [coords] has a different length than [MaskedArray.rank],
  /// or if any coordinate is out of range.
  E? getCell(List<int> coords) => getCellUntyped(coords) as E?;

  /// Returns a flat list of elements of type `E?`, where masked elements are
  /// `null`.
  List<E?> toList() => toListUntyped().cast<E?>();
}

/// Fallback element access when the type argument is widened to [DTypeTag].
extension MaskedArrayBaseElements on MaskedArray<DTypeTag> {
  /// The single value of a 0-dimensional [MaskedArray], or `null` if masked.
  dynamic get scalar => scalarRaw;

  /// The element at the given multi-dimensional [coords], or `null` if masked.
  dynamic getCell(List<int> coords) => getCellUntyped(coords);

  /// Returns a flat list of elements, where masked elements are `null`.
  List<Object?> toList() => toListUntyped();
}

/// True division for a [MaskedArray], inferring the concrete result tag [M]
/// from the [DivideOf] projection of [T] (`Float64` for integer and
/// boolean arrays; [T] itself for floating-point and complex arrays).
extension MaskedArrayDivide<T extends DivideOf<M>, M extends DTypeTag>
    on MaskedArray<T> {
  /// Element-wise true division, propagating masks and masking zero divisors.
  ///
  /// Positions where the divisor is zero (or masked) are excluded from the
  /// computation and masked in the result. See [MaskedArray.add] for the
  /// operand and error contract.
  MaskedArray<M> divide<Out extends M>(
    Object? other, {
    MaskedArray<Out>? out,
  }) => _maDivide<T, M>(this, other, out: out);

  /// Element-wise true division (`this / other`); see [divide].
  MaskedArray<M> operator /(Object? other) => _maDivide<T, M>(this, other);
}

/// Fallback true division when the type argument is widened to [DTypeTag].
extension MaskedArrayBaseDivide on MaskedArray<DTypeTag> {
  /// Element-wise true division, propagating masks and masking zero divisors.
  MaskedArray<DTypeTag> divide<Out extends DTypeTag>(
    Object? other, {
    MaskedArray<Out>? out,
  }) => _maDivide<DTypeTag, DTypeTag>(this, other, out: out);

  /// Element-wise true division (`this / other`); see [divide].
  MaskedArray<DTypeTag> operator /(Object? other) =>
      _maDivide<DTypeTag, DTypeTag>(this, other);
}

/// Bitwise operations (`&`, `|`, `^`, `~`) for [MaskedArray]s whose dtype tag
/// implements [BitwiseDType]: all integer tags, [Boolean], and the least upper
/// bounds `MaskedArray<IntegerDType>` / `MaskedArray<BitwiseDType>`.
///
/// Applying one of these operators to a floating-point or complex array is a
/// compile-time error. [MaskedArrayBitwiseSpec] provides the same operations
/// for the run-time-checked [AnyBitwiseSpec] row returned by
/// [MaskedArray.asBitwiseDType].
extension MaskedArrayBitwise<T extends BitwiseDType> on MaskedArray<T> {
  /// Element-wise bitwise AND, propagating masks.
  MaskedArray<T> bitwiseAnd<Out extends T>(
    Object? other, {
    MaskedArray<Out>? out,
  }) => _maBitwiseAnd<T>(this, other, out: out);

  /// Element-wise bitwise OR, propagating masks.
  MaskedArray<T> bitwiseOr<Out extends T>(
    Object? other, {
    MaskedArray<Out>? out,
  }) => _maBitwiseOr<T>(this, other, out: out);

  /// Element-wise bitwise XOR, propagating masks.
  MaskedArray<T> bitwiseXor<Out extends T>(
    Object? other, {
    MaskedArray<Out>? out,
  }) => _maBitwiseXor<T>(this, other, out: out);

  /// Element-wise bitwise NOT / inversion (`~this`), preserving the mask.
  MaskedArray<T> invert<Out extends T>({MaskedArray<Out>? out}) =>
      _maInvert<T>(this, out: out);

  /// Element-wise bitwise AND (`this & other`); see [bitwiseAnd].
  MaskedArray<T> operator &(Object? other) => _maBitwiseAnd<T>(this, other);

  /// Element-wise bitwise OR (`this | other`); see [bitwiseOr].
  MaskedArray<T> operator |(Object? other) => _maBitwiseOr<T>(this, other);

  /// Element-wise bitwise XOR (`this ^ other`); see [bitwiseXor].
  MaskedArray<T> operator ^(Object? other) => _maBitwiseXor<T>(this, other);

  /// Element-wise bitwise NOT (`~this`); see [invert].
  MaskedArray<T> operator ~() => _maInvert<T>(this);
}

/// Bitwise operations (`&`, `|`, `^`, `~`) for the run-time-checked
/// [AnyBitwiseSpec] row produced by [MaskedArray.asBitwiseDType].
extension MaskedArrayBitwiseSpec on MaskedArray<AnyBitwiseSpec> {
  /// Element-wise bitwise AND, propagating masks.
  MaskedArray<AnyBitwiseSpec> bitwiseAnd<Out extends AnyBitwiseSpec>(
    Object? other, {
    MaskedArray<Out>? out,
  }) => _maBitwiseAnd<AnyBitwiseSpec>(this, other, out: out);

  /// Element-wise bitwise OR, propagating masks.
  MaskedArray<AnyBitwiseSpec> bitwiseOr<Out extends AnyBitwiseSpec>(
    Object? other, {
    MaskedArray<Out>? out,
  }) => _maBitwiseOr<AnyBitwiseSpec>(this, other, out: out);

  /// Element-wise bitwise XOR, propagating masks.
  MaskedArray<AnyBitwiseSpec> bitwiseXor<Out extends AnyBitwiseSpec>(
    Object? other, {
    MaskedArray<Out>? out,
  }) => _maBitwiseXor<AnyBitwiseSpec>(this, other, out: out);

  /// Element-wise bitwise NOT (`~this`), preserving the mask.
  MaskedArray<AnyBitwiseSpec> invert<Out extends AnyBitwiseSpec>({
    MaskedArray<Out>? out,
  }) => _maInvert<AnyBitwiseSpec>(this, out: out);

  /// Element-wise bitwise AND (`this & other`).
  MaskedArray<AnyBitwiseSpec> operator &(Object? other) =>
      _maBitwiseAnd<AnyBitwiseSpec>(this, other);

  /// Element-wise bitwise OR (`this | other`).
  MaskedArray<AnyBitwiseSpec> operator |(Object? other) =>
      _maBitwiseOr<AnyBitwiseSpec>(this, other);

  /// Element-wise bitwise XOR (`this ^ other`).
  MaskedArray<AnyBitwiseSpec> operator ^(Object? other) =>
      _maBitwiseXor<AnyBitwiseSpec>(this, other);

  /// Element-wise bitwise NOT (`~this`).
  MaskedArray<AnyBitwiseSpec> operator ~() => _maInvert<AnyBitwiseSpec>(this);
}

/// Shift operations (`<<`, `>>`) for [MaskedArray]s whose dtype tag implements
/// [IntegerDType]: all signed and unsigned integer tags and the least upper
/// bound `MaskedArray<IntegerDType>`.
///
/// Applying one of these operators to a boolean, floating-point, or complex
/// array is a compile-time error. [MaskedArrayShiftSpec] provides the same
/// operations for the run-time-checked [AnyIntegerSpec] row returned by
/// [MaskedArray.asIntegerDType].
extension MaskedArrayShift<T extends IntegerDType> on MaskedArray<T> {
  /// Element-wise left shift, propagating masks.
  MaskedArray<T> leftShift<Out extends T>(
    Object? other, {
    MaskedArray<Out>? out,
  }) => _maLeftShift<T>(this, other, out: out);

  /// Element-wise right shift, propagating masks.
  MaskedArray<T> rightShift<Out extends T>(
    Object? other, {
    MaskedArray<Out>? out,
  }) => _maRightShift<T>(this, other, out: out);

  /// Element-wise left shift (`this << other`); see [leftShift].
  MaskedArray<T> operator <<(Object? other) => _maLeftShift<T>(this, other);

  /// Element-wise right shift (`this >> other`); see [rightShift].
  MaskedArray<T> operator >>(Object? other) => _maRightShift<T>(this, other);
}

/// Shift operations (`<<`, `>>`) for the run-time-checked [AnyIntegerSpec] row
/// produced by [MaskedArray.asIntegerDType].
extension MaskedArrayShiftSpec on MaskedArray<AnyIntegerSpec> {
  /// Element-wise left shift, propagating masks.
  MaskedArray<AnyIntegerSpec> leftShift<Out extends AnyIntegerSpec>(
    Object? other, {
    MaskedArray<Out>? out,
  }) => _maLeftShift<AnyIntegerSpec>(this, other, out: out);

  /// Element-wise right shift, propagating masks.
  MaskedArray<AnyIntegerSpec> rightShift<Out extends AnyIntegerSpec>(
    Object? other, {
    MaskedArray<Out>? out,
  }) => _maRightShift<AnyIntegerSpec>(this, other, out: out);

  /// Element-wise left shift (`this << other`).
  MaskedArray<AnyIntegerSpec> operator <<(Object? other) =>
      _maLeftShift<AnyIntegerSpec>(this, other);

  /// Element-wise right shift (`this >> other`).
  MaskedArray<AnyIntegerSpec> operator >>(Object? other) =>
      _maRightShift<AnyIntegerSpec>(this, other);
}

/// Sum and product reductions for a [MaskedArray], inferring the concrete
/// accumulator tag [R] from the [AccumulatorOf] projection of [T]
/// (`Int64` for boolean and signed integer arrays, `Uint64` for unsigned
/// integer arrays, and [T] itself otherwise), matching `package:ndarray`'s
/// `sum` and `prod`.
extension MaskedArrayAccumulatingReductions<
  T extends AccumulatorOf<R>,
  R extends DTypeTag
>
    on MaskedArray<T> {
  /// Returns the sum of elements along the given [axis], ignoring masked
  /// elements.
  ///
  /// Masked elements contribute `0`. The result is masked only where every
  /// element along the reduction is masked.
  MaskedArray<R> sum<Out extends R>({int? axis, MaskedArray<Out>? out}) =>
      _maSum<T, R>(this, axis: axis, out: out);

  /// Returns the product of elements along the given [axis], ignoring masked
  /// elements.
  ///
  /// Masked elements contribute `1`. The result is masked only where every
  /// element along the reduction is masked.
  MaskedArray<R> prod<Out extends R>({int? axis, MaskedArray<Out>? out}) =>
      _maProd<T, R>(this, axis: axis, out: out);
}

/// Mean reduction for a [MaskedArray], inferring the concrete result tag [D]
/// from the [DoublePrecisionOf] projection of [T] (`Complex128` for
/// complex arrays and `Float64` otherwise), matching `package:ndarray`'s
/// `mean`.
extension MaskedArrayDoublePrecisionReductions<
  T extends DoublePrecisionOf<D>,
  D extends DTypeTag
>
    on MaskedArray<T> {
  /// Returns the mean of elements along the given [axis], ignoring masked
  /// elements.
  ///
  /// Computed as `sum / count` over the unmasked elements. The result is
  /// masked only where every element along the reduction is masked.
  MaskedArray<D> mean<Out extends D>({int? axis, MaskedArray<Out>? out}) =>
      _maMean<T, D>(this, axis: axis, out: out);
}

/// Fallback sum, product, and mean reductions when the type argument is
/// widened to [DTypeTag].
extension MaskedArrayBaseReductions on MaskedArray<DTypeTag> {
  /// Returns the sum of elements along the given [axis], ignoring masked
  /// elements.
  MaskedArray<DTypeTag> sum<Out extends DTypeTag>({
    int? axis,
    MaskedArray<Out>? out,
  }) => _maSum<DTypeTag, DTypeTag>(this, axis: axis, out: out);

  /// Returns the product of elements along the given [axis], ignoring masked
  /// elements.
  MaskedArray<DTypeTag> prod<Out extends DTypeTag>({
    int? axis,
    MaskedArray<Out>? out,
  }) => _maProd<DTypeTag, DTypeTag>(this, axis: axis, out: out);

  /// Returns the mean of elements along the given [axis], ignoring masked
  /// elements.
  MaskedArray<DTypeTag> mean<Out extends DTypeTag>({
    int? axis,
    MaskedArray<Out>? out,
  }) => _maMean<DTypeTag, DTypeTag>(this, axis: axis, out: out);
}

/// Real-component / magnitude operations for a [MaskedArray], inferring the
/// concrete result tag [R] from the [RealOf] projection of [T] (`Float32` for
/// `Complex64`, `Float64` for `Complex128`, and [T] itself for real dtypes).
extension MaskedArrayRealReductions<T extends RealOf<R>, R extends DTypeTag>
    on MaskedArray<T> {
  /// Element-wise absolute value (or complex magnitude), preserving the mask.
  MaskedArray<R> abs<Out extends R>({MaskedArray<Out>? out}) =>
      mapUnary<R, Out>((d) => ndops.abs(d), out: out);
}

/// Fallback [abs] when the type argument is widened to [DTypeTag].
extension MaskedArrayBaseRealReductions on MaskedArray<DTypeTag> {
  /// Element-wise absolute value (or complex magnitude), preserving the mask.
  MaskedArray<DTypeTag> abs<Out extends DTypeTag>({MaskedArray<Out>? out}) =>
      mapUnary<DTypeTag, Out>((d) => ndops.abs(d.asAnySpec), out: out);
}

/// Transcendental and inexact math operations for a [MaskedArray], inferring
/// the concrete result tag [R] from the [InexactOf] projection of [T]
/// (`Float64` for integer, boolean, `Float16`, `BFloat16`, and `Float64`;
/// `Float32`, `Complex64`, and `Complex128` preserved).
extension MaskedArrayInexactMath<T extends InexactOf<R>, R extends DTypeTag>
    on MaskedArray<T> {
  /// Element-wise square root, preserving the mask.
  MaskedArray<R> sqrt<Out extends R>({MaskedArray<Out>? out}) =>
      mapUnary<R, Out>((d) => ndops.sqrt(d), out: out);

  /// Element-wise exponential ($e^x$), preserving the mask.
  MaskedArray<R> exp<Out extends R>({MaskedArray<Out>? out}) =>
      mapUnary<R, Out>((d) => ndops.exp(d), out: out);

  /// Element-wise natural logarithm ($\ln x$), preserving the mask.
  MaskedArray<R> log<Out extends R>({MaskedArray<Out>? out}) =>
      mapUnary<R, Out>((d) => ndops.log(d), out: out);

  /// Element-wise sine, preserving the mask.
  MaskedArray<R> sin<Out extends R>({MaskedArray<Out>? out}) =>
      mapUnary<R, Out>((d) => ndops.sin(d), out: out);

  /// Element-wise cosine, preserving the mask.
  MaskedArray<R> cos<Out extends R>({MaskedArray<Out>? out}) =>
      mapUnary<R, Out>((d) => ndops.cos(d), out: out);

  /// Element-wise tangent, preserving the mask.
  MaskedArray<R> tan<Out extends R>({MaskedArray<Out>? out}) =>
      mapUnary<R, Out>((d) => ndops.tan(d), out: out);

  /// Element-wise four-quadrant inverse tangent (`atan2(this, other)`),
  /// propagating masks.
  MaskedArray<R> atan2<Out extends R>(Object? other, {MaskedArray<Out>? out}) =>
      _maAtan2<T, R>(this, other, out: out);

  /// Element-wise Euclidean hypotenuse (`sqrt(this^2 + other^2)`),
  /// propagating masks.
  MaskedArray<R> hypot<Out extends R>(Object? other, {MaskedArray<Out>? out}) =>
      _maHypot<T, R>(this, other, out: out);
}

/// Fallback transcendental and inexact math operations when the type argument
/// is widened to [DTypeTag].
extension MaskedArrayBaseInexactMath on MaskedArray<DTypeTag> {
  /// Element-wise square root, preserving the mask.
  MaskedArray<DTypeTag> sqrt<Out extends DTypeTag>({MaskedArray<Out>? out}) =>
      mapUnary<DTypeTag, Out>((d) => ndops.sqrt(d.asAnySpec), out: out);

  /// Element-wise exponential ($e^x$), preserving the mask.
  MaskedArray<DTypeTag> exp<Out extends DTypeTag>({MaskedArray<Out>? out}) =>
      mapUnary<DTypeTag, Out>((d) => ndops.exp(d.asAnySpec), out: out);

  /// Element-wise natural logarithm ($\ln x$), preserving the mask.
  MaskedArray<DTypeTag> log<Out extends DTypeTag>({MaskedArray<Out>? out}) =>
      mapUnary<DTypeTag, Out>((d) => ndops.log(d.asAnySpec), out: out);

  /// Element-wise sine, preserving the mask.
  MaskedArray<DTypeTag> sin<Out extends DTypeTag>({MaskedArray<Out>? out}) =>
      mapUnary<DTypeTag, Out>((d) => ndops.sin(d.asAnySpec), out: out);

  /// Element-wise cosine, preserving the mask.
  MaskedArray<DTypeTag> cos<Out extends DTypeTag>({MaskedArray<Out>? out}) =>
      mapUnary<DTypeTag, Out>((d) => ndops.cos(d.asAnySpec), out: out);

  /// Element-wise tangent, preserving the mask.
  MaskedArray<DTypeTag> tan<Out extends DTypeTag>({MaskedArray<Out>? out}) =>
      mapUnary<DTypeTag, Out>((d) => ndops.tan(d.asAnySpec), out: out);

  /// Element-wise four-quadrant inverse tangent (`atan2(this, other)`),
  /// propagating masks.
  MaskedArray<DTypeTag> atan2<Out extends DTypeTag>(
    Object? other, {
    MaskedArray<Out>? out,
  }) => _maAtan2<DTypeTag, DTypeTag>(this, other, out: out);

  /// Element-wise Euclidean hypotenuse (`sqrt(this^2 + other^2)`),
  /// propagating masks.
  MaskedArray<DTypeTag> hypot<Out extends DTypeTag>(
    Object? other, {
    MaskedArray<Out>? out,
  }) => _maHypot<DTypeTag, DTypeTag>(this, other, out: out);
}

/// Top-level helper for `divide` on [MaskedArray]s.
MaskedArray<M> maDivide<T extends DTypeTag, M extends DTypeTag>(
  MaskedArray<T> a,
  MaskedArray<T> b, {
  MaskedArray<DTypeTag>? out,
}) => _maDivide<T, M>(a, b, out: out);

/// Top-level helper for `atan2` on [MaskedArray]s.
MaskedArray<R> maAtan2<T extends DTypeTag, R extends DTypeTag>(
  MaskedArray<T> y,
  MaskedArray<T> x, {
  MaskedArray<DTypeTag>? out,
}) => _maAtan2<T, R>(y, x, out: out);

/// Top-level helper for `hypot` on [MaskedArray]s.
MaskedArray<R> maHypot<T extends DTypeTag, R extends DTypeTag>(
  MaskedArray<T> x1,
  MaskedArray<T> x2, {
  MaskedArray<DTypeTag>? out,
}) => _maHypot<T, R>(x1, x2, out: out);

/// Top-level helper for [where] on [MaskedArray]s.
MaskedArray<T> maWhere<T extends DTypeTag, Out extends T>(
  Object condition,
  MaskedArray<T> x,
  MaskedArray<T> y, {
  MaskedArray<Out>? out,
}) => _maWhere<T>(condition, x, y, out: out);

/// Top-level helper for [concatenate] on [MaskedArray]s.
MaskedArray<T> maConcatenate<T extends DTypeTag, Out extends T>(
  List<MaskedArray<T>> arrays, {
  int axis = 0,
  MaskedArray<Out>? out,
}) => _maConcatenate<T>(arrays, axis: axis, out: out);
