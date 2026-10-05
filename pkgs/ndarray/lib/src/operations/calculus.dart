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

import '../ndarray.dart';
import 'dart:ffi' as ffi;
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';

// Standalone operational relative cross-imports
import 'helpers.dart';

/// Represents the spacing between points for calculus operations along a single axis.
///
/// Spacing can be a constant scalar (step) or a list of coordinates
/// (variable/non-uniform spacing). The type parameter [V] represents the numeric
/// type (typically [double] or [Complex]).
///
/// Note: To specify spacings for multiple axes in [gradientArray], use a
/// `List<Spacing>`.
sealed class Spacing<V extends Object> {
  const Spacing();

  /// Constant spacing of value [value] (e.g. [dx]).
  const factory Spacing.step(V value) = StepSpacing<V>;

  /// Variable (non-uniform) spacing using a coordinate list [values].
  /// The length of [values] must match the dimension size of the axis.
  const factory Spacing.coordinates(List<V> values) = CoordinateSpacing<V>;
}

/// Constant spacing implementation.
final class StepSpacing<V extends Object> extends Spacing<V> {
  /// The constant step size along the axis.
  final V value;

  /// Creates a constant spacing of [value].
  const StepSpacing(this.value);
}

/// Variable coordinate spacing implementation.
final class CoordinateSpacing<V extends Object> extends Spacing<V> {
  final List<V> _values;

  /// The coordinate values along the axis.
  List<V> get values => List<V>.unmodifiable(_values);

  /// Creates a variable spacing from the coordinate list [_values].
  const CoordinateSpacing(this._values);
}

// Helper for list equality comparison
bool _listEquals(List<Object?> a, List<Object?> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

/// Integrates along the given axis using the composite trapezoidal rule.
///
/// The composite trapezoidal rule approximates the integral of a function by
/// dividing the area under the curve into trapezoids:
///
/// \[
/// \int_a^b f(x)\,dx \approx \sum_{i=1}^{N-1} \frac{f(x_{i-1}) + f(x_i)}{2} \Delta x_i
/// \]
///
/// This approximation is significantly more accurate than simple rectangular integration.
///
/// The [axis] parameter specifies the dimension of the array along which to
/// integrate. A negative value is resolved relative to the end of the dimensions,
/// where `-1` represents the last dimension, `-2` represents the second-to-last,
/// and so on.
///
/// Spacing along the axis is specified by [spacing].
///
/// **Preconditions:**
/// - Input [y] must not be disposed.
/// - Input [y] must be a numeric (floating-point, integer, or complex) type.
/// - [axis] must be within bounds `[-y.rank, y.rank - 1]`.
/// - If [spacing] is [CoordinateSpacing], its length must match `y.shape[axis]`.
/// - If [spacing] is complex, input [y] must also be complex.
/// - If [out] is provided, it must match the resolved shape and dtype.
/// - It is an error if [y] or [out] is disposed.
/// - It is an error if [y] has a boolean dtype.
/// - It is an error if complex spacing is used with a real input array.
/// - It is an error if [axis] is out of bounds or coordinate spacing length is mismatched.
///
/// **Example:**
/// {@example /example/calculus_example.dart lang=dart}
NDArray<T> trapz<T extends DTypeTag>(
  NDArray<
    DTypeSpec<DTypeTag, Object?, DTypeTag, DTypeTag, T, DTypeTag, DTypeTag>
  >
  y, {
  Spacing spacing = const Spacing.step(1.0),
  int axis = -1,
  NDArray<T>? out,
}) {
  if (y.isDisposed) {
    throw StateError('Cannot execute trapz() on a disposed array.');
  }
  if (out != null) {
    if (out.isDisposed) {
      throw StateError('Cannot write trapz result to a disposed output array.');
    }
    validateOutBuffer(out);
  }

  final DType<DTypeTag> yDType = y.dtype;
  if (yDType == DType.boolean) {
    throw ArgumentError.value(
      y.dtype,
      'y',
      'Must not be boolean (calculus operations are not supported on boolean arrays; cast to a floating-point or complex type first)',
    );
  }

  final bool isComplexSpacing = switch (spacing) {
    Spacing<Complex>() => true,
    StepSpacing(:final value) => value is Complex,
    CoordinateSpacing(:final values) =>
      values.isNotEmpty && values.first is Complex,
  };
  if (isComplexSpacing && !y.dtype.isComplex) {
    throw ArgumentError.value(
      y.dtype,
      'y',
      'Must be complex when spacing is complex (cast the array to complex first)',
    );
  }

  if (axis < -y.rank || axis >= y.rank) {
    throw RangeError.range(
      axis,
      -y.rank,
      y.rank - 1,
      'axis',
      'Must be within valid rank range',
    );
  }
  final targetAxis = axis < 0 ? y.rank + axis : axis;

  final N = y.shape[targetAxis];
  if (spacing is CoordinateSpacing) {
    if (spacing.values.length != N) {
      throw ArgumentError.value(
        spacing.values.length,
        'spacing',
        'Must have length matching dimension size $N (got ${spacing.values.length})',
      );
    }
  }

  final targetShape = List<int>.from(y.shape)..removeAt(targetAxis);

  if (y.dtype.isInteger) {
    if (out != null) {
      if (!_listEquals(out.shape, targetShape) || out.dtype == DType.boolean) {
        throw ArgumentError.value(
          out,
          'out',
          'Must have compatible shape $targetShape and non-boolean dtype (got shape ${out.shape}, dtype ${out.dtype})',
        );
      }
    }
    return NDArray.scope(() {
      final doubleY = castNDArray<Float64>(y, DType.float64);
      final Spacing doubleSpacing = switch (spacing) {
        StepSpacing(:final value) => Spacing.step((value as num).toDouble()),
        CoordinateSpacing(:final values) => Spacing.coordinates([
          for (final v in values) (v as num).toDouble(),
        ]),
      };
      final doubleRes = trapz<Float64>(
        doubleY,
        spacing: doubleSpacing,
        axis: axis,
      );
      if (out != null) {
        final casted = castNDArray(doubleRes, out.dtype);
        casted.copy(out: out);
        return out;
      }
      return doubleRes.detachToParentScope() as NDArray<T>;
    });
  }

  if (out != null) {
    final validDType =
        out.dtype == yDType ||
        ((yDType == DType.float16 || yDType == DType.bfloat16) &&
            out.dtype == DType.float64);
    if (!_listEquals(out.shape, targetShape) || !validDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape $targetShape and dtype ${y.dtype} (got shape ${out.shape}, dtype ${out.dtype})',
      );
    }
    if (sharesMemory(y, out)) {
      return NDArray.scope(() {
        final temp = trapz<T>(y, spacing: spacing, axis: axis);
        temp.copy(out: out);
        return out;
      });
    }
  }

  switch (yDType) {
    case DType.float16:
    case DType.bfloat16:
      return NDArray.scope(() {
        final doubleY = castNDArray<Float64>(y, DType.float64);
        final doubleRes = trapz<Float64>(doubleY, spacing: spacing, axis: axis);
        if (out != null) {
          if (out.dtype == DType.float64) {
            doubleRes.copy(out: out as NDArray<Float64>);
          } else {
            final casted = switch (y.dtype) {
              DType.float16 => castNDArray<Float16>(doubleRes, DType.float16),
              _ => castNDArray<BFloat16>(doubleRes, DType.bfloat16),
            };
            casted.copy(out: out);
          }
          return out;
        }
        if (T == Float16) {
          return castNDArray<Float16>(
                doubleRes,
                DType.float16,
              ).detachToParentScope()
              as NDArray<T>;
        }
        if (T == BFloat16) {
          return castNDArray<BFloat16>(
                doubleRes,
                DType.bfloat16,
              ).detachToParentScope()
              as NDArray<T>;
        }
        if (T == Float64) {
          return doubleRes.detachToParentScope() as NDArray<T>;
        }
        final NDArray<DTypeTag> casted = switch (y.dtype) {
          DType.float16 => castNDArray<Float16>(doubleRes, DType.float16),
          _ => castNDArray<BFloat16>(doubleRes, DType.bfloat16),
        };
        return casted.detachToParentScope() as NDArray<T>;
      });
    case DType.float64:
    case DType.float32:
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
    case DType.boolean:
    case DType.complex128:
    case DType.complex64:
      break;
  }

  return NDArray.scope(() {
    final NDArray<T> result =
        out ??
        (switch (y.dtype) {
              DType.float64 => NDArray<Float64>.zeros(
                targetShape,
                DType.float64,
              ),
              DType.float32 => NDArray<Float32>.zeros(
                targetShape,
                DType.float32,
              ),
              DType.complex128 => NDArray<Complex128>.zeros(
                targetShape,
                DType.complex128,
              ),
              DType.complex64 => NDArray<Complex64>.zeros(
                targetShape,
                DType.complex64,
              ),
              _ => NDArray.zeros(targetShape, y.dtype),
            }
            as NDArray<T>);

    final rank = y.shape.length;
    final marker = ScratchArena.marker;
    try {
      final cShape = ScratchArena.copyInts(y.shape);
      final cStridesY = ScratchArena.copyInts(y.strides);
      final cStridesRes = ScratchArena.copyInts(result.strides);
      switch (spacing) {
        case StepSpacing():
          final value = spacing.value;
          if (value is Complex) {
            switch (y.dtype) {
              case DType.complex128:
                final dxStruct = ScratchArena.allocate<cpx_t>(
                  ffi.sizeOf<cpx_t>(),
                );
                dxStruct.ref.r = value.real;
                dxStruct.ref.i = value.imag;
                s_trapz_complex128_all(
                  y.pointer.cast(),
                  cStridesY,
                  ffi.nullptr,
                  0,
                  dxStruct.ref,
                  result.pointer.cast(),
                  cStridesRes,
                  cShape,
                  rank,
                  targetAxis,
                );
                checkNativeOom();
              case DType.complex64:
                final dxStruct = ScratchArena.allocate<cpx_f_t>(
                  ffi.sizeOf<cpx_f_t>(),
                );
                dxStruct.ref.r = value.real;
                dxStruct.ref.i = value.imag;
                s_trapz_complex64_all(
                  y.pointer.cast(),
                  cStridesY,
                  ffi.nullptr,
                  0,
                  dxStruct.ref,
                  result.pointer.cast(),
                  cStridesRes,
                  cShape,
                  rank,
                  targetAxis,
                );
                checkNativeOom();
              case DType.float64:
              case DType.float32:
              case DType.float16:
              case DType.bfloat16:
              case DType.int64:
              case DType.int32:
              case DType.int16:
              case DType.int8:
              case DType.uint64:
              case DType.uint32:
              case DType.uint16:
              case DType.uint8:
              case DType.boolean:
                throw ArgumentError.value(
                  y.dtype,
                  'y.dtype',
                  'Must be a supported dtype for trapz with complex spacing (unsupported DType for trapz: ${y.dtype})',
                );
            }
          } else if (value is num) {
            final dxVal = value.toDouble();
            final dtype = y.dtype;
            switch (dtype) {
              case DType.float64:
                s_trapz_double(
                  y.pointer.cast(),
                  cStridesY,
                  ffi.nullptr,
                  0,
                  dxVal,
                  result.pointer.cast(),
                  cStridesRes,
                  cShape,
                  rank,
                  targetAxis,
                );
                checkNativeOom();
              case DType.float32:
                s_trapz_float(
                  y.pointer.cast(),
                  cStridesY,
                  ffi.nullptr,
                  0,
                  dxVal,
                  result.pointer.cast(),
                  cStridesRes,
                  cShape,
                  rank,
                  targetAxis,
                );
                checkNativeOom();
              case DType.complex128:
                s_trapz_complex128(
                  y.pointer.cast(),
                  cStridesY,
                  ffi.nullptr,
                  0,
                  dxVal,
                  result.pointer.cast(),
                  cStridesRes,
                  cShape,
                  rank,
                  targetAxis,
                );
                checkNativeOom();
              case DType.complex64:
                s_trapz_complex64(
                  y.pointer.cast(),
                  cStridesY,
                  ffi.nullptr,
                  0,
                  dxVal,
                  result.pointer.cast(),
                  cStridesRes,
                  cShape,
                  rank,
                  targetAxis,
                );
                checkNativeOom();
              case DType.float16:
              case DType.bfloat16:
              case DType.int64:
              case DType.int32:
              case DType.int16:
              case DType.int8:
              case DType.uint64:
              case DType.uint32:
              case DType.uint16:
              case DType.uint8:
              case DType.boolean:
                throw ArgumentError.value(
                  dtype,
                  'dtype',
                  'Must be a supported dtype for trapz (unsupported DType for trapz: $dtype)',
                );
            }
          }

        case CoordinateSpacing():
          final values = spacing.values;
          if (values.every((e) => e is Complex)) {
            final complexValues = values.cast<Complex>();
            switch (y.dtype) {
              case DType.complex128:
                final dxStruct = ScratchArena.allocate<cpx_t>(
                  ffi.sizeOf<cpx_t>(),
                );
                dxStruct.ref.r = 1.0;
                dxStruct.ref.i = 0.0;
                NDArray<DTypeTag>? spacingArray;
                try {
                  spacingArray = NDArray<DTypeTag>.fromList(complexValues, [
                    N,
                  ], DType.complex128);
                  s_trapz_complex128_all(
                    y.pointer.cast(),
                    cStridesY,
                    spacingArray.pointer.cast(),
                    spacingArray.strides[0],
                    dxStruct.ref,
                    result.pointer.cast(),
                    cStridesRes,
                    cShape,
                    rank,
                    targetAxis,
                  );
                  checkNativeOom();
                } finally {
                  spacingArray?.dispose();
                }
              case DType.complex64:
                final dxStruct = ScratchArena.allocate<cpx_f_t>(
                  ffi.sizeOf<cpx_f_t>(),
                );
                dxStruct.ref.r = 1.0;
                dxStruct.ref.i = 0.0;
                NDArray<DTypeTag>? spacingArray;
                try {
                  spacingArray = NDArray<DTypeTag>.fromList(complexValues, [
                    N,
                  ], DType.complex64);
                  s_trapz_complex64_all(
                    y.pointer.cast(),
                    cStridesY,
                    spacingArray.pointer.cast(),
                    spacingArray.strides[0],
                    dxStruct.ref,
                    result.pointer.cast(),
                    cStridesRes,
                    cShape,
                    rank,
                    targetAxis,
                  );
                  checkNativeOom();
                } finally {
                  spacingArray?.dispose();
                }
              case DType.float64:
              case DType.float32:
              case DType.float16:
              case DType.bfloat16:
              case DType.int64:
              case DType.int32:
              case DType.int16:
              case DType.int8:
              case DType.uint64:
              case DType.uint32:
              case DType.uint16:
              case DType.uint8:
              case DType.boolean:
                throw ArgumentError.value(
                  y.dtype,
                  'y.dtype',
                  'Must be a supported dtype for trapz with complex coordinate spacing (unsupported DType for trapz: ${y.dtype})',
                );
            }
          } else {
            NDArray<DTypeTag>? spacingArray;
            try {
              final bool useFloat =
                  yDType == DType.float32 || yDType == DType.complex64;
              if (useFloat) {
                spacingArray = NDArray<Float32>.create([N], DType.float32);
                var i = 0;
                for (final val in values) {
                  spacingArray.setCellFlat(i++, (val as num).toDouble());
                }
              } else {
                spacingArray = NDArray<Float64>.create([N], DType.float64);
                var i = 0;
                for (final val in values) {
                  spacingArray.setCellFlat(i++, (val as num).toDouble());
                }
              }

              final dtype = yDType;
              switch (dtype) {
                case DType.float64:
                  s_trapz_double(
                    y.pointer.cast(),
                    cStridesY,
                    spacingArray.pointer.cast(),
                    spacingArray.strides[0],
                    0.0,
                    result.pointer.cast(),
                    cStridesRes,
                    cShape,
                    rank,
                    targetAxis,
                  );
                  checkNativeOom();
                case DType.float32:
                  s_trapz_float(
                    y.pointer.cast(),
                    cStridesY,
                    spacingArray.pointer.cast(),
                    spacingArray.strides[0],
                    0.0,
                    result.pointer.cast(),
                    cStridesRes,
                    cShape,
                    rank,
                    targetAxis,
                  );
                  checkNativeOom();
                case DType.complex128:
                  s_trapz_complex128(
                    y.pointer.cast(),
                    cStridesY,
                    spacingArray.pointer.cast(),
                    spacingArray.strides[0],
                    0.0,
                    result.pointer.cast(),
                    cStridesRes,
                    cShape,
                    rank,
                    targetAxis,
                  );
                  checkNativeOom();
                case DType.complex64:
                  s_trapz_complex64(
                    y.pointer.cast(),
                    cStridesY,
                    spacingArray.pointer.cast(),
                    spacingArray.strides[0],
                    0.0,
                    result.pointer.cast(),
                    cStridesRes,
                    cShape,
                    rank,
                    targetAxis,
                  );
                  checkNativeOom();
                case DType.float16:
                case DType.bfloat16:
                case DType.int64:
                case DType.int32:
                case DType.int16:
                case DType.int8:
                case DType.uint64:
                case DType.uint32:
                case DType.uint16:
                case DType.uint8:
                case DType.boolean:
                  throw ArgumentError.value(
                    dtype,
                    'dtype',
                    'Must be a supported dtype for trapz (unsupported DType for trapz: $dtype)',
                  );
              }
            } finally {
              spacingArray?.dispose();
            }
          }
      }
    } finally {
      ScratchArena.reset(marker);
    }

    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}

/// Calculates the N-Dimensional gradient along a single axis.
///
/// Returns a single [NDArray] representing the derivative along [axis].
/// For a 1D array, this is equivalent to `gradientArray(f)[0]`.
/// To calculate gradients along multiple axes at once, use [gradientArray].
///
/// The gradient is calculated using second-order accurate central differences
/// for interior points:
///
/// \[
/// f'(x_i) \approx \frac{f(x_{i+1}) - f(x_{i-1})}{x_{i+1} - x_{i-1}}
/// \]
///
/// **Boundary accuracy ([edgeOrder]):**
/// At the edges of the array, central differences cannot be used:
/// - **`edgeOrder = 1` (First-order one-sided differences):**
///   - Start boundary: \(f'(x_0) \approx \frac{f(x_1) - f(x_0)}{x_1 - x_0}\)
///   - End boundary: \(f'(x_{N-1}) \approx \frac{f(x_{N-1}) - f(x_{N-2})}{x_{N-1} - x_{N-2}}\)
/// - **`edgeOrder = 2` (Second-order one-sided differences):**
///   Provides higher precision at the boundaries by utilizing three neighboring points.
///
/// The [axis] parameter specifies the dimension of the array along which the
/// derivative is computed. A negative value is resolved relative to the end of
/// the dimensions, where `-1` represents the last dimension, `-2` represents
/// the second-to-last, and so on.
///
/// Spacing along the axis is specified by [spacing].
///
/// **Preconditions:**
/// - Input [f] must not be disposed.
/// - Input [f] must be a floating-point or complex type.
/// - [axis] must be within bounds `[-f.rank, f.rank - 1]`.
/// - If [spacing] is [CoordinateSpacing], its length must match `f.shape[axis]`.
/// - If [spacing] is complex, input [f] must also be complex.
/// - If [out] is provided, it must match the resolved shape and dtype.
/// - It is an error if [f] or [out] is disposed.
/// - It is an error if [f] has a boolean dtype.
/// - It is an error if complex spacing is used with a real input array.
/// - It is an error if [axis] is out of bounds or spacing is invalid.
/// - It is an error if [edgeOrder] is not 1 or 2.
///
/// **Memory Ownership & Lifetime:**
/// - Allocates a new array on the unmanaged C heap. **The caller takes full ownership** of this memory and **must explicitly call [dispose]** to prevent native leaks, unless executing inside a managed [NDArray.scope()].
///
/// **Example:**
/// {@example /example/calculus_example.dart lang=dart}
NDArray<T> gradient<T extends DTypeTag>(
  NDArray<
    DTypeSpec<DTypeTag, Object?, DTypeTag, DTypeTag, T, DTypeTag, DTypeTag>
  >
  f, {
  Spacing spacing = const Spacing.step(1.0),
  int axis = 0,
  int edgeOrder = 1,
  NDArray<T>? out,
}) {
  if (f.isDisposed) {
    throw StateError('Cannot execute gradient() on a disposed array.');
  }
  if (out != null) {
    if (out.isDisposed) {
      throw StateError(
        'Cannot write gradient result to a disposed output array.',
      );
    }
    validateOutBuffer(out);
  }
  if (edgeOrder != 1 && edgeOrder != 2) {
    throw ArgumentError.value(
      edgeOrder,
      'edgeOrder',
      'Must be 1 or 2 (was $edgeOrder)',
    );
  }

  final DType<DTypeTag> fDType = f.dtype;
  if (fDType == DType.boolean) {
    throw ArgumentError.value(
      f.dtype,
      'f',
      'Must not be boolean (calculus operations are not supported on boolean arrays; cast to a floating-point or complex type first)',
    );
  }

  final bool isComplexSpacing = switch (spacing) {
    Spacing<Complex>() => true,
    StepSpacing(:final value) => value is Complex,
    CoordinateSpacing(:final values) => values.any((v) => v is Complex),
  };
  if (isComplexSpacing && !f.dtype.isComplex) {
    throw ArgumentError.value(
      f.dtype,
      'f',
      'Must be complex when spacing is complex (cast the array to complex first)',
    );
  }

  if (axis < -f.rank || axis >= f.rank) {
    throw RangeError.range(
      axis,
      -f.rank,
      f.rank - 1,
      'axis',
      'Must be within valid rank range',
    );
  }
  final targetAxis = axis < 0 ? f.rank + axis : axis;

  final N = f.shape[targetAxis];
  final minSize = edgeOrder == 2 ? 3 : 2;
  if (N < minSize) {
    throw ArgumentError.value(
      N,
      'f.shape[$targetAxis]',
      'Must be at least $minSize for edgeOrder=$edgeOrder (dimension size $N along axis $targetAxis is too small)',
    );
  }
  if (spacing is CoordinateSpacing) {
    if (spacing.values.length != N) {
      throw ArgumentError.value(
        spacing.values.length,
        'spacing',
        'Must have length matching dimension size $N (got ${spacing.values.length})',
      );
    }
  }

  if (f.dtype.isInteger) {
    if (out != null) {
      if (!_listEquals(out.shape, f.shape) || out.dtype == DType.boolean) {
        throw ArgumentError.value(
          out,
          'out',
          'Must have compatible shape ${f.shape} and non-boolean dtype (got shape ${out.shape}, dtype ${out.dtype})',
        );
      }
    }
    return NDArray.scope(() {
      final doubleF = castNDArray<Float64>(f, DType.float64);
      final doubleRes = gradient<Float64>(
        doubleF,
        spacing: spacing,
        axis: axis,
        edgeOrder: edgeOrder,
      );
      if (out != null) {
        if (out.dtype == DType.float64) {
          doubleRes.copy(out: out as NDArray<Float64>);
        } else {
          final casted = castNDArray(doubleRes, out.dtype);
          casted.copy(out: out);
        }
        return out;
      }
      return doubleRes.detachToParentScope() as NDArray<T>;
    });
  }

  if (out != null) {
    final validDType =
        out.dtype == fDType ||
        ((fDType == DType.float16 || fDType == DType.bfloat16) &&
            out.dtype == DType.float64);
    if (!_listEquals(out.shape, f.shape) || !validDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape ${f.shape} and dtype ${f.dtype} (got shape ${out.shape}, dtype ${out.dtype})',
      );
    }
    if (sharesMemory(f, out)) {
      return NDArray.scope(() {
        final temp = gradient<T>(
          f,
          spacing: spacing,
          axis: axis,
          edgeOrder: edgeOrder,
        );
        temp.copy(out: out);
        return out;
      });
    }
  }

  if (fDType == DType.float16 || fDType == DType.bfloat16) {
    return NDArray.scope(() {
      final doubleF = castNDArray<Float64>(f, DType.float64);
      final doubleRes = gradient<Float64>(
        doubleF,
        spacing: spacing,
        axis: axis,
        edgeOrder: edgeOrder,
      );
      if (out != null) {
        if (out.dtype == DType.float64) {
          doubleRes.copy(out: out as NDArray<Float64>);
        } else {
          final casted = fDType == DType.float16
              ? castNDArray<Float16>(doubleRes, DType.float16)
              : castNDArray<BFloat16>(doubleRes, DType.bfloat16);
          casted.copy(out: out);
        }
        return out;
      }
      if (T == Float16) {
        return castNDArray<Float16>(
              doubleRes,
              DType.float16,
            ).detachToParentScope()
            as NDArray<T>;
      }
      if (T == BFloat16) {
        return castNDArray<BFloat16>(
              doubleRes,
              DType.bfloat16,
            ).detachToParentScope()
            as NDArray<T>;
      }
      if (T == Float64) {
        return doubleRes.detachToParentScope() as NDArray<T>;
      }
      final NDArray<DTypeTag> casted = fDType == DType.float16
          ? castNDArray<Float16>(doubleRes, DType.float16)
          : castNDArray<BFloat16>(doubleRes, DType.bfloat16);
      return casted.detachToParentScope() as NDArray<T>;
    });
  }

  return NDArray.scope(() {
    final NDArray<T> result =
        out ??
        (switch (f.dtype) {
              DType.float64 => NDArray<Float64>.zeros(f.shape, DType.float64),
              DType.float32 => NDArray<Float32>.zeros(f.shape, DType.float32),
              DType.complex128 => NDArray<Complex128>.zeros(
                f.shape,
                DType.complex128,
              ),
              DType.complex64 => NDArray<Complex64>.zeros(
                f.shape,
                DType.complex64,
              ),
              _ => throw ArgumentError.value(
                f.dtype,
                'f.dtype',
                'Must be a supported dtype for gradient (unsupported DType for gradient: ${f.dtype})',
              ),
            }
            as NDArray<T>);

    final rank = f.shape.length;
    final marker = ScratchArena.marker;
    try {
      final cShape = ScratchArena.copyInts(f.shape);
      final cStridesF = ScratchArena.copyInts(f.strides);
      final cStridesRes = ScratchArena.copyInts(result.strides);
      switch (spacing) {
        case StepSpacing():
          final value = spacing.value;
          if (value is Complex) {
            switch (f.dtype) {
              case DType.complex128:
                final dxStruct = ScratchArena.allocate<cpx_t>(
                  ffi.sizeOf<cpx_t>(),
                );
                dxStruct.ref.r = value.real;
                dxStruct.ref.i = value.imag;
                s_gradient_complex128_all(
                  f.pointer.cast(),
                  cStridesF,
                  ffi.nullptr,
                  0,
                  dxStruct.ref,
                  result.pointer.cast(),
                  cStridesRes,
                  cShape,
                  rank,
                  targetAxis,
                  edgeOrder,
                );
                checkNativeOom();
              case DType.complex64:
                final dxStruct = ScratchArena.allocate<cpx_f_t>(
                  ffi.sizeOf<cpx_f_t>(),
                );
                dxStruct.ref.r = value.real;
                dxStruct.ref.i = value.imag;
                s_gradient_complex64_all(
                  f.pointer.cast(),
                  cStridesF,
                  ffi.nullptr,
                  0,
                  dxStruct.ref,
                  result.pointer.cast(),
                  cStridesRes,
                  cShape,
                  rank,
                  targetAxis,
                  edgeOrder,
                );
                checkNativeOom();
              case DType.float64:
              case DType.float32:
              case DType.float16:
              case DType.bfloat16:
              case DType.int64:
              case DType.int32:
              case DType.int16:
              case DType.int8:
              case DType.uint64:
              case DType.uint32:
              case DType.uint16:
              case DType.uint8:
              case DType.boolean:
                throw ArgumentError.value(
                  f.dtype,
                  'f.dtype',
                  'Must be a supported dtype for gradient with complex spacing (unsupported DType for gradient: ${f.dtype})',
                );
            }
          } else if (value is num) {
            final dxVal = value.toDouble();
            final dtype = f.dtype;
            switch (dtype) {
              case DType.float64:
                s_gradient_double(
                  f.pointer.cast(),
                  cStridesF,
                  ffi.nullptr,
                  0,
                  dxVal,
                  result.pointer.cast(),
                  cStridesRes,
                  cShape,
                  rank,
                  targetAxis,
                  edgeOrder,
                );
                checkNativeOom();
              case DType.float32:
                s_gradient_float(
                  f.pointer.cast(),
                  cStridesF,
                  ffi.nullptr,
                  0,
                  dxVal,
                  result.pointer.cast(),
                  cStridesRes,
                  cShape,
                  rank,
                  targetAxis,
                  edgeOrder,
                );
                checkNativeOom();
              case DType.complex128:
                s_gradient_complex128(
                  f.pointer.cast(),
                  cStridesF,
                  ffi.nullptr,
                  0,
                  dxVal,
                  result.pointer.cast(),
                  cStridesRes,
                  cShape,
                  rank,
                  targetAxis,
                  edgeOrder,
                );
                checkNativeOom();
              case DType.complex64:
                s_gradient_complex64(
                  f.pointer.cast(),
                  cStridesF,
                  ffi.nullptr,
                  0,
                  dxVal,
                  result.pointer.cast(),
                  cStridesRes,
                  cShape,
                  rank,
                  targetAxis,
                  edgeOrder,
                );
                checkNativeOom();
              case DType.float16:
              case DType.bfloat16:
              case DType.int64:
              case DType.int32:
              case DType.int16:
              case DType.int8:
              case DType.uint64:
              case DType.uint32:
              case DType.uint16:
              case DType.uint8:
              case DType.boolean:
                throw ArgumentError.value(
                  dtype,
                  'dtype',
                  'Must be a supported dtype for gradient (unsupported DType for gradient: $dtype)',
                );
            }
          }

        case CoordinateSpacing():
          final values = spacing.values;
          if (values.every((e) => e is Complex)) {
            final complexValues = values.cast<Complex>();
            switch (f.dtype) {
              case DType.complex128:
                final dxStruct = ScratchArena.allocate<cpx_t>(
                  ffi.sizeOf<cpx_t>(),
                );
                dxStruct.ref.r = 1.0;
                dxStruct.ref.i = 0.0;
                NDArray<DTypeTag>? spacingArray;
                try {
                  spacingArray = NDArray<DTypeTag>.fromList(complexValues, [
                    N,
                  ], DType.complex128);
                  s_gradient_complex128_all(
                    f.pointer.cast(),
                    cStridesF,
                    spacingArray.pointer.cast(),
                    spacingArray.strides[0],
                    dxStruct.ref,
                    result.pointer.cast(),
                    cStridesRes,
                    cShape,
                    rank,
                    targetAxis,
                    edgeOrder,
                  );
                  checkNativeOom();
                } finally {
                  spacingArray?.dispose();
                }
              case DType.complex64:
                final dxStruct = ScratchArena.allocate<cpx_f_t>(
                  ffi.sizeOf<cpx_f_t>(),
                );
                dxStruct.ref.r = 1.0;
                dxStruct.ref.i = 0.0;
                NDArray<DTypeTag>? spacingArray;
                try {
                  spacingArray = NDArray<DTypeTag>.fromList(complexValues, [
                    N,
                  ], DType.complex64);
                  s_gradient_complex64_all(
                    f.pointer.cast(),
                    cStridesF,
                    spacingArray.pointer.cast(),
                    spacingArray.strides[0],
                    dxStruct.ref,
                    result.pointer.cast(),
                    cStridesRes,
                    cShape,
                    rank,
                    targetAxis,
                    edgeOrder,
                  );
                  checkNativeOom();
                } finally {
                  spacingArray?.dispose();
                }
              case DType.float64:
              case DType.float32:
              case DType.float16:
              case DType.bfloat16:
              case DType.int64:
              case DType.int32:
              case DType.int16:
              case DType.int8:
              case DType.uint64:
              case DType.uint32:
              case DType.uint16:
              case DType.uint8:
              case DType.boolean:
                throw ArgumentError.value(
                  f.dtype,
                  'f.dtype',
                  'Must be a supported dtype for gradient with complex coordinate spacing (unsupported DType for gradient: ${f.dtype})',
                );
            }
          } else {
            NDArray<DTypeTag>? spacingArray;
            try {
              final bool useFloat =
                  fDType == DType.float32 || fDType == DType.complex64;
              if (useFloat) {
                spacingArray = NDArray<Float32>.create([N], DType.float32);
                var i = 0;
                for (final val in values) {
                  spacingArray.setCellFlat(i++, (val as num).toDouble());
                }
              } else {
                spacingArray = NDArray<Float64>.create([N], DType.float64);
                var i = 0;
                for (final val in values) {
                  spacingArray.setCellFlat(i++, (val as num).toDouble());
                }
              }
              final dtype = fDType;
              switch (dtype) {
                case DType.float64:
                  s_gradient_double(
                    f.pointer.cast(),
                    cStridesF,
                    spacingArray.pointer.cast(),
                    spacingArray.strides[0],
                    0.0,
                    result.pointer.cast(),
                    cStridesRes,
                    cShape,
                    rank,
                    targetAxis,
                    edgeOrder,
                  );
                  checkNativeOom();
                case DType.float32:
                  s_gradient_float(
                    f.pointer.cast(),
                    cStridesF,
                    spacingArray.pointer.cast(),
                    spacingArray.strides[0],
                    0.0,
                    result.pointer.cast(),
                    cStridesRes,
                    cShape,
                    rank,
                    targetAxis,
                    edgeOrder,
                  );
                  checkNativeOom();
                case DType.complex128:
                  s_gradient_complex128(
                    f.pointer.cast(),
                    cStridesF,
                    spacingArray.pointer.cast(),
                    spacingArray.strides[0],
                    0.0,
                    result.pointer.cast(),
                    cStridesRes,
                    cShape,
                    rank,
                    targetAxis,
                    edgeOrder,
                  );
                  checkNativeOom();
                case DType.complex64:
                  s_gradient_complex64(
                    f.pointer.cast(),
                    cStridesF,
                    spacingArray.pointer.cast(),
                    spacingArray.strides[0],
                    0.0,
                    result.pointer.cast(),
                    cStridesRes,
                    cShape,
                    rank,
                    targetAxis,
                    edgeOrder,
                  );
                  checkNativeOom();
                case DType.float16:
                case DType.bfloat16:
                case DType.int64:
                case DType.int32:
                case DType.int16:
                case DType.int8:
                case DType.uint64:
                case DType.uint32:
                case DType.uint16:
                case DType.uint8:
                case DType.boolean:
                  throw ArgumentError.value(
                    dtype,
                    'dtype',
                    'Must be a supported dtype for gradient (unsupported DType for gradient: $dtype)',
                  );
              }
            } finally {
              spacingArray?.dispose();
            }
          }
      }
    } finally {
      ScratchArena.reset(marker);
    }

    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}

/// Calculates the n-dimensional gradient along multiple axes.
///
/// Returns a [List<NDArray>] containing the partial derivatives along each
/// specified [axis]. For a 1D array, this is equivalent to `gradient(f)`.
///
/// To calculate the gradient along a single specific axis, use [gradient].
///
/// Differentiates along the axes specified by [axis] (defaulting to all axes).
/// Returns a list of single-axis gradients.
///
/// Each index in [axis] specifies the dimension along which the derivative
/// is computed. Negative indices are resolved relative to the end of the
/// dimensions, where `-1` represents the last dimension, `-2` represents the
/// second-to-last, and so on.
///
/// Spacing along the axes can be specified in two ways:
/// - [spacing]: A single [Spacing] object applied to all axes (shortcut).
/// - [spacings]: A list of [Spacing] objects, one for each axis being differentiated.
/// - [edgeOrder]: Accuracy of the calculation at the boundaries (1 or 2).
///   See [gradient] for details.
///
/// If neither is provided, constant spacing [Spacing.step(1.0)] is used for all axes.
///
/// **Preconditions:**
/// - Input [f] must not be disposed.
/// - Input [f] must be a floating-point or complex type.
/// - If provided, [axis] elements must be unique and within bounds `[-f.rank, f.rank - 1]`.
/// - [spacing] and [spacings] are mutually exclusive.
/// - If provided, [spacings] length must match the number of axes being differentiated.
/// - It is an error if [f] is disposed.
/// - It is an error if [f] has a boolean dtype.
/// - It is an error if [axis] contains out of bounds or duplicate indices.
/// - It is an error if both [spacing] and [spacings] are provided.
/// - It is an error if [spacings] length does not match the number of axes.
/// - It is an error if [edgeOrder] is not 1 or 2.
///
/// **Memory Ownership & Lifetime:**
/// - Allocates a new list of arrays on the unmanaged C heap. **The caller takes full ownership** of this memory and **must explicitly call [dispose]** on all returned arrays in the list to prevent native leaks, unless executing inside a managed [NDArray.scope()].
///
/// **Example:**
/// {@example /example/calculus_example.dart lang=dart}
List<NDArray<T>> gradientArray<T extends DTypeTag>(
  NDArray<
    DTypeSpec<DTypeTag, Object?, DTypeTag, DTypeTag, T, DTypeTag, DTypeTag>
  >
  f, {
  Spacing? spacing,
  List<Spacing>? spacings,
  List<int>? axis,
  int edgeOrder = 1,
  List<NDArray<T>>? out,
}) {
  if (f.isDisposed) {
    throw StateError('Cannot execute gradientArray() on a disposed array.');
  }

  final DType<DTypeTag> fDType = f.dtype;
  if (fDType == DType.boolean) {
    throw ArgumentError.value(
      f.dtype,
      'f',
      'Must not be boolean (calculus operations are not supported on boolean arrays; cast to a floating-point or complex type first)',
    );
  }

  if (spacing != null && spacings != null) {
    throw ArgumentError.value(
      [spacing, spacings],
      'spacing, spacings',
      'Must not specify both (spacing and spacings are mutually exclusive)',
    );
  }

  // Resolve axes
  final List<int> targetAxes;
  if (axis == null) {
    targetAxes = List<int>.generate(f.shape.length, (i) => i);
  } else {
    targetAxes = [];
    for (final ax in axis) {
      if (ax < -f.rank || ax >= f.rank) {
        throw RangeError.range(
          ax,
          -f.rank,
          f.rank - 1,
          'axis',
          'Must be within valid rank range',
        );
      }
      final resolvedAx = ax < 0 ? f.rank + ax : ax;
      if (targetAxes.contains(resolvedAx)) {
        throw ArgumentError.value(
          ax,
          'axis',
          'Must not contain duplicate axes (axis index $ax specified multiple times)',
        );
      }
      targetAxes.add(resolvedAx);
    }
  }

  final minSize = edgeOrder == 2 ? 3 : 2;
  for (final ax in targetAxes) {
    if (f.shape[ax] < minSize) {
      throw ArgumentError.value(
        f.shape[ax],
        'f.shape[$ax]',
        'Must be at least $minSize for edgeOrder=$edgeOrder (dimension size ${f.shape[ax]} along axis $ax is too small)',
      );
    }
  }

  if (spacings != null && spacings.length != targetAxes.length) {
    throw ArgumentError.value(
      spacings.length,
      'spacings',
      'Must match the number of axes (${targetAxes.length}) (got ${spacings.length})',
    );
  }

  if (out != null) {
    if (out.length != targetAxes.length) {
      throw ArgumentError.value(
        out.length,
        'out',
        'Must match the number of axes (${targetAxes.length}) (got ${out.length})',
      );
    }
    for (var i = 0; i < out.length; i++) {
      if (out[i].isDisposed) {
        throw StateError(
          'Cannot write gradient result to a disposed output array at index $i.',
        );
      }
      validateOutBuffer(out[i], 'out[$i]');
      final validOutDType = fDType.isInteger
          ? out[i].dtype != DType.boolean
          : (out[i].dtype == fDType ||
                ((fDType == DType.float16 || fDType == DType.bfloat16) &&
                    out[i].dtype == DType.float64));
      if (!listEquals(out[i].shape, f.shape) || !validOutDType) {
        throw ArgumentError.value(
          out[i],
          'out[$i]',
          'Must have compatible shape and dtype (provided out buffer at index $i has incompatible shape or dtype)',
        );
      }
      for (var j = 0; j < i; j++) {
        if (sharesMemory(out[i], out[j])) {
          throw ArgumentError.value(
            out[i],
            'out[$i]',
            'Must not share memory with out[$j]',
          );
        }
      }
    }
    if (targetAxes.length > 1 && out.any((o) => sharesMemory(f, o))) {
      return NDArray.scope(() {
        final fCopy = f.copy();
        return gradientArray<T>(
          fCopy,
          spacing: spacing,
          spacings: spacings,
          axis: axis,
          edgeOrder: edgeOrder,
          out: out,
        );
      });
    }
  }

  final List<NDArray<T>> results = [];
  try {
    for (var i = 0; i < targetAxes.length; i++) {
      final singleGrad = gradient<T>(
        f,
        spacing: spacings?[i] ?? spacing ?? const Spacing.step(1.0),
        axis: targetAxes[i],
        edgeOrder: edgeOrder,
        out: out?[i],
      );
      results.add(singleGrad);
    }
  } catch (e) {
    // Clean up any successful allocations if one fails
    if (out == null) {
      for (final res in results) {
        res.dispose();
      }
    }
    rethrow;
  }

  return out ?? results;
}
