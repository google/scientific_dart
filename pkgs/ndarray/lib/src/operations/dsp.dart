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

import 'dart:math' as math;
import 'dart:ffi' as ffi;
import '../ndarray.dart';
import '../ndarray_bindings.dart' as bindings;

import 'padding.dart';
import '../scratch_arena.dart';
import 'helpers.dart';

/// Computes the element-wise phase/argument of complex or real numbers.
///
/// Returns an array of float or double with values in $[-\pi, \pi]$.
/// For real inputs, the angle is $0.0$ for non-negative values and $\pi$ for negative values.
///
/// **Preconditions:**
/// - Input array [a] must not be disposed.
/// - If provided, [out] must match the shape of [a] and the corresponding floating-point dtype.
///
/// **Throws:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] has boolean dtype.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Uses native vectorized C `v_angle` for contiguous arrays and `s_angle` for strided layouts.
///
/// **Example:**
/// {@example /example/dsp_example.dart lang=dart}
///
/// Reference: [NumPy angle](https://numpy.org/doc/stable/reference/generated/numpy.angle.html)
NDArray<R> angle<R extends DTypeTag>(
  NDArray<RealFloatOf<R>> a, {
  NDArray<R>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute angle() on a disposed array.');
  }

  final DType<R> targetDType = switch (a.dtype) {
    DType.complex128 => DType.float64 as DType<R>,
    DType.complex64 => DType.float32 as DType<R>,
    DType.float32 => DType.float32 as DType<R>,
    DType.float16 ||
    DType.bfloat16 ||
    DType.float64 ||
    DType.int64 ||
    DType.int32 ||
    DType.int16 ||
    DType.int8 ||
    DType.uint64 ||
    DType.uint32 ||
    DType.uint16 ||
    DType.uint8 => DType.float64 as DType<R>,
    DType.boolean => throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      'Must not be boolean',
    ),
  };

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype',
      );
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = angle<R>(a);
        temp.copy(out: out);
        return out;
      });
    }
  }

  final NDArray<R> result = out ?? NDArray.create(a.shape, targetDType);

  try {
    switch (a.dtype) {
      case DType.complex128:
        if (a.isContiguous && result.isContiguous) {
          bindings.v_angle_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            ffi.nullptr,
          );
        } else {
          final rank = a.shape.length;
          final marker = ScratchArena.marker;
          try {
            final cBuffer = ScratchArena.getStridedBuffer(rank);
            final cShape = cBuffer;
            final cStridesA = cBuffer + rank;
            final cStridesRes = cBuffer + (rank * 2);
            for (var i = 0; i < rank; i++) {
              cShape[i] = a.shape[i];
              cStridesA[i] = a.strides[i];
              cStridesRes[i] = result.strides[i];
            }
            bindings.s_angle_complex128(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            checkNativeOom();
          } finally {
            ScratchArena.reset(marker);
          }
        }
        return result;
      case DType.complex64:
        if (a.isContiguous && result.isContiguous) {
          bindings.v_angle_complex64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            ffi.nullptr,
          );
        } else {
          final rank = a.shape.length;
          final marker = ScratchArena.marker;
          try {
            final cBuffer = ScratchArena.getStridedBuffer(rank);
            final cShape = cBuffer;
            final cStridesA = cBuffer + rank;
            final cStridesRes = cBuffer + (rank * 2);
            for (var i = 0; i < rank; i++) {
              cShape[i] = a.shape[i];
              cStridesA[i] = a.strides[i];
              cStridesRes[i] = result.strides[i];
            }
            bindings.s_angle_complex64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            checkNativeOom();
          } finally {
            ScratchArena.reset(marker);
          }
        }
        return result;
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
        NDArray.scope(() {
          final doubleA = castNDArray<Float64>(a, DType.float64);
          final doubleRes = NDArray<Float64>.create(a.shape, DType.float64);
          unaryOp<Float64, Float64>(
            doubleRes,
            doubleA,
            a.shape,
            doubleA.strides,
            doubleRes.strides,
            0,
            0,
            0,
            (val) {
              // `doubleA` was just cast to float64, so the element is a double.
              final v = val as double;
              if (v.isNaN) return double.nan;
              return (v < 0.0 || identical(v, -0.0)) ? math.pi : 0.0;
            },
          );
          final casted = castNDArray<R>(doubleRes, result.dtype);
          casted.copy(out: result);
        });
        return result;
      case DType.boolean:
        throw ArgumentError.value(a.dtype, 'a.dtype', 'Must not be boolean');
    }
  } catch (_) {
    if (out == null) {
      result.dispose();
    }
    rethrow;
  }
}

/// Unwraps radian phase angles by changing absolute jumps greater than [discont]
/// to their $2\pi$ complement along the given [axis].
///
/// **Preconditions:**
/// - Input array [a] must not be disposed.
/// - The specified [axis] must be within valid rank bounds `[-a.rank, a.rank - 1]`.
/// - If provided, [out] must match the shape and expected dtype of [a].
///
/// **Throws:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] has boolean or complex dtype.
/// - It is an error if [axis] is out of bounds.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Uses native C strided routines (`s_unwrap_double` / `s_unwrap_float`) to efficiently unwrap along any axis.
///
/// **Example:**
/// {@example /example/dsp_example.dart lang=dart}
///
/// Reference: [NumPy unwrap](https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html)
NDArray<T> unwrap<T extends DTypeTag>(
  NDArray<InexactOf<T>> a, {
  double discont = math.pi,
  int axis = -1,
  NDArray<T>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute unwrap() on a disposed array.');
  }

  final DType<DTypeTag> aDType = a.dtype;
  if (aDType == DType.boolean || aDType.isComplex) {
    throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      'Must not be boolean or complex',
    );
  }

  final rank = a.shape.length;
  if (axis < -rank || axis >= rank) {
    throw RangeError.range(
      axis,
      -rank,
      rank - 1,
      'axis',
      'Must be within valid rank range',
    );
  }
  final resolvedAxis = axis < 0 ? rank + axis : axis;

  final DType targetDType =
      out?.dtype ??
      ((aDType.isInteger || aDType == DType.float16 || aDType == DType.bfloat16)
          ? DType.float64
          : aDType);

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) ||
        out.dtype == DType.boolean ||
        out.dtype.isComplex ||
        ((aDType == DType.float64 || aDType == DType.float32) &&
            out.dtype != aDType)) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype',
      );
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<T>.create(out.shape, out.dtype);
        unwrap<T>(a, discont: discont, axis: axis, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  final NDArray<T> result =
      out ??
      (switch (targetDType) {
            DType.float64 => NDArray<Float64>.create(a.shape, DType.float64),
            DType.float32 => NDArray<Float32>.create(a.shape, DType.float32),
            DType.float16 => NDArray<Float16>.create(a.shape, DType.float16),
            DType.bfloat16 => NDArray<BFloat16>.create(a.shape, DType.bfloat16),
            DType.int64 => NDArray<Int64>.create(a.shape, DType.int64),
            DType.int32 => NDArray<Int32>.create(a.shape, DType.int32),
            DType.int16 => NDArray<Int16>.create(a.shape, DType.int16),
            DType.int8 => NDArray<Int8>.create(a.shape, DType.int8),
            DType.uint64 => NDArray<Uint64>.create(a.shape, DType.uint64),
            DType.uint32 => NDArray<Uint32>.create(a.shape, DType.uint32),
            DType.uint16 => NDArray<Uint16>.create(a.shape, DType.uint16),
            DType.uint8 => NDArray<Uint8>.create(a.shape, DType.uint8),
            _ => throw ArgumentError.value(
              targetDType,
              'targetDType',
              'Must not be boolean or complex',
            ),
          }
          as NDArray<T>);

  try {
    switch (a.dtype) {
      case DType.float64:
        final marker = ScratchArena.marker;
        try {
          final rankForBuffer = a.shape.length;
          final cBuffer = ScratchArena.getStridedBuffer(rankForBuffer);
          final cShape = cBuffer;
          final cStridesA = cBuffer + rankForBuffer;
          final cStridesRes = cBuffer + (rankForBuffer * 2);
          for (var i = 0; i < rankForBuffer; i++) {
            cShape[i] = a.shape[i];
            cStridesA[i] = a.strides[i];
            cStridesRes[i] = result.strides[i];
          }
          bindings.s_unwrap_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rankForBuffer,
            resolvedAxis,
            discont,
          );
          checkNativeOom();
        } finally {
          ScratchArena.reset(marker);
        }
        return result;
      case DType.float32:
        final marker = ScratchArena.marker;
        try {
          final rankForBuffer = a.shape.length;
          final cBuffer = ScratchArena.getStridedBuffer(rankForBuffer);
          final cShape = cBuffer;
          final cStridesA = cBuffer + rankForBuffer;
          final cStridesRes = cBuffer + (rankForBuffer * 2);
          for (var i = 0; i < rankForBuffer; i++) {
            cShape[i] = a.shape[i];
            cStridesA[i] = a.strides[i];
            cStridesRes[i] = result.strides[i];
          }
          bindings.s_unwrap_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rankForBuffer,
            resolvedAxis,
            discont,
          );
          checkNativeOom();
        } finally {
          ScratchArena.reset(marker);
        }
        return result;
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
        NDArray.scope(() {
          final doubleA = castNDArray<Float64>(a, DType.float64);
          final doubleRes = unwrap<Float64>(
            doubleA,
            discont: discont,
            axis: axis,
          );
          final casted = castNDArray(doubleRes, result.dtype);
          casted.copy(out: result);
        });
        return result;
      case DType.boolean:
      case DType.complex128:
      case DType.complex64:
        throw ArgumentError.value(
          a.dtype,
          'a.dtype',
          'Must not be boolean or complex',
        );
    }
  } catch (_) {
    if (out == null) {
      result.dispose();
    }
    rethrow;
  }
}

/// Internal helper executing direct stencil N-D valid cross-correlation.
NDArray<R> _correlateValid<
  T extends DTypeTag,
  K extends DTypeTag,
  R extends DTypeTag
>(NDArray<T> in1, NDArray<K> in2, {NDArray<R>? out}) {
  final rank = in1.rank;
  final outShape = List<int>.generate(
    rank,
    (i) => in1.shape[i] - in2.shape[i] + 1,
  );

  if (out != null && out.dtype != in1.dtype) {
    throw ArgumentError.value(
      out.dtype,
      'out.dtype',
      'Must match in1 dtype (${in1.dtype})',
    );
  }
  if (out != null && (sharesMemory(in1, out) || sharesMemory(in2, out))) {
    return NDArray.scope(() {
      final temp = _correlateValid<T, K, R>(in1, in2);
      temp.copy(out: out);
      return out;
    });
  }
  final DType<R> targetDType = out?.dtype ?? (in1.dtype as DType<R>);
  final result = out ?? NDArray<R>.zeros(outShape, targetDType);

  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.allocate<ffi.Int64>(
      rank * 5 * ffi.sizeOf<ffi.Int64>(),
    );
    final cStrides1 = cBuffer;
    final cStrides2 = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    final cShapeRes = cBuffer + (rank * 3);
    final cShapeK = cBuffer + (rank * 4);

    for (var i = 0; i < rank; i++) {
      cStrides1[i] = in1.strides[i];
      cStrides2[i] = in2.strides[i];
      cStridesRes[i] = result.strides[i];
      cShapeRes[i] = result.shape[i];
      cShapeK[i] = in2.shape[i];
    }

    switch (in1.dtype) {
      case DType.float64:
        bindings.s_correlate_valid_double(
          in1.pointer.cast(),
          cStrides1,
          in2.pointer.cast(),
          cStrides2,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          cShapeK,
          rank,
        );
        break;
      case DType.float32:
        bindings.s_correlate_valid_float(
          in1.pointer.cast(),
          cStrides1,
          in2.pointer.cast(),
          cStrides2,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          cShapeK,
          rank,
        );
        break;
      case DType.complex128:
        bindings.s_correlate_valid_complex128(
          in1.pointer.cast(),
          cStrides1,
          in2.pointer.cast(),
          cStrides2,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          cShapeK,
          rank,
        );
        break;
      case DType.complex64:
        bindings.s_correlate_valid_complex64(
          in1.pointer.cast(),
          cStrides1,
          in2.pointer.cast(),
          cStrides2,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          cShapeK,
          rank,
        );
        break;
      case DType.int64:
        bindings.s_correlate_valid_int64(
          in1.pointer.cast(),
          cStrides1,
          in2.pointer.cast(),
          cStrides2,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          cShapeK,
          rank,
        );
        break;
      case DType.int32:
        bindings.s_correlate_valid_int32(
          in1.pointer.cast(),
          cStrides1,
          in2.pointer.cast(),
          cStrides2,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          cShapeK,
          rank,
        );
        break;
      case DType.int16:
      case DType.int8:
      case DType.uint8:
      case DType.uint16:
      case DType.uint32:
        NDArray.scope(() {
          final aPromoted = castNDArray<Int64>(in1, DType.int64);
          final vPromoted = castNDArray<Int64>(in2, DType.int64);
          final outInt64 = NDArray<Int64>.zeros(outShape, DType.int64);
          _correlateValid<Int64, Int64, Int64>(
            aPromoted,
            vPromoted,
            out: outInt64,
          );
          castNDArray<R>(outInt64, targetDType).copy(out: result);
        });
        break;
      case DType.uint64:
        bindings.s_correlate_valid_uint64(
          in1.pointer.cast(),
          cStrides1,
          in2.pointer.cast(),
          cStrides2,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          cShapeK,
          rank,
        );
        break;
      case DType.float16:
      case DType.bfloat16:
        NDArray.scope(() {
          final aPromoted = castNDArray<Float64>(in1, DType.float64);
          final vPromoted = castNDArray<Float64>(in2, DType.float64);
          final outFloat64 = NDArray<Float64>.zeros(outShape, DType.float64);
          _correlateValid<Float64, Float64, Float64>(
            aPromoted,
            vPromoted,
            out: outFloat64,
          );
          castNDArray<R>(outFloat64, targetDType).copy(out: result);
        });
        break;
      case DType.boolean:
        if (out == null) result.dispose();
        throw ArgumentError.value(
          in1.dtype,
          'in1.dtype',
          'Must not be boolean',
        );
    }
  } finally {
    ScratchArena.reset(marker);
  }

  return result;
}

/// Specifies the output array shape mode for cross-correlation and convolution operations.
enum ConvMode {
  /// Output is generated at all positions where input arrays overlap by at least one element.
  ///
  /// For inputs of shape $N$ and $K$, the output shape along axis $i$ is $N_i + K_i - 1$.
  full,

  /// Output is generated only at positions where kernel is fully contained inside input without padding.
  ///
  /// For inputs of shape $N$ and $K$, the output shape along axis $i$ is $N_i - K_i + 1$.
  /// It is an error if any input dimension is smaller than kernel dimension.
  valid,

  /// Output shape matches input array shape $N_i$.
  ///
  /// Output is centered with respect to full correlation/convolution.
  same,
}

/// Computes the N-dimensional discrete cross-correlation of two arrays [in1] and [in2].
///
/// Cross-correlation evaluates the similarity of two signals as a function of the displacement of one relative to the other.
/// For $d$-dimensional input arrays $y = \text{in1}$ and $w = \text{in2}$, the cross-correlation at index $\mathbf{n}$ is:
/// $$z[\mathbf{n}] = \sum_{\mathbf{m}} y[\mathbf{n} + \mathbf{m}] \cdot \overline{w[\mathbf{m}]}$$
///
/// Contrast with [convolve], where the kernel array [in2] is flipped across all axes prior to cross-correlation:
/// $$\text{convolve}(y, w) = \text{correlate}(y, \text{flip}(w))$$
///
/// ### Mode Parameter ([ConvMode])
/// The [mode] parameter controls the output array shape:
/// - [ConvMode.full]: Returns cross-correlation at all overlap positions. Output shape along axis $i$ is $N_i + K_i - 1$.
/// - [ConvMode.valid]: Returns output only where [in2] is completely inside [in1]. Output shape along axis $i$ is $N_i - K_i + 1$.
/// - [ConvMode.same]: Returns output centered to match the shape of [in1] ($N_i$).
///
/// It is an error if [in1], [in2], or [out] is disposed, if [in1] and [in2] have different ranks or rank 0,
/// if [in1] and [in2] have different [DType]s, if [mode] is [ConvMode.valid] and any dimension of [in1] is smaller
/// than the corresponding dimension of [in2], or if [out] has an incompatible shape or dtype.
///
/// ### References & Further Reading
/// - [NumPy correlate Documentation](https://numpy.org/doc/stable/reference/generated/numpy.correlate.html)
/// - [SciPy signal.correlate Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html)
/// - [Wikipedia: Cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation)
NDArray<T> correlate<T extends SelfOf<DTypeTag>>(
  NDArray<T> in1,
  NDArray<T> in2, {
  ConvMode mode = ConvMode.valid,
  NDArray<T>? out,
}) {
  if (in1.isDisposed || in2.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute correlate() on a disposed array.');
  }
  if (in1.rank != in2.rank || in1.rank == 0) {
    throw ArgumentError.value(
      in2.rank,
      'in2.rank',
      'Must have the same non-zero rank as in1 (${in1.rank})',
    );
  }
  if (in1.size == 0 || in2.size == 0) {
    throw ArgumentError.value(
      in1.size == 0 ? in1.size : in2.size,
      in1.size == 0 ? 'in1' : 'in2',
      'Must not be empty',
    );
  }
  if (in1.dtype != in2.dtype) {
    throw ArgumentError.value(
      in2.dtype,
      'in2.dtype',
      'Must match in1 dtype (${in1.dtype})',
    );
  }
  if (out != null) {
    validateOutBuffer(out);
    if (out.dtype != in1.dtype) {
      throw ArgumentError.value(
        out.dtype,
        'out.dtype',
        'Must match in1 dtype (${in1.dtype})',
      );
    }
  }

  final rank = in1.rank;

  switch (mode) {
    case ConvMode.valid:
      for (var i = 0; i < rank; i++) {
        if (in1.shape[i] < in2.shape[i]) {
          throw ArgumentError.value(
            in1.shape[i],
            'in1.shape[$i]',
            'Must be greater than or equal to in2 dimension (${in2.shape[i]})',
          );
        }
      }
      final expectedShape = List<int>.generate(
        rank,
        (i) => in1.shape[i] - in2.shape[i] + 1,
      );
      if (out != null && !listEquals(out.shape, expectedShape)) {
        throw ArgumentError.value(
          out.shape,
          'out.shape',
          'Must have shape $expectedShape',
        );
      }
      if (out != null && (sharesMemory(in1, out) || sharesMemory(in2, out))) {
        return NDArray.scope(() {
          final temp = _correlateValid<T, T, T>(in1, in2);
          temp.copy(out: out);
          return out;
        });
      }
      return _correlateValid<T, T, T>(in1, in2, out: out);
    case ConvMode.full:
      final expectedShape = List<int>.generate(
        rank,
        (i) => in1.shape[i] + in2.shape[i] - 1,
      );
      if (out != null && !listEquals(out.shape, expectedShape)) {
        throw ArgumentError.value(
          out.shape,
          'out.shape',
          'Must have shape $expectedShape',
        );
      }
      if (out != null && (sharesMemory(in1, out) || sharesMemory(in2, out))) {
        return NDArray.scope(() {
          final temp = correlate<T>(in1, in2, mode: ConvMode.full);
          temp.copy(out: out);
          return out;
        });
      }

      return NDArray.scope(() {
        final padWidths = List<(int, int)>.generate(
          rank,
          (i) => (in2.shape[i] - 1, in2.shape[i] - 1),
        );
        final padded1 = pad<T>(
          in1,
          PadWidth.axes(padWidths),
          mode: PaddingMode.constant,
        );
        final res = _correlateValid<T, T, T>(padded1, in2, out: out);
        if (out != null) return out;
        return res.detachToParentScope();
      });
    case ConvMode.same:
      if (out != null && !listEquals(out.shape, in1.shape)) {
        throw ArgumentError.value(
          out.shape,
          'out.shape',
          'Must have shape ${in1.shape}',
        );
      }
      return NDArray.scope(() {
        final fullCorr = correlate<T>(in1, in2, mode: ConvMode.full);
        final selectors = List<Selector>.generate(rank, (i) {
          final start = (in2.shape[i] - 1) ~/ 2;
          return Slice(start: start, stop: start + in1.shape[i]);
        });
        final sliced = fullCorr.slice(selectors);
        if (out != null) {
          sliced.copy(out: out);
          return out;
        }
        return sliced.copy().detachToParentScope();
      });
  }
}

/// Computes the N-dimensional discrete linear convolution of two multi-dimensional arrays [in1] and [in2].
///
/// Linear convolution evaluates the response of a linear time-invariant (LTI) system to an input signal.
/// For $d$-dimensional input arrays $y = \text{in1}$ and $w = \text{in2}$, convolution at index $\mathbf{n}$ is:
/// $$z[\mathbf{n}] = \sum_{\mathbf{m}} y[\mathbf{n} - \mathbf{m}] \cdot w[\mathbf{m}]$$
///
/// Contrast with [correlate], where the kernel array [in2] is used without spatial reversal:
/// $$\text{convolve}(y, w) = \text{correlate}(y, \text{flip}(w))$$
///
/// ### Mode Parameter ([ConvMode])
/// The [mode] parameter controls the output array shape:
/// - [ConvMode.full]: Returns linear convolution at all positions where arrays overlap by at least 1 element.
///   Output shape along axis $i$ is $N_i + K_i - 1$.
/// - [ConvMode.valid]: Returns output only at positions where [in2] is completely inside [in1] without zero padding.
///   Output shape along axis $i$ is $N_i - K_i + 1$.
/// - [ConvMode.same]: Returns output centered to match the exact shape of [in1] ($N_i$).
///
/// It is an error if [in1], [in2], or [out] is disposed, if [in1] and [in2] have different ranks or rank 0,
/// if [in1] and [in2] have different [DType]s, if [mode] is [ConvMode.valid] and any dimension of [in1] is smaller
/// than [in2], or if [out] has an incompatible shape or dtype.
///
/// ### References & Further Reading
/// - [NumPy convolve Documentation](https://numpy.org/doc/stable/reference/generated/numpy.convolve.html)
/// - [SciPy signal.convolve Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html)
/// - [Wikipedia: Convolution](https://en.wikipedia.org/wiki/Convolution)
NDArray<T> convolve<T extends SelfOf<DTypeTag>>(
  NDArray<T> in1,
  NDArray<T> in2, {
  ConvMode mode = ConvMode.full,
  NDArray<T>? out,
}) {
  if (in1.isDisposed || in2.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute convolve() on a disposed array.');
  }
  if (in1.rank != in2.rank || in1.rank == 0) {
    throw ArgumentError.value(
      in2.rank,
      'in2.rank',
      'Must have the same non-zero rank as in1 (${in1.rank})',
    );
  }
  if (in1.size == 0 || in2.size == 0) {
    throw ArgumentError.value(
      in1.size == 0 ? in1.size : in2.size,
      in1.size == 0 ? 'in1' : 'in2',
      'Must not be empty',
    );
  }
  if (in1.dtype != in2.dtype) {
    throw ArgumentError.value(
      in2.dtype,
      'in2.dtype',
      'Must match in1 dtype (${in1.dtype})',
    );
  }
  if (out != null) {
    validateOutBuffer(out);
    if (out.dtype != in1.dtype) {
      throw ArgumentError.value(
        out.dtype,
        'out.dtype',
        'Must match in1 dtype (${in1.dtype})',
      );
    }
  }
  if (out != null && (sharesMemory(in1, out) || sharesMemory(in2, out))) {
    return NDArray.scope(() {
      final temp = convolve<T>(in1, in2, mode: mode);
      if (out.dtype != temp.dtype || !listEquals(out.shape, temp.shape)) {
        throw ArgumentError.value(
          out,
          'out',
          'Must have compatible shape and dtype',
        );
      }
      temp.copy(out: out);
      return out;
    });
  }

  return NDArray.scope(() {
    final reversedSelectors = List<Selector>.generate(
      in2.rank,
      (_) => const Slice(step: -1),
    );
    final flippedKernel = in2.slice(reversedSelectors);
    final contiguousKernel =
        (flippedKernel.isContiguous && !in2.dtype.isComplex)
        ? flippedKernel
        : flippedKernel.copy();
    switch (in2.dtype) {
      case DType.complex128:
        bindings.v_conj_complex128(
          contiguousKernel.pointer.cast(),
          contiguousKernel.pointer.cast(),
          contiguousKernel.size,
          ffi.nullptr,
        );
      case DType.complex64:
        bindings.v_conj_complex64(
          contiguousKernel.pointer.cast(),
          contiguousKernel.pointer.cast(),
          contiguousKernel.size,
          ffi.nullptr,
        );
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
        break;
    }
    final res = correlate<T>(in1, contiguousKernel, mode: mode, out: out);
    if (out != null) return out;
    return res.detachToParentScope();
  });
}

/// Computes 2-dimensional spatial linear convolution of two 2D matrix arrays [in1] and [in2].
///
/// For 2D matrix input $Y = \text{in1}$ of shape $(H_1, W_1)$ and kernel $W = \text{in2}$ of shape $(H_2, W_2)$,
/// 2D spatial convolution at matrix coordinates $(i, j)$ is defined as:
/// $$Z[i, j] = \sum_{m} \sum_{n} Y[i - m, j - n] \cdot W[m, n]$$
///
/// ### Mode Parameter ([ConvMode])
/// The [mode] parameter specifies output matrix dimensions:
/// - [ConvMode.full]: Returns output shape $(H_1 + H_2 - 1, W_1 + W_2 - 1)$.
/// - [ConvMode.valid]: Returns output shape $(H_1 - H_2 + 1, W_1 - W_2 + 1)$.
/// - [ConvMode.same]: Returns output shape $(H_1, W_1)$, centered relative to full convolution.
///
/// It is an error if [in1] or [in2] is not a 2-dimensional array, if [in1], [in2], or [out] is disposed,
/// if [mode] is [ConvMode.valid] and $H_1 < H_2$ or $W_1 < W_2$, or if [out] shape or dtype is invalid.
///
/// ### References & Further Reading
/// - [SciPy signal.convolve2d Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html)
NDArray<T> convolve2d<T extends SelfOf<DTypeTag>>(
  NDArray<T> in1,
  NDArray<T> in2, {
  ConvMode mode = ConvMode.full,
  NDArray<T>? out,
}) {
  if (in1.rank != 2 || in2.rank != 2) {
    throw ArgumentError.value(
      in1.rank != 2 ? in1.rank : in2.rank,
      in1.rank != 2 ? 'in1.rank' : 'in2.rank',
      'Must be 2-dimensional',
    );
  }
  return convolve<T>(in1, in2, mode: mode, out: out);
}
