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

/// Computes the element-wise phase/argument of complex numbers.
///
/// Returns an array of float or double (matching the precision of the complex
/// input, i.e., Float64 for Complex128, Float32 for Complex64) with values
/// in $[-\pi, \pi]$.
///
/// **Preconditions:**
/// - Input array [a] must not be disposed and must have a complex dtype (`complex64` or `complex128`).
/// - If provided, [out] must match the shape of [a] and the corresponding floating-point dtype.
///
/// **Throws:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] dtype is not complex.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Uses native vectorized C `v_angle` for contiguous arrays and `s_angle` for strided layouts.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([Complex(1.0, 1.0)], [1], DType.complex128);
/// final p = angle(a);
/// print(p.toList()); // [0.7853981633974483]
/// ```
///
/// Reference: [NumPy angle](https://numpy.org/doc/stable/reference/generated/numpy.angle.html)
NDArray<R> angle<T extends Complex, R extends double>(
  NDArray<T> a, {
  NDArray<R>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute angle() on a disposed array.');
  }

  if (a.dtype != DType.complex128 && a.dtype != DType.complex64) {
    throw ArgumentError('Input array must be complex for angle().');
  }

  final DType<R> targetDType;
  switch (a.dtype) {
    case DType.complex128:
      targetDType = DType.float64 as DType<R>;
      break;
    case DType.complex64:
      targetDType = DType.float32 as DType<R>;
      break;
  }

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for angle.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType);
  }

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.complex128:
        bindings.v_angle_complex128(
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          ffi.nullptr,
        );
        return result;
      case DType.complex64:
        bindings.v_angle_complex64(
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          ffi.nullptr,
        );
        return result;
    }
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
      switch (a.dtype) {
        case DType.complex128:
          bindings.s_angle_complex128(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.complex64:
          bindings.s_angle_complex64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }
}

/// Unwraps radian phase angles by changing absolute jumps greater than [discont]
/// to their $2\pi$ complement along the given [axis].
///
/// **Preconditions:**
/// - Input array [a] must not be disposed and must have a floating-point dtype (`float32` or `float64`).
/// - The specified [axis] must be within valid rank bounds `[-a.rank, a.rank - 1]`.
/// - If provided, [out] must match the shape and dtype of [a].
///
/// **Throws:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] dtype is not floating-point (`float32` or `float64`).
/// - It is an error if [axis] is out of bounds.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Uses native C strided routines (`s_unwrap_double` / `s_unwrap_float`) to efficiently unwrap along any axis.
///
/// **Example:**
/// ```dart
/// final phase = NDArray.fromList([0.0, math.pi + 0.1, -(math.pi + 0.1)], [3], DType.float64);
/// final unwrapped = unwrap(phase);
/// ```
///
/// Reference: [NumPy unwrap](https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html)
NDArray<T> unwrap<T extends double>(
  NDArray<T> a, {
  double discont = math.pi,
  int axis = -1,
  NDArray<T>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute unwrap() on a disposed array.');
  }

  if (!a.dtype.isFloating) {
    throw ArgumentError('Input array must be float32 or float64 for unwrap().');
  }

  final rank = a.shape.length;
  final resolvedAxis = axis < 0 ? rank + axis : axis;
  if (resolvedAxis < 0 || resolvedAxis >= rank) {
    throw ArgumentError('Invalid axis $axis for shape ${a.shape}');
  }

  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for unwrap.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, a.dtype);
  }

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

    switch (a.dtype) {
      case DType.float64:
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
        return result;
      case DType.float32:
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
        return result;
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
      case DType.int64:
      case DType.int32:
      case DType.int16:
      case DType.uint8:
      case DType.boolean:
      case DType.complex128:
      case DType.complex64:
        final doubleA = NDArray.fromList(
          a.toList().cast<num>().map((e) => e.toDouble()).toList(),
          a.shape,
          DType.float64,
        );
        final doubleRes = unwrap(doubleA, discont: discont, axis: axis);
        final casted = NDArray.fromList(
          doubleRes.toList(),
          doubleRes.shape,
          result.dtype,
        );
        casted.copy(out: result);
        doubleA.dispose();
        doubleRes.dispose();
        casted.dispose();
    }
  } finally {
    ScratchArena.reset(marker);
  }
  return result;
}

/// Internal helper executing direct stencil N-D valid cross-correlation.
NDArray<R> _correlateValid<
  T extends Object,
  K extends Object,
  R extends Object
>(NDArray<T> in1, NDArray<K> in2, {NDArray<R>? out}) {
  final rank = in1.rank;
  final outShape = List<int>.generate(
    rank,
    (i) => in1.shape[i] - in2.shape[i] + 1,
  );

  if (out != null && out.dtype != in1.dtype) {
    throw ArgumentError(
      'Provided out buffer dtype (${out.dtype}) must match in1 dtype (${in1.dtype}).',
    );
  }
  final DType<R> targetDType = out?.dtype ?? (in1.dtype as DType<R>);
  final result = out ?? NDArray<R>.zeros(outShape, targetDType);

  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.allocate<ffi.Int>(
      rank * 5 * ffi.sizeOf<ffi.Int>(),
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
      default:
        throw ArgumentError('Unsupported dtype for correlate');
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
/// $$z[\mathbf{n}] = \sum_{\mathbf{m}} y[\mathbf{n} + \mathbf{m}] \cdot w[\mathbf{m}]$$
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
NDArray<R> correlate<T extends Object, K extends Object, R extends Object>(
  NDArray<T> in1,
  NDArray<K> in2, {
  ConvMode mode = ConvMode.valid,
  NDArray<R>? out,
}) {
  if (in1.isDisposed || in2.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute correlate() on a disposed array.');
  }
  if (in1.rank != in2.rank || in1.rank == 0) {
    throw ArgumentError('in1 and in2 must have the same non-zero rank.');
  }
  if (in1.dtype != in2.dtype) {
    throw ArgumentError('in1 and in2 must have matching DType.');
  }

  final rank = in1.rank;

  switch (mode) {
    case ConvMode.valid:
      for (var i = 0; i < rank; i++) {
        if (in1.shape[i] < in2.shape[i]) {
          throw ArgumentError(
            'in1 dimensions must be >= in2 dimensions for valid mode.',
          );
        }
      }
      final expectedShape = List<int>.generate(
        rank,
        (i) => in1.shape[i] - in2.shape[i] + 1,
      );
      if (out != null && !listEquals(out.shape, expectedShape)) {
        throw ArgumentError('Provided out buffer has incompatible shape.');
      }
      return _correlateValid<T, K, R>(in1, in2, out: out);
    case ConvMode.full:
      final expectedShape = List<int>.generate(
        rank,
        (i) => in1.shape[i] + in2.shape[i] - 1,
      );
      if (out != null && !listEquals(out.shape, expectedShape)) {
        throw ArgumentError('Provided out buffer has incompatible shape.');
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
        final res = _correlateValid<T, K, R>(padded1, in2, out: out);
        if (out != null) return out;
        return res.detachToParentScope();
      });
    case ConvMode.same:
      if (out != null && !listEquals(out.shape, in1.shape)) {
        throw ArgumentError('Provided out buffer has incompatible shape.');
      }
      return NDArray.scope(() {
        final fullCorr = correlate<T, K, R>(in1, in2, mode: ConvMode.full);
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
NDArray<R> convolve<T extends Object, K extends Object, R extends Object>(
  NDArray<T> in1,
  NDArray<K> in2, {
  ConvMode mode = ConvMode.full,
  NDArray<R>? out,
}) {
  if (in1.isDisposed || in2.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute convolve() on a disposed array.');
  }
  if (in1.rank != in2.rank || in1.rank == 0) {
    throw ArgumentError('in1 and in2 must have the same non-zero rank.');
  }

  return NDArray.scope(() {
    final reversedSelectors = List<Selector>.generate(
      in2.rank,
      (_) => const Slice(step: -1),
    );
    final flippedKernel = in2.slice(reversedSelectors);
    final contiguousKernel = flippedKernel.isContiguous
        ? flippedKernel
        : flippedKernel.copy();
    final res = correlate<T, K, R>(in1, contiguousKernel, mode: mode, out: out);
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
NDArray<R> convolve2d<T extends Object, K extends Object, R extends Object>(
  NDArray<T> in1,
  NDArray<K> in2, {
  ConvMode mode = ConvMode.full,
  NDArray<R>? out,
}) {
  if (in1.rank != 2 || in2.rank != 2) {
    throw ArgumentError('convolve2d requires 2-dimensional arrays.');
  }
  return convolve<T, K, R>(in1, in2, mode: mode, out: out);
}
