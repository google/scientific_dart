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

// ignore_for_file: non_constant_identifier_names
import 'package:pocketfft/pocketfft.dart';

import '../ndarray.dart';
import '../float16_utils.dart';

import 'dart:ffi' as ffi;

import '../scratch_arena.dart';
import 'padding.dart';

// Standalone operational relative cross-imports
import 'manipulation.dart';

/// Helper to allocate a KissFFT plan configuration on the ScratchArena stack.

void _loadSignalToKissInput<T>(
  NDArray<T> inputA,
  int srcStart,
  int copyLen,
  int targetLen,
  ffi.Pointer<kiss_fft_cpx> pin,
) {
  switch (inputA.dtype) {
    case DType.complex128:
      final inPtr = inputA.pointer.cast<kiss_fft_cpx>() + srcStart;
      for (var i = 0; i < copyLen; i++) {
        pin[i].r = inPtr[i].r;
        pin[i].i = inPtr[i].i;
      }
    case DType.complex64:
      final inPtr = inputA.pointer.cast<ffi.Float>() + (srcStart * 2);
      for (var i = 0; i < copyLen; i++) {
        pin[i].r = inPtr[2 * i];
        pin[i].i = inPtr[2 * i + 1];
      }
    case DType.float64:
      final inPtr = inputA.pointer.cast<ffi.Double>() + srcStart;
      for (var i = 0; i < copyLen; i++) {
        pin[i].r = inPtr[i];
        pin[i].i = 0.0;
      }
    case DType.float32:
      final inPtr = inputA.pointer.cast<ffi.Float>() + srcStart;
      for (var i = 0; i < copyLen; i++) {
        pin[i].r = inPtr[i];
        pin[i].i = 0.0;
      }
    case DType.int64:
      final inPtr = inputA.pointer.cast<ffi.Int64>() + srcStart;
      for (var i = 0; i < copyLen; i++) {
        pin[i].r = inPtr[i].toDouble();
        pin[i].i = 0.0;
      }
    case DType.int32:
      final inPtr = inputA.pointer.cast<ffi.Int32>() + srcStart;
      for (var i = 0; i < copyLen; i++) {
        pin[i].r = inPtr[i].toDouble();
        pin[i].i = 0.0;
      }
    case DType.int16:
      final inPtr = inputA.pointer.cast<ffi.Int16>() + srcStart;
      for (var i = 0; i < copyLen; i++) {
        pin[i].r = inPtr[i].toDouble();
        pin[i].i = 0.0;
      }
    case DType.uint8:
      final inPtr = inputA.pointer.cast<ffi.Uint8>() + srcStart;
      for (var i = 0; i < copyLen; i++) {
        pin[i].r = inPtr[i].toDouble();
        pin[i].i = 0.0;
      }
    case DType.int8:
      final inPtr = inputA.pointer.cast<ffi.Int8>() + srcStart;
      for (var i = 0; i < copyLen; i++) {
        pin[i].r = inPtr[i].toDouble();
        pin[i].i = 0.0;
      }
    case DType.uint64:
      final inPtr = inputA.pointer.cast<ffi.Uint64>() + srcStart;
      for (var i = 0; i < copyLen; i++) {
        pin[i].r = inPtr[i].toDouble();
        pin[i].i = 0.0;
      }
    case DType.uint32:
      final inPtr = inputA.pointer.cast<ffi.Uint32>() + srcStart;
      for (var i = 0; i < copyLen; i++) {
        pin[i].r = inPtr[i].toDouble();
        pin[i].i = 0.0;
      }
    case DType.uint16:
      final inPtr = inputA.pointer.cast<ffi.Uint16>() + srcStart;
      for (var i = 0; i < copyLen; i++) {
        pin[i].r = inPtr[i].toDouble();
        pin[i].i = 0.0;
      }
    case DType.float16:
      final inPtr = inputA.pointer.cast<ffi.Uint16>() + srcStart;
      for (var i = 0; i < copyLen; i++) {
        pin[i].r = Float16Utils.decodeFloat16(inPtr[i]);
        pin[i].i = 0.0;
      }
    case DType.bfloat16:
      final inPtr = inputA.pointer.cast<ffi.Uint16>() + srcStart;
      for (var i = 0; i < copyLen; i++) {
        pin[i].r = Float16Utils.decodeBFloat16(inPtr[i]);
        pin[i].i = 0.0;
      }
    case DType.boolean:
      final inPtr = inputA.pointer.cast<ffi.Uint8>() + srcStart;
      for (var i = 0; i < copyLen; i++) {
        pin[i].r = inPtr[i] != 0 ? 1.0 : 0.0;
        pin[i].i = 0.0;
      }
  }
  for (var i = copyLen; i < targetLen; i++) {
    pin[i].r = 0.0;
    pin[i].i = 0.0;
  }
}

void _storeKissOutputToResult<R>(
  NDArray<R> result,
  int destStart,
  int targetLen,
  ffi.Pointer<kiss_fft_cpx> pout, {
  double scale = 1.0,
}) {
  if (result.dtype == DType.complex128) {
    final outPtr = result.pointer.cast<kiss_fft_cpx>() + destStart;
    for (var i = 0; i < targetLen; i++) {
      outPtr[i].r = pout[i].r * scale;
      outPtr[i].i = pout[i].i * scale;
    }
  } else if (result.dtype == DType.complex64) {
    final outPtr = result.pointer.cast<ffi.Float>() + (destStart * 2);
    for (var i = 0; i < targetLen; i++) {
      outPtr[2 * i] = (pout[i].r * scale);
      outPtr[2 * i + 1] = (pout[i].i * scale);
    }
  }
}

kiss_fft_cfg _getKissFFTPlan(int nfft, int inverse_fft) {
  return PocketFFTPlanCache.instance.getPlan(nfft, isInverse: inverse_fft != 0);
}

/// Computes the 1D discrete Fourier Transform (FFT) along the specified [axis].
///
/// Transforms discrete sequences from the time/space domain into frequency coefficients
/// using the standard Discrete Fourier Transform (DFT) formula:
///
/// \[
/// X_k = \sum_{n=0}^{N-1} x_n e^{-i \frac{2\pi}{N} k n}
/// \]
///
/// The resulting array is **always complex** (DType.complex128 or DType.complex64 depending on precision).
///
/// Uses pocketfft's pocketfft/KissFFT mixed-radix prime factoring, supporting
/// arbitrary non-power-of-two sequence lengths.
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if the input array [a] shape is empty (scalar 0D, rank < 1).
/// - It is an error if the specified [axis] is out of bounds `[-a.rank, a.rank - 1]`.
/// - It is an error if target length [n] is provided but is less than or equal to 0.
/// - It is an error if [out] has incompatible shape, dtype, or is not contiguous.
///
/// **Memory Ownership & Lifetime:**
/// - Allocates a new array on the unmanaged C heap. The caller takes full ownership of this memory and must explicitly call [dispose] to prevent native leaks, unless executing inside a managed [NDArray.scope].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N \log N)$ where $N$ is the transform length, scaling linearly even for prime lengths.
/// - Swapping the selected [axis] to the final dimension is a zero-copy, zero-allocation transpose view ($O(1)$ complexity).
///
/// **Example:**
/// {@example /example/fft_example.dart lang=dart}
///
/// Reference: [Cooley-Tukey FFT Algorithm](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)
NDArray<R> fft<T, R extends Complex>(
  NDArray<T> a, {
  int? n,
  int axis = -1,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute fft() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write FFT result to a disposed output array.');
  }
  if (a.shape.isEmpty) {
    throw ArgumentError(
      'Cannot compute FFT on a 0-dimensional or empty scalar array',
    );
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(axis, -rank, rank - 1, 'axis');
  }

  final lastAxisDim = a.shape[normAxis];
  final targetLen = n ?? lastAxisDim;
  if (targetLen <= 0) {
    throw ArgumentError(
      'Target transform length [n] must be greater than 0 (was $n)',
    );
  }

  // Formulate output shape by replacing the transform axis with targetLen
  final outShape = List<int>.from(a.shape);
  outShape[normAxis] = targetLen;

  final isFloatOrComplex = a.dtype.isFloating || a.dtype.isComplex;
  final expectedDType = (a.dtype == DType.float32 || a.dtype == DType.complex64)
      ? DType.complex64
      : DType.complex128;
  final targetDType = out?.dtype ?? expectedDType;

  if (out != null) {
    if (isFloatOrComplex && out.dtype != expectedDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible dtype (expected $expectedDType, got ${out.dtype}).',
      );
    }
    if (!isFloatOrComplex &&
        out.dtype != DType.complex64 &&
        out.dtype != DType.complex128) {
      throw ArgumentError(
        'Provided out buffer has incompatible dtype (expected complex64 or complex128, got ${out.dtype}).',
      );
    }
    if (!listEquals(out.shape, outShape)) {
      throw ArgumentError('Provided out buffer has incompatible shape.');
    }
    if (!out.isContiguous) {
      throw ArgumentError('Provided out buffer must be contiguous.');
    }
  }

  if (normAxis != rank - 1) {
    final axes = List.generate(rank, (i) => i);
    axes[normAxis] = rank - 1;
    axes[rank - 1] = normAxis;

    final transposedInput = a.transpose(axes);
    if (out != null) {
      final transposedResult = fft<T, R>(transposedInput, n: n);
      final finalResult = transposedResult.transpose(axes);
      finalResult.copy(out: out);
      transposedResult.dispose();
      finalResult.dispose();
      transposedInput.dispose();
      return out;
    } else {
      final transposedResult = fft<T, R>(transposedInput, n: n);
      final finalResult = transposedResult.transpose(axes);
      final resCopy = finalResult.copy();
      finalResult.dispose();
      transposedResult.dispose();
      transposedInput.dispose();
      return resCopy;
    }
  }

  final NDArray<T> inputA;
  final bool wasCopied;
  if (!a.isContiguous) {
    inputA = a.copy();
    wasCopied = true;
  } else {
    inputA = a;
    wasCopied = false;
  }

  final result = out ?? NDArray<R>.zeros(outShape, targetDType as DType<R>);

  // Count how many 1D row sub-signals exist to execute strided walks
  final totalElements = inputA.shape.reduce((x, y) => x * y);
  final signalsCount = totalElements ~/ lastAxisDim;

  final isZeroCopyFastPath =
      inputA.dtype == DType.complex128 &&
      targetLen == lastAxisDim &&
      inputA.isContiguous;

  kiss_fft_cfg cfg = ffi.nullptr.cast();

  if (isZeroCopyFastPath) {
    final marker = ScratchArena.marker;
    try {
      cfg = _getKissFFTPlan(targetLen, 0);

      for (var s = 0; s < signalsCount; s++) {
        final rowPin = inputA.pointer.cast<kiss_fft_cpx>() + s * lastAxisDim;
        final rowPout = result.pointer.cast<kiss_fft_cpx>() + s * targetLen;
        kiss_fft(cfg, rowPin, rowPout);
      }
    } finally {
      ScratchArena.reset(marker);
      if (wasCopied) {
        inputA.dispose();
      }
    }
    return result;
  }

  final marker = ScratchArena.marker;
  ffi.Pointer<kiss_fft_cpx> pin = ffi.nullptr.cast();
  ffi.Pointer<kiss_fft_cpx> pout = ffi.nullptr.cast();

  try {
    cfg = _getKissFFTPlan(targetLen, 0);
    pin = ScratchArena.allocate<kiss_fft_cpx>(
      targetLen * ffi.sizeOf<kiss_fft_cpx>(),
    );
    pout = ScratchArena.allocate<kiss_fft_cpx>(
      targetLen * ffi.sizeOf<kiss_fft_cpx>(),
    );

    final copyLen = targetLen < lastAxisDim ? targetLen : lastAxisDim;
    for (var s = 0; s < signalsCount; s++) {
      final srcStart = s * lastAxisDim;
      final destStart = s * targetLen;

      _loadSignalToKissInput(inputA, srcStart, copyLen, targetLen, pin);

      // 3. Fire high-speed native FFT on the C heap components
      kiss_fft(cfg, pin, pout);

      // 4. Collect results from pout back into result array
      _storeKissOutputToResult(result, destStart, targetLen, pout);
    }
  } finally {
    ScratchArena.reset(marker);
    if (wasCopied) {
      inputA.dispose();
    }
  }

  return result;
}

/// Computes the 1D inverse discrete Fourier Transform (IFFT) along the specified [axis].
///
/// Transforms frequency domain coefficients back into complex time/space domain signals, applying
/// standard `1 / N` normalization scaling automatically:
///
/// \[
/// x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{i \frac{2\pi}{N} k n}
/// \]
///
/// The resulting array is **always complex** (DType.complex128 or DType.complex64 depending on precision).
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if the input array [a] shape is empty (scalar 0D, rank < 1).
/// - It is an error if the specified [axis] is out of bounds `[-a.rank, a.rank - 1]`.
/// - It is an error if target length [n] is provided but is less than or equal to 0.
/// - It is an error if [out] has incompatible shape, dtype, or is not contiguous.
///
/// **Memory Ownership & Lifetime:**
/// - Allocates a new array on the unmanaged C heap. The caller takes full ownership of this memory and must explicitly call [dispose] to prevent native leaks, unless executing inside a managed [NDArray.scope].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N \log N)$ where $N$ is the transform length.
/// - Swapping the selected [axis] to the final dimension is a zero-copy, zero-allocation transpose view ($O(1)$ complexity).
///
/// **Example:**
/// {@example /example/fft_example.dart lang=dart}
NDArray<R> ifft<T, R extends Complex>(
  NDArray<T> a, {
  int? n,
  int axis = -1,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute ifft() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write IFFT result to a disposed output array.');
  }
  if (a.shape.isEmpty) {
    throw ArgumentError(
      'Cannot compute IFFT on a 0-dimensional or empty scalar array',
    );
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(axis, -rank, rank - 1, 'axis');
  }

  final lastAxisDim = a.shape[normAxis];
  final targetLen = n ?? lastAxisDim;
  if (targetLen <= 0) {
    throw ArgumentError(
      'Target transform length [n] must be greater than 0 (was $n)',
    );
  }

  // Formulate output shape by replacing the transform axis with targetLen
  final outShape = List<int>.from(a.shape);
  outShape[normAxis] = targetLen;

  final isFloatOrComplex = a.dtype.isFloating || a.dtype.isComplex;
  final expectedDType = (a.dtype == DType.float32 || a.dtype == DType.complex64)
      ? DType.complex64
      : DType.complex128;
  final targetDType = out?.dtype ?? expectedDType;

  if (out != null) {
    if (isFloatOrComplex && out.dtype != expectedDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible dtype (expected $expectedDType, got ${out.dtype}).',
      );
    }
    if (!isFloatOrComplex &&
        out.dtype != DType.complex64 &&
        out.dtype != DType.complex128) {
      throw ArgumentError(
        'Provided out buffer has incompatible dtype (expected complex64 or complex128, got ${out.dtype}).',
      );
    }
    if (!listEquals(out.shape, outShape)) {
      throw ArgumentError('Provided out buffer has incompatible shape.');
    }
    if (!out.isContiguous) {
      throw ArgumentError('Provided out buffer must be contiguous.');
    }
  }

  if (normAxis != rank - 1) {
    final axes = List.generate(rank, (i) => i);
    axes[normAxis] = rank - 1;
    axes[rank - 1] = normAxis;

    final transposedInput = a.transpose(axes);
    if (out != null) {
      final transposedResult = ifft<T, R>(transposedInput, n: n);
      final finalResult = transposedResult.transpose(axes);
      finalResult.copy(out: out);
      transposedResult.dispose();
      finalResult.dispose();
      transposedInput.dispose();
      return out;
    } else {
      final transposedResult = ifft<T, R>(transposedInput, n: n);
      final finalResult = transposedResult.transpose(axes);
      final resCopy = finalResult.copy();
      finalResult.dispose();
      transposedResult.dispose();
      transposedInput.dispose();
      return resCopy;
    }
  }

  final NDArray<T> inputA;
  final bool wasCopied;
  if (!a.isContiguous) {
    inputA = a.copy();
    wasCopied = true;
  } else {
    inputA = a;
    wasCopied = false;
  }

  final result = out ?? NDArray<R>.zeros(outShape, targetDType as DType<R>);

  final totalElements = inputA.shape.reduce((x, y) => x * y);
  final signalsCount = totalElements ~/ lastAxisDim;

  final isZeroCopyFastPath =
      inputA.dtype == DType.complex128 &&
      targetLen == lastAxisDim &&
      inputA.isContiguous;

  kiss_fft_cfg cfg = ffi.nullptr.cast();

  if (isZeroCopyFastPath) {
    final marker = ScratchArena.marker;
    try {
      cfg = _getKissFFTPlan(targetLen, 1);

      final scaleFactor = 1.0 / targetLen;
      for (var s = 0; s < signalsCount; s++) {
        final rowPin = inputA.pointer.cast<kiss_fft_cpx>() + s * lastAxisDim;
        final rowPout = result.pointer.cast<kiss_fft_cpx>() + s * targetLen;
        kiss_fft(cfg, rowPin, rowPout);

        final pDoubles = rowPout.cast<ffi.Double>();
        final count = targetLen * 2;
        for (var i = 0; i < count; i++) {
          pDoubles[i] *= scaleFactor;
        }
      }
    } finally {
      ScratchArena.reset(marker);
      if (wasCopied) {
        inputA.dispose();
      }
    }
    return result;
  }

  final marker = ScratchArena.marker;
  ffi.Pointer<kiss_fft_cpx> pin = ffi.nullptr.cast();
  ffi.Pointer<kiss_fft_cpx> pout = ffi.nullptr.cast();

  try {
    cfg = _getKissFFTPlan(targetLen, 1);
    pin = ScratchArena.allocate<kiss_fft_cpx>(
      targetLen * ffi.sizeOf<kiss_fft_cpx>(),
    );
    pout = ScratchArena.allocate<kiss_fft_cpx>(
      targetLen * ffi.sizeOf<kiss_fft_cpx>(),
    );

    final copyLen = targetLen < lastAxisDim ? targetLen : lastAxisDim;
    final scaleFactor = 1.0 / targetLen;
    for (var s = 0; s < signalsCount; s++) {
      final srcStart = s * lastAxisDim;
      final destStart = s * targetLen;

      _loadSignalToKissInput(inputA, srcStart, copyLen, targetLen, pin);

      // 2. Fire high-speed native inverse transform
      kiss_fft(cfg, pin, pout);

      // 3. Apply standard 1/N scaling factor normalization (KissFFT leaves it unscaled)
      _storeKissOutputToResult(
        result,
        destStart,
        targetLen,
        pout,
        scale: scaleFactor,
      );
    }
  } finally {
    ScratchArena.reset(marker);
    if (wasCopied) {
      inputA.dispose();
    }
  }

  return result;
}

/// Shifts the zero-frequency component to the center of the spectrum.
///
/// Swaps half-spaces along all specified [axes].
/// This is extremely useful for re-positioning frequency components so that the zero-frequency (DC)
/// component is centered in the middle of the grid/spectrum.
///
/// **Preconditions:**
/// - It is an error if [a] is disposed.
/// - It is an error if [axes] contains duplicate indices or invalid types.
/// - It is an error if any axis index in [axes] is out of bounds `[-a.rank, a.rank - 1].
///
/// **Memory Ownership & Lifetime:**
/// - Allocates a new array on the unmanaged C heap. The caller takes full ownership of this memory and must explicitly call [dispose] to prevent native leaks, unless executing inside a managed [NDArray.scope].
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(N)$ where $N$ is the total number of elements.
/// - Space Complexity is $O(N)$ as it allocates a new array and copies elements to construct the shifted spectrum.
///
/// **Example:**
/// {@example /example/fftshift_example.dart lang=dart}
///
/// Reference: [NumPy fftshift](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html)
NDArray<T> fftshift<T extends Object>(
  NDArray<T> a, {
  dynamic axes,
  NDArray<T>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot shift a disposed array.');
  }

  final rank = a.rank;
  if (rank == 0) {
    return a.copy(out: out);
  }

  // Resolve axes
  final List<int> resolvedAxes;
  if (axes == null) {
    resolvedAxes = List.generate(rank, (i) => i);
  } else if (axes is int) {
    final norm = axes < 0 ? rank + axes : axes;
    if (norm < 0 || norm >= rank) {
      throw RangeError.range(axes, -rank, rank - 1, 'axes');
    }
    resolvedAxes = [norm];
  } else if (axes is List<int>) {
    resolvedAxes = [];
    for (var axis in axes) {
      final norm = axis < 0 ? rank + axis : axis;
      if (norm < 0 || norm >= rank) {
        throw RangeError.range(axis, -rank, rank - 1, 'axes');
      }
      if (resolvedAxes.contains(norm)) {
        throw ArgumentError('Duplicate axis $norm specified in axes.');
      }
      resolvedAxes.add(norm);
    }
  } else {
    throw ArgumentError(
      'axes must be null, an integer, or a list of integers.',
    );
  }

  return NDArray.scope(() {
    NDArray<T> current = a;
    for (final axis in resolvedAxes) {
      final dimSize = current.shape[axis];
      final shift = dimSize ~/ 2;
      current = roll(current, shift, axis: axis);
    }
    final result = current.copy(out: out);
    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}

/// Inverse of [fftshift].
///
/// Shifts the zero-frequency component back to the beginning of the spectrum.
///
/// **Preconditions:**
/// - It is an error if [a] is disposed.
/// - It is an error if [axes] contains duplicate indices or invalid types.
/// - It is an error if any axis index in [axes] is out of bounds `[-a.rank, a.rank - 1].
///
/// **Memory Ownership & Lifetime:**
/// - Allocates a new array on the unmanaged C heap. The caller takes full ownership of this memory and must explicitly call [dispose] to prevent native leaks, unless executing inside a managed [NDArray.scope].
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(N)$ where $N$ is the total number of elements.
/// - Space Complexity is $O(N)$ as it allocates a new array and copies elements to construct the shifted spectrum.
///
/// **Example:**
/// {@example /example/fftshift_example.dart lang=dart}
///
/// Reference: [NumPy ifftshift](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftshift.html)
NDArray<T> ifftshift<T extends Object>(
  NDArray<T> a, {
  dynamic axes,
  NDArray<T>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot shift a disposed array.');
  }

  final rank = a.rank;
  if (rank == 0) {
    return a.copy(out: out);
  }

  // Resolve axes
  final List<int> resolvedAxes;
  if (axes == null) {
    resolvedAxes = List.generate(rank, (i) => i);
  } else if (axes is int) {
    final norm = axes < 0 ? rank + axes : axes;
    if (norm < 0 || norm >= rank) {
      throw RangeError.range(axes, -rank, rank - 1, 'axes');
    }
    resolvedAxes = [norm];
  } else if (axes is List<int>) {
    resolvedAxes = [];
    for (var axis in axes) {
      final norm = axis < 0 ? rank + axis : axis;
      if (norm < 0 || norm >= rank) {
        throw RangeError.range(axis, -rank, rank - 1, 'axes');
      }
      if (resolvedAxes.contains(norm)) {
        throw ArgumentError('Duplicate axis $norm specified in axes.');
      }
      resolvedAxes.add(norm);
    }
  } else {
    throw ArgumentError(
      'axes must be null, an integer, or a list of integers.',
    );
  }

  return NDArray.scope(() {
    NDArray<T> current = a;
    for (final axis in resolvedAxes) {
      final dimSize = current.shape[axis];
      final shift = (dimSize + 1) ~/ 2;
      current = roll(current, shift, axis: axis);
    }
    final result = current.copy(out: out);
    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}

// Helper to retrieve a KissFFTR plan configuration from the cache.
kiss_fftr_cfg _getKissFFTRPlan(int nfft, int inverse_fft) {
  return PocketFFTPlanCache.instance.getRealPlan(
    nfft,
    isInverse: inverse_fft != 0,
  );
}

// Helper to retrieve a KissFFTND plan configuration from the cache.
kiss_fftnd_cfg _getKissFFTNDPlan(List<int> dims, int inverse_fft) {
  return PocketFFTPlanCache.instance.getNDPlan(
    dims,
    isInverse: inverse_fft != 0,
  );
}

List<int> _inversePermutation(List<int> p) {
  final inv = List<int>.filled(p.length, 0);
  for (var i = 0; i < p.length; i++) {
    inv[p[i]] = i;
  }
  return inv;
}

void _copyRealToDouble(
  NDArray a,
  int offset,
  int count,
  ffi.Pointer<ffi.Double> dest,
) {
  switch (a.dtype) {
    case DType.float64:
      final pIn = a.pointer.cast<ffi.Double>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i] = pIn[i];
      }
    case DType.float32:
      final pIn = a.pointer.cast<ffi.Float>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i] = pIn[i];
      }
    case DType.int64:
      final pIn = a.pointer.cast<ffi.Int64>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i] = pIn[i].toDouble();
      }
    case DType.int32:
      final pIn = a.pointer.cast<ffi.Int32>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i] = pIn[i].toDouble();
      }
    case DType.int16:
      final pIn = a.pointer.cast<ffi.Int16>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i] = pIn[i].toDouble();
      }
    case DType.uint8:
      final pIn = a.pointer.cast<ffi.Uint8>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i] = pIn[i].toDouble();
      }
    default:
      throw UnsupportedError(
        'Unsupported dtype for rfft input copy: ${a.dtype}',
      );
  }
}

void _copyComplexToDoubleCpx(
  NDArray a,
  int offset,
  int count,
  ffi.Pointer<kiss_fft_cpx> dest,
) {
  switch (a.dtype) {
    case DType.complex128:
      final pIn = a.pointer.cast<kiss_fft_cpx>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i].r = pIn[i].r;
        dest[i].i = pIn[i].i;
      }
    case DType.complex64:
      final pIn = a.pointer.cast<ffi.Float>() + offset * 2;
      for (var i = 0; i < count; i++) {
        dest[i].r = pIn[2 * i];
        dest[i].i = pIn[2 * i + 1];
      }
    case DType.float64:
      final pIn = a.pointer.cast<ffi.Double>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i].r = pIn[i];
        dest[i].i = 0.0;
      }
    case DType.float32:
      final pIn = a.pointer.cast<ffi.Float>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i].r = pIn[i];
        dest[i].i = 0.0;
      }
    case DType.int64:
      final pIn = a.pointer.cast<ffi.Int64>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i].r = pIn[i].toDouble();
        dest[i].i = 0.0;
      }
    case DType.int32:
      final pIn = a.pointer.cast<ffi.Int32>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i].r = pIn[i].toDouble();
        dest[i].i = 0.0;
      }
    case DType.int16:
      final pIn = a.pointer.cast<ffi.Int16>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i].r = pIn[i].toDouble();
        dest[i].i = 0.0;
      }
    case DType.uint8:
      final pIn = a.pointer.cast<ffi.Uint8>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i].r = pIn[i].toDouble();
        dest[i].i = 0.0;
      }
    default:
      throw UnsupportedError(
        'Unsupported dtype for irfft input copy: ${a.dtype}',
      );
  }
}

void _copyIntToDoubleCpx(NDArray a, int count, ffi.Pointer<kiss_fft_cpx> dest) {
  switch (a.dtype) {
    case DType.int64:
      final pIn = a.pointer.cast<ffi.Int64>();
      for (var i = 0; i < count; i++) {
        dest[i].r = pIn[i].toDouble();
        dest[i].i = 0.0;
      }
    case DType.int32:
      final pIn = a.pointer.cast<ffi.Int32>();
      for (var i = 0; i < count; i++) {
        dest[i].r = pIn[i].toDouble();
        dest[i].i = 0.0;
      }
    case DType.int16:
      final pIn = a.pointer.cast<ffi.Int16>();
      for (var i = 0; i < count; i++) {
        dest[i].r = pIn[i].toDouble();
        dest[i].i = 0.0;
      }
    case DType.uint8:
      final pIn = a.pointer.cast<ffi.Uint8>();
      for (var i = 0; i < count; i++) {
        dest[i].r = pIn[i].toDouble();
        dest[i].i = 0.0;
      }
    default:
      throw UnsupportedError('Expected integer dtype, got ${a.dtype}');
  }
}

void _copyIntToFloatCpx(NDArray a, int count, ffi.Pointer<ffi.Float> dest) {
  switch (a.dtype) {
    case DType.int64:
      final pIn = a.pointer.cast<ffi.Int64>();
      for (var i = 0; i < count; i++) {
        dest[2 * i] = pIn[i].toDouble();
        dest[2 * i + 1] = 0.0;
      }
    case DType.int32:
      final pIn = a.pointer.cast<ffi.Int32>();
      for (var i = 0; i < count; i++) {
        dest[2 * i] = pIn[i].toDouble();
        dest[2 * i + 1] = 0.0;
      }
    case DType.int16:
      final pIn = a.pointer.cast<ffi.Int16>();
      for (var i = 0; i < count; i++) {
        dest[2 * i] = pIn[i].toDouble();
        dest[2 * i + 1] = 0.0;
      }
    case DType.uint8:
      final pIn = a.pointer.cast<ffi.Uint8>();
      for (var i = 0; i < count; i++) {
        dest[2 * i] = pIn[i].toDouble();
        dest[2 * i + 1] = 0.0;
      }
    default:
      throw UnsupportedError('Expected integer dtype, got ${a.dtype}');
  }
}

NDArray<R> _promoteToComplex<T, R extends Complex>(
  NDArray<T> a,
  DType<R> targetDType,
) {
  final result = NDArray<R>.zeros(a.shape, targetDType);
  final numElements = a.size;
  final contiguousA = a.isContiguous ? a : a.copy();

  if (targetDType == DType.complex128) {
    final pOut = result.pointer.cast<kiss_fft_cpx>();
    final pOutD = pOut.cast<ffi.Double>();
    switch (contiguousA.dtype) {
      case DType.complex128:
        final pInD = contiguousA.pointer.cast<ffi.Double>();
        final count = numElements * 2;
        for (var i = 0; i < count; i++) {
          pOutD[i] = pInD[i];
        }
        break;
      case DType.complex64:
        final pInF = contiguousA.pointer.cast<ffi.Float>();
        for (var i = 0; i < numElements; i++) {
          pOutD[2 * i] = pInF[2 * i];
          pOutD[2 * i + 1] = pInF[2 * i + 1];
        }
        break;
      case DType.float64:
        final pInD = contiguousA.pointer.cast<ffi.Double>();
        for (var i = 0; i < numElements; i++) {
          pOutD[2 * i] = pInD[i];
          pOutD[2 * i + 1] = 0.0;
        }
        break;
      case DType.float32:
        final pInF = contiguousA.pointer.cast<ffi.Float>();
        for (var i = 0; i < numElements; i++) {
          pOutD[2 * i] = pInF[i];
          pOutD[2 * i + 1] = 0.0;
        }
        break;
      default:
        _copyIntToDoubleCpx(contiguousA, numElements, pOut);
    }
  } else {
    // complex64
    final pOut = result.pointer.cast<ffi.Float>();
    switch (contiguousA.dtype) {
      case DType.complex128:
        final pInD = contiguousA.pointer.cast<ffi.Double>();
        for (var i = 0; i < numElements; i++) {
          pOut[2 * i] = pInD[2 * i];
          pOut[2 * i + 1] = pInD[2 * i + 1];
        }
        break;
      case DType.complex64:
        final pInF = contiguousA.pointer.cast<ffi.Float>();
        final count = numElements * 2;
        for (var i = 0; i < count; i++) {
          pOut[i] = pInF[i];
        }
        break;
      case DType.float64:
        final pInD = contiguousA.pointer.cast<ffi.Double>();
        for (var i = 0; i < numElements; i++) {
          pOut[2 * i] = pInD[i];
          pOut[2 * i + 1] = 0.0;
        }
        break;
      case DType.float32:
        final pInF = contiguousA.pointer.cast<ffi.Float>();
        for (var i = 0; i < numElements; i++) {
          pOut[2 * i] = pInF[i];
          pOut[2 * i + 1] = 0.0;
        }
        break;
      default:
        _copyIntToFloatCpx(contiguousA, numElements, pOut);
    }
  }

  if (!a.isContiguous) {
    contiguousA.dispose();
  }
  return result;
}

NDArray<R> _padOrTruncate<T, R extends Complex>(
  NDArray<T> arr,
  List<int> s,
  List<int> axes,
  DType<R> targetDType,
) {
  var needsSliceOrPad = false;
  for (var i = 0; i < axes.length; i++) {
    if (s[i] != arr.shape[axes[i]]) {
      needsSliceOrPad = true;
      break;
    }
  }

  if (!needsSliceOrPad &&
      arr.dtype == targetDType &&
      arr.isContiguous &&
      arr is NDArray<R>) {
    return (arr as NDArray<R>).copy();
  }

  return NDArray.scope(() {
    NDArray<R> current = _promoteToComplex<T, R>(arr, targetDType);

    if (needsSliceOrPad) {
      for (var i = 0; i < axes.length; i++) {
        final axis = axes[i];
        final targetSize = s[i];
        final currentSize = current.shape[axis];

        if (targetSize < currentSize) {
          final slices = List<Selector>.generate(current.rank, (dim) {
            if (dim == axis) {
              return Slice(start: 0, stop: targetSize);
            }
            return const Slice.all();
          });
          current = current.slice(slices).copy();
        } else if (targetSize > currentSize) {
          final padWidthList = List<(int, int)>.generate(current.rank, (dim) {
            if (dim == axis) {
              return (0, targetSize - currentSize);
            }
            return (0, 0);
          });
          current = pad(current, PadWidth.axes(padWidthList));
        }
      }
    }
    return current.detachToParentScope();
  });
}

/// Returns the Discrete Fourier Transform sample frequencies.
///
/// The returned float array contains the frequency bin centers in cycles
/// per unit of the sample spacing (with spacing given by [d]). For instance,
/// if the sample spacing is in seconds, then the frequency unit is cycles/second (Hz).
///
/// Given a window length [n] and sample spacing [d], the frequencies are:
///
///   f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
///   f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd
///
/// **Preconditions:**
/// - It is an error if window length [n] is less than or equal to 0.
/// - It is an error if sample spacing [d] is 0.0.
///
/// **Performance considerations:**
/// - $O(n)$ time complexity and space complexity.
///
/// Reference: [DFT sample frequencies](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html)
NDArray<Float64> fftfreq(int n, {double d = 1.0}) {
  if (n <= 0) {
    throw ArgumentError('n must be strictly positive (was $n)');
  }
  if (d == 0.0) {
    throw ArgumentError('sample spacing d must be non-zero');
  }
  final val = 1.0 / (d * n);
  final list = List<Float64>.filled(n, Float64(0.0));
  if (n % 2 == 0) {
    final half = n ~/ 2;
    for (var i = 0; i < half; i++) {
      list[i] = Float64(i * val);
    }
    for (var i = half; i < n; i++) {
      list[i] = Float64((i - n) * val);
    }
  } else {
    final half = (n - 1) ~/ 2;
    for (var i = 0; i <= half; i++) {
      list[i] = Float64(i * val);
    }
    for (var i = half + 1; i < n; i++) {
      list[i] = Float64((i - n) * val);
    }
  }
  return NDArray<Float64>.fromList(list, [n], DType.float64);
}

/// Returns the Discrete Fourier Transform sample frequencies for real inputs.
///
/// The returned float array contains the frequency bin centers in cycles
/// per unit of the sample spacing (with spacing given by [d]). For instance,
/// if the sample spacing is in seconds, then the frequency unit is cycles/second (Hz).
///
/// Given a window length [n] and sample spacing [d], the frequencies are
/// the first `n // 2 + 1` elements returned by [fftfreq]:
///
///   f = [0, 1, ..., n//2] / (d*n)
///
/// **Preconditions:**
/// - It is an error if window length [n] is less than or equal to 0.
/// - It is an error if sample spacing [d] is 0.0.
///
/// **Performance considerations:**
/// - $O(n)$ time complexity and space complexity.
///
/// Reference: [Real DFT sample frequencies](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html)
NDArray<Float64> rfftfreq(int n, {double d = 1.0}) {
  if (n <= 0) {
    throw ArgumentError('n must be strictly positive (was $n)');
  }
  if (d == 0.0) {
    throw ArgumentError('sample spacing d must be non-zero');
  }
  final val = 1.0 / (d * n);
  final limit = n ~/ 2 + 1;
  final list = List<Float64>.generate(limit, (i) => Float64(i * val));
  return NDArray<Float64>.fromList(list, [limit], DType.float64);
}

/// Computes the 1D discrete Fourier Transform for real input along the specified [axis].
///
/// Computes the one-dimensional discrete Fourier Transform for a real-valued
/// input signal. Since the DFT of a real signal is conjugate symmetric, the
/// negative frequency terms are redundant and are omitted. The output has
/// length `n // 2 + 1` along the transformed axis.
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if the input array [a] shape is empty (scalar 0D, rank < 1).
/// - It is an error if the specified [axis] is out of bounds `[-a.rank, a.rank - 1]`.
/// - It is an error if target length [n] is provided but is less than or equal to 0.
/// - It is an error if [out] is provided and has incompatible shape (`(n ?? a.shape[axis]) // 2 + 1` along [axis]), incompatible dtype (`complex64` if input is `float32`, `complex128` otherwise), or is not contiguous.
///
/// **Throws:**
/// - [ArgumentError] if the input array shape is empty (scalar 0D).
/// - [RangeError] if the specified [axis] is out of bounds.
/// - [ArgumentError] if [n] is provided but is less than or equal to 0.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
/// - [StateError] if native FFI memory allocations fail.
///
/// **Performance considerations:**
/// - Even lengths use an optimized C pathway via `kiss_fftr` which is faster than complex FFT.
/// - Odd lengths fall back to casting to complex and running standard complex [fft] and slicing.
///
/// Reference: [Real 1D FFT](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html)
NDArray<R> rfft<T, R extends Complex>(
  NDArray<T> a, {
  int? n,
  int axis = -1,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute rfft() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write FFT result to a disposed output array.');
  }
  if (a.shape.isEmpty) {
    throw ArgumentError(
      'Cannot compute FFT on a 0-dimensional or empty scalar array',
    );
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(axis, -rank, rank - 1, 'axis');
  }

  final lastAxisDim = a.shape[normAxis];
  final targetLen = n ?? lastAxisDim;
  if (targetLen <= 0) {
    throw ArgumentError('Target transform length [n] must be greater than 0');
  }

  final outShape = List<int>.from(a.shape);
  outShape[normAxis] = targetLen ~/ 2 + 1;

  final isFloatOrComplex = a.dtype.isFloating || a.dtype.isComplex;
  final expectedDType = (a.dtype == DType.float32 || a.dtype == DType.complex64)
      ? DType.complex64
      : DType.complex128;
  final targetDType = out?.dtype ?? expectedDType;

  if (out != null) {
    if (isFloatOrComplex && out.dtype != expectedDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible dtype (expected $expectedDType, got ${out.dtype}).',
      );
    }
    if (!isFloatOrComplex &&
        out.dtype != DType.complex64 &&
        out.dtype != DType.complex128) {
      throw ArgumentError(
        'Provided out buffer has incompatible dtype (expected complex64 or complex128, got ${out.dtype}).',
      );
    }
    if (!listEquals(out.shape, outShape)) {
      throw ArgumentError('Provided out buffer has incompatible shape.');
    }
    if (!out.isContiguous) {
      throw ArgumentError('Provided out buffer must be contiguous.');
    }
  }

  if (targetLen % 2 == 0) {
    // Even targetLen: Optimized Path
    if (normAxis != rank - 1) {
      final axes = List.generate(rank, (i) => i);
      axes[normAxis] = rank - 1;
      axes[rank - 1] = normAxis;

      final transposedInput = a.transpose(axes);
      if (out != null) {
        final transposedResult = rfft<T, R>(transposedInput, n: n);
        final finalResult = transposedResult.transpose(axes);
        finalResult.copy(out: out);
        transposedResult.dispose();
        finalResult.dispose();
        transposedInput.dispose();
        return out;
      } else {
        final transposedResult = rfft<T, R>(transposedInput, n: n);
        final finalResult = transposedResult.transpose(axes);
        final resCopy = finalResult.copy();
        finalResult.dispose();
        transposedResult.dispose();
        transposedInput.dispose();
        return resCopy;
      }
    }

    final NDArray<T> inputA = a.isContiguous ? a : a.copy();
    final bool wasCopied = !a.isContiguous;

    final result = out ?? NDArray<R>.zeros(outShape, targetDType as DType<R>);

    final totalElements = inputA.shape.reduce((x, y) => x * y);
    final signalsCount = totalElements ~/ lastAxisDim;

    final isZeroCopyFastPath =
        inputA.dtype == DType.float64 &&
        targetLen == lastAxisDim &&
        inputA.isContiguous &&
        result.dtype == DType.complex128;

    kiss_fftr_cfg cfg = ffi.nullptr.cast();

    if (isZeroCopyFastPath) {
      final marker = ScratchArena.marker;
      try {
        cfg = _getKissFFTRPlan(targetLen, 0);
        final outLen = targetLen ~/ 2 + 1;
        for (var s = 0; s < signalsCount; s++) {
          final rowPin = inputA.pointer.cast<ffi.Double>() + s * lastAxisDim;
          final rowPout = result.pointer.cast<kiss_fft_cpx>() + s * outLen;
          kiss_fftr(cfg, rowPin, rowPout);
        }
      } finally {
        ScratchArena.reset(marker);
        if (wasCopied) inputA.dispose();
      }
      return result;
    }

    final marker = ScratchArena.marker;
    try {
      cfg = _getKissFFTRPlan(targetLen, 0);
      final pin = ScratchArena.allocate<ffi.Double>(
        targetLen * ffi.sizeOf<ffi.Double>(),
      );
      final outLen = targetLen ~/ 2 + 1;
      final pout = ScratchArena.allocate<kiss_fft_cpx>(
        outLen * ffi.sizeOf<kiss_fft_cpx>(),
      );

      final copyLen = targetLen < lastAxisDim ? targetLen : lastAxisDim;

      for (var s = 0; s < signalsCount; s++) {
        final srcStart = s * lastAxisDim;
        final destStart = s * outLen;

        _copyRealToDouble(inputA, srcStart, copyLen, pin);
        for (var i = copyLen; i < targetLen; i++) {
          pin[i] = 0.0;
        }

        kiss_fftr(cfg, pin, pout);

        if (result.dtype == DType.complex128) {
          final pOut = result.pointer.cast<ffi.Double>() + destStart * 2;
          final pPoutD = pout.cast<ffi.Double>();
          final numDoubles = outLen * 2;
          for (var i = 0; i < numDoubles; i++) {
            pOut[i] = pPoutD[i];
          }
        } else {
          final pOut = result.pointer.cast<ffi.Float>() + destStart * 2;
          for (var i = 0; i < outLen; i++) {
            pOut[2 * i] = pout[i].r;
            pOut[2 * i + 1] = pout[i].i;
          }
        }
      }
    } finally {
      ScratchArena.reset(marker);
      if (wasCopied) inputA.dispose();
    }
    return result;
  } else {
    // Odd targetLen: Fallback Path
    return NDArray.scope(() {
      final complexFFT = fft<T, Complex>(a, n: targetLen, axis: axis);
      final slices = List<Selector>.generate(rank, (i) {
        if (i == normAxis) {
          return Slice(start: 0, stop: targetLen ~/ 2 + 1);
        }
        return const Slice.all();
      });
      final sliced = complexFFT.slice(slices);
      final finalResult =
          out ?? NDArray<R>.zeros(outShape, targetDType as DType<R>);
      sliced.copy(out: finalResult);
      if (out == null) {
        finalResult.detachToParentScope();
      }
      return finalResult;
    });
  }
}

/// Computes the 1D inverse discrete Fourier Transform for real input along the specified [axis].
///
/// Computes the inverse of [rfft]. The input signal along [axis] is expected
/// to be conjugate symmetric and contain only the non-negative frequency terms
/// (as returned by [rfft]). The output is real-valued.
///
/// The length of the output along [axis] is given by [n]. If [n] is not provided,
/// it defaults to `2 * (a.shape[axis] - 1)`.
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if the input array [a] shape is empty (scalar 0D, rank < 1).
/// - It is an error if the specified [axis] is out of bounds `[-a.rank, a.rank - 1]`.
/// - It is an error if target length [n] is provided but is less than or equal to 0.
/// - It is an error if [out] is provided and has incompatible shape ([n] along [axis]), incompatible dtype (`float32` if input is `complex64` or `float32`, `float64` otherwise), or is not contiguous.
///
/// **Throws:**
/// - [ArgumentError] if the input array shape is empty (scalar 0D).
/// - [RangeError] if the specified [axis] is out of bounds.
/// - [ArgumentError] if [n] is provided but is less than or equal to 0.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
/// - [StateError] if native FFI memory allocations fail.
///
/// **Performance considerations:**
/// - Even [n] uses an optimized C pathway via `kiss_fftri`.
/// - Odd [n] reconstructs the full conjugate symmetric spectrum, runs complex [ifft], and discards imaginary part.
///
/// Reference: [Inverse Real 1D FFT](https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html)
NDArray<R> irfft<T, R extends double>(
  NDArray<T> a, {
  int? n,
  int axis = -1,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute irfft() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write IFFT result to a disposed output array.');
  }
  if (a.shape.isEmpty) {
    throw ArgumentError(
      'Cannot compute IFFT on a 0-dimensional or empty scalar array',
    );
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(axis, -rank, rank - 1, 'axis');
  }

  final lastAxisDim = a.shape[normAxis];
  final targetLen = n ?? 2 * (lastAxisDim - 1);
  if (targetLen <= 0) {
    throw ArgumentError('Target transform length [n] must be greater than 0');
  }

  final outShape = List<int>.from(a.shape);
  outShape[normAxis] = targetLen;

  final isFloatOrComplex = a.dtype.isFloating || a.dtype.isComplex;
  final expectedDType = (a.dtype == DType.complex64 || a.dtype == DType.float32)
      ? DType.float32
      : DType.float64;
  final targetDType = out?.dtype ?? expectedDType;

  if (out != null) {
    if (isFloatOrComplex && out.dtype != expectedDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible dtype (expected $expectedDType, got ${out.dtype}).',
      );
    }
    if (!isFloatOrComplex &&
        out.dtype != DType.float32 &&
        out.dtype != DType.float64) {
      throw ArgumentError(
        'Provided out buffer has incompatible dtype (expected float32 or float64, got ${out.dtype}).',
      );
    }
    if (!listEquals(out.shape, outShape)) {
      throw ArgumentError('Provided out buffer has incompatible shape.');
    }
    if (!out.isContiguous) {
      throw ArgumentError('Provided out buffer must be contiguous.');
    }
  }

  if (targetLen % 2 == 0) {
    // Even targetLen: Optimized Path
    final targetInputLen = targetLen ~/ 2 + 1;

    if (normAxis != rank - 1) {
      final axes = List.generate(rank, (i) => i);
      axes[normAxis] = rank - 1;
      axes[rank - 1] = normAxis;

      final transposedInput = a.transpose(axes);
      if (out != null) {
        final transposedResult = irfft<T, R>(transposedInput, n: n);
        final finalResult = transposedResult.transpose(axes);
        finalResult.copy(out: out);
        transposedResult.dispose();
        finalResult.dispose();
        transposedInput.dispose();
        return out;
      } else {
        final transposedResult = irfft<T, R>(transposedInput, n: n);
        final finalResult = transposedResult.transpose(axes);
        final resCopy = finalResult.copy();
        finalResult.dispose();
        transposedResult.dispose();
        transposedInput.dispose();
        return resCopy;
      }
    }

    final NDArray<T> inputA = a.isContiguous ? a : a.copy();
    final bool wasCopied = !a.isContiguous;

    final result = out ?? NDArray<R>.zeros(outShape, targetDType as DType<R>);

    final totalElements = inputA.shape.reduce((x, y) => x * y);
    final signalsCount = totalElements ~/ lastAxisDim;

    final isZeroCopyFastPath =
        inputA.dtype == DType.complex128 &&
        targetInputLen == lastAxisDim &&
        inputA.isContiguous &&
        result.dtype == DType.float64;

    kiss_fftr_cfg cfg = ffi.nullptr.cast();

    if (isZeroCopyFastPath) {
      final marker = ScratchArena.marker;
      try {
        cfg = _getKissFFTRPlan(targetLen, 1);
        final scaleFactor = 1.0 / targetLen;
        for (var s = 0; s < signalsCount; s++) {
          final rowPin = inputA.pointer.cast<kiss_fft_cpx>() + s * lastAxisDim;
          final rowPout = result.pointer.cast<ffi.Double>() + s * targetLen;
          kiss_fftri(cfg, rowPin, rowPout);
          for (var i = 0; i < targetLen; i++) {
            rowPout[i] *= scaleFactor;
          }
        }
      } finally {
        ScratchArena.reset(marker);
        if (wasCopied) inputA.dispose();
      }
      return result;
    }

    final marker = ScratchArena.marker;
    try {
      cfg = _getKissFFTRPlan(targetLen, 1);
      final pin = ScratchArena.allocate<kiss_fft_cpx>(
        targetInputLen * ffi.sizeOf<kiss_fft_cpx>(),
      );
      final pout = ScratchArena.allocate<ffi.Double>(
        targetLen * ffi.sizeOf<ffi.Double>(),
      );

      final copyLen = targetInputLen < lastAxisDim
          ? targetInputLen
          : lastAxisDim;
      final scaleFactor = 1.0 / targetLen;

      for (var s = 0; s < signalsCount; s++) {
        final srcStart = s * lastAxisDim;
        final destStart = s * targetLen;

        _copyComplexToDoubleCpx(inputA, srcStart, copyLen, pin);
        for (var i = copyLen; i < targetInputLen; i++) {
          pin[i].r = 0.0;
          pin[i].i = 0.0;
        }

        kiss_fftri(cfg, pin, pout);

        if (result.dtype == DType.float64) {
          final pOut = result.pointer.cast<ffi.Double>() + destStart;
          for (var i = 0; i < targetLen; i++) {
            pOut[i] = pout[i] * scaleFactor;
          }
        } else {
          final pOut = result.pointer.cast<ffi.Float>() + destStart;
          for (var i = 0; i < targetLen; i++) {
            pOut[i] = pout[i] * scaleFactor;
          }
        }
      }
    } finally {
      ScratchArena.reset(marker);
      if (wasCopied) inputA.dispose();
    }
    return result;
  } else {
    // Odd targetLen: Fallback Path
    return NDArray.scope(() {
      final rank = a.rank;
      final normAxis = axis < 0 ? rank + axis : axis;
      final lastAxisDim = a.shape[normAxis]; // M

      final reconDType =
          (a.dtype == DType.complex64 || a.dtype == DType.float32)
          ? DType.complex64
          : DType.complex128;

      final resolvedAxes = List.generate(rank, (i) => i);
      if (normAxis != rank - 1) {
        resolvedAxes[normAxis] = rank - 1;
        resolvedAxes[rank - 1] = normAxis;
      }

      final transposedInput = a.transpose(resolvedAxes);
      final contiguousInput = transposedInput.isContiguous
          ? transposedInput
          : transposedInput.copy();

      final transposedReconShape = List<int>.from(transposedInput.shape);
      transposedReconShape[rank - 1] = targetLen;

      final contiguousRecon = NDArray.zeros(transposedReconShape, reconDType);
      final M = lastAxisDim;
      final totalElements = contiguousInput.size;
      final signalsCount = totalElements ~/ M;

      if (reconDType == DType.complex128) {
        final pIn = contiguousInput.pointer.cast<kiss_fft_cpx>();
        final pOut = contiguousRecon.pointer.cast<kiss_fft_cpx>();

        for (var s = 0; s < signalsCount; s++) {
          final inOffset = s * M;
          final outOffset = s * targetLen;

          for (var i = 0; i < M; i++) {
            pOut[outOffset + i].r = pIn[inOffset + i].r;
            pOut[outOffset + i].i = pIn[inOffset + i].i;
          }
          for (var i = M; i < targetLen; i++) {
            final srcIdx = targetLen - i;
            pOut[outOffset + i].r = pIn[inOffset + srcIdx].r;
            pOut[outOffset + i].i = -pIn[inOffset + srcIdx].i;
          }
        }
      } else {
        final pIn = contiguousInput.pointer.cast<ffi.Float>();
        final pOut = contiguousRecon.pointer.cast<ffi.Float>();

        for (var s = 0; s < signalsCount; s++) {
          final inOffset = s * M * 2;
          final outOffset = s * targetLen * 2;

          for (var i = 0; i < M; i++) {
            pOut[outOffset + 2 * i] = pIn[inOffset + 2 * i];
            pOut[outOffset + 2 * i + 1] = pIn[inOffset + 2 * i + 1];
          }
          for (var i = M; i < targetLen; i++) {
            final srcIdx = targetLen - i;
            pOut[outOffset + 2 * i] = pIn[inOffset + 2 * srcIdx];
            pOut[outOffset + 2 * i + 1] = -pIn[inOffset + 2 * srcIdx + 1];
          }
        }
      }

      if (!transposedInput.isContiguous) {
        contiguousInput.dispose();
      }

      final complexIFFTTransposed = ifft(contiguousRecon);
      final complexIFFT = complexIFFTTransposed.transpose(resolvedAxes);

      final outShape = List<int>.from(a.shape);
      outShape[normAxis] = targetLen;

      final finalResult =
          out ?? NDArray<R>.zeros(outShape, targetDType as DType<R>);
      final numElements = finalResult.size;
      final contiguousIFFT = complexIFFT.isContiguous
          ? complexIFFT
          : complexIFFT.copy();

      if (contiguousIFFT.dtype == DType.complex128) {
        final pIn = contiguousIFFT.pointer.cast<ffi.Double>();
        if (finalResult.dtype == DType.float64) {
          final pOut = finalResult.pointer.cast<ffi.Double>();
          for (var i = 0; i < numElements; i++) {
            pOut[i] = pIn[2 * i];
          }
        } else {
          final pOut = finalResult.pointer.cast<ffi.Float>();
          for (var i = 0; i < numElements; i++) {
            pOut[i] = pIn[2 * i];
          }
        }
      } else {
        final pIn = contiguousIFFT.pointer.cast<ffi.Float>();
        if (finalResult.dtype == DType.float64) {
          final pOut = finalResult.pointer.cast<ffi.Double>();
          for (var i = 0; i < numElements; i++) {
            pOut[i] = pIn[2 * i];
          }
        } else {
          final pOut = finalResult.pointer.cast<ffi.Float>();
          for (var i = 0; i < numElements; i++) {
            pOut[i] = pIn[2 * i];
          }
        }
      }

      if (!complexIFFT.isContiguous) {
        contiguousIFFT.dispose();
      }
      if (out == null) {
        finalResult.detachToParentScope();
      }
      return finalResult;
    });
  }
}

NDArray<R> _fftnND<T, R extends Complex>(
  NDArray<T> a, {
  List<int>? s,
  List<int>? axes,
  bool inverse = false,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute fftn() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write FFT result to a disposed output array.');
  }

  final rank = a.rank;
  if (rank == 0) {
    throw ArgumentError('Cannot compute FFT on a 0-dimensional array.');
  }

  final List<int> axesResolved;
  if (axes == null) {
    if (s == null) {
      axesResolved = List.generate(rank, (i) => i);
    } else {
      axesResolved = List.generate(s.length, (i) => rank - s.length + i);
    }
  } else {
    axesResolved = axes.map((ax) => ax < 0 ? rank + ax : ax).toList();
  }

  for (final ax in axesResolved) {
    if (ax < 0 || ax >= rank) {
      throw RangeError.range(ax, 0, rank - 1, 'axis');
    }
  }
  if (axesResolved.toSet().length != axesResolved.length) {
    throw ArgumentError('axes must be unique');
  }

  final List<int> sResolved;
  if (s == null) {
    sResolved = axesResolved.map((ax) => a.shape[ax]).toList();
  } else {
    sResolved = s;
  }

  if (axesResolved.length != sResolved.length) {
    throw ArgumentError('axes and s must have the same length');
  }
  for (final sz in sResolved) {
    if (sz <= 0) {
      throw ArgumentError('transform size must be positive');
    }
  }

  final isFloatOrComplex = a.dtype.isFloating || a.dtype.isComplex;
  final expectedDType = (a.dtype == DType.float32 || a.dtype == DType.complex64)
      ? DType.complex64
      : DType.complex128;
  final targetDType = out?.dtype ?? expectedDType;

  final outShape = List<int>.from(a.shape);
  for (var i = 0; i < axesResolved.length; i++) {
    outShape[axesResolved[i]] = sResolved[i];
  }

  if (out != null) {
    if (isFloatOrComplex && out.dtype != expectedDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible dtype (expected $expectedDType, got ${out.dtype}).',
      );
    }
    if (!isFloatOrComplex &&
        out.dtype != DType.complex64 &&
        out.dtype != DType.complex128) {
      throw ArgumentError(
        'Provided out buffer has incompatible dtype (expected complex64 or complex128, got ${out.dtype}).',
      );
    }
    if (!listEquals(out.shape, outShape)) {
      throw ArgumentError('Provided out buffer has incompatible shape.');
    }
    if (!out.isContiguous) {
      throw ArgumentError('Provided out buffer must be contiguous.');
    }
  }

  return NDArray.scope(() {
    final prepA = _padOrTruncate<T, R>(
      a,
      sResolved,
      axesResolved,
      targetDType as DType<R>,
    );

    final otherAxes = <int>[];
    for (var i = 0; i < rank; i++) {
      if (!axesResolved.contains(i)) {
        otherAxes.add(i);
      }
    }
    final permutation = [...otherAxes, ...axesResolved];
    final transposed = prepA.transpose(permutation);
    final contiguousTransposed = transposed.isContiguous
        ? transposed
        : transposed.copy();

    final resultTransposed = NDArray<R>.zeros(
      contiguousTransposed.shape,
      targetDType,
    );

    final signalSize = sResolved.reduce((x, y) => x * y);
    final totalElements = contiguousTransposed.size;
    final signalsCount = totalElements ~/ signalSize;
    final scale = inverse ? 1.0 / signalSize : 1.0;

    final marker = ScratchArena.marker;
    try {
      final cfg = _getKissFFTNDPlan(sResolved, inverse ? 1 : 0);

      if (targetDType == DType.complex128) {
        final pIn = contiguousTransposed.pointer.cast<kiss_fft_cpx>();
        final pOut = resultTransposed.pointer.cast<kiss_fft_cpx>();
        for (var s = 0; s < signalsCount; s++) {
          final rowPin = pIn + s * signalSize;
          final rowPout = pOut + s * signalSize;
          kiss_fftnd(cfg, rowPin, rowPout);
          if (inverse) {
            final pOutD = rowPout.cast<ffi.Double>();
            final numDoubles = signalSize * 2;
            for (var i = 0; i < numDoubles; i++) {
              pOutD[i] *= scale;
            }
          }
        }
      } else {
        final pin = ScratchArena.allocate<kiss_fft_cpx>(
          signalSize * ffi.sizeOf<kiss_fft_cpx>(),
        );
        final pout = ScratchArena.allocate<kiss_fft_cpx>(
          signalSize * ffi.sizeOf<kiss_fft_cpx>(),
        );

        final pIn = contiguousTransposed.pointer.cast<ffi.Float>();
        final pOut = resultTransposed.pointer.cast<ffi.Float>();

        for (var s = 0; s < signalsCount; s++) {
          final inOffset = s * signalSize * 2;
          final outOffset = s * signalSize * 2;

          for (var i = 0; i < signalSize; i++) {
            pin[i].r = pIn[inOffset + 2 * i];
            pin[i].i = pIn[inOffset + 2 * i + 1];
          }

          kiss_fftnd(cfg, pin, pout);

          for (var i = 0; i < signalSize; i++) {
            pOut[outOffset + 2 * i] = pout[i].r * scale;
            pOut[outOffset + 2 * i + 1] = pout[i].i * scale;
          }
        }
      }
    } finally {
      ScratchArena.reset(marker);
    }

    final invPermutation = _inversePermutation(permutation);
    final result = resultTransposed.transpose(invPermutation);

    if (out != null) {
      result.copy(out: out);
      return out;
    } else {
      return result.isContiguous
          ? result.detachToParentScope()
          : result.copy().detachToParentScope();
    }
  });
}

/// Computes the N-dimensional discrete Fourier Transform.
///
/// Computes the discrete Fourier Transform over multiple axes of a
/// multidimensional array.
///
/// **Preconditions:**
/// - It is an error if input array [a] or [out] is disposed.
/// - It is an error if input array [a] is 0-dimensional (rank < 1).
/// - It is an error if [axes] and [s] length mismatch.
/// - It is an error if any axis index in [axes] is out of bounds `[-a.rank, a.rank - 1]` or contains duplicates.
/// - It is an error if any dimension in [s] is $\le 0$.
/// - It is an error if [out] has incompatible shape, dtype, or is not contiguous.
///
/// **Performance considerations:**
/// - Uses `kiss_fftnd` plan which is optimized for multi-dimensional transforms.
/// - Transposes the array to bring target [axes] to the end before calling native C code, which is fast but might require a copy to make it contiguous.
///
/// Reference: [N-dimensional FFT](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftn.html)
NDArray<R> fftn<T, R extends Complex>(
  NDArray<T> a, {
  List<int>? s,
  List<int>? axes,
  NDArray<R>? out,
}) => _fftnND<T, R>(a, s: s, axes: axes, inverse: false, out: out);

/// Computes the N-dimensional inverse discrete Fourier Transform.
///
/// Computes the inverse of [fftn].
///
/// **Preconditions:**
/// - Same as [fftn].
///
/// **Performance considerations:**
/// - Same as [fftn].
///
/// Reference: [Inverse N-dimensional FFT](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftn.html)
NDArray<R> ifftn<T, R extends Complex>(
  NDArray<T> a, {
  List<int>? s,
  List<int>? axes,
  NDArray<R>? out,
}) => _fftnND<T, R>(a, s: s, axes: axes, inverse: true, out: out);

/// Computes the 2-dimensional discrete Fourier Transform.
///
/// Equivalent to calling [fftn] with [axes] defaulting to the last two axes `[-2, -1]`.
///
/// **Preconditions:**
/// - It is an error if input array [a] or [out] is disposed.
/// - It is an error if input array [a] has rank < 2.
/// - It is an error if [axes] is provided and does not have length 2.
/// - It is an error if [s] is provided and does not have length 2.
/// - It is an error if any axis index in [axes] is out of bounds `[-a.rank, a.rank - 1]` or contains duplicates.
/// - It is an error if any dimension in [s] is $\le 0$.
/// - It is an error if [out] has incompatible shape, dtype, or is not contiguous.
///
/// Reference: [2-dimensional FFT](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html)
NDArray<R> fft2<T, R extends Complex>(
  NDArray<T> a, {
  List<int>? s,
  List<int>? axes = const [-2, -1],
  NDArray<R>? out,
}) {
  final resolvedAxes = axes ?? const [-2, -1];
  if (resolvedAxes.length != 2) {
    throw ArgumentError('axes must have length 2');
  }
  return fftn<T, R>(a, s: s, axes: resolvedAxes, out: out);
}

/// Computes the 2-dimensional inverse discrete Fourier Transform.
///
/// Equivalent to calling [ifftn] with [axes] defaulting to the last two axes `[-2, -1]`.
///
/// **Preconditions:**
/// - It is an error if input array [a] or [out] is disposed.
/// - It is an error if input array [a] has rank < 2.
/// - It is an error if [axes] is provided and does not have length 2.
/// - It is an error if [s] is provided and does not have length 2.
/// - It is an error if any axis index in [axes] is out of bounds `[-a.rank, a.rank - 1]` or contains duplicates.
/// - It is an error if any dimension in [s] is $\le 0$.
/// - It is an error if [out] has incompatible shape, dtype, or is not contiguous.
///
/// Reference: [Inverse 2-dimensional FFT](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft2.html)
NDArray<R> ifft2<T, R extends Complex>(
  NDArray<T> a, {
  List<int>? s,
  List<int>? axes = const [-2, -1],
  NDArray<R>? out,
}) {
  final resolvedAxes = axes ?? const [-2, -1];
  if (resolvedAxes.length != 2) {
    throw ArgumentError('axes must have length 2');
  }
  return ifftn<T, R>(a, s: s, axes: resolvedAxes, out: out);
}

/// Clears all precomputed native FFT plans from the isolate plan cache.
///
/// Frees unmanaged C heap memory allocated for twiddle factor tables and
/// multidimensional transform buffers across 1D complex, 1D real, and
/// N-dimensional FFT plans.
void clearFFTPlanCache() {
  PocketFFTPlanCache.instance.clear();
}
