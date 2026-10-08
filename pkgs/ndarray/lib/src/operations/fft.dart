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
import 'helpers.dart';
import 'manipulation.dart';

NDArray _createZeros(List<int> shape, DType dtype) => switch (dtype) {
  DType.float64 => NDArray<Float64>.zeros(shape, DType.float64),
  DType.float32 => NDArray<Float32>.zeros(shape, DType.float32),
  DType.float16 => NDArray<Float16>.zeros(shape, DType.float16),
  DType.bfloat16 => NDArray<BFloat16>.zeros(shape, DType.bfloat16),
  DType.int64 => NDArray<Int64>.zeros(shape, DType.int64),
  DType.int32 => NDArray<Int32>.zeros(shape, DType.int32),
  DType.int16 => NDArray<Int16>.zeros(shape, DType.int16),
  DType.int8 => NDArray<Int8>.zeros(shape, DType.int8),
  DType.uint64 => NDArray<Uint64>.zeros(shape, DType.uint64),
  DType.uint32 => NDArray<Uint32>.zeros(shape, DType.uint32),
  DType.uint16 => NDArray<Uint16>.zeros(shape, DType.uint16),
  DType.uint8 => NDArray<Uint8>.zeros(shape, DType.uint8),
  DType.complex128 => NDArray<Complex128>.zeros(shape, DType.complex128),
  DType.complex64 => NDArray<Complex64>.zeros(shape, DType.complex64),
  DType.boolean => NDArray<Boolean>.zeros(shape, DType.boolean),
};

/// Helper to allocate a KissFFT plan configuration on the ScratchArena stack.

int _getSignalOffset(int s, List<int> shape, List<int> strides) {
  final rank = shape.length;
  if (rank <= 1) return 0;
  int offset = 0;
  int temp = s;
  for (var d = rank - 2; d >= 0; d--) {
    final dim = shape[d];
    final coord = temp % dim;
    temp ~/= dim;
    offset += coord * strides[d];
  }
  return offset;
}

void _loadSignalToKissInput<T extends DTypeTag>(
  NDArray<T> inputA,
  int srcStart,
  int copyLen,
  int targetLen,
  ffi.Pointer<kiss_fft_cpx> pin, {
  int elemStride = 1,
}) {
  if (elemStride == 1) {
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
          pin[i].r = BigInt.from(inPtr[i]).toUnsigned(64).toDouble();
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
  } else {
    switch (inputA.dtype) {
      case DType.complex128:
        final inPtr = inputA.pointer.cast<kiss_fft_cpx>() + srcStart;
        for (var i = 0; i < copyLen; i++) {
          final idx = i * elemStride;
          pin[i].r = inPtr[idx].r;
          pin[i].i = inPtr[idx].i;
        }
      case DType.complex64:
        final inPtr = inputA.pointer.cast<ffi.Float>() + (srcStart * 2);
        for (var i = 0; i < copyLen; i++) {
          final idx = (i * elemStride) * 2;
          pin[i].r = inPtr[idx];
          pin[i].i = inPtr[idx + 1];
        }
      case DType.float64:
        final inPtr = inputA.pointer.cast<ffi.Double>() + srcStart;
        for (var i = 0; i < copyLen; i++) {
          pin[i].r = inPtr[i * elemStride];
          pin[i].i = 0.0;
        }
      case DType.float32:
        final inPtr = inputA.pointer.cast<ffi.Float>() + srcStart;
        for (var i = 0; i < copyLen; i++) {
          pin[i].r = inPtr[i * elemStride];
          pin[i].i = 0.0;
        }
      case DType.int64:
        final inPtr = inputA.pointer.cast<ffi.Int64>() + srcStart;
        for (var i = 0; i < copyLen; i++) {
          pin[i].r = inPtr[i * elemStride].toDouble();
          pin[i].i = 0.0;
        }
      case DType.int32:
        final inPtr = inputA.pointer.cast<ffi.Int32>() + srcStart;
        for (var i = 0; i < copyLen; i++) {
          pin[i].r = inPtr[i * elemStride].toDouble();
          pin[i].i = 0.0;
        }
      case DType.int16:
        final inPtr = inputA.pointer.cast<ffi.Int16>() + srcStart;
        for (var i = 0; i < copyLen; i++) {
          pin[i].r = inPtr[i * elemStride].toDouble();
          pin[i].i = 0.0;
        }
      case DType.uint8:
        final inPtr = inputA.pointer.cast<ffi.Uint8>() + srcStart;
        for (var i = 0; i < copyLen; i++) {
          pin[i].r = inPtr[i * elemStride].toDouble();
          pin[i].i = 0.0;
        }
      case DType.int8:
        final inPtr = inputA.pointer.cast<ffi.Int8>() + srcStart;
        for (var i = 0; i < copyLen; i++) {
          pin[i].r = inPtr[i * elemStride].toDouble();
          pin[i].i = 0.0;
        }
      case DType.uint64:
        final inPtr = inputA.pointer.cast<ffi.Uint64>() + srcStart;
        for (var i = 0; i < copyLen; i++) {
          pin[i].r = BigInt.from(
            inPtr[i * elemStride],
          ).toUnsigned(64).toDouble();
          pin[i].i = 0.0;
        }
      case DType.uint32:
        final inPtr = inputA.pointer.cast<ffi.Uint32>() + srcStart;
        for (var i = 0; i < copyLen; i++) {
          pin[i].r = inPtr[i * elemStride].toDouble();
          pin[i].i = 0.0;
        }
      case DType.uint16:
        final inPtr = inputA.pointer.cast<ffi.Uint16>() + srcStart;
        for (var i = 0; i < copyLen; i++) {
          pin[i].r = inPtr[i * elemStride].toDouble();
          pin[i].i = 0.0;
        }
      case DType.float16:
        final inPtr = inputA.pointer.cast<ffi.Uint16>() + srcStart;
        for (var i = 0; i < copyLen; i++) {
          pin[i].r = Float16Utils.decodeFloat16(inPtr[i * elemStride]);
          pin[i].i = 0.0;
        }
      case DType.bfloat16:
        final inPtr = inputA.pointer.cast<ffi.Uint16>() + srcStart;
        for (var i = 0; i < copyLen; i++) {
          pin[i].r = Float16Utils.decodeBFloat16(inPtr[i * elemStride]);
          pin[i].i = 0.0;
        }
      case DType.boolean:
        final inPtr = inputA.pointer.cast<ffi.Uint8>() + srcStart;
        for (var i = 0; i < copyLen; i++) {
          pin[i].r = inPtr[i * elemStride] != 0 ? 1.0 : 0.0;
          pin[i].i = 0.0;
        }
    }
  }
  for (var i = copyLen; i < targetLen; i++) {
    pin[i].r = 0.0;
    pin[i].i = 0.0;
  }
}

void _storeKissOutputToResult<R extends DTypeTag>(
  NDArray<R> result,
  int destStart,
  int targetLen,
  ffi.Pointer<kiss_fft_cpx> pout, {
  double scale = 1.0,
  int elemStride = 1,
}) {
  if (elemStride == 1) {
    switch (result.dtype) {
      case DType.complex128:
        final outPtr = result.pointer.cast<kiss_fft_cpx>() + destStart;
        for (var i = 0; i < targetLen; i++) {
          outPtr[i].r = pout[i].r * scale;
          outPtr[i].i = pout[i].i * scale;
        }
      case DType.complex64:
        final outPtr = result.pointer.cast<ffi.Float>() + (destStart * 2);
        for (var i = 0; i < targetLen; i++) {
          outPtr[2 * i] = (pout[i].r * scale);
          outPtr[2 * i + 1] = (pout[i].i * scale);
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
        throw UnsupportedError('Unsupported FFT output dtype: ${result.dtype}');
    }
  } else {
    switch (result.dtype) {
      case DType.complex128:
        final outPtr = result.pointer.cast<kiss_fft_cpx>() + destStart;
        for (var i = 0; i < targetLen; i++) {
          final idx = i * elemStride;
          outPtr[idx].r = pout[i].r * scale;
          outPtr[idx].i = pout[i].i * scale;
        }
      case DType.complex64:
        final outPtr = result.pointer.cast<ffi.Float>() + (destStart * 2);
        for (var i = 0; i < targetLen; i++) {
          final idx = (i * elemStride) * 2;
          outPtr[idx] = (pout[i].r * scale);
          outPtr[idx + 1] = (pout[i].i * scale);
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
        throw UnsupportedError('Unsupported FFT output dtype: ${result.dtype}');
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
/// - It is an error if [out] has incompatible shape or dtype.
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
NDArray<R> fft<R extends DTypeTag>(
  NDArray<ComplexOf<R>> a, {
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
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must not be 0-dimensional or empty scalar array (cannot compute FFT on a 0-dimensional or empty scalar array)',
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
    throw ArgumentError.value(
      n,
      'n',
      'Must be greater than 0 (target transform length [n] was $n)',
    );
  }

  // Formulate output shape by replacing the transform axis with targetLen
  final outShape = List<int>.from(a.shape);
  outShape[normAxis] = targetLen;

  final isFloatOrComplex = a.dtype.isFloating || a.dtype.isComplex;
  final expectedDType =
      ((a.dtype as DType<DTypeTag>) == DType.float32 ||
          (a.dtype as DType<DTypeTag>) == DType.complex64)
      ? DType.complex64
      : DType.complex128;
  final targetDType = out?.dtype ?? expectedDType;

  if (out != null) {
    validateOutBuffer(out);
    if (isFloatOrComplex && out.dtype != expectedDType) {
      throw ArgumentError.value(
        out.dtype,
        'out.dtype',
        'Must be $expectedDType (provided out buffer has incompatible dtype, got ${out.dtype})',
      );
    }
    if (!isFloatOrComplex &&
        out.dtype != DType.complex64 &&
        out.dtype != DType.complex128) {
      throw ArgumentError.value(
        out.dtype,
        'out.dtype',
        'Must be complex64 or complex128 (provided out buffer has incompatible dtype, got ${out.dtype})',
      );
    }
    if (!listEquals(out.shape, outShape)) {
      throw ArgumentError.value(
        out.shape,
        'out.shape',
        'Must match expected output shape $outShape (provided out buffer has incompatible shape)',
      );
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = _createZeros(outShape, out.dtype) as NDArray<R>;
        fft<R>(a, n: n, axis: axis, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  if (a.size == 0 || lastAxisDim == 0) {
    return NDArray.scope(() {
      final zeroOut = _createZeros(outShape, targetDType) as NDArray<R>;
      if (out != null) {
        zeroOut.copy(out: out);
        return out;
      }
      zeroOut.detachToParentScope();
      return zeroOut;
    });
  }

  if (normAxis != rank - 1) {
    final axes = List.generate(rank, (i) => i);
    axes[normAxis] = rank - 1;
    axes[rank - 1] = normAxis;

    return NDArray.scope(() {
      final transposedInput = a.transpose(axes);
      if (out != null) {
        final transposedOut = out.transpose(axes);
        fft<R>(transposedInput, n: n, out: transposedOut);
        return out;
      } else {
        final result = _createZeros(outShape, targetDType) as NDArray<R>;
        final transposedOut = result.transpose(axes);
        fft<R>(transposedInput, n: n, out: transposedOut);
        result.detachToParentScope();
        return result;
      }
    });
  }

  return NDArray.scope(() {
    final NDArray<DTypeTag> inputA = a;
    final result = out ?? _createZeros(outShape, targetDType) as NDArray<R>;

    // Count how many 1D row sub-signals exist to execute strided walks
    final signalsCount = rank <= 1
        ? 1
        : inputA.shape.take(rank - 1).reduce((x, y) => x * y);

    final isZeroCopyFastPath =
        inputA.dtype == DType.complex128 &&
        result.dtype == DType.complex128 &&
        targetLen == lastAxisDim &&
        inputA.isContiguous &&
        result.isContiguous;

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
      }
      if (out == null) {
        result.detachToParentScope();
      }
      return result;
    }

    final marker = ScratchArena.marker;
    try {
      cfg = _getKissFFTPlan(targetLen, 0);
      final pin = ScratchArena.allocate<kiss_fft_cpx>(
        targetLen * ffi.sizeOf<kiss_fft_cpx>(),
      );
      final pout = ScratchArena.allocate<kiss_fft_cpx>(
        targetLen * ffi.sizeOf<kiss_fft_cpx>(),
      );

      final copyLen = targetLen < lastAxisDim ? targetLen : lastAxisDim;
      final inStride = inputA.strides.isEmpty ? 1 : inputA.strides.last;
      final outStride = result.strides.isEmpty ? 1 : result.strides.last;

      for (var s = 0; s < signalsCount; s++) {
        final srcStart = _getSignalOffset(s, inputA.shape, inputA.strides);
        final destStart = _getSignalOffset(s, result.shape, result.strides);

        _loadSignalToKissInput(
          inputA,
          srcStart,
          copyLen,
          targetLen,
          pin,
          elemStride: inStride,
        );

        // 3. Fire high-speed native FFT on the C heap components
        kiss_fft(cfg, pin, pout);

        // 4. Collect results from pout back into result array
        _storeKissOutputToResult(
          result,
          destStart,
          targetLen,
          pout,
          elemStride: outStride,
        );
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
/// - It is an error if [out] has incompatible shape or dtype.
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
NDArray<R> ifft<R extends DTypeTag>(
  NDArray<ComplexOf<R>> a, {
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
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must not be 0-dimensional or empty scalar array (cannot compute IFFT on a 0-dimensional or empty scalar array)',
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
    throw ArgumentError.value(
      n,
      'n',
      'Must be greater than 0 (target transform length [n] was $n)',
    );
  }

  // Formulate output shape by replacing the transform axis with targetLen
  final outShape = List<int>.from(a.shape);
  outShape[normAxis] = targetLen;

  final isFloatOrComplex = a.dtype.isFloating || a.dtype.isComplex;
  final expectedDType =
      ((a.dtype as DType<DTypeTag>) == DType.float32 ||
          (a.dtype as DType<DTypeTag>) == DType.complex64)
      ? DType.complex64
      : DType.complex128;
  final targetDType = out?.dtype ?? expectedDType;

  if (out != null) {
    validateOutBuffer(out);
    if (isFloatOrComplex && out.dtype != expectedDType) {
      throw ArgumentError.value(
        out.dtype,
        'out.dtype',
        'Must be $expectedDType (provided out buffer has incompatible dtype, got ${out.dtype})',
      );
    }
    if (!isFloatOrComplex &&
        out.dtype != DType.complex64 &&
        out.dtype != DType.complex128) {
      throw ArgumentError.value(
        out.dtype,
        'out.dtype',
        'Must be complex64 or complex128 (provided out buffer has incompatible dtype, got ${out.dtype})',
      );
    }
    if (!listEquals(out.shape, outShape)) {
      throw ArgumentError.value(
        out.shape,
        'out.shape',
        'Must match expected output shape $outShape (provided out buffer has incompatible shape)',
      );
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = _createZeros(outShape, out.dtype) as NDArray<R>;
        ifft<R>(a, n: n, axis: axis, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  if (a.size == 0 || lastAxisDim == 0) {
    return NDArray.scope(() {
      final zeroOut = _createZeros(outShape, targetDType) as NDArray<R>;
      if (out != null) {
        zeroOut.copy(out: out);
        return out;
      }
      zeroOut.detachToParentScope();
      return zeroOut;
    });
  }

  if (normAxis != rank - 1) {
    final axes = List.generate(rank, (i) => i);
    axes[normAxis] = rank - 1;
    axes[rank - 1] = normAxis;

    return NDArray.scope(() {
      final transposedInput = a.transpose(axes);
      if (out != null) {
        final transposedOut = out.transpose(axes);
        ifft<R>(transposedInput, n: n, out: transposedOut);
        return out;
      } else {
        final result = _createZeros(outShape, targetDType) as NDArray<R>;
        final transposedOut = result.transpose(axes);
        ifft<R>(transposedInput, n: n, out: transposedOut);
        result.detachToParentScope();
        return result;
      }
    });
  }

  return NDArray.scope(() {
    final NDArray<DTypeTag> inputA = a;
    final result = out ?? _createZeros(outShape, targetDType) as NDArray<R>;

    final signalsCount = rank <= 1
        ? 1
        : inputA.shape.take(rank - 1).reduce((x, y) => x * y);

    final isZeroCopyFastPath =
        inputA.dtype == DType.complex128 &&
        result.dtype == DType.complex128 &&
        targetLen == lastAxisDim &&
        inputA.isContiguous &&
        result.isContiguous;

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
      }
      if (out == null) {
        result.detachToParentScope();
      }
      return result;
    }

    final marker = ScratchArena.marker;
    try {
      cfg = _getKissFFTPlan(targetLen, 1);
      final pin = ScratchArena.allocate<kiss_fft_cpx>(
        targetLen * ffi.sizeOf<kiss_fft_cpx>(),
      );
      final pout = ScratchArena.allocate<kiss_fft_cpx>(
        targetLen * ffi.sizeOf<kiss_fft_cpx>(),
      );

      final copyLen = targetLen < lastAxisDim ? targetLen : lastAxisDim;
      final scaleFactor = 1.0 / targetLen;
      final inStride = inputA.strides.isEmpty ? 1 : inputA.strides.last;
      final outStride = result.strides.isEmpty ? 1 : result.strides.last;

      for (var s = 0; s < signalsCount; s++) {
        final srcStart = _getSignalOffset(s, inputA.shape, inputA.strides);
        final destStart = _getSignalOffset(s, result.shape, result.strides);

        _loadSignalToKissInput(
          inputA,
          srcStart,
          copyLen,
          targetLen,
          pin,
          elemStride: inStride,
        );

        // 2. Fire high-speed native inverse transform
        kiss_fft(cfg, pin, pout);

        // 3. Apply standard 1/N scaling factor normalization (KissFFT leaves it unscaled)
        _storeKissOutputToResult(
          result,
          destStart,
          targetLen,
          pout,
          scale: scaleFactor,
          elemStride: outStride,
        );
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
NDArray<T> fftshift<T extends DTypeTag>(
  NDArray<T> a, {
  Object? axes,
  NDArray<T>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot shift a disposed array.');
  }
  if (out != null) {
    validateOutBuffer(out);
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
    for (final axis in axes) {
      final norm = axis < 0 ? rank + axis : axis;
      if (norm < 0 || norm >= rank) {
        throw RangeError.range(axis, -rank, rank - 1, 'axes');
      }
      if (resolvedAxes.contains(norm)) {
        throw ArgumentError.value(
          axis,
          'axes',
          'Must not contain duplicate axes (duplicate axis $norm specified in axes)',
        );
      }
      resolvedAxes.add(norm);
    }
  } else {
    throw ArgumentError.value(
      axes,
      'axes',
      'Must be null, an integer, or a list of integers',
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
NDArray<T> ifftshift<T extends DTypeTag>(
  NDArray<T> a, {
  Object? axes,
  NDArray<T>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot shift a disposed array.');
  }
  if (out != null) {
    validateOutBuffer(out);
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
    for (final axis in axes) {
      final norm = axis < 0 ? rank + axis : axis;
      if (norm < 0 || norm >= rank) {
        throw RangeError.range(axis, -rank, rank - 1, 'axes');
      }
      if (resolvedAxes.contains(norm)) {
        throw ArgumentError.value(
          axis,
          'axes',
          'Must not contain duplicate axes (duplicate axis $norm specified in axes)',
        );
      }
      resolvedAxes.add(norm);
    }
  } else {
    throw ArgumentError.value(
      axes,
      'axes',
      'Must be null, an integer, or a list of integers',
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
    case DType.int8:
      final pIn = a.pointer.cast<ffi.Int8>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i] = pIn[i].toDouble();
      }
    case DType.uint64:
      final pIn = a.pointer.cast<ffi.Uint64>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i] = BigInt.from(pIn[i]).toUnsigned(64).toDouble();
      }
    case DType.uint32:
      final pIn = a.pointer.cast<ffi.Uint32>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i] = pIn[i].toDouble();
      }
    case DType.uint16:
      final pIn = a.pointer.cast<ffi.Uint16>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i] = pIn[i].toDouble();
      }
    case DType.uint8:
      final pIn = a.pointer.cast<ffi.Uint8>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i] = pIn[i].toDouble();
      }
    case DType.float16:
      final pIn = a.pointer.cast<ffi.Uint16>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i] = Float16Utils.decodeFloat16(pIn[i]);
      }
    case DType.bfloat16:
      final pIn = a.pointer.cast<ffi.Uint16>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i] = Float16Utils.decodeBFloat16(pIn[i]);
      }
    case DType.boolean:
      final pIn = a.pointer.cast<ffi.Uint8>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i] = pIn[i] != 0 ? 1.0 : 0.0;
      }
    case DType.complex128:
    case DType.complex64:
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
    case DType.int8:
      final pIn = a.pointer.cast<ffi.Int8>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i].r = pIn[i].toDouble();
        dest[i].i = 0.0;
      }
    case DType.uint64:
      final pIn = a.pointer.cast<ffi.Uint64>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i].r = BigInt.from(pIn[i]).toUnsigned(64).toDouble();
        dest[i].i = 0.0;
      }
    case DType.uint32:
      final pIn = a.pointer.cast<ffi.Uint32>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i].r = pIn[i].toDouble();
        dest[i].i = 0.0;
      }
    case DType.uint16:
      final pIn = a.pointer.cast<ffi.Uint16>() + offset;
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
    case DType.float16:
      final pIn = a.pointer.cast<ffi.Uint16>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i].r = Float16Utils.decodeFloat16(pIn[i]);
        dest[i].i = 0.0;
      }
    case DType.bfloat16:
      final pIn = a.pointer.cast<ffi.Uint16>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i].r = Float16Utils.decodeBFloat16(pIn[i]);
        dest[i].i = 0.0;
      }
    case DType.boolean:
      final pIn = a.pointer.cast<ffi.Uint8>() + offset;
      for (var i = 0; i < count; i++) {
        dest[i].r = pIn[i] != 0 ? 1.0 : 0.0;
        dest[i].i = 0.0;
      }
  }
}

NDArray<R> _promoteToComplex<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a,
  DType<R> targetDType,
) {
  if (a.dtype == targetDType) {
    if (a is NDArray<R>) {
      return (a as NDArray<R>).copy();
    }
    final view = NDArray<R>.view(a, shape: a.shape, strides: a.strides);
    try {
      return view.copy();
    } finally {
      view.dispose();
    }
  }
  return castNDArray(a, targetDType);
}

NDArray<R> _padOrTruncate<T extends DTypeTag, R extends DTypeTag>(
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

  if (!needsSliceOrPad && arr.dtype == targetDType && arr.isContiguous) {
    if (arr is NDArray<R>) {
      return (arr as NDArray<R>).copy();
    }
    final view = NDArray<R>.view(arr, shape: arr.shape, strides: arr.strides);
    try {
      return view.copy();
    } finally {
      view.dispose();
    }
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
NDArray<Float64> fftfreq(int n, {double d = 1.0, NDArray<Float64>? out}) {
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write fftfreq result to a disposed output array.');
  }
  if (n <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be strictly positive (was $n)');
  }
  if (d == 0.0) {
    throw ArgumentError.value(
      d,
      'd',
      'Must be non-zero (sample spacing d must be non-zero)',
    );
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, [n]) || out.dtype != DType.float64) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape [$n] and dtype ${DType.float64} (incompatible out buffer shape or dtype)',
      );
    }
  }
  final val = 1.0 / (d * n);
  final result =
      out ?? NDArray<Float64>.create([n], DType.float64, zeroInit: false);
  final half = n.isEven ? (n ~/ 2) : ((n - 1) ~/ 2 + 1);
  if (result.isContiguous) {
    final ptr = result.pointer.cast<ffi.Double>();
    for (var i = 0; i < half; i++) {
      ptr[i] = i * val;
    }
    for (var i = half; i < n; i++) {
      ptr[i] = (i - n) * val;
    }
  } else {
    for (var i = 0; i < half; i++) {
      result.setCell([i], i * val);
    }
    for (var i = half; i < n; i++) {
      result.setCell([i], (i - n) * val);
    }
  }
  return result;
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
NDArray<Float64> rfftfreq(int n, {double d = 1.0, NDArray<Float64>? out}) {
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write rfftfreq result to a disposed output array.',
    );
  }
  if (n <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be strictly positive (was $n)');
  }
  if (d == 0.0) {
    throw ArgumentError.value(
      d,
      'd',
      'Must be non-zero (sample spacing d must be non-zero)',
    );
  }
  final val = 1.0 / (d * n);
  final limit = n ~/ 2 + 1;
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, [limit]) || out.dtype != DType.float64) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape [$limit] and dtype ${DType.float64} (incompatible out buffer shape or dtype)',
      );
    }
  }
  final result =
      out ?? NDArray<Float64>.create([limit], DType.float64, zeroInit: false);
  if (result.isContiguous) {
    final ptr = result.pointer.cast<ffi.Double>();
    for (var i = 0; i < limit; i++) {
      ptr[i] = i * val;
    }
  } else {
    for (var i = 0; i < limit; i++) {
      result.setCell([i], i * val);
    }
  }
  return result;
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
/// - It is an error if [a] has a complex dtype.
/// - It is an error if the input array [a] shape is empty (scalar 0D, rank < 1).
/// - It is an error if the specified [axis] is out of bounds `[-a.rank, a.rank - 1]`.
/// - It is an error if target length [n] is provided but is less than or equal to 0.
/// - It is an error if [out] is provided and has incompatible shape (`(n ?? a.shape[axis]) // 2 + 1` along [axis]) or incompatible dtype (`complex64` if input is `float32`, `complex128` otherwise).
///
/// **Throws:**
/// - [ArgumentError] if [a] is complex or has an empty shape (scalar 0D).
/// - [RangeError] if the specified [axis] is out of bounds.
/// - [ArgumentError] if [n] is provided but is less than or equal to 0.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
/// - [StateError] if native FFI memory allocations fail.
///
/// **Performance considerations:**
/// - Both even and odd lengths use the native C++ PocketFFT real-to-complex pathway via `kiss_fftr`.
///
/// Reference: [Real 1D FFT](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html)
NDArray<R> rfft<R extends DTypeTag>(
  NDArray<ComplexOf<R>> a, {
  int? n,
  int axis = -1,
  NDArray<R>? out,
}) => _rfftImpl<R>(a, n: n, axis: axis, out: out, isIhfft: false);

/// Computes the inverse FFT of a signal that has Hermitian symmetry.
///
/// Computes the one-dimensional inverse discrete Fourier Transform of a
/// real-valued input signal [a], returning the non-negative frequency terms of length
/// `n // 2 + 1` along [axis]. This is the exact inverse of [hfft].
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] has a complex dtype.
/// - It is an error if the input array [a] shape is empty (scalar 0D, rank < 1).
/// - It is an error if the specified [axis] is out of bounds `[-a.rank, a.rank - 1]`.
/// - It is an error if target length [n] is provided but is less than or equal to 0.
/// - It is an error if [out] is provided and has incompatible shape (`(n ?? a.shape[axis]) // 2 + 1` along [axis]) or incompatible dtype (`complex64` if input is `float32`, `complex128` otherwise).
///
/// Reference: [NumPy ihfft](https://numpy.org/doc/stable/reference/generated/numpy.fft.ihfft.html)
NDArray<R> ihfft<R extends DTypeTag>(
  NDArray<ComplexOf<R>> a, {
  int? n,
  int axis = -1,
  NDArray<R>? out,
}) => _rfftImpl<R>(a, n: n, axis: axis, out: out, isIhfft: true);

NDArray<R> _rfftImpl<R extends DTypeTag>(
  NDArray<ComplexOf<R>> a, {
  int? n,
  int axis = -1,
  NDArray<R>? out,
  required bool isIhfft,
}) {
  if (a.isDisposed) {
    throw StateError(
      'Cannot execute ${isIhfft ? 'ihfft' : 'rfft'}() on a disposed array.',
    );
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write FFT result to a disposed output array.');
  }
  if (a.dtype.isComplex) {
    throw ArgumentError.value(a.dtype, 'a', 'Must be a real-valued array');
  }
  if (a.shape.isEmpty) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must not be a 0-dimensional or empty scalar array',
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
    throw ArgumentError.value(n, 'n', 'Must be greater than 0');
  }

  final outShape = List<int>.from(a.shape);
  outShape[normAxis] = targetLen ~/ 2 + 1;

  final isFloat = a.dtype.isFloating;
  final expectedDType = (a.dtype as DType<DTypeTag>) == DType.float32
      ? DType.complex64
      : DType.complex128;
  final targetDType = out?.dtype ?? expectedDType;

  if (out != null) {
    validateOutBuffer(out);
    if (isFloat && out.dtype != expectedDType) {
      throw ArgumentError.value(
        out.dtype,
        'out',
        'Must have dtype $expectedDType',
      );
    }
    if (!isFloat &&
        out.dtype != DType.complex64 &&
        out.dtype != DType.complex128) {
      throw ArgumentError.value(
        out.dtype,
        'out',
        'Must have dtype complex64 or complex128',
      );
    }
    if (!listEquals(out.shape, outShape)) {
      throw ArgumentError.value(out.shape, 'out', 'Must have shape $outShape');
    }
    if (!out.isContiguous || sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = _createZeros(outShape, out.dtype) as NDArray<R>;
        _rfftImpl<R>(a, n: n, axis: axis, out: temp, isIhfft: isIhfft);
        temp.copy(out: out);
        return out;
      });
    }
  }

  if (a.size == 0 || lastAxisDim == 0) {
    return NDArray.scope(() {
      final zeroOut = _createZeros(outShape, targetDType) as NDArray<R>;
      if (out != null) {
        zeroOut.copy(out: out);
        return out;
      }
      zeroOut.detachToParentScope();
      return zeroOut;
    });
  }

  if (normAxis != rank - 1) {
    final axes = List.generate(rank, (i) => i);
    axes[normAxis] = rank - 1;
    axes[rank - 1] = normAxis;

    return NDArray.scope(() {
      final transposedInput = a.transpose(axes);
      final transposedResult = _rfftImpl<R>(
        transposedInput,
        n: n,
        isIhfft: isIhfft,
      );
      final finalResult = transposedResult.transpose(axes);
      if (out != null) {
        finalResult.copy(out: out);
        return out;
      } else {
        final resCopy = finalResult.copy();
        resCopy.detachToParentScope();
        return resCopy;
      }
    });
  }

  return NDArray.scope(() {
    final inputA = a.isContiguous ? a : a.copy();
    final result = out ?? _createZeros(outShape, targetDType) as NDArray<R>;

    final totalElements = inputA.shape.reduce((x, y) => x * y);
    final signalsCount = totalElements ~/ lastAxisDim;

    final isZeroCopyFastPath =
        (inputA.dtype as DType<DTypeTag>) == DType.float64 &&
        targetLen == lastAxisDim &&
        inputA.isContiguous &&
        result.dtype == DType.complex128;

    kiss_fftr_cfg cfg = ffi.nullptr.cast();
    final scale = isIhfft ? 1.0 / targetLen : 1.0;

    if (isZeroCopyFastPath) {
      final marker = ScratchArena.marker;
      try {
        cfg = _getKissFFTRPlan(targetLen, 0);
        final outLen = targetLen ~/ 2 + 1;
        for (var s = 0; s < signalsCount; s++) {
          final rowPin = inputA.pointer.cast<ffi.Double>() + s * lastAxisDim;
          final rowPout = result.pointer.cast<kiss_fft_cpx>() + s * outLen;
          kiss_fftr(cfg, rowPin, rowPout);
          if (isIhfft) {
            for (var i = 0; i < outLen; i++) {
              rowPout[i].r = rowPout[i].r * scale;
              rowPout[i].i = -rowPout[i].i * scale;
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
          if (!isIhfft) {
            final pPoutD = pout.cast<ffi.Double>();
            final numDoubles = outLen * 2;
            for (var i = 0; i < numDoubles; i++) {
              pOut[i] = pPoutD[i];
            }
          } else {
            for (var i = 0; i < outLen; i++) {
              pOut[2 * i] = pout[i].r * scale;
              pOut[2 * i + 1] = -pout[i].i * scale;
            }
          }
        } else {
          final pOut = result.pointer.cast<ffi.Float>() + destStart * 2;
          if (!isIhfft) {
            for (var i = 0; i < outLen; i++) {
              pOut[2 * i] = pout[i].r;
              pOut[2 * i + 1] = pout[i].i;
            }
          } else {
            for (var i = 0; i < outLen; i++) {
              pOut[2 * i] = pout[i].r * scale;
              pOut[2 * i + 1] = -pout[i].i * scale;
            }
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
/// - It is an error if [out] is provided and has incompatible shape ([n] along [axis]) or incompatible dtype (`float32` if input is `complex64` or `float32`, `float64` otherwise).
///
/// **Throws:**
/// - [ArgumentError] if the input array shape is empty (scalar 0D).
/// - [RangeError] if the specified [axis] is out of bounds.
/// - [ArgumentError] if [n] is provided but is less than or equal to 0.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
/// - [StateError] if native FFI memory allocations fail.
///
/// **Performance considerations:**
/// - Both even and odd [n] use the native C++ PocketFFT complex-to-real pathway via `kiss_fftri`.
///
/// Reference: [Inverse Real 1D FFT](https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html)
NDArray<R> irfft<R extends DTypeTag>(
  NDArray<RealFloatOf<R>> a, {
  int? n,
  int axis = -1,
  NDArray<R>? out,
}) => _irfftImpl<R>(a, n: n, axis: axis, out: out, isHfft: false);

/// Computes the FFT of a signal that has Hermitian symmetry (real spectrum).
///
/// Computes the one-dimensional discrete Fourier Transform of a Hermitian-symmetric
/// input signal [a], producing a real-valued spectrum of length [n] along [axis].
/// If [n] is not provided, it defaults to `2 * (a.shape[axis] - 1)`.
/// This is the exact inverse of [ihfft].
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if the input array [a] shape is empty (scalar 0D, rank < 1).
/// - It is an error if the specified [axis] is out of bounds `[-a.rank, a.rank - 1]`.
/// - It is an error if target length [n] is provided but is less than or equal to 0.
/// - It is an error if [out] is provided and has incompatible shape ([n] along [axis]) or incompatible dtype (`float32` if input is `complex64` or `float32`, `float64` otherwise).
///
/// Reference: [NumPy hfft](https://numpy.org/doc/stable/reference/generated/numpy.fft.hfft.html)
NDArray<R> hfft<R extends DTypeTag>(
  NDArray<RealFloatOf<R>> a, {
  int? n,
  int axis = -1,
  NDArray<R>? out,
}) => _irfftImpl<R>(a, n: n, axis: axis, out: out, isHfft: true);

NDArray<R> _irfftImpl<R extends DTypeTag>(
  NDArray<RealFloatOf<R>> a, {
  int? n,
  int axis = -1,
  NDArray<R>? out,
  required bool isHfft,
}) {
  if (a.isDisposed) {
    throw StateError(
      'Cannot execute ${isHfft ? 'hfft' : 'irfft'}() on a disposed array.',
    );
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write IFFT result to a disposed output array.');
  }
  if (a.shape.isEmpty) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must not be a 0-dimensional or empty scalar array',
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
    throw ArgumentError.value(n, 'n', 'Must be greater than 0');
  }

  final outShape = List<int>.from(a.shape);
  outShape[normAxis] = targetLen;

  final isFloatOrComplex = a.dtype.isFloating || a.dtype.isComplex;
  final expectedDType =
      ((a.dtype as DType<DTypeTag>) == DType.complex64 ||
          (a.dtype as DType<DTypeTag>) == DType.float32)
      ? DType.float32
      : DType.float64;
  final targetDType = out?.dtype ?? expectedDType;

  if (out != null) {
    validateOutBuffer(out);
    if (isFloatOrComplex && out.dtype != expectedDType) {
      throw ArgumentError.value(
        out.dtype,
        'out',
        'Must have dtype $expectedDType',
      );
    }
    if (!isFloatOrComplex &&
        out.dtype != DType.float32 &&
        out.dtype != DType.float64) {
      throw ArgumentError.value(
        out.dtype,
        'out',
        'Must have dtype float32 or float64',
      );
    }
    if (!listEquals(out.shape, outShape)) {
      throw ArgumentError.value(out.shape, 'out', 'Must have shape $outShape');
    }
    if (!out.isContiguous || sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = _createZeros(outShape, out.dtype) as NDArray<R>;
        _irfftImpl<R>(a, n: n, axis: axis, out: temp, isHfft: isHfft);
        temp.copy(out: out);
        return out;
      });
    }
  }

  if (a.size == 0 || lastAxisDim == 0) {
    return NDArray.scope(() {
      final zeroOut = _createZeros(outShape, targetDType) as NDArray<R>;
      if (out != null) {
        zeroOut.copy(out: out);
        return out;
      }
      zeroOut.detachToParentScope();
      return zeroOut;
    });
  }

  final targetInputLen = targetLen ~/ 2 + 1;

  if (normAxis != rank - 1) {
    final axes = List.generate(rank, (i) => i);
    axes[normAxis] = rank - 1;
    axes[rank - 1] = normAxis;

    return NDArray.scope(() {
      final transposedInput = a.transpose(axes);
      final transposedResult = _irfftImpl<R>(
        transposedInput,
        n: n,
        isHfft: isHfft,
      );
      final finalResult = transposedResult.transpose(axes);
      if (out != null) {
        finalResult.copy(out: out);
        return out;
      } else {
        final resCopy = finalResult.copy();
        resCopy.detachToParentScope();
        return resCopy;
      }
    });
  }

  return NDArray.scope(() {
    final inputA = a.isContiguous ? a : a.copy();
    final result = out ?? _createZeros(outShape, targetDType) as NDArray<R>;

    final totalElements = inputA.shape.reduce((x, y) => x * y);
    final signalsCount = totalElements ~/ lastAxisDim;

    final isZeroCopyFastPath =
        !isHfft &&
        (inputA.dtype as DType<DTypeTag>) == DType.complex128 &&
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
      }
      if (out == null) {
        result.detachToParentScope();
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
      final scaleFactor = isHfft ? 1.0 : 1.0 / targetLen;

      for (var s = 0; s < signalsCount; s++) {
        final srcStart = s * lastAxisDim;
        final destStart = s * targetLen;

        _copyComplexToDoubleCpx(inputA, srcStart, copyLen, pin);
        if (isHfft) {
          for (var i = 0; i < copyLen; i++) {
            pin[i].i = -pin[i].i;
          }
        }
        for (var i = copyLen; i < targetInputLen; i++) {
          pin[i].r = 0.0;
          pin[i].i = 0.0;
        }

        kiss_fftri(cfg, pin, pout);

        if (result.dtype == DType.float64) {
          final pOut = result.pointer.cast<ffi.Double>() + destStart;
          if (isHfft) {
            for (var i = 0; i < targetLen; i++) {
              pOut[i] = pout[i];
            }
          } else {
            for (var i = 0; i < targetLen; i++) {
              pOut[i] = pout[i] * scaleFactor;
            }
          }
        } else {
          final pOut = result.pointer.cast<ffi.Float>() + destStart;
          if (isHfft) {
            for (var i = 0; i < targetLen; i++) {
              pOut[i] = pout[i];
            }
          } else {
            for (var i = 0; i < targetLen; i++) {
              pOut[i] = pout[i] * scaleFactor;
            }
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

NDArray<R> _fftnND<T extends DTypeTag, R extends DTypeTag>(
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
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must not be 0-dimensional (cannot compute FFT on a 0-dimensional array)',
    );
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
    throw ArgumentError.value(
      axes,
      'axes',
      'Must be unique (axes must be unique)',
    );
  }

  final List<int> sResolved;
  if (s == null) {
    sResolved = axesResolved.map((ax) => a.shape[ax]).toList();
  } else {
    sResolved = s;
  }

  if (axesResolved.length != sResolved.length) {
    throw ArgumentError.value(
      [axesResolved.length, sResolved.length],
      'axes, s',
      'Must have the same length (axes and s must have the same length, got axes length ${axesResolved.length} and s length ${sResolved.length})',
    );
  }
  for (final sz in sResolved) {
    if (sz <= 0) {
      throw ArgumentError.value(
        sz,
        's',
        'Must be positive (transform size must be positive, got $sz)',
      );
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
    validateOutBuffer(out);
    if (isFloatOrComplex && out.dtype != expectedDType) {
      throw ArgumentError.value(
        out.dtype,
        'out.dtype',
        'Must be $expectedDType (provided out buffer has incompatible dtype, got ${out.dtype})',
      );
    }
    if (!isFloatOrComplex &&
        out.dtype != DType.complex64 &&
        out.dtype != DType.complex128) {
      throw ArgumentError.value(
        out.dtype,
        'out.dtype',
        'Must be complex64 or complex128 (provided out buffer has incompatible dtype, got ${out.dtype})',
      );
    }
    if (!listEquals(out.shape, outShape)) {
      throw ArgumentError.value(
        out.shape,
        'out.shape',
        'Must match expected output shape $outShape (provided out buffer has incompatible shape)',
      );
    }
    if (!out.isContiguous || sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = _createZeros(outShape, out.dtype) as NDArray<R>;
        _fftnND<DTypeTag, R>(a, s: s, axes: axes, inverse: inverse, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  if (a.size == 0 || outShape.contains(0)) {
    final zeroOut = _createZeros(outShape, targetDType) as NDArray<R>;
    if (out != null) {
      zeroOut.copy(out: out);
      zeroOut.dispose();
      return out;
    }
    return zeroOut;
  }

  return NDArray.scope(() {
    final prepA = _padOrTruncate<DTypeTag, R>(
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

    final resultTransposed =
        _createZeros(contiguousTransposed.shape, targetDType) as NDArray<R>;

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
      return result.copy().detachToParentScope();
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
/// - It is an error if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Uses `kiss_fftnd` plan which is optimized for multi-dimensional transforms.
/// - Transposes the array to bring target [axes] to the end before calling native C code, which is fast but might require a copy to make it contiguous.
///
/// Reference: [N-dimensional FFT](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftn.html)
NDArray<R> fftn<R extends DTypeTag>(
  NDArray<ComplexOf<R>> a, {
  List<int>? s,
  List<int>? axes,
  NDArray<R>? out,
}) => _fftnND<DTypeTag, R>(a, s: s, axes: axes, inverse: false, out: out);

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
NDArray<R> ifftn<R extends DTypeTag>(
  NDArray<ComplexOf<R>> a, {
  List<int>? s,
  List<int>? axes,
  NDArray<R>? out,
}) => _fftnND<DTypeTag, R>(a, s: s, axes: axes, inverse: true, out: out);

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
NDArray<R> fft2<R extends DTypeTag>(
  NDArray<ComplexOf<R>> a, {
  List<int>? s,
  List<int>? axes = const [-2, -1],
  NDArray<R>? out,
}) {
  final resolvedAxes = axes ?? const [-2, -1];
  if (resolvedAxes.length != 2) {
    throw ArgumentError.value(
      axes,
      'axes',
      'Must have length 2 (axes must have length 2, got length ${resolvedAxes.length})',
    );
  }
  return fftn<R>(a, s: s, axes: resolvedAxes, out: out);
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
NDArray<R> ifft2<R extends DTypeTag>(
  NDArray<ComplexOf<R>> a, {
  List<int>? s,
  List<int>? axes = const [-2, -1],
  NDArray<R>? out,
}) {
  final resolvedAxes = axes ?? const [-2, -1];
  if (resolvedAxes.length != 2) {
    throw ArgumentError.value(
      axes,
      'axes',
      'Must have length 2 (axes must have length 2, got length ${resolvedAxes.length})',
    );
  }
  return ifftn<R>(a, s: s, axes: resolvedAxes, out: out);
}

/// Computes the N-dimensional discrete Fourier Transform for real input.
///
/// Computes the N-dimensional discrete Fourier Transform over any number of axes
/// in an M-dimensional real array by means of the Fast Fourier Transform (FFT).
/// By default, all axes are transformed, with the real transform performed over
/// the last axis, while the remaining transforms are complex.
///
/// The length of the output along the last transformed axis is `s.last // 2 + 1`,
/// while the remaining transformed axes have lengths given by the corresponding
/// entries of [s] (or the input shape along those axes if [s] is omitted).
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] has a complex dtype.
/// - It is an error if [a] is 0-dimensional (rank < 1).
/// - It is an error if [axes] or [s] is empty, or if their lengths mismatch.
/// - It is an error if any axis index in [axes] is out of bounds `[-a.rank, a.rank - 1]` or contains duplicates.
/// - It is an error if any dimension in [s] is $\le 0$.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// Reference: [NumPy rfftn](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftn.html)
NDArray<R> rfftn<R extends DTypeTag>(
  NDArray<ComplexOf<R>> a, {
  List<int>? s,
  List<int>? axes,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute rfftn() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write FFT result to a disposed output array.');
  }
  if (a.dtype.isComplex) {
    throw ArgumentError.value(a.dtype, 'a', 'Must be a real-valued array');
  }

  final rank = a.rank;
  if (rank == 0) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must not be a 0-dimensional array',
    );
  }

  final List<int> axesResolved;
  if (axes == null) {
    if (s == null) {
      axesResolved = List.generate(rank, (i) => i);
    } else {
      if (s.isEmpty) {
        throw ArgumentError.value(s, 's', 'Must not be empty');
      }
      axesResolved = List.generate(s.length, (i) => rank - s.length + i);
    }
  } else {
    if (axes.isEmpty) {
      throw ArgumentError.value(axes, 'axes', 'Must not be empty');
    }
    axesResolved = axes.map((ax) => ax < 0 ? rank + ax : ax).toList();
  }

  for (final ax in axesResolved) {
    if (ax < 0 || ax >= rank) {
      throw RangeError.range(ax, 0, rank - 1, 'axis');
    }
  }
  if (axesResolved.toSet().length != axesResolved.length) {
    throw ArgumentError.value(axes, 'axes', 'Must contain unique axes');
  }

  final List<int> sResolved;
  if (s == null) {
    sResolved = axesResolved.map((ax) => a.shape[ax]).toList();
  } else {
    sResolved = s;
  }

  if (axesResolved.length != sResolved.length) {
    throw ArgumentError.value(
      s,
      's',
      'Must have the same length as axes (${axesResolved.length})',
    );
  }
  for (final sz in sResolved) {
    if (sz <= 0) {
      throw ArgumentError.value(
        s,
        's',
        'Must contain positive transform sizes',
      );
    }
  }

  final isFloat = a.dtype.isFloating;
  final expectedDType = (a.dtype as DType<DTypeTag>) == DType.float32
      ? DType.complex64
      : DType.complex128;
  final targetDType = out?.dtype ?? expectedDType;

  final outShape = List<int>.from(a.shape);
  for (var i = 0; i < axesResolved.length - 1; i++) {
    outShape[axesResolved[i]] = sResolved[i];
  }
  outShape[axesResolved.last] = sResolved.last ~/ 2 + 1;

  if (out != null) {
    validateOutBuffer(out);
    if (isFloat && out.dtype != expectedDType) {
      throw ArgumentError.value(
        out.dtype,
        'out',
        'Must have dtype $expectedDType',
      );
    }
    if (!isFloat &&
        out.dtype != DType.complex64 &&
        out.dtype != DType.complex128) {
      throw ArgumentError.value(
        out.dtype,
        'out',
        'Must have dtype complex64 or complex128',
      );
    }
    if (!listEquals(out.shape, outShape)) {
      throw ArgumentError.value(out.shape, 'out', 'Must have shape $outShape');
    }
    if (!out.isContiguous || sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = _createZeros(outShape, out.dtype) as NDArray<R>;
        rfftn<R>(a, s: s, axes: axes, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  if (a.size == 0 || outShape.contains(0)) {
    return NDArray.scope(() {
      final zeroOut = _createZeros(outShape, targetDType) as NDArray<R>;
      if (out != null) {
        zeroOut.copy(out: out);
        return out;
      }
      zeroOut.detachToParentScope();
      return zeroOut;
    });
  }

  if (axesResolved.length == 1) {
    return rfft<R>(a, n: sResolved.single, axis: axesResolved.single, out: out);
  }

  return NDArray.scope(() {
    final NDArray<R> rfftLast;
    if (!isFloat && targetDType == DType.complex64) {
      final tmpLastShape = List<int>.from(a.shape);
      tmpLastShape[axesResolved.last] = sResolved.last ~/ 2 + 1;
      final tmpLast = _createZeros(tmpLastShape, targetDType) as NDArray<R>;
      rfftLast = rfft<R>(
        a,
        n: sResolved.last,
        axis: axesResolved.last,
        out: tmpLast,
      );
    } else {
      rfftLast = rfft<R>(a, n: sResolved.last, axis: axesResolved.last);
    }
    final result = _fftnND<R, R>(
      rfftLast,
      s: sResolved.sublist(0, sResolved.length - 1),
      axes: axesResolved.sublist(0, axesResolved.length - 1),
      inverse: false,
      out: out,
    );
    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}

/// Computes the inverse of [rfftn].
///
/// Computes the inverse of the N-dimensional discrete Fourier Transform for
/// real input over any number of axes in an M-dimensional array by means of
/// the Fast Fourier Transform (FFT).
///
/// By default, all axes are inverted, with the complex-to-real inverse transform
/// performed over the last axis (defaulting to length `2 * (a.shape[axes.last] - 1)`
/// if [s] is not provided).
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] is 0-dimensional (rank < 1).
/// - It is an error if [axes] or [s] is empty, or if their lengths mismatch.
/// - It is an error if any axis index in [axes] is out of bounds `[-a.rank, a.rank - 1]` or contains duplicates.
/// - It is an error if any dimension in [s] (or inferred last-axis length) is $\le 0$.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// Reference: [NumPy irfftn](https://numpy.org/doc/stable/reference/generated/numpy.fft.irfftn.html)
NDArray<R> irfftn<R extends DTypeTag>(
  NDArray<RealFloatOf<R>> a, {
  List<int>? s,
  List<int>? axes,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute irfftn() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write IFFT result to a disposed output array.');
  }

  final rank = a.rank;
  if (rank == 0) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must not be a 0-dimensional array',
    );
  }

  final List<int> axesResolved;
  if (axes == null) {
    if (s == null) {
      axesResolved = List.generate(rank, (i) => i);
    } else {
      if (s.isEmpty) {
        throw ArgumentError.value(s, 's', 'Must not be empty');
      }
      axesResolved = List.generate(s.length, (i) => rank - s.length + i);
    }
  } else {
    if (axes.isEmpty) {
      throw ArgumentError.value(axes, 'axes', 'Must not be empty');
    }
    axesResolved = axes.map((ax) => ax < 0 ? rank + ax : ax).toList();
  }

  for (final ax in axesResolved) {
    if (ax < 0 || ax >= rank) {
      throw RangeError.range(ax, 0, rank - 1, 'axis');
    }
  }
  if (axesResolved.toSet().length != axesResolved.length) {
    throw ArgumentError.value(axes, 'axes', 'Must contain unique axes');
  }

  final List<int> sResolved;
  if (s == null) {
    sResolved = [
      for (var i = 0; i < axesResolved.length - 1; i++)
        a.shape[axesResolved[i]],
      2 * (a.shape[axesResolved.last] - 1),
    ];
  } else {
    sResolved = s;
  }

  if (axesResolved.length != sResolved.length) {
    throw ArgumentError.value(
      s,
      's',
      'Must have the same length as axes (${axesResolved.length})',
    );
  }
  for (final sz in sResolved) {
    if (sz <= 0) {
      throw ArgumentError.value(
        sResolved,
        's',
        'Must contain positive transform sizes',
      );
    }
  }

  final isFloatOrComplex = a.dtype.isFloating || a.dtype.isComplex;
  final expectedDType =
      ((a.dtype as DType<DTypeTag>) == DType.complex64 ||
          (a.dtype as DType<DTypeTag>) == DType.float32)
      ? DType.float32
      : DType.float64;
  final targetDType = out?.dtype ?? expectedDType;

  final outShape = List<int>.from(a.shape);
  for (var i = 0; i < axesResolved.length; i++) {
    outShape[axesResolved[i]] = sResolved[i];
  }

  if (out != null) {
    validateOutBuffer(out);
    if (isFloatOrComplex && out.dtype != expectedDType) {
      throw ArgumentError.value(
        out.dtype,
        'out',
        'Must have dtype $expectedDType',
      );
    }
    if (!isFloatOrComplex &&
        out.dtype != DType.float32 &&
        out.dtype != DType.float64) {
      throw ArgumentError.value(
        out.dtype,
        'out',
        'Must have dtype float32 or float64',
      );
    }
    if (!listEquals(out.shape, outShape)) {
      throw ArgumentError.value(out.shape, 'out', 'Must have shape $outShape');
    }
    if (!out.isContiguous || sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = _createZeros(outShape, out.dtype) as NDArray<R>;
        irfftn<R>(a, s: s, axes: axes, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  if (a.size == 0 || outShape.contains(0)) {
    return NDArray.scope(() {
      final zeroOut = _createZeros(outShape, targetDType) as NDArray<R>;
      if (out != null) {
        zeroOut.copy(out: out);
        return out;
      }
      zeroOut.detachToParentScope();
      return zeroOut;
    });
  }

  if (axesResolved.length == 1) {
    return irfft<R>(
      a,
      n: sResolved.single,
      axis: axesResolved.single,
      out: out,
    );
  }

  return NDArray.scope(() {
    final leadingAxes = axesResolved.sublist(0, axesResolved.length - 1);
    final leadingS = sResolved.sublist(0, sResolved.length - 1);
    final complexDType = targetDType == DType.float32
        ? DType.complex64
        : DType.complex128;
    final tmpShape = List<int>.from(a.shape);
    for (var i = 0; i < leadingAxes.length; i++) {
      tmpShape[leadingAxes[i]] = leadingS[i];
    }
    final tmpComplex = _createZeros(tmpShape, complexDType);
    _fftnND<DTypeTag, DTypeTag>(
      a,
      s: leadingS,
      axes: leadingAxes,
      inverse: true,
      out: tmpComplex,
    );
    final result = irfft<R>(
      tmpComplex as NDArray<RealFloatOf<R>>,
      n: sResolved.last,
      axis: axesResolved.last,
      out: out,
    );
    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}

/// Computes the 2-dimensional discrete Fourier Transform for real input.
///
/// Equivalent to calling [rfftn] with [axes] defaulting to the last two axes `[-2, -1]`.
///
/// **Preconditions:**
/// - It is an error if input array [a] or [out] is disposed.
/// - It is an error if [a] has a complex dtype.
/// - It is an error if input array [a] has rank < 2.
/// - It is an error if [axes] is provided and does not have length 2.
/// - It is an error if [s] is provided and does not have length 2.
/// - It is an error if any axis index in [axes] is out of bounds `[-a.rank, a.rank - 1]` or contains duplicates.
/// - It is an error if any dimension in [s] is $\le 0$.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// Reference: [NumPy rfft2](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft2.html)
NDArray<R> rfft2<R extends DTypeTag>(
  NDArray<ComplexOf<R>> a, {
  List<int>? s,
  List<int>? axes = const [-2, -1],
  NDArray<R>? out,
}) {
  final resolvedAxes = axes ?? const [-2, -1];
  if (resolvedAxes.length != 2) {
    throw ArgumentError.value(axes, 'axes', 'Must have length 2');
  }
  return rfftn<R>(a, s: s, axes: resolvedAxes, out: out);
}

/// Computes the 2-dimensional inverse discrete Fourier Transform for real input.
///
/// Equivalent to calling [irfftn] with [axes] defaulting to the last two axes `[-2, -1]`.
///
/// **Preconditions:**
/// - It is an error if input array [a] or [out] is disposed.
/// - It is an error if input array [a] has rank < 2.
/// - It is an error if [axes] is provided and does not have length 2.
/// - It is an error if [s] is provided and does not have length 2.
/// - It is an error if any axis index in [axes] is out of bounds `[-a.rank, a.rank - 1]` or contains duplicates.
/// - It is an error if any dimension in [s] is $\le 0$.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// Reference: [NumPy irfft2](https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft2.html)
NDArray<R> irfft2<R extends DTypeTag>(
  NDArray<RealFloatOf<R>> a, {
  List<int>? s,
  List<int>? axes = const [-2, -1],
  NDArray<R>? out,
}) {
  final resolvedAxes = axes ?? const [-2, -1];
  if (resolvedAxes.length != 2) {
    throw ArgumentError.value(axes, 'axes', 'Must have length 2');
  }
  return irfftn<R>(a, s: s, axes: resolvedAxes, out: out);
}

/// Clears all precomputed native FFT plans from the isolate plan cache.
///
/// Frees unmanaged C heap memory allocated for twiddle factor tables and
/// multidimensional transform buffers across 1D complex, 1D real, and
/// N-dimensional FFT plans.
void clearFFTPlanCache() {
  PocketFFTPlanCache.instance.clear();
}
