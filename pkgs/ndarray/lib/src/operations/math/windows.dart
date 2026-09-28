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

import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../helpers.dart';

/// Returns the Hanning (Hann) window.
///
/// The Hanning window is a taper formed by using a weighted cosine:
///
/// $$w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{M - 1}\right), \quad 0 \le n \le M-1$$
///
/// Unlike the Hamming window, the Hanning window tapers all the way to exactly
/// **zero** at the boundaries ($w[0] = w[M-1] = 0.0$). It features a fast side-lobe
/// roll-off rate of $18 \text{ dB/octave}$, making it highly suitable for general
/// spectral analysis where suppression of distant side lobes is critical.
///
/// **Preconditions:**
/// - If [out] is provided, it must not be disposed and must have shape `[M]` (or `[0]` if `M < 1`)
///   and matching [DType].
///
/// - It is an error if [out] is disposed (throws [StateError]).
/// - It is an error if [out] has incompatible shape or [DType] (throws [ArgumentError]).
///
/// **Example:**
/// ```dart
/// final window = hanning(512);
/// ```
NDArray<T> hanning<T extends DTypeTag>(
  int M, {
  DType<T>? dtype,
  NDArray<T>? out,
}) {
  if (out != null && out.isDisposed) {
    throw StateError('Cannot execute hanning() on a disposed out buffer.');
  }
  final resolvedDType = dtype ?? (out?.dtype ?? (DType.float64 as DType<T>));
  final targetShape = [M < 1 ? 0 : M];
  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != resolvedDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for hanning.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(targetShape, resolvedDType);
  }

  if (M < 1) return result;
  if (M == 1) {
    result.fill(castValue(1.0, resolvedDType));
    return result;
  }

  switch (resolvedDType) {
    case DType.float32:
      if (result.isContiguous) {
        v_hanning_float(result.pointer.cast(), M);
      } else {
        final temp = NDArray<Float32>.create([M], DType.float32);
        try {
          v_hanning_float(temp.pointer.cast(), M);
          temp.copy(out: result as NDArray<Float32>);
        } finally {
          temp.dispose();
        }
      }
      return result;
    case DType.float64:
      if (result.isContiguous) {
        v_hanning_double(result.pointer.cast(), M);
      } else {
        final temp = NDArray<Float64>.create([M], DType.float64);
        try {
          v_hanning_double(temp.pointer.cast(), M);
          temp.copy(out: result as NDArray<Float64>);
        } finally {
          temp.dispose();
        }
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
    case DType.boolean:
    case DType.complex128:
    case DType.complex64:
      final temp = NDArray<Float64>.create([M], DType.float64);
      try {
        v_hanning_double(temp.pointer.cast(), M);
        final casted = castNDArray(temp, resolvedDType);
        try {
          casted.copy(out: result);
        } finally {
          casted.dispose();
        }
      } finally {
        temp.dispose();
      }
      return result;
  }
}

/// Returns the Hamming window.
///
/// The Hamming window is a taper formed by using an optimized weighted cosine:
///
/// $$w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{M - 1}\right), \quad 0 \le n \le M-1$$
///
/// Unlike the Hanning window, the Hamming window does not taper to zero at the boundaries,
/// leaving a small pedestal/discontinuity ($w[0] = w[M-1] = 0.08$). It is optimized to
/// minimize the maximum side-lobe level (achieving a first side lobe of $-43 \text{ dB}$
/// compared to Hanning's $-32 \text{ dB}$), at the expense of a slower side-lobe roll-off
/// rate of $6 \text{ dB/octave}$.
///
/// **Preconditions:**
/// - If [out] is provided, it must not be disposed and must have shape `[M]` (or `[0]` if `M < 1`)
///   and matching [DType].
///
/// - It is an error if [out] is disposed (throws [StateError]).
/// - It is an error if [out] has incompatible shape or [DType] (throws [ArgumentError]).
///
/// **Example:**
/// ```dart
/// final window = hamming(512);
/// ```
NDArray<T> hamming<T extends DTypeTag>(
  int M, {
  DType<T>? dtype,
  NDArray<T>? out,
}) {
  if (out != null && out.isDisposed) {
    throw StateError('Cannot execute hamming() on a disposed out buffer.');
  }
  final resolvedDType = dtype ?? (out?.dtype ?? (DType.float64 as DType<T>));
  final targetShape = [M < 1 ? 0 : M];
  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != resolvedDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for hamming.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(targetShape, resolvedDType);
  }

  if (M < 1) return result;
  if (M == 1) {
    result.fill(castValue(1.0, resolvedDType));
    return result;
  }

  switch (resolvedDType) {
    case DType.float32:
      if (result.isContiguous) {
        v_hamming_float(result.pointer.cast(), M);
      } else {
        final temp = NDArray<Float32>.create([M], DType.float32);
        try {
          v_hamming_float(temp.pointer.cast(), M);
          temp.copy(out: result as NDArray<Float32>);
        } finally {
          temp.dispose();
        }
      }
      return result;
    case DType.float64:
      if (result.isContiguous) {
        v_hamming_double(result.pointer.cast(), M);
      } else {
        final temp = NDArray<Float64>.create([M], DType.float64);
        try {
          v_hamming_double(temp.pointer.cast(), M);
          temp.copy(out: result as NDArray<Float64>);
        } finally {
          temp.dispose();
        }
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
    case DType.boolean:
    case DType.complex128:
    case DType.complex64:
      final temp = NDArray<Float64>.create([M], DType.float64);
      try {
        v_hamming_double(temp.pointer.cast(), M);
        final casted = castNDArray(temp, resolvedDType);
        try {
          casted.copy(out: result);
        } finally {
          casted.dispose();
        }
      } finally {
        temp.dispose();
      }
      return result;
  }
}
