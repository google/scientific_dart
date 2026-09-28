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

import "dart:ffi" as ffi;

// ignore_for_file: non_constant_identifier_names
import "../../ndarray.dart";
import "../../ndarray_bindings.dart";
import "../../scratch_arena.dart";
import "../helpers.dart";

/// Returns the real part of a complex array element-wise.
///
/// If the input array [a] is already real (integer or float), returns a zero-copy
/// view of the array [a] (when no [out] or [where] mask is provided).
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - If provided, the output recycler [out] must match the expected target shape and float [DType].
///
/// It is an error if the array has been disposed (throws [StateError]), or if
/// [out] is provided with incompatible shape or dtype (throws [ArgumentError]).
///
/// **Example:**
/// ```dart
/// final a = NDArray<Complex>.create([2], DType.complex128);
/// a.setCell([0], Complex(3.0, 4.0));
/// a.setCell([1], Complex(-1.0, 0.0));
/// final r = real(a); // [3.0, -1.0] (DType.float64)
/// ```
NDArray<R> real<T, R>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError("Cannot execute real() on a disposed array.");
  }

  final DType<dynamic> targetDType;
  switch (a.dtype) {
    case DType.complex64:
      targetDType = DType.float32;
    case DType.complex128:
      targetDType = DType.float64;
    default:
      targetDType = a.dtype;
  }

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype for real.",
      );
    }
    result = out;
  } else {
    if (where == null &&
        a.dtype != DType.complex128 &&
        a.dtype != DType.complex64) {
      return NDArray.view(a, shape: a.shape, strides: a.strides)
          as NDArray<R>; // Zero-copy view for already real arrays!
    }
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  final maskHolder = prepareMask(where, result.shape);
  try {
    switch (a.dtype) {
      case DType.complex128:
      case DType.complex64:
        final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
        final coord = List<int>.filled(a.shape.length, 0);
        final maskPtr = maskHolder.pointer;

        for (var i = 0; i < size; i++) {
          if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
            final c = a.getCell(coord) as Complex;
            result.setCell(coord, c.real as R);
          }
          for (var d = a.shape.length - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < a.shape[d]) break;
            coord[d] = 0;
          }
        }
        return result;
      default:
        // This path is taken if out != null or where != null and a is not complex.
        if (where == null) {
          a.copy(out: result as dynamic);
        } else {
          final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
          final coord = List<int>.filled(a.shape.length, 0);
          final maskPtr = maskHolder.pointer;

          for (var i = 0; i < size; i++) {
            if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
              result.setCell(coord, a.getCell(coord) as R);
            }
            for (var d = a.shape.length - 1; d >= 0; d--) {
              coord[d]++;
              if (coord[d] < a.shape[d]) break;
              coord[d] = 0;
            }
          }
        }
        return result;
    }
  } finally {
    maskHolder.dispose();
  }
}

/// Returns the imaginary part of a complex array element-wise.
///
/// If the input array [a] is already real, returns a zero-filled array of matching shape
/// and target float [DType].
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - If provided, the output recycler [out] must match the expected target shape and float [DType].
///
/// It is an error if the array has been disposed (throws [StateError]), or if
/// [out] is provided with incompatible shape or dtype (throws [ArgumentError]).
///
/// **Example:**
/// ```dart
/// final a = NDArray<Complex>.create([2], DType.complex128);
/// a.setCell([0], Complex(3.0, 4.0));
/// a.setCell([1], Complex(-1.0, 0.0));
/// final im = imag(a); // [4.0, 0.0] (DType.float64)
/// ```
NDArray<R> imag<T, R>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError("Cannot execute imag() on a disposed array.");
  }

  final DType<dynamic> targetDType = a.dtype == DType.complex64
      ? DType.float32
      : DType.float64;

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype for imag.",
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  final maskHolder = prepareMask(where, result.shape);
  try {
    if (a.dtype != DType.complex128 && a.dtype != DType.complex64) {
      if (where == null) {
        result.fill(0.0 as R);
      } else {
        final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
        final coord = List<int>.filled(a.shape.length, 0);
        final maskPtr = maskHolder.pointer;
        final zeroVal = 0.0 as R;

        for (var i = 0; i < size; i++) {
          if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
            result.setCell(coord, zeroVal);
          }
          for (var d = a.shape.length - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < a.shape[d]) break;
            coord[d] = 0;
          }
        }
      }
      return result;
    }

    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final coord = List<int>.filled(a.shape.length, 0);
    final maskPtr = maskHolder.pointer;

    for (var i = 0; i < size; i++) {
      if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
        final c = a.getCell(coord) as Complex;
        result.setCell(coord, c.imag as R);
      }
      for (var d = a.shape.length - 1; d >= 0; d--) {
        coord[d]++;
        if (coord[d] < a.shape[d]) break;
        coord[d] = 0;
      }
    }

    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise complex conjugate of the array elements.
///
/// **Preconditions:**
/// - The array must not be disposed.
///
/// It is an error if the array has been disposed (throws [StateError]), or if
/// [out] is provided with incompatible shape or dtype (throws [ArgumentError]).
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([Complex(1.0, 2.0)], [1], DType.complex128);
/// final c = conj(a); // [Complex(1.0, -2.0)]
/// ```
NDArray<T> conj<T>(NDArray<T> a, {NDArray<dynamic>? where, NDArray<T>? out}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError("Cannot execute conj() on a disposed array.");
  }
  final targetDType = a.dtype;
  final result = out ?? NDArray<T>.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype for conj.",
      );
    }
  }

  final maskHolder = prepareMask(where, result.shape);
  try {
    switch (targetDType) {
      case DType.complex128:
        if (a.isContiguous && result.isContiguous) {
          v_conj_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        } else {
          final rank = a.shape.length;
          final marker = ScratchArena.marker;
          final cShape = ScratchArena.copyInts(a.shape);
          final cStridesA = ScratchArena.copyInts(a.strides);
          final cStridesRes = ScratchArena.copyInts(result.strides);
          try {
            s_conj_complex128(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          } finally {
            ScratchArena.reset(marker);
          }
        }
      case DType.complex64:
        if (a.isContiguous && result.isContiguous) {
          v_conj_complex64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        } else {
          final rank = a.shape.length;
          final marker = ScratchArena.marker;
          final cShape = ScratchArena.copyInts(a.shape);
          final cStridesA = ScratchArena.copyInts(a.strides);
          final cStridesRes = ScratchArena.copyInts(result.strides);
          try {
            s_conj_complex64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          } finally {
            ScratchArena.reset(marker);
          }
        }
      default:
        // Real/boolean numbers are their own complex conjugates!
        if (where == null) {
          a.copy(out: result);
        } else {
          final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
          final coord = List<int>.filled(a.shape.length, 0);
          final maskPtr = maskHolder.pointer;

          for (var i = 0; i < size; i++) {
            if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
              result.setCell(coord, a.getCell(coord));
            }
            for (var d = a.shape.length - 1; d >= 0; d--) {
              coord[d]++;
              if (coord[d] < a.shape[d]) break;
              coord[d] = 0;
            }
          }
        }
        return result;
    }
  } finally {
    maskHolder.dispose();
  }
}

/// Alias for [conj].
NDArray<T> conjugate<T>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
  NDArray<T>? out,
}) => conj(a, where: where, out: out);
