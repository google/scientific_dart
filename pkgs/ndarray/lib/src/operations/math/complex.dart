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
import "../../ndarray.dart";
import "../../ndarray_bindings.dart";
import "../../scratch_arena.dart";
import "../helpers.dart";

NDArray<DTypeTag> _complexPartView(
  NDArray<DTypeTag> a,
  DType<DTypeTag> floatDType, {
  required bool isImag,
}) {
  final floatStrides = a.strides.map((s) => s * 2).toList();
  var minRelativeOffset = 0;
  if (!a.shape.contains(0)) {
    for (var d = 0; d < a.shape.length; d++) {
      final s = floatStrides[d];
      if (s < 0) {
        minRelativeOffset += (a.shape[d] - 1) * s;
      }
    }
  }
  final elementOffset = (isImag ? 1 : 0) + minRelativeOffset;
  final ffi.Pointer<ffi.Void> basePtr = floatDType == DType.float64
      ? (a.pointer.cast<ffi.Double>() + elementOffset).cast<ffi.Void>()
      : (a.pointer.cast<ffi.Float>() + elementOffset).cast<ffi.Void>();
  return NDArray<DTypeTag>.fromPointer(
    basePtr,
    a.shape,
    floatDType,
    strides: floatStrides,
  );
}

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
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<R> real<R extends DTypeTag>(
  NDArray<
    DTypeSpec<R, Object?, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>
  >
  a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError("Cannot execute real() on a disposed array.");
  }

  final DType<DTypeTag> targetDType;
  switch (a.dtype) {
    case DType.complex64:
      targetDType = DType.float32;
    case DType.complex128:
      targetDType = DType.float64;
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
      targetDType = a.dtype;
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for real',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(a.shape, out.dtype);
        real<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  } else if (where == null &&
      (a.dtype as DType<DTypeTag>) != DType.complex128 &&
      (a.dtype as DType<DTypeTag>) != DType.complex64) {
    return NDArray<R>.view(
      a,
      shape: a.shape,
      strides: a.strides,
    ); // Zero-copy view for already real arrays!
  }

  if (where == null &&
      ((a.dtype as DType<DTypeTag>) == DType.complex128 ||
          (a.dtype as DType<DTypeTag>) == DType.complex64)) {
    final NDArray<R> result =
        out ?? NDArray<R>.create(a.shape, targetDType as DType<R>);
    final view = _complexPartView(a, targetDType, isImag: false);
    try {
      view.copy(out: result);
    } finally {
      view.dispose();
    }
    return result;
  }

  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<R> result =
        out ??
        NDArray<R>.create(
          a.shape,
          targetDType as DType<R>,
          zeroInit: where != null,
        );
    switch (a.dtype) {
      case DType.complex128:
      case DType.complex64:
        final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
        final coord = List<int>.filled(a.shape.length, 0);
        final maskPtr = maskHolder.pointer;

        for (var i = 0; i < size; i++) {
          if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
            final c = a.getCell(coord) as Complex;
            result.setCell(coord, c.real);
          }
          for (var d = a.shape.length - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < a.shape[d]) break;
            coord[d] = 0;
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
      case DType.boolean:
        // This path is taken if out != null or where != null and a is not complex.
        if (where == null) {
          (a as NDArray<DTypeTag>).copy(out: result);
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
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<R> imag<R extends DTypeTag>(
  NDArray<
    DTypeSpec<DTypeTag, Object?, R, DTypeTag, DTypeTag, DTypeTag, DTypeTag>
  >
  a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError("Cannot execute imag() on a disposed array.");
  }

  final DType<DTypeTag> targetDType = switch (a.dtype) {
    DType.complex64 || DType.float32 => DType.float32,
    _ => DType.float64,
  };

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for imag',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(a.shape, out.dtype);
        imag<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  if (where == null &&
      ((a.dtype as DType<DTypeTag>) == DType.complex128 ||
          (a.dtype as DType<DTypeTag>) == DType.complex64)) {
    final NDArray<R> result =
        out ?? NDArray<R>.create(a.shape, targetDType as DType<R>);
    final view = _complexPartView(a, targetDType, isImag: true);
    try {
      view.copy(out: result);
    } finally {
      view.dispose();
    }
    return result;
  }

  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<R> result =
        out ??
        NDArray<R>.create(
          a.shape,
          targetDType as DType<R>,
          zeroInit: where != null,
        );
    if ((a.dtype as DType<DTypeTag>) != DType.complex128 &&
        (a.dtype as DType<DTypeTag>) != DType.complex64) {
      if (where == null) {
        result.fill(0.0);
      } else {
        final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
        final coord = List<int>.filled(a.shape.length, 0);
        final maskPtr = maskHolder.pointer;
        final zeroVal = 0.0;

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
        result.setCell(coord, c.imag);
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
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<T> conj<T extends DTypeTag>(
  NDArray<T> a, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError("Cannot execute conj() on a disposed array.");
  }
  final targetDType = a.dtype;
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for conj',
      );
    }
    if ((sharesMemory(a, out) &&
            (!a.isContiguous ||
                !out.isContiguous ||
                a.offsetElements != out.offsetElements ||
                !listEquals(a.strides, out.strides))) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(a.shape, targetDType);
        conj<T>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  final maskHolder = prepareMask(where, a.shape);
  try {
    final result =
        out ?? NDArray<T>.create(a.shape, targetDType, zeroInit: where != null);
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
          try {
            final cShape = ScratchArena.copyInts(a.shape);
            final cStridesA = ScratchArena.copyInts(a.strides);
            final cStridesRes = ScratchArena.copyInts(result.strides);
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
          try {
            final cShape = ScratchArena.copyInts(a.shape);
            final cStridesA = ScratchArena.copyInts(a.strides);
            final cStridesRes = ScratchArena.copyInts(result.strides);
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
NDArray<T> conjugate<T extends DTypeTag>(
  NDArray<T> a, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) => conj(a, where: where, out: out);
