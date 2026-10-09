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

import 'dart:ffi' as ffi;
import '../ndarray.dart';
import '../scratch_arena.dart';
import '../ndarray_bindings.dart' as bindings;
import '../ndarray_extensions_bindings.dart' as ext_bindings;
import '../float16_utils.dart';
import 'helpers.dart';

/// Supported padding modes.
enum PadMode {
  /// Pads with a constant value.
  constant,

  /// Pads with the edge values of the array.
  edge,

  /// Pads with the reflection of the vector mirrored on the first and last
  /// values of the vector along each axis.
  reflect,

  /// Pads with the reflection of the vector mirrored along the edge of the
  /// array.
  symmetric,

  /// Pads with the wrap of the vector along the axis.
  wrap,

  /// Pads with the linear ramp between end value and the array edge value.
  linearRamp,

  /// Pads with the mean value of all or part of the vector along each axis.
  mean,

  /// Pads with the median value of all or part of the vector along each axis.
  median,

  /// Pads with the minimum value of all or part of the vector along each axis.
  min,

  /// Pads with the maximum value of all or part of the vector along each axis.
  max;

  /// Alias for [min].
  static const PadMode minimum = PadMode.min;

  /// Alias for [max].
  static const PadMode maximum = PadMode.max;
}

/// Backwards compatibility alias for [PadMode].
typedef PaddingMode = PadMode;

/// Represents padding widths before and after for each axis.
final class PadWidth {
  final (int before, int after)? _uniform;
  final List<(int before, int after)>? _axes;

  const PadWidth._(this._uniform, this._axes);

  /// Creates a uniform [PadWidth] for all axes.
  ///
  /// If [after] is not specified, it defaults to [before].
  ///
  /// **Preconditions:**
  /// - [before] must be non-negative.
  /// - [after] must be non-negative if specified.
  ///
  /// **Throws:**
  /// - It is an error if [before] or [after] is negative.
  factory PadWidth.all(int before, [int? after]) {
    RangeError.checkNotNegative(before, 'before');
    if (after != null) {
      RangeError.checkNotNegative(after, 'after');
    }
    return PadWidth._((before, after ?? before), null);
  }

  /// Creates a [PadWidth] with specific widths for each axis.
  ///
  /// **Preconditions:**
  /// - All before and after widths must be non-negative.
  ///
  /// **Throws:**
  /// - It is an error if any width in [widths] is negative.
  factory PadWidth.axes(List<(int before, int after)> widths) {
    for (final (before, after) in widths) {
      RangeError.checkNotNegative(before, 'before');
      RangeError.checkNotNegative(after, 'after');
    }
    return PadWidth._(null, List.unmodifiable(widths));
  }

  /// Normalizes the pad widths to a list of length [rank].
  List<(int before, int after)> normalize(int rank) {
    final uniform = _uniform;
    if (uniform != null) {
      return List.filled(rank, uniform);
    }
    final axes = _axes;
    if (axes != null) {
      if (axes.length != rank) {
        throw ArgumentError.value(
          axes,
          'padWidth',
          'Length of padding widths (${axes.length}) must match array rank ($rank)',
        );
      }
      return axes;
    }
    throw StateError('Invalid PadWidth state');
  }
}

/// Represents constant padding values for each axis.
///
/// The type parameter [T] is the [DTypeTag] of the array these values are
/// meant for; it ties a `PadValues` to a matching [NDArray] at the [pad] call
/// site. The values themselves are untyped ([Object?]) because a tag does not
/// name its element type; they must match the array's dtype (`double` for the
/// float dtypes, `int` for the integer dtypes, [Complex] for the complex
/// dtypes and `bool` for [DType.boolean]).
final class PadValues<T extends DTypeTag> {
  final (Object? before, Object? after)? _uniform;
  final List<(Object? before, Object? after)>? _axes;

  const PadValues._(this._uniform, this._axes);

  /// Creates a uniform [PadValues] for all axes.
  ///
  /// If [after] is not specified, it defaults to [before].
  factory PadValues.all(Object? before, [Object? after]) {
    return PadValues._((before, after ?? before), null);
  }

  /// Creates a [PadValues] with specific values for each axis.
  factory PadValues.axes(List<(Object? before, Object? after)> values) {
    return PadValues._(null, List.unmodifiable(values));
  }

  /// Normalizes the pad values to a list of length [rank].
  List<(Object? before, Object? after)> normalize(
    int rank,
    Object? defaultValue,
  ) {
    final uniform = _uniform;
    if (uniform != null) {
      return List.filled(rank, uniform);
    }
    final axes = _axes;
    if (axes != null) {
      if (axes.length != rank) {
        throw ArgumentError.value(
          axes,
          'constantValues',
          'Length of padding values (${axes.length}) must match array rank ($rank)',
        );
      }
      return axes;
    }
    return List.filled(rank, (defaultValue, defaultValue));
  }
}

/// Represents the window length for statistical padding modes.
final class StatLength {
  final (int before, int after)? _uniform;
  final List<(int before, int after)>? _axes;

  const StatLength._(this._uniform, this._axes);

  /// Creates a uniform [StatLength] for all axes.
  ///
  /// If [after] is not specified, it defaults to [before].
  ///
  /// **Preconditions:**
  /// - [before] must be positive.
  /// - [after] must be positive if specified.
  ///
  /// **Throws:**
  /// - It is an error if [before] or [after] is not positive.
  factory StatLength.all(int before, [int? after]) {
    if (before <= 0) {
      throw ArgumentError.value(before, 'before', 'Must be positive');
    }
    if (after != null && after <= 0) {
      throw ArgumentError.value(after, 'after', 'Must be positive');
    }
    return StatLength._((before, after ?? before), null);
  }

  /// Creates a [StatLength] with specific lengths for each axis.
  ///
  /// **Preconditions:**
  /// - All before and after lengths must be positive.
  ///
  /// **Throws:**
  /// - It is an error if any length in [lengths] is not positive.
  factory StatLength.axes(List<(int before, int after)> lengths) {
    for (final (before, after) in lengths) {
      if (before <= 0) {
        throw ArgumentError.value(before, 'before', 'Must be positive');
      }
      if (after <= 0) {
        throw ArgumentError.value(after, 'after', 'Must be positive');
      }
    }
    return StatLength._(null, List.unmodifiable(lengths));
  }

  /// Normalizes the stat lengths to a list of length [rank], capping them by the [shape].
  List<(int before, int after)> normalize(List<int> shape) {
    final rank = shape.length;
    final uniform = _uniform;
    if (uniform != null) {
      return List.generate(rank, (i) {
        if (shape[i] == 0) return (0, 0);
        final b = uniform.$1.clamp(1, shape[i]);
        final a = uniform.$2.clamp(1, shape[i]);
        return (b, a);
      });
    }
    final axes = _axes;
    if (axes != null) {
      if (axes.length != rank) {
        throw ArgumentError.value(
          axes,
          'statLength',
          'Length of stat lengths (${axes.length}) must match array rank ($rank)',
        );
      }
      return List.generate(rank, (i) {
        if (shape[i] == 0) return (0, 0);
        final b = axes[i].$1.clamp(1, shape[i]);
        final a = axes[i].$2.clamp(1, shape[i]);
        return (b, a);
      });
    }
    return List.generate(rank, (i) => (shape[i], shape[i]));
  }
}

/// Helper to get default constant value for a DType.
Object _getDefaultValue(DType dtype) {
  switch (dtype) {
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
      return 0.0;
    case DType.complex128:
    case DType.complex64:
      return Complex(0.0, 0.0);
    case DType.boolean:
      return false;
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
      return 0;
  }
}

/// Pads an N-dimensional array according to the specified [padWidth] and [mode].
///
/// This operation returns a new array with padded values along each axis, or
/// writes to [out] if provided.
///
/// **Preconditions:**
/// - The [array] must not be disposed.
/// - If [out] is provided, it must not be disposed, and its shape must match the
///   calculated padded shape.
/// - The [array] must have rank >= 1 (cannot pad 0-dimensional arrays).
/// - The length of [padWidth], [constantValues], [endValues], and [statLength]
///   must match the rank of the [array] when specified per-axis.
///
/// **Throws:**
/// - It is an error if [array] or [out] is disposed.
/// - It is an error if [array] is 0-dimensional.
/// - It is an error if parameter dimensions mismatch.
///
/// **Performance considerations:**
/// - The operation performs padding sequentially axis-by-axis.
/// - If an axis has 0 padding (both before and after), it is skipped to save
///   memory and copy overhead, unless it is the only axis or `out` is provided
///   (in which case at least one copy is performed).
/// - Statistical modes (`median`, `mean`, etc.) cache the statistics per slice
///   to avoid redundant calculations (like sorting for median) for every padded
///   element.
///
/// **Example:**
/// {@example /example/padding_example.dart lang=dart}
NDArray<T> pad<T extends DTypeTag, Out extends T>(
  NDArray<T> array,
  PadWidth padWidth, {
  PadMode mode = PadMode.constant,
  PadValues<T>? constantValues,
  PadValues<T>? endValues,
  StatLength? statLength,
  NDArray<Out>? out,
}) {
  if (array.isDisposed) {
    throw StateError('Source array is disposed.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Output array is disposed.');
  }

  final rank = array.rank;
  if (rank == 0) {
    throw ArgumentError.value(
      array,
      'array',
      'Cannot pad a 0-dimensional array',
    );
  }

  // Normalize parameters
  final normPadWidths = padWidth.normalize(rank);
  final defaultValue = _getDefaultValue(array.dtype);
  final normConstantValues = (constantValues ?? PadValues<T>.all(defaultValue))
      .normalize(rank, defaultValue);
  final normEndValues = (endValues ?? PadValues<T>.all(defaultValue)).normalize(
    rank,
    defaultValue,
  );
  final normStatLengths =
      statLength?.normalize(array.shape) ??
      List.generate(rank, (i) => (array.shape[i], array.shape[i]));

  // Calculate expected output shape
  final finalShape = List<int>.generate(rank, (i) {
    final (before, after) = normPadWidths[i];
    return array.shape[i] + before + after;
  });

  if (out != null) {
    validateOutBuffer(out);
    if (out.dtype != array.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Output array dtype (${out.dtype}) must match source array dtype (${array.dtype})',
      );
    }
    if (out.rank != rank) {
      throw ArgumentError.value(
        out,
        'out',
        'Output array rank (${out.rank}) must match expected rank ($rank)',
      );
    }
    for (var i = 0; i < rank; i++) {
      if (out.shape[i] != finalShape[i]) {
        throw ArgumentError.value(
          out,
          'out',
          'Output array shape at dim $i (${out.shape[i]}) must match expected shape (${finalShape[i]})',
        );
      }
    }
  }

  for (var i = 0; i < rank; i++) {
    final (before, after) = normPadWidths[i];
    if ((before > 0 || after > 0) &&
        array.shape[i] == 0 &&
        mode != PadMode.constant) {
      throw ArgumentError.value(
        array,
        'array',
        'Cannot pad empty axis $i of shape ${array.shape} with mode $mode',
      );
    }
  }

  var needsPadding = false;
  for (final (before, after) in normPadWidths) {
    if (before > 0 || after > 0) {
      needsPadding = true;
      break;
    }
  }

  if (!needsPadding) {
    if (out != null) {
      return array.copy(out: out);
    } else {
      return array.copy();
    }
  }

  final isFastNativeMode = switch (mode) {
    PadMode.constant ||
    PadMode.edge ||
    PadMode.reflect ||
    PadMode.symmetric ||
    PadMode.wrap => true,
    _ => false,
  };

  if (isFastNativeMode && array.size > 0) {
    return _padNativeFast<T>(
      array,
      normPadWidths,
      mode,
      normConstantValues,
      finalShape,
      out,
    );
  }

  return _padAxisByAxis<T>(
    array,
    normPadWidths,
    mode,
    normConstantValues,
    normEndValues,
    normStatLengths,
    out,
  );
}

ffi.Pointer<ffi.Int16> _copyInt16s(List<int> list) {
  final ptr = ScratchArena.allocate<ffi.Int16>(
    list.length * ffi.sizeOf<ffi.Int16>(),
  );
  for (var i = 0; i < list.length; i++) {
    ptr[i] = list[i];
  }
  return ptr;
}

ffi.Pointer<ffi.Uint32> _copyUint32s(List<int> list) {
  final ptr = ScratchArena.allocate<ffi.Uint32>(
    list.length * ffi.sizeOf<ffi.Uint32>(),
  );
  for (var i = 0; i < list.length; i++) {
    ptr[i] = list[i];
  }
  return ptr;
}

ffi.Pointer<ffi.Uint8> _copyUint8s(List<int> list) {
  final ptr = ScratchArena.allocate<ffi.Uint8>(
    list.length * ffi.sizeOf<ffi.Uint8>(),
  );
  for (var i = 0; i < list.length; i++) {
    ptr[i] = list[i];
  }
  return ptr;
}

(
  ffi.Pointer<ffi.Void> constBeforePtr,
  ffi.Pointer<ffi.Void> constAfterPtr,
  int isUniformConstant,
)
_prepareConstants(
  DType dtype,
  List<(Object? before, Object? after)> normConstantValues,
  int rank,
) {
  var isUniform = 1;
  final firstVal = normConstantValues[0].$1;
  for (var i = 0; i < rank; i++) {
    if (normConstantValues[i].$1 != firstVal ||
        normConstantValues[i].$2 != firstVal) {
      isUniform = 0;
      break;
    }
  }

  final beforeVals = [for (final v in normConstantValues) v.$1];
  final afterVals = [for (final v in normConstantValues) v.$2];

  int toIntVal(Object? e) => e is int ? e : (e as num).toInt();
  bool toBoolVal(Object? e) => e is bool ? e : (e as num) != 0;
  Complex toComplexVal(Object? e) =>
      e is Complex ? e : Complex((e as num).toDouble(), 0.0);

  final (cbPtr, caPtr) = switch (dtype) {
    DType.float64 => (
      ScratchArena.copyDoubles([
        for (final e in beforeVals) (e as num).toDouble(),
      ]).cast<ffi.Void>(),
      ScratchArena.copyDoubles([
        for (final e in afterVals) (e as num).toDouble(),
      ]).cast<ffi.Void>(),
    ),
    DType.float32 => (
      ScratchArena.copyFloats([
        for (final e in beforeVals) (e as num).toDouble(),
      ]).cast<ffi.Void>(),
      ScratchArena.copyFloats([
        for (final e in afterVals) (e as num).toDouble(),
      ]).cast<ffi.Void>(),
    ),
    DType.int64 => (
      ScratchArena.copyInt64s([
        for (final e in beforeVals) toIntVal(e),
      ]).cast<ffi.Void>(),
      ScratchArena.copyInt64s([
        for (final e in afterVals) toIntVal(e),
      ]).cast<ffi.Void>(),
    ),
    DType.int32 => (
      ScratchArena.copyInt32s([
        for (final e in beforeVals) toIntVal(e),
      ]).cast<ffi.Void>(),
      ScratchArena.copyInt32s([
        for (final e in afterVals) toIntVal(e),
      ]).cast<ffi.Void>(),
    ),
    DType.int16 => (
      _copyInt16s([for (final e in beforeVals) toIntVal(e)]).cast<ffi.Void>(),
      _copyInt16s([for (final e in afterVals) toIntVal(e)]).cast<ffi.Void>(),
    ),
    DType.uint8 => (
      _copyUint8s([for (final e in beforeVals) toIntVal(e)]).cast<ffi.Void>(),
      _copyUint8s([for (final e in afterVals) toIntVal(e)]).cast<ffi.Void>(),
    ),
    DType.boolean => (
      ScratchArena.copyBools([
        for (final e in beforeVals) toBoolVal(e),
      ]).cast<ffi.Void>(),
      ScratchArena.copyBools([
        for (final e in afterVals) toBoolVal(e),
      ]).cast<ffi.Void>(),
    ),
    DType.complex128 => (
      ScratchArena.copyComplexes([
        for (final e in beforeVals) toComplexVal(e),
      ]).cast<ffi.Void>(),
      ScratchArena.copyComplexes([
        for (final e in afterVals) toComplexVal(e),
      ]).cast<ffi.Void>(),
    ),
    DType.complex64 => (
      ScratchArena.copyFloatComplexes([
        for (final e in beforeVals) toComplexVal(e),
      ]).cast<ffi.Void>(),
      ScratchArena.copyFloatComplexes([
        for (final e in afterVals) toComplexVal(e),
      ]).cast<ffi.Void>(),
    ),
    DType.float16 => (
      _copyInt16s([
        for (final e in beforeVals)
          Float16Utils.encodeFloat16((e as num).toDouble()),
      ]).cast<ffi.Void>(),
      _copyInt16s([
        for (final e in afterVals)
          Float16Utils.encodeFloat16((e as num).toDouble()),
      ]).cast<ffi.Void>(),
    ),
    DType.bfloat16 => (
      _copyInt16s([
        for (final e in beforeVals)
          Float16Utils.encodeBFloat16((e as num).toDouble()),
      ]).cast<ffi.Void>(),
      _copyInt16s([
        for (final e in afterVals)
          Float16Utils.encodeBFloat16((e as num).toDouble()),
      ]).cast<ffi.Void>(),
    ),
    DType.int8 => (
      _copyUint8s([for (final e in beforeVals) toIntVal(e)]).cast<ffi.Void>(),
      _copyUint8s([for (final e in afterVals) toIntVal(e)]).cast<ffi.Void>(),
    ),
    DType.uint64 => (
      ScratchArena.copyInt64s([
        for (final e in beforeVals) toIntVal(e),
      ]).cast<ffi.Void>(),
      ScratchArena.copyInt64s([
        for (final e in afterVals) toIntVal(e),
      ]).cast<ffi.Void>(),
    ),
    DType.uint32 => (
      _copyUint32s([for (final e in beforeVals) toIntVal(e)]).cast<ffi.Void>(),
      _copyUint32s([for (final e in afterVals) toIntVal(e)]).cast<ffi.Void>(),
    ),
    DType.uint16 => (
      _copyInt16s([for (final e in beforeVals) toIntVal(e)]).cast<ffi.Void>(),
      _copyInt16s([for (final e in afterVals) toIntVal(e)]).cast<ffi.Void>(),
    ),
  };

  return (cbPtr, caPtr, isUniform);
}

bool _padArgsShareMemoryWithOut<T extends DTypeTag>(
  NDArray<T> array,
  NDArray<T> out,
  List<(Object? before, Object? after)> normConstantValues,
  List<(Object? before, Object? after)>? normEndValues,
) {
  if (sharesMemory(array, out)) return true;
  for (final (b, a) in normConstantValues) {
    if (b is NDArray && sharesMemory(b, out)) return true;
    if (a is NDArray && sharesMemory(a, out)) return true;
  }
  if (normEndValues != null) {
    for (final (b, a) in normEndValues) {
      if (b is NDArray && sharesMemory(b, out)) return true;
      if (a is NDArray && sharesMemory(a, out)) return true;
    }
  }
  return false;
}

NDArray<T> _padNativeFast<T extends DTypeTag>(
  NDArray<T> array,
  List<(int before, int after)> normPadWidths,
  PadMode mode,
  List<(Object? before, Object? after)> normConstantValues,
  List<int> finalShape,
  NDArray<T>? out,
) {
  final rank = array.rank;
  final modeInt = switch (mode) {
    PadMode.constant => 0,
    PadMode.edge => 1,
    PadMode.reflect => 2,
    PadMode.symmetric => 3,
    PadMode.wrap => 4,
    _ => -1,
  };

  final dtypeInt = array.dtype.index;

  return NDArray.scope(() {
    final bool copyToOut =
        out != null &&
        (!out.isContiguous ||
            _padArgsShareMemoryWithOut(array, out, normConstantValues, null));
    final NDArray<T> targetDest = (out != null && !copyToOut)
        ? out
        : NDArray<T>.create(finalShape, array.dtype, zeroInit: false);

    final marker = ScratchArena.marker;
    try {
      final (cbPtr, caPtr, isUniform) = _prepareConstants(
        array.dtype,
        normConstantValues,
        rank,
      );

      if (rank == 2 && targetDest.isContiguous) {
        final padTop = normPadWidths[0].$1;
        final padBottom = normPadWidths[0].$2;
        final padLeft = normPadWidths[1].$1;
        final padRight = normPadWidths[1].$2;

        final res = ext_bindings.native_pad_2d(
          dtypeInt,
          array.pointer.cast(),
          array.shape[0],
          array.shape[1],
          array.strides[0],
          array.strides[1],
          targetDest.pointer.cast(),
          padTop,
          padBottom,
          padLeft,
          padRight,
          modeInt,
          cbPtr,
          caPtr,
          isUniform,
        );
        if (res == -4) {
          throw OutOfMemoryError();
        }
        if (res < 0) {
          throw StateError('native_pad_2d failed with code $res');
        }
      } else {
        final srcShapePtr = ScratchArena.copyInt64s(array.shape);
        final srcStridesPtr = ScratchArena.copyInt64s(array.strides);
        final destShapePtr = ScratchArena.copyInt64s(targetDest.shape);
        final destStridesPtr = ScratchArena.copyInt64s(targetDest.strides);
        final padBeforePtr = ScratchArena.copyInt64s([
          for (final w in normPadWidths) w.$1,
        ]);
        final padAfterPtr = ScratchArena.copyInt64s([
          for (final w in normPadWidths) w.$2,
        ]);

        final res = ext_bindings.native_pad_nd(
          dtypeInt,
          array.pointer.cast(),
          srcShapePtr,
          srcStridesPtr,
          targetDest.pointer.cast(),
          destShapePtr,
          destStridesPtr,
          padBeforePtr,
          padAfterPtr,
          rank,
          modeInt,
          cbPtr,
          caPtr,
          isUniform,
        );
        if (res == -4) {
          throw OutOfMemoryError();
        }
        if (res < 0) {
          throw StateError('native_pad_nd failed with code $res');
        }
      }
    } finally {
      ScratchArena.reset(marker);
    }

    if (copyToOut) {
      targetDest.copy(out: out);
      return out;
    }

    if (out == null) {
      targetDest.detachToParentScope();
    }
    return targetDest;
  });
}

NDArray<T> _padAxisByAxis<T extends DTypeTag>(
  NDArray<T> array,
  List<(int before, int after)> normPadWidths,
  PadMode mode,
  List<(Object? before, Object? after)> normConstantValues,
  List<(Object? before, Object? after)> normEndValues,
  List<(int before, int after)> normStatLengths,
  NDArray<T>? out,
) {
  final rank = array.rank;
  return NDArray.scope(() {
    final bool useTempOut =
        out != null &&
        (!out.isContiguous ||
            _padArgsShareMemoryWithOut(
              array,
              out,
              normConstantValues,
              normEndValues,
            ));
    var currentSrc = array;

    for (int axis = 0; axis < rank; axis++) {
      final padBefore = normPadWidths[axis].$1;
      final padAfter = normPadWidths[axis].$2;

      final isLastAxis = axis == rank - 1;
      final skipThisAxis =
          padBefore == 0 &&
          padAfter == 0 &&
          (!isLastAxis || out == null || useTempOut);

      if (skipThisAxis) {
        continue;
      }

      final nextShape = List<int>.from(currentSrc.shape);
      nextShape[axis] += padBefore + padAfter;

      final NDArray<T> currentDest;
      if (isLastAxis && out != null && !useTempOut) {
        currentDest = out;
      } else {
        currentDest = NDArray<T>.create(
          nextShape,
          array.dtype,
          zeroInit: false,
        );
      }

      try {
        _padAxis(
          currentSrc,
          currentDest,
          axis,
          padBefore,
          padAfter,
          mode,
          normConstantValues[axis].$1,
          normConstantValues[axis].$2,
          normEndValues[axis].$1,
          normEndValues[axis].$2,
          normStatLengths[axis].$1,
          normStatLengths[axis].$2,
        );
      } finally {
        if (!identical(currentSrc, array)) {
          currentSrc.dispose();
        }
      }
      currentSrc = currentDest;
    }

    if (identical(currentSrc, array)) {
      if (out != null) {
        array.copy(out: out);
        return out;
      }
      final copy = array.copy();
      copy.detachToParentScope();
      return copy;
    }
    if (useTempOut) {
      currentSrc.copy(out: out);
      return out;
    }
    if (out == null) {
      currentSrc.detachToParentScope();
    }
    return currentSrc;
  });
}

void _padAxis<T extends DTypeTag>(
  NDArray<T> src,
  NDArray<T> dest,
  int axis,
  int padBefore,
  int padAfter,
  PadMode mode,
  Object? constantBefore,
  Object? constantAfter,
  Object? endBefore,
  Object? endAfter,
  int statLengthBefore,
  int statLengthAfter,
) {
  final marker = ScratchArena.marker;
  try {
    final shapeSrcPtr = ScratchArena.copyInts(src.shape);
    final stridesSrcPtr = ScratchArena.copyInts(src.strides);
    final shapeDestPtr = ScratchArena.copyInts(dest.shape);
    final rank = src.rank;
    final modeInt = switch (mode) {
      PadMode.constant => 0,
      PadMode.edge => 1,
      PadMode.reflect => 2,
      PadMode.symmetric => 3,
      PadMode.wrap => 4,
      PadMode.linearRamp => 5,
      PadMode.max => 6,
      PadMode.mean => 7,
      PadMode.median => 8,
      PadMode.min => 9,
    };

    switch (src.dtype) {
      case DType.float64:
        bindings.pad_axis_double(
          src.pointer.cast(),
          shapeSrcPtr,
          stridesSrcPtr,
          dest.pointer.cast(),
          shapeDestPtr,
          rank,
          axis,
          padBefore,
          padAfter,
          modeInt,
          (constantBefore as num).toDouble(),
          (constantAfter as num).toDouble(),
          (endBefore as num).toDouble(),
          (endAfter as num).toDouble(),
          statLengthBefore,
          statLengthAfter,
        );
      case DType.float32:
        bindings.pad_axis_float(
          src.pointer.cast(),
          shapeSrcPtr,
          stridesSrcPtr,
          dest.pointer.cast(),
          shapeDestPtr,
          rank,
          axis,
          padBefore,
          padAfter,
          modeInt,
          (constantBefore as num).toDouble(),
          (constantAfter as num).toDouble(),
          (endBefore as num).toDouble(),
          (endAfter as num).toDouble(),
          statLengthBefore,
          statLengthAfter,
        );
      case DType.int64:
        bindings.pad_axis_int64(
          src.pointer.cast(),
          shapeSrcPtr,
          stridesSrcPtr,
          dest.pointer.cast(),
          shapeDestPtr,
          rank,
          axis,
          padBefore,
          padAfter,
          modeInt,
          constantBefore is int
              ? constantBefore
              : (constantBefore as num).toInt(),
          constantAfter is int ? constantAfter : (constantAfter as num).toInt(),
          endBefore is int ? endBefore : (endBefore as num).toInt(),
          endAfter is int ? endAfter : (endAfter as num).toInt(),
          statLengthBefore,
          statLengthAfter,
        );
      case DType.uint64:
        bindings.pad_axis_uint64(
          src.pointer.cast(),
          shapeSrcPtr,
          stridesSrcPtr,
          dest.pointer.cast(),
          shapeDestPtr,
          rank,
          axis,
          padBefore,
          padAfter,
          modeInt,
          constantBefore is int
              ? constantBefore
              : (constantBefore as num).toInt(),
          constantAfter is int ? constantAfter : (constantAfter as num).toInt(),
          endBefore is int ? endBefore : (endBefore as num).toInt(),
          endAfter is int ? endAfter : (endAfter as num).toInt(),
          statLengthBefore,
          statLengthAfter,
        );
      case DType.int32:
        bindings.pad_axis_int32(
          src.pointer.cast(),
          shapeSrcPtr,
          stridesSrcPtr,
          dest.pointer.cast(),
          shapeDestPtr,
          rank,
          axis,
          padBefore,
          padAfter,
          modeInt,
          constantBefore is int
              ? constantBefore
              : (constantBefore as num).toInt(),
          constantAfter is int ? constantAfter : (constantAfter as num).toInt(),
          endBefore is int ? endBefore : (endBefore as num).toInt(),
          endAfter is int ? endAfter : (endAfter as num).toInt(),
          statLengthBefore,
          statLengthAfter,
        );
      case DType.int16:
        bindings.pad_axis_int16(
          src.pointer.cast(),
          shapeSrcPtr,
          stridesSrcPtr,
          dest.pointer.cast(),
          shapeDestPtr,
          rank,
          axis,
          padBefore,
          padAfter,
          modeInt,
          constantBefore is int
              ? constantBefore
              : (constantBefore as num).toInt(),
          constantAfter is int ? constantAfter : (constantAfter as num).toInt(),
          endBefore is int ? endBefore : (endBefore as num).toInt(),
          endAfter is int ? endAfter : (endAfter as num).toInt(),
          statLengthBefore,
          statLengthAfter,
        );
      case DType.uint8:
      case DType.boolean:
        int cb = 0;
        int ca = 0;
        int eb = 0;
        int ea = 0;
        int toBoolBit(Object? v) =>
            v is bool ? (v ? 1 : 0) : ((v as num) != 0 ? 1 : 0);
        int toIntVal(Object? v) => v is int ? v : (v as num).toInt();
        if (src.dtype == DType.boolean) {
          cb = toBoolBit(constantBefore);
          ca = toBoolBit(constantAfter);
          eb = toBoolBit(endBefore);
          ea = toBoolBit(endAfter);
        } else {
          cb = toIntVal(constantBefore);
          ca = toIntVal(constantAfter);
          eb = toIntVal(endBefore);
          ea = toIntVal(endAfter);
        }
        bindings.pad_axis_uint8(
          src.pointer.cast(),
          shapeSrcPtr,
          stridesSrcPtr,
          dest.pointer.cast(),
          shapeDestPtr,
          rank,
          axis,
          padBefore,
          padAfter,
          modeInt,
          cb,
          ca,
          eb,
          ea,
          statLengthBefore,
          statLengthAfter,
        );
      case DType.complex128:
        final cbPtr = ScratchArena.allocate<bindings.cpx_t>(
          ffi.sizeOf<bindings.cpx_t>(),
        );
        final caPtr = ScratchArena.allocate<bindings.cpx_t>(
          ffi.sizeOf<bindings.cpx_t>(),
        );
        final ebPtr = ScratchArena.allocate<bindings.cpx_t>(
          ffi.sizeOf<bindings.cpx_t>(),
        );
        final eaPtr = ScratchArena.allocate<bindings.cpx_t>(
          ffi.sizeOf<bindings.cpx_t>(),
        );

        Complex toComplexVal(Object? v) =>
            v is Complex ? v : Complex((v as num).toDouble(), 0.0);
        final cb = toComplexVal(constantBefore);
        final ca = toComplexVal(constantAfter);
        final eb = toComplexVal(endBefore);
        final ea = toComplexVal(endAfter);

        cbPtr.ref.r = cb.real;
        cbPtr.ref.i = cb.imag;
        caPtr.ref.r = ca.real;
        caPtr.ref.i = ca.imag;
        ebPtr.ref.r = eb.real;
        ebPtr.ref.i = eb.imag;
        eaPtr.ref.r = ea.real;
        eaPtr.ref.i = ea.imag;

        bindings.pad_axis_complex128(
          src.pointer.cast(),
          shapeSrcPtr,
          stridesSrcPtr,
          dest.pointer.cast(),
          shapeDestPtr,
          rank,
          axis,
          padBefore,
          padAfter,
          modeInt,
          cbPtr,
          caPtr,
          ebPtr,
          eaPtr,
          statLengthBefore,
          statLengthAfter,
        );
      case DType.complex64:
        final cbPtr = ScratchArena.allocate<bindings.cpx_f_t>(
          ffi.sizeOf<bindings.cpx_f_t>(),
        );
        final caPtr = ScratchArena.allocate<bindings.cpx_f_t>(
          ffi.sizeOf<bindings.cpx_f_t>(),
        );
        final ebPtr = ScratchArena.allocate<bindings.cpx_f_t>(
          ffi.sizeOf<bindings.cpx_f_t>(),
        );
        final eaPtr = ScratchArena.allocate<bindings.cpx_f_t>(
          ffi.sizeOf<bindings.cpx_f_t>(),
        );

        Complex toComplexVal(Object? v) =>
            v is Complex ? v : Complex((v as num).toDouble(), 0.0);
        final cb = toComplexVal(constantBefore);
        final ca = toComplexVal(constantAfter);
        final eb = toComplexVal(endBefore);
        final ea = toComplexVal(endAfter);

        cbPtr.ref.r = cb.real;
        cbPtr.ref.i = cb.imag;
        caPtr.ref.r = ca.real;
        caPtr.ref.i = ca.imag;
        ebPtr.ref.r = eb.real;
        ebPtr.ref.i = eb.imag;
        eaPtr.ref.r = ea.real;
        eaPtr.ref.i = ea.imag;

        bindings.pad_axis_complex64(
          src.pointer.cast(),
          shapeSrcPtr,
          stridesSrcPtr,
          dest.pointer.cast(),
          shapeDestPtr,
          rank,
          axis,
          padBefore,
          padAfter,
          modeInt,
          cbPtr,
          caPtr,
          ebPtr,
          eaPtr,
          statLengthBefore,
          statLengthAfter,
        );
      case DType.int8:
        NDArray.scope(() {
          final i64Src = castNDArray<Int64>(src, DType.int64);
          final i64Dest = NDArray<Int64>.zeros(dest.shape, DType.int64);
          _padAxis<Int64>(
            i64Src,
            i64Dest,
            axis,
            padBefore,
            padAfter,
            mode,
            (constantBefore as num).toInt(),
            (constantAfter as num).toInt(),
            (endBefore as num).toInt(),
            (endAfter as num).toInt(),
            statLengthBefore,
            statLengthAfter,
          );
          final castedDest = castNDArray(i64Dest, dest.dtype);
          castedDest.copy(out: dest);
        });
      case DType.uint32:
      case DType.uint16:
        NDArray.scope(() {
          final u64Src = castNDArray<Uint64>(src, DType.uint64);
          final u64Dest = NDArray<Uint64>.zeros(dest.shape, DType.uint64);
          _padAxis<Uint64>(
            u64Src,
            u64Dest,
            axis,
            padBefore,
            padAfter,
            mode,
            (constantBefore as num).toInt(),
            (constantAfter as num).toInt(),
            (endBefore as num).toInt(),
            (endAfter as num).toInt(),
            statLengthBefore,
            statLengthAfter,
          );
          final castedDest = castNDArray(u64Dest, dest.dtype);
          castedDest.copy(out: dest);
        });
      case DType.float16:
      case DType.bfloat16:
        NDArray.scope(() {
          double toDoubleUnsignedAware(Object v) => (v as num).toDouble();
          final doubleSrc = castNDArray<Float64>(src, DType.float64);
          final doubleDest = NDArray<Float64>.zeros(dest.shape, DType.float64);
          _padAxis<Float64>(
            doubleSrc,
            doubleDest,
            axis,
            padBefore,
            padAfter,
            mode,
            toDoubleUnsignedAware(constantBefore!),
            toDoubleUnsignedAware(constantAfter!),
            toDoubleUnsignedAware(endBefore!),
            toDoubleUnsignedAware(endAfter!),
            statLengthBefore,
            statLengthAfter,
          );
          final castedDest = castNDArray(doubleDest, dest.dtype);
          castedDest.copy(out: dest);
        });
    }
    checkNativeOom();
  } finally {
    ScratchArena.reset(marker);
  }
}
