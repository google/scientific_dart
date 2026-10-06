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

import '../autograd/autograd.dart';
import '../backend/compute_engine.dart';
import '../backend/kernels.dart';
import '../dtype.dart';
import '../exceptions.dart';
import '../gpu_array.dart';
import '../slice.dart';

/// Padding modes for [pad].
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
}

void _checkNotDisposed(GpuArray<DTypeTag> a, String name) {
  if (a.isDisposed || a.buffer.isDisposed) {
    throw GpuDeviceDisposedException(
      'Cannot operate on disposed GpuArray ($name).',
    );
  }
}

GpuArray<T> _allocateOrValidateOut<T extends DTypeTag>(
  String opName,
  List<int> outShape,
  DType<T> outDType,
  GpuArray<DTypeTag> reference,
  GpuArray<T>? out,
) {
  if (out != null) {
    _checkNotDisposed(out, 'out');
    if (out.size > 1 && out.strides.contains(0)) {
      throw ArgumentError.value(
        out,
        'out',
        'Must be writeable and not a broadcasted view.',
      );
    }
    if (!areShapesEqual(out.shape, outShape)) {
      throw ArgumentError.value(
        out.shape,
        'out.shape',
        'Must match expected output shape $outShape for $opName.',
      );
    }
    if (out.dtype != outDType) {
      throw ArgumentError.value(
        out.dtype,
        'out.dtype',
        'Must match output dtype $outDType.',
      );
    }
    if (out.device != reference.device) {
      throw ArgumentError.value(
        out.device,
        'out.device',
        'Must be on the same device as input (${reference.device}).',
      );
    }
    return out;
  }
  return GpuArray<T>.empty(outShape, outDType, device: reference.device);
}

/// Joins a sequence of [arrays] along an existing [axis].
GpuArray<T> concatenate<T extends DTypeTag>(
  List<GpuArray<DTypeTag>> arrays, {
  int axis = 0,
  GpuArray<T>? out,
}) {
  if (arrays.isEmpty) {
    throw ArgumentError.value(arrays, 'arrays', 'Must not be empty.');
  }
  final first = arrays[0];
  _checkNotDisposed(first, 'arrays[0]');
  final rank = first.shape.length;
  final normAxis = axis < 0 ? axis + rank : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw GpuAxisOutOfBoundsException(axis, rank);
  }

  var outDType = first.dtype;
  var totalAxisLength = 0;

  for (var i = 0; i < arrays.length; i++) {
    final arr = arrays[i];
    _checkNotDisposed(arr, 'arrays[$i]');
    if (arr.device != first.device) {
      throw ArgumentError.value(
        arr.device,
        'arrays[$i].device',
        'Must be on the same device as arrays[0] (${first.device}).',
      );
    }
    if (arr.shape.length != rank) {
      throw GpuShapeMismatchException('concatenate', arr.shape, first.shape);
    }
    for (var d = 0; d < rank; d++) {
      if (d != normAxis && arr.shape[d] != first.shape[d]) {
        throw GpuShapeMismatchException('concatenate', arr.shape, first.shape);
      }
    }
    totalAxisLength += arr.shape[normAxis];
    outDType = GpuArray.promoteDTypes(outDType, arr.dtype);
  }

  final outShape = List<int>.of(first.shape);
  outShape[normAxis] = totalAxisLength;
  final result = _allocateOrValidateOut<T>(
    'concatenate',
    outShape,
    outDType as DType<T>,
    first,
    out,
  );

  GpuKernels.executeConcatenate(
    srcBuffers: arrays.map((a) => a.buffer).toList(),
    srcShapes: arrays.map((a) => a.shape).toList(),
    srcStrides: arrays.map((a) => a.strides).toList(),
    srcOffsets: arrays.map((a) => a.offsetElements).toList(),
    srcDtypes: arrays.map((a) => a.dtype).toList(),
    dst: result.buffer,
    outShape: outShape,
    outStrides: result.strides,
    offsetDst: result.offsetElements,
    dtypeDst: outDType,
    axis: normAxis,
  );

  if (isGradEnabled && arrays.any((a) => a.requiresGrad)) {
    result.requiresGrad = true;
    result.gradFn = ConcatenateBackward(arrays, axis: normAxis);
  }

  return result;
}

/// Joins a sequence of [arrays] along a new [axis].
GpuArray<T> stack<T extends DTypeTag>(
  List<GpuArray<DTypeTag>> arrays, {
  int axis = 0,
  GpuArray<T>? out,
}) {
  if (arrays.isEmpty) {
    throw ArgumentError.value(arrays, 'arrays', 'Must not be empty.');
  }
  final expanded = arrays.map((a) => expandDims(a, axis)).toList();
  try {
    return concatenate<T>(expanded, axis: axis, out: out);
  } finally {
    for (final view in expanded) {
      view.dispose();
    }
  }
}

/// Stacks arrays in sequence vertically (row-wise / along axis 0).
GpuArray<T> vstack<T extends DTypeTag>(
  List<GpuArray<DTypeTag>> arrays, {
  GpuArray<T>? out,
}) {
  if (arrays.isEmpty) {
    throw ArgumentError.value(arrays, 'arrays', 'Must not be empty.');
  }
  final tempViews = <GpuArray<DTypeTag>>[];
  final prepared = arrays.map((a) {
    _checkNotDisposed(a, 'arrays');
    if (a.shape.length == 1) {
      final view = a.reshape([1, a.shape[0]]);
      tempViews.add(view);
      return view;
    }
    return a;
  }).toList();
  try {
    return concatenate<T>(prepared, axis: 0, out: out);
  } finally {
    for (final v in tempViews) {
      v.dispose();
    }
  }
}

/// Stacks arrays in sequence horizontally (column-wise / along axis 1).
GpuArray<T> hstack<T extends DTypeTag>(
  List<GpuArray<DTypeTag>> arrays, {
  GpuArray<T>? out,
}) {
  if (arrays.isEmpty) {
    throw ArgumentError.value(arrays, 'arrays', 'Must not be empty.');
  }
  final tempViews = <GpuArray<DTypeTag>>[];
  final prepared = arrays.map((a) {
    _checkNotDisposed(a, 'arrays');
    if (a.shape.length == 1) {
      final view = a.reshape([a.shape[0], 1]);
      tempViews.add(view);
      return view;
    }
    return a;
  }).toList();
  final axis = (arrays[0].shape.length == 1) ? 0 : 1;
  try {
    return concatenate<T>(prepared, axis: axis, out: out);
  } finally {
    for (final v in tempViews) {
      v.dispose();
    }
  }
}

/// Stacks arrays in sequence depth-wise (along axis 2).
GpuArray<T> dstack<T extends DTypeTag>(
  List<GpuArray<DTypeTag>> arrays, {
  GpuArray<T>? out,
}) {
  if (arrays.isEmpty) {
    throw ArgumentError.value(arrays, 'arrays', 'Must not be empty.');
  }
  final tempViews = <GpuArray<DTypeTag>>[];
  final prepared = arrays.map((a) {
    _checkNotDisposed(a, 'arrays');
    if (a.shape.length == 1) {
      final view = a.reshape([1, a.shape[0], 1]);
      tempViews.add(view);
      return view;
    } else if (a.shape.length == 2) {
      final view = a.reshape([a.shape[0], a.shape[1], 1]);
      tempViews.add(view);
      return view;
    }
    return a;
  }).toList();
  try {
    return concatenate<T>(prepared, axis: 2, out: out);
  } finally {
    for (final v in tempViews) {
      v.dispose();
    }
  }
}

/// Stacks 1D or 2D arrays as columns to create a 2D array.
GpuArray<T> columnStack<T extends DTypeTag>(
  List<GpuArray<DTypeTag>> arrays, {
  GpuArray<T>? out,
}) {
  if (arrays.isEmpty) {
    throw ArgumentError.value(arrays, 'arrays', 'Must not be empty.');
  }
  final tempViews = <GpuArray<DTypeTag>>[];
  final prepared = arrays.map((a) {
    _checkNotDisposed(a, 'arrays');
    if (a.shape.length == 1) {
      final view = a.reshape([a.shape[0], 1]);
      tempViews.add(view);
      return view;
    }
    return a;
  }).toList();
  try {
    return concatenate<T>(prepared, axis: 1, out: out);
  } finally {
    for (final v in tempViews) {
      v.dispose();
    }
  }
}

/// Splits an array into multiple sub-arrays along [axis].
List<GpuArray<T>> split<T extends DTypeTag>(
  GpuArray<T> a,
  Object indicesOrSections, {
  int axis = 0,
}) {
  _checkNotDisposed(a, 'a');
  final rank = a.shape.length;
  final normAxis = axis < 0 ? axis + rank : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw GpuAxisOutOfBoundsException(axis, rank);
  }
  final dimensionLength = a.shape[normAxis];

  final splitPoints = <int>[0];

  if (indicesOrSections is int) {
    final sections = indicesOrSections;
    if (sections <= 0 || dimensionLength % sections != 0) {
      throw ArgumentError.value(
        indicesOrSections,
        'indicesOrSections',
        'Must evenly divide axis length ($dimensionLength).',
      );
    }
    final step = dimensionLength ~/ sections;
    for (var i = 1; i < sections; i++) {
      splitPoints.add(i * step);
    }
  } else if (indicesOrSections is List<int>) {
    splitPoints.addAll(indicesOrSections);
  } else {
    throw ArgumentError.value(
      indicesOrSections,
      'indicesOrSections',
      'Must be an int or List<int>.',
    );
  }
  splitPoints.add(dimensionLength);

  final result = <GpuArray<T>>[];
  for (var i = 0; i < splitPoints.length - 1; i++) {
    final start = splitPoints[i];
    final stop = splitPoints[i + 1];

    final sliceSpecs = List<Object>.generate(rank, (d) {
      if (d == normAxis) {
        return Slice(start, stop);
      }
      return const All();
    });

    result.add(a.slice(sliceSpecs));
  }
  return result;
}

/// Splits an array into multiple sub-arrays (allowing unequal division).
List<GpuArray<T>> arraySplit<T extends DTypeTag>(
  GpuArray<T> a,
  Object indicesOrSections, {
  int axis = 0,
}) {
  _checkNotDisposed(a, 'a');
  if (indicesOrSections is! int) {
    return split<T>(a, indicesOrSections, axis: axis);
  }
  final rank = a.shape.length;
  final normAxis = axis < 0 ? axis + rank : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw GpuAxisOutOfBoundsException(axis, rank);
  }
  final dimensionLength = a.shape[normAxis];
  final n = indicesOrSections;
  if (n <= 0) {
    throw ArgumentError.value(
      indicesOrSections,
      'indicesOrSections',
      'Must be positive.',
    );
  }

  final div = dimensionLength ~/ n;
  final mod = dimensionLength % n;

  final splitPoints = <int>[0];
  var current = 0;
  for (var i = 0; i < n; i++) {
    final size = i < mod ? div + 1 : div;
    current += size;
    if (i < n - 1) {
      splitPoints.add(current);
    }
  }
  splitPoints.add(dimensionLength);

  final result = <GpuArray<T>>[];
  for (var i = 0; i < splitPoints.length - 1; i++) {
    final start = splitPoints[i];
    final stop = splitPoints[i + 1];

    final sliceSpecs = List<Object>.generate(rank, (d) {
      if (d == normAxis) {
        return Slice(start, stop);
      }
      return const All();
    });

    result.add(a.slice(sliceSpecs));
  }
  return result;
}

/// Splits array horizontally (along axis 1 for >=2D, or axis 0 for 1D).
List<GpuArray<T>> hsplit<T extends DTypeTag>(
  GpuArray<T> a,
  Object indicesOrSections,
) {
  _checkNotDisposed(a, 'a');
  if (a.shape.isEmpty) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must have at least 1 dimension for hsplit.',
    );
  }
  final axis = a.shape.length == 1 ? 0 : 1;
  return split<T>(a, indicesOrSections, axis: axis);
}

/// Splits array vertically (along axis 0).
List<GpuArray<T>> vsplit<T extends DTypeTag>(
  GpuArray<T> a,
  Object indicesOrSections,
) {
  _checkNotDisposed(a, 'a');
  if (a.shape.length < 2) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must have at least 2 dimensions for vsplit.',
    );
  }
  return split<T>(a, indicesOrSections, axis: 0);
}

/// Splits array depth-wise (along axis 2).
List<GpuArray<T>> dsplit<T extends DTypeTag>(
  GpuArray<T> a,
  Object indicesOrSections,
) {
  _checkNotDisposed(a, 'a');
  if (a.shape.length < 3) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must have at least 3 dimensions for dsplit.',
    );
  }
  return split<T>(a, indicesOrSections, axis: 2);
}

/// Constructs an array by repeating [a] the number of times given by [reps].
GpuArray<T> tile<T extends DTypeTag>(
  GpuArray<T> a,
  List<int> reps, {
  GpuArray<T>? out,
}) {
  _checkNotDisposed(a, 'a');
  for (final r in reps) {
    if (r < 0) {
      throw ArgumentError.value(
        reps,
        'reps',
        'Must contain non-negative ints.',
      );
    }
  }
  final rank = reps.length > a.shape.length ? reps.length : a.shape.length;
  final padRankA = rank - a.shape.length;
  final padRankR = rank - reps.length;

  final shapeA = List<int>.filled(padRankA, 1, growable: true)..addAll(a.shape);
  final repsNormalized = List<int>.filled(padRankR, 1, growable: true)
    ..addAll(reps);

  final outShape = List<int>.generate(
    rank,
    (d) => shapeA[d] * repsNormalized[d],
  );
  final result = _allocateOrValidateOut<T>('tile', outShape, a.dtype, a, out);

  GpuKernels.executeTile(
    src: a.buffer,
    shapeSrc: a.shape,
    stridesSrc: a.strides,
    offsetSrc: a.offsetElements,
    dtypeSrc: a.dtype,
    dst: result.buffer,
    outShape: outShape,
    outStrides: result.strides,
    offsetDst: result.offsetElements,
    dtypeDst: a.dtype,
  );

  return result;
}

/// Repeats elements of an array [repeats] times along [axis].
GpuArray<T> repeat<T extends DTypeTag>(
  GpuArray<T> a,
  int repeats, {
  int? axis,
  GpuArray<T>? out,
}) {
  _checkNotDisposed(a, 'a');
  RangeError.checkNotNegative(repeats, 'repeats');
  if (axis == null) {
    final flat = a.flatten();
    try {
      final total = flat.shape[0];
      final outShape = [total * repeats];
      final result = _allocateOrValidateOut<T>(
        'repeat',
        outShape,
        a.dtype,
        a,
        out,
      );

      GpuKernels.executeRepeat(
        src: flat.buffer,
        shapeSrc: flat.shape,
        stridesSrc: flat.strides,
        offsetSrc: flat.offsetElements,
        dtypeSrc: flat.dtype,
        dst: result.buffer,
        outShape: outShape,
        outStrides: result.strides,
        offsetDst: result.offsetElements,
        dtypeDst: a.dtype,
        repeats: repeats,
        axis: 0,
      );

      return result;
    } finally {
      if (!identical(flat, a)) {
        flat.dispose();
      }
    }
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? axis + rank : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw GpuAxisOutOfBoundsException(axis, rank);
  }
  final outShape = List<int>.of(a.shape);
  outShape[normAxis] *= repeats;

  final result = _allocateOrValidateOut<T>('repeat', outShape, a.dtype, a, out);

  GpuKernels.executeRepeat(
    src: a.buffer,
    shapeSrc: a.shape,
    stridesSrc: a.strides,
    offsetSrc: a.offsetElements,
    dtypeSrc: a.dtype,
    dst: result.buffer,
    outShape: outShape,
    outStrides: result.strides,
    offsetDst: result.offsetElements,
    dtypeDst: a.dtype,
    repeats: repeats,
    axis: normAxis,
  );

  return result;
}

/// Pads an array with [padWidth] according to [mode].
GpuArray<T> pad<T extends DTypeTag>(
  GpuArray<T> a,
  List<List<int>> padWidth, {
  PadMode mode = PadMode.constant,
  Object constantValues = 0,
  GpuArray<T>? out,
}) {
  _checkNotDisposed(a, 'a');
  final rank = a.shape.length;
  if (padWidth.length != rank) {
    throw ArgumentError.value(
      padWidth,
      'padWidth',
      'Must have length matching tensor rank ($rank).',
    );
  }
  for (var d = 0; d < rank; d++) {
    if (padWidth[d].length != 2 || padWidth[d][0] < 0 || padWidth[d][1] < 0) {
      throw ArgumentError.value(
        padWidth[d],
        'padWidth[$d]',
        'Must be a [before, after] pair of non-negative integers.',
      );
    }
    if (mode != PadMode.constant && a.shape[d] == 0) {
      throw ArgumentError.value(
        a.shape,
        'a.shape',
        'Must not contain empty dimensions when padding with $mode.',
      );
    }
  }

  final outShape = List<int>.generate(
    rank,
    (d) => a.shape[d] + padWidth[d][0] + padWidth[d][1],
  );
  final result = _allocateOrValidateOut<T>('pad', outShape, a.dtype, a, out);

  GpuKernels.executePad(
    src: a.buffer,
    shapeSrc: a.shape,
    stridesSrc: a.strides,
    offsetSrc: a.offsetElements,
    dtypeSrc: a.dtype,
    dst: result.buffer,
    outShape: outShape,
    outStrides: result.strides,
    offsetDst: result.offsetElements,
    dtypeDst: a.dtype,
    padWidth: padWidth,
    constantValue: constantValues,
    padMode: mode.index,
  );
  return result;
}

/// Rolls array elements along a given [axis].
GpuArray<T> roll<T extends DTypeTag>(
  GpuArray<T> a,
  Object shift, {
  Object? axis,
  GpuArray<T>? out,
}) {
  _checkNotDisposed(a, 'a');
  if (axis == null) {
    final total = a.size;
    if (total == 0) {
      return _allocateOrValidateOut<T>('roll', a.shape, a.dtype, a, out);
    }
    final s = (shift is int) ? shift : (shift as List<int>)[0];

    final result = _allocateOrValidateOut<T>('roll', a.shape, a.dtype, a, out);

    GpuKernels.executeRoll(
      src: a.buffer,
      shapeSrc: a.shape,
      stridesSrc: a.strides,
      offsetSrc: a.offsetElements,
      dtypeSrc: a.dtype,
      dst: result.buffer,
      outShape: a.shape,
      outStrides: result.strides,
      offsetDst: result.offsetElements,
      dtypeDst: a.dtype,
      shift: s,
      axis: null,
    );

    return result;
  }

  final rank = a.shape.length;
  final axes = (axis is int) ? [axis] : (axis as List<int>);
  final shifts = (shift is int) ? [shift] : (shift as List<int>);

  if (axes.length != shifts.length) {
    throw ArgumentError.value(
      shifts,
      'shift',
      'Must have the same length as axis (${axes.length}).',
    );
  }

  var current = a;
  for (var i = 0; i < axes.length; i++) {
    final ax = axes[i] < 0 ? axes[i] + rank : axes[i];
    if (ax < 0 || ax >= rank) {
      throw GpuAxisOutOfBoundsException(axes[i], rank);
    }
    final s = shifts[i];

    final isLast = i == axes.length - 1;
    final nextArr = (isLast && out != null)
        ? _allocateOrValidateOut<T>('roll', current.shape, a.dtype, a, out)
        : GpuArray<T>.empty(current.shape, a.dtype, device: a.device);

    GpuKernels.executeRoll(
      src: current.buffer,
      shapeSrc: current.shape,
      stridesSrc: current.strides,
      offsetSrc: current.offsetElements,
      dtypeSrc: current.dtype,
      dst: nextArr.buffer,
      outShape: current.shape,
      outStrides: nextArr.strides,
      offsetDst: nextArr.offsetElements,
      dtypeDst: a.dtype,
      shift: s,
      axis: ax,
    );

    if (!identical(current, a)) {
      current.dispose();
    }
    current = nextArr;
  }

  return current;
}

/// Reverses the order of elements along the given [axis].
GpuArray<T> flip<T extends DTypeTag>(GpuArray<T> a, {Object? axis}) {
  _checkNotDisposed(a, 'a');
  final rank = a.shape.length;
  final List<int> axes;
  if (axis == null) {
    axes = List.generate(rank, (i) => i);
  } else if (axis is int) {
    final norm = axis < 0 ? axis + rank : axis;
    if (norm < 0 || norm >= rank) {
      throw GpuAxisOutOfBoundsException(axis, rank);
    }
    axes = [norm];
  } else if (axis is List<int>) {
    axes = axis.map((ax) {
      final norm = ax < 0 ? ax + rank : ax;
      if (norm < 0 || norm >= rank) {
        throw GpuAxisOutOfBoundsException(ax, rank);
      }
      return norm;
    }).toList();
  } else {
    throw ArgumentError.value(axis, 'axis', 'Must be int, List<int>, or null.');
  }

  final sliceSpecs = List<Object>.generate(rank, (d) {
    if (axes.contains(d)) {
      return const Slice(null, null, -1);
    }
    return const All();
  });

  return a.slice(sliceSpecs);
}

/// Flips an array in the left/right direction (along axis 1).
GpuArray<T> fliplr<T extends DTypeTag>(GpuArray<T> a) {
  _checkNotDisposed(a, 'a');
  if (a.shape.length < 2) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must be at least 2-D for fliplr.',
    );
  }
  return flip(a, axis: 1);
}

/// Flips an array in the up/down direction (along axis 0).
GpuArray<T> flipud<T extends DTypeTag>(GpuArray<T> a) {
  _checkNotDisposed(a, 'a');
  if (a.shape.isEmpty) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must be at least 1-D for flipud.',
    );
  }
  return flip(a, axis: 0);
}

/// Flattens [a] into a contiguous 1-D array view (or copy if non-contiguous).
GpuArray<T> ravel<T extends DTypeTag>(GpuArray<T> a) {
  _checkNotDisposed(a, 'a');
  return a.flatten();
}

/// Rotates an array by 90 degrees in the plane specified by [axes].
GpuArray<T> rot90<T extends DTypeTag>(
  GpuArray<T> a, {
  int k = 1,
  List<int> axes = const [0, 1],
}) {
  _checkNotDisposed(a, 'a');
  final rank = a.shape.length;
  if (axes.length != 2) {
    throw ArgumentError.value(axes, 'axes', 'Must have length 2.');
  }
  final ax1 = axes[0] < 0 ? axes[0] + rank : axes[0];
  final ax2 = axes[1] < 0 ? axes[1] + rank : axes[1];
  if (ax1 < 0 || ax1 >= rank) {
    throw GpuAxisOutOfBoundsException(axes[0], rank);
  }
  if (ax2 < 0 || ax2 >= rank) {
    throw GpuAxisOutOfBoundsException(axes[1], rank);
  }
  if (ax1 == ax2) {
    throw ArgumentError.value(axes, 'axes', 'Must specify two distinct axes.');
  }

  final rot = ((k % 4) + 4) % 4;
  if (rot == 0) return a.copy();
  if (rot == 1) return flip(swapaxes(a, ax1, ax2), axis: ax1);
  if (rot == 2) return flip(flip(a, axis: ax1), axis: ax2);
  return flip(swapaxes(a, ax1, ax2), axis: ax2);
}

/// Extracts a diagonal or constructs a diagonal array.
GpuArray<T> diag<T extends DTypeTag>(
  GpuArray<T> v, {
  int k = 0,
  GpuArray<T>? out,
}) {
  _checkNotDisposed(v, 'v');
  if (v.shape.length == 1) {
    final n = v.shape[0];
    final size = n + k.abs();
    final result = _allocateOrValidateOut<T>(
      'diag',
      [size, size],
      v.dtype,
      v,
      out,
    );
    GpuKernels.executeDiag1DTo2D(
      src: v.buffer,
      srcLength: n,
      strideSrc: v.strides[0],
      offsetSrc: v.offsetElements,
      dtypeSrc: v.dtype,
      dst: result.buffer,
      outSize: size,
      outStrides: result.strides,
      offsetDst: result.offsetElements,
      dtypeDst: result.dtype,
      k: k,
    );
    return result;
  } else if (v.shape.length == 2) {
    return diagonal(v, offset: k, out: out);
  }
  throw ArgumentError.value(v.shape, 'v.shape', 'Must be 1-D or 2-D for diag.');
}

/// Extracts specified diagonals of an N-D array.
///
/// For tensors with rank > 2, the axes [axis1] and [axis2] are removed and the
/// extracted diagonal dimension is appended at the end of the output shape,
/// matching `numpy.diagonal`.
GpuArray<T> diagonal<T extends DTypeTag>(
  GpuArray<T> a, {
  int offset = 0,
  int axis1 = 0,
  int axis2 = 1,
  GpuArray<T>? out,
}) {
  _checkNotDisposed(a, 'a');
  final rank = a.shape.length;
  if (rank < 2) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must be at least 2-D for diagonal.',
    );
  }
  final ax1 = axis1 < 0 ? axis1 + rank : axis1;
  final ax2 = axis2 < 0 ? axis2 + rank : axis2;
  if (ax1 < 0 || ax1 >= rank) {
    throw GpuAxisOutOfBoundsException(axis1, rank);
  }
  if (ax2 < 0 || ax2 >= rank) {
    throw GpuAxisOutOfBoundsException(axis2, rank);
  }
  if (ax1 == ax2) {
    throw ArgumentError.value(
      axis2,
      'axis2',
      'Must be distinct from axis1 ($axis1).',
    );
  }

  final rows = a.shape[ax1];
  final cols = a.shape[ax2];

  var diagonalLength = offset >= 0
      ? ((rows < cols - offset) ? rows : (cols - offset))
      : ((rows + offset < cols) ? (rows + offset) : cols);
  if (diagonalLength < 0) diagonalLength = 0;

  final outShape = <int>[];
  for (var d = 0; d < rank; d++) {
    if (d != ax1 && d != ax2) {
      outShape.add(a.shape[d]);
    }
  }
  outShape.add(diagonalLength);

  final result = _allocateOrValidateOut<T>(
    'diagonal',
    outShape,
    a.dtype,
    a,
    out,
  );
  final totalOut = computeSize(outShape);
  if (totalOut == 0) return result;

  GpuKernels.executeDiagonal(
    src: a.buffer,
    shapeSrc: a.shape,
    stridesSrc: a.strides,
    offsetSrc: a.offsetElements,
    dtypeSrc: a.dtype,
    dst: result.buffer,
    outShape: outShape,
    outStrides: result.strides,
    offsetDst: result.offsetElements,
    dtypeDst: result.dtype,
    offset: offset,
    axis1: ax1,
    axis2: ax2,
  );

  return result;
}

/// Computes the sum along diagonals of the array as a [GpuArray].
///
/// For a 2D array, returns a 0D scalar [GpuArray] (use `.scalar` to read the
/// host scalar value). For N-D arrays (`N > 2`), returns an `(N - 2)`-D array
/// summing the diagonal along [axis1] and [axis2].
GpuArray<T> trace<T extends DTypeTag>(
  GpuArray<T> a, {
  int offset = 0,
  int axis1 = 0,
  int axis2 = 1,
  GpuArray<T>? out,
}) {
  final d = diagonal<T>(a, offset: offset, axis1: axis1, axis2: axis2);
  try {
    return d.sum(axis: -1, out: out);
  } finally {
    d.dispose();
  }
}

/// Extracts the upper triangle of an array.
GpuArray<T> triu<T extends DTypeTag>(
  GpuArray<T> m, {
  int k = 0,
  GpuArray<T>? out,
}) {
  _checkNotDisposed(m, 'm');
  final rank = m.shape.length;
  if (rank < 2) {
    throw ArgumentError.value(
      m.shape,
      'm.shape',
      'Must have at least 2 dimensions for triu.',
    );
  }

  final result = _allocateOrValidateOut<T>('triu', m.shape, m.dtype, m, out);

  GpuKernels.executeTriangular(
    src: m.buffer,
    shapeSrc: m.shape,
    stridesSrc: m.strides,
    offsetSrc: m.offsetElements,
    dtypeSrc: m.dtype,
    dst: result.buffer,
    outStrides: result.strides,
    offsetDst: result.offsetElements,
    dtypeDst: m.dtype,
    k: k,
    upper: true,
  );

  return result;
}

/// Extracts the lower triangle of an array.
GpuArray<T> tril<T extends DTypeTag>(
  GpuArray<T> m, {
  int k = 0,
  GpuArray<T>? out,
}) {
  _checkNotDisposed(m, 'm');
  final rank = m.shape.length;
  if (rank < 2) {
    throw ArgumentError.value(
      m.shape,
      'm.shape',
      'Must have at least 2 dimensions for tril.',
    );
  }

  final result = _allocateOrValidateOut<T>('tril', m.shape, m.dtype, m, out);

  GpuKernels.executeTriangular(
    src: m.buffer,
    shapeSrc: m.shape,
    stridesSrc: m.strides,
    offsetSrc: m.offsetElements,
    dtypeSrc: m.dtype,
    dst: result.buffer,
    outStrides: result.strides,
    offsetDst: result.offsetElements,
    dtypeDst: m.dtype,
    k: k,
    upper: false,
  );

  return result;
}

/// Moves axes of an array to new positions.
GpuArray<T> moveaxis<T extends DTypeTag>(
  GpuArray<T> a,
  Object source,
  Object destination,
) {
  _checkNotDisposed(a, 'a');
  final rank = a.shape.length;
  final srcList = (source is int) ? [source] : (source as List<int>);
  final dstList = (destination is int)
      ? [destination]
      : (destination as List<int>);

  if (srcList.length != dstList.length) {
    throw ArgumentError.value(
      destination,
      'destination',
      'Must have the same length as source (${srcList.length}).',
    );
  }

  final normSrc = srcList.map((x) {
    final norm = x < 0 ? x + rank : x;
    if (norm < 0 || norm >= rank) {
      throw GpuAxisOutOfBoundsException(x, rank);
    }
    return norm;
  }).toList();
  final normDst = dstList.map((x) {
    final norm = x < 0 ? x + rank : x;
    if (norm < 0 || norm >= rank) {
      throw GpuAxisOutOfBoundsException(x, rank);
    }
    return norm;
  }).toList();

  final order = <int>[];
  for (var i = 0; i < rank; i++) {
    if (!normSrc.contains(i)) order.add(i);
  }
  final pairs = List.generate(
    normDst.length,
    (i) => (dst: normDst[i], src: normSrc[i]),
  )..sort((x, y) => x.dst.compareTo(y.dst));
  for (final pair in pairs) {
    order.insert(pair.dst, pair.src);
  }

  return a.transpose(order);
}

/// Interchanges two axes of an array.
GpuArray<T> swapaxes<T extends DTypeTag>(GpuArray<T> a, int axis1, int axis2) {
  _checkNotDisposed(a, 'a');
  final rank = a.shape.length;
  final ax1 = axis1 < 0 ? axis1 + rank : axis1;
  final ax2 = axis2 < 0 ? axis2 + rank : axis2;
  if (ax1 < 0 || ax1 >= rank) {
    throw GpuAxisOutOfBoundsException(axis1, rank);
  }
  if (ax2 < 0 || ax2 >= rank) {
    throw GpuAxisOutOfBoundsException(axis2, rank);
  }

  final order = List<int>.generate(rank, (i) => i);
  order[ax1] = ax2;
  order[ax2] = ax1;

  return a.transpose(order);
}

/// Expands the shape of an array by inserting a new axis at [axis].
GpuArray<T> expandDims<T extends DTypeTag>(GpuArray<T> a, Object axis) {
  _checkNotDisposed(a, 'a');
  final rank = a.shape.length;
  final axes = (axis is int) ? [axis] : (axis as List<int>);
  final outRank = rank + axes.length;

  final normAxes = axes.map((ax) {
    final norm = ax < 0 ? ax + outRank : ax;
    if (norm < 0 || norm >= outRank) {
      throw GpuAxisOutOfBoundsException(ax, outRank);
    }
    return norm;
  }).toList()..sort();

  final newShape = List<int>.of(a.shape);
  for (final ax in normAxes) {
    newShape.insert(ax, 1);
  }

  return a.reshape(newShape);
}

/// Broadcasts an array to a new [shape].
GpuArray<T> broadcastTo<T extends DTypeTag>(GpuArray<T> a, List<int> shape) {
  _checkNotDisposed(a, 'a');
  final bStrides = broadcastStrides(a.shape, a.strides, shape);
  final isContig = isContiguousLayout(shape, bStrides);

  return GpuArray<T>.fromBuffer(
    buffer: a.buffer,
    shape: List.unmodifiable(shape),
    strides: List.unmodifiable(bStrides),
    dtype: a.dtype,
    device: a.device,
    offsetElements: a.offsetElements,
    isContiguous: isContig,
    parent: a,
  );
}

/// Broadcasts any number of [arrays] against each other to a common shape.
List<GpuArray<DTypeTag>> broadcastArrays(List<GpuArray<DTypeTag>> arrays) {
  if (arrays.isEmpty) return const [];
  var commonShape = arrays[0].shape;
  for (var i = 0; i < arrays.length; i++) {
    _checkNotDisposed(arrays[i], 'arrays[$i]');
    commonShape = broadcastShapes(commonShape, arrays[i].shape);
  }
  return arrays.map((a) => a.broadcastTo(commonShape)).toList();
}

/// Views [a] as an array with at least one dimension.
GpuArray<T> atleast1d<T extends DTypeTag>(GpuArray<T> a) {
  _checkNotDisposed(a, 'a');
  if (a.shape.isEmpty) {
    return a.reshape(const [1]);
  }
  return a.reshape(a.shape);
}

/// Views [a] as an array with at least two dimensions.
GpuArray<T> atleast2d<T extends DTypeTag>(GpuArray<T> a) {
  _checkNotDisposed(a, 'a');
  if (a.shape.isEmpty) {
    return a.reshape(const [1, 1]);
  }
  if (a.shape.length == 1) {
    return a.reshape([1, a.shape[0]]);
  }
  return a.reshape(a.shape);
}

/// Views [a] as an array with at least three dimensions.
GpuArray<T> atleast3d<T extends DTypeTag>(GpuArray<T> a) {
  _checkNotDisposed(a, 'a');
  if (a.shape.isEmpty) {
    return a.reshape(const [1, 1, 1]);
  }
  if (a.shape.length == 1) {
    return a.reshape([1, a.shape[0], 1]);
  }
  if (a.shape.length == 2) {
    return a.reshape([a.shape[0], a.shape[1], 1]);
  }
  return a.reshape(a.shape);
}
