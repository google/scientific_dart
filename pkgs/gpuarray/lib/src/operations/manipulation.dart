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
import '../dtype.dart';
import '../gpu_array.dart';
import '../exceptions.dart';
import '../slice.dart';
import '../backend/compute_engine.dart';
import '../backend/kernels.dart';
import '../autograd/autograd.dart';

/// Joins a sequence of [arrays] along an existing [axis].
GpuArray<T> concatenate<T extends DTypeTag>(
  List<GpuArray> arrays, {
  int axis = 0,
}) {
  if (arrays.isEmpty) {
    throw ArgumentError('Cannot concatenate an empty list of arrays.');
  }
  final first = arrays[0];
  final rank = first.shape.length;
  final normAxis = axis < 0 ? axis + rank : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw GpuAxisOutOfBoundsException(axis, rank);
  }

  // Determine promoted dtype and verify matching shapes along other axes
  var outDType = first.dtype;
  var totalAxisLen = 0;

  for (final arr in arrays) {
    if (arr.shape.length != rank) {
      throw GpuShapeMismatchException('concatenate', arr.shape, first.shape);
    }
    for (var d = 0; d < rank; d++) {
      if (d != normAxis && arr.shape[d] != first.shape[d]) {
        throw GpuShapeMismatchException('concatenate', arr.shape, first.shape);
      }
    }
    totalAxisLen += arr.shape[normAxis];
    outDType = GpuArray.promoteDTypes(outDType, arr.dtype);
  }

  final outShape = List<int>.from(first.shape);
  outShape[normAxis] = totalAxisLen;
  final result = GpuArray<T>.empty(
    outShape,
    outDType as DType<T>,
    device: first.device,
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
GpuArray<T> stack<T extends DTypeTag>(List<GpuArray> arrays, {int axis = 0}) {
  if (arrays.isEmpty) {
    throw ArgumentError('Cannot stack an empty list of arrays.');
  }
  final expanded = arrays.map((a) => expand_dims(a, axis)).toList();
  return concatenate<T>(expanded, axis: axis);
}

/// Stacks arrays in sequence vertically (row wise / along axis 0).
GpuArray<T> vstack<T extends DTypeTag>(List<GpuArray> arrays) {
  if (arrays.isEmpty) {
    throw ArgumentError('Cannot vstack an empty list of arrays.');
  }
  final prepared = arrays.map((a) {
    if (a.shape.length == 1) {
      return a.reshape([1, a.shape[0]]);
    }
    return a;
  }).toList();
  return concatenate<T>(prepared, axis: 0);
}

/// Stacks arrays in sequence horizontally (column wise / along axis 1).
GpuArray<T> hstack<T extends DTypeTag>(List<GpuArray> arrays) {
  if (arrays.isEmpty) {
    throw ArgumentError('Cannot hstack an empty list of arrays.');
  }
  final prepared = arrays.map((a) {
    if (a.shape.length == 1) {
      return a.reshape([a.shape[0], 1]);
    }
    return a;
  }).toList();
  final axis = (arrays[0].shape.length == 1) ? 0 : 1;
  return concatenate<T>(prepared, axis: axis);
}

/// Stacks arrays in sequence depth wise (along axis 2).
GpuArray<T> dstack<T extends DTypeTag>(List<GpuArray> arrays) {
  if (arrays.isEmpty) {
    throw ArgumentError('Cannot dstack an empty list of arrays.');
  }
  final prepared = arrays.map((a) {
    if (a.shape.length == 1) {
      return a.reshape([1, a.shape[0], 1]);
    } else if (a.shape.length == 2) {
      return a.reshape([a.shape[0], a.shape[1], 1]);
    }
    return a;
  }).toList();
  return concatenate<T>(prepared, axis: 2);
}

/// Stacks 1D or 2D arrays as columns to create a 2D array.
GpuArray<T> column_stack<T extends DTypeTag>(List<GpuArray> arrays) {
  if (arrays.isEmpty) {
    throw ArgumentError('Cannot column_stack an empty list of arrays.');
  }
  final prepared = arrays.map((a) {
    if (a.shape.length == 1) {
      return a.reshape([a.shape[0], 1]);
    }
    return a;
  }).toList();
  return concatenate<T>(prepared, axis: 1);
}

/// Splits an array into multiple sub-arrays along [axis].
List<GpuArray<T>> split<T extends DTypeTag>(
  GpuArray<T> a,
  dynamic indicesOrSections, {
  int axis = 0,
}) {
  final rank = a.shape.length;
  final normAxis = axis < 0 ? axis + rank : axis;
  final dimLen = a.shape[normAxis];

  final splitPoints = <int>[0];

  if (indicesOrSections is int) {
    final sections = indicesOrSections;
    if (dimLen % sections != 0) {
      throw ArgumentError(
        'Array split does not result in an equal division: $dimLen is not divisible by $sections.',
      );
    }
    final step = dimLen ~/ sections;
    for (var i = 1; i < sections; i++) {
      splitPoints.add(i * step);
    }
  } else if (indicesOrSections is List<int>) {
    splitPoints.addAll(indicesOrSections);
  } else {
    throw ArgumentError(
      'indicesOrSections must be an integer (section count) or List<int> of split indices.',
    );
  }
  splitPoints.add(dimLen);

  final result = <GpuArray<T>>[];
  for (var i = 0; i < splitPoints.length - 1; i++) {
    final start = splitPoints[i];
    final stop = splitPoints[i + 1];

    final sliceSpecs = List<dynamic>.generate(rank, (d) {
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
List<GpuArray<T>> array_split<T extends DTypeTag>(
  GpuArray<T> a,
  dynamic indicesOrSections, {
  int axis = 0,
}) {
  if (indicesOrSections is! int) {
    return split<T>(a, indicesOrSections, axis: axis);
  }
  final rank = a.shape.length;
  final normAxis = axis < 0 ? axis + rank : axis;
  final dimLen = a.shape[normAxis];
  final n = indicesOrSections;

  final div = dimLen ~/ n;
  final mod = dimLen % n;

  final splitPoints = <int>[0];
  var current = 0;
  for (var i = 0; i < n; i++) {
    final size = i < mod ? div + 1 : div;
    current += size;
    if (i < n - 1) {
      splitPoints.add(current);
    }
  }
  splitPoints.add(dimLen);

  final result = <GpuArray<T>>[];
  for (var i = 0; i < splitPoints.length - 1; i++) {
    final start = splitPoints[i];
    final stop = splitPoints[i + 1];

    final sliceSpecs = List<dynamic>.generate(rank, (d) {
      if (d == normAxis) {
        return Slice(start, stop);
      }
      return const All();
    });

    result.add(a.slice(sliceSpecs));
  }
  return result;
}

/// Splits array horizontally (along axis 1).
List<GpuArray<T>> hsplit<T extends DTypeTag>(
  GpuArray<T> a,
  dynamic indicesOrSections,
) {
  final axis = a.shape.length == 1 ? 0 : 1;
  return split<T>(a, indicesOrSections, axis: axis);
}

/// Splits array vertically (along axis 0).
List<GpuArray<T>> vsplit<T extends DTypeTag>(
  GpuArray<T> a,
  dynamic indicesOrSections,
) {
  if (a.shape.length < 2) {
    throw ArgumentError('vsplit only works on arrays of 2 or more dimensions.');
  }
  return split<T>(a, indicesOrSections, axis: 0);
}

/// Splits array depth-wise (along axis 2).
List<GpuArray<T>> dsplit<T extends DTypeTag>(
  GpuArray<T> a,
  dynamic indicesOrSections,
) {
  if (a.shape.length < 3) {
    throw ArgumentError('dsplit only works on arrays of 3 or more dimensions.');
  }
  return split<T>(a, indicesOrSections, axis: 2);
}

/// Constructs an array by repeating [a] the number of times given by [reps].
GpuArray<T> tile<T extends DTypeTag>(GpuArray<T> a, List<int> reps) {
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
  final result = GpuArray<T>.empty(outShape, a.dtype, device: a.device);

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
}) {
  if (repeats < 0) {
    throw ArgumentError('repeats must be non-negative.');
  }
  if (axis == null) {
    final flat = a.flatten();
    final total = flat.shape[0];
    final outShape = [total * repeats];
    final result = GpuArray<T>.empty(outShape, a.dtype, device: a.device);

    for (var i = 0; i < total; i++) {
      final val = ComputeEngine.readAny(
        flat.buffer,
        flat.dtype,
        i,
        offsetElements: flat.offsetElements,
      );
      for (var r = 0; r < repeats; r++) {
        ComputeEngine.writeAny(result.buffer, a.dtype, i * repeats + r, val);
      }
    }

    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? axis + rank : axis;
  final outShape = List<int>.from(a.shape);
  outShape[normAxis] *= repeats;

  final result = GpuArray<T>.empty(outShape, a.dtype, device: a.device);

  final total = ShapeUtils.computeSize(a.shape);
  final coords = List<int>.filled(rank, 0);

  for (var i = 0; i < total; i++) {
    var srcIdx = 0;
    for (var d = 0; d < rank; d++) {
      srcIdx += coords[d] * a.strides[d];
    }
    final val = ComputeEngine.readAny(
      a.buffer,
      a.dtype,
      srcIdx,
      offsetElements: a.offsetElements,
    );

    for (var r = 0; r < repeats; r++) {
      var dstIdx = 0;
      for (var d = 0; d < rank; d++) {
        final coordDst = (d == normAxis) ? coords[d] * repeats + r : coords[d];
        dstIdx += coordDst * result.strides[d];
      }
      ComputeEngine.writeAny(result.buffer, a.dtype, dstIdx, val);
    }

    for (var d = rank - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < a.shape[d]) break;
      coords[d] = 0;
    }
  }

  return result;
}

/// Pads an array with [padWidth].
GpuArray<T> pad<T extends DTypeTag>(
  GpuArray<T> a,
  List<List<int>> padWidth, {
  String mode = 'constant',
  dynamic constantValues = 0,
}) {
  final rank = a.shape.length;
  if (padWidth.length != rank) {
    throw ArgumentError(
      'padWidth length (${padWidth.length}) must match tensor rank ($rank).',
    );
  }

  final outShape = List<int>.generate(
    rank,
    (d) => a.shape[d] + padWidth[d][0] + padWidth[d][1],
  );
  final result = GpuArray<T>.empty(outShape, a.dtype, device: a.device);

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
  );

  return result;
}

/// Roll array elements along a given [axis].
GpuArray<T> roll<T extends DTypeTag>(
  GpuArray<T> a,
  dynamic shift, {
  dynamic axis,
}) {
  if (axis == null) {
    final flat = a.flatten();
    final total = flat.shape[0];
    final s = (shift is int) ? shift : (shift as List<int>)[0];
    final sNorm = ((s % total) + total) % total;

    final rolledFlat = GpuArray<T>.empty([total], a.dtype, device: a.device);

    for (var i = 0; i < total; i++) {
      final srcIdx = (i - sNorm + total) % total;
      final val = ComputeEngine.readAny(
        flat.buffer,
        flat.dtype,
        srcIdx,
        offsetElements: flat.offsetElements,
      );
      ComputeEngine.writeAny(rolledFlat.buffer, a.dtype, i, val);
    }

    return rolledFlat.reshape(a.shape);
  }

  final rank = a.shape.length;
  final axes = (axis is int) ? [axis] : (axis as List<int>);
  final shifts = (shift is int) ? [shift] : (shift as List<int>);

  if (axes.length != shifts.length) {
    throw ArgumentError('shift and axis must have the same length.');
  }

  var current = a;
  for (var i = 0; i < axes.length; i++) {
    final ax = axes[i] < 0 ? axes[i] + rank : axes[i];
    final s = shifts[i];
    final dim = current.shape[ax];
    final sNorm = ((s % dim) + dim) % dim;

    final nextArr = GpuArray<T>.empty(current.shape, a.dtype, device: a.device);
    final total = ShapeUtils.computeSize(current.shape);
    final coords = List<int>.filled(rank, 0);

    for (var j = 0; j < total; j++) {
      var dstIdx = 0;
      var srcIdx = 0;
      for (var d = 0; d < rank; d++) {
        dstIdx += coords[d] * nextArr.strides[d];
        final srcCoord = (d == ax)
            ? (coords[d] - sNorm + dim) % dim
            : coords[d];
        srcIdx += srcCoord * current.strides[d];
      }

      final val = ComputeEngine.readAny(
        current.buffer,
        current.dtype,
        srcIdx,
        offsetElements: current.offsetElements,
      );
      ComputeEngine.writeAny(nextArr.buffer, a.dtype, dstIdx, val);

      for (var d = rank - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < current.shape[d]) break;
        coords[d] = 0;
      }
    }

    current = nextArr;
  }

  return current;
}

/// Reverses the order of elements along the given [axis].
GpuArray<T> flip<T extends DTypeTag>(GpuArray<T> a, {dynamic axis}) {
  final rank = a.shape.length;
  final List<int> axes;
  if (axis == null) {
    axes = List.generate(rank, (i) => i);
  } else if (axis is int) {
    axes = [axis < 0 ? axis + rank : axis];
  } else if (axis is List<int>) {
    axes = axis.map((ax) => ax < 0 ? ax + rank : ax).toList();
  } else {
    throw ArgumentError('axis must be int, List<int>, or null.');
  }

  final sliceSpecs = List<dynamic>.generate(rank, (d) {
    if (axes.contains(d)) {
      return const Slice(null, null, -1);
    }
    return const All();
  });

  return a.slice(sliceSpecs);
}

/// Rotates an array by 90 degrees in the plane specified by [axes].
GpuArray<T> rot90<T extends DTypeTag>(
  GpuArray<T> a, {
  int k = 1,
  List<int> axes = const [0, 1],
}) {
  final rank = a.shape.length;
  if (axes.length != 2) {
    throw ArgumentError('len(axes) must be 2.');
  }
  final ax1 = axes[0] < 0 ? axes[0] + rank : axes[0];
  final ax2 = axes[1] < 0 ? axes[1] + rank : axes[1];
  if (ax1 == ax2) {
    throw ArgumentError('axes must be different.');
  }

  final rot = ((k % 4) + 4) % 4;
  if (rot == 0) return a.copy();
  if (rot == 1) return flip(swapaxes(a, ax1, ax2), axis: ax1);
  if (rot == 2) return flip(flip(a, axis: ax1), axis: ax2);
  // rot == 3
  return flip(swapaxes(a, ax1, ax2), axis: ax2);
}

/// Extracts a diagonal or constructs a diagonal array.
GpuArray<T> diag<T extends DTypeTag>(GpuArray<T> v, {int k = 0}) {
  if (v.shape.length == 1) {
    final n = v.shape[0];
    final size = n + k.abs();
    final out = GpuArray<T>.zeros([size, size], v.dtype, device: v.device);
    final rowOffset = k < 0 ? -k : 0;
    final colOffset = k > 0 ? k : 0;

    for (var i = 0; i < n; i++) {
      final val = ComputeEngine.readAny(
        v.buffer,
        v.dtype,
        i,
        offsetElements: v.offsetElements,
      );
      final r = i + rowOffset;
      final c = i + colOffset;
      final dstIdx = r * size + c;
      ComputeEngine.writeAny(out.buffer, out.dtype, dstIdx, val);
    }
    return out;
  } else if (v.shape.length == 2) {
    return diagonal(v, offset: k);
  }
  throw ArgumentError('diag only works on 1D or 2D arrays.');
}

/// Returns specified diagonals of an array.
GpuArray<T> diagonal<T extends DTypeTag>(
  GpuArray<T> a, {
  int offset = 0,
  int axis1 = 0,
  int axis2 = 1,
}) {
  final rank = a.shape.length;
  final ax1 = axis1 < 0 ? axis1 + rank : axis1;
  final ax2 = axis2 < 0 ? axis2 + rank : axis2;

  final rows = a.shape[ax1];
  final cols = a.shape[ax2];

  int diagLen;
  if (offset >= 0) {
    diagLen = (rows < cols - offset) ? rows : (cols - offset);
  } else {
    diagLen = (rows + offset < cols) ? (rows + offset) : cols;
  }
  if (diagLen < 0) diagLen = 0;

  final result = GpuArray<T>.empty([diagLen], a.dtype, device: a.device);

  final startRow = offset < 0 ? -offset : 0;
  final startCol = offset > 0 ? offset : 0;

  for (var i = 0; i < diagLen; i++) {
    final r = startRow + i;
    final c = startCol + i;
    final srcIdx = r * a.strides[ax1] + c * a.strides[ax2];
    final val = ComputeEngine.readAny(
      a.buffer,
      a.dtype,
      srcIdx,
      offsetElements: a.offsetElements,
    );
    ComputeEngine.writeAny(result.buffer, a.dtype, i, val);
  }

  return result;
}

/// Returns the sum along diagonals of the array.
dynamic trace<T extends DTypeTag>(
  GpuArray<T> a, {
  int offset = 0,
  int axis1 = 0,
  int axis2 = 1,
}) {
  final d = diagonal(a, offset: offset, axis1: axis1, axis2: axis2);
  final s = d.sum();
  return s.shape.isEmpty ? s.item() : s;
}

/// Upper triangle of an array.
GpuArray<T> triu<T extends DTypeTag>(GpuArray<T> m, {int k = 0}) {
  final rank = m.shape.length;
  if (rank < 2) {
    throw ArgumentError('triu requires an array of at least 2 dimensions.');
  }

  final result = GpuArray<T>.empty(m.shape, m.dtype, device: m.device);

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

/// Lower triangle of an array.
GpuArray<T> tril<T extends DTypeTag>(GpuArray<T> m, {int k = 0}) {
  final rank = m.shape.length;
  if (rank < 2) {
    throw ArgumentError('tril requires an array of at least 2 dimensions.');
  }

  final result = GpuArray<T>.empty(m.shape, m.dtype, device: m.device);

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

/// Move axes of an array to new positions.
GpuArray<T> moveaxis<T extends DTypeTag>(
  GpuArray<T> a,
  dynamic source,
  dynamic destination,
) {
  final rank = a.shape.length;
  final srcList = (source is int) ? [source] : (source as List<int>);
  final dstList = (destination is int)
      ? [destination]
      : (destination as List<int>);

  if (srcList.length != dstList.length) {
    throw ArgumentError('source and destination must have the same length.');
  }

  final normSrc = srcList.map((x) => x < 0 ? x + rank : x).toList();
  final normDst = dstList.map((x) => x < 0 ? x + rank : x).toList();

  final order = <int>[];
  for (var i = 0; i < rank; i++) {
    if (!normSrc.contains(i)) order.add(i);
  }
  for (var i = 0; i < normDst.length; i++) {
    order.insert(normDst[i], normSrc[i]);
  }

  return a.transpose(order);
}

/// Interchange two axes of an array.
GpuArray<T> swapaxes<T extends DTypeTag>(GpuArray<T> a, int axis1, int axis2) {
  final rank = a.shape.length;
  final ax1 = axis1 < 0 ? axis1 + rank : axis1;
  final ax2 = axis2 < 0 ? axis2 + rank : axis2;

  final order = List<int>.generate(rank, (i) => i);
  order[ax1] = ax2;
  order[ax2] = ax1;

  return a.transpose(order);
}

/// Expand the shape of an array by inserting a new axis at [axis].
GpuArray<T> expand_dims<T extends DTypeTag>(GpuArray<T> a, dynamic axis) {
  final rank = a.shape.length;
  final axes = (axis is int) ? [axis] : (axis as List<int>);
  final outRank = rank + axes.length;

  final normAxes = axes.map((ax) => ax < 0 ? ax + outRank : ax).toList()
    ..sort();

  final newShape = List<int>.from(a.shape);
  for (final ax in normAxes) {
    newShape.insert(ax, 1);
  }

  return a.reshape(newShape);
}

/// Broadcast an array to a new shape.
GpuArray<T> broadcast_to<T extends DTypeTag>(GpuArray<T> a, List<int> shape) {
  final bStrides = ShapeUtils.broadcastStrides(a.shape, a.strides, shape);
  final isContig = ShapeUtils.isContiguous(shape, bStrides);

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
