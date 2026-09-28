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

import '../backend/compute_engine.dart';
import '../backend/kernels.dart';
import '../dtype.dart';
import '../exceptions.dart';
import '../gpu_array.dart';

void _checkArrayNotDisposed(GpuArray arr, String name) {
  if (arr.isDisposed || arr.buffer.isDisposed) {
    throw GpuDeviceDisposedException(
      'Cannot operate on disposed GpuArray ($name).',
    );
  }
}

/// Selects elements from [x] or [y] depending on [condition].
///
/// For finding the indices of non-zero elements in a single array, use
/// [nonzero], [flatnonzero], or [argwhere].
///
/// If [out] is provided, the result is written into [out] and returned.
GpuArray<T> where<T extends DTypeTag>(
  GpuArray condition,
  GpuArray x,
  GpuArray y, {
  GpuArray<T>? out,
}) {
  _checkArrayNotDisposed(condition, 'condition');
  _checkArrayNotDisposed(x, 'x');
  _checkArrayNotDisposed(y, 'y');

  final outShape = broadcastShapes(
    condition.shape,
    broadcastShapes(x.shape, y.shape),
  );

  final outDType = GpuArray.promoteDTypes(x.dtype, y.dtype) as DType<T>;
  final GpuArray<T> result;
  if (out != null) {
    _checkArrayNotDisposed(out, 'out');
    if (out.size > 1 && out.strides.contains(0)) {
      throw ArgumentError.value(
        out,
        'out',
        'Must be writeable and not a broadcasted view.',
      );
    }
    if (!areShapesEqual(out.shape, outShape)) {
      throw GpuShapeMismatchException('where(out)', out.shape, outShape);
    }
    if (out.dtype != outDType) {
      throw ArgumentError.value(
        out.dtype,
        'out.dtype',
        'Must match promoted output dtype $outDType.',
      );
    }
    result = out;
  } else {
    result = GpuArray<T>.empty(outShape, outDType, device: x.device);
  }

  GpuKernels.executeWhere(
    cond: condition.buffer,
    shapeCond: condition.shape,
    stridesCond: condition.strides,
    offsetCond: condition.offsetElements,
    srcX: x.buffer,
    shapeX: x.shape,
    stridesX: x.strides,
    offsetX: x.offsetElements,
    dtypeX: x.dtype,
    srcY: y.buffer,
    shapeY: y.shape,
    stridesY: y.strides,
    offsetY: y.offsetElements,
    dtypeY: y.dtype,
    dst: result.buffer,
    outShape: outShape,
    outStrides: result.strides,
    offsetDst: result.offsetElements,
    dtypeDst: outDType,
  );

  return result;
}

/// Constructs an array drawn from elements in [choicelist], depending on
/// conditions in [condlist].
///
/// If [out] is provided, the final result is written into [out] and returned.
GpuArray<T> select<T extends DTypeTag>(
  List<GpuArray> condlist,
  List<GpuArray> choicelist, {
  GpuArray? defaultValue,
  GpuArray? defaultArr,
  GpuArray<T>? out,
}) {
  if (condlist.isEmpty) {
    throw ArgumentError.value(condlist, 'condlist', 'Must not be empty.');
  }
  if (choicelist.isEmpty) {
    throw ArgumentError.value(choicelist, 'choicelist', 'Must not be empty.');
  }
  if (condlist.length != choicelist.length) {
    throw ArgumentError.value(
      choicelist.length,
      'choicelist.length',
      'Must match condlist.length (${condlist.length}).',
    );
  }

  var outShape = choicelist[0].shape;
  var outDType = choicelist[0].dtype;

  for (var i = 0; i < condlist.length; i++) {
    _checkArrayNotDisposed(condlist[i], 'condlist[$i]');
    _checkArrayNotDisposed(choicelist[i], 'choicelist[$i]');
    outShape = broadcastShapes(outShape, condlist[i].shape);
    outShape = broadcastShapes(outShape, choicelist[i].shape);
    outDType = GpuArray.promoteDTypes(outDType, choicelist[i].dtype);
  }

  final fallback = defaultValue ?? defaultArr;
  var current = (fallback != null)
      ? fallback.astype<T>(outDType as DType<T>)
      : GpuArray<T>.zeros(
          outShape,
          outDType as DType<T>,
          device: choicelist[0].device,
        );

  for (var i = condlist.length - 1; i >= 0; i--) {
    final cond = condlist[i];
    final choice = choicelist[i];
    final isLast = i == 0;
    final next = where<T>(cond, choice, current, out: isLast ? out : null);
    if (!identical(next, current)) {
      current.dispose();
    }
    current = next;
  }

  return current;
}

/// Extracts a 1D array of the elements of [arr] that satisfy the boolean
/// [condition].
GpuArray<T> extract<T extends DTypeTag>(GpuArray condition, GpuArray<T> arr) {
  _checkArrayNotDisposed(condition, 'condition');
  _checkArrayNotDisposed(arr, 'arr');

  final flatIndices = flatnonzero(condition);
  final count = flatIndices.shape[0];
  if (count == 0) {
    flatIndices.dispose();
    return GpuArray<T>.zeros([0], arr.dtype, device: arr.device);
  }

  final flatArr = arr.flatten();
  final result = GpuArray<T>.empty([count], arr.dtype, device: arr.device);

  try {
    final flatIndicesList = flatIndices.toList().cast<int>();
    for (var i = 0; i < count; i++) {
      final srcIndex = flatIndicesList[i];
      final val = readBufferAny(
        flatArr.buffer,
        flatArr.dtype,
        srcIndex,
        offsetElements: flatArr.offsetElements,
      );
      writeBufferAny(result.buffer, arr.dtype, i, val);
    }
  } finally {
    flatIndices.dispose();
    if (!identical(flatArr, arr)) {
      flatArr.dispose();
    }
  }

  return result;
}

/// Takes elements from [arr] along [axis] (or from the flattened array if
/// [axis] is `null`) at the given [indices].
GpuArray<T> take<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray indices, {
  int? axis,
  GpuArray<T>? out,
}) {
  _checkArrayNotDisposed(arr, 'arr');
  _checkArrayNotDisposed(indices, 'indices');

  final indexList = indices.toList().map((e) => (e as num).toInt()).toList();
  if (axis == null) {
    final flat = arr.flatten();
    try {
      final totalLen = flat.shape[0];
      final outShape = List<int>.unmodifiable(indices.shape);
      final GpuArray<T> result;
      if (out != null) {
        _checkArrayNotDisposed(out, 'out');
        if (out.size > 1 && out.strides.contains(0)) {
          throw ArgumentError.value(
            out,
            'out',
            'Must be writeable and not a broadcasted view.',
          );
        }
        if (!areShapesEqual(out.shape, outShape)) {
          throw GpuShapeMismatchException('take(out)', out.shape, outShape);
        }
        result = out;
      } else {
        result = GpuArray<T>.empty(outShape, arr.dtype, device: arr.device);
      }

      for (var i = 0; i < indexList.length; i++) {
        var k = indexList[i];
        if (k < 0) k += totalLen;
        if (k < 0 || k >= totalLen) {
          throw IndexError.withLength(k, totalLen, name: 'indices');
        }
        final srcElemOffset = k * flat.strides[0];
        final val = readBufferAny(
          flat.buffer,
          flat.dtype,
          srcElemOffset,
          offsetElements: flat.offsetElements,
        );
        writeBufferAny(
          result.buffer,
          result.dtype,
          i,
          val,
          offsetElements: result.offsetElements,
        );
      }
      return result;
    } finally {
      if (!identical(flat, arr)) {
        flat.dispose();
      }
    }
  }

  final rank = arr.shape.length;
  final normAxis = axis < 0 ? axis + rank : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw GpuAxisOutOfBoundsException(axis, rank);
  }

  final axisLen = arr.shape[normAxis];
  final outShape = <int>[
    ...arr.shape.sublist(0, normAxis),
    ...indices.shape,
    ...arr.shape.sublist(normAxis + 1),
  ];

  final GpuArray<T> result;
  if (out != null) {
    _checkArrayNotDisposed(out, 'out');
    if (out.size > 1 && out.strides.contains(0)) {
      throw ArgumentError.value(
        out,
        'out',
        'Must be writeable and not a broadcasted view.',
      );
    }
    if (!areShapesEqual(out.shape, outShape)) {
      throw GpuShapeMismatchException('take(out)', out.shape, outShape);
    }
    result = out;
  } else {
    result = GpuArray<T>.empty(outShape, arr.dtype, device: arr.device);
  }

  final totalOut = computeSize(outShape);
  if (totalOut == 0) return result;

  final indexRank = indices.shape.length;
  final indexStrides = computeCStrides(indices.shape);
  final outRank = outShape.length;
  final coords = List<int>.filled(outRank, 0);

  for (var i = 0; i < totalOut; i++) {
    var flatIndexPos = 0;
    for (var d = 0; d < indexRank; d++) {
      flatIndexPos += coords[normAxis + d] * indexStrides[d];
    }
    var k = indexList[flatIndexPos];
    if (k < 0) k += axisLen;
    if (k < 0 || k >= axisLen) {
      throw IndexError.withLength(k, axisLen, name: 'indices');
    }

    var srcElemOffset = 0;
    for (var d = 0; d < rank; d++) {
      final int coord;
      if (d < normAxis) {
        coord = coords[d];
      } else if (d == normAxis) {
        coord = k;
      } else {
        coord = coords[d - 1 + indexRank];
      }
      srcElemOffset += coord * arr.strides[d];
    }

    var dstElemOffset = 0;
    for (var d = 0; d < outRank; d++) {
      dstElemOffset += coords[d] * result.strides[d];
    }

    final val = readBufferAny(
      arr.buffer,
      arr.dtype,
      srcElemOffset,
      offsetElements: arr.offsetElements,
    );
    writeBufferAny(
      result.buffer,
      result.dtype,
      dstElemOffset,
      val,
      offsetElements: result.offsetElements,
    );

    for (var d = outRank - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < outShape[d]) break;
      coords[d] = 0;
    }
  }

  return result;
}

/// Replaces specified elements of [arr] with [values] using flat 1D [indices].
void put<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray indices,
  GpuArray values,
) {
  _checkArrayNotDisposed(arr, 'arr');
  _checkArrayNotDisposed(indices, 'indices');
  _checkArrayNotDisposed(values, 'values');

  final totalLen = arr.size;
  final indexList = indices.toList().map((e) => (e as num).toInt()).toList();
  final valList = values.toList();
  if (valList.isEmpty && indexList.isNotEmpty) {
    throw ArgumentError.value(values, 'values', 'Must not be empty.');
  }

  final rank = arr.shape.length;
  final cStrides = computeCStrides(arr.shape);

  for (var i = 0; i < indexList.length; i++) {
    var k = indexList[i];
    if (k < 0) k += totalLen;
    if (k < 0 || k >= totalLen) {
      throw IndexError.withLength(k, totalLen, name: 'indices');
    }

    var rem = k;
    var dstElemOffset = 0;
    for (var d = 0; d < rank; d++) {
      final coord = rem ~/ cStrides[d];
      rem = rem % cStrides[d];
      dstElemOffset += coord * arr.strides[d];
    }

    final val = valList[i % valList.length];
    writeBufferAny(
      arr.buffer,
      arr.dtype,
      dstElemOffset,
      val,
      offsetElements: arr.offsetElements,
    );
  }
}

/// Takes values from [arr] along [axis] at specified 1D or multi-dimensional
/// [indices].
GpuArray<T> takeAlongAxis<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray indices,
  int axis, {
  GpuArray<T>? out,
}) {
  _checkArrayNotDisposed(arr, 'arr');
  _checkArrayNotDisposed(indices, 'indices');

  final rank = arr.shape.length;
  final normAxis = axis < 0 ? axis + rank : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw GpuAxisOutOfBoundsException(axis, rank);
  }

  final outShape = indices.shape;
  final GpuArray<T> result;
  if (out != null) {
    _checkArrayNotDisposed(out, 'out');
    if (out.size > 1 && out.strides.contains(0)) {
      throw ArgumentError.value(
        out,
        'out',
        'Must be writeable and not a broadcasted view.',
      );
    }
    if (!areShapesEqual(out.shape, outShape)) {
      throw GpuShapeMismatchException(
        'takeAlongAxis(out)',
        out.shape,
        outShape,
      );
    }
    result = out;
  } else {
    result = GpuArray<T>.empty(outShape, arr.dtype, device: arr.device);
  }

  GpuKernels.executeTakeAlongAxis(
    src: arr.buffer,
    shapeSrc: arr.shape,
    stridesSrc: arr.strides,
    offsetSrc: arr.offsetElements,
    dtypeSrc: arr.dtype,
    indices: indices.buffer,
    shapeIndices: indices.shape,
    stridesIndices: indices.strides,
    offsetIndices: indices.offsetElements,
    dtypeIndices: indices.dtype,
    dst: result.buffer,
    outShape: outShape,
    outStrides: result.strides,
    offsetDst: result.offsetElements,
    dtypeDst: arr.dtype,
    axis: normAxis,
  );

  return result;
}

/// Takes values from [arr] along [axis] at specified [indices].
@Deprecated('Use takeAlongAxis instead.')
// ignore: non_constant_identifier_names
GpuArray<T> take_along_axis<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray indices,
  int axis, {
  GpuArray<T>? out,
}) => takeAlongAxis<T>(arr, indices, axis, out: out);

/// Puts [values] into [arr] along [axis] at positions specified by [indices].
void putAlongAxis<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray indices,
  GpuArray values,
  int axis,
) {
  _checkArrayNotDisposed(arr, 'arr');
  _checkArrayNotDisposed(indices, 'indices');
  _checkArrayNotDisposed(values, 'values');

  final rank = arr.shape.length;
  final normAxis = axis < 0 ? axis + rank : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw GpuAxisOutOfBoundsException(axis, rank);
  }

  GpuKernels.executePutAlongAxis(
    arr: arr.buffer,
    shapeArr: arr.shape,
    stridesArr: arr.strides,
    offsetArr: arr.offsetElements,
    dtypeArr: arr.dtype,
    indices: indices.buffer,
    shapeIndices: indices.shape,
    stridesIndices: indices.strides,
    offsetIndices: indices.offsetElements,
    dtypeIndices: indices.dtype,
    values: values.buffer,
    shapeVal: values.shape,
    stridesVal: values.strides,
    offsetVal: values.offsetElements,
    dtypeVal: values.dtype,
    axis: normAxis,
  );
}

/// Puts [values] into [arr] along [axis] at positions specified by [indices].
@Deprecated('Use putAlongAxis instead.')
// ignore: non_constant_identifier_names
void put_along_axis<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray indices,
  GpuArray values,
  int axis,
) => putAlongAxis<T>(arr, indices, values, axis);

bool _isNonZero(Object? val) {
  if (val is Complex) {
    return val.real != 0.0 || val.imag != 0.0;
  }
  if (val is bool) {
    return val;
  }
  if (val is num) {
    return val != 0;
  }
  return false;
}

/// Finds the indices of non-zero elements as a list of 1D arrays, one per
/// dimension.
List<GpuArray<Int32>> nonzero(GpuArray arr) {
  _checkArrayNotDisposed(arr, 'arr');
  final rank = arr.shape.length;
  final total = computeSize(arr.shape);
  final coords = List<int>.filled(rank, 0);

  final matchingCoords = List.generate(rank, (_) => <int>[]);

  for (var i = 0; i < total; i++) {
    var elemOffset = 0;
    for (var d = 0; d < rank; d++) {
      elemOffset += coords[d] * arr.strides[d];
    }

    final val = readBufferAny(
      arr.buffer,
      arr.dtype,
      elemOffset,
      offsetElements: arr.offsetElements,
    );

    if (_isNonZero(val)) {
      for (var d = 0; d < rank; d++) {
        matchingCoords[d].add(coords[d]);
      }
    }

    for (var d = rank - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < arr.shape[d]) break;
      coords[d] = 0;
    }
  }

  return matchingCoords.map((coordList) {
    return GpuArray<Int32>.fromList(
      coordList,
      [coordList.length],
      DType.int32,
      device: arr.device,
    );
  }).toList();
}

/// Finds indices that are non-zero in the flattened version of [arr].
GpuArray<Int32> flatnonzero(GpuArray arr) {
  _checkArrayNotDisposed(arr, 'arr');
  final flat = arr.flatten();
  try {
    final total = flat.shape[0];
    final matching = <int>[];

    for (var i = 0; i < total; i++) {
      final val = readBufferAny(
        flat.buffer,
        flat.dtype,
        i * flat.strides[0],
        offsetElements: flat.offsetElements,
      );
      if (_isNonZero(val)) {
        matching.add(i);
      }
    }

    return GpuArray<Int32>.fromList(
      matching,
      [matching.length],
      DType.int32,
      device: arr.device,
    );
  } finally {
    if (!identical(flat, arr)) {
      flat.dispose();
    }
  }
}

/// Finds the indices of non-zero elements as a 2D array of shape `(N, rank)`.
GpuArray<Int32> argwhere(GpuArray arr) {
  _checkArrayNotDisposed(arr, 'arr');
  final nz = nonzero(arr);
  try {
    if (nz.isEmpty || nz[0].shape[0] == 0) {
      return GpuArray<Int32>.zeros(
        [0, arr.shape.length],
        DType.int32,
        device: arr.device,
      );
    }

    final count = nz[0].shape[0];
    final rank = arr.shape.length;
    final list2D = <int>[];

    final dimLists = nz.map((a) => a.toList().cast<int>()).toList();
    for (var i = 0; i < count; i++) {
      for (var d = 0; d < rank; d++) {
        list2D.add(dimLists[d][i]);
      }
    }

    return GpuArray<Int32>.fromList(
      list2D,
      [count, rank],
      DType.int32,
      device: arr.device,
    );
  } finally {
    for (final coordArr in nz) {
      coordArr.dispose();
    }
  }
}
