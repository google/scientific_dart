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

import '../backend/compute_engine.dart';
import '../backend/kernels.dart';
import '../buffer.dart';
import '../device.dart';
import '../dtype.dart';
import '../exceptions.dart';
import '../gpu_array.dart';
import 'manipulation.dart';

void _checkArrayNotDisposed(GpuArray<DTypeTag> arr, String name) {
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
GpuArray<T> where<T extends AnySpec>(
  GpuArray<DTypeTag> condition,
  GpuArray<DTypeTag> x,
  GpuArray<DTypeTag> y, {
  GpuArray<T>? out,
}) {
  _checkArrayNotDisposed(condition, 'condition');
  _checkArrayNotDisposed(x, 'x');
  _checkArrayNotDisposed(y, 'y');
  if (condition.device != x.device || y.device != x.device) {
    throw ArgumentError.value(
      y.device,
      'device',
      'Operands must be on the same WebGPU device.',
    );
  }

  final outShape = broadcastShapes(
    condition.shape,
    broadcastShapes(x.shape, y.shape),
  );

  final outDType = GpuArray.promoteDTypes(x.dtype, y.dtype) as DType<T>;
  final GpuArray<T> result;
  if (out != null) {
    _checkArrayNotDisposed(out, 'out');
    if (out.device != x.device) {
      throw ArgumentError.value(
        out.device,
        'out.device',
        'Must be on the same WebGPU device as inputs.',
      );
    }
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
        'Must match expected output shape $outShape.',
      );
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

  final condBool = condition.dtype == DType.boolean
      ? condition
      : condition.astype(DType.boolean);
  try {
    GpuKernels.executeWhere(
      cond: condBool.buffer,
      shapeCond: condBool.shape,
      stridesCond: condBool.strides,
      offsetCond: condBool.offsetElements,
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
  } finally {
    if (!identical(condBool, condition)) {
      condBool.dispose();
    }
  }

  return result;
}

/// Constructs an array drawn from elements in [choicelist], depending on
/// conditions in [condlist].
///
/// If [out] is provided, the final result is written into [out] and returned.
GpuArray<T> select<T extends AnySpec>(
  List<GpuArray<DTypeTag>> condlist,
  List<GpuArray<DTypeTag>> choicelist, {
  GpuArray<DTypeTag>? defaultValue,
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

  var current = (defaultValue != null)
      ? defaultValue.astype(outDType as DType<T>)
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
    if (!identical(current, defaultValue) && !identical(next, current)) {
      current.dispose();
    }
    current = next;
  }

  return current;
}

/// Extracts a 1D array of the elements of [arr] that satisfy the boolean
/// [condition].
GpuArray<T> extract<T extends DTypeTag>(
  GpuArray<DTypeTag> condition,
  GpuArray<T> arr,
) {
  _checkArrayNotDisposed(condition, 'condition');
  _checkArrayNotDisposed(arr, 'arr');
  if (condition.device != arr.device) {
    throw ArgumentError.value(
      condition.device,
      'condition.device',
      'Must reside on the same GpuDevice (${arr.device}) as arr.',
    );
  }

  final flatIndices = flatnonzero(condition);
  try {
    final count = flatIndices.shape[0];
    if (count == 0) {
      return GpuArray<T>.zeros([0], arr.dtype, device: arr.device);
    }
    final flatArr = arr.flatten();
    try {
      return take(flatArr, flatIndices);
    } finally {
      if (!identical(flatArr, arr)) {
        flatArr.dispose();
      }
    }
  } finally {
    flatIndices.dispose();
  }
}

/// Extracts elements of [arr] matching the boolean mask [mask] (alias for [extract]).
GpuArray<T> booleanMask<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray<DTypeTag> mask,
) => extract<T>(mask, arr);

/// Takes elements from [arr] along [axis] (or from the flattened array if
/// [axis] is `null`) at the given [indices].
GpuArray<T> take<T extends DTypeTag, Out extends T>(
  GpuArray<T> arr,
  GpuArray<DTypeTag> indices, {
  int? axis,
  GpuArray<Out>? out,
}) {
  _checkArrayNotDisposed(arr, 'arr');
  _checkArrayNotDisposed(indices, 'indices');
  if (indices.device != arr.device) {
    throw ArgumentError.value(
      indices.device,
      'indices.device',
      'Must reside on the same GpuDevice (${arr.device}) as arr.',
    );
  }
  if (!indices.dtype.isInteger) {
    throw ArgumentError.value(
      indices.dtype,
      'indices.dtype',
      'Must be an integer DType.',
    );
  }

  if (axis == null) {
    final outShape = List<int>.unmodifiable(indices.shape);
    final GpuArray<T> result;
    if (out != null) {
      _checkArrayNotDisposed(out, 'out');
      if (out.device != arr.device) {
        throw ArgumentError.value(
          out.device,
          'out.device',
          'Must reside on the same GpuDevice (${arr.device}) as arr.',
        );
      }
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
          'Must match expected output shape $outShape.',
        );
      }
      if (out.dtype != arr.dtype) {
        throw ArgumentError.value(
          out.dtype,
          'out.dtype',
          'Must match arr.dtype (${arr.dtype}).',
        );
      }
      result = out;
    } else {
      result = GpuArray<T>.empty(outShape, arr.dtype, device: arr.device);
    }

    if (indices.size == 0) return result;
    if (arr.size == 0) {
      throw IndexError.withLength(0, 0, name: 'indices');
    }

    GpuKernels.executeTake(
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
      axis: null,
    );
    return result;
  }

  final rank = arr.shape.length;
  final normAxis = axis < 0 ? axis + rank : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw GpuAxisOutOfBoundsException(axis, rank);
  }

  final axisLength = arr.shape[normAxis];
  final outShape = <int>[
    ...arr.shape.sublist(0, normAxis),
    ...indices.shape,
    ...arr.shape.sublist(normAxis + 1),
  ];

  final GpuArray<T> result;
  if (out != null) {
    _checkArrayNotDisposed(out, 'out');
    if (out.device != arr.device) {
      throw ArgumentError.value(
        out.device,
        'out.device',
        'Must reside on the same GpuDevice (${arr.device}) as arr.',
      );
    }
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
        'Must match expected output shape $outShape.',
      );
    }
    if (out.dtype != arr.dtype) {
      throw ArgumentError.value(
        out.dtype,
        'out.dtype',
        'Must match arr.dtype (${arr.dtype}).',
      );
    }
    result = out;
  } else {
    result = GpuArray<T>.empty(outShape, arr.dtype, device: arr.device);
  }

  final totalOut = computeSize(outShape);
  if (totalOut == 0) return result;
  if (axisLength == 0) {
    throw IndexError.withLength(0, 0, name: 'indices');
  }

  GpuKernels.executeTake(
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

/// Gathers elements from [arr] along [axis] (alias for [take]).
GpuArray<T> gather<T extends DTypeTag, Out extends T>(
  GpuArray<T> arr,
  GpuArray<DTypeTag> indices, {
  int? axis,
  GpuArray<Out>? out,
}) => take(arr, indices, axis: axis, out: out);

/// Replaces specified elements of [arr] with [values] using flat 1D [indices].
void put<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray<DTypeTag> indices,
  GpuArray<DTypeTag> values,
) {
  _checkArrayNotDisposed(arr, 'arr');
  _checkArrayNotDisposed(indices, 'indices');
  _checkArrayNotDisposed(values, 'values');
  if (indices.device != arr.device) {
    throw ArgumentError.value(
      indices.device,
      'indices.device',
      'Must reside on the same GpuDevice (${arr.device}) as arr.',
    );
  }
  if (values.device != arr.device) {
    throw ArgumentError.value(
      values.device,
      'values.device',
      'Must reside on the same GpuDevice (${arr.device}) as arr.',
    );
  }
  if (!indices.dtype.isInteger) {
    throw ArgumentError.value(
      indices.dtype,
      'indices.dtype',
      'Must be an integer DType.',
    );
  }
  if (values.size == 0 && indices.size > 0) {
    throw ArgumentError.value(values, 'values', 'Must not be empty.');
  }
  if (indices.size == 0) return;
  if (arr.size == 0) {
    throw IndexError.withLength(0, 0, name: 'indices');
  }

  GpuKernels.executePut(
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
  );
}

/// Scatters [values] into [arr] at flat [indices] (alias for [put]).
void scatter<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray<DTypeTag> indices,
  GpuArray<DTypeTag> values,
) => put<T>(arr, indices, values);

/// Takes values from [arr] along [axis] at specified 1D or multi-dimensional
/// [indices].
GpuArray<T> takeAlongAxis<T extends DTypeTag, Out extends T>(
  GpuArray<T> arr,
  GpuArray<DTypeTag> indices,
  int axis, {
  GpuArray<Out>? out,
}) {
  _checkArrayNotDisposed(arr, 'arr');
  _checkArrayNotDisposed(indices, 'indices');
  if (indices.device != arr.device) {
    throw ArgumentError.value(
      indices.device,
      'indices.device',
      'Must reside on the same GpuDevice (${arr.device}) as arr.',
    );
  }
  if (!indices.dtype.isInteger) {
    throw ArgumentError.value(
      indices.dtype,
      'indices.dtype',
      'Must be an integer DType.',
    );
  }

  final rank = arr.shape.length;
  if (indices.shape.length != rank) {
    throw ArgumentError.value(
      indices.shape,
      'indices.shape',
      'Must have the same rank ($rank) as arr (${arr.shape}).',
    );
  }
  final normAxis = axis < 0 ? axis + rank : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw GpuAxisOutOfBoundsException(axis, rank);
  }
  final outShape = List<int>.filled(rank, 0);
  final bStridesSrc = List<int>.filled(rank, 0);
  final bStridesIndices = List<int>.filled(rank, 0);
  for (var d = 0; d < rank; d++) {
    if (d == normAxis) {
      outShape[d] = indices.shape[d];
      bStridesSrc[d] = arr.strides[d];
      bStridesIndices[d] = indices.strides[d];
    } else if (indices.shape[d] == arr.shape[d]) {
      outShape[d] = indices.shape[d];
      bStridesSrc[d] = arr.strides[d];
      bStridesIndices[d] = indices.strides[d];
    } else if (indices.shape[d] == 1) {
      outShape[d] = arr.shape[d];
      bStridesSrc[d] = arr.strides[d];
      bStridesIndices[d] = 0;
    } else if (arr.shape[d] == 1) {
      outShape[d] = indices.shape[d];
      bStridesSrc[d] = 0;
      bStridesIndices[d] = indices.strides[d];
    } else {
      throw GpuShapeMismatchException(
        'takeAlongAxis',
        arr.shape,
        indices.shape,
      );
    }
  }

  final GpuArray<T> result;
  if (out != null) {
    _checkArrayNotDisposed(out, 'out');
    if (out.device != arr.device) {
      throw ArgumentError.value(
        out.device,
        'out.device',
        'Must reside on the same GpuDevice (${arr.device}) as arr.',
      );
    }
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
        'Must match expected output shape $outShape.',
      );
    }
    if (out.dtype != arr.dtype) {
      throw ArgumentError.value(
        out.dtype,
        'out.dtype',
        'Must match arr.dtype (${arr.dtype}).',
      );
    }
    result = out;
  } else {
    result = GpuArray<T>.empty(outShape, arr.dtype, device: arr.device);
  }

  GpuKernels.executeTakeAlongAxis(
    src: arr.buffer,
    shapeSrc: arr.shape,
    stridesSrc: bStridesSrc,
    offsetSrc: arr.offsetElements,
    dtypeSrc: arr.dtype,
    indices: indices.buffer,
    shapeIndices: outShape,
    stridesIndices: bStridesIndices,
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

/// Puts [values] into [arr] along [axis] at positions specified by [indices].
void putAlongAxis<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray<DTypeTag> indices,
  GpuArray<DTypeTag> values,
  int axis,
) {
  _checkArrayNotDisposed(arr, 'arr');
  _checkArrayNotDisposed(indices, 'indices');
  _checkArrayNotDisposed(values, 'values');
  if (indices.device != arr.device) {
    throw ArgumentError.value(
      indices.device,
      'indices.device',
      'Must reside on the same GpuDevice (${arr.device}) as arr.',
    );
  }
  if (values.device != arr.device) {
    throw ArgumentError.value(
      values.device,
      'values.device',
      'Must reside on the same GpuDevice (${arr.device}) as arr.',
    );
  }
  if (!indices.dtype.isInteger) {
    throw ArgumentError.value(
      indices.dtype,
      'indices.dtype',
      'Must be an integer DType.',
    );
  }

  final rank = arr.shape.length;
  if (indices.shape.length != rank) {
    throw ArgumentError.value(
      indices.shape,
      'indices.shape',
      'Must have the same rank ($rank) as arr (${arr.shape}).',
    );
  }
  final normAxis = axis < 0 ? axis + rank : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw GpuAxisOutOfBoundsException(axis, rank);
  }
  final effShapeIndices = List<int>.filled(rank, 0);
  final bStridesIndices = List<int>.filled(rank, 0);
  for (var d = 0; d < rank; d++) {
    if (d == normAxis) {
      effShapeIndices[d] = indices.shape[d];
      bStridesIndices[d] = indices.strides[d];
    } else if (indices.shape[d] == arr.shape[d]) {
      effShapeIndices[d] = arr.shape[d];
      bStridesIndices[d] = indices.strides[d];
    } else if (indices.shape[d] == 1) {
      effShapeIndices[d] = arr.shape[d];
      bStridesIndices[d] = 0;
    } else {
      throw GpuShapeMismatchException('putAlongAxis', arr.shape, indices.shape);
    }
  }

  GpuKernels.executePutAlongAxis(
    arr: arr.buffer,
    shapeArr: arr.shape,
    stridesArr: arr.strides,
    offsetArr: arr.offsetElements,
    dtypeArr: arr.dtype,
    indices: indices.buffer,
    shapeIndices: effShapeIndices,
    stridesIndices: bStridesIndices,
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

/// Finds the indices of non-zero elements as a list of 1D [Int64] arrays, one
/// per dimension.
List<GpuArray<Int64>> nonzero(GpuArray<DTypeTag> arr) {
  _checkArrayNotDisposed(arr, 'arr');
  final rank = arr.shape.length;
  if (rank == 0) {
    final flat = arr.reshape([1]);
    try {
      return nonzero(flat);
    } finally {
      flat.dispose();
    }
  }

  final (prefixBuffer, count) = GpuKernels.executeNonZeroScan(
    src: arr.buffer,
    shapeSrc: arr.shape,
    stridesSrc: arr.strides,
    offsetSrc: arr.offsetElements,
    dtypeSrc: arr.dtype,
  );
  try {
    final results = <GpuArray<Int64>>[];
    for (var d = 0; d < rank; d++) {
      final axisOut = GpuArray<Int64>.empty(
        [count],
        DType.int64,
        device: arr.device,
      );
      if (count > 0) {
        GpuKernels.executeNonZeroScatter(
          mode: 'nonzero_axis',
          cond: arr.buffer,
          shapeCond: arr.shape,
          stridesCond: arr.strides,
          offsetCond: arr.offsetElements,
          dtypeCond: arr.dtype,
          prefixBuffer: prefixBuffer,
          src: arr.buffer,
          stridesSrc: arr.strides,
          offsetSrc: arr.offsetElements,
          dtypeSrc: arr.dtype,
          dst: axisOut.buffer,
          offsetDst: 0,
          dtypeDst: DType.int64,
          targetAxis: d,
        );
      }
      results.add(axisOut);
    }
    return results;
  } finally {
    prefixBuffer.dispose();
  }
}

/// Finds indices that are non-zero in the flattened version of [arr] as
/// [GpuArray<Int64>].
GpuArray<Int64> flatnonzero(GpuArray<DTypeTag> arr) {
  _checkArrayNotDisposed(arr, 'arr');
  final flat = arr.flatten();
  try {
    final (prefixBuffer, count) = GpuKernels.executeNonZeroScan(
      src: flat.buffer,
      shapeSrc: flat.shape,
      stridesSrc: flat.strides,
      offsetSrc: flat.offsetElements,
      dtypeSrc: flat.dtype,
    );
    try {
      final result = GpuArray<Int64>.empty(
        [count],
        DType.int64,
        device: arr.device,
      );
      if (count > 0) {
        GpuKernels.executeNonZeroScatter(
          mode: 'flatnonzero',
          cond: flat.buffer,
          shapeCond: flat.shape,
          stridesCond: flat.strides,
          offsetCond: flat.offsetElements,
          dtypeCond: flat.dtype,
          prefixBuffer: prefixBuffer,
          src: flat.buffer,
          stridesSrc: flat.strides,
          offsetSrc: flat.offsetElements,
          dtypeSrc: flat.dtype,
          dst: result.buffer,
          offsetDst: 0,
          dtypeDst: DType.int64,
        );
      }
      return result;
    } finally {
      prefixBuffer.dispose();
    }
  } finally {
    if (!identical(flat, arr)) {
      flat.dispose();
    }
  }
}

/// Finds the indices of non-zero elements as a 2D [Int64] array of shape
/// `(N, rank)`.
GpuArray<Int64> argwhere(GpuArray<DTypeTag> arr) {
  _checkArrayNotDisposed(arr, 'arr');
  final rank = arr.shape.length;
  final (prefixBuffer, count) = GpuKernels.executeNonZeroScan(
    src: arr.buffer,
    shapeSrc: arr.shape,
    stridesSrc: arr.strides,
    offsetSrc: arr.offsetElements,
    dtypeSrc: arr.dtype,
  );
  try {
    final result = GpuArray<Int64>.empty(
      [count, rank],
      DType.int64,
      device: arr.device,
    );
    if (count > 0 && rank > 0) {
      GpuKernels.executeNonZeroScatter(
        mode: 'argwhere',
        cond: arr.buffer,
        shapeCond: arr.shape,
        stridesCond: arr.strides,
        offsetCond: arr.offsetElements,
        dtypeCond: arr.dtype,
        prefixBuffer: prefixBuffer,
        src: arr.buffer,
        stridesSrc: arr.strides,
        offsetSrc: arr.offsetElements,
        dtypeSrc: arr.dtype,
        dst: result.buffer,
        offsetDst: 0,
        dtypeDst: DType.int64,
      );
    }
    return result;
  } finally {
    prefixBuffer.dispose();
  }
}

int _normalizeRequiredAxis(int axis, int rank) {
  final normAxis = axis < 0 ? axis + rank : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw GpuAxisOutOfBoundsException(axis, rank);
  }
  return normAxis;
}

GpuArray<T> _prepareOutputArray<T extends DTypeTag>(
  GpuArray<DTypeTag> source,
  List<int> outShape,
  DType<T> outDType,
  GpuArray<T>? out,
  String outName,
) {
  if (out == null) {
    return GpuArray<T>.empty(outShape, outDType, device: source.device);
  }
  _checkArrayNotDisposed(out, outName);
  if (out.device != source.device) {
    throw ArgumentError.value(
      out.device,
      '$outName.device',
      'Must reside on the same GpuDevice (${source.device}) as input.',
    );
  }
  if (out.size > 1 && out.strides.contains(0)) {
    throw ArgumentError.value(
      out,
      outName,
      'Must be writeable and not a broadcasted view.',
    );
  }
  if (!areShapesEqual(out.shape, outShape)) {
    throw ArgumentError.value(
      out.shape,
      '$outName.shape',
      'Must match expected output shape $outShape.',
    );
  }
  if (out.dtype != outDType) {
    throw ArgumentError.value(
      out.dtype,
      '$outName.dtype',
      'Must match expected output dtype $outDType.',
    );
  }
  return out;
}

/// Sorts elements of [a] in ascending order along [axis].
///
/// When [axis] is `null`, the flattened array is sorted into a 1-D array.
/// If [out] is provided, the sorted values are written into [out] and
/// returned.
GpuArray<T> sort<T extends DTypeTag, Out extends T>(
  GpuArray<T> a, {
  int? axis = -1,
  SortKind kind = SortKind.quicksort,
  GpuArray<Out>? out,
}) {
  _checkArrayNotDisposed(a, 'a');
  if (axis == null) {
    final flat = a.flatten();
    try {
      final outShape = flat.shape;
      final result = _prepareOutputArray<T>(a, outShape, a.dtype, out, 'out');
      if (flat.size == 0) return result;
      GpuKernels.executeAxisSort(
        src: flat.buffer,
        shapeSrc: flat.shape,
        stridesSrc: flat.strides,
        offsetSrc: flat.offsetElements,
        dtypeSrc: flat.dtype,
        axis: 0,
        outAxisSize: flat.size,
        descending: false,
        dstValues: result.buffer,
        outValShape: result.shape,
        outValStrides: result.strides,
        offsetDstValues: result.offsetElements,
      );
      return result;
    } finally {
      if (!identical(flat, a)) {
        flat.dispose();
      }
    }
  }

  final normAxis = _normalizeRequiredAxis(axis, a.ndim);
  final result = _prepareOutputArray<T>(a, a.shape, a.dtype, out, 'out');
  if (a.size == 0) return result;

  GpuKernels.executeAxisSort(
    src: a.buffer,
    shapeSrc: a.shape,
    stridesSrc: a.strides,
    offsetSrc: a.offsetElements,
    dtypeSrc: a.dtype,
    axis: normAxis,
    outAxisSize: a.shape[normAxis],
    descending: false,
    dstValues: result.buffer,
    outValShape: result.shape,
    outValStrides: result.strides,
    offsetDstValues: result.offsetElements,
  );
  return result;
}

/// Computes the 64-bit integer indices that would sort [a] along [axis].
///
/// When [axis] is `null`, the indices into the flattened array are produced.
/// If [out] is provided, the sorted indices are written into [out] and
/// returned.
GpuArray<Int64> argsort<T extends DTypeTag>(
  GpuArray<T> a, {
  int? axis = -1,
  SortKind kind = SortKind.quicksort,
  GpuArray<Int64>? out,
}) {
  _checkArrayNotDisposed(a, 'a');
  if (axis == null) {
    final flat = a.flatten();
    try {
      final outShape = flat.shape;
      final result = _prepareOutputArray<Int64>(
        a,
        outShape,
        DType.int64,
        out,
        'out',
      );
      if (flat.size == 0) return result;
      GpuKernels.executeAxisSort(
        src: flat.buffer,
        shapeSrc: flat.shape,
        stridesSrc: flat.strides,
        offsetSrc: flat.offsetElements,
        dtypeSrc: flat.dtype,
        axis: 0,
        outAxisSize: flat.size,
        descending: false,
        dstIndices: result.buffer,
        outIndicesShape: result.shape,
        outIndicesStrides: result.strides,
        offsetDstIndices: result.offsetElements,
      );
      return result;
    } finally {
      if (!identical(flat, a)) {
        flat.dispose();
      }
    }
  }

  final normAxis = _normalizeRequiredAxis(axis, a.ndim);
  final result = _prepareOutputArray<Int64>(
    a,
    a.shape,
    DType.int64,
    out,
    'out',
  );
  if (a.size == 0) return result;

  GpuKernels.executeAxisSort(
    src: a.buffer,
    shapeSrc: a.shape,
    stridesSrc: a.strides,
    offsetSrc: a.offsetElements,
    dtypeSrc: a.dtype,
    axis: normAxis,
    outAxisSize: a.shape[normAxis],
    descending: false,
    dstIndices: result.buffer,
    outIndicesShape: result.shape,
    outIndicesStrides: result.strides,
    offsetDstIndices: result.offsetElements,
  );
  return result;
}

/// Finds the [k] largest or smallest elements and their 64-bit indices
/// along [axis].
///
/// When [largest] is `true`, the [k] largest elements are selected in
/// descending order; otherwise the [k] smallest elements are selected in
/// ascending order.
({GpuArray<T> values, GpuArray<Int64> indices})
topk<T extends DTypeTag, OutValues extends T>(
  GpuArray<T> a,
  int k, {
  int axis = -1,
  bool largest = true,
  bool sorted = true,
  GpuArray<OutValues>? outValues,
  GpuArray<Int64>? outIndices,
}) {
  _checkArrayNotDisposed(a, 'a');
  final normAxis = _normalizeRequiredAxis(axis, a.ndim);
  final axisSize = a.shape[normAxis];
  if (k < 0 || k > axisSize) {
    throw RangeError.value(k, 'k', 'Must be in range [0, $axisSize].');
  }

  final outShape = List<int>.of(a.shape);
  outShape[normAxis] = k;

  final valuesResult = _prepareOutputArray<T>(
    a,
    outShape,
    a.dtype,
    outValues,
    'outValues',
  );
  final indicesResult = _prepareOutputArray<Int64>(
    a,
    outShape,
    DType.int64,
    outIndices,
    'outIndices',
  );

  if (k == 0 || a.size == 0) {
    return (values: valuesResult, indices: indicesResult);
  }

  GpuKernels.executeAxisSort(
    src: a.buffer,
    shapeSrc: a.shape,
    stridesSrc: a.strides,
    offsetSrc: a.offsetElements,
    dtypeSrc: a.dtype,
    axis: normAxis,
    outAxisSize: k,
    descending: largest,
    dstValues: valuesResult.buffer,
    outValShape: valuesResult.shape,
    outValStrides: valuesResult.strides,
    offsetDstValues: valuesResult.offsetElements,
    dstIndices: indicesResult.buffer,
    outIndicesShape: indicesResult.shape,
    outIndicesStrides: indicesResult.strides,
    offsetDstIndices: indicesResult.offsetElements,
  );
  return (values: valuesResult, indices: indicesResult);
}

void _validateKth(Object kth, int axisSize) {
  final List<int> indices;
  if (kth is int) {
    indices = <int>[kth];
  } else if (kth is List<int>) {
    if (kth.isEmpty) {
      throw ArgumentError.value(kth, 'kth', 'Must not be empty.');
    }
    indices = kth;
  } else {
    throw ArgumentError.value(kth, 'kth', 'Must be an int or List<int>.');
  }

  for (final k in indices) {
    final normK = k < 0 ? k + axisSize : k;
    if (normK < 0 || normK >= axisSize) {
      throw RangeError.value(
        k,
        'kth',
        'Must be in range [-$axisSize, ${axisSize - 1}].',
      );
    }
  }
}

/// Rearranges elements of [a] along [axis] so that the [kth] element is in
/// its final sorted position.
///
/// All elements smaller than the [kth] element are moved before it and all
/// equal or greater elements are moved behind it. Accepts an `int` or
/// `List<int>` for [kth].
GpuArray<T> partition<T extends DTypeTag, Out extends T>(
  GpuArray<T> a,
  Object kth, {
  int? axis = -1,
  GpuArray<Out>? out,
}) {
  _checkArrayNotDisposed(a, 'a');
  final axisSize = axis == null
      ? a.size
      : a.shape[_normalizeRequiredAxis(axis, a.ndim)];
  _validateKth(kth, axisSize);
  return sort(a, axis: axis, out: out);
}

/// Computes the 64-bit integer indices that would partition [a] at [kth]
/// along [axis].
///
/// Accepts an `int` or `List<int>` for [kth]. If [out] is provided, the
/// indices are written into [out] and returned.
GpuArray<Int64> argpartition<T extends DTypeTag>(
  GpuArray<T> a,
  Object kth, {
  int? axis = -1,
  GpuArray<Int64>? out,
}) {
  _checkArrayNotDisposed(a, 'a');
  final axisSize = axis == null
      ? a.size
      : a.shape[_normalizeRequiredAxis(axis, a.ndim)];
  _validateKth(kth, axisSize);
  return argsort<T>(a, axis: axis, out: out);
}

List<int> _inferListShape(List<Object?> list) {
  final shape = <int>[];
  Object? current = list;
  while (current is List) {
    shape.add(current.length);
    current = current.isEmpty ? null : current.first;
  }
  return shape;
}

(GpuArray<DTypeTag>, bool) _coerceObjectToGpuArray(
  Object value,
  DType defaultDType,
  GpuDevice device,
) {
  if (value is GpuArray<DTypeTag>) {
    _checkArrayNotDisposed(value, 'v');
    if (value.device != device) {
      throw ArgumentError.value(
        value.device,
        'v.device',
        'Must reside on the same GpuDevice ($device) as a.',
      );
    }
    return (value, false);
  }
  if (value is List) {
    final shape = _inferListShape(value);
    return (
      GpuArray.fromList(value, shape, defaultDType, device: device),
      true,
    );
  }
  if (value is num || value is bool || value is BigInt || value is Complex) {
    return (
      GpuArray.full(const <int>[], value, defaultDType, device: device),
      true,
    );
  }
  throw ArgumentError.value(
    value,
    'v',
    'Must be a GpuArray, List, num, bool, BigInt, or Complex.',
  );
}

/// Finds 64-bit indices where elements of [v] should be inserted into
/// sorted 1-D array [a] to maintain order.
///
/// Uses [side] ([SearchSide.left] or [SearchSide.right]) to control tie
/// placement, and optional 1-D integer [sorter] permutation indices.
GpuArray<Int64> searchsorted<T extends DTypeTag>(
  GpuArray<T> a,
  Object v, {
  SearchSide side = SearchSide.left,
  GpuArray<DTypeTag>? sorter,
  GpuArray<Int64>? out,
}) {
  _checkArrayNotDisposed(a, 'a');
  if (a.ndim != 1) {
    throw ArgumentError.value(a.shape, 'a.shape', 'Must be a 1-D array.');
  }
  if (sorter != null) {
    _checkArrayNotDisposed(sorter, 'sorter');
    if (sorter.device != a.device) {
      throw ArgumentError.value(
        sorter.device,
        'sorter.device',
        'Must reside on the same GpuDevice (${a.device}) as a.',
      );
    }
    if (sorter.ndim != 1 || sorter.shape[0] != a.shape[0]) {
      throw ArgumentError.value(
        sorter.shape,
        'sorter.shape',
        'Must be a 1-D array matching a.shape (${a.shape}).',
      );
    }
    if (!sorter.dtype.isInteger) {
      throw ArgumentError.value(
        sorter.dtype,
        'sorter.dtype',
        'Must be an integer DType.',
      );
    }
  }

  final (vArr, ownsVArr) = _coerceObjectToGpuArray(v, a.dtype, a.device);
  GpuArray<DTypeTag>? castA;
  GpuArray<DTypeTag>? castV;
  try {
    final result = _prepareOutputArray<Int64>(
      a,
      vArr.shape,
      DType.int64,
      out,
      'out',
    );
    if (vArr.size == 0) return result;

    final commonDType = GpuArray.promoteDTypes(a.dtype, vArr.dtype);
    final effectiveA = a.dtype == commonDType
        ? a
        : (castA = a.astype(commonDType));
    final effectiveV = vArr.dtype == commonDType
        ? vArr
        : (castV = vArr.astype(commonDType));

    GpuKernels.executeSearchSorted(
      arr: effectiveA.buffer,
      lengthA: effectiveA.shape[0],
      strideA: effectiveA.strides[0],
      offsetA: effectiveA.offsetElements,
      dtype: commonDType,
      values: effectiveV.buffer,
      shapeV: effectiveV.shape,
      stridesV: effectiveV.strides,
      offsetV: effectiveV.offsetElements,
      dst: result.buffer,
      outStrides: result.strides,
      offsetDst: result.offsetElements,
      side: side == SearchSide.right ? 1 : 0,
      sorter: sorter?.buffer,
      strideSorter: sorter != null ? sorter.strides[0] : 0,
      offsetSorter: sorter != null ? sorter.offsetElements : 0,
      sorterDType: sorter?.dtype ?? DType.int64,
    );
    return result;
  } finally {
    castA?.dispose();
    castV?.dispose();
    if (ownsVArr) {
      vArr.dispose();
    }
  }
}

GpuArray<R> _wrapUniqueBuffer<R extends DTypeTag>(
  GpuBuffer buffer,
  List<int> shape,
  DType<R> dtype,
  GpuDevice device,
) {
  buffer.detachFromScope();
  final array = GpuArray<R>.fromBuffer(
    buffer: buffer,
    shape: shape,
    strides: computeCStrides(shape),
    dtype: dtype,
    device: device,
  );
  buffer.dispose();
  return array;
}

({
  GpuArray<T> values,
  GpuArray<Int64>? indices,
  GpuArray<Int64>? inverse,
  GpuArray<Int64>? counts,
})
_uniqueCore<T extends DTypeTag>(
  GpuArray<T> ar, {
  int? axis,
  required bool wantIndices,
  required bool wantInverse,
  required bool wantCounts,
}) {
  _checkArrayNotDisposed(ar, 'ar');
  final device = ar.device;
  final dtype = ar.dtype;

  if (axis == null) {
    final flat = ar.isContiguous ? ar : ar.copy();
    try {
      final numRows = ar.size;
      final (
        valuesBuffer,
        indicesBuffer,
        inverseBuffer,
        countsBuffer,
        numUnique,
      ) = GpuKernels.executeUnique(
        src: flat.buffer,
        numRows: numRows,
        rowLength: 1,
        dtype: dtype,
      );

      final valuesArr = _wrapUniqueBuffer<T>(
        valuesBuffer,
        <int>[numUnique],
        dtype,
        device,
      );
      final GpuArray<Int64>? indicesArr;
      if (wantIndices) {
        indicesArr = _wrapUniqueBuffer<Int64>(
          indicesBuffer,
          <int>[numUnique],
          DType.int64,
          device,
        );
      } else {
        indicesBuffer.dispose();
        indicesArr = null;
      }

      final GpuArray<Int64>? inverseArr;
      if (wantInverse) {
        inverseArr = _wrapUniqueBuffer<Int64>(
          inverseBuffer,
          <int>[numRows],
          DType.int64,
          device,
        );
      } else {
        inverseBuffer.dispose();
        inverseArr = null;
      }

      final GpuArray<Int64>? countsArr;
      if (wantCounts) {
        countsArr = _wrapUniqueBuffer<Int64>(
          countsBuffer,
          <int>[numUnique],
          DType.int64,
          device,
        );
      } else {
        countsBuffer.dispose();
        countsArr = null;
      }

      return (
        values: valuesArr,
        indices: indicesArr,
        inverse: inverseArr,
        counts: countsArr,
      );
    } finally {
      if (!identical(flat, ar)) {
        flat.dispose();
      }
    }
  }

  final normAxis = _normalizeRequiredAxis(axis, ar.ndim);
  final moved = normAxis == 0 ? ar : moveaxis<T>(ar, normAxis, 0);
  final contiguousMoved = moved.isContiguous ? moved : moved.copy();
  try {
    final numRows = contiguousMoved.shape[0];
    var rowLength = 1;
    for (var d = 1; d < contiguousMoved.shape.length; d++) {
      rowLength *= contiguousMoved.shape[d];
    }

    final (
      valuesBuffer,
      indicesBuffer,
      inverseBuffer,
      countsBuffer,
      numUnique,
    ) = GpuKernels.executeUnique(
      src: contiguousMoved.buffer,
      numRows: numRows,
      rowLength: rowLength,
      dtype: dtype,
    );

    final movedOutShape = <int>[numUnique, ...contiguousMoved.shape.sublist(1)];
    final movedValues = _wrapUniqueBuffer<T>(
      valuesBuffer,
      movedOutShape,
      dtype,
      device,
    );
    final GpuArray<T> valuesArr;
    if (normAxis == 0) {
      valuesArr = movedValues;
    } else {
      final restored = moveaxis<T>(movedValues, 0, normAxis);
      try {
        valuesArr = restored.copy();
      } finally {
        restored.dispose();
        movedValues.dispose();
      }
    }

    final GpuArray<Int64>? indicesArr;
    if (wantIndices) {
      indicesArr = _wrapUniqueBuffer<Int64>(
        indicesBuffer,
        <int>[numUnique],
        DType.int64,
        device,
      );
    } else {
      indicesBuffer.dispose();
      indicesArr = null;
    }

    final GpuArray<Int64>? inverseArr;
    if (wantInverse) {
      inverseArr = _wrapUniqueBuffer<Int64>(
        inverseBuffer,
        <int>[numRows],
        DType.int64,
        device,
      );
    } else {
      inverseBuffer.dispose();
      inverseArr = null;
    }

    final GpuArray<Int64>? countsArr;
    if (wantCounts) {
      countsArr = _wrapUniqueBuffer<Int64>(
        countsBuffer,
        <int>[numUnique],
        DType.int64,
        device,
      );
    } else {
      countsBuffer.dispose();
      countsArr = null;
    }

    return (
      values: valuesArr,
      indices: indicesArr,
      inverse: inverseArr,
      counts: countsArr,
    );
  } finally {
    if (!identical(contiguousMoved, moved)) {
      contiguousMoved.dispose();
    }
    if (!identical(moved, ar)) {
      moved.dispose();
    }
  }
}

/// Finds the sorted unique elements of [ar] (or unique slices along [axis]).
GpuArray<T> unique<T extends DTypeTag>(GpuArray<T> ar, {int? axis}) {
  return _uniqueCore<T>(
    ar,
    axis: axis,
    wantIndices: false,
    wantInverse: false,
    wantCounts: false,
  ).values;
}

/// Finds the sorted unique elements of [ar] and the 64-bit indices of their
/// first occurrences in [ar].
({GpuArray<T> values, GpuArray<Int64> indices})
uniqueWithIndex<T extends DTypeTag>(GpuArray<T> ar, {int? axis}) {
  final res = _uniqueCore<T>(
    ar,
    axis: axis,
    wantIndices: true,
    wantInverse: false,
    wantCounts: false,
  );
  return (values: res.values, indices: res.indices!);
}

/// Finds the sorted unique elements of [ar] and the 64-bit inverse indices
/// that reconstruct [ar] from the unique array.
({GpuArray<T> values, GpuArray<Int64> inverse})
uniqueWithInverse<T extends DTypeTag>(GpuArray<T> ar, {int? axis}) {
  final res = _uniqueCore<T>(
    ar,
    axis: axis,
    wantIndices: false,
    wantInverse: true,
    wantCounts: false,
  );
  return (values: res.values, inverse: res.inverse!);
}

/// Finds the sorted unique elements of [ar] and the 64-bit occurrence counts
/// of each unique element.
({GpuArray<T> values, GpuArray<Int64> counts})
uniqueWithCounts<T extends DTypeTag>(GpuArray<T> ar, {int? axis}) {
  final res = _uniqueCore<T>(
    ar,
    axis: axis,
    wantIndices: false,
    wantInverse: false,
    wantCounts: true,
  );
  return (values: res.values, counts: res.counts!);
}

/// Finds the sorted unique elements of [ar] along with first-occurrence
/// `indices`, reconstruction `inverse` indices, and occurrence `counts`.
({
  GpuArray<T> values,
  GpuArray<Int64> indices,
  GpuArray<Int64> inverse,
  GpuArray<Int64> counts,
})
uniqueAll<T extends DTypeTag>(GpuArray<T> ar, {int? axis}) {
  final res = _uniqueCore<T>(
    ar,
    axis: axis,
    wantIndices: true,
    wantInverse: true,
    wantCounts: true,
  );
  return (
    values: res.values,
    indices: res.indices!,
    inverse: res.inverse!,
    counts: res.counts!,
  );
}

bool _isSignedInteger(DType dtype) => switch (dtype) {
  DType.int8 || DType.int16 || DType.int32 || DType.int64 => true,
  DType.uint8 ||
  DType.uint16 ||
  DType.uint32 ||
  DType.uint64 ||
  DType.float16 ||
  DType.bfloat16 ||
  DType.float32 ||
  DType.float64 ||
  DType.complex64 ||
  DType.complex128 ||
  DType.boolean => false,
};

/// Counts occurrences of each value in 1-D non-negative integer array [x].
///
/// Produces a [GpuArray<Int64>] when [weights] is `null`, or a
/// [GpuArray<Float64>] of weighted bin sums when [weights] is provided.
GpuArray<DTypeTag> bincount(
  GpuArray<DTypeTag> x, {
  GpuArray<DTypeTag>? weights,
  int minlength = 0,
  GpuArray<DTypeTag>? out,
}) {
  _checkArrayNotDisposed(x, 'x');
  if (x.ndim != 1) {
    throw ArgumentError.value(x.shape, 'x.shape', 'Must be a 1-D array.');
  }
  if (!x.dtype.isInteger) {
    throw ArgumentError.value(x.dtype, 'x.dtype', 'Must be an integer DType.');
  }
  if (minlength < 0) {
    throw ArgumentError.value(minlength, 'minlength', 'Must be non-negative.');
  }
  if (weights != null) {
    _checkArrayNotDisposed(weights, 'weights');
    if (weights.device != x.device) {
      throw ArgumentError.value(
        weights.device,
        'weights.device',
        'Must reside on the same GpuDevice (${x.device}) as x.',
      );
    }
    if (!areShapesEqual(weights.shape, x.shape)) {
      throw ArgumentError.value(
        weights.shape,
        'weights.shape',
        'Must match x.shape (${x.shape}).',
      );
    }
  }

  var outLength = minlength;
  if (x.size > 0) {
    if (_isSignedInteger(x.dtype)) {
      final minArr = x.min();
      try {
        final minVal = (minArr.scalar as num).toInt();
        if (minVal < 0) {
          throw ArgumentError.value(
            minVal,
            'x',
            'Must not contain negative values.',
          );
        }
      } finally {
        minArr.dispose();
      }
    }
    final maxArr = x.max();
    try {
      final maxVal = (maxArr.scalar as num).toInt();
      outLength = math.max(maxVal + 1, minlength);
    } finally {
      maxArr.dispose();
    }
  }

  final outShape = <int>[outLength];
  final outDType = weights != null ? DType.float64 : DType.int64;
  final result = _prepareOutputArray<DTypeTag>(
    x,
    outShape,
    outDType,
    out,
    'out',
  );

  if (outLength == 0) return result;
  if (x.size == 0) {
    GpuKernels.executeFill(
      dst: result.buffer,
      outShape: outShape,
      outStrides: result.strides,
      offsetDst: result.offsetElements,
      dtypeDst: outDType,
      value: 0,
    );
    return result;
  }

  GpuKernels.executeBincount(
    x: x.buffer,
    lengthX: x.shape[0],
    strideX: x.strides[0],
    offsetX: x.offsetElements,
    dtypeX: x.dtype,
    dst: result.buffer,
    outLength: outLength,
    strideOut: result.strides[0],
    offsetDst: result.offsetElements,
    weights: weights?.buffer,
    strideWeights: weights != null ? weights.strides[0] : 0,
    offsetWeights: weights != null ? weights.offsetElements : 0,
    dtypeWeights: weights?.dtype,
  );
  return result;
}

DType _defaultScanDType(DType dtype) => switch (dtype) {
  DType.boolean ||
  DType.int8 ||
  DType.int16 ||
  DType.int32 ||
  DType.int64 => DType.int64,
  DType.uint8 || DType.uint16 || DType.uint32 || DType.uint64 => DType.uint64,
  DType.float16 ||
  DType.bfloat16 ||
  DType.float32 ||
  DType.float64 ||
  DType.complex64 ||
  DType.complex128 => dtype,
};

GpuArray<DTypeTag> _cumulativeScanImpl(
  String op,
  GpuArray<DTypeTag> a, {
  int? axis,
  DType? dtype,
  GpuArray<DTypeTag>? out,
}) {
  _checkArrayNotDisposed(a, 'a');
  final targetDType = dtype ?? _defaultScanDType(a.dtype);

  if (axis == null) {
    final flat = a.flatten();
    try {
      final outShape = flat.shape;
      final result = _prepareOutputArray<DTypeTag>(
        a,
        outShape,
        targetDType,
        out,
        'out',
      );
      if (flat.size == 0) return result;
      GpuKernels.executeCumulativeScan(
        op: op,
        src: flat.buffer,
        shapeSrc: flat.shape,
        stridesSrc: flat.strides,
        offsetSrc: flat.offsetElements,
        dtypeSrc: flat.dtype,
        dst: result.buffer,
        outShape: result.shape,
        outStrides: result.strides,
        offsetDst: result.offsetElements,
        dtypeDst: targetDType,
        axis: 0,
      );
      return result;
    } finally {
      if (!identical(flat, a)) {
        flat.dispose();
      }
    }
  }

  final normAxis = _normalizeRequiredAxis(axis, a.ndim);
  final result = _prepareOutputArray<DTypeTag>(
    a,
    a.shape,
    targetDType,
    out,
    'out',
  );
  if (a.size == 0) return result;

  GpuKernels.executeCumulativeScan(
    op: op,
    src: a.buffer,
    shapeSrc: a.shape,
    stridesSrc: a.strides,
    offsetSrc: a.offsetElements,
    dtypeSrc: a.dtype,
    dst: result.buffer,
    outShape: result.shape,
    outStrides: result.strides,
    offsetDst: result.offsetElements,
    dtypeDst: targetDType,
    axis: normAxis,
  );
  return result;
}

/// Computes the cumulative sum of elements along [axis] (or flattened when
/// [axis] is `null`).
///
/// Signed integer and boolean inputs accumulate into [DType.int64] by
/// default, and unsigned integer inputs accumulate into [DType.uint64]
/// unless [dtype] is explicitly specified.
GpuArray<DTypeTag> cumsum(
  GpuArray<DTypeTag> a, {
  int? axis,
  DType? dtype,
  GpuArray<DTypeTag>? out,
}) {
  return _cumulativeScanImpl('cumsum', a, axis: axis, dtype: dtype, out: out);
}

/// Computes the cumulative product of elements along [axis] (or flattened
/// when [axis] is `null`).
///
/// Signed integer and boolean inputs accumulate into [DType.int64] by
/// default, and unsigned integer inputs accumulate into [DType.uint64]
/// unless [dtype] is explicitly specified.
GpuArray<DTypeTag> cumprod(
  GpuArray<DTypeTag> a, {
  int? axis,
  DType? dtype,
  GpuArray<DTypeTag>? out,
}) {
  return _cumulativeScanImpl('cumprod', a, axis: axis, dtype: dtype, out: out);
}

GpuArray<T> _coerceDiffBoundary<T extends DTypeTag>(
  Object boundary,
  GpuArray<T> a,
  int normAxis,
  String paramName,
) {
  if (boundary is GpuArray<DTypeTag>) {
    _checkArrayNotDisposed(boundary, paramName);
    if (boundary.device != a.device) {
      throw ArgumentError.value(
        boundary.device,
        '$paramName.device',
        'Must reside on the same GpuDevice (${a.device}) as a.',
      );
    }
    if (boundary.ndim == 0) {
      final bShape = List<int>.of(a.shape);
      bShape[normAxis] = 1;
      final castScalar = boundary.dtype == a.dtype
          ? boundary as GpuArray<T>
          : boundary.astype(a.dtype);
      try {
        return broadcastTo<T>(castScalar, bShape).copy();
      } finally {
        if (!identical(castScalar, boundary)) {
          castScalar.dispose();
        }
      }
    }
    return boundary.dtype == a.dtype
        ? boundary.copy() as GpuArray<T>
        : boundary.astype(a.dtype);
  }
  if (boundary is List) {
    final shape = _inferListShape(boundary);
    return GpuArray<T>.fromList(boundary, shape, a.dtype, device: a.device);
  }
  if (boundary is num ||
      boundary is bool ||
      boundary is BigInt ||
      boundary is Complex) {
    final bShape = List<int>.of(a.shape);
    bShape[normAxis] = 1;
    return GpuArray<T>.full(bShape, boundary, a.dtype, device: a.device);
  }
  throw ArgumentError.value(
    boundary,
    paramName,
    'Must be a GpuArray, List, num, bool, BigInt, or Complex.',
  );
}

/// Computes the [n]-th discrete difference along [axis].
///
/// Optional [prepend] and [append] values are concatenated along [axis] prior
/// to computing differences.
GpuArray<T> diff<T extends DTypeTag, Out extends T>(
  GpuArray<T> a, {
  int n = 1,
  int axis = -1,
  Object? prepend,
  Object? append,
  GpuArray<Out>? out,
}) {
  _checkArrayNotDisposed(a, 'a');
  if (n < 0) {
    throw ArgumentError.value(n, 'n', 'Must be non-negative.');
  }
  if (a.ndim == 0) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must have at least 1 dimension for diff.',
    );
  }
  final normAxis = _normalizeRequiredAxis(axis, a.ndim);

  GpuArray<T> working = a;
  var ownsWorking = false;

  if (prepend != null || append != null) {
    GpuArray<T>? prepArr;
    GpuArray<T>? appArr;
    try {
      final parts = <GpuArray<T>>[];
      if (prepend != null) {
        prepArr = _coerceDiffBoundary<T>(prepend, a, normAxis, 'prepend');
        parts.add(prepArr);
      }
      parts.add(a);
      if (append != null) {
        appArr = _coerceDiffBoundary<T>(append, a, normAxis, 'append');
        parts.add(appArr);
      }
      working = concatenate(parts, axis: normAxis) as GpuArray<T>;
      ownsWorking = true;
    } finally {
      prepArr?.dispose();
      appArr?.dispose();
    }
  }

  try {
    final finalAxisLength = math.max(0, working.shape[normAxis] - n);
    final outShape = List<int>.of(working.shape);
    outShape[normAxis] = finalAxisLength;

    final result = _prepareOutputArray<T>(a, outShape, a.dtype, out, 'out');
    if (n == 0) {
      if (out != null) {
        if (result.size > 0) {
          GpuKernels.copyStrided(
            src: working.buffer,
            shape: working.shape,
            strides: working.strides,
            offsetSrc: working.offsetElements,
            dtypeSrc: working.dtype,
            dst: result.buffer,
            outStrides: result.strides,
            offsetDst: result.offsetElements,
            dtypeDst: result.dtype,
          );
        }
        return result;
      }
      return ownsWorking ? working : working.copy();
    }

    if (result.size == 0) {
      return result;
    }

    var current = working;
    var ownsCurrent = false;
    try {
      for (var step = 0; step < n; step++) {
        final isLast = step == n - 1;
        final stepShape = List<int>.of(current.shape);
        stepShape[normAxis] = current.shape[normAxis] - 1;
        final stepDst = isLast
            ? result
            : GpuArray<T>.empty(stepShape, a.dtype, device: a.device);
        GpuKernels.executeDiff1(
          src: current.buffer,
          stridesSrc: current.strides,
          offsetSrc: current.offsetElements,
          dtype: a.dtype,
          dst: stepDst.buffer,
          outShape: stepShape,
          outStrides: stepDst.strides,
          offsetDst: stepDst.offsetElements,
          axis: normAxis,
        );
        if (ownsCurrent) {
          current.dispose();
        }
        current = stepDst;
        ownsCurrent = !isLast;
      }
      return result;
    } finally {
      if (ownsCurrent) {
        current.dispose();
      }
    }
  } finally {
    if (ownsWorking && !(n == 0 && out == null)) {
      working.dispose();
    }
  }
}
