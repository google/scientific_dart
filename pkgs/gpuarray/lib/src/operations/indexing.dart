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
GpuArray<T> where<T extends DTypeTag>(
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
      : condition.astype<Boolean>(DType.boolean);
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
GpuArray<T> select<T extends DTypeTag>(
  List<GpuArray<DTypeTag>> condlist,
  List<GpuArray<DTypeTag>> choicelist, {
  GpuArray<DTypeTag>? defaultValue,
  GpuArray<DTypeTag>? defaultArr,
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
    if (!identical(current, fallback) && !identical(next, current)) {
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
      return take<T>(flatArr, flatIndices);
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
GpuArray<T> take<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray<DTypeTag> indices, {
  int? axis,
  GpuArray<T>? out,
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
GpuArray<T> gather<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray<DTypeTag> indices, {
  int? axis,
  GpuArray<T>? out,
}) => take<T>(arr, indices, axis: axis, out: out);

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
GpuArray<T> takeAlongAxis<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray<DTypeTag> indices,
  int axis, {
  GpuArray<T>? out,
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

/// Takes values from [arr] along [axis] at specified [indices].
@Deprecated('Use takeAlongAxis instead.')
// ignore: non_constant_identifier_names
GpuArray<T> take_along_axis<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray<DTypeTag> indices,
  int axis, {
  GpuArray<T>? out,
}) => takeAlongAxis<T>(arr, indices, axis, out: out);

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

/// Puts [values] into [arr] along [axis] at positions specified by [indices].
@Deprecated('Use putAlongAxis instead.')
// ignore: non_constant_identifier_names
void put_along_axis<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray<DTypeTag> indices,
  GpuArray<DTypeTag> values,
  int axis,
) => putAlongAxis<T>(arr, indices, values, axis);

/// Finds the indices of non-zero elements as a list of 1D arrays, one per
/// dimension.
List<GpuArray<Int32>> nonzero(GpuArray<DTypeTag> arr) {
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
    final results = <GpuArray<Int32>>[];
    for (var d = 0; d < rank; d++) {
      final axisOut = GpuArray<Int32>.empty(
        [count],
        DType.int32,
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
          dtypeDst: DType.int32,
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

/// Finds indices that are non-zero in the flattened version of [arr].
GpuArray<Int32> flatnonzero(GpuArray<DTypeTag> arr) {
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
      final result = GpuArray<Int32>.empty(
        [count],
        DType.int32,
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
          dtypeDst: DType.int32,
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

/// Finds the indices of non-zero elements as a 2D array of shape `(N, rank)`.
GpuArray<Int32> argwhere(GpuArray<DTypeTag> arr) {
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
    final result = GpuArray<Int32>.empty(
      [count, rank],
      DType.int32,
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
        dtypeDst: DType.int32,
      );
    }
    return result;
  } finally {
    prefixBuffer.dispose();
  }
}
