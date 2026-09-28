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
import '../backend/compute_engine.dart';
import '../backend/kernels.dart';

/// Returns elements chosen from [x] or [y] depending on [condition].
///
/// If [x] and [y] are omitted, returns indices where [condition] is `true`
/// as a tuple of index arrays (identical to [nonzero]).
dynamic where<T extends DTypeTag>(
  GpuArray condition, [
  GpuArray? x,
  GpuArray? y,
]) {
  if (x == null && y == null) {
    return nonzero(condition);
  }
  if (x == null || y == null) {
    throw ArgumentError('where() requires either both x and y or neither');
  }

  final outShape = ShapeUtils.broadcastShapes(
    condition.shape,
    ShapeUtils.broadcastShapes(x.shape, y.shape),
  );

  // Type promotion between x and y
  final outDType = GpuArray.promoteDTypes(x.dtype, y.dtype) as DType<T>;
  final result = GpuArray<T>.empty(outShape, outDType, device: x.device);

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

/// Returns an array drawn from elements in [choicelist], depending on conditions in [condlist].
GpuArray<T> select<T extends DTypeTag>(
  List<GpuArray> condlist,
  List<GpuArray> choicelist, [
  GpuArray? defaultArr,
]) {
  if (condlist.isEmpty || choicelist.isEmpty) {
    throw ArgumentError('condlist and choicelist must not be empty');
  }
  if (condlist.length != choicelist.length) {
    throw ArgumentError('condlist and choicelist must have the same length');
  }

  var outShape = choicelist[0].shape;
  var outDType = choicelist[0].dtype;

  for (var i = 0; i < condlist.length; i++) {
    outShape = ShapeUtils.broadcastShapes(outShape, condlist[i].shape);
    outShape = ShapeUtils.broadcastShapes(outShape, choicelist[i].shape);
    outDType = GpuArray.promoteDTypes(outDType, choicelist[i].dtype);
  }

  GpuArray current = (defaultArr != null)
      ? defaultArr.astype(outDType)
      : GpuArray.zeros(outShape, outDType, device: choicelist[0].device);

  for (var i = condlist.length - 1; i >= 0; i--) {
    final cond = condlist[i];
    final choice = choicelist[i];
    current = where<T>(cond, choice, current) as GpuArray<T>;
  }

  return current as GpuArray<T>;
}

/// Returns the elements of an array that satisfy the boolean [condition].
GpuArray<T> extract<T extends DTypeTag>(GpuArray condition, GpuArray<T> arr) {
  final flatIndices = flatnonzero(condition);
  final count = flatIndices.shape[0];
  if (count == 0) {
    return GpuArray<T>.zeros([0], arr.dtype, device: arr.device);
  }

  final flatArr = arr.flatten();
  final result = GpuArray<T>.empty([count], arr.dtype, device: arr.device);

  final flatIndicesList = flatIndices.toList().cast<int>();
  for (var i = 0; i < count; i++) {
    final srcIdx = flatIndicesList[i];
    final val = ComputeEngine.readAny(
      flatArr.buffer,
      flatArr.dtype,
      srcIdx,
      offsetElements: flatArr.offsetElements,
    );
    ComputeEngine.writeAny(result.buffer, arr.dtype, i, val);
  }

  return result;
}

/// Takes values from [arr] along [axis] at specified 1D or multi-dimensional [indices].
GpuArray<T> take_along_axis<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray indices,
  int axis,
) {
  final rank = arr.shape.length;
  final normAxis = axis < 0 ? axis + rank : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw GpuAxisOutOfBoundsException(axis, rank);
  }

  final outShape = indices.shape;
  final result = GpuArray<T>.empty(outShape, arr.dtype, device: arr.device);

  GpuKernels.executeTakeAlongAxis(
    src: arr.buffer,
    shapeSrc: arr.shape,
    stridesSrc: arr.strides,
    offsetSrc: arr.offsetElements,
    dtypeSrc: arr.dtype,
    indices: indices.buffer,
    shapeIdx: indices.shape,
    stridesIdx: indices.strides,
    offsetIdx: indices.offsetElements,
    dtypeIdx: indices.dtype,
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
void put_along_axis<T extends DTypeTag>(
  GpuArray<T> arr,
  GpuArray indices,
  GpuArray values,
  int axis,
) {
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
    shapeIdx: indices.shape,
    stridesIdx: indices.strides,
    offsetIdx: indices.offsetElements,
    dtypeIdx: indices.dtype,
    values: values.buffer,
    shapeVal: values.shape,
    stridesVal: values.strides,
    offsetVal: values.offsetElements,
    dtypeVal: values.dtype,
    axis: normAxis,
  );
}

bool _isNonZero(dynamic val) {
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

/// Returns the indices of non-zero elements as a list of 1D arrays, one per dimension.
List<GpuArray<Int32>> nonzero(GpuArray arr) {
  final rank = arr.shape.length;
  final total = ShapeUtils.computeSize(arr.shape);
  final coords = List<int>.filled(rank, 0);

  final matchingCoords = List.generate(rank, (_) => <int>[]);

  for (var i = 0; i < total; i++) {
    var elemIdx = 0;
    for (var d = 0; d < rank; d++) {
      elemIdx += coords[d] * arr.strides[d];
    }

    final val = ComputeEngine.readAny(
      arr.buffer,
      arr.dtype,
      elemIdx,
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

/// Return indices that are non-zero in the flattened version of [arr].
GpuArray<Int32> flatnonzero(GpuArray arr) {
  final flat = arr.flatten();
  final total = flat.shape[0];
  final matching = <int>[];

  for (var i = 0; i < total; i++) {
    final val = ComputeEngine.readAny(
      flat.buffer,
      flat.dtype,
      i,
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
}

/// Returns the indices of non-zero elements as a 2D array of shape `(N, rank)`.
GpuArray<Int32> argwhere(GpuArray arr) {
  final nz = nonzero(arr);
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
}
