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

import '../device.dart';
import '../dtype.dart';
import '../gpu_array.dart';
import 'linalg_buffer_ops.dart';
import 'linalg_wgsl_df64.dart';
import 'tensor_kernels.dart';
import 'tensors.dart';

export 'decompositions.dart';
export 'solvers.dart';
export 'tensors.dart';

void _checkBinaryInputs(GpuArray a, GpuArray b) {
  if (a.isDisposed) {
    throw StateError('Cannot operate on a disposed GpuArray (a).');
  }
  if (b.isDisposed) {
    throw StateError('Cannot operate on a disposed GpuArray (b).');
  }
  if (a.device != b.device) {
    throw ArgumentError.value(
      b,
      'b',
      'Must reside on the same GpuDevice as a.',
    );
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b,
      'b',
      'Must have the same DType as a (${a.dtype}), got ${b.dtype}.',
    );
  }
}

({List<int> batchShape, List<int> aPadded, List<int> bPadded})
_broadcastBatchShapes(List<int> shapeA, List<int> shapeB) {
  final maxRank = math.max(shapeA.length, shapeB.length);
  final batchShape = List<int>.filled(maxRank, 1);
  final aPadded = List<int>.filled(maxRank, 1);
  final bPadded = List<int>.filled(maxRank, 1);
  for (var i = 0; i < maxRank; i++) {
    final dimA = i >= maxRank - shapeA.length
        ? shapeA[i - (maxRank - shapeA.length)]
        : 1;
    final dimB = i >= maxRank - shapeB.length
        ? shapeB[i - (maxRank - shapeB.length)]
        : 1;
    if (dimA != dimB && dimA != 1 && dimB != 1) {
      throw ArgumentError.value(
        shapeB,
        'b',
        'Must have broadcast-compatible batch dimensions with $shapeA.',
      );
    }
    aPadded[i] = dimA;
    bPadded[i] = dimB;
    batchShape[i] = math.max(dimA, dimB);
  }
  return (batchShape: batchShape, aPadded: aPadded, bPadded: bPadded);
}

/// Matrix product of two [GpuArray] tensors [a] and [b].
///
/// The behavior depends on the dimensionality of [a] and [b]:
/// - If both arguments are 2-D (`[M, K]` and `[K, N]`), they are multiplied as
///   conventional matrices producing shape `[M, N]`.
/// - If either argument is N-D (`N > 2`), it is treated as a stack of matrices
///   residing in the last two indices and broadcast accordingly.
/// - If the first argument is 1-D (`[K]`), it is promoted to a matrix by
///   prepending a `1` to its dimensions (`[1, K]`), and the prepended `1` is
///   removed after matrix multiplication.
/// - If the second argument is 1-D (`[K]`), it is promoted to a matrix by
///   appending a `1` to its dimensions (`[K, 1]`), and the appended `1` is
///   removed after matrix multiplication.
/// - If both arguments are 1-D (`[K]`), the inner product (0-D scalar) is
///   computed.
///
/// Both [a] and [b] must have at least 1 dimension (`ndim >= 1`), reside on the
/// same [GpuDevice], have matching [DType]s, and have matching inner dimensions.
/// If [out] is provided, it must match the result shape, dtype, and device.
GpuArray<T> matmul<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  GpuArray<T>? out,
}) {
  _checkBinaryInputs(a, b);
  if (a.ndim == 0) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must be at least 1-dimensional for matmul (0-D scalars are not allowed).',
    );
  }
  if (b.ndim == 0) {
    throw ArgumentError.value(
      b.shape,
      'b',
      'Must be at least 1-dimensional for matmul (0-D scalars are not allowed).',
    );
  }

  if (a.ndim == 1 && b.ndim == 1) {
    return dot(a, b, out: out);
  }

  final squeezeA = a.ndim == 1;
  final squeezeB = b.ndim == 1;
  final shapeA = squeezeA ? <int>[1, a.shape[0]] : a.shape;
  final shapeB = squeezeB ? <int>[b.shape[0], 1] : b.shape;

  final m = shapeA[shapeA.length - 2];
  final kA = shapeA[shapeA.length - 1];
  final kB = shapeB[shapeB.length - 2];
  final n = shapeB[shapeB.length - 1];
  if (kA != kB) {
    throw ArgumentError.value(
      b.shape,
      'b',
      'Must have inner dimension matching a ($kA != $kB).',
    );
  }

  final batchInfo = _broadcastBatchShapes(
    shapeA.sublist(0, shapeA.length - 2),
    shapeB.sublist(0, shapeB.length - 2),
  );
  final resultShape = <int>[
    ...batchInfo.batchShape,
    if (!squeezeA) m,
    if (!squeezeB) n,
  ];
  validateLinalgOut(out, a.device, resultShape, a.dtype);

  final batchCount = batchInfo.batchShape.isEmpty
      ? 1
      : batchInfo.batchShape.reduce((x, y) => x * y);
  final single = isSinglePrecisionDType(a.dtype);

  return ResourceScope.scope(() {
    if (isComplexDType(a.dtype)) {
      final bufferA = single
          ? toContiguousComplex64Buffer(a)
          : toContiguousComplex128Buffer(a);
      final bufferB = single
          ? toContiguousComplex64Buffer(b)
          : toContiguousComplex128Buffer(b);
      final bufferC = dispatchBatchedMatmulC128Gpu(
        a.device,
        bufferA,
        bufferB,
        batchCount: batchCount,
        m: m,
        k: kA,
        n: n,
        batchShape: batchInfo.batchShape,
        aBatchShape: batchInfo.aPadded,
        bBatchShape: batchInfo.bPadded,
        singlePrecision: single,
      );
      final output = single
          ? writeComplex64BufferToArray<T>(
              a.device,
              bufferC,
              resultShape,
              a.dtype,
              out: out,
            )
          : writeComplex128BufferToArray<T>(
              a.device,
              bufferC,
              resultShape,
              a.dtype,
              out: out,
            );
      if (out == null) output.detachToParentScope();
      return output;
    } else {
      final bufferA = single
          ? toContiguousFloat32Buffer(a)
          : toContiguousFloat64Buffer(a);
      final bufferB = single
          ? toContiguousFloat32Buffer(b)
          : toContiguousFloat64Buffer(b);
      final bufferC = dispatchBatchedMatmulF64Gpu(
        a.device,
        bufferA,
        bufferB,
        batchCount: batchCount,
        m: m,
        k: kA,
        n: n,
        batchShape: batchInfo.batchShape,
        aBatchShape: batchInfo.aPadded,
        bBatchShape: batchInfo.bPadded,
        singlePrecision: single,
      );
      final output = single
          ? writeFloat32BufferToArray<T>(
              a.device,
              bufferC,
              resultShape,
              a.dtype,
              out: out,
            )
          : writeFloat64BufferToArray<T>(
              a.device,
              bufferC,
              resultShape,
              a.dtype,
              out: out,
            );
      if (out == null) output.detachToParentScope();
      return output;
    }
  });
}

/// Dot product of two [GpuArray] tensors [a] and [b].
///
/// Specifically:
/// - If both [a] and [b] are 1-D arrays, computes the inner product of vectors
///   (without complex conjugation) as a 0-D scalar [GpuArray].
/// - If both [a] and [b] are 2-D arrays, computes matrix multiplication.
/// - If either [a] or [b] is 0-D (scalar), computes elementwise multiplication.
/// - If [a] is an N-D array and [b] is a 1-D array, computes the sum product
///   over the last axis of [a] and [b].
/// - If [a] is an N-D array and [b] is an M-D array (where `M >= 2`), computes
///   the sum product over the last axis of [a] and the second-to-last axis of
///   [b].
///
/// The [a] and [b] arrays must reside on the same [GpuDevice], have matching
/// [DType]s, and have compatible contracted dimensions.
GpuArray<T> dot<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  GpuArray<T>? out,
}) {
  _checkBinaryInputs(a, b);
  final single = isSinglePrecisionDType(a.dtype);

  if (a.ndim == 0 || b.ndim == 0) {
    final resultShape = a.ndim == 0 ? b.shape : a.shape;
    validateLinalgOut(out, a.device, resultShape, a.dtype);
    final m = a.ndim == 0 ? 1 : a.size;
    final n = b.ndim == 0 ? 1 : b.size;
    return ResourceScope.scope(() {
      if (isComplexDType(a.dtype)) {
        final bufferA = single
            ? toContiguousComplex64Buffer(a)
            : toContiguousComplex128Buffer(a);
        final bufferB = single
            ? toContiguousComplex64Buffer(b)
            : toContiguousComplex128Buffer(b);
        final bufferC = dispatchBatchedMatmulC128Gpu(
          a.device,
          bufferA,
          bufferB,
          batchCount: 1,
          m: m,
          k: 1,
          n: n,
          singlePrecision: single,
        );
        final output = single
            ? writeComplex64BufferToArray<T>(
                a.device,
                bufferC,
                resultShape,
                a.dtype,
                out: out,
              )
            : writeComplex128BufferToArray<T>(
                a.device,
                bufferC,
                resultShape,
                a.dtype,
                out: out,
              );
        if (out == null) output.detachToParentScope();
        return output;
      } else {
        final bufferA = single
            ? toContiguousFloat32Buffer(a)
            : toContiguousFloat64Buffer(a);
        final bufferB = single
            ? toContiguousFloat32Buffer(b)
            : toContiguousFloat64Buffer(b);
        final bufferC = dispatchBatchedMatmulF64Gpu(
          a.device,
          bufferA,
          bufferB,
          batchCount: 1,
          m: m,
          k: 1,
          n: n,
          singlePrecision: single,
        );
        final output = single
            ? writeFloat32BufferToArray<T>(
                a.device,
                bufferC,
                resultShape,
                a.dtype,
                out: out,
              )
            : writeFloat64BufferToArray<T>(
                a.device,
                bufferC,
                resultShape,
                a.dtype,
                out: out,
              );
        if (out == null) output.detachToParentScope();
        return output;
      }
    });
  }

  if (a.ndim == 1 && b.ndim == 1) {
    if (a.shape[0] != b.shape[0]) {
      throw ArgumentError.value(
        b.shape,
        'b',
        'Must have the same length as a (${a.shape[0]} != ${b.shape[0]}).',
      );
    }
    validateLinalgOut(out, a.device, const <int>[], a.dtype);
    final k = a.shape[0];
    return ResourceScope.scope(() {
      if (isComplexDType(a.dtype)) {
        final bufferA = single
            ? toContiguousComplex64Buffer(a)
            : toContiguousComplex128Buffer(a);
        final bufferB = single
            ? toContiguousComplex64Buffer(b)
            : toContiguousComplex128Buffer(b);
        final bufferC = dispatchBatchedMatmulC128Gpu(
          a.device,
          bufferA,
          bufferB,
          batchCount: 1,
          m: 1,
          k: k,
          n: 1,
          singlePrecision: single,
        );
        final output = single
            ? writeComplex64BufferToArray<T>(
                a.device,
                bufferC,
                const <int>[],
                a.dtype,
                out: out,
              )
            : writeComplex128BufferToArray<T>(
                a.device,
                bufferC,
                const <int>[],
                a.dtype,
                out: out,
              );
        if (out == null) output.detachToParentScope();
        return output;
      } else {
        final bufferA = single
            ? toContiguousFloat32Buffer(a)
            : toContiguousFloat64Buffer(a);
        final bufferB = single
            ? toContiguousFloat32Buffer(b)
            : toContiguousFloat64Buffer(b);
        final bufferC = dispatchBatchedMatmulF64Gpu(
          a.device,
          bufferA,
          bufferB,
          batchCount: 1,
          m: 1,
          k: k,
          n: 1,
          singlePrecision: single,
        );
        final output = single
            ? writeFloat32BufferToArray<T>(
                a.device,
                bufferC,
                const <int>[],
                a.dtype,
                out: out,
              )
            : writeFloat64BufferToArray<T>(
                a.device,
                bufferC,
                const <int>[],
                a.dtype,
                out: out,
              );
        if (out == null) output.detachToParentScope();
        return output;
      }
    });
  }

  if (a.ndim == 2 && b.ndim == 2) {
    return matmul(a, b, out: out);
  }

  if (b.ndim == 1) {
    final kA = a.shape.last;
    final kB = b.shape[0];
    if (kA != kB) {
      throw ArgumentError.value(
        b.shape,
        'b',
        'Must match the last dimension of a ($kA != $kB).',
      );
    }
    final resultShape = a.shape.sublist(0, a.ndim - 1);
    validateLinalgOut(out, a.device, resultShape, a.dtype);
    final m = kA == 0 ? 0 : a.size ~/ kA;
    return ResourceScope.scope(() {
      if (isComplexDType(a.dtype)) {
        final bufferA = single
            ? toContiguousComplex64Buffer(a)
            : toContiguousComplex128Buffer(a);
        final bufferB = single
            ? toContiguousComplex64Buffer(b)
            : toContiguousComplex128Buffer(b);
        final bufferC = dispatchBatchedMatmulC128Gpu(
          a.device,
          bufferA,
          bufferB,
          batchCount: 1,
          m: m,
          k: kA,
          n: 1,
          singlePrecision: single,
        );
        final output = single
            ? writeComplex64BufferToArray<T>(
                a.device,
                bufferC,
                resultShape,
                a.dtype,
                out: out,
              )
            : writeComplex128BufferToArray<T>(
                a.device,
                bufferC,
                resultShape,
                a.dtype,
                out: out,
              );
        if (out == null) output.detachToParentScope();
        return output;
      } else {
        final bufferA = single
            ? toContiguousFloat32Buffer(a)
            : toContiguousFloat64Buffer(a);
        final bufferB = single
            ? toContiguousFloat32Buffer(b)
            : toContiguousFloat64Buffer(b);
        final bufferC = dispatchBatchedMatmulF64Gpu(
          a.device,
          bufferA,
          bufferB,
          batchCount: 1,
          m: m,
          k: kA,
          n: 1,
          singlePrecision: single,
        );
        final output = single
            ? writeFloat32BufferToArray<T>(
                a.device,
                bufferC,
                resultShape,
                a.dtype,
                out: out,
              )
            : writeFloat64BufferToArray<T>(
                a.device,
                bufferC,
                resultShape,
                a.dtype,
                out: out,
              );
        if (out == null) output.detachToParentScope();
        return output;
      }
    });
  }

  return tensordot(
    a,
    b,
    axes: (<int>[a.ndim - 1], <int>[b.ndim - 2]),
    out: out,
  );
}

/// Vector dot product of two [GpuArray] tensors [a] and [b].
///
/// Flattens multidimensional inputs before computing the inner product and
/// returns a 0-D scalar [GpuArray]. If [a] has a complex data type
/// ([DType.complex64] or [DType.complex128]), the complex conjugate of [a] is
/// used for the calculation.
///
/// The [a] and [b] arrays must reside on the same [GpuDevice], have matching
/// [DType]s, and contain the same total number of elements (`a.size == b.size`).
GpuArray<T> vdot<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  GpuArray<T>? out,
}) {
  _checkBinaryInputs(a, b);
  if (a.size != b.size) {
    throw ArgumentError.value(
      b.shape,
      'b',
      'Must have the same number of elements as a (${a.size} != ${b.size}).',
    );
  }
  validateLinalgOut(out, a.device, const <int>[], a.dtype);

  final k = a.size;
  final single = isSinglePrecisionDType(a.dtype);
  return ResourceScope.scope(() {
    if (isComplexDType(a.dtype)) {
      final bufferA = single
          ? toContiguousComplex64Buffer(a)
          : toContiguousComplex128Buffer(a);
      final bufferB = single
          ? toContiguousComplex64Buffer(b)
          : toContiguousComplex128Buffer(b);
      final bufferC = dispatchBatchedMatmulC128Gpu(
        a.device,
        bufferA,
        bufferB,
        batchCount: 1,
        m: 1,
        k: k,
        n: 1,
        conjugateA: true,
        singlePrecision: single,
      );
      final output = single
          ? writeComplex64BufferToArray<T>(
              a.device,
              bufferC,
              const <int>[],
              a.dtype,
              out: out,
            )
          : writeComplex128BufferToArray<T>(
              a.device,
              bufferC,
              const <int>[],
              a.dtype,
              out: out,
            );
      if (out == null) output.detachToParentScope();
      return output;
    } else {
      final bufferA = single
          ? toContiguousFloat32Buffer(a)
          : toContiguousFloat64Buffer(a);
      final bufferB = single
          ? toContiguousFloat32Buffer(b)
          : toContiguousFloat64Buffer(b);
      final bufferC = dispatchBatchedMatmulF64Gpu(
        a.device,
        bufferA,
        bufferB,
        batchCount: 1,
        m: 1,
        k: k,
        n: 1,
        singlePrecision: single,
      );
      final output = single
          ? writeFloat32BufferToArray<T>(
              a.device,
              bufferC,
              const <int>[],
              a.dtype,
              out: out,
            )
          : writeFloat64BufferToArray<T>(
              a.device,
              bufferC,
              const <int>[],
              a.dtype,
              out: out,
            );
      if (out == null) output.detachToParentScope();
      return output;
    }
  });
}

/// Extracts the specified diagonal of [a] along [axis1] and [axis2].
///
/// If [a] is 2-D, returns a 1-D [GpuArray] containing the diagonal elements at
/// [offset]. If `a.ndim > 2`, the axes specified by [axis1] and [axis2] are
/// used to extract the 2-D diagonals, and the diagonal axis is appended to the
/// end of the output shape.
///
/// The [a] array must have at least 2 dimensions (`ndim >= 2`), and [axis1] and
/// [axis2] must be distinct valid axes.
GpuArray<T> diagonal<T extends DTypeTag>(
  GpuArray<T> a, {
  int offset = 0,
  int axis1 = 0,
  int axis2 = 1,
  GpuArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot extract diagonal of a disposed GpuArray.');
  }
  if (a.ndim < 2) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must have at least 2 dimensions for diagonal, got ${a.ndim}-D.',
    );
  }
  final normAxis1 = axis1 < 0 ? axis1 + a.ndim : axis1;
  final normAxis2 = axis2 < 0 ? axis2 + a.ndim : axis2;
  if (normAxis1 < 0 || normAxis1 >= a.ndim) {
    throw RangeError.range(axis1, -a.ndim, a.ndim - 1, 'axis1');
  }
  if (normAxis2 < 0 || normAxis2 >= a.ndim) {
    throw RangeError.range(axis2, -a.ndim, a.ndim - 1, 'axis2');
  }
  if (normAxis1 == normAxis2) {
    throw ArgumentError.value(
      axis2,
      'axis2',
      'Must be distinct from axis1 ($axis1).',
    );
  }

  final dim1 = a.shape[normAxis1];
  final dim2 = a.shape[normAxis2];
  final int diagonalLength;
  final int startOffsetElements;
  if (offset >= 0) {
    diagonalLength = math.max(0, math.min(dim1, dim2 - offset));
    startOffsetElements = a.offsetElements + offset * a.strides[normAxis2];
  } else {
    diagonalLength = math.max(0, math.min(dim1 + offset, dim2));
    startOffsetElements = a.offsetElements + (-offset) * a.strides[normAxis1];
  }

  final diagShape = <int>[];
  final diagStrides = <int>[];
  for (var i = 0; i < a.ndim; i++) {
    if (i != normAxis1 && i != normAxis2) {
      diagShape.add(a.shape[i]);
      diagStrides.add(a.strides[i]);
    }
  }
  diagShape.add(diagonalLength);
  diagStrides.add(a.strides[normAxis1] + a.strides[normAxis2]);

  validateLinalgOut(out, a.device, diagShape, a.dtype);

  final view = GpuArray<T>.fromBuffer(
    buffer: a.buffer,
    shape: diagShape,
    strides: diagStrides,
    dtype: a.dtype,
    device: a.device,
    offsetElements: diagonalLength == 0
        ? a.offsetElements
        : startOffsetElements,
    parent: a,
  );
  if (out != null) {
    try {
      return copyGpuArray(view, out: out);
    } finally {
      view.dispose();
    }
  }
  return view;
}

/// Sum along the diagonals of [a] at [offset] across [axis1] and [axis2].
///
/// If [a] is 2-D, returns a 0-D scalar [GpuArray] containing the sum along the
/// diagonal. If [a] has more than two dimensions, the axes specified by [axis1]
/// and [axis2] are used to determine the 2-D sub-arrays whose traces are
/// returned.
///
/// The [a] array must have at least 2 dimensions (`ndim >= 2`), and [axis1] and
/// [axis2] must be distinct valid axes.
GpuArray<T> trace<T extends DTypeTag>(
  GpuArray<T> a, {
  int offset = 0,
  int axis1 = 0,
  int axis2 = 1,
  GpuArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute trace of a disposed GpuArray.');
  }
  return ResourceScope.scope(() {
    final diagView = diagonal(a, offset: offset, axis1: axis1, axis2: axis2);
    final resultShape = diagView.shape.sublist(0, diagView.ndim - 1);
    final result = sumLastAxisGpu(diagView, resultShape: resultShape, out: out);
    if (out == null) result.detachToParentScope();
    return result;
  });
}
