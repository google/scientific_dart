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

import 'dart:ffi' as ffi;
import 'dart:math' as math;

import 'package:ndarray/ndarray.dart' as nd;
import 'package:ndarray/ndarray.dart'
    show DType, DTypeTag, Float64, Int64, NDArray;

import '../device.dart';
import '../exceptions.dart';
import '../gpu_array.dart';

/// Supported norm orders for vector and matrix [norm] and [cond] computations.
enum NormOrd {
  /// Frobenius matrix norm ($\sqrt{\sum_{i,j} |a_{ij}|^2}$).
  frobenius(isMatrixOnly: true, ndKind: nd.NormKind.frobenius),

  /// Nuclear (trace) matrix norm ($\sum_i \sigma_i$).
  nuclear(isMatrixOnly: true, ndKind: nd.NormKind.nuclear),

  /// 1-norm (sum of absolute values for vectors; maximum column sum for
  /// matrices).
  l1(isMatrixOnly: false, ndKind: nd.NormKind.l1),

  /// Negative 1-norm (minimum column sum for matrices;
  /// $\left(\sum |x_i|^{-1}\right)^{-1}$ for vectors).
  negL1(isMatrixOnly: false, ndKind: nd.NormKind.negL1),

  /// 2-norm (Euclidean norm for vectors; largest singular value for matrices).
  l2(isMatrixOnly: false, ndKind: nd.NormKind.l2),

  /// Negative 2-norm (smallest singular value for matrices;
  /// $\left(\sum |x_i|^{-2}\right)^{-1/2}$ for vectors).
  negL2(isMatrixOnly: false, ndKind: nd.NormKind.negL2),

  /// Infinity norm (maximum absolute value for vectors; maximum row sum for
  /// matrices).
  infinity(isMatrixOnly: false, ndKind: nd.NormKind.infinity),

  /// Negative infinity norm (minimum absolute value for vectors; minimum row
  /// sum for matrices).
  negInfinity(isMatrixOnly: false, ndKind: nd.NormKind.negInfinity);

  const NormOrd({required this.isMatrixOnly, required this.ndKind});

  /// Whether this norm order is only valid for 2D matrix norms.
  final bool isMatrixOnly;

  /// The corresponding `package:ndarray` norm kind.
  final nd.NormKind ndKind;
}

/// Resource disposal extension for the record returned by [slogdet].
extension SlogdetRecordDispose<T extends DTypeTag>
    on ({GpuArray<T> sign, GpuArray<T> logabsdet}) {
  /// Disposes [sign] and [logabsdet].
  void dispose() {
    sign.dispose();
    logabsdet.dispose();
  }
}

bool _shapesEqual(List<int> first, List<int> second) {
  if (first.length != second.length) return false;
  for (var i = 0; i < first.length; i++) {
    if (first[i] != second[i]) return false;
  }
  return true;
}

void _validateOutBuffer<R extends DTypeTag>(
  GpuArray<R> out,
  List<int> expectedShape,
  DType<R> expectedDType, {
  String name = 'out',
}) {
  if (out.size > 1 && out.strides.contains(0)) {
    throw ArgumentError.value(
      out,
      name,
      'Must be writeable and not a broadcasted view.',
    );
  }
  if (!_shapesEqual(out.shape, expectedShape) || out.dtype != expectedDType) {
    throw ArgumentError.value(
      out,
      name,
      'Must be an array with shape $expectedShape and dtype $expectedDType, '
      'got shape ${out.shape} and dtype ${out.dtype}.',
    );
  }
}

GpuArray<R> _writeOrWrapResult<R extends DTypeTag>(
  NDArray<R> hostResult,
  GpuDevice device,
  GpuArray<R>? out, {
  String outName = 'out',
  bool promote = true,
}) {
  if (out != null) {
    _validateOutBuffer(out, hostResult.shape, hostResult.dtype, name: outName);
    if (out.isContiguous) {
      final contiguous = hostResult.isContiguous
          ? hostResult
          : hostResult.copy();
      if (out.byteSize > 0) {
        out.buffer.copyFromHost(
          contiguous.pointer.cast<ffi.Void>(),
          out.byteSize,
          offset: out.offsetElements * out.dtype.byteWidth,
        );
      }
    } else if (out.byteSize > 0) {
      out.buffer.copyToHost(
        out.buffer.address.cast<ffi.Void>(),
        out.buffer.sizeInBytes,
      );
      final totalBufferElements = out.buffer.sizeInBytes ~/ out.dtype.byteWidth;
      final rootBufferView = NDArray<R>.fromPointer(
        out.buffer.address.cast<ffi.Void>(),
        <int>[totalBufferElements],
        out.dtype,
      );
      final outView = NDArray<R>.view(
        rootBufferView,
        shape: out.shape,
        strides: out.strides,
        offsetElements: out.offsetElements,
      );
      hostResult.copy(out: outView);
      out.buffer.copyFromHost(
        out.buffer.address.cast<ffi.Void>(),
        out.buffer.sizeInBytes,
      );
    }
    return out;
  }
  final gpuResult = GpuArray<R>.fromNDArray(hostResult, device: device);
  if (promote) {
    gpuResult.detachToParentScope();
  }
  return gpuResult;
}

NDArray<Float64> _toContiguousFloat64(NDArray array) {
  if (array.dtype == DType.float64) {
    return array.isContiguous
        ? array as NDArray<Float64>
        : (array as NDArray<Float64>).copy();
  }
  return array.astype<Float64>(DType.float64);
}

NDArray<T> _matchDType<T extends DTypeTag>(
  NDArray<Float64> array,
  DType<T> targetDType,
) {
  if (targetDType == DType.float64) {
    return array as NDArray<T>;
  }
  return array.astype<T>(targetDType);
}

/// Solves a linear matrix equation $A x = b$ for a square, non-singular matrix
/// [a].
///
/// Supports 1D right-hand side vectors `(n,)` and 2D right-hand side matrices
/// `(n, k)`. None of [a], [b], or [out] may be disposed.
GpuArray<T> solve<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  GpuArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute solve on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write solve result to a disposed output GpuArray.',
    );
  }
  if (a.rank < 2 || a.shape[a.rank - 2] != a.shape[a.rank - 1]) {
    throw GpuShapeMismatchException('solve', a.shape, b.shape);
  }
  final n = a.shape[a.rank - 1];
  if (b.rank < 1 || b.shape[0] != n) {
    throw GpuShapeMismatchException('solve', a.shape, b.shape);
  }
  if (out != null) {
    _validateOutBuffer(out, b.shape, b.dtype);
  }

  return NDArray.scope(() {
    final aF64 = _toContiguousFloat64(a.toNDArray());
    final bF64 = _toContiguousFloat64(b.toNDArray());
    final NDArray<Float64> solvedF64;
    try {
      solvedF64 = nd.solve<Float64>(aF64, bF64);
    } on nd.LinAlgException catch (error) {
      throw StateError(error.message);
    }
    final hostResult = _matchDType<T>(solvedF64, a.dtype);
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the multiplicative inverse $A^{-1}$ of a square matrix [a].
///
/// Neither [a] nor [out] (if provided) may be disposed, and [a] must be square
/// in its last two dimensions and non-singular.
GpuArray<T> inv<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute inv on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write inv result to a disposed output GpuArray.');
  }
  if (a.rank < 2 || a.shape[a.rank - 2] != a.shape[a.rank - 1]) {
    throw GpuShapeMismatchException('inv', a.shape, const <int>[]);
  }
  if (out != null) {
    _validateOutBuffer(out, a.shape, a.dtype);
  }

  return NDArray.scope(() {
    final aF64 = _toContiguousFloat64(a.toNDArray());
    final NDArray<Float64> invF64;
    try {
      invF64 = nd.inv<Float64>(aF64);
    } on nd.LinAlgException catch (error) {
      throw StateError(error.message);
    }
    final hostResult = _matchDType<T>(invF64, a.dtype);
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the Moore-Penrose pseudoinverse $A^+$ of a 2D matrix [a] via SVD.
///
/// Singular values smaller than `rcond * max(singular_values)` are treated as
/// zero. Neither [a] nor [out] (if provided) may be disposed, and [a] must be
/// 2-dimensional.
GpuArray<T> pinv<T extends DTypeTag>(
  GpuArray<T> a, {
  double rcond = 1e-15,
  GpuArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute pinv on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write pinv result to a disposed output GpuArray.');
  }
  if (a.rank != 2) {
    throw GpuShapeMismatchException('pinv', a.shape, const <int>[]);
  }
  final expectedShape = <int>[a.shape[1], a.shape[0]];
  if (out != null) {
    _validateOutBuffer(out, expectedShape, a.dtype);
  }

  return NDArray.scope(() {
    final aF64 = _toContiguousFloat64(a.toNDArray());
    final NDArray<Float64> pinvF64;
    try {
      pinvF64 = nd.pinv<Float64>(aF64, rcond: rcond);
    } on nd.LinAlgException catch (error) {
      throw StateError(error.message);
    }
    final hostResult = _matchDType<T>(pinvF64, a.dtype);
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the determinant of a square matrix or batch of square matrices [a].
///
/// Returns a 0D [GpuArray] (shape `[]`) when [a] is 2D, or an array of shape
/// `a.shape.sublist(0, a.rank - 2)` for batched inputs.
///
/// Neither [a] nor [out] (if provided) may be disposed, and [a] must be square
/// in its last two dimensions.
GpuArray<T> det<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute det on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write det result to a disposed output GpuArray.');
  }
  if (a.rank < 2 || a.shape[a.rank - 2] != a.shape[a.rank - 1]) {
    throw GpuShapeMismatchException('det', a.shape, const <int>[]);
  }
  final expectedShape = a.shape.sublist(0, a.rank - 2);
  if (out != null) {
    _validateOutBuffer(out, expectedShape, a.dtype);
  }

  return NDArray.scope(() {
    final aF64 = _toContiguousFloat64(a.toNDArray());
    final detF64 = nd.det<Float64>(aF64);
    final hostResult = _matchDType<T>(detF64, a.dtype);
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the sign and natural logarithm of the absolute value of the
/// determinant of a square matrix or batch of square matrices [a].
///
/// Returns a record `(sign, logabsdet)`. For a singular matrix, `sign` is `0.0`
/// and `logabsdet` is `-double.infinity`.
///
/// Neither [a] nor any array in [out] may be disposed, and [a] must be square
/// in its last two dimensions.
({GpuArray<T> sign, GpuArray<T> logabsdet}) slogdet<T extends DTypeTag>(
  GpuArray<T> a, {
  ({GpuArray<T> sign, GpuArray<T> logabsdet})? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute slogdet on a disposed GpuArray.');
  }
  if (out != null && (out.sign.isDisposed || out.logabsdet.isDisposed)) {
    throw StateError(
      'Cannot write slogdet result to a disposed output GpuArray.',
    );
  }
  if (a.rank < 2 || a.shape[a.rank - 2] != a.shape[a.rank - 1]) {
    throw GpuShapeMismatchException('slogdet', a.shape, const <int>[]);
  }
  final expectedShape = a.shape.sublist(0, a.rank - 2);
  if (out != null) {
    _validateOutBuffer(out.sign, expectedShape, a.dtype, name: 'out.sign');
    _validateOutBuffer(
      out.logabsdet,
      expectedShape,
      a.dtype,
      name: 'out.logabsdet',
    );
  }

  return NDArray.scope(() {
    final aF64 = _toContiguousFloat64(a.toNDArray());
    final rawSlogdet = nd.slogdet<Float64, Float64>(aF64);
    final signHost = _matchDType<T>(rawSlogdet.sign, a.dtype);
    final logabsdetHost = _matchDType<T>(rawSlogdet.logabsdet, a.dtype);

    final signGpu = _writeOrWrapResult(
      signHost,
      a.device,
      out?.sign,
      outName: 'out.sign',
      promote: false,
    );
    final logabsdetGpu = _writeOrWrapResult(
      logabsdetHost,
      a.device,
      out?.logabsdet,
      outName: 'out.logabsdet',
      promote: false,
    );
    if (out == null) {
      signGpu.detachToParentScope();
      logabsdetGpu.detachToParentScope();
    }
    return (sign: signGpu, logabsdet: logabsdetGpu);
  });
}

/// Raises a square matrix [a] to the integer power [n] using binary
/// exponentiation.
///
/// For $n = 0$, returns the identity matrix $I$. For $n < 0$, computes
/// $(A^{-1})^{|n|}$.
///
/// Neither [a] nor [out] (if provided) may be disposed, and [a] must be a
/// square 2D matrix.
GpuArray<T> matrixPower<T extends DTypeTag>(
  GpuArray<T> a,
  int n, {
  GpuArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute matrixPower on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write matrixPower result to a disposed output GpuArray.',
    );
  }
  if (a.rank != 2 || a.shape[0] != a.shape[1]) {
    throw GpuShapeMismatchException('matrixPower', a.shape, const <int>[]);
  }
  if (out != null) {
    _validateOutBuffer(out, a.shape, a.dtype);
  }

  return NDArray.scope(() {
    final aF64 = _toContiguousFloat64(a.toNDArray());
    final NDArray<Float64> powerF64;
    try {
      powerF64 = nd.matrix_power<Float64>(aF64, n);
    } on nd.LinAlgException catch (error) {
      throw StateError(error.message);
    }
    final hostResult = _matchDType<T>(powerF64, a.dtype);
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Snake-case alias for [matrixPower].
GpuArray<T> matrix_power<T extends DTypeTag>(
  GpuArray<T> a,
  int n, {
  GpuArray<T>? out,
}) => matrixPower(a, n, out: out);

/// Computes the numerical rank of a tensor [a] using Singular Value
/// Decomposition.
///
/// Returns a [GpuArray] of [Int64] (shape `[]` for 1D or 2D inputs, or batch
/// shape `a.shape.sublist(0, a.rank - 2)` for higher-rank inputs).
/// Singular values greater than [tolerance] (or $\max(m, n) \cdot \epsilon \cdot \sigma_{\max}$
/// if [tolerance] is omitted) are counted as non-zero.
///
/// Neither [a] nor [out] (if provided) may be disposed.
GpuArray<Int64> matrixRank<T extends DTypeTag>(
  GpuArray<T> a, {
  double? tolerance,
  GpuArray<Int64>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute matrixRank on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write matrixRank result to a disposed output GpuArray.',
    );
  }
  if (a.rank == 0) {
    throw GpuShapeMismatchException('matrixRank', a.shape, const <int>[]);
  }

  final expectedShape = a.rank < 2
      ? const <int>[]
      : a.shape.sublist(0, a.rank - 2);
  if (out != null) {
    _validateOutBuffer(out, expectedShape, DType.int64);
  }

  return NDArray.scope(() {
    final f64Input = _toContiguousFloat64(a.toNDArray());
    if (a.rank == 1) {
      final pointer = f64Input.pointer.cast<ffi.Double>();
      var nonZero = false;
      final threshold = tolerance ?? 1e-15;
      for (var i = 0; i < f64Input.size; i++) {
        if (pointer[i].abs() > threshold) {
          nonZero = true;
          break;
        }
      }
      final rankHost = NDArray.scalar(nonZero ? 1 : 0, dtype: DType.int64);
      return _writeOrWrapResult(rankHost, a.device, out);
    }

    final rank = f64Input.rank;
    final m = f64Input.shape[rank - 2];
    final n = f64Input.shape[rank - 1];
    final k = math.min(m, n);
    final ({NDArray<Float64> u, NDArray<Float64> s, NDArray<Float64> vh})
    rawSvd;
    try {
      rawSvd = nd.svd<Float64, Float64>(f64Input);
    } on nd.LinAlgException catch (error) {
      throw StateError(error.message);
    }
    final sPointer = rawSvd.s.pointer.cast<ffi.Double>();
    final rankHost = NDArray<Int64>.zeros(expectedShape, DType.int64);
    final rankPointer = rankHost.pointer.cast<ffi.Int64>();
    final batchCount = rankHost.size;

    for (var b = 0; b < batchCount; b++) {
      final offset = b * k;
      final maxSigma = k > 0 ? sPointer[offset] : 0.0;
      final cutoff =
          tolerance ?? (math.max(m, n) * 2.220446049250313e-16 * maxSigma);
      var count = 0;
      for (var i = 0; i < k; i++) {
        if (sPointer[offset + i] > cutoff) {
          count++;
        }
      }
      rankPointer[b] = count;
    }

    return _writeOrWrapResult(rankHost, a.device, out);
  });
}

/// Snake-case alias for [matrixRank].
GpuArray<Int64> matrix_rank<T extends DTypeTag>(
  GpuArray<T> a, {
  double? tolerance,
  GpuArray<Int64>? out,
}) => matrixRank(a, tolerance: tolerance, out: out);

/// Computes a vector or matrix norm of [a].
///
/// Always returns a [GpuArray] of [Float64] (a 0D scalar array with shape `[]`
/// when [axis] is `null` and [keepDims] is `false`, or a reduced tensor along
/// [axis]).
///
/// The [axis] parameter may be `null`, an `int` (for vector norms along a
/// single axis), or a 2-element `List<int>` (for 2D matrix norms along two
/// axes). Neither [a] nor [out] (if provided) may be disposed.
GpuArray<Float64> norm<T extends DTypeTag>(
  GpuArray<T> a, {
  NormOrd? ord,
  Object? axis,
  bool keepDims = false,
  GpuArray<Float64>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute norm on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write norm result to a disposed output GpuArray.');
  }

  if (a.rank == 0) {
    if (axis != null) {
      throw ArgumentError.value(
        axis,
        'axis',
        'Must be null for 0-dimensional input.',
      );
    }
    if (ord != null && ord.isMatrixOnly) {
      throw ArgumentError.value(
        ord,
        'ord',
        'Must not use a matrix-only norm on a 0-dimensional array.',
      );
    }
    if (out != null) {
      _validateOutBuffer(out, const <int>[], DType.float64);
    }
    return NDArray.scope(() {
      final f64Input = _toContiguousFloat64(a.toNDArray());
      final absScalar = f64Input.pointer.cast<ffi.Double>()[0].abs();
      final hostResult = NDArray.scalar(absScalar, dtype: DType.float64);
      return _writeOrWrapResult(hostResult, a.device, out);
    });
  }

  final isVectorReduction = (axis is int) || (axis == null && a.rank == 1);
  if (isVectorReduction && ord != null && ord.isMatrixOnly) {
    throw ArgumentError.value(
      ord,
      'ord',
      'Must not use a matrix-only norm (${ord.name}) for a 1D vector norm.',
    );
  }

  return NDArray.scope(() {
    final f64Input = _toContiguousFloat64(a.toNDArray());
    final NDArray<Float64> hostResult;
    try {
      hostResult = nd.norm<Float64>(
        f64Input,
        ord: ord?.ndKind,
        axis: axis,
        keepdims: keepDims,
      );
    } on nd.LinAlgException catch (error) {
      throw StateError(error.message);
    }
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the condition number of a 2D matrix or batch of matrices [a] with
/// respect to [ord] (defaulting to [NormOrd.l2]).
///
/// Always returns a [GpuArray] of [Float64] (shape `[]` for a 2D matrix, or
/// batch shape `a.shape.sublist(0, a.rank - 2)` for batched matrices).
///
/// Neither [a] nor [out] (if provided) may be disposed, and [a] must have at
/// least 2 dimensions.
GpuArray<Float64> cond<T extends DTypeTag>(
  GpuArray<T> a, {
  NormOrd ord = NormOrd.l2,
  GpuArray<Float64>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cond on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write cond result to a disposed output GpuArray.');
  }
  if (a.rank < 2) {
    throw GpuShapeMismatchException('cond', a.shape, const <int>[]);
  }
  if (ord == NormOrd.nuclear) {
    throw ArgumentError.value(
      ord,
      'ord',
      'Must not be NormOrd.nuclear for condition number computation.',
    );
  }
  final rank = a.rank;
  final m = a.shape[rank - 2];
  final n = a.shape[rank - 1];
  if (ord != NormOrd.l2 && ord != NormOrd.negL2 && m != n) {
    throw GpuShapeMismatchException('cond', a.shape, const <int>[]);
  }
  final expectedShape = a.shape.sublist(0, rank - 2);
  if (out != null) {
    _validateOutBuffer(out, expectedShape, DType.float64);
  }

  return NDArray.scope(() {
    final f64Input = _toContiguousFloat64(a.toNDArray());
    final NDArray<Float64> hostResult;
    try {
      hostResult = nd.cond<Float64>(f64Input, p: ord.ndKind);
    } on nd.LinAlgException catch (error) {
      throw StateError(error.message);
    }
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the matrix chain product of two or more arrays in [arrays] using
/// dynamic programming to select the optimal parenthesization.
///
/// The [arrays] list must contain at least 2 non-disposed arrays with
/// compatible inner dimensions. If [out] is provided, it must not be disposed.
GpuArray<T> multiDot<T extends DTypeTag>(
  List<GpuArray<T>> arrays, {
  GpuArray<T>? out,
}) {
  for (final array in arrays) {
    if (array.isDisposed) {
      throw StateError('Cannot execute multiDot on a disposed GpuArray.');
    }
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write multiDot result to a disposed output GpuArray.',
    );
  }
  if (arrays.length < 2) {
    throw ArgumentError.value(
      arrays,
      'arrays',
      'Must be a list of at least 2 arrays.',
    );
  }

  final count = arrays.length;
  for (var i = 0; i < count; i++) {
    final rank = arrays[i].rank;
    if (i == 0 || i == count - 1) {
      if (rank != 1 && rank != 2) {
        throw GpuShapeMismatchException(
          'multiDot',
          arrays[i].shape,
          const <int>[],
        );
      }
    } else if (rank != 2) {
      throw GpuShapeMismatchException(
        'multiDot',
        arrays[i].shape,
        const <int>[],
      );
    }
  }

  var previousCols = arrays[0].rank == 1
      ? arrays[0].shape[0]
      : arrays[0].shape[1];
  for (var i = 1; i < count; i++) {
    final currentRows = arrays[i].shape[0];
    if (currentRows != previousCols) {
      throw GpuShapeMismatchException(
        'multiDot',
        arrays[i - 1].shape,
        arrays[i].shape,
      );
    }
    if (i < count - 1) {
      previousCols = arrays[i].shape[1];
    }
  }

  return NDArray.scope(() {
    final hostArrays = <NDArray<T>>[
      for (final array in arrays) array.toNDArray(),
    ];
    final hostResult = nd.multi_dot<T>(hostArrays);
    return _writeOrWrapResult(hostResult, arrays.first.device, out);
  });
}

/// Snake-case alias for [multiDot].
GpuArray<T> multi_dot<T extends DTypeTag>(
  List<GpuArray<T>> arrays, {
  GpuArray<T>? out,
}) => multiDot(arrays, out: out);
