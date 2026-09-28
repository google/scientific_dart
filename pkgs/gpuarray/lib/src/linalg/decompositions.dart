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
    show Complex128, DType, DTypeTag, Float64, Int32, NDArray, Slice;

import '../device.dart';
import '../exceptions.dart';
import '../gpu_array.dart';

/// Mode for QR matrix decomposition.
enum QrMode {
  /// Reduced (thin) QR decomposition where $Q$ has shape `(m, min(m, n))`
  /// and $R$ has shape `(min(m, n), n)`.
  reduced(isReduced: true),

  /// Complete (full) QR decomposition where $Q$ has shape `(m, m)` and $R$
  /// has shape `(m, n)`.
  complete(isReduced: false);

  const QrMode({required this.isReduced});

  /// Whether this mode produces the reduced (thin) factorization.
  final bool isReduced;

  /// Computes the inner factor dimension ($\min(m, n)$ for [reduced], $m$ for
  /// [complete]) for a matrix with [rows] rows and [cols] columns.
  int factorDimension(int rows, int cols) =>
      isReduced ? math.min(rows, cols) : rows;
}

/// Matrix triangle selection for symmetric, Hermitian, and Cholesky
/// operations.
enum MatrixTriangle {
  /// Uses the lower triangular portion of the matrix.
  lower(isUpper: false),

  /// Uses the upper triangular portion of the matrix.
  upper(isUpper: true);

  const MatrixTriangle({required this.isUpper});

  /// Whether the upper triangle is selected.
  final bool isUpper;
}

/// Resource disposal extension for the record returned by [svd].
extension SvdRecordDispose<T extends DTypeTag>
    on ({GpuArray<T> u, GpuArray<T> s, GpuArray<T> vt}) {
  /// Disposes [u], [s], and [vt].
  void dispose() {
    u.dispose();
    s.dispose();
    vt.dispose();
  }
}

/// Resource disposal extension for the record returned by [qr].
extension QrRecordDispose<T extends DTypeTag>
    on ({GpuArray<T> q, GpuArray<T> r}) {
  /// Disposes [q] and [r].
  void dispose() {
    q.dispose();
    r.dispose();
  }
}

/// Resource disposal extension for the record returned by [eigh] and [eig].
extension EigRecordDispose<T extends DTypeTag, V extends DTypeTag>
    on ({GpuArray<T> eigenvalues, GpuArray<V> eigenvectors}) {
  /// Disposes [eigenvalues] and [eigenvectors].
  void dispose() {
    eigenvalues.dispose();
    eigenvectors.dispose();
  }
}

/// Resource disposal extension for the record returned by [lu].
extension LuRecordDispose<T extends DTypeTag>
    on ({GpuArray<T> p, GpuArray<T> l, GpuArray<T> u}) {
  /// Disposes [p], [l], and [u].
  void dispose() {
    p.dispose();
    l.dispose();
    u.dispose();
  }
}

/// Resource disposal extension for the record returned by [luFactor].
extension LuFactorRecordDispose<T extends DTypeTag>
    on ({GpuArray<T> lu, GpuArray<Int32> pivots}) {
  /// Disposes [lu] and [pivots].
  void dispose() {
    this.lu.dispose();
    pivots.dispose();
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

/// Computes the Singular Value Decomposition (SVD) of a 2D matrix or batch of
/// matrices [a]: $A = U \cdot \text{diag}(S) \cdot V^T$.
///
/// If [fullMatrices] is `true` (the default), `u` has shape `(..., m, m)` and
/// `vt` has shape `(..., n, n)`. If [fullMatrices] is `false`, `u` has shape
/// `(..., m, k)` and `vt` has shape `(..., k, n)` where $k = \min(m, n)$.
///
/// Neither [a] nor any array in [out] may be disposed, and [a] must have at
/// least 2 dimensions.
({GpuArray<T> u, GpuArray<T> s, GpuArray<T> vt}) svd<T extends DTypeTag>(
  GpuArray<T> a, {
  bool fullMatrices = true,
  ({GpuArray<T> u, GpuArray<T> s, GpuArray<T> vt})? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute svd on a disposed GpuArray.');
  }
  if (out != null &&
      (out.u.isDisposed || out.s.isDisposed || out.vt.isDisposed)) {
    throw StateError('Cannot write svd result to a disposed output GpuArray.');
  }
  if (a.rank < 2) {
    throw GpuShapeMismatchException('svd', a.shape, const <int>[]);
  }

  final rank = a.rank;
  final m = a.shape[rank - 2];
  final n = a.shape[rank - 1];
  final k = math.min(m, n);
  final batchShape = a.shape.sublist(0, rank - 2);
  final expectedUShape = <int>[...batchShape, m, fullMatrices ? m : k];
  final expectedSShape = <int>[...batchShape, k];
  final expectedVtShape = <int>[...batchShape, fullMatrices ? n : k, n];

  if (out != null) {
    _validateOutBuffer(out.u, expectedUShape, a.dtype, name: 'out.u');
    _validateOutBuffer(out.s, expectedSShape, a.dtype, name: 'out.s');
    _validateOutBuffer(out.vt, expectedVtShape, a.dtype, name: 'out.vt');
  }

  return NDArray.scope(() {
    final hostInput = a.toNDArray();
    final f64Input = _toContiguousFloat64(hostInput);
    final ({NDArray<Float64> u, NDArray<Float64> s, NDArray<Float64> vh})
    rawSvd;
    try {
      rawSvd = nd.svd<Float64, Float64>(f64Input);
    } on nd.LinAlgException catch (error) {
      throw StateError(error.message);
    }

    NDArray<Float64> uF64 = rawSvd.u;
    final NDArray<Float64> sF64 = rawSvd.s;
    NDArray<Float64> vtF64 = rawSvd.vh;

    if (!fullMatrices) {
      if (m > k) {
        final sliceSpecs = <Slice>[
          for (var i = 0; i < batchShape.length; i++) const Slice.all(),
          const Slice.all(),
          Slice(start: 0, stop: k),
        ];
        uF64 = uF64.slice(sliceSpecs).copy();
      }
      if (n > k) {
        final sliceSpecs = <Slice>[
          for (var i = 0; i < batchShape.length; i++) const Slice.all(),
          Slice(start: 0, stop: k),
          const Slice.all(),
        ];
        vtF64 = vtF64.slice(sliceSpecs).copy();
      }
    }

    final uHost = _matchDType<T>(uF64, a.dtype);
    final sHost = _matchDType<T>(sF64, a.dtype);
    final vtHost = _matchDType<T>(vtF64, a.dtype);

    final uGpu = _writeOrWrapResult(
      uHost,
      a.device,
      out?.u,
      outName: 'out.u',
      promote: false,
    );
    final sGpu = _writeOrWrapResult(
      sHost,
      a.device,
      out?.s,
      outName: 'out.s',
      promote: false,
    );
    final vtGpu = _writeOrWrapResult(
      vtHost,
      a.device,
      out?.vt,
      outName: 'out.vt',
      promote: false,
    );
    if (out == null) {
      uGpu.detachToParentScope();
      sGpu.detachToParentScope();
      vtGpu.detachToParentScope();
    }
    return (u: uGpu, s: sGpu, vt: vtGpu);
  });
}

/// Computes only the singular values of a 2D matrix or batch of matrices [a]
/// in descending order.
///
/// Neither [a] nor [out] (if provided) may be disposed, and [a] must have at
/// least 2 dimensions.
GpuArray<T> svdvals<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute svdvals on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write svdvals result to a disposed output GpuArray.',
    );
  }
  if (a.rank < 2) {
    throw GpuShapeMismatchException('svdvals', a.shape, const <int>[]);
  }
  final rank = a.rank;
  final k = math.min(a.shape[rank - 2], a.shape[rank - 1]);
  final expectedShape = <int>[...a.shape.sublist(0, rank - 2), k];
  if (out != null) {
    _validateOutBuffer(out, expectedShape, a.dtype);
  }

  return NDArray.scope(() {
    final hostInput = a.toNDArray();
    final f64Input = _toContiguousFloat64(hostInput);
    final ({NDArray<Float64> u, NDArray<Float64> s, NDArray<Float64> vh})
    rawSvd;
    try {
      rawSvd = nd.svd<Float64, Float64>(f64Input);
    } on nd.LinAlgException catch (error) {
      throw StateError(error.message);
    }
    final sHost = _matchDType<T>(rawSvd.s, a.dtype);
    return _writeOrWrapResult(sHost, a.device, out);
  });
}

/// Computes the QR decomposition of a 2D matrix [a]: $A = Q \cdot R$.
///
/// Uses Householder reflections. When [mode] is [QrMode.reduced] (default),
/// `q` has shape `(m, k)` and `r` has shape `(k, n)` where $k = \min(m, n)$.
/// When [mode] is [QrMode.complete], `q` has shape `(m, m)` and `r` has shape
/// `(m, n)`.
///
/// Neither [a] nor any array in [out] may be disposed, and [a] must be
/// 2-dimensional.
({GpuArray<T> q, GpuArray<T> r}) qr<T extends DTypeTag>(
  GpuArray<T> a, {
  QrMode mode = QrMode.reduced,
  ({GpuArray<T> q, GpuArray<T> r})? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute qr on a disposed GpuArray.');
  }
  if (out != null && (out.q.isDisposed || out.r.isDisposed)) {
    throw StateError('Cannot write qr result to a disposed output GpuArray.');
  }
  if (a.rank != 2) {
    throw GpuShapeMismatchException('qr', a.shape, const <int>[]);
  }

  final m = a.shape[0];
  final n = a.shape[1];
  final k = math.min(m, n);
  final factorDimension = mode.factorDimension(m, n);
  final expectedQShape = <int>[m, factorDimension];
  final expectedRShape = <int>[factorDimension, n];

  if (out != null) {
    _validateOutBuffer(out.q, expectedQShape, a.dtype, name: 'out.q');
    _validateOutBuffer(out.r, expectedRShape, a.dtype, name: 'out.r');
  }

  return NDArray.scope(() {
    final hostInput = a.toNDArray();
    final rWork = _toContiguousFloat64(hostInput).copy();
    final rWorkPointer = rWork.pointer.cast<ffi.Double>();

    final qFull = NDArray<Float64>.zeros(<int>[m, m], DType.float64);
    final qFullPointer = qFull.pointer.cast<ffi.Double>();
    for (var i = 0; i < m; i++) {
      qFullPointer[i * m + i] = 1.0;
    }

    final householderVector = List<double>.filled(m, 0.0);

    for (var col = 0; col < k; col++) {
      var normX = 0.0;
      for (var row = col; row < m; row++) {
        final entry = rWorkPointer[row * n + col];
        normX += entry * entry;
      }
      normX = math.sqrt(normX);
      if (normX < 1e-15) continue;

      final x0 = rWorkPointer[col * n + col];
      final sign = x0 >= 0.0 ? 1.0 : -1.0;
      final u1 = x0 + sign * normX;

      for (var i = 0; i < col; i++) {
        householderVector[i] = 0.0;
      }
      householderVector[col] = 1.0;
      var vNormSq = 1.0;
      for (var row = col + 1; row < m; row++) {
        final scaled = rWorkPointer[row * n + col] / u1;
        householderVector[row] = scaled;
        vNormSq += scaled * scaled;
      }
      final tau = 2.0 / vNormSq;

      for (var j = col; j < n; j++) {
        var dotProduct = 0.0;
        for (var row = col; row < m; row++) {
          dotProduct += householderVector[row] * rWorkPointer[row * n + j];
        }
        final factor = tau * dotProduct;
        for (var row = col; row < m; row++) {
          rWorkPointer[row * n + j] -= factor * householderVector[row];
        }
      }

      for (var row = 0; row < m; row++) {
        var dotProduct = 0.0;
        for (var j = col; j < m; j++) {
          dotProduct += qFullPointer[row * m + j] * householderVector[j];
        }
        final factor = tau * dotProduct;
        for (var j = col; j < m; j++) {
          qFullPointer[row * m + j] -= factor * householderVector[j];
        }
      }
    }

    for (var i = 0; i < m; i++) {
      final limit = math.min(i, n);
      for (var j = 0; j < limit; j++) {
        rWorkPointer[i * n + j] = 0.0;
      }
    }

    final qOutF64 = factorDimension == m
        ? qFull
        : qFull.slice(<Slice>[
            const Slice.all(),
            Slice(start: 0, stop: factorDimension),
          ]).copy();
    final rOutF64 = factorDimension == m
        ? rWork
        : rWork.slice(<Slice>[
            Slice(start: 0, stop: factorDimension),
            const Slice.all(),
          ]).copy();

    final qHost = _matchDType<T>(qOutF64, a.dtype);
    final rHost = _matchDType<T>(rOutF64, a.dtype);

    final qGpu = _writeOrWrapResult(
      qHost,
      a.device,
      out?.q,
      outName: 'out.q',
      promote: false,
    );
    final rGpu = _writeOrWrapResult(
      rHost,
      a.device,
      out?.r,
      outName: 'out.r',
      promote: false,
    );
    if (out == null) {
      qGpu.detachToParentScope();
      rGpu.detachToParentScope();
    }
    return (q: qGpu, r: rGpu);
  });
}

/// Computes the Cholesky decomposition of a symmetric positive-definite matrix
/// [a].
///
/// By default ([uplo] == [MatrixTriangle.lower] and [upper] == `false`),
/// returns the lower-triangular factor $L$ such that $A = L \cdot L^T$.
/// When [uplo] is [MatrixTriangle.upper] or [upper] is `true`, returns the
/// upper-triangular factor $U$ such that $A = U^T \cdot U$.
///
/// Neither [a] nor [out] (if provided) may be disposed, and [a] must be a
/// square 2D matrix that is symmetric positive-definite.
GpuArray<T> cholesky<T extends DTypeTag>(
  GpuArray<T> a, {
  MatrixTriangle uplo = MatrixTriangle.lower,
  bool upper = false,
  GpuArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cholesky on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write cholesky result to a disposed output GpuArray.',
    );
  }
  if (a.rank < 2 || a.shape[a.rank - 2] != a.shape[a.rank - 1]) {
    throw GpuShapeMismatchException('cholesky', a.shape, const <int>[]);
  }
  if (out != null) {
    _validateOutBuffer(out, a.shape, a.dtype);
  }

  final useUpper = uplo.isUpper || upper;
  return NDArray.scope(() {
    final hostInput = a.toNDArray();
    final f64Input = _toContiguousFloat64(hostInput);
    final NDArray<Float64> lowerFactor;
    try {
      lowerFactor = nd.cholesky<Float64>(f64Input);
    } on nd.LinAlgException catch (error) {
      throw StateError(error.message);
    }

    final NDArray<Float64> factorF64;
    if (useUpper) {
      final rank = f64Input.rank;
      final axes = List<int>.generate(rank, (index) => index);
      axes[rank - 2] = rank - 1;
      axes[rank - 1] = rank - 2;
      factorF64 = lowerFactor.transpose(axes).copy();
    } else {
      factorF64 = lowerFactor;
    }

    final hostResult = _matchDType<T>(factorF64, a.dtype);
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the eigenvalues and eigenvectors of a real symmetric (or Hermitian)
/// matrix [a]: $A \cdot V = V \cdot \text{diag}(\lambda)$.
///
/// Returns a record `(eigenvalues, eigenvectors)` where `eigenvalues` are in
/// ascending order and the columns of `eigenvectors` are the corresponding
/// orthonormal eigenvectors.
///
/// Neither [a] nor any array in [out] may be disposed, and [a] must be square
/// in its last two dimensions.
({GpuArray<T> eigenvalues, GpuArray<T> eigenvectors}) eigh<T extends DTypeTag>(
  GpuArray<T> a, {
  MatrixTriangle uplo = MatrixTriangle.lower,
  ({GpuArray<T> eigenvalues, GpuArray<T> eigenvectors})? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute eigh on a disposed GpuArray.');
  }
  if (out != null &&
      (out.eigenvalues.isDisposed || out.eigenvectors.isDisposed)) {
    throw StateError('Cannot write eigh result to a disposed output GpuArray.');
  }
  if (a.rank < 2 || a.shape[a.rank - 2] != a.shape[a.rank - 1]) {
    throw GpuShapeMismatchException('eigh', a.shape, const <int>[]);
  }

  final rank = a.rank;
  final n = a.shape[rank - 1];
  final batchShape = a.shape.sublist(0, rank - 2);
  final expectedValuesShape = <int>[...batchShape, n];
  final expectedVectorsShape = <int>[...batchShape, n, n];

  if (out != null) {
    _validateOutBuffer(
      out.eigenvalues,
      expectedValuesShape,
      a.dtype,
      name: 'out.eigenvalues',
    );
    _validateOutBuffer(
      out.eigenvectors,
      expectedVectorsShape,
      a.dtype,
      name: 'out.eigenvectors',
    );
  }

  return NDArray.scope(() {
    final hostInput = a.toNDArray();
    final f64Input = _toContiguousFloat64(hostInput);
    final ({NDArray<Float64> eigenvalues, NDArray<Float64> eigenvectors})
    rawEigh;
    try {
      rawEigh = nd.eigh<Float64, Float64>(
        f64Input,
        uplo: uplo.isUpper ? nd.MatrixTriangle.upper : nd.MatrixTriangle.lower,
      );
    } on nd.LinAlgException catch (error) {
      throw StateError(error.message);
    }

    final valuesHost = _matchDType<T>(rawEigh.eigenvalues, a.dtype);
    final vectorsHost = _matchDType<T>(rawEigh.eigenvectors, a.dtype);

    final valuesGpu = _writeOrWrapResult(
      valuesHost,
      a.device,
      out?.eigenvalues,
      outName: 'out.eigenvalues',
      promote: false,
    );
    final vectorsGpu = _writeOrWrapResult(
      vectorsHost,
      a.device,
      out?.eigenvectors,
      outName: 'out.eigenvectors',
      promote: false,
    );
    if (out == null) {
      valuesGpu.detachToParentScope();
      vectorsGpu.detachToParentScope();
    }
    return (eigenvalues: valuesGpu, eigenvectors: vectorsGpu);
  });
}

/// Computes only the eigenvalues of a real symmetric (or Hermitian) matrix [a]
/// in ascending order.
///
/// Neither [a] nor [out] (if provided) may be disposed, and [a] must be square
/// in its last two dimensions.
GpuArray<T> eigvalsh<T extends DTypeTag>(
  GpuArray<T> a, {
  MatrixTriangle uplo = MatrixTriangle.lower,
  GpuArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute eigvalsh on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write eigvalsh result to a disposed output GpuArray.',
    );
  }
  if (a.rank < 2 || a.shape[a.rank - 2] != a.shape[a.rank - 1]) {
    throw GpuShapeMismatchException('eigvalsh', a.shape, const <int>[]);
  }

  final rank = a.rank;
  final n = a.shape[rank - 1];
  final expectedShape = <int>[...a.shape.sublist(0, rank - 2), n];
  if (out != null) {
    _validateOutBuffer(out, expectedShape, a.dtype);
  }

  return NDArray.scope(() {
    final hostInput = a.toNDArray();
    final f64Input = _toContiguousFloat64(hostInput);
    final NDArray<Float64> rawValues;
    try {
      rawValues = nd.eigvalsh<Float64>(
        f64Input,
        uplo: uplo.isUpper ? nd.MatrixTriangle.upper : nd.MatrixTriangle.lower,
      );
    } on nd.LinAlgException catch (error) {
      throw StateError(error.message);
    }
    final valuesHost = _matchDType<T>(rawValues, a.dtype);
    return _writeOrWrapResult(valuesHost, a.device, out);
  });
}

/// Computes the eigenvalues and right eigenvectors of a general square matrix
/// [a].
///
/// Because a general real matrix may have complex conjugate eigenvalue pairs,
/// both `eigenvalues` and `eigenvectors` are returned as [Complex128] arrays.
///
/// Neither [a] nor any array in [out] may be disposed, and [a] must be square
/// in its last two dimensions.
({GpuArray<Complex128> eigenvalues, GpuArray<Complex128> eigenvectors})
eig<T extends DTypeTag>(
  GpuArray<T> a, {
  ({GpuArray<Complex128> eigenvalues, GpuArray<Complex128> eigenvectors})? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute eig on a disposed GpuArray.');
  }
  if (out != null &&
      (out.eigenvalues.isDisposed || out.eigenvectors.isDisposed)) {
    throw StateError('Cannot write eig result to a disposed output GpuArray.');
  }
  if (a.rank < 2 || a.shape[a.rank - 2] != a.shape[a.rank - 1]) {
    throw GpuShapeMismatchException('eig', a.shape, const <int>[]);
  }

  final rank = a.rank;
  final n = a.shape[rank - 1];
  final batchShape = a.shape.sublist(0, rank - 2);
  final expectedValuesShape = <int>[...batchShape, n];
  final expectedVectorsShape = <int>[...batchShape, n, n];

  if (out != null) {
    _validateOutBuffer(
      out.eigenvalues,
      expectedValuesShape,
      DType.complex128,
      name: 'out.eigenvalues',
    );
    _validateOutBuffer(
      out.eigenvectors,
      expectedVectorsShape,
      DType.complex128,
      name: 'out.eigenvectors',
    );
  }

  return NDArray.scope(() {
    final hostInput = a.toNDArray();
    final f64Input = _toContiguousFloat64(hostInput);
    final ({NDArray<Complex128> eigenvalues, NDArray<Complex128> eigenvectors})
    rawEig;
    try {
      rawEig = nd.eig<Complex128>(f64Input);
    } on nd.LinAlgException catch (error) {
      throw StateError(error.message);
    }

    final valuesGpu = _writeOrWrapResult(
      rawEig.eigenvalues,
      a.device,
      out?.eigenvalues,
      outName: 'out.eigenvalues',
      promote: false,
    );
    final vectorsGpu = _writeOrWrapResult(
      rawEig.eigenvectors,
      a.device,
      out?.eigenvectors,
      outName: 'out.eigenvectors',
      promote: false,
    );
    if (out == null) {
      valuesGpu.detachToParentScope();
      vectorsGpu.detachToParentScope();
    }
    return (eigenvalues: valuesGpu, eigenvectors: vectorsGpu);
  });
}

/// Computes the eigenvalues of a general square matrix [a] as a [Complex128]
/// array.
///
/// Neither [a] nor [out] (if provided) may be disposed, and [a] must be square
/// in its last two dimensions.
GpuArray<Complex128> eigvals<T extends DTypeTag>(
  GpuArray<T> a, {
  GpuArray<Complex128>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute eigvals on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write eigvals result to a disposed output GpuArray.',
    );
  }
  if (a.rank < 2 || a.shape[a.rank - 2] != a.shape[a.rank - 1]) {
    throw GpuShapeMismatchException('eigvals', a.shape, const <int>[]);
  }

  final rank = a.rank;
  final n = a.shape[rank - 1];
  final expectedShape = <int>[...a.shape.sublist(0, rank - 2), n];
  if (out != null) {
    _validateOutBuffer(out, expectedShape, DType.complex128);
  }

  return NDArray.scope(() {
    final hostInput = a.toNDArray();
    final f64Input = _toContiguousFloat64(hostInput);
    final NDArray<Complex128> rawValues;
    try {
      rawValues = nd.eigvals<Complex128>(f64Input);
    } on nd.LinAlgException catch (error) {
      throw StateError(error.message);
    }
    return _writeOrWrapResult(rawValues, a.device, out);
  });
}

/// Computes the pivoted LU decomposition of a 2D matrix [a] such that
/// $A = P \cdot L \cdot U$.
///
/// Returns a record `(p, l, u)` where `p` is the $(m, m)$ permutation matrix,
/// `l` is the $(m, k)$ lower-triangular matrix with unit diagonal, and `u` is
/// the $(k, n)$ upper-triangular matrix, with $k = \min(m, n)$.
///
/// Neither [a] nor any array in [out] may be disposed, and [a] must be
/// 2-dimensional.
({GpuArray<T> p, GpuArray<T> l, GpuArray<T> u}) lu<T extends DTypeTag>(
  GpuArray<T> a, {
  ({GpuArray<T> p, GpuArray<T> l, GpuArray<T> u})? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute lu on a disposed GpuArray.');
  }
  if (out != null &&
      (out.p.isDisposed || out.l.isDisposed || out.u.isDisposed)) {
    throw StateError('Cannot write lu result to a disposed output GpuArray.');
  }
  if (a.rank != 2) {
    throw GpuShapeMismatchException('lu', a.shape, const <int>[]);
  }

  final m = a.shape[0];
  final n = a.shape[1];
  final k = math.min(m, n);

  if (out != null) {
    _validateOutBuffer(out.p, <int>[m, m], a.dtype, name: 'out.p');
    _validateOutBuffer(out.l, <int>[m, k], a.dtype, name: 'out.l');
    _validateOutBuffer(out.u, <int>[k, n], a.dtype, name: 'out.u');
  }

  return NDArray.scope(() {
    final hostInput = a.toNDArray();
    final work = _toContiguousFloat64(hostInput).copy();
    final workPointer = work.pointer.cast<ffi.Double>();
    final permutationIndices = List<int>.generate(m, (index) => index);

    for (var j = 0; j < k; j++) {
      var maxRow = j;
      var maxAbs = workPointer[j * n + j].abs();
      for (var i = j + 1; i < m; i++) {
        final candidate = workPointer[i * n + j].abs();
        if (candidate > maxAbs) {
          maxAbs = candidate;
          maxRow = i;
        }
      }

      if (maxRow != j) {
        final tempIndex = permutationIndices[j];
        permutationIndices[j] = permutationIndices[maxRow];
        permutationIndices[maxRow] = tempIndex;
        for (var col = 0; col < n; col++) {
          final temp = workPointer[j * n + col];
          workPointer[j * n + col] = workPointer[maxRow * n + col];
          workPointer[maxRow * n + col] = temp;
        }
      }

      final pivotValue = workPointer[j * n + j];
      if (pivotValue.abs() > 1e-15) {
        for (var i = j + 1; i < m; i++) {
          workPointer[i * n + j] /= pivotValue;
          final multiplier = workPointer[i * n + j];
          for (var col = j + 1; col < n; col++) {
            workPointer[i * n + col] -= multiplier * workPointer[j * n + col];
          }
        }
      }
    }

    final pF64 = NDArray<Float64>.zeros(<int>[m, m], DType.float64);
    final lF64 = NDArray<Float64>.zeros(<int>[m, k], DType.float64);
    final uF64 = NDArray<Float64>.zeros(<int>[k, n], DType.float64);

    final pPointer = pF64.pointer.cast<ffi.Double>();
    final lPointer = lF64.pointer.cast<ffi.Double>();
    final uPointer = uF64.pointer.cast<ffi.Double>();

    for (var i = 0; i < m; i++) {
      pPointer[permutationIndices[i] * m + i] = 1.0;
    }

    for (var i = 0; i < m; i++) {
      for (var j = 0; j < k; j++) {
        if (i == j) {
          lPointer[i * k + j] = 1.0;
        } else if (i > j) {
          lPointer[i * k + j] = workPointer[i * n + j];
        }
      }
    }

    for (var i = 0; i < k; i++) {
      for (var j = i; j < n; j++) {
        uPointer[i * n + j] = workPointer[i * n + j];
      }
    }

    final pHost = _matchDType<T>(pF64, a.dtype);
    final lHost = _matchDType<T>(lF64, a.dtype);
    final uHost = _matchDType<T>(uF64, a.dtype);

    final pGpu = _writeOrWrapResult(
      pHost,
      a.device,
      out?.p,
      outName: 'out.p',
      promote: false,
    );
    final lGpu = _writeOrWrapResult(
      lHost,
      a.device,
      out?.l,
      outName: 'out.l',
      promote: false,
    );
    final uGpu = _writeOrWrapResult(
      uHost,
      a.device,
      out?.u,
      outName: 'out.u',
      promote: false,
    );
    if (out == null) {
      pGpu.detachToParentScope();
      lGpu.detachToParentScope();
      uGpu.detachToParentScope();
    }
    return (p: pGpu, l: lGpu, u: uGpu);
  });
}

/// Computes the compact LU factorization of a square 2D matrix [a] for use with
/// [luSolve].
///
/// Returns a record `(lu, pivots)` where `lu` packs $L$ (strictly lower
/// triangle) and $U$ (upper triangle including diagonal) in a single `(n, n)`
/// matrix, and `pivots` is a 1D `Int32` array of row swap indices.
///
/// Neither [a] nor any array in [out] may be disposed, and [a] must be a
/// square 2D matrix.
({GpuArray<T> lu, GpuArray<Int32> pivots}) luFactor<T extends DTypeTag>(
  GpuArray<T> a, {
  ({GpuArray<T> lu, GpuArray<Int32> pivots})? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute luFactor on a disposed GpuArray.');
  }
  if (out != null && (out.lu.isDisposed || out.pivots.isDisposed)) {
    throw StateError(
      'Cannot write luFactor result to a disposed output GpuArray.',
    );
  }
  if (a.rank != 2 || a.shape[0] != a.shape[1]) {
    throw GpuShapeMismatchException('luFactor', a.shape, const <int>[]);
  }

  final n = a.shape[0];
  if (out != null) {
    _validateOutBuffer(out.lu, <int>[n, n], a.dtype, name: 'out.lu');
    _validateOutBuffer(out.pivots, <int>[n], DType.int32, name: 'out.pivots');
  }

  return NDArray.scope(() {
    final hostInput = a.toNDArray();
    final luF64 = _toContiguousFloat64(hostInput).copy();
    final luPointer = luF64.pointer.cast<ffi.Double>();
    final pivotsNd = NDArray<Int32>.zeros(<int>[n], DType.int32);
    final pivotsPointer = pivotsNd.pointer.cast<ffi.Int32>();

    for (var j = 0; j < n; j++) {
      var maxRow = j;
      var maxAbs = luPointer[j * n + j].abs();
      for (var i = j + 1; i < n; i++) {
        final candidate = luPointer[i * n + j].abs();
        if (candidate > maxAbs) {
          maxAbs = candidate;
          maxRow = i;
        }
      }
      pivotsPointer[j] = maxRow;
      if (maxRow != j) {
        for (var col = 0; col < n; col++) {
          final temp = luPointer[j * n + col];
          luPointer[j * n + col] = luPointer[maxRow * n + col];
          luPointer[maxRow * n + col] = temp;
        }
      }
      final pivotValue = luPointer[j * n + j];
      if (pivotValue.abs() < 1e-15) {
        throw StateError('Matrix is singular in luFactor at column $j.');
      }
      for (var i = j + 1; i < n; i++) {
        luPointer[i * n + j] /= pivotValue;
        final multiplier = luPointer[i * n + j];
        for (var col = j + 1; col < n; col++) {
          luPointer[i * n + col] -= multiplier * luPointer[j * n + col];
        }
      }
    }

    final luHost = _matchDType<T>(luF64, a.dtype);
    final luGpu = _writeOrWrapResult(
      luHost,
      a.device,
      out?.lu,
      outName: 'out.lu',
      promote: false,
    );
    final pivotsGpu = _writeOrWrapResult(
      pivotsNd,
      a.device,
      out?.pivots,
      outName: 'out.pivots',
      promote: false,
    );
    if (out == null) {
      luGpu.detachToParentScope();
      pivotsGpu.detachToParentScope();
    }
    return (lu: luGpu, pivots: pivotsGpu);
  });
}

/// Snake-case alias for [luFactor].
({GpuArray<T> lu, GpuArray<Int32> pivots}) lu_factor<T extends DTypeTag>(
  GpuArray<T> a, {
  ({GpuArray<T> lu, GpuArray<Int32> pivots})? out,
}) => luFactor(a, out: out);

/// Solves a linear system $A x = b$ given the compact LU factorization
/// `(lu, pivots)` from [luFactor].
///
/// Supports 1D right-hand side vectors `(n,)` and 2D right-hand side matrices
/// `(n, k)`. None of [lu], [pivots], [b], or [out] may be disposed.
GpuArray<T> luSolve<T extends DTypeTag>(
  GpuArray<T> lu,
  GpuArray<Int32> pivots,
  GpuArray<T> b, {
  GpuArray<T>? out,
}) {
  if (lu.isDisposed || pivots.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute luSolve on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write luSolve result to a disposed output GpuArray.',
    );
  }
  if (lu.rank != 2 || lu.shape[0] != lu.shape[1]) {
    throw GpuShapeMismatchException('luSolve', lu.shape, b.shape);
  }
  final n = lu.shape[0];
  if (pivots.rank != 1 || pivots.shape[0] != n) {
    throw GpuShapeMismatchException('luSolve', lu.shape, pivots.shape);
  }
  if ((b.rank != 1 && b.rank != 2) || b.shape[0] != n) {
    throw GpuShapeMismatchException('luSolve', lu.shape, b.shape);
  }
  if (out != null) {
    _validateOutBuffer(out, b.shape, b.dtype);
  }

  final is1D = b.rank == 1;
  final rightHandSides = is1D ? 1 : b.shape[1];

  return NDArray.scope(() {
    final luF64 = _toContiguousFloat64(lu.toNDArray());
    final luPointer = luF64.pointer.cast<ffi.Double>();

    final pivotsHost = pivots.toNDArray();
    final contiguousPivots = pivotsHost.isContiguous
        ? pivotsHost
        : pivotsHost.copy();
    final pivotsPointer = contiguousPivots.pointer.cast<ffi.Int32>();

    final xF64 = _toContiguousFloat64(b.toNDArray()).copy().reshape(b.shape);
    final xPointer = xF64.pointer.cast<ffi.Double>();

    for (var i = 0; i < n; i++) {
      final pivotRow = pivotsPointer[i];
      if (pivotRow != i) {
        for (var col = 0; col < rightHandSides; col++) {
          final temp = xPointer[i * rightHandSides + col];
          xPointer[i * rightHandSides + col] =
              xPointer[pivotRow * rightHandSides + col];
          xPointer[pivotRow * rightHandSides + col] = temp;
        }
      }
    }

    for (var i = 1; i < n; i++) {
      for (var k = 0; k < i; k++) {
        final lowerEntry = luPointer[i * n + k];
        for (var col = 0; col < rightHandSides; col++) {
          xPointer[i * rightHandSides + col] -=
              lowerEntry * xPointer[k * rightHandSides + col];
        }
      }
    }

    for (var i = n - 1; i >= 0; i--) {
      final diagonalEntry = luPointer[i * n + i];
      if (diagonalEntry.abs() < 1e-15) {
        throw StateError('Matrix is singular in luSolve.');
      }
      for (var col = 0; col < rightHandSides; col++) {
        var sum = xPointer[i * rightHandSides + col];
        for (var k = i + 1; k < n; k++) {
          sum -= luPointer[i * n + k] * xPointer[k * rightHandSides + col];
        }
        xPointer[i * rightHandSides + col] = sum / diagonalEntry;
      }
    }

    final xHost = _matchDType<T>(xF64, b.dtype);
    return _writeOrWrapResult(xHost, b.device, out);
  });
}

/// Snake-case alias for [luSolve].
GpuArray<T> lu_solve<T extends DTypeTag>(
  GpuArray<T> lu,
  GpuArray<Int32> pivots,
  GpuArray<T> b, {
  GpuArray<T>? out,
}) => luSolve(lu, pivots, b, out: out);
