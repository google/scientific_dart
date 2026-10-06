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
import 'dart:math' as math;
import '../ndarray.dart';
import 'package:openblas/openblas.dart';
import 'dart:ffi' as ffi;
import '../scratch_arena.dart';
import '../exceptions.dart';
import '../ndarray_bindings.dart';

// Standalone operational relative cross-imports
import 'broadcasting.dart';
import 'math.dart';
import 'helpers.dart';

NDArray<DTypeTag> _createZeros(List<int> shape, DType<DTypeTag> dtype) =>
    switch (dtype) {
      DType.float64 => NDArray<Float64>.zeros(shape, DType.float64),
      DType.float32 => NDArray<Float32>.zeros(shape, DType.float32),
      DType.float16 => NDArray<Float16>.zeros(shape, DType.float16),
      DType.bfloat16 => NDArray<BFloat16>.zeros(shape, DType.bfloat16),
      DType.int64 => NDArray<Int64>.zeros(shape, DType.int64),
      DType.int32 => NDArray<Int32>.zeros(shape, DType.int32),
      DType.int16 => NDArray<Int16>.zeros(shape, DType.int16),
      DType.int8 => NDArray<Int8>.zeros(shape, DType.int8),
      DType.uint64 => NDArray<Uint64>.zeros(shape, DType.uint64),
      DType.uint32 => NDArray<Uint32>.zeros(shape, DType.uint32),
      DType.uint16 => NDArray<Uint16>.zeros(shape, DType.uint16),
      DType.uint8 => NDArray<Uint8>.zeros(shape, DType.uint8),
      DType.complex128 => NDArray<Complex128>.zeros(shape, DType.complex128),
      DType.complex64 => NDArray<Complex64>.zeros(shape, DType.complex64),
      DType.boolean => NDArray<Boolean>.zeros(shape, DType.boolean),
    };

(int, int) _physicalByteSpan(NDArray<DTypeTag> x) {
  if (x.size == 0) {
    return (x.pointer.address, x.pointer.address);
  }
  var minElemOffset = 0;
  var maxElemOffset = 0;
  for (var d = 0; d < x.shape.length; d++) {
    final stride = x.strides[d];
    final size = x.shape[d];
    if (stride > 0) {
      maxElemOffset += (size - 1) * stride;
    } else if (stride < 0) {
      minElemOffset += (size - 1) * stride;
    }
  }
  final byteWidth = x.dtype.byteWidth;
  final startAddr = x.pointer.address + minElemOffset * byteWidth;
  final endAddr = x.pointer.address + (maxElemOffset + 1) * byteWidth;
  return (startAddr, endAddr);
}

bool _isMemoryAliased(NDArray<DTypeTag> out, NDArray<DTypeTag> other) {
  if (identical(out, other) || out.pointer.address == other.pointer.address) {
    return true;
  }
  if (out.size == 0 || other.size == 0) return false;
  final (startOut, endOut) = _physicalByteSpan(out);
  final (startOther, endOther) = _physicalByteSpan(other);
  return startOut < endOther && startOther < endOut;
}

({bool hasNaN, bool hasInf}) _analyzeNonFinitePtr(
  ffi.Pointer<ffi.Void> ptr,
  int elementCount,
  DType dtype,
) {
  var hasNaN = false;
  var hasInf = false;
  switch (dtype) {
    case DType.float64:
      final p = ptr.cast<ffi.Double>();
      for (var i = 0; i < elementCount; i++) {
        final v = p[i];
        if (v.isNaN) {
          hasNaN = true;
        } else if (v.isInfinite) {
          hasInf = true;
        }
        if (hasNaN && hasInf) break;
      }
    case DType.float32:
      final p = ptr.cast<ffi.Float>();
      for (var i = 0; i < elementCount; i++) {
        final v = p[i];
        if (v.isNaN) {
          hasNaN = true;
        } else if (v.isInfinite) {
          hasInf = true;
        }
        if (hasNaN && hasInf) break;
      }
    case DType.complex128:
      final p = ptr.cast<ffi.Double>();
      final n = elementCount * 2;
      for (var i = 0; i < n; i++) {
        final v = p[i];
        if (v.isNaN) {
          hasNaN = true;
        } else if (v.isInfinite) {
          hasInf = true;
        }
        if (hasNaN && hasInf) break;
      }
    case DType.complex64:
      final p = ptr.cast<ffi.Float>();
      final n = elementCount * 2;
      for (var i = 0; i < n; i++) {
        final v = p[i];
        if (v.isNaN) {
          hasNaN = true;
        } else if (v.isInfinite) {
          hasInf = true;
        }
        if (hasNaN && hasInf) break;
      }
    default:
      throw UnsupportedError('Unsupported dtype: $dtype');
  }
  return (hasNaN: hasNaN, hasInf: hasInf);
}

({bool hasNaN, bool hasInf}) _analyzeNonFinite(NDArray<DTypeTag> arr) {
  if (!arr.dtype.isFloating && !arr.dtype.isComplex) {
    return (hasNaN: false, hasInf: false);
  }
  if (arr.isContiguous) {
    return _analyzeNonFinitePtr(arr.pointer, arr.size, arr.dtype);
  }
  final contig = arr.copy();
  try {
    return _analyzeNonFinitePtr(contig.pointer, contig.size, contig.dtype);
  } finally {
    contig.dispose();
  }
}

void _fillPtrWithNaN(ffi.Pointer<ffi.Void> ptr, int elementCount, DType dtype) {
  switch (dtype) {
    case DType.float64:
      final p = ptr.cast<ffi.Double>();
      for (var i = 0; i < elementCount; i++) {
        p[i] = double.nan;
      }
    case DType.float32:
      final p = ptr.cast<ffi.Float>();
      for (var i = 0; i < elementCount; i++) {
        p[i] = double.nan;
      }
    case DType.complex128:
      final p = ptr.cast<ffi.Double>();
      final n = elementCount * 2;
      for (var i = 0; i < n; i++) {
        p[i] = double.nan;
      }
    case DType.complex64:
      final p = ptr.cast<ffi.Float>();
      final n = elementCount * 2;
      for (var i = 0; i < n; i++) {
        p[i] = double.nan;
      }
    default:
      throw UnsupportedError('Unsupported dtype: $dtype');
  }
}

void _fillUpperTriangleWithNaN(
  ffi.Pointer<ffi.Void> ptr,
  int rows,
  int cols,
  DType dtype,
) {
  switch (dtype) {
    case DType.float64:
      final p = ptr.cast<ffi.Double>();
      for (var i = 0; i < rows; i++) {
        for (var j = i; j < cols; j++) {
          p[i * cols + j] = double.nan;
        }
      }
    case DType.float32:
      final p = ptr.cast<ffi.Float>();
      for (var i = 0; i < rows; i++) {
        for (var j = i; j < cols; j++) {
          p[i * cols + j] = double.nan;
        }
      }
    case DType.complex128:
      final p = ptr.cast<ffi.Double>();
      for (var i = 0; i < rows; i++) {
        for (var j = i; j < cols; j++) {
          final idx = (i * cols + j) * 2;
          p[idx] = double.nan;
          p[idx + 1] = double.nan;
        }
      }
    case DType.complex64:
      final p = ptr.cast<ffi.Float>();
      for (var i = 0; i < rows; i++) {
        for (var j = i; j < cols; j++) {
          final idx = (i * cols + j) * 2;
          p[idx] = double.nan;
          p[idx + 1] = double.nan;
        }
      }
    default:
      throw UnsupportedError('Unsupported dtype: $dtype');
  }
}

enum _LapackFailureKind {
  singularMatrix,
  nonPositiveDefinite,
  iterationsExceeded,
  general,
}

void _checkLapackInfo(
  int info,
  String routine, {
  _LapackFailureKind positiveKind = _LapackFailureKind.general,
  String? positiveMessage,
}) {
  if (info == 0) return;
  if (info < 0) {
    throw LinAlgException(
      '$routine failed (argument ${-info} had an illegal or non-finite value, info = $info).',
    );
  }
  final msg = positiveMessage ?? '$routine failed with info = $info.';
  switch (positiveKind) {
    case _LapackFailureKind.singularMatrix:
      throw SingularMatrixException(msg);
    case _LapackFailureKind.nonPositiveDefinite:
      throw NonPositiveDefiniteException(msg);
    case _LapackFailureKind.iterationsExceeded:
      throw IterationsExceededException(msg);
    case _LapackFailureKind.general:
      throw LinAlgException(msg);
  }
}

/// Matrix product of two arrays.
///
/// Behavior depends on the ranks of [a] and [b] in the same manner as NumPy's
/// `matmul`:
/// - If both arguments are 2-D, they are multiplied like conventional matrices.
/// - If either argument is N-D ($N > 2$), it is treated as a stack of matrices
///   residing in the last two indexes and broadcast accordingly.
/// - If the first argument is 1-D, it is promoted to a matrix by prepending a 1
///   to its dimensions; after matrix multiplication the prepended 1 is removed.
/// - If the second argument is 1-D, it is promoted to a matrix by appending a 1
///   to its dimensions; after matrix multiplication the appended 1 is removed.
/// - If both arguments are 1-D of length $K$, their inner product is returned
///   as a 0-D scalar array.
///
/// **Preconditions:**
/// - Neither [a], [b], nor [out] (if provided) may be disposed.
/// - Both [a] and [b] must have rank $\ge 1$ (0-D scalars are not allowed; use
///   `multiply` instead).
/// - Both [a] and [b] must have the same [DType].
/// - The last dimension of [a] must match the second-to-last dimension of [b]
///   (or the only dimension of [b] if [b] is 1-D).
/// - Leading batch dimensions of [a] and [b] must be broadcast-compatible.
/// - If [out] is provided, it must be writeable and match the output shape and
///   [DType].
///
/// **Throws:**
/// - It is an error if [a], [b], or [out] is disposed.
/// - It is an error if [a] or [b] is 0-dimensional, if their [DType]s differ,
///   or if their inner or batch dimensions are incompatible.
/// - It is an error if [out] is read-only or has an incompatible shape or
///   [DType].
///
/// **Performance considerations:**
/// - Dispatches 2-D and batched floating-point and complex matrix products to
///   OpenBLAS (`cblas_dgemm`, `cblas_sgemm`, `cblas_zgemm`, `cblas_cgemm`,
///   `cblas_dgemv`, etc.) with $O(M \cdot K \cdot N)$ complexity per matrix
///   slice.
/// - Uses vectorized C kernels (`native_matmul_2d` / `native_matmul_batched`)
///   for integer and boolean arrays.
///
/// **Example:**
/// {@example /example/linalg_example.dart lang=dart}
///
/// Reference: [NumPy matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
NDArray<T> matmul<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute matmul() on a disposed array.');
  }
  if (out != null) {
    if (out.isDisposed) {
      throw StateError(
        'Cannot write matmul result to a disposed output array.',
      );
    }
    validateOutBuffer(out);
  }
  if (a.rank == 0 || b.rank == 0) {
    throw ArgumentError.value(
      [a.shape, b.shape],
      'a, b',
      'Must not be 0D scalar arrays (matmul does not support 0D scalar arrays, got shapes ${a.shape} and ${b.shape}).',
    );
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }
  final targetDType = a.dtype;

  if (a.shape.length == 1 && b.shape.length == 1) {
    if (a.shape[0] != b.shape[0]) {
      throw ArgumentError.value(
        [a.shape, b.shape],
        'a, b',
        'Must have compatible vector dimensions for 1D dot product in matmul: ${a.shape} and ${b.shape}',
      );
    }
    if (targetDType.isFloating || targetDType.isComplex) {
      checkBlasIntDim(a.shape[0], 'n', 'matmul');
      checkBlasIntStride(a.strides[0], 'inca', 'matmul');
      checkBlasIntStride(b.strides[0], 'incb', 'matmul');
    }
  } else {
    final kA = a.shape[a.shape.length - 1];
    final kB = b.shape.length == 1 ? b.shape[0] : b.shape[b.shape.length - 2];
    if (kA != kB) {
      throw ArgumentError.value(
        [a.shape, b.shape],
        'a, b',
        'Must have compatible inner matrix dimensions for matmul: kA($kA) != kB($kB). Shapes: ${a.shape} and ${b.shape}',
      );
    }
    if (targetDType.isFloating || targetDType.isComplex) {
      final m = a.shape.length == 1 ? 1 : a.shape[a.shape.length - 2];
      final n = b.shape.length == 1 ? 1 : b.shape[b.shape.length - 1];
      checkBlasIntDim(m, 'm', 'matmul');
      checkBlasIntDim(kA, 'k', 'matmul');
      checkBlasIntDim(n, 'n', 'matmul');
      if (a.shape.length == 1) {
        checkBlasIntStride(a.strides[0], 'inca', 'matmul');
      } else {
        checkBlasIntStride(a.strides[a.shape.length - 2], 'lda', 'matmul');
        checkBlasIntStride(a.strides[a.shape.length - 1], 'lda', 'matmul');
      }
      if (b.shape.length == 1) {
        checkBlasIntStride(b.strides[0], 'incb', 'matmul');
      } else {
        checkBlasIntStride(b.strides[b.shape.length - 2], 'ldb', 'matmul');
        checkBlasIntStride(b.strides[b.shape.length - 1], 'ldb', 'matmul');
      }
    }
  }

  switch (targetDType) {
    case DType.float16:
    case DType.bfloat16:
      return NDArray.scope(() {
        final aF32 = castNDArray<Float32>(a, DType.float32);
        final bF32 = castNDArray<Float32>(b, DType.float32);
        final resF32 = matmul<Float32>(aF32, bF32);
        final res = castNDArray<T>(resF32, targetDType);
        if (out != null) {
          if (!listEquals(out.shape, res.shape) || out.dtype != targetDType) {
            throw ArgumentError.value(
              out,
              'out',
              'Must have compatible shape ${res.shape} and dtype $targetDType (provided out buffer has incompatible shape or dtype, got shape ${out.shape} and dtype ${out.dtype}).',
            );
          }
          res.copy(out: out);
          return out;
        }
        return res.detachToParentScope();
      });
    default:
      break;
  }

  NDArray? aCast;
  NDArray? bCast;
  NDArray? aCopy;
  NDArray? bCopy;
  NDArray? aPromotedView;
  NDArray? bPromotedView;
  NDArray<T>? result;
  var success = false;

  try {
    aCast = a.dtype == targetDType ? a : castNDArray(a, targetDType);
    bCast = b.dtype == targetDType ? b : castNDArray(b, targetDType);

    if (aCast.shape.length == 1 && bCast.shape.length == 1) {
      final n = aCast.shape[0];
      if (n != bCast.shape[0]) {
        throw ArgumentError.value(
          [aCast.shape, bCast.shape],
          'a, b',
          'Must have compatible vector dimensions for 1D dot product in matmul: ${aCast.shape} and ${bCast.shape}',
        );
      }
      if (out != null) {
        if (!listEquals(out.shape, []) || out.dtype != targetDType) {
          throw ArgumentError.value(
            out,
            'out',
            'Must have compatible shape [] and dtype $targetDType (provided out buffer has incompatible shape or dtype, got shape ${out.shape} and dtype ${out.dtype}).',
          );
        }
      }
      if (aCast.strides[0] == 0) {
        aCopy = aCast.copy();
      }
      if (bCast.strides[0] == 0) {
        bCopy = bCast.copy();
      }
      final aVec = aCopy ?? aCast;
      final bVec = bCopy ?? bCast;
      final incA = aVec.strides[0];
      final incB = bVec.strides[0];
      final offsetA = (n > 0 && incA < 0) ? (n - 1) * incA : 0;
      final offsetB = (n > 0 && incB < 0) ? (n - 1) * incB : 0;
      switch (targetDType) {
        case DType.float64:
          final scalarRes = cblas_ddot(
            n,
            aVec.pointer.cast<ffi.Double>() + offsetA,
            incA,
            bVec.pointer.cast<ffi.Double>() + offsetB,
            incB,
          );
          if (out != null) {
            out.pointer.cast<ffi.Double>()[0] = scalarRes;
            result = out;
          } else {
            result =
                (NDArray.scalar(scalarRes, dtype: DType.float64) as NDArray<T>);
          }
          success = true;
          return result;
        case DType.float32:
          final scalarRes = cblas_sdot(
            n,
            aVec.pointer.cast<ffi.Float>() + offsetA,
            incA,
            bVec.pointer.cast<ffi.Float>() + offsetB,
            incB,
          );
          if (out != null) {
            out.pointer.cast<ffi.Float>()[0] = scalarRes;
            result = out;
          } else {
            result =
                (NDArray.scalar(scalarRes, dtype: DType.float32) as NDArray<T>);
          }
          success = true;
          return result;
        case DType.complex128:
          final aPtr = aVec.pointer.cast<ffi.Double>();
          final bPtr = bVec.pointer.cast<ffi.Double>();
          var realSum = 0.0;
          var imagSum = 0.0;
          for (var i = 0; i < n; i++) {
            final ar = aPtr[i * incA * 2];
            final ai = aPtr[i * incA * 2 + 1];
            final br = bPtr[i * incB * 2];
            final bi = bPtr[i * incB * 2 + 1];
            realSum += ar * br - ai * bi;
            imagSum += ar * bi + ai * br;
          }
          final resVal = Complex(realSum, imagSum);
          if (out != null) {
            final outPtr = out.pointer.cast<ffi.Double>();
            outPtr[0] = realSum;
            outPtr[1] = imagSum;
            result = out;
          } else {
            result =
                (NDArray.scalar(resVal, dtype: DType.complex128) as NDArray<T>);
          }
          success = true;
          return result;
        case DType.complex64:
          final aPtr = aVec.pointer.cast<ffi.Float>();
          final bPtr = bVec.pointer.cast<ffi.Float>();
          var realSum = 0.0;
          var imagSum = 0.0;
          for (var i = 0; i < n; i++) {
            final ar = aPtr[i * incA * 2];
            final ai = aPtr[i * incA * 2 + 1];
            final br = bPtr[i * incB * 2];
            final bi = bPtr[i * incB * 2 + 1];
            realSum += ar * br - ai * bi;
            imagSum += ar * bi + ai * br;
          }
          final resVal = Complex(realSum, imagSum);
          if (out != null) {
            final outPtr = out.pointer.cast<ffi.Float>();
            outPtr[0] = realSum;
            outPtr[1] = imagSum;
            result = out;
          } else {
            result =
                (NDArray.scalar(resVal, dtype: DType.complex64) as NDArray<T>);
          }
          success = true;
          return result;
        default:
          break;
      }
    }

    bool needsCopyForGemm(NDArray arr) {
      if (arr.strides.any((s) => s < 0)) return true;
      if (arr.shape.length == 1) {
        return arr.strides[0] != 1;
      }
      final r = arr.shape.length;
      final rows = arr.shape[r - 2];
      final cols = arr.shape[r - 1];
      final s0 = arr.strides[r - 2];
      final s1 = arr.strides[r - 1];
      if (s0 <= 0 || s1 <= 0) return true;
      final validNoTrans = s1 == 1 && s0 >= math.max(1, cols);
      final validTrans = s0 == 1 && s1 >= math.max(1, rows);
      return !validNoTrans && !validTrans;
    }

    if (needsCopyForGemm(aCast)) {
      aCopy = aCast.copy();
    }
    if (needsCopyForGemm(bCast)) {
      bCopy = bCast.copy();
    }

    final aToUse = aCopy ?? aCast;
    final bToUse = bCopy ?? bCast;

    var aPromoted = false;
    var bPromoted = false;

    NDArray aView = aToUse;
    if (aToUse.shape.length == 1) {
      aPromotedView = NDArray.view(
        aToUse,
        shape: [1, aToUse.shape[0]],
        strides: [0, aToUse.strides[0]],
        offsetElements: 0,
      );
      aView = aPromotedView;
      aPromoted = true;
    }

    NDArray bView = bToUse;
    if (bToUse.shape.length == 1) {
      bPromotedView = NDArray.view(
        bToUse,
        shape: [bToUse.shape[0], 1],
        strides: [bToUse.strides[0], 0],
        offsetElements: 0,
      );
      bView = bPromotedView;
      bPromoted = true;
    }

    final rankA = aView.shape.length;
    final rankB = bView.shape.length;

    final m = aView.shape[rankA - 2];
    final kA = aView.shape[rankA - 1];
    final kB = bView.shape[rankB - 2];
    final n = bView.shape[rankB - 1];

    if (kA != kB) {
      throw ArgumentError.value(
        [aCast.shape, bCast.shape],
        'a, b',
        'Must have compatible inner matrix dimensions for matmul: kA($kA) != kB($kB). Shapes: ${aCast.shape} and ${bCast.shape}',
      );
    }

    final stackA = aView.shape.sublist(0, rankA - 2);
    final stackB = bView.shape.sublist(0, rankB - 2);
    final broadcastStack = broadcastStackShapes(stackA, stackB);

    final expectedFinalShape = <int>[];
    if (aPromoted && bPromoted) {
      // empty []
    } else if (aPromoted) {
      expectedFinalShape.addAll([...broadcastStack, n]);
    } else if (bPromoted) {
      expectedFinalShape.addAll([...broadcastStack, m]);
    } else {
      expectedFinalShape.addAll([...broadcastStack, m, n]);
    }

    if (out != null) {
      if (!listEquals(out.shape, expectedFinalShape) ||
          out.dtype != targetDType) {
        throw ArgumentError.value(
          out,
          'out',
          'Must have compatible shape $expectedFinalShape and dtype $targetDType (provided out buffer has incompatible shape or dtype, got shape ${out.shape} and dtype ${out.dtype}).',
        );
      }
    }

    final resShape = [...broadcastStack, m, n];
    final bool isAliased =
        out != null &&
        (sharesMemory(a, out) ||
            sharesMemory(b, out) ||
            _isMemoryAliased(out, a) ||
            _isMemoryAliased(out, b) ||
            _isMemoryAliased(out, aToUse) ||
            _isMemoryAliased(out, bToUse));
    final bool canUseOutDirectly =
        out != null &&
        !isAliased &&
        out.isContiguous &&
        listEquals(out.shape, resShape);
    result = canUseOutDirectly
        ? out
        : _createZeros(resShape, targetDType) as NDArray<T>;

    // Stride resolution logic for 100% copy-free BLAS matrix multiplication
    var transA = 111; // CblasNoTrans
    var lda = math.max(1, kA);
    if (!aPromoted) {
      if (aView.strides[rankA - 1] == 1 &&
          aView.strides[rankA - 2] >= math.max(1, kA)) {
        transA = 111;
        lda = math.max(aView.strides[rankA - 2], math.max(1, kA));
      } else {
        transA = 112; // CblasTrans
        lda = math.max(aView.strides[rankA - 1], math.max(1, m));
      }
    }

    var transB = 111; // CblasNoTrans
    var ldb = math.max(1, n);
    if (!bPromoted) {
      if (bView.strides[rankB - 1] == 1 &&
          bView.strides[rankB - 2] >= math.max(1, n)) {
        transB = 111;
        ldb = math.max(bView.strides[rankB - 2], math.max(1, n));
      } else {
        transB = 112; // CblasTrans
        ldb = math.max(bView.strides[rankB - 1], math.max(1, kB));
      }
    }

    final lenA = stackA.length;
    final lenB = stackB.length;
    final lenResult = broadcastStack.length;

    final walkStridesA = List<int>.filled(lenResult, 0);
    final walkStridesB = List<int>.filled(lenResult, 0);

    for (var i = 0; i < lenResult; i++) {
      final resAxis = lenResult - 1 - i;
      final axisA = lenA - 1 - i;
      final axisB = lenB - 1 - i;

      if (axisA >= 0) {
        walkStridesA[resAxis] = (stackA[axisA] == broadcastStack[resAxis])
            ? aView.strides[axisA]
            : 0;
      } else {
        walkStridesA[resAxis] = 0;
      }

      if (axisB >= 0) {
        walkStridesB[resAxis] = (stackB[axisB] == broadcastStack[resAxis])
            ? bView.strides[axisB]
            : 0;
      } else {
        walkStridesB[resAxis] = 0;
      }
    }

    final walkStridesRes = List<int>.filled(lenResult, 0);
    var resStride = m * n;
    for (var i = lenResult - 1; i >= 0; i--) {
      walkStridesRes[i] = resStride;
      resStride *= broadcastStack[i];
    }

    final marker = ScratchArena.marker;
    try {
      ffi.Pointer<ffi.Double> alphaZ = ffi.nullptr.cast();
      ffi.Pointer<ffi.Double> betaZ = ffi.nullptr.cast();
      ffi.Pointer<ffi.Float> alphaC = ffi.nullptr.cast();
      ffi.Pointer<ffi.Float> betaC = ffi.nullptr.cast();

      switch (targetDType) {
        case DType.complex128:
          alphaZ = ScratchArena.allocate<ffi.Double>(
            2 * ffi.sizeOf<ffi.Double>(),
          );
          alphaZ[0] = 1.0;
          alphaZ[1] = 0.0;
          betaZ = ScratchArena.allocate<ffi.Double>(
            2 * ffi.sizeOf<ffi.Double>(),
          );
          betaZ[0] = 0.0;
          betaZ[1] = 0.0;
        case DType.complex64:
          alphaC = ScratchArena.allocate<ffi.Float>(
            2 * ffi.sizeOf<ffi.Float>(),
          );
          alphaC[0] = 1.0;
          alphaC[1] = 0.0;
          betaC = ScratchArena.allocate<ffi.Float>(2 * ffi.sizeOf<ffi.Float>());
          betaC[0] = 0.0;
          betaC[1] = 0.0;
        default:
          break;
      }

      void walk(int dim, int offsetA, int offsetB, int offsetRes) {
        if (dim == lenResult) {
          switch (targetDType) {
            case DType.float64:
              if (aPromoted && bPromoted) {
                final incA = aView.strides[rankA - 1];
                final incB = bView.strides[rankB - 2];
                final dot = cblas_ddot(
                  kA,
                  aView.pointer.cast<ffi.Double>() + offsetA,
                  incA,
                  bView.pointer.cast<ffi.Double>() + offsetB,
                  incB,
                );
                result!.pointer.cast<ffi.Double>()[offsetRes] = dot;
              } else if (bPromoted) {
                final incB = bView.strides[rankB - 2];
                if (transA == 111) {
                  cblas_dgemv(
                    101, // CblasRowMajor
                    111, // CblasNoTrans
                    m,
                    kA,
                    1.0,
                    aView.pointer.cast<ffi.Double>() + offsetA,
                    lda,
                    bView.pointer.cast<ffi.Double>() + offsetB,
                    incB,
                    0.0,
                    result!.pointer.cast<ffi.Double>() + offsetRes,
                    1,
                  );
                } else {
                  cblas_dgemv(
                    101, // CblasRowMajor
                    112, // CblasTrans
                    kA,
                    m,
                    1.0,
                    aView.pointer.cast<ffi.Double>() + offsetA,
                    lda,
                    bView.pointer.cast<ffi.Double>() + offsetB,
                    incB,
                    0.0,
                    result!.pointer.cast<ffi.Double>() + offsetRes,
                    1,
                  );
                }
              } else if (aPromoted) {
                final incA = aView.strides[rankA - 1];
                if (transB == 111) {
                  cblas_dgemv(
                    101, // CblasRowMajor
                    112, // CblasTrans (y = a B = B^T a)
                    kB,
                    n,
                    1.0,
                    bView.pointer.cast<ffi.Double>() + offsetB,
                    ldb,
                    aView.pointer.cast<ffi.Double>() + offsetA,
                    incA,
                    0.0,
                    result!.pointer.cast<ffi.Double>() + offsetRes,
                    1,
                  );
                } else {
                  cblas_dgemv(
                    101, // CblasRowMajor
                    111, // CblasNoTrans (y = a B_mem^T = B_mem a)
                    n,
                    kB,
                    1.0,
                    bView.pointer.cast<ffi.Double>() + offsetB,
                    ldb,
                    aView.pointer.cast<ffi.Double>() + offsetA,
                    incA,
                    0.0,
                    result!.pointer.cast<ffi.Double>() + offsetRes,
                    1,
                  );
                }
              } else {
                cblas_dgemm(
                  101, // CblasRowMajor
                  transA,
                  transB,
                  m,
                  n,
                  kA,
                  1.0,
                  aView.pointer.cast<ffi.Double>() + offsetA,
                  lda,
                  bView.pointer.cast<ffi.Double>() + offsetB,
                  ldb,
                  0.0,
                  result!.pointer.cast<ffi.Double>() + offsetRes,
                  n, // ldc (result is always contiguous row-major)
                );
              }
            case DType.float32:
              if (aPromoted && bPromoted) {
                final incA = aView.strides[rankA - 1];
                final incB = bView.strides[rankB - 2];
                final dot = cblas_sdot(
                  kA,
                  aView.pointer.cast<ffi.Float>() + offsetA,
                  incA,
                  bView.pointer.cast<ffi.Float>() + offsetB,
                  incB,
                );
                result!.pointer.cast<ffi.Float>()[offsetRes] = dot;
              } else if (bPromoted) {
                final incB = bView.strides[rankB - 2];
                if (transA == 111) {
                  cblas_sgemv(
                    101,
                    111,
                    m,
                    kA,
                    1.0,
                    aView.pointer.cast<ffi.Float>() + offsetA,
                    lda,
                    bView.pointer.cast<ffi.Float>() + offsetB,
                    incB,
                    0.0,
                    result!.pointer.cast<ffi.Float>() + offsetRes,
                    1,
                  );
                } else {
                  cblas_sgemv(
                    101,
                    112,
                    kA,
                    m,
                    1.0,
                    aView.pointer.cast<ffi.Float>() + offsetA,
                    lda,
                    bView.pointer.cast<ffi.Float>() + offsetB,
                    incB,
                    0.0,
                    result!.pointer.cast<ffi.Float>() + offsetRes,
                    1,
                  );
                }
              } else if (aPromoted) {
                final incA = aView.strides[rankA - 1];
                if (transB == 111) {
                  cblas_sgemv(
                    101,
                    112,
                    kB,
                    n,
                    1.0,
                    bView.pointer.cast<ffi.Float>() + offsetB,
                    ldb,
                    aView.pointer.cast<ffi.Float>() + offsetA,
                    incA,
                    0.0,
                    result!.pointer.cast<ffi.Float>() + offsetRes,
                    1,
                  );
                } else {
                  cblas_sgemv(
                    101,
                    111,
                    n,
                    kB,
                    1.0,
                    bView.pointer.cast<ffi.Float>() + offsetB,
                    ldb,
                    aView.pointer.cast<ffi.Float>() + offsetA,
                    incA,
                    0.0,
                    result!.pointer.cast<ffi.Float>() + offsetRes,
                    1,
                  );
                }
              } else {
                cblas_sgemm(
                  101, // CblasRowMajor
                  transA,
                  transB,
                  m,
                  n,
                  kA,
                  1.0,
                  aView.pointer.cast<ffi.Float>() + offsetA,
                  lda,
                  bView.pointer.cast<ffi.Float>() + offsetB,
                  ldb,
                  0.0,
                  result!.pointer.cast<ffi.Float>() + offsetRes,
                  n, // ldc (result is always contiguous row-major)
                );
              }
            case DType.complex128:
              if (aPromoted && bPromoted) {
                final incA = aView.strides[rankA - 1];
                final incB = bView.strides[rankB - 2];
                final aPtr = aView.pointer.cast<ffi.Double>() + (offsetA * 2);
                final bPtr = bView.pointer.cast<ffi.Double>() + (offsetB * 2);
                final resPtr =
                    result!.pointer.cast<ffi.Double>() + (offsetRes * 2);
                var realSum = 0.0;
                var imagSum = 0.0;
                for (var i = 0; i < kA; i++) {
                  final ar = aPtr[i * incA * 2];
                  final ai = aPtr[i * incA * 2 + 1];
                  final br = bPtr[i * incB * 2];
                  final bi = bPtr[i * incB * 2 + 1];
                  realSum += ar * br - ai * bi;
                  imagSum += ar * bi + ai * br;
                }
                resPtr[0] = realSum;
                resPtr[1] = imagSum;
              } else if (bPromoted) {
                final incB = bView.strides[rankB - 2];
                if (transA == 111) {
                  cblas_zgemv(
                    101,
                    111,
                    m,
                    kA,
                    alphaZ,
                    aView.pointer.cast<ffi.Double>() + (offsetA * 2),
                    lda,
                    bView.pointer.cast<ffi.Double>() + (offsetB * 2),
                    incB,
                    betaZ,
                    result!.pointer.cast<ffi.Double>() + (offsetRes * 2),
                    1,
                  );
                } else {
                  cblas_zgemv(
                    101,
                    112,
                    kA,
                    m,
                    alphaZ,
                    aView.pointer.cast<ffi.Double>() + (offsetA * 2),
                    lda,
                    bView.pointer.cast<ffi.Double>() + (offsetB * 2),
                    incB,
                    betaZ,
                    result!.pointer.cast<ffi.Double>() + (offsetRes * 2),
                    1,
                  );
                }
              } else if (aPromoted) {
                final incA = aView.strides[rankA - 1];
                if (transB == 111) {
                  cblas_zgemv(
                    101,
                    112,
                    kB,
                    n,
                    alphaZ,
                    bView.pointer.cast<ffi.Double>() + (offsetB * 2),
                    ldb,
                    aView.pointer.cast<ffi.Double>() + (offsetA * 2),
                    incA,
                    betaZ,
                    result!.pointer.cast<ffi.Double>() + (offsetRes * 2),
                    1,
                  );
                } else {
                  cblas_zgemv(
                    101,
                    111,
                    n,
                    kB,
                    alphaZ,
                    bView.pointer.cast<ffi.Double>() + (offsetB * 2),
                    ldb,
                    aView.pointer.cast<ffi.Double>() + (offsetA * 2),
                    incA,
                    betaZ,
                    result!.pointer.cast<ffi.Double>() + (offsetRes * 2),
                    1,
                  );
                }
              } else {
                cblas_zgemm(
                  101,
                  transA,
                  transB,
                  m,
                  n,
                  kA,
                  alphaZ,
                  aView.pointer.cast<ffi.Double>() + (offsetA * 2),
                  lda,
                  bView.pointer.cast<ffi.Double>() + (offsetB * 2),
                  ldb,
                  betaZ,
                  result!.pointer.cast<ffi.Double>() + (offsetRes * 2),
                  n,
                );
              }
            case DType.complex64:
              if (aPromoted && bPromoted) {
                final incA = aView.strides[rankA - 1];
                final incB = bView.strides[rankB - 2];
                final aPtr = aView.pointer.cast<ffi.Float>() + (offsetA * 2);
                final bPtr = bView.pointer.cast<ffi.Float>() + (offsetB * 2);
                final resPtr =
                    result!.pointer.cast<ffi.Float>() + (offsetRes * 2);
                var realSum = 0.0;
                var imagSum = 0.0;
                for (var i = 0; i < kA; i++) {
                  final ar = aPtr[i * incA * 2];
                  final ai = aPtr[i * incA * 2 + 1];
                  final br = bPtr[i * incB * 2];
                  final bi = bPtr[i * incB * 2 + 1];
                  realSum += ar * br - ai * bi;
                  imagSum += ar * bi + ai * br;
                }
                resPtr[0] = realSum;
                resPtr[1] = imagSum;
              } else if (bPromoted) {
                final incB = bView.strides[rankB - 2];
                if (transA == 111) {
                  cblas_cgemv(
                    101,
                    111,
                    m,
                    kA,
                    alphaC,
                    aView.pointer.cast<ffi.Float>() + (offsetA * 2),
                    lda,
                    bView.pointer.cast<ffi.Float>() + (offsetB * 2),
                    incB,
                    betaC,
                    result!.pointer.cast<ffi.Float>() + (offsetRes * 2),
                    1,
                  );
                } else {
                  cblas_cgemv(
                    101,
                    112,
                    kA,
                    m,
                    alphaC,
                    aView.pointer.cast<ffi.Float>() + (offsetA * 2),
                    lda,
                    bView.pointer.cast<ffi.Float>() + (offsetB * 2),
                    incB,
                    betaC,
                    result!.pointer.cast<ffi.Float>() + (offsetRes * 2),
                    1,
                  );
                }
              } else if (aPromoted) {
                final incA = aView.strides[rankA - 1];
                if (transB == 111) {
                  cblas_cgemv(
                    101,
                    112,
                    kB,
                    n,
                    alphaC,
                    bView.pointer.cast<ffi.Float>() + (offsetB * 2),
                    ldb,
                    aView.pointer.cast<ffi.Float>() + (offsetA * 2),
                    incA,
                    betaC,
                    result!.pointer.cast<ffi.Float>() + (offsetRes * 2),
                    1,
                  );
                } else {
                  cblas_cgemv(
                    101,
                    111,
                    n,
                    kB,
                    alphaC,
                    bView.pointer.cast<ffi.Float>() + (offsetB * 2),
                    ldb,
                    aView.pointer.cast<ffi.Float>() + (offsetA * 2),
                    incA,
                    betaC,
                    result!.pointer.cast<ffi.Float>() + (offsetRes * 2),
                    1,
                  );
                }
              } else {
                cblas_cgemm(
                  101,
                  transA,
                  transB,
                  m,
                  n,
                  kA,
                  alphaC,
                  aView.pointer.cast<ffi.Float>() + (offsetA * 2),
                  lda,
                  bView.pointer.cast<ffi.Float>() + (offsetB * 2),
                  ldb,
                  betaC,
                  result!.pointer.cast<ffi.Float>() + (offsetRes * 2),
                  n,
                );
              }
            case DType.uint64:
            case DType.int64:
            case DType.uint32:
            case DType.int32:
            case DType.uint16:
            case DType.int16:
            case DType.uint8:
            case DType.int8:
            case DType.boolean:
              final strideARow = aView.strides[rankA - 2];
              final strideACol = aView.strides[rankA - 1];

              final strideBRow = bView.strides[rankB - 2];
              final strideBCol = bView.strides[rankB - 1];

              final strideResRow = result!.strides[resShape.length - 2];
              final strideResCol = result.strides[resShape.length - 1];

              switch (targetDType) {
                case DType.uint64:
                case DType.int64:
                  matmul_int64(
                    result.pointer.cast<ffi.Int64>() + offsetRes,
                    strideResRow,
                    strideResCol,
                    aView.pointer.cast<ffi.Int64>() + offsetA,
                    strideARow,
                    strideACol,
                    bView.pointer.cast<ffi.Int64>() + offsetB,
                    strideBRow,
                    strideBCol,
                    m,
                    n,
                    kA,
                  );
                case DType.uint32:
                case DType.int32:
                  matmul_int32(
                    result.pointer.cast<ffi.Int32>() + offsetRes,
                    strideResRow,
                    strideResCol,
                    aView.pointer.cast<ffi.Int32>() + offsetA,
                    strideARow,
                    strideACol,
                    bView.pointer.cast<ffi.Int32>() + offsetB,
                    strideBRow,
                    strideBCol,
                    m,
                    n,
                    kA,
                  );
                case DType.uint16:
                case DType.int16:
                  matmul_int16(
                    result.pointer.cast<ffi.Int16>() + offsetRes,
                    strideResRow,
                    strideResCol,
                    aView.pointer.cast<ffi.Int16>() + offsetA,
                    strideARow,
                    strideACol,
                    bView.pointer.cast<ffi.Int16>() + offsetB,
                    strideBRow,
                    strideBCol,
                    m,
                    n,
                    kA,
                  );
                case DType.uint8:
                case DType.int8:
                  matmul_uint8(
                    result.pointer.cast<ffi.Uint8>() + offsetRes,
                    strideResRow,
                    strideResCol,
                    aView.pointer.cast<ffi.Uint8>() + offsetA,
                    strideARow,
                    strideACol,
                    bView.pointer.cast<ffi.Uint8>() + offsetB,
                    strideBRow,
                    strideBCol,
                    m,
                    n,
                    kA,
                  );
                case DType.boolean:
                  final aPtr = aView.pointer.cast<ffi.Uint8>() + offsetA;
                  final bPtr = bView.pointer.cast<ffi.Uint8>() + offsetB;
                  final resPtr = result.pointer.cast<ffi.Uint8>() + offsetRes;
                  for (var r = 0; r < m; r++) {
                    for (var c = 0; c < n; c++) {
                      var acc = 0;
                      for (var i = 0; i < kA; i++) {
                        if (aPtr[r * strideARow + i * strideACol] != 0 &&
                            bPtr[i * strideBRow + c * strideBCol] != 0) {
                          acc = 1;
                          break;
                        }
                      }
                      resPtr[r * strideResRow + c * strideResCol] = acc;
                    }
                  }
                default:
                  throw UnsupportedError(
                    'Unsupported integer type: $targetDType',
                  );
              }
            default:
              throw UnsupportedError('Unsupported type: $targetDType');
          }
          return;
        }

        final size = broadcastStack[dim];
        final strideA = walkStridesA[dim];
        final strideB = walkStridesB[dim];
        final strideRes = walkStridesRes[dim];

        for (var i = 0; i < size; i++) {
          walk(
            dim + 1,
            offsetA + i * strideA,
            offsetB + i * strideB,
            offsetRes + i * strideRes,
          );
        }
      }

      walk(0, 0, 0, 0);
    } finally {
      ScratchArena.reset(marker);
    }

    if (out != null) {
      if (!canUseOutDirectly) {
        final reshaped = result.reshape(out.shape);
        reshaped.copy(out: out);
        reshaped.dispose();
        result.dispose();
      }
      success = true;
      return out;
    }

    // Post-calculation 1D dummy dimensions demotions
    if (aPromoted && bPromoted) {
      final reshaped = result.reshape([]);
      final finalRes = reshaped.copy();
      reshaped.dispose();
      result.dispose();
      success = true;
      return finalRes; // 0D scalar array for pure vector dot products
    } else if (aPromoted) {
      final newShape = List<int>.from(result.shape)
        ..removeAt(result.shape.length - 2);
      final reshaped = result.reshape(newShape);
      final finalRes = reshaped.copy();
      reshaped.dispose();
      result.dispose();
      success = true;
      return finalRes;
    } else if (bPromoted) {
      final newShape = List<int>.from(result.shape)
        ..removeAt(result.shape.length - 1);
      final reshaped = result.reshape(newShape);
      final finalRes = reshaped.copy();
      reshaped.dispose();
      result.dispose();
      success = true;
      return finalRes;
    }

    success = true;
    return result;
  } finally {
    aPromotedView?.dispose();
    bPromotedView?.dispose();
    if (aCast != null && !identical(aCast, a)) aCast.dispose();
    if (bCast != null && !identical(bCast, b)) bCast.dispose();
    aCopy?.dispose();
    bCopy?.dispose();
    if (!success) {
      if (result != null && !identical(result, out)) {
        result.dispose();
      }
    }
  }
}

/// Computes the product of two or more arrays in a single function call,
/// while automatically selecting the fastest evaluation order.
///
/// Solves the matrix chain multiplication problem using standard dynamic programming in $O(N^3)$ time.
///
/// **Preconditions:**
/// - It is an error if any input array in [arrays] or [out] is disposed.
/// - It is an error if [arrays] has fewer than 2 elements.
/// - It is an error if any intermediate array is not 2-dimensional.
/// - It is an error if first or last array has rank > 2 or rank < 1.
/// - It is an error if inner dimensions of adjacent matrices are incompatible.
/// - It is an error if [out] shape or dtype is incompatible.
///
/// **Performance considerations:**
/// - Automatically optimizes the order of operations to minimize total scalar multiplications.
/// - All intermediate transient arrays are automatically disposed of to guarantee zero memory leaks.
///
/// **Example:**
/// {@example /example/linalg_multi_dot_example.dart lang=dart}
///
/// Reference: [NumPy linalg.multi_dot](https://numpy.org/doc/stable/reference/generated/numpy.linalg.multi_dot.html)
NDArray<T> multi_dot<T extends DTypeTag>(
  List<NDArray<DTypeTag>> arrays, {
  NDArray<T>? out,
}) {
  for (final a in arrays) {
    if (a.isDisposed) {
      throw StateError(
        'Cannot execute multi_dot() with a disposed array in the list.',
      );
    }
  }
  if (out != null) {
    if (out.isDisposed) {
      throw StateError(
        'Cannot write multi_dot result to a disposed output array.',
      );
    }
    validateOutBuffer(out);
  }
  if (arrays.length < 2) {
    throw ArgumentError.value(
      arrays.length,
      'arrays',
      'Must contain at least 2 arrays (multi_dot requires at least 2 arrays, got ${arrays.length}).',
    );
  }

  final n = arrays.length;

  // Check dimensions & validate rank preconditions
  for (var i = 0; i < n; i++) {
    final rank = arrays[i].shape.length;
    if (i == 0 || i == n - 1) {
      if (rank != 1 && rank != 2) {
        throw ArgumentError.value(
          arrays[i].shape,
          'arrays[$i]',
          'Must be 1D or 2D (first and last arrays in multi_dot must be 1D or 2D, array $i was shape ${arrays[i].shape}).',
        );
      }
    } else {
      if (rank != 2) {
        throw ArgumentError.value(
          arrays[i].shape,
          'arrays[$i]',
          'Must be 2D (all intermediate arrays in multi_dot must be 2D, array $i was shape ${arrays[i].shape}).',
        );
      }
    }
  }

  // Build dimensions list p
  final p = List<int>.filled(n + 1, 0);
  if (arrays[0].shape.length == 1) {
    p[0] = 1;
    p[1] = arrays[0].shape[0];
  } else {
    p[0] = arrays[0].shape[0];
    p[1] = arrays[0].shape[1];
  }

  for (var i = 1; i < n - 1; i++) {
    final shape = arrays[i].shape;
    if (shape[0] != p[i]) {
      throw ArgumentError.value(
        shape[0],
        'arrays[$i].shape[0]',
        'Must match previous dimension (${p[i]}) (incompatible matrix dimensions in multi_dot: array $i first dimension (${shape[0]}) must match previous dimension (${p[i]})).',
      );
    }
    p[i + 1] = shape[1];
  }

  // Last array
  final lastIdx = n - 1;
  final lastShape = arrays[lastIdx].shape;
  if (lastShape[0] != p[lastIdx]) {
    throw ArgumentError.value(
      lastShape[0],
      'arrays[lastIdx].shape[0]',
      'Must match previous dimension (${p[lastIdx]}) (incompatible matrix dimensions in multi_dot: last array first dimension (${lastShape[0]}) must match previous dimension (${p[lastIdx]})).',
    );
  }
  if (lastShape.length == 1) {
    p[n] = 1;
  } else {
    p[n] = lastShape[1];
  }

  for (var i = 0; i <= n; i++) {
    checkBlasIntDim(p[i], 'dim', 'multi_dot');
  }
  for (var i = 0; i < n; i++) {
    for (final s in arrays[i].strides) {
      checkBlasIntStride(s, 'lda', 'multi_dot');
    }
  }

  // Resolve target DType and upcasted type
  DType<DTypeTag> targetDType = arrays[0].dtype;
  for (var i = 1; i < n; i++) {
    targetDType = resolveDType(targetDType, arrays[i].dtype);
  }
  if (!targetDType.isFloating && !targetDType.isComplex) {
    if (T == Float64 || (out != null && out.dtype == DType.float64)) {
      targetDType = DType.float64;
    }
  }

  // If out is provided, validate it
  final expectedFinalShape = <int>[];
  final first1D = arrays[0].shape.length == 1;
  final last1D = arrays[lastIdx].shape.length == 1;
  if (first1D && last1D) {
    // Result is 0D scalar shape []
  } else if (first1D) {
    expectedFinalShape.add(p[n]);
  } else if (last1D) {
    expectedFinalShape.add(p[0]);
  } else {
    expectedFinalShape.addAll([p[0], p[n]]);
  }

  if (out != null) {
    if (!listEquals(out.shape, expectedFinalShape) ||
        out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape $expectedFinalShape and dtype $targetDType (provided out recycler has incompatible shape or dtype, got shape ${out.shape} and dtype ${out.dtype}).',
      );
    }
  }

  return NDArray.scope(() {
    // Dynamic programming to find the optimal parenthesization
    final m = List.generate(n + 1, (_) => List<int>.filled(n + 1, 0));
    final s = List.generate(n + 1, (_) => List<int>.filled(n + 1, 0));

    for (var l = 2; l <= n; l++) {
      for (var i = 1; i <= n - l + 1; i++) {
        final j = i + l - 1;
        m[i][j] = -1;
        s[i][j] = i;
        for (var k = i; k < j; k++) {
          final cost = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
          if (m[i][j] < 0 || cost < m[i][j]) {
            m[i][j] = cost;
            s[i][j] = k;
          }
        }
      }
    }

    // Helper function to recursively evaluate matrix multiplication chain
    NDArray<DTypeTag> eval(int i, int j) {
      if (i == j) {
        // Return a contiguous copy of arrays[i-1] casted to the correct targetDType
        final src = arrays[i - 1];
        if (src.dtype == targetDType) {
          return src.copy();
        } else {
          return castNDArray(src, targetDType);
        }
      }

      final k = s[i][j];
      final left = eval(i, k);
      final right = eval(k + 1, j);

      // Perform matrix multiplication
      final res = matmul<DTypeTag>(left, right);
      left.dispose();
      right.dispose();
      return res;
    }

    // Top-level split point evaluation
    final k = s[1][n];
    final left = eval(1, k);
    final right = eval(k + 1, n);

    final finalResult = matmul<DTypeTag>(left, right, out: out);
    left.dispose();
    right.dispose();

    if (out != null) return out;
    return (finalResult as NDArray<T>).detachToParentScope();
  });
}

/// Computes the multiplicative inverse of a square 2D matrix.
///
/// Uses OpenBLAS LAPACK LU decomposition routines
/// (`LAPACKE_dgetrf`/`LAPACKE_dgetri` for Float64, and `LAPACKE_sgetrf`/`LAPACKE_sgetri` for Float32).
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] is not square in its last two dimensions (`shape.length == 2` and `shape[0] == shape[1]`).
/// - It is an error if [a] has an unsupported dtype (only float and complex dtypes are supported).
/// - It is an error if [out] is provided and has incompatible shape or dtype.
/// - The matrix must be non-singular (invertible).
///
/// **Throws:**
/// - Throws a [SingularMatrixException] if the matrix is singular (non-invertible) during LU pivoting.
/// - Throws a [LinAlgException] if the underlying LAPACK routine fails.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N^3)$ where $N$ is the matrix dimension length.
/// - For non-contiguous views, automatically flattens the matrix first, recycling allocation views
///   where safe to minimize heap churn.
///
/// **Example:**
/// {@example /example/linalg_example.dart lang=dart}
///
/// Reference: [Matrix Inversion](https://en.wikipedia.org/wiki/Invertible_matrix)
NDArray<T> inv<T extends DTypeTag>(NDArray<T> a, {NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute inverse of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write inverse to a disposed output array.');
  }
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 2] != a.shape[rank - 1]) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must be square in the last 2 dimensions and rank >= 2 (was ${a.shape})',
    );
  }

  switch (a.dtype) {
    case DType.float32:
    case DType.float64:
    case DType.float16:
    case DType.bfloat16:
    case DType.complex64:
    case DType.complex128:
      break;
    default:
      throw ArgumentError.value(
        a.dtype,
        'a.dtype',
        'Must be float or complex (matrix inversion only supports float or complex dtypes, got ${a.dtype}).',
      );
  }
  checkBlasIntDim(a.shape[rank - 1], 'n', 'inv');
  checkBlasIntStride(a.strides[rank - 2], 'lda', 'inv');
  checkBlasIntStride(a.strides[rank - 1], 'lda', 'inv');
  if (a.dtype == DType.float16 || a.dtype == DType.bfloat16) {
    if (out != null) {
      validateOutBuffer(out);
      if (!listEquals(out.shape, a.shape) ||
          (out.dtype != DType.float64 && out.dtype != a.dtype)) {
        throw ArgumentError.value(
          out,
          'out',
          'Must have compatible shape ${a.shape} and float64 or ${a.dtype} dtype for matrix inversion (provided out buffer has incompatible shape or dtype).',
        );
      }
    }
    return NDArray.scope(() {
      final aF64 = castNDArray<Float64>(a, DType.float64);
      final resF64 = inv<Float64>(aF64);
      if (out != null) {
        if (out.dtype == DType.float64) {
          resF64.copy(out: out as NDArray<Float64>);
        } else {
          castNDArray<T>(resF64, out.dtype).copy(out: out);
        }
        return out;
      }
      return castNDArray<T>(resF64, a.dtype).detachToParentScope();
    });
  }
  final n = a.shape[rank - 1];
  final stackShape = a.shape.sublist(0, rank - 2);
  final DType<T> targetDType = a.dtype;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape ${a.shape} and dtype $targetDType for matrix inversion (provided out buffer has incompatible shape or dtype).',
      );
    }
    if (!out.isContiguous || sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = inv<T>(a);
        temp.copy(out: out);
        return out;
      });
    }
  }

  return NDArray.scope(() {
    final NDArray<T> result;
    if (out != null) {
      result = out;
      a.copy(out: result);
    } else {
      result = a.copy();
    }

    if (n == 0) {
      if (out == null) {
        result.detachToParentScope();
      }
      return result;
    }

    final marker = ScratchArena.marker;
    try {
      final ipiv = ScratchArena.allocate<ffi.Int>(n * ffi.sizeOf<ffi.Int>());
      walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
        coords,
      ) {
        var offsetRes = 0;
        for (var i = 0; i < coords.length; i++) {
          offsetRes += coords[i] * result.strides[i];
        }
        final sliceRes = NDArray<T>.view(
          result,
          shape: [n, n],
          strides: result.strides.sublist(rank - 2),
          offsetElements: offsetRes,
        );

        final nf = _analyzeNonFinitePtr(sliceRes.pointer, n * n, targetDType);
        if (nf.hasNaN) {
          _fillPtrWithNaN(sliceRes.pointer, n * n, targetDType);
          sliceRes.dispose();
          return;
        }
        switch (targetDType) {
          case DType.float32:
            final info = LAPACKE_sgetrf(
              101,
              n,
              n,
              sliceRes.pointer.cast<ffi.Float>(),
              n,
              ipiv,
            );
            _checkLapackInfo(
              info,
              'LAPACKE_sgetrf',
              positiveKind: _LapackFailureKind.singularMatrix,
              positiveMessage: 'Matrix is singular and cannot be inverted',
            );
            final nfAfterLu = _analyzeNonFinitePtr(
              sliceRes.pointer,
              n * n,
              targetDType,
            );
            if (nfAfterLu.hasNaN) {
              _fillPtrWithNaN(sliceRes.pointer, n * n, targetDType);
            } else {
              final infoTri = LAPACKE_sgetri(
                101,
                n,
                sliceRes.pointer.cast<ffi.Float>(),
                n,
                ipiv,
              );
              _checkLapackInfo(
                infoTri,
                'LAPACKE_sgetri',
                positiveKind: _LapackFailureKind.singularMatrix,
                positiveMessage: 'Matrix is singular and cannot be inverted',
              );
            }
          case DType.float64:
            final info = LAPACKE_dgetrf(
              101,
              n,
              n,
              sliceRes.pointer.cast<ffi.Double>(),
              n,
              ipiv,
            );
            _checkLapackInfo(
              info,
              'LAPACKE_dgetrf',
              positiveKind: _LapackFailureKind.singularMatrix,
              positiveMessage: 'Matrix is singular and cannot be inverted',
            );
            final nfAfterLu = _analyzeNonFinitePtr(
              sliceRes.pointer,
              n * n,
              targetDType,
            );
            if (nfAfterLu.hasNaN) {
              _fillPtrWithNaN(sliceRes.pointer, n * n, targetDType);
            } else {
              final infoTri = LAPACKE_dgetri(
                101,
                n,
                sliceRes.pointer.cast<ffi.Double>(),
                n,
                ipiv,
              );
              _checkLapackInfo(
                infoTri,
                'LAPACKE_dgetri',
                positiveKind: _LapackFailureKind.singularMatrix,
                positiveMessage: 'Matrix is singular and cannot be inverted',
              );
            }
          case DType.complex64:
            final info = LAPACKE_cgetrf(
              101,
              n,
              n,
              sliceRes.pointer.cast<ffi.Float>(),
              n,
              ipiv,
            );
            _checkLapackInfo(
              info,
              'LAPACKE_cgetrf',
              positiveKind: _LapackFailureKind.singularMatrix,
              positiveMessage: 'Matrix is singular and cannot be inverted',
            );
            final nfAfterLu = _analyzeNonFinitePtr(
              sliceRes.pointer,
              n * n,
              targetDType,
            );
            if (nfAfterLu.hasNaN) {
              _fillPtrWithNaN(sliceRes.pointer, n * n, targetDType);
            } else {
              final infoTri = LAPACKE_cgetri(
                101,
                n,
                sliceRes.pointer.cast<ffi.Float>(),
                n,
                ipiv,
              );
              _checkLapackInfo(
                infoTri,
                'LAPACKE_cgetri',
                positiveKind: _LapackFailureKind.singularMatrix,
                positiveMessage: 'Matrix is singular and cannot be inverted',
              );
            }
          case DType.complex128:
            final info = LAPACKE_zgetrf(
              101,
              n,
              n,
              sliceRes.pointer.cast<ffi.Double>(),
              n,
              ipiv,
            );
            _checkLapackInfo(
              info,
              'LAPACKE_zgetrf',
              positiveKind: _LapackFailureKind.singularMatrix,
              positiveMessage: 'Matrix is singular and cannot be inverted',
            );
            final nfAfterLu = _analyzeNonFinitePtr(
              sliceRes.pointer,
              n * n,
              targetDType,
            );
            if (nfAfterLu.hasNaN) {
              _fillPtrWithNaN(sliceRes.pointer, n * n, targetDType);
            } else {
              final infoTri = LAPACKE_zgetri(
                101,
                n,
                sliceRes.pointer.cast<ffi.Double>(),
                n,
                ipiv,
              );
              _checkLapackInfo(
                infoTri,
                'LAPACKE_zgetri',
                positiveKind: _LapackFailureKind.singularMatrix,
                positiveMessage: 'Matrix is singular and cannot be inverted',
              );
            }
          default:
            throw UnsupportedError(
              'Unsupported type for matrix inversion: $targetDType',
            );
        }
        sliceRes.dispose();
      });

      if (out == null) {
        result.detachToParentScope();
      }
      return result;
    } finally {
      ScratchArena.reset(marker);
    }
  });
}

/// Computes the determinant of a square matrix or a stack of square matrices using OpenBLAS/LAPACK.
///
/// Transforms the matrix and calculates its determinant natively via LAPACK LU decomposition.
/// Supports both real (float32, float64) and complex (complex64, complex128) data types.
/// Returns the determinant stack as an array of corresponding types (float64 for real inputs,
/// and complex64/complex128 for complex inputs).
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] is not square in its last two dimensions or is less than 2-dimensional.
/// - It is an error if [a.dtype] is not float32, float64, complex64, or complex128.
/// - It is an error if [out] is provided and has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N^3)$ using LAPACK linear algebra solvers.
/// - Fully vectorized and batched in native C for float64, complex64, and complex128, minimizing FFI transitions.
///
/// **Example:**
/// {@example /example/linalg_example.dart lang=dart}
///
/// Refer to the [determinant](https://en.wikipedia.org/wiki/Determinant)
/// and [LAPACK LU solver](https://en.wikipedia.org/wiki/LU_decomposition) for additional details.
///
/// Returns a 0-dimensional [NDArray] if [a] is a 2D matrix, or a new [NDArray] with stack dimensions if [a] is a stack of matrices.
NDArray<T> det<T extends DTypeTag>(NDArray<T> a, {NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute determinant of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write determinant to a disposed output array.');
  }
  if (a.dtype != DType.float64 &&
      a.dtype != DType.float32 &&
      a.dtype != DType.float16 &&
      a.dtype != DType.bfloat16 &&
      a.dtype != DType.complex128 &&
      a.dtype != DType.complex64) {
    throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      'Must be float or complex (det only supports float and complex dtypes)',
    );
  }
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 1] != a.shape[rank - 2]) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must be square and at least 2D (was ${a.shape})',
    );
  }
  checkBlasIntDim(a.shape[rank - 1], 'n', 'det');
  checkBlasIntStride(a.strides[rank - 2], 'lda', 'det');
  checkBlasIntStride(a.strides[rank - 1], 'lda', 'det');
  final stackShape = a.shape.sublist(0, rank - 2);
  if (a.dtype == DType.float16 || a.dtype == DType.bfloat16) {
    if (out != null) {
      validateOutBuffer(out);
      if (!listEquals(out.shape, stackShape) ||
          (out.dtype != DType.float64 && out.dtype != a.dtype)) {
        throw ArgumentError.value(
          out,
          'out',
          'Must have compatible shape $stackShape and float64 or ${a.dtype} dtype (provided out buffer has incompatible shape or dtype).',
        );
      }
    }
    return NDArray.scope(() {
      final aF64 = castNDArray<Float64>(a, DType.float64);
      final resF64 = det<Float64>(aF64);
      if (out != null) {
        if (out.dtype == DType.float64) {
          resF64.copy(out: out as NDArray<Float64>);
        } else {
          castNDArray<T>(resF64, out.dtype).copy(out: out);
        }
        return out;
      }
      return castNDArray<T>(resF64, a.dtype).detachToParentScope();
    });
  }
  final expectedDType = a.dtype;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, stackShape) || out.dtype != expectedDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape $stackShape and dtype $expectedDType (provided out buffer has incompatible shape or dtype).',
      );
    }
    if (!out.isContiguous || sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = det<T>(a);
        temp.copy(out: out);
        return out;
      });
    }
  }

  return NDArray.scope(() {
    if (a.shape[rank - 1] == 0) {
      final result = out ?? NDArray.zeros(stackShape, a.dtype);
      result.fill(
        castValue(a.dtype.isComplex ? Complex(1.0, 0.0) : 1.0, a.dtype),
      );
      if (out == null) {
        result.detachToParentScope();
      }
      return result;
    }
    switch (a.dtype) {
      case DType.float64:
        final result =
            out ?? (NDArray.zeros(stackShape, DType.float64) as NDArray<T>);
        final marker = ScratchArena.marker;
        try {
          final cStridesA = ScratchArena.copyInts(a.strides);
          final cStridesRes = ScratchArena.copyInts(result.strides);
          final cShape = ScratchArena.copyInts(a.shape);

          final n = a.shape[rank - 1];
          final cCopy = ScratchArena.allocate<ffi.Double>(
            n * n * ffi.sizeOf<ffi.Double>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );

          s_det_double(
            a.pointer.cast<ffi.Double>(),
            cStridesA,
            result.pointer.cast<ffi.Double>(),
            cStridesRes,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_dgetrf_ptr().cast(),
          );
          checkNativeOom();
        } finally {
          ScratchArena.reset(marker);
        }
        if (out == null) {
          result.detachToParentScope();
        }
        return result;
      case DType.complex128:
        final result =
            out ?? (NDArray.zeros(stackShape, DType.complex128) as NDArray<T>);
        final marker = ScratchArena.marker;
        try {
          final cStridesA = ScratchArena.copyInts(a.strides);
          final cStridesRes = ScratchArena.copyInts(result.strides);
          final cShape = ScratchArena.copyInts(a.shape);

          final n = a.shape[rank - 1];
          final cCopy = ScratchArena.allocate<cpx_t>(
            n * n * ffi.sizeOf<cpx_t>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );

          s_det_complex_double(
            a.pointer.cast<cpx_t>(),
            cStridesA,
            result.pointer.cast<cpx_t>(),
            cStridesRes,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_zgetrf_ptr().cast(),
          );
          checkNativeOom();
        } finally {
          ScratchArena.reset(marker);
        }
        if (out == null) {
          result.detachToParentScope();
        }
        return result;
      case DType.complex64:
        final result =
            out ?? (NDArray.zeros(stackShape, DType.complex64) as NDArray<T>);
        final marker = ScratchArena.marker;
        try {
          final cStridesA = ScratchArena.copyInts(a.strides);
          final cStridesRes = ScratchArena.copyInts(result.strides);
          final cShape = ScratchArena.copyInts(a.shape);

          final n = a.shape[rank - 1];
          final cCopy = ScratchArena.allocate<cpx_f_t>(
            n * n * ffi.sizeOf<cpx_f_t>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );

          s_det_complex_float(
            a.pointer.cast<cpx_f_t>(),
            cStridesA,
            result.pointer.cast<cpx_f_t>(),
            cStridesRes,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_cgetrf_ptr().cast(),
          );
          checkNativeOom();
        } finally {
          ScratchArena.reset(marker);
        }
        if (out == null) {
          result.detachToParentScope();
        }
        return result;
      case DType.float32:
        final result =
            out ?? (NDArray.zeros(stackShape, DType.float32) as NDArray<T>);
        final marker = ScratchArena.marker;
        try {
          final cStridesA = ScratchArena.copyInts(a.strides);
          final cStridesRes = ScratchArena.copyInts(result.strides);
          final cShape = ScratchArena.copyInts(a.shape);

          final n = a.shape[rank - 1];
          final cCopy = ScratchArena.allocate<ffi.Float>(
            n * n * ffi.sizeOf<ffi.Float>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );

          s_det_float(
            a.pointer.cast<ffi.Float>(),
            cStridesA,
            result.pointer.cast<ffi.Float>(),
            cStridesRes,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_sgetrf_ptr().cast(),
          );
          checkNativeOom();
        } finally {
          ScratchArena.reset(marker);
        }
        if (out == null) {
          result.detachToParentScope();
        }
        return result;
      default:
        throw ArgumentError.value(
          a.dtype,
          'a.dtype',
          'Must be a supported float or complex dtype (unsupported dtype for determinant)',
        );
    }
  });
}

/// Computes the sign and natural logarithm of the absolute value of the determinant of a square 2D matrix or stack of matrices.
///
/// **Preconditions:**
/// - It is an error if [a], [outSign], or [outLogdet] is disposed.
/// - It is an error if [a] rank < 2, or the last two dimensions are not square.
/// - It is an error if [a] dtype is not float32, float64, complex64, or complex128.
/// - It is an error if [outSign] or [outLogdet] is provided and has incompatible shape or dtype.
///
/// **Returns:**
/// - A record `(sign, logdet)` of two NDArrays, representing the sign (or phase) and log of the absolute determinant.
///
/// Reference: [NumPy linalg.slogdet](https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html)
({NDArray<T> sign, NDArray<R> logabsdet})
slogdet<T extends DTypeTag, R extends DTypeTag>(
  NDArray<
    DTypeSpec<
      R,
      Object?,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      T,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag
    >
  >
  a, {
  NDArray<T>? outSign,
  NDArray<R>? outLogdet,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute slogdet of a disposed array.');
  }
  if (outSign != null && outSign.isDisposed) {
    throw StateError('Cannot write slogdet sign to a disposed output array.');
  }
  if (outLogdet != null && outLogdet.isDisposed) {
    throw StateError('Cannot write slogdet logdet to a disposed output array.');
  }
  if ((a.dtype as DType<DTypeTag>) != DType.float64 &&
      (a.dtype as DType<DTypeTag>) != DType.float32 &&
      (a.dtype as DType<DTypeTag>) != DType.float16 &&
      (a.dtype as DType<DTypeTag>) != DType.bfloat16 &&
      (a.dtype as DType<DTypeTag>) != DType.complex128 &&
      (a.dtype as DType<DTypeTag>) != DType.complex64) {
    throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      'Must be float or complex (slogdet only supports float and complex dtypes)',
    );
  }
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 1] != a.shape[rank - 2]) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must be square and at least 2D (was ${a.shape})',
    );
  }
  checkBlasIntDim(a.shape[rank - 1], 'n', 'slogdet');
  checkBlasIntStride(a.strides[rank - 2], 'lda', 'slogdet');
  checkBlasIntStride(a.strides[rank - 1], 'lda', 'slogdet');
  final stackShape = a.shape.sublist(0, rank - 2);

  if (outSign != null &&
      outLogdet != null &&
      sharesMemory(outSign, outLogdet)) {
    throw ArgumentError.value(
      outLogdet,
      'outLogdet',
      'Must not share memory with outSign.',
    );
  }

  if ((a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16) {
    if (outSign != null) {
      validateOutBuffer(outSign, 'outSign');
      if (!listEquals(outSign.shape, stackShape) ||
          (outSign.dtype != DType.float64 && outSign.dtype != a.dtype)) {
        throw ArgumentError.value(
          outSign,
          'outSign',
          'Must have compatible shape $stackShape and float64 or ${a.dtype} dtype (provided outSign buffer has incompatible shape or dtype).',
        );
      }
    }
    if (outLogdet != null) {
      validateOutBuffer(outLogdet, 'outLogdet');
      if (!listEquals(outLogdet.shape, stackShape) ||
          (outLogdet.dtype != DType.float64 && outLogdet.dtype != a.dtype)) {
        throw ArgumentError.value(
          outLogdet,
          'outLogdet',
          'Must have compatible shape $stackShape and float64 or ${a.dtype} dtype (provided outLogdet buffer has incompatible shape or dtype).',
        );
      }
    }
    return NDArray.scope(() {
      final aF64 = castNDArray<Float64>(a, DType.float64);
      final resF64 = slogdet<Float64, Float64>(aF64);
      if (outSign != null) {
        if (outSign.dtype == DType.float64) {
          resF64.sign.copy(out: outSign as NDArray<Float64>);
        } else {
          castNDArray<T>(resF64.sign, outSign.dtype).copy(out: outSign);
        }
      }
      if (outLogdet != null) {
        if (outLogdet.dtype == DType.float64) {
          resF64.logabsdet.copy(out: outLogdet as NDArray<Float64>);
        } else {
          castNDArray<R>(
            resF64.logabsdet,
            outLogdet.dtype,
          ).copy(out: outLogdet);
        }
      }
      final finalSign =
          outSign ??
          castNDArray<T>(
            resF64.sign,
            a.dtype as DType<T>,
          ).detachToParentScope();
      final finalLogdet =
          outLogdet ??
          castNDArray<R>(
            resF64.logabsdet,
            a.dtype as DType<R>,
          ).detachToParentScope();
      return (sign: finalSign, logabsdet: finalLogdet);
    });
  }

  final DType<R> logdetDType =
      ((a.dtype as DType<DTypeTag>) == DType.float32 ||
          (a.dtype as DType<DTypeTag>) == DType.complex64)
      ? DType.float32 as DType<R>
      : DType.float64 as DType<R>;

  if (outSign != null) {
    validateOutBuffer(outSign, 'outSign');
    if (!listEquals(outSign.shape, stackShape) || outSign.dtype != a.dtype) {
      throw ArgumentError.value(
        outSign,
        'outSign',
        'Must have compatible shape $stackShape and dtype ${a.dtype} (provided outSign buffer has incompatible shape or dtype).',
      );
    }
  }

  if (outLogdet != null) {
    validateOutBuffer(outLogdet, 'outLogdet');
    if (!listEquals(outLogdet.shape, stackShape) ||
        outLogdet.dtype != logdetDType) {
      throw ArgumentError.value(
        outLogdet,
        'outLogdet',
        'Must have compatible shape $stackShape and dtype $logdetDType (provided outLogdet buffer has incompatible shape or dtype).',
      );
    }
  }

  final bool needTempSign =
      outSign != null && (!outSign.isContiguous || sharesMemory(a, outSign));
  final bool needTempLogdet =
      outLogdet != null &&
      (!outLogdet.isContiguous || sharesMemory(a, outLogdet));
  if (needTempSign || needTempLogdet) {
    return NDArray.scope(() {
      final res = slogdet<T, R>(
        a,
        outSign: needTempSign ? null : outSign,
        outLogdet: needTempLogdet ? null : outLogdet,
      );
      if (needTempSign) res.sign.copy(out: outSign);
      if (needTempLogdet) res.logabsdet.copy(out: outLogdet);
      final finalSign = outSign ?? res.sign.detachToParentScope();
      final finalLogdet = outLogdet ?? res.logabsdet.detachToParentScope();
      return (sign: finalSign, logabsdet: finalLogdet);
    });
  }

  return NDArray.scope(() {
    final signResult =
        outSign ?? NDArray<T>.zeros(stackShape, a.dtype as DType<T>);
    final logdetResult = outLogdet ?? NDArray<R>.zeros(stackShape, logdetDType);

    if (a.shape[rank - 1] == 0) {
      signResult.fill(
        castValue(a.dtype.isComplex ? Complex(1.0, 0.0) : 1.0, a.dtype),
      );
      logdetResult.fill(castValue(0.0, logdetDType));
      if (outSign == null) signResult.detachToParentScope();
      if (outLogdet == null) logdetResult.detachToParentScope();
      return (sign: signResult, logabsdet: logdetResult);
    }

    final marker = ScratchArena.marker;
    try {
      final cStridesA = ScratchArena.copyInts(a.strides);
      final cStridesSign = ScratchArena.copyInts(signResult.strides);
      final cStridesLogdet = ScratchArena.copyInts(logdetResult.strides);
      final cShape = ScratchArena.copyInts(a.shape);

      final n = a.shape[rank - 1];

      switch (a.dtype) {
        case DType.float64:
          final cCopy = ScratchArena.allocate<ffi.Double>(
            n * n * ffi.sizeOf<ffi.Double>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );
          s_slogdet_double(
            a.pointer.cast<ffi.Double>(),
            cStridesA,
            signResult.pointer.cast<ffi.Double>(),
            cStridesSign,
            logdetResult.pointer.cast<ffi.Double>(),
            cStridesLogdet,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_dgetrf_ptr().cast(),
          );
        case DType.float32:
          final cCopy = ScratchArena.allocate<ffi.Float>(
            n * n * ffi.sizeOf<ffi.Float>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );
          s_slogdet_float(
            a.pointer.cast<ffi.Float>(),
            cStridesA,
            signResult.pointer.cast<ffi.Float>(),
            cStridesSign,
            logdetResult.pointer.cast<ffi.Float>(),
            cStridesLogdet,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_sgetrf_ptr().cast(),
          );
        case DType.complex128:
          final cCopy = ScratchArena.allocate<cpx_t>(
            n * n * ffi.sizeOf<cpx_t>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );
          s_slogdet_complex_double(
            a.pointer.cast<cpx_t>(),
            cStridesA,
            signResult.pointer.cast<cpx_t>(),
            cStridesSign,
            logdetResult.pointer.cast<ffi.Double>(),
            cStridesLogdet,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_zgetrf_ptr().cast(),
          );
        case DType.complex64:
          final cCopy = ScratchArena.allocate<cpx_f_t>(
            n * n * ffi.sizeOf<cpx_f_t>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );
          s_slogdet_complex_float(
            a.pointer.cast<cpx_f_t>(),
            cStridesA,
            signResult.pointer.cast<cpx_f_t>(),
            cStridesSign,
            logdetResult.pointer.cast<ffi.Float>(),
            cStridesLogdet,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_cgetrf_ptr().cast(),
          );
        default:
          throw UnsupportedError('Unsupported dtype ${a.dtype}');
      }
      checkNativeOom();
    } finally {
      ScratchArena.reset(marker);
    }

    if (outSign == null) {
      signResult.detachToParentScope();
    }
    if (outLogdet == null) {
      logdetResult.detachToParentScope();
    }

    return (sign: signResult, logabsdet: logdetResult);
  });
}

/// Extension on [slogdet] result record type to support easy disposal of both arrays.
extension SlogdetRecordDispose<T extends DTypeTag, R extends DTypeTag>
    on ({NDArray<T> sign, NDArray<R> logabsdet}) {
  /// Disposes both [sign] and [logabsdet] arrays simultaneously.
  void dispose() {
    this.sign.dispose();
    this.logabsdet.dispose();
  }
}

/// Solve a linear matrix equation, or system of linear scalar equations.
///
/// Computes the "exact" solution, `x`, of the linear equation `a * x = b`.
/// Natively offloads to LAPACK solvers (`dgesv`, `sgesv`, `zgesv`, `cgesv`) depending on precision.
///
/// **Preconditions:**
/// - It is an error if [a], [b], or [out] is disposed.
/// - It is an error if [a] is not square (size $N \times N$) or not 2-dimensional.
/// - It is an error if [b] dimensions do not match [a], or dtypes mismatch, or dtypes are unsupported.
/// - It is an error if [out] is provided and has incompatible shape or dtype.
/// - The matrix [a] must be non-singular (invertible).
///
/// **Throws:**
/// - [ArgumentError] if [a] is singular and cannot be solved.
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N^3)$ executed natively.
///
/// **Example:**
/// {@example /example/linalg_example.dart lang=dart}
void _copyStrided2DMatrix(
  ffi.Pointer<ffi.Void> src,
  List<int> strides,
  int offsetElements,
  int n,
  DType dtype,
  ffi.Pointer<ffi.Void> dest,
) {
  final stride0 = strides[strides.length - 2];
  final stride1 = strides[strides.length - 1];
  final byteWidth = dtype.byteWidth;
  if (stride1 == 1) {
    for (var r = 0; r < n; r++) {
      final srcRow = ffi.Pointer<ffi.Uint8>.fromAddress(
        src.address + (offsetElements + r * stride0) * byteWidth,
      );
      final destRow = ffi.Pointer<ffi.Uint8>.fromAddress(
        dest.address + (r * n) * byteWidth,
      );
      custom_memcpy(destRow.cast(), srcRow.cast(), n * byteWidth);
    }
  } else {
    final marker = ScratchArena.marker;
    try {
      final cBuf = ScratchArena.allocate<ffi.Int64>(
        4 * ffi.sizeOf<ffi.Int64>(),
      );
      cBuf[0] = stride0;
      cBuf[1] = stride1;
      cBuf[2] = n;
      cBuf[3] = n;
      final cStrides = cBuf;
      final cShape = cBuf + 2;
      final srcPtr = ffi.Pointer<ffi.Void>.fromAddress(
        src.address + offsetElements * byteWidth,
      );
      switch (dtype) {
        case DType.float64:
          s_flatten_double(srcPtr.cast(), cStrides, dest.cast(), cShape, 2);
        case DType.float32:
          s_flatten_float(srcPtr.cast(), cStrides, dest.cast(), cShape, 2);
        case DType.complex128:
          s_flatten_complex128(srcPtr.cast(), cStrides, dest.cast(), cShape, 2);
        case DType.complex64:
          s_flatten_complex64(srcPtr.cast(), cStrides, dest.cast(), cShape, 2);
        default:
          throw UnsupportedError('Unsupported type: $dtype');
      }
      checkNativeOom();
    } finally {
      ScratchArena.reset(marker);
    }
  }
}

void _lapackeSolve(
  DType dtype,
  int n,
  int nrhs,
  ffi.Pointer<ffi.Void> aPtr,
  ffi.Pointer<ffi.Int> ipiv,
  ffi.Pointer<ffi.Void> bPtr,
) {
  final nfA = _analyzeNonFinitePtr(aPtr, n * n, dtype);
  final nfB = _analyzeNonFinitePtr(bPtr, n * nrhs, dtype);
  if (nfA.hasNaN || nfB.hasNaN) {
    _fillPtrWithNaN(bPtr, n * nrhs, dtype);
    return;
  }
  final int info;
  final String routine;
  switch (dtype) {
    case DType.float64:
      routine = 'LAPACKE_dgesv';
      info = LAPACKE_dgesv(
        101,
        n,
        nrhs,
        aPtr.cast<ffi.Double>(),
        n,
        ipiv,
        bPtr.cast<ffi.Double>(),
        nrhs,
      );
    case DType.float32:
      routine = 'LAPACKE_sgesv';
      info = LAPACKE_sgesv(
        101,
        n,
        nrhs,
        aPtr.cast<ffi.Float>(),
        n,
        ipiv,
        bPtr.cast<ffi.Float>(),
        nrhs,
      );
    case DType.complex128:
      routine = 'LAPACKE_zgesv';
      info = LAPACKE_zgesv(
        101,
        n,
        nrhs,
        aPtr.cast<ffi.Double>(),
        n,
        ipiv,
        bPtr.cast<ffi.Double>(),
        nrhs,
      );
    case DType.complex64:
      routine = 'LAPACKE_cgesv';
      info = LAPACKE_cgesv(
        101,
        n,
        nrhs,
        aPtr.cast<ffi.Float>(),
        n,
        ipiv,
        bPtr.cast<ffi.Float>(),
        nrhs,
      );
    default:
      throw UnsupportedError('Unsupported type for solve: $dtype');
  }
  _checkLapackInfo(
    info,
    routine,
    positiveKind: _LapackFailureKind.singularMatrix,
    positiveMessage: 'Matrix is singular and cannot be solved',
  );
}

/// Solves a linear matrix equation, or framework of equations $AX = B$.
///
/// Computes the exact solution of the matrix equation $AX = B$, where $A$ is a square
/// matrix (or batch of square matrices) of shape `(..., N, N)`, and $B$ is a vector or
/// matrix (or batch) of shape `(..., N)` or `(..., N, K)`.
///
/// **Preconditions:**
/// - $A$ and $B$ must not be disposed.
/// - The last two dimensions of $A$ must be square ($M = N$).
/// - $A$ and $B$ must have matching floating-point or complex data types.
/// - If [out] is provided, it must match the result shape and data type.
///
/// It is an error if $A$ or $B$ is disposed, non-square, or has incompatible shapes/dtypes.
///
/// Throws a [SingularMatrixException] if the matrix $A$ is singular or ill-conditioned.
///
/// **Example:**
/// {@example /example/linalg_example.dart#solve_system lang=dart}
NDArray<T> solve<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute solve() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write solve result to a disposed output array.');
  }
  final rankA = a.shape.length;
  if (rankA < 2 || a.shape[rankA - 2] != a.shape[rankA - 1]) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must be square in the last 2 dimensions and rank >= 2 (was ${a.shape})',
    );
  }
  final n = a.shape[rankA - 1];
  final rankB = b.shape.length;

  if (rankA == 2) {
    if (rankB == 1) {
      if (b.shape[0] != n) {
        throw ArgumentError.value(
          b.shape,
          'b.shape',
          'Must match matrix dimension $n of a (${a.shape}) (dimensions of b (${b.shape}) must match matrix dimension $n of a (${a.shape}))',
        );
      }
    } else if (rankB == 2) {
      if (b.shape[0] != n) {
        throw ArgumentError.value(
          b.shape,
          'b.shape',
          'Must match matrix dimension $n of a (${a.shape}) (dimensions of b (${b.shape}) must match matrix dimension $n of a (${a.shape}))',
        );
      }
    } else {
      throw ArgumentError.value(
        b.shape,
        'b.shape',
        'Must have rank 1 or 2 (dimensions of b (${b.shape}) are incompatible with a (${a.shape}). Expected rank 1 or 2.)',
      );
    }
  } else {
    final stackShapeA = a.shape.sublist(0, rankA - 2);
    if (rankB == 1) {
      if (b.shape[0] != n) {
        throw ArgumentError.value(
          b.shape,
          'b.shape',
          'Must match matrix dimension $n of a (${a.shape}) (dimensions of b (${b.shape}) must match matrix dimension $n of a (${a.shape}))',
        );
      }
    } else if (rankB == rankA) {
      if (!listEquals(b.shape.sublist(0, rankA - 2), stackShapeA) ||
          b.shape[rankB - 2] != n) {
        throw ArgumentError.value(
          b.shape,
          'b.shape',
          'Must match stack shape $stackShapeA and matrix dimension $n of a (${a.shape}) (dimensions of b (${b.shape}) must match stack shape $stackShapeA and matrix dimension $n of a (${a.shape}))',
        );
      }
    } else {
      throw ArgumentError.value(
        b.shape,
        'b.shape',
        'Must have rank 1 or $rankA (dimensions of b (${b.shape}) are incompatible with a (${a.shape}). Expected rank 1 or $rankA.)',
      );
    }
  }

  final expectedOutShape = (rankA > 2 && rankB == 1)
      ? [...a.shape.sublist(0, rankA - 2), n]
      : b.shape;

  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b.dtype',
      'Must match dtype of a (${a.dtype}) (mismatched dtypes for solve: a has dtype ${a.dtype}, b has dtype ${b.dtype}).',
    );
  }

  if (a.dtype != DType.float64 &&
      a.dtype != DType.float32 &&
      a.dtype != DType.float16 &&
      a.dtype != DType.bfloat16 &&
      a.dtype != DType.complex128 &&
      a.dtype != DType.complex64) {
    throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      'Must be float or complex (solve only supports float64, float32, float16, bfloat16, complex128, or complex64 dtypes, got ${a.dtype}).',
    );
  }

  final nrhs = rankB == rankA ? b.shape[rankB - 1] : 1;
  checkBlasIntDim(n, 'n', 'solve');
  checkBlasIntDim(nrhs, 'nrhs', 'solve');
  checkBlasIntStride(a.strides[rankA - 2], 'lda', 'solve');
  checkBlasIntStride(a.strides[rankA - 1], 'lda', 'solve');
  if (rankB == 1) {
    checkBlasIntStride(b.strides[0], 'ldb', 'solve');
  } else {
    checkBlasIntStride(b.strides[rankB - 2], 'ldb', 'solve');
    checkBlasIntStride(b.strides[rankB - 1], 'ldb', 'solve');
  }

  if (a.dtype == DType.float16 || a.dtype == DType.bfloat16) {
    if (out != null) {
      validateOutBuffer(out);
      if (!listEquals(out.shape, expectedOutShape) ||
          (out.dtype != DType.float64 && out.dtype != b.dtype)) {
        throw ArgumentError.value(
          out,
          'out',
          'Must have compatible shape $expectedOutShape and dtype ${DType.float64} (provided out buffer has incompatible shape or dtype, expected shape $expectedOutShape and dtype ${DType.float64}, got shape ${out.shape} and dtype ${out.dtype}).',
        );
      }
    }
    return NDArray.scope(() {
      final aF64 = castNDArray<Float64>(a, DType.float64);
      final bF64 = castNDArray<Float64>(b, DType.float64);
      final resF64 = solve<Float64>(aF64, bF64);
      if (out != null) {
        if (out.dtype == DType.float64) {
          resF64.copy(out: out as NDArray<Float64>);
        } else {
          castNDArray<T>(resF64, out.dtype).copy(out: out);
        }
        return out;
      }
      return castNDArray<T>(resF64, a.dtype).detachToParentScope();
    });
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, expectedOutShape) || out.dtype != b.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape $expectedOutShape and dtype ${b.dtype} (provided out buffer has incompatible shape or dtype, expected shape $expectedOutShape and dtype ${b.dtype}, got shape ${out.shape} and dtype ${out.dtype}).',
      );
    }
    if (!out.isContiguous || sharesMemory(a, out) || sharesMemory(b, out)) {
      return NDArray.scope(() {
        final temp = solve<T>(a, b);
        temp.copy(out: out);
        return out;
      });
    }
  }

  final NDArray<T> bCopy;
  if (rankA > 2 && rankB == 1) {
    bCopy = out ?? NDArray<T>.create(expectedOutShape, b.dtype);
    final bBroadcast = broadcastTo<T>(b, expectedOutShape);
    bBroadcast.copy(out: bCopy);
    bBroadcast.dispose();
  } else if (out != null) {
    bCopy = out;
    b.copy(out: bCopy);
  } else {
    bCopy = b.copy();
  }

  if (n == 0) {
    return bCopy;
  }

  var success = false;
  final marker = ScratchArena.marker;
  try {
    final ipiv = ScratchArena.allocate<ffi.Int>(n * ffi.sizeOf<ffi.Int>());
    final aCopyPtr = ScratchArena.allocate<ffi.Uint8>(
      n * n * a.dtype.byteWidth,
    );

    if (rankA == 2) {
      if (a.isContiguous) {
        custom_memcpy(aCopyPtr.cast(), a.pointer, n * n * a.dtype.byteWidth);
      } else {
        _copyStrided2DMatrix(
          a.pointer,
          a.strides,
          0,
          n,
          a.dtype,
          aCopyPtr.cast(),
        );
      }
      _lapackeSolve(a.dtype, n, nrhs, aCopyPtr.cast(), ipiv, bCopy.pointer);
      success = true;
      return bCopy;
    }

    final stackDims = rankA - 2;
    final stackShape = a.shape.sublist(0, stackDims);
    var stackSize = 1;
    for (var d = 0; d < stackDims; d++) {
      stackSize *= stackShape[d];
    }

    final coords = List<int>.filled(stackDims, 0);
    final aStrides = a.strides;
    final bCopyStrides = bCopy.strides;
    final bElemSize = b.dtype.byteWidth;

    for (var step = 0; step < stackSize; step++) {
      var offsetA = 0;
      var offsetB = 0;
      for (var d = 0; d < stackDims; d++) {
        offsetA += coords[d] * aStrides[d];
        offsetB += coords[d] * bCopyStrides[d];
      }

      final aDType = a.dtype;
      final aByteWidth = aDType.byteWidth;
      final isSliceAContig =
          aStrides[rankA - 2] == n && aStrides[rankA - 1] == 1;
      if (isSliceAContig) {
        final srcSliceA = ffi.Pointer<ffi.Uint8>.fromAddress(
          a.pointer.address + offsetA * aByteWidth,
        );
        custom_memcpy(aCopyPtr.cast(), srcSliceA.cast(), n * n * aByteWidth);
      } else {
        _copyStrided2DMatrix(
          a.pointer,
          aStrides,
          offsetA,
          n,
          aDType,
          aCopyPtr.cast(),
        );
      }

      final bSlicePtr = ffi.Pointer<ffi.Void>.fromAddress(
        bCopy.pointer.address + offsetB * bElemSize,
      );

      _lapackeSolve(aDType, n, nrhs, aCopyPtr.cast(), ipiv, bSlicePtr);

      for (var d = stackDims - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < stackShape[d]) break;
        coords[d] = 0;
      }
    }

    success = true;
    return bCopy;
  } finally {
    ScratchArena.reset(marker);
    if (out == null && !success) {
      bCopy.dispose();
    }
  }
}

/// Computes the eigenvalues and right eigenvectors of a square array or stack of square arrays.
///
/// Returns a record `(eigenvalues, eigenvectors)` containing:
/// - **eigenvalues**: An `NDArray<DTypeTag>` of shape `[..., N]` containing the eigenvalues.
/// - **eigenvectors**: An `NDArray<DTypeTag>` of shape `[..., N, N]` containing the corresponding right eigenvectors as columns.
///
/// Both are returned with `Complex` elements because eigenvalues and eigenvectors can be complex
/// even for real matrices.
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] is not square in its last two dimensions or is less than 2-dimensional.
/// - It is an error if the DType of [a] is not supported.
/// - It is an error if [out] is provided and has incompatible shape or dtype.
///
/// **Throws:**
/// - Throws an [IterationsExceededException] if the eigenvalue computation does not converge.
/// - Throws a [LinAlgException] if [a] contains non-finite values (`NaN` or `±Infinity`) or if the LAPACK routine fails.
({NDArray<R> eigenvalues, NDArray<R> eigenvectors}) eig<R extends DTypeTag>(
  NDArray<
    DTypeSpec<
      DTypeTag,
      Object?,
      DTypeTag,
      R,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag
    >
  >
  a, {
  ({NDArray<R> eigenvalues, NDArray<R> eigenvectors})? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute eig of a disposed array.');
  }
  if (out != null &&
      (out.eigenvalues.isDisposed || out.eigenvectors.isDisposed)) {
    throw StateError('Cannot write eig result to a disposed output array.');
  }
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 1] != a.shape[rank - 2]) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must be square and at least 2D (was ${a.shape})',
    );
  }
  final n = a.shape[rank - 1];
  checkBlasIntDim(n, 'n', 'eig');
  checkBlasIntStride(a.strides[rank - 2], 'lda', 'eig');
  checkBlasIntStride(a.strides[rank - 1], 'lda', 'eig');
  final stackShape = a.shape.sublist(0, rank - 2);

  final compDType =
      ((a.dtype as DType<DTypeTag>) == DType.float32 ||
          (a.dtype as DType<DTypeTag>) == DType.complex64)
      ? DType.complex64
      : DType.complex128;

  final wShape = [...stackShape, n];
  final vrShape = [...stackShape, n, n];

  return NDArray.scope(() {
    final NDArray<R> w;
    final NDArray<R> vr;

    if (out != null) {
      w = out.eigenvalues;
      vr = out.eigenvectors;
      validateOutBuffer(w, 'out.eigenvalues');
      validateOutBuffer(vr, 'out.eigenvectors');
      if (!listEquals(w.shape, wShape) || w.dtype != compDType) {
        throw ArgumentError.value(
          w,
          'out.eigenvalues',
          'Must have compatible shape $wShape and dtype $compDType (provided out eigenvalues buffer has incompatible shape or dtype, got shape ${w.shape} and dtype ${w.dtype}).',
        );
      }
      if (!listEquals(vr.shape, vrShape) || vr.dtype != compDType) {
        throw ArgumentError.value(
          vr,
          'out.eigenvectors',
          'Must have compatible shape $vrShape and dtype $compDType (provided out eigenvectors buffer has incompatible shape or dtype, got shape ${vr.shape} and dtype ${vr.dtype}).',
        );
      }
      if (sharesMemory(w, vr)) {
        throw ArgumentError.value(
          vr,
          'out.eigenvectors',
          'Must not share memory with out.eigenvalues.',
        );
      }
      if (!w.isContiguous ||
          !vr.isContiguous ||
          sharesMemory(a, w) ||
          sharesMemory(a, vr)) {
        final temp = eig<R>(a);
        temp.eigenvalues.copy(out: w);
        temp.eigenvectors.copy(out: vr);
        return (eigenvalues: w, eigenvectors: vr);
      }
    } else {
      w = NDArray<R>.create(wShape, compDType as DType<R>);
      vr = NDArray<R>.create(vrShape, compDType);
    }

    if (n == 0) {
      if (out == null) {
        w.detachToParentScope();
        vr.detachToParentScope();
      }
      return (eigenvalues: w, eigenvectors: vr);
    }

    final jobvl = 'N'.codeUnitAt(0);
    final jobvr = 'V'.codeUnitAt(0);

    final bool wasCast =
        a.dtype.isInteger ||
        (a.dtype as DType<DTypeTag>) == DType.float16 ||
        (a.dtype as DType<DTypeTag>) == DType.bfloat16;
    final NDArray src = wasCast ? castNDArray(a, DType.float64) : a;
    if (src.dtype != DType.complex128 &&
        src.dtype != DType.complex64 &&
        src.dtype != DType.float64 &&
        src.dtype != DType.float32) {
      throw UnimplementedError('Type ${src.dtype} not supported for eig');
    }
    try {
      walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
        coords,
      ) {
        var offsetA = 0;
        for (var i = 0; i < coords.length; i++) {
          offsetA += coords[i] * src.strides[i];
        }

        final sliceView = NDArray.view(
          src,
          shape: [n, n],
          strides: src.strides.sublist(rank - 2),
          offsetElements: offsetA,
        );
        final sliceCopy = sliceView.copy();
        sliceView.dispose();

        try {
          var offsetW = 0;
          for (var i = 0; i < coords.length; i++) {
            offsetW += coords[i] * w.strides[i];
          }
          var offsetVR = 0;
          for (var i = 0; i < coords.length; i++) {
            offsetVR += coords[i] * vr.strides[i];
          }

          final nf = _analyzeNonFinitePtr(sliceCopy.pointer, n * n, src.dtype);
          if (nf.hasNaN || nf.hasInf) {
            throw const LinAlgException(
              'Array must not contain infs or NaNs in eig.',
            );
          }

          switch (src.dtype) {
            case DType.complex128:
              final w2D = NDArray<DTypeTag>.create([n], DType.complex128);
              final vr2D = NDArray<DTypeTag>.create([n, n], DType.complex128);
              try {
                final info = LAPACKE_zgeev(
                  101, // ROW_MAJOR
                  jobvl,
                  jobvr,
                  n,
                  sliceCopy.pointer.cast<ffi.Double>(),
                  n,
                  w2D.pointer.cast<ffi.Double>(),
                  ffi.nullptr.cast<ffi.Double>(),
                  n,
                  vr2D.pointer.cast<ffi.Double>(),
                  n,
                );

                _checkLapackInfo(
                  info,
                  'LAPACKE_zgeev',
                  positiveKind: _LapackFailureKind.iterationsExceeded,
                  positiveMessage:
                      'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
                );

                final wView = NDArray<DTypeTag>.view(
                  w,
                  shape: [n],
                  strides: w.strides.isEmpty ? [1] : [w.strides.last],
                  offsetElements: offsetW,
                );
                w2D.copy(out: wView);
                wView.dispose();

                final vrView = NDArray<DTypeTag>.view(
                  vr,
                  shape: [n, n],
                  strides: vr.strides.sublist(rank - 2),
                  offsetElements: offsetVR,
                );
                vr2D.copy(out: vrView);
                vrView.dispose();
              } finally {
                w2D.dispose();
                vr2D.dispose();
              }
            case DType.complex64:
              final w2D = NDArray<DTypeTag>.create([n], DType.complex64);
              final vr2D = NDArray<DTypeTag>.create([n, n], DType.complex64);
              try {
                final info = LAPACKE_cgeev(
                  101, // ROW_MAJOR
                  jobvl,
                  jobvr,
                  n,
                  sliceCopy.pointer.cast<ffi.Float>(),
                  n,
                  w2D.pointer.cast<ffi.Float>(),
                  ffi.nullptr.cast<ffi.Float>(),
                  n,
                  vr2D.pointer.cast<ffi.Float>(),
                  n,
                );

                _checkLapackInfo(
                  info,
                  'LAPACKE_cgeev',
                  positiveKind: _LapackFailureKind.iterationsExceeded,
                  positiveMessage:
                      'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
                );

                final wView = NDArray<DTypeTag>.view(
                  w,
                  shape: [n],
                  strides: w.strides.isEmpty ? [1] : [w.strides.last],
                  offsetElements: offsetW,
                );
                w2D.copy(out: wView);
                wView.dispose();

                final vrView = NDArray<DTypeTag>.view(
                  vr,
                  shape: [n, n],
                  strides: vr.strides.sublist(rank - 2),
                  offsetElements: offsetVR,
                );
                vr2D.copy(out: vrView);
                vrView.dispose();
              } finally {
                w2D.dispose();
                vr2D.dispose();
              }
            case DType.float64:
              final wr = NDArray<Float64>.zeros([n], DType.float64);
              final wi = NDArray<Float64>.zeros([n], DType.float64);
              final vrReal = NDArray<Float64>.create([n, n], DType.float64);
              try {
                final info = LAPACKE_dgeev(
                  101,
                  jobvl,
                  jobvr,
                  n,
                  sliceCopy.pointer.cast<ffi.Double>(),
                  n,
                  wr.pointer.cast<ffi.Double>(),
                  wi.pointer.cast<ffi.Double>(),
                  ffi.nullptr.cast<ffi.Double>(),
                  n,
                  vrReal.pointer.cast<ffi.Double>(),
                  n,
                );

                _checkLapackInfo(
                  info,
                  'LAPACKE_dgeev',
                  positiveKind: _LapackFailureKind.iterationsExceeded,
                  positiveMessage:
                      'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
                );

                final strideWLast = w.strides.isEmpty ? 1 : w.strides.last;
                final strideVR1 = vr.strides[rank - 2];
                final strideVR2 = vr.strides[rank - 1];
                assemble_eigenvectors_double(
                  w.pointer.cast<cpx_t>() + offsetW,
                  strideWLast,
                  vr.pointer.cast<cpx_t>() + offsetVR,
                  strideVR1,
                  strideVR2,
                  wr.pointer.cast<ffi.Double>(),
                  wi.pointer.cast<ffi.Double>(),
                  vrReal.pointer.cast<ffi.Double>(),
                  n,
                );
              } finally {
                wr.dispose();
                wi.dispose();
                vrReal.dispose();
              }
            case DType.float32:
              final wr = NDArray<Float32>.zeros([n], DType.float32);
              final wi = NDArray<Float32>.zeros([n], DType.float32);
              final vrReal = NDArray<Float32>.create([n, n], DType.float32);
              try {
                final info = LAPACKE_sgeev(
                  101,
                  jobvl,
                  jobvr,
                  n,
                  sliceCopy.pointer.cast<ffi.Float>(),
                  n,
                  wr.pointer.cast<ffi.Float>(),
                  wi.pointer.cast<ffi.Float>(),
                  ffi.nullptr.cast<ffi.Float>(),
                  n,
                  vrReal.pointer.cast<ffi.Float>(),
                  n,
                );

                _checkLapackInfo(
                  info,
                  'LAPACKE_sgeev',
                  positiveKind: _LapackFailureKind.iterationsExceeded,
                  positiveMessage:
                      'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
                );

                final strideWLast = w.strides.isEmpty ? 1 : w.strides.last;
                final strideVR1 = vr.strides[rank - 2];
                final strideVR2 = vr.strides[rank - 1];
                assemble_eigenvectors_float(
                  w.pointer.cast<cpx_f_t>() + offsetW,
                  strideWLast,
                  vr.pointer.cast<cpx_f_t>() + offsetVR,
                  strideVR1,
                  strideVR2,
                  wr.pointer.cast<ffi.Float>(),
                  wi.pointer.cast<ffi.Float>(),
                  vrReal.pointer.cast<ffi.Float>(),
                  n,
                );
              } finally {
                wr.dispose();
                wi.dispose();
                vrReal.dispose();
              }
            default:
              throw UnimplementedError(
                'Type ${src.dtype} not supported for eig',
              );
          }
        } finally {
          sliceCopy.dispose();
        }
      });
    } finally {
      if (wasCast) {
        src.dispose();
      }
    }

    if (out == null) {
      w.detachToParentScope();
      vr.detachToParentScope();
    }
    return (eigenvalues: w, eigenvectors: vr);
  });
}

/// Computes only the eigenvalues of a general square 2D matrix or stack of matrices.
///
/// Unlike [eig], this function does not compute eigenvectors, making it much faster.
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] is not square or rank < 2.
/// - It is an error if [a] has integer dtype or an unsupported dtype.
/// - It is an error if [out] is provided and has incompatible shape or dtype.
///
/// **Throws:**
/// - Throws an [IterationsExceededException] if the eigenvalue computation does not converge.
/// - Throws a [LinAlgException] if [a] contains non-finite values (`NaN` or `±Infinity`) or if the LAPACK routine fails.
///
/// **Returns:**
/// - A contiguous `NDArray<DTypeTag>` containing the computed eigenvalues.
///
/// Reference: [NumPy linalg.eigvals](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html)
NDArray<R> eigvals<R extends DTypeTag>(
  NDArray<
    DTypeSpec<
      DTypeTag,
      Object?,
      DTypeTag,
      R,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag
    >
  >
  a, {
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute eigvals of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write eigvals result to a disposed output array.');
  }
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 1] != a.shape[rank - 2]) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must be square and at least 2D (was ${a.shape})',
    );
  }
  final n = a.shape[rank - 1];
  checkBlasIntDim(n, 'n', 'eigvals');
  checkBlasIntStride(a.strides[rank - 2], 'lda', 'eigvals');
  checkBlasIntStride(a.strides[rank - 1], 'lda', 'eigvals');
  final stackShape = a.shape.sublist(0, rank - 2);

  final compDType =
      ((a.dtype as DType<DTypeTag>) == DType.float32 ||
          (a.dtype as DType<DTypeTag>) == DType.complex64)
      ? DType.complex64
      : DType.complex128;

  final wShape = [...stackShape, n];

  return NDArray.scope(() {
    final NDArray<R> w;

    if (out != null) {
      validateOutBuffer(out);
      w = out;
      if (!listEquals(w.shape, wShape) || w.dtype != compDType) {
        throw ArgumentError.value(
          w,
          'out',
          'Must have compatible shape $wShape and dtype $compDType (provided out eigenvalues buffer has incompatible shape or dtype, got shape ${w.shape} and dtype ${w.dtype}).',
        );
      }
      if (!w.isContiguous || sharesMemory(a, w)) {
        final temp = eigvals<R>(a);
        temp.copy(out: w);
        return w;
      }
    } else {
      w = NDArray<R>.create(wShape, compDType as DType<R>);
    }

    if (n == 0) {
      if (out == null) {
        w.detachToParentScope();
      }
      return w;
    }

    final jobvl = 'N'.codeUnitAt(0);
    final jobvr = 'N'.codeUnitAt(0);

    final bool wasCast =
        a.dtype.isInteger ||
        (a.dtype as DType<DTypeTag>) == DType.float16 ||
        (a.dtype as DType<DTypeTag>) == DType.bfloat16;
    final NDArray src = wasCast ? castNDArray(a, DType.float64) : a;
    try {
      if (src.dtype != DType.complex128 &&
          src.dtype != DType.complex64 &&
          src.dtype != DType.float64 &&
          src.dtype != DType.float32) {
        throw UnimplementedError('Type ${src.dtype} not supported for eigvals');
      }

      walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
        coords,
      ) {
        var offsetA = 0;
        for (var i = 0; i < coords.length; i++) {
          offsetA += coords[i] * src.strides[i];
        }

        final sliceView = NDArray.view(
          src,
          shape: [n, n],
          strides: src.strides.sublist(rank - 2),
          offsetElements: offsetA,
        );
        final sliceCopy = sliceView.copy();
        sliceView.dispose();

        try {
          var offsetW = 0;
          for (var i = 0; i < coords.length; i++) {
            offsetW += coords[i] * w.strides[i];
          }

          final nf = _analyzeNonFinitePtr(sliceCopy.pointer, n * n, src.dtype);
          if (nf.hasNaN || nf.hasInf) {
            throw const LinAlgException(
              'Array must not contain infs or NaNs in eigvals.',
            );
          }

          switch (src.dtype) {
            case DType.complex128:
              final w2D = NDArray<DTypeTag>.create([n], DType.complex128);
              try {
                final info = LAPACKE_zgeev(
                  101, // ROW_MAJOR
                  jobvl,
                  jobvr,
                  n,
                  sliceCopy.pointer.cast<ffi.Double>(),
                  n,
                  w2D.pointer.cast<ffi.Double>(),
                  ffi.nullptr.cast<ffi.Double>(),
                  1, // ldvl
                  ffi.nullptr.cast<ffi.Double>(),
                  1, // ldvr
                );

                _checkLapackInfo(
                  info,
                  'LAPACKE_zgeev',
                  positiveKind: _LapackFailureKind.iterationsExceeded,
                  positiveMessage:
                      'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
                );

                final wView = NDArray<DTypeTag>.view(
                  w,
                  shape: [n],
                  strides: w.strides.isEmpty ? [1] : [w.strides.last],
                  offsetElements: offsetW,
                );
                w2D.copy(out: wView);
                wView.dispose();
              } finally {
                w2D.dispose();
              }

            case DType.complex64:
              final w2D = NDArray<DTypeTag>.create([n], DType.complex64);
              try {
                final info = LAPACKE_cgeev(
                  101, // ROW_MAJOR
                  jobvl,
                  jobvr,
                  n,
                  sliceCopy.pointer.cast<ffi.Float>(),
                  n,
                  w2D.pointer.cast<ffi.Float>(),
                  ffi.nullptr.cast<ffi.Float>(),
                  1, // ldvl
                  ffi.nullptr.cast<ffi.Float>(),
                  1, // ldvr
                );

                _checkLapackInfo(
                  info,
                  'LAPACKE_cgeev',
                  positiveKind: _LapackFailureKind.iterationsExceeded,
                  positiveMessage:
                      'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
                );

                final wView = NDArray<DTypeTag>.view(
                  w,
                  shape: [n],
                  strides: w.strides.isEmpty ? [1] : [w.strides.last],
                  offsetElements: offsetW,
                );
                w2D.copy(out: wView);
                wView.dispose();
              } finally {
                w2D.dispose();
              }

            case DType.float64:
              final wr = NDArray<Float64>.zeros([n], DType.float64);
              final wi = NDArray<Float64>.zeros([n], DType.float64);
              try {
                final info = LAPACKE_dgeev(
                  101,
                  jobvl,
                  jobvr,
                  n,
                  sliceCopy.pointer.cast<ffi.Double>(),
                  n,
                  wr.pointer.cast<ffi.Double>(),
                  wi.pointer.cast<ffi.Double>(),
                  ffi.nullptr.cast<ffi.Double>(),
                  1, // ldvl
                  ffi.nullptr.cast<ffi.Double>(),
                  1, // ldvr
                );

                _checkLapackInfo(
                  info,
                  'LAPACKE_dgeev',
                  positiveKind: _LapackFailureKind.iterationsExceeded,
                  positiveMessage:
                      'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
                );

                final strideWLast = w.strides.isEmpty ? 1 : w.strides.last;
                assemble_eigenvalues_double(
                  w.pointer.cast<cpx_t>() + offsetW,
                  strideWLast,
                  wr.pointer.cast<ffi.Double>(),
                  wi.pointer.cast<ffi.Double>(),
                  n,
                );
              } finally {
                wr.dispose();
                wi.dispose();
              }

            case DType.float32:
              final wr = NDArray<Float32>.zeros([n], DType.float32);
              final wi = NDArray<Float32>.zeros([n], DType.float32);
              try {
                final info = LAPACKE_sgeev(
                  101,
                  jobvl,
                  jobvr,
                  n,
                  sliceCopy.pointer.cast<ffi.Float>(),
                  n,
                  wr.pointer.cast<ffi.Float>(),
                  wi.pointer.cast<ffi.Float>(),
                  ffi.nullptr.cast<ffi.Float>(),
                  1, // ldvl
                  ffi.nullptr.cast<ffi.Float>(),
                  1, // ldvr
                );

                _checkLapackInfo(
                  info,
                  'LAPACKE_sgeev',
                  positiveKind: _LapackFailureKind.iterationsExceeded,
                  positiveMessage:
                      'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
                );

                final strideWLast = w.strides.isEmpty ? 1 : w.strides.last;
                assemble_eigenvalues_float(
                  w.pointer.cast<cpx_f_t>() + offsetW,
                  strideWLast,
                  wr.pointer.cast<ffi.Float>(),
                  wi.pointer.cast<ffi.Float>(),
                  n,
                );
              } finally {
                wr.dispose();
                wi.dispose();
              }
            default:
              throw UnimplementedError(
                'Type ${src.dtype} not supported for eigvals',
              );
          }
        } finally {
          sliceCopy.dispose();
        }
      });

      if (out == null) {
        w.detachToParentScope();
      }
      return w;
    } finally {
      if (wasCast) src.dispose();
    }
  });
}

/// Computes the Moore-Penrose pseudo-inverse of a 2D matrix.
///
/// Uses Singular Value Decomposition (SVD) to resolve the pseudo-inverse.
/// Singular values smaller than [rcond] * max(singular_value) are treated as zero.
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] does not have rank == 2.
/// - It is an error if [out] is provided and has incompatible shape or dtype.
///
/// **Example:**
/// {@example /example/linalg_premium_example.dart lang=dart}
NDArray<T> pinv<T extends DTypeTag>(
  NDArray<T> a, {
  double? rcond,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute pinv() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write pinv result to a disposed output array.');
  }
  if (a.shape.length != 2) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must be a 2D matrix (Moore-Penrose pseudo-inverse is only defined for 2D matrices, was shape ${a.shape}).',
    );
  }
  final m = a.shape[0];
  final n = a.shape[1];
  checkBlasIntDim(m, 'm', 'pinv');
  checkBlasIntDim(n, 'n', 'pinv');
  checkBlasIntStride(a.strides[0], 'lda', 'pinv');
  checkBlasIntStride(a.strides[1], 'lda', 'pinv');

  final targetShape = [n, m];
  if (a.dtype == DType.float16 || a.dtype == DType.bfloat16) {
    if (out != null) {
      validateOutBuffer(out);
      if (!listEquals(out.shape, targetShape) ||
          (out.dtype != DType.float64 && out.dtype != a.dtype)) {
        throw ArgumentError.value(
          out,
          'out',
          'Must have compatible shape $targetShape and float64 or ${a.dtype} dtype (provided out buffer has incompatible shape or dtype).',
        );
      }
    }
    return NDArray.scope(() {
      final aF64 = castNDArray<Float64>(a, DType.float64);
      final resF64 = pinv<Float64>(aF64, rcond: rcond);
      if (out != null) {
        if (out.dtype == DType.float64) {
          resF64.copy(out: out as NDArray<Float64>);
        } else {
          castNDArray<T>(resF64, out.dtype).copy(out: out);
        }
        return out;
      }
      return castNDArray<T>(resF64, a.dtype).detachToParentScope();
    });
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape $targetShape and dtype ${a.dtype} (provided out buffer has incompatible shape or dtype).',
      );
    }
    if (!out.isContiguous || sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = pinv<T>(a, rcond: rcond);
        temp.copy(out: out);
        return out;
      });
    }
  }

  return NDArray.scope(() {
    final result = out ?? NDArray<T>.create(targetShape, a.dtype);
    if (m == 0 || n == 0) {
      if (out == null) {
        result.detachToParentScope();
      }
      return result;
    }
    final svdResult = _svd<T>(a);
    final u = svdResult.u;
    final s = svdResult.s;
    final vt = svdResult.vh;

    final double maxSingularVal = (s.dtype == DType.float32)
        ? s.pointer.cast<ffi.Float>()[0]
        : s.pointer.cast<ffi.Double>()[0];
    final epsilon = 2.220446049250313e-16;
    final maxDim = m > n ? m : n;
    final resolvedRcond = rcond ?? (maxDim * epsilon);
    final threshold = resolvedRcond * maxSingularVal;

    final aDType = a.dtype;
    final sIsF32 = s.dtype == DType.float32;
    final sPlus = NDArray.zeros([n, m], aDType);
    for (var i = 0; i < s.shape[0]; i++) {
      final double sVal = sIsF32
          ? s.pointer.cast<ffi.Float>()[i]
          : s.pointer.cast<ffi.Double>()[i];
      if (sVal > threshold) {
        sPlus.setCell([i, i], castValue(1.0 / sVal, aDType));
      }
    }

    final v = conjugate(vt.transpose());
    final ut = conjugate(u.transpose());

    final temp = matmul(v, sPlus);
    matmul(temp, ut, out: result);

    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}

/// Raise a square 2D matrix to the integer power [n].
///
/// Computes $A^n$ using binary exponentiation (square-and-multiply)
/// in $O(\log n)$ matrix multiplications.
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] has rank != 2 or is not square.
/// - It is an error if [out] has mismatched shape or dtype.
///
/// **Example:**
/// {@example /example/linalg_premium_example.dart lang=dart}
NDArray<T> matrix_power<T extends DTypeTag>(
  NDArray<T> a,
  int n, {
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute matrix_power() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write matrix_power result to a disposed output array.',
    );
  }
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must be a 2D square matrix (matrix_power is only defined for 2D square matrices, was shape ${a.shape}).',
    );
  }
  if (n < 0 && a.dtype.isInteger) {
    throw ArgumentError.value(
      n,
      'n',
      'Must be non-negative for integer matrices (integer matrices cannot be raised to negative powers because matrix inversion requires floating point types. Please convert the matrix to float64 or float32 first).',
    );
  }

  final size = a.shape[0];
  if (a.dtype.isFloating || a.dtype.isComplex || n < 0) {
    checkBlasIntDim(size, 'n', 'matrix_power');
    checkBlasIntStride(a.strides[0], 'lda', 'matrix_power');
    checkBlasIntStride(a.strides[1], 'lda', 'matrix_power');
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape ${a.shape} and dtype ${a.dtype} (provided out buffer has incompatible shape or dtype).',
      );
    }
    if (!out.isContiguous || sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = matrix_power<T>(a, n);
        temp.copy(out: out);
        return out;
      });
    }
  }

  return NDArray.scope(() {
    final result = out ?? NDArray<T>.create(a.shape, a.dtype);
    if (n == 0) {
      final eye = NDArray.eye(size, a.dtype);
      result.fill(normalizeScalar(0, a.dtype));
      for (var i = 0; i < size; i++) {
        result.setCell([i, i], eye.getCell([i, i]));
      }
      if (out == null) {
        result.detachToParentScope();
      }
      return result;
    }

    NDArray base;
    if (n < 0) {
      final invA = inv(a);
      base = invA.dtype == a.dtype ? invA : castNDArray<T>(invA, a.dtype);
      n = -n;
    } else {
      base = a;
    }

    if (n == 1) {
      base.copy(out: result);
      if (out == null) {
        result.detachToParentScope();
      }
      return result;
    }

    var res = NDArray<T>.eye(size, a.dtype);
    var tempRes = NDArray<T>.zeros(a.shape, a.dtype);

    var current = base.copy() as NDArray<T>;
    var tempCurrent = NDArray<T>.zeros(a.shape, a.dtype);

    var exponent = n;
    while (exponent > 0) {
      if ((exponent & 1) == 1) {
        matmul(res, current, out: tempRes);
        final tmp = res;
        res = tempRes;
        tempRes = tmp;
      }
      if (exponent > 1) {
        matmul(current, current, out: tempCurrent);
        final tmp = current;
        current = tempCurrent;
        tempCurrent = tmp;
      }
      exponent >>= 1;
    }

    res.copy(out: result);

    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}

/// Computes the Cholesky decomposition of a square symmetric/Hermitian positive-definite matrix or stack of matrices.
///
/// Returns the lower triangular factor $L$ such that $A = L L^*$ (or $A = L L^T$).
/// Natively offloads to LAPACK solvers (`dpotrf`, `spotrf`, `cpotrf`, `zpotrf`) depending on precision and complexity.
///
/// **Preconditions:**
/// - The input matrix [a] must not be disposed.
/// - The input matrix [a] must have rank $\\ge 2$ and square trailing dimensions (`a.shape[a.rank - 2] == a.shape[a.rank - 1]`).
/// - The input matrix [a] must have a floating-point or complex data type (`float32`, `float64`, `complex64`, or `complex128`).
/// - Each matrix slice in [a] must be symmetric/Hermitian positive-definite.
/// - If provided, the [out] destination matrix must have the same shape and dtype as [a].
///
/// **Throws:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] has rank < 2 or trailing dimensions are not square.
/// - It is an error if [a] has an unsupported dtype (e.g. integer or boolean).
/// - It is an error if the provided [out] buffer has an incompatible shape or dtype.
/// - Throws a [NonPositiveDefiniteException] if any matrix slice is not positive-definite or contains `NaN`.
/// - Throws a [LinAlgException] if the underlying LAPACK routine fails.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(B \\times n^3)$ flops for $B$ batches of $n \times n$ matrices.
/// - Uses LAPACK solvers.
/// - Performs zero memory allocations if a pre-allocated [out] buffer is provided and the input [a] is contiguous.
///
/// **Example:**
/// {@example /example/linalg_example.dart lang=dart}
///
/// Reference: [NumPy linalg.cholesky](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html)
NDArray<T> cholesky<T extends DTypeTag>(NDArray<T> a, {NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cholesky() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write cholesky result to a disposed output array.',
    );
  }
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 2] != a.shape[rank - 1]) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must be square in the last 2 dimensions and rank >= 2 (was ${a.shape})',
    );
  }
  if (!a.dtype.isFloating && !a.dtype.isComplex) {
    throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      'Must be float or complex (Cholesky decomposition is only supported for float and complex dtypes, was ${a.dtype})',
    );
  }
  checkBlasIntDim(a.shape[rank - 1], 'n', 'cholesky');
  checkBlasIntStride(a.strides[rank - 2], 'lda', 'cholesky');
  checkBlasIntStride(a.strides[rank - 1], 'lda', 'cholesky');
  if (a.dtype == DType.float16 || a.dtype == DType.bfloat16) {
    if (out != null) {
      validateOutBuffer(out);
      if (!listEquals(out.shape, a.shape) ||
          (out.dtype != DType.float64 && out.dtype != a.dtype)) {
        throw ArgumentError.value(
          out,
          'out',
          'Must have compatible shape ${a.shape} and float64 or ${a.dtype} dtype (provided out L buffer has incompatible shape or dtype).',
        );
      }
    }
    return NDArray.scope(() {
      final aF64 = castNDArray<Float64>(a, DType.float64);
      final resF64 = cholesky<Float64>(aF64);
      if (out != null) {
        if (out.dtype == DType.float64) {
          resF64.copy(out: out as NDArray<Float64>);
        } else {
          castNDArray<T>(resF64, out.dtype).copy(out: out);
        }
        return out;
      }
      return castNDArray<T>(resF64, a.dtype).detachToParentScope();
    });
  }
  final n = a.shape[rank - 1];
  final stackShape = a.shape.sublist(0, rank - 2);
  final targetDType = a.dtype;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape ${a.shape} and dtype ${a.dtype} (provided out L buffer has incompatible shape or dtype).',
      );
    }
    if (!out.isContiguous || sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = cholesky<T>(a);
        temp.copy(out: out);
        return out;
      });
    }
  }

  return NDArray.scope(() {
    final NDArray<T> lMat;
    if (out != null) {
      lMat = out;
      a.copy(out: lMat);
    } else {
      lMat = a.copy();
    }

    if (n == 0) {
      if (out == null) {
        lMat.detachToParentScope();
      }
      return lMat;
    }

    // Char 'L' in ASCII is 76
    const uploL = 76;

    walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
      coords,
    ) {
      var offsetL = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetL += coords[i] * lMat.strides[i];
      }
      final lSlice = NDArray<T>.view(
        lMat,
        shape: [n, n],
        strides: lMat.strides.sublist(rank - 2),
        offsetElements: offsetL,
      );

      try {
        final nf = _analyzeNonFinitePtr(lSlice.pointer, n * n, targetDType);
        if (nf.hasNaN) {
          throw const NonPositiveDefiniteException(
            'Matrix is not positive definite (contains NaN).',
          );
        }

        final int info;
        final String routine;
        switch (targetDType) {
          case DType.float64:
            routine = 'LAPACKE_dpotrf';
            info = LAPACKE_dpotrf(
              101, // ROW_MAJOR
              uploL,
              n,
              lSlice.pointer.cast<ffi.Double>(),
              n,
            );
          case DType.float32:
            routine = 'LAPACKE_spotrf';
            info = LAPACKE_spotrf(
              101, // ROW_MAJOR
              uploL,
              n,
              lSlice.pointer.cast<ffi.Float>(),
              n,
            );
          case DType.complex128:
            routine = 'LAPACKE_zpotrf';
            info = LAPACKE_zpotrf(
              101, // ROW_MAJOR
              uploL,
              n,
              lSlice.pointer.cast<ffi.Double>(),
              n,
            );
          case DType.complex64:
            routine = 'LAPACKE_cpotrf';
            info = LAPACKE_cpotrf(
              101, // ROW_MAJOR
              uploL,
              n,
              lSlice.pointer.cast<ffi.Float>(),
              n,
            );
          default:
            throw UnimplementedError(
              'Unsupported dtype for Cholesky: $targetDType',
            );
        }

        _checkLapackInfo(
          info,
          routine,
          positiveKind: _LapackFailureKind.nonPositiveDefinite,
          positiveMessage:
              'Matrix must be positive-definite for Cholesky decomposition: the leading minor of order $info is not positive definite.',
        );

        final nfAfter = _analyzeNonFinitePtr(
          lSlice.pointer,
          n * n,
          targetDType,
        );
        if (nfAfter.hasNaN) {
          throw const NonPositiveDefiniteException(
            'Matrix is not positive definite.',
          );
        }

        v_zero_upper_triangular(
          lSlice.pointer.cast<ffi.Void>(),
          n,
          encodeDType(targetDType),
        );
      } finally {
        lSlice.dispose();
      }
    });

    if (out == null) {
      lMat.detachToParentScope();
    }
    return lMat;
  });
}

/// Computes the QR decomposition of a matrix or a stack of matrices $A = Q R$.
///
/// Decomposes a matrix [a] out an orthogonal matrix `Q` and an upper triangular matrix `R`
/// such that `a = Q * R`.
/// Uses LAPACK solvers (`dgeqrf` / `sgeqrf` and `dorgqr` / `sorgqr`) depending on precision.
///
/// **Preconditions:**
/// - Input matrix [a] must be at least 2-dimensional.
///
/// **Throws:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] rank is less than 2.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// **Example:**
/// {@example /example/linalg_example.dart lang=dart}
({NDArray<T> q, NDArray<T> r}) qr<T extends DTypeTag>(
  NDArray<T> a, {
  ({NDArray<T> q, NDArray<T> r})? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute qr() on a disposed array.');
  }
  if (out != null && (out.q.isDisposed || out.r.isDisposed)) {
    throw StateError('Cannot write qr result to a disposed output array.');
  }
  final rank = a.shape.length;
  if (rank < 2) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must be at least 2D (matrix must be at least 2D, was ${a.shape})',
    );
  }
  if (!a.dtype.isFloating && !a.dtype.isComplex) {
    throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      'Must be float or complex (QR decomposition is only supported for float and complex dtypes, was ${a.dtype})',
    );
  }
  final m = a.shape[rank - 2];
  final n = a.shape[rank - 1];
  checkBlasIntDim(m, 'm', 'qr');
  checkBlasIntDim(n, 'n', 'qr');
  checkBlasIntStride(a.strides[rank - 2], 'lda', 'qr');
  checkBlasIntStride(a.strides[rank - 1], 'lda', 'qr');
  final k = m < n ? m : n;
  final stackShape = a.shape.sublist(0, rank - 2);

  final qShape = [...stackShape, m, k];
  final rShape = [...stackShape, k, n];

  if (out != null && sharesMemory(out.q, out.r)) {
    throw ArgumentError.value(
      out.r,
      'out.r',
      'Must not share memory with out.q.',
    );
  }

  if (a.dtype == DType.float16 || a.dtype == DType.bfloat16) {
    if (out != null) {
      validateOutBuffer(out.q, 'out.q');
      validateOutBuffer(out.r, 'out.r');
      if (!listEquals(out.q.shape, qShape) ||
          (out.q.dtype != DType.float64 && out.q.dtype != a.dtype)) {
        throw ArgumentError.value(
          out.q,
          'out.q',
          'Must have compatible shape $qShape and float64 or ${a.dtype} dtype (provided out Q buffer has incompatible shape or dtype).',
        );
      }
      if (!listEquals(out.r.shape, rShape) ||
          (out.r.dtype != DType.float64 && out.r.dtype != a.dtype)) {
        throw ArgumentError.value(
          out.r,
          'out.r',
          'Must have compatible shape $rShape and float64 or ${a.dtype} dtype (provided out R buffer has incompatible shape or dtype).',
        );
      }
    }
    return NDArray.scope(() {
      final aF64 = castNDArray<Float64>(a, DType.float64);
      final resF64 = qr<Float64>(aF64);
      if (out != null) {
        if (out.q.dtype == DType.float64) {
          resF64.q.copy(out: out.q as NDArray<Float64>);
        } else {
          castNDArray<T>(resF64.q, out.q.dtype).copy(out: out.q);
        }
        if (out.r.dtype == DType.float64) {
          resF64.r.copy(out: out.r as NDArray<Float64>);
        } else {
          castNDArray<T>(resF64.r, out.r.dtype).copy(out: out.r);
        }
        return (q: out.q, r: out.r);
      }
      return (
        q: castNDArray<T>(resF64.q, a.dtype).detachToParentScope(),
        r: castNDArray<T>(resF64.r, a.dtype).detachToParentScope(),
      );
    });
  }

  final DType<T> targetDType = a.dtype;

  return NDArray.scope(() {
    final NDArray<T> qMat;
    final NDArray<T> rMat;
    if (out != null) {
      qMat = out.q;
      rMat = out.r;
      validateOutBuffer(qMat, 'out.q');
      validateOutBuffer(rMat, 'out.r');
      if (!listEquals(qMat.shape, qShape) || qMat.dtype != targetDType) {
        throw ArgumentError.value(
          qMat,
          'out.q',
          'Must have compatible shape $qShape and dtype $targetDType (provided out Q buffer has incompatible shape or dtype).',
        );
      }
      if (!listEquals(rMat.shape, rShape) || rMat.dtype != targetDType) {
        throw ArgumentError.value(
          rMat,
          'out.r',
          'Must have compatible shape $rShape and dtype $targetDType (provided out R buffer has incompatible shape or dtype).',
        );
      }
      if (!qMat.isContiguous ||
          !rMat.isContiguous ||
          sharesMemory(a, qMat) ||
          sharesMemory(a, rMat)) {
        final temp = qr<T>(a);
        temp.q.copy(out: qMat);
        temp.r.copy(out: rMat);
        return (q: qMat, r: rMat);
      }
    } else {
      qMat = NDArray<T>.zeros(qShape, targetDType);
      rMat = NDArray<T>.zeros(rShape, targetDType);
    }

    if (m == 0 || n == 0) {
      if (out == null) {
        qMat.detachToParentScope();
        rMat.detachToParentScope();
      }
      return (q: qMat, r: rMat);
    }

    final aCopy = NDArray.create([m, n], targetDType);
    final marker = ScratchArena.marker;
    try {
      final ffi.Pointer<ffi.Void> tau;
      switch (targetDType) {
        case DType.float64:
          tau = ScratchArena.allocate<ffi.Double>(
            k * ffi.sizeOf<ffi.Double>(),
          ).cast<ffi.Void>();
        case DType.float32:
          tau = ScratchArena.allocate<ffi.Float>(
            k * ffi.sizeOf<ffi.Float>(),
          ).cast<ffi.Void>();
        case DType.complex128:
          tau = ScratchArena.allocate<ffi.Double>(
            2 * k * ffi.sizeOf<ffi.Double>(),
          ).cast<ffi.Void>();
        case DType.complex64:
          tau = ScratchArena.allocate<ffi.Float>(
            2 * k * ffi.sizeOf<ffi.Float>(),
          ).cast<ffi.Void>();
        default:
          throw UnimplementedError('Unsupported DType for QR: $targetDType');
      }

      walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
        coords,
      ) {
        var offsetA = 0;
        for (var i = 0; i < coords.length; i++) {
          offsetA += coords[i] * a.strides[i];
        }

        final sliceView = NDArray.view(
          a,
          shape: [m, n],
          strides: a.strides.sublist(rank - 2),
          offsetElements: offsetA,
        );
        sliceView.copy(out: aCopy);
        sliceView.dispose();

        final NDArray r2D = NDArray.zeros([k, n], targetDType);
        final NDArray q2D = NDArray.zeros([m, k], targetDType);

        final nf = _analyzeNonFinitePtr(aCopy.pointer, m * n, targetDType);
        if (nf.hasNaN || nf.hasInf) {
          _fillPtrWithNaN(q2D.pointer, m * k, targetDType);
          _fillUpperTriangleWithNaN(r2D.pointer, k, n, targetDType);
        } else {
          switch (targetDType) {
            case DType.float64:
              final info = LAPACKE_dgeqrf(
                101, // ROW_MAJOR
                m,
                n,
                aCopy.pointer.cast<ffi.Double>(),
                n,
                tau.cast<ffi.Double>(),
              );
              _checkLapackInfo(info, 'LAPACKE_dgeqrf');
              final rPtr = r2D.pointer.cast<ffi.Double>();
              final aPtr = aCopy.pointer.cast<ffi.Double>();
              for (var i = 0; i < k; i++) {
                for (var j = i; j < n; j++) {
                  rPtr[i * n + j] = aPtr[i * n + j];
                }
              }
              final qPtr = q2D.pointer.cast<ffi.Double>();
              for (var i = 0; i < m; i++) {
                for (var j = 0; j < k; j++) {
                  qPtr[i * k + j] = aPtr[i * n + j];
                }
              }
              final infoOrg = LAPACKE_dorgqr(
                101, // ROW_MAJOR
                m,
                k,
                k,
                q2D.pointer.cast<ffi.Double>(),
                k,
                tau.cast<ffi.Double>(),
              );
              _checkLapackInfo(infoOrg, 'LAPACKE_dorgqr');
            case DType.float32:
              final info = LAPACKE_sgeqrf(
                101, // ROW_MAJOR
                m,
                n,
                aCopy.pointer.cast<ffi.Float>(),
                n,
                tau.cast<ffi.Float>(),
              );
              _checkLapackInfo(info, 'LAPACKE_sgeqrf');
              final rPtr = r2D.pointer.cast<ffi.Float>();
              final aPtr = aCopy.pointer.cast<ffi.Float>();
              for (var i = 0; i < k; i++) {
                for (var j = i; j < n; j++) {
                  rPtr[i * n + j] = aPtr[i * n + j];
                }
              }
              final qPtr = q2D.pointer.cast<ffi.Float>();
              for (var i = 0; i < m; i++) {
                for (var j = 0; j < k; j++) {
                  qPtr[i * k + j] = aPtr[i * n + j];
                }
              }
              final infoOrg = LAPACKE_sorgqr(
                101, // ROW_MAJOR
                m,
                k,
                k,
                q2D.pointer.cast<ffi.Float>(),
                k,
                tau.cast<ffi.Float>(),
              );
              _checkLapackInfo(infoOrg, 'LAPACKE_sorgqr');
            case DType.complex128:
              final info = LAPACKE_zgeqrf(
                101, // ROW_MAJOR
                m,
                n,
                aCopy.pointer.cast<ffi.Double>(),
                n,
                tau.cast<ffi.Double>(),
              );
              _checkLapackInfo(info, 'LAPACKE_zgeqrf');
              final rPtr = r2D.pointer.cast<ffi.Double>();
              final aPtr = aCopy.pointer.cast<ffi.Double>();
              for (var i = 0; i < k; i++) {
                for (var j = i; j < n; j++) {
                  rPtr[(i * n + j) * 2] = aPtr[(i * n + j) * 2];
                  rPtr[(i * n + j) * 2 + 1] = aPtr[(i * n + j) * 2 + 1];
                }
              }
              final qPtr = q2D.pointer.cast<ffi.Double>();
              for (var i = 0; i < m; i++) {
                for (var j = 0; j < k; j++) {
                  qPtr[(i * k + j) * 2] = aPtr[(i * n + j) * 2];
                  qPtr[(i * k + j) * 2 + 1] = aPtr[(i * n + j) * 2 + 1];
                }
              }
              final infoOrg = LAPACKE_zungqr(
                101, // ROW_MAJOR
                m,
                k,
                k,
                q2D.pointer.cast<ffi.Double>(),
                k,
                tau.cast<ffi.Double>(),
              );
              _checkLapackInfo(infoOrg, 'LAPACKE_zungqr');
            case DType.complex64:
              final info = LAPACKE_cgeqrf(
                101, // ROW_MAJOR
                m,
                n,
                aCopy.pointer.cast<ffi.Float>(),
                n,
                tau.cast<ffi.Float>(),
              );
              _checkLapackInfo(info, 'LAPACKE_cgeqrf');
              final rPtr = r2D.pointer.cast<ffi.Float>();
              final aPtr = aCopy.pointer.cast<ffi.Float>();
              for (var i = 0; i < k; i++) {
                for (var j = i; j < n; j++) {
                  rPtr[(i * n + j) * 2] = aPtr[(i * n + j) * 2];
                  rPtr[(i * n + j) * 2 + 1] = aPtr[(i * n + j) * 2 + 1];
                }
              }
              final qPtr = q2D.pointer.cast<ffi.Float>();
              for (var i = 0; i < m; i++) {
                for (var j = 0; j < k; j++) {
                  qPtr[(i * k + j) * 2] = aPtr[(i * n + j) * 2];
                  qPtr[(i * k + j) * 2 + 1] = aPtr[(i * n + j) * 2 + 1];
                }
              }
              final infoOrg = LAPACKE_cungqr(
                101, // ROW_MAJOR
                m,
                k,
                k,
                q2D.pointer.cast<ffi.Float>(),
                k,
                tau.cast<ffi.Float>(),
              );
              _checkLapackInfo(infoOrg, 'LAPACKE_cungqr');
            default:
              break;
          }
        }

        var offsetQ = 0;
        for (var i = 0; i < coords.length; i++) {
          offsetQ += coords[i] * qMat.strides[i];
        }
        var offsetR = 0;
        for (var i = 0; i < coords.length; i++) {
          offsetR += coords[i] * rMat.strides[i];
        }

        final qSlice = NDArray<T>.view(
          qMat,
          shape: [m, k],
          strides: qMat.strides.sublist(rank - 2),
          offsetElements: offsetQ,
        );
        q2D.copy(out: qSlice);
        qSlice.dispose();

        final rSlice = NDArray<T>.view(
          rMat,
          shape: [k, n],
          strides: rMat.strides.sublist(rank - 2),
          offsetElements: offsetR,
        );
        r2D.copy(out: rSlice);
        rSlice.dispose();

        q2D.dispose();
        r2D.dispose();
      });
    } finally {
      ScratchArena.reset(marker);
      aCopy.dispose();
    }

    if (out == null) {
      qMat.detachToParentScope();
      rMat.detachToParentScope();
    }
    return (q: qMat, r: rMat);
  });
}

/// Computes the Singular Value Decomposition (SVD) of a matrix or a stack of matrices $A = U S V^h$.
///
/// Decomposes a matrix [a] out left singular vectors `U`, singular values `S`,
/// and right singular vectors Vh such that `a = U * diag(S) * Vh`.
/// Uses LAPACK solvers (`dgesdd` / `sgesdd`) depending on precision.
///
/// **Preconditions:**
/// - Input matrix [a] must be at least 2-dimensional.
///
/// **Throws:**
/// - It is an error if [a] or any buffer in [out] is disposed.
/// - It is an error if [a] rank is less than 2.
/// - It is an error if [a] has an unsupported dtype (e.g. integer or boolean).
/// - It is an error if any buffer in [out] has incompatible shape or dtype.
/// - Throws an [IterationsExceededException] if the SVD computation does not converge.
/// - Throws a [LinAlgException] if [a] contains non-finite values (`NaN` or `±Infinity`).
///
/// **Example:**
/// {@example /example/linalg_example.dart lang=dart}
({NDArray<T> u, NDArray<R> s, NDArray<T> vh})
svd<T extends DTypeTag, R extends DTypeTag>(
  NDArray<
    DTypeSpec<
      R,
      Object?,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      T,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag
    >
  >
  a, {
  ({NDArray<T> u, NDArray<R> s, NDArray<T> vh})? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute svd() on a disposed array.');
  }
  if (out != null) {
    if (out.u.isDisposed || out.s.isDisposed || out.vh.isDisposed) {
      throw StateError('Cannot write SVD result to a disposed output array.');
    }
  }
  if (!a.dtype.isFloating && !a.dtype.isComplex) {
    throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      'Must be float or complex (SVD decomposition is only supported for float and complex dtypes, was ${a.dtype})',
    );
  }
  final rank = a.shape.length;
  if (rank < 2) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must be at least 2D (matrix must be at least 2D, was ${a.shape})',
    );
  }
  final m = a.shape[rank - 2];
  final n = a.shape[rank - 1];
  checkBlasIntDim(m, 'm', 'svd');
  checkBlasIntDim(n, 'n', 'svd');
  checkBlasIntStride(a.strides[rank - 2], 'lda', 'svd');
  checkBlasIntStride(a.strides[rank - 1], 'lda', 'svd');
  final stackShape = a.shape.sublist(0, rank - 2);

  final uShape = [...stackShape, m, m];
  final sShape = m < n ? [...stackShape, m] : [...stackShape, n];
  final vtShape = [...stackShape, n, n];

  if (out != null &&
      (sharesMemory(out.u, out.s) ||
          sharesMemory(out.u, out.vh) ||
          sharesMemory(out.s, out.vh))) {
    throw ArgumentError.value(
      out,
      'out',
      'Must not have overlapping output buffers (u, s, vh must not share memory).',
    );
  }

  if ((a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16) {
    if (out != null) {
      validateOutBuffer(out.u, 'out.u');
      validateOutBuffer(out.s, 'out.s');
      validateOutBuffer(out.vh, 'out.vh');
      if (!listEquals(out.u.shape, uShape) ||
          (out.u.dtype != DType.float64 && out.u.dtype != a.dtype)) {
        throw ArgumentError.value(
          out.u,
          'out.u',
          'Must have compatible shape $uShape and float64 or ${a.dtype} dtype (provided out U buffer has incompatible shape or dtype).',
        );
      }
      if (!listEquals(out.s.shape, sShape) ||
          (out.s.dtype != DType.float64 && out.s.dtype != a.dtype)) {
        throw ArgumentError.value(
          out.s,
          'out.s',
          'Must have compatible shape $sShape and float64 or ${a.dtype} dtype (provided out S buffer has incompatible shape or dtype).',
        );
      }
      if (!listEquals(out.vh.shape, vtShape) ||
          (out.vh.dtype != DType.float64 && out.vh.dtype != a.dtype)) {
        throw ArgumentError.value(
          out.vh,
          'out.vh',
          'Must have compatible shape $vtShape and float64 or ${a.dtype} dtype (provided out Vh buffer has incompatible shape or dtype).',
        );
      }
    }
    return NDArray.scope(() {
      final aF64 = castNDArray<Float64>(a, DType.float64);
      final resF64 = _svd<Float64>(aF64);
      if (out != null) {
        if (out.u.dtype == DType.float64) {
          resF64.u.copy(out: out.u as NDArray<Float64>);
        } else {
          castNDArray<T>(resF64.u, out.u.dtype).copy(out: out.u);
        }
        if (out.s.dtype == DType.float64) {
          resF64.s.copy(out: out.s);
        } else {
          castNDArray<R>(resF64.s, out.s.dtype).copy(out: out.s);
        }
        if (out.vh.dtype == DType.float64) {
          resF64.vh.copy(out: out.vh as NDArray<Float64>);
        } else {
          castNDArray<T>(resF64.vh, out.vh.dtype).copy(out: out.vh);
        }
        return (u: out.u, s: out.s, vh: out.vh);
      }
      return (
        u: castNDArray<T>(resF64.u, a.dtype as DType<T>).detachToParentScope(),
        s: castNDArray<R>(resF64.s, a.dtype as DType<R>).detachToParentScope(),
        vh: castNDArray<T>(
          resF64.vh,
          a.dtype as DType<T>,
        ).detachToParentScope(),
      );
    });
  }

  final dtypeS = a.dtype.isComplex
      ? ((a.dtype as DType<DTypeTag>) == DType.complex128
            ? DType.float64
            : DType.float32)
      : a.dtype;

  if (out != null) {
    validateOutBuffer(out.u, 'out.u');
    validateOutBuffer(out.s, 'out.s');
    validateOutBuffer(out.vh, 'out.vh');
    if (!listEquals(out.u.shape, uShape) || out.u.dtype != a.dtype) {
      throw ArgumentError.value(
        out.u,
        'out.u',
        'Must have compatible shape $uShape and dtype ${a.dtype} (provided out U buffer has incompatible shape or dtype).',
      );
    }
    if (!listEquals(out.s.shape, sShape) || out.s.dtype != dtypeS) {
      throw ArgumentError.value(
        out.s,
        'out.s',
        'Must have compatible shape $sShape and dtype $dtypeS (provided out S buffer has incompatible shape or dtype).',
      );
    }
    if (!listEquals(out.vh.shape, vtShape) || out.vh.dtype != a.dtype) {
      throw ArgumentError.value(
        out.vh,
        'out.vh',
        'Must have compatible shape $vtShape and dtype ${a.dtype} (provided out Vh buffer has incompatible shape or dtype).',
      );
    }
    if (!out.u.isContiguous ||
        !out.s.isContiguous ||
        !out.vh.isContiguous ||
        sharesMemory(a, out.u) ||
        sharesMemory(a, out.s) ||
        sharesMemory(a, out.vh)) {
      return NDArray.scope(() {
        final temp = _svd<T>(a as NDArray<T>);
        temp.u.copy(out: out.u);
        temp.s.copy(out: out.s);
        temp.vh.copy(out: out.vh);
        return (u: out.u, s: out.s, vh: out.vh);
      });
    }
  }

  final res = _svd<T>(
    a as NDArray<T>,
    out: out == null ? null : (u: out.u, s: out.s, vh: out.vh),
  );
  return (u: res.u, s: res.s as NDArray<R>, vh: res.vh);
}

({NDArray<T> u, NDArray<DTypeTag> s, NDArray<T> vh}) _svd<T extends DTypeTag>(
  NDArray<T> a, {
  ({NDArray<T> u, NDArray<DTypeTag> s, NDArray<T> vh})? out,
}) {
  final rank = a.shape.length;
  final m = a.shape[rank - 2];
  final n = a.shape[rank - 1];
  checkBlasIntDim(m, 'm', 'svd');
  checkBlasIntDim(n, 'n', 'svd');
  checkBlasIntStride(a.strides[rank - 2], 'lda', 'svd');
  checkBlasIntStride(a.strides[rank - 1], 'lda', 'svd');
  final stackShape = a.shape.sublist(0, rank - 2);

  return NDArray.scope(() {
    if (m == 0 || n == 0) {
      final dtypeS = a.dtype.isComplex
          ? (a.dtype == DType.complex128 ? DType.float64 : DType.float32)
          : a.dtype;
      final uShape = [...stackShape, m, m];
      final sShape = [...stackShape, 0];
      final vtShape = [...stackShape, n, n];

      final uMat = out?.u ?? NDArray<T>.zeros(uShape, a.dtype);
      final sMat =
          out?.s ?? NDArray<Float64>.zeros(sShape, dtypeS as DType<Float64>);
      final vhMat = out?.vh ?? NDArray<T>.zeros(vtShape, a.dtype);

      final oneTyped = castValue(1.0, a.dtype);
      if (m > 0) {
        walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
          coords,
        ) {
          var offsetU = 0;
          for (var i = 0; i < coords.length; i++) {
            offsetU += coords[i] * uMat.strides[i];
          }
          final uSlice = NDArray<T>.view(
            uMat,
            shape: [m, m],
            strides: uMat.strides.sublist(rank - 2),
            offsetElements: offsetU,
          );
          for (var i = 0; i < m; i++) {
            uSlice.setCell([i, i], oneTyped);
          }
          uSlice.dispose();
        });
      }
      if (n > 0) {
        walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
          coords,
        ) {
          var offsetVt = 0;
          for (var i = 0; i < coords.length; i++) {
            offsetVt += coords[i] * vhMat.strides[i];
          }
          final vtSlice = NDArray<T>.view(
            vhMat,
            shape: [n, n],
            strides: vhMat.strides.sublist(rank - 2),
            offsetElements: offsetVt,
          );
          for (var i = 0; i < n; i++) {
            vtSlice.setCell([i, i], oneTyped);
          }
          vtSlice.dispose();
        });
      }

      if (out == null) {
        uMat.detachToParentScope();
        sMat.detachToParentScope();
        vhMat.detachToParentScope();
      }
      return (u: uMat, s: sMat, vh: vhMat);
    }

    if (m < n) {
      // Compute SVD of A^H (or A^T for real)
      final axes = List<int>.generate(rank, (i) => i);
      axes[rank - 2] = rank - 1;
      axes[rank - 1] = rank - 2;
      final aT = a.dtype.isComplex
          ? conjugate(a.transpose(axes))
          : a.transpose(axes).copy();
      try {
        final resT = _svd<T>(aT);
        final uNew = a.dtype.isComplex
            ? conjugate(resT.vh.transpose(axes))
            : resT.vh.transpose(axes);
        final sNew = resT.s;
        final vhNew = a.dtype.isComplex
            ? conjugate(resT.u.transpose(axes))
            : resT.u.transpose(axes);

        final uResult = out?.u ?? uNew;
        final sResult = out?.s ?? sNew;
        final vhResult = out?.vh ?? vhNew;

        if (out != null) {
          uNew.copy(out: out.u);
          sNew.copy(out: out.s);
          vhNew.copy(out: out.vh);
          uNew.dispose();
          sNew.dispose();
          vhNew.dispose();
          return (u: uResult, s: sResult, vh: vhResult);
        } else {
          final uCopy = uResult.copy();
          final vhCopy = vhResult.copy();
          uNew.dispose();
          vhNew.dispose();
          uResult.dispose();
          vhResult.dispose();
          uCopy.detachToParentScope();
          sNew.detachToParentScope();
          vhCopy.detachToParentScope();
          return (u: uCopy, s: sNew, vh: vhCopy);
        }
      } finally {
        aT.dispose();
      }
    }

    final dtypeS = a.dtype.isComplex
        ? (a.dtype == DType.complex128 ? DType.float64 : DType.float32)
        : a.dtype;

    final uShape = [...stackShape, m, m];
    final sShape = [...stackShape, n];
    final vtShape = [...stackShape, n, n];

    final NDArray<T> uMat = out?.u ?? NDArray<T>.zeros(uShape, a.dtype);
    final NDArray<DTypeTag> sMat =
        out?.s ?? NDArray<DTypeTag>.zeros(sShape, dtypeS);
    final NDArray<T> vtMat = out?.vh ?? NDArray<T>.zeros(vtShape, a.dtype);

    final aCopy = NDArray<T>.create([m, n], a.dtype);
    final marker = ScratchArena.marker;
    try {
      final superbLen = math.max(1, n - 1);
      final ffi.Pointer<ffi.Void> superb = switch (a.dtype) {
        DType.float64 || DType.complex128 => ScratchArena.allocate<ffi.Double>(
          superbLen * ffi.sizeOf<ffi.Double>(),
        ).cast<ffi.Void>(),
        _ => ScratchArena.allocate<ffi.Float>(
          superbLen * ffi.sizeOf<ffi.Float>(),
        ).cast<ffi.Void>(),
      };

      walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
        coords,
      ) {
        var offsetA = 0;
        for (var i = 0; i < coords.length; i++) {
          offsetA += coords[i] * a.strides[i];
        }

        final sliceView = NDArray.view(
          a,
          shape: [m, n],
          strides: a.strides.sublist(rank - 2),
          offsetElements: offsetA,
        );
        sliceView.copy(out: aCopy);
        sliceView.dispose();

        final NDArray<DTypeTag> s2D =
            (a.dtype == DType.float32 || a.dtype == DType.complex64)
            ? NDArray<Float32>.zeros([n], DType.float32)
            : NDArray<Float64>.zeros([n], DType.float64);
        final NDArray u2D = NDArray.zeros([m, m], a.dtype);
        final NDArray vt2D = NDArray.zeros([n, n], a.dtype);

        final nf = _analyzeNonFinitePtr(aCopy.pointer, m * n, a.dtype);
        if (nf.hasNaN || nf.hasInf) {
          s2D.dispose();
          u2D.dispose();
          vt2D.dispose();
          throw const IterationsExceededException(
            'SVD did not converge (input contains non-finite values).',
          );
        }

        final int info;
        final String routine;
        switch (a.dtype) {
          case DType.float64:
            routine = 'LAPACKE_dgesvd';
            info = LAPACKE_dgesvd(
              101,
              65,
              65,
              m,
              n,
              aCopy.pointer.cast<ffi.Double>(),
              n,
              s2D.pointer.cast<ffi.Double>(),
              u2D.pointer.cast<ffi.Double>(),
              m,
              vt2D.pointer.cast<ffi.Double>(),
              n,
              superb.cast<ffi.Double>(),
            );

          case DType.float32:
            routine = 'LAPACKE_sgesvd';
            info = LAPACKE_sgesvd(
              101,
              65,
              65,
              m,
              n,
              aCopy.pointer.cast<ffi.Float>(),
              n,
              s2D.pointer.cast<ffi.Float>(),
              u2D.pointer.cast<ffi.Float>(),
              m,
              vt2D.pointer.cast<ffi.Float>(),
              n,
              superb.cast<ffi.Float>(),
            );

          case DType.complex128:
            routine = 'LAPACKE_zgesvd';
            info = LAPACKE_zgesvd(
              101,
              65,
              65,
              m,
              n,
              aCopy.pointer.cast<ffi.Double>(),
              n,
              s2D.pointer.cast<ffi.Double>(),
              u2D.pointer.cast<ffi.Double>(),
              m,
              vt2D.pointer.cast<ffi.Double>(),
              n,
              superb.cast<ffi.Double>(),
            );

          case DType.complex64:
            routine = 'LAPACKE_cgesvd';
            info = LAPACKE_cgesvd(
              101,
              65,
              65,
              m,
              n,
              aCopy.pointer.cast<ffi.Float>(),
              n,
              s2D.pointer.cast<ffi.Float>(),
              u2D.pointer.cast<ffi.Float>(),
              m,
              vt2D.pointer.cast<ffi.Float>(),
              n,
              superb.cast<ffi.Float>(),
            );
          default:
            s2D.dispose();
            u2D.dispose();
            vt2D.dispose();
            throw ArgumentError.value(
              a.dtype,
              'a.dtype',
              'Must be float or complex (unsupported dtype for SVD: ${a.dtype})',
            );
        }
        if (info != 0) {
          s2D.dispose();
          u2D.dispose();
          vt2D.dispose();
          _checkLapackInfo(
            info,
            routine,
            positiveKind: _LapackFailureKind.iterationsExceeded,
            positiveMessage: '$routine failed to converge (info = $info).',
          );
        }

        var offsetU = 0;
        for (var i = 0; i < coords.length; i++) {
          offsetU += coords[i] * uMat.strides[i];
        }
        var offsetS = 0;
        for (var i = 0; i < coords.length; i++) {
          offsetS += coords[i] * sMat.strides[i];
        }
        var offsetVt = 0;
        for (var i = 0; i < coords.length; i++) {
          offsetVt += coords[i] * vtMat.strides[i];
        }

        final uSlice = NDArray<T>.view(
          uMat,
          shape: [m, m],
          strides: uMat.strides.sublist(rank - 2),
          offsetElements: offsetU,
        );
        u2D.copy(out: uSlice);
        uSlice.dispose();

        final sSlice = NDArray<DTypeTag>.view(
          sMat,
          shape: [n],
          strides: sMat.strides.isEmpty ? [1] : [sMat.strides.last],
          offsetElements: offsetS,
        );
        s2D.copy(out: sSlice);
        sSlice.dispose();

        final vtSlice = NDArray<T>.view(
          vtMat,
          shape: [n, n],
          strides: vtMat.strides.sublist(rank - 2),
          offsetElements: offsetVt,
        );
        vt2D.copy(out: vtSlice);
        vtSlice.dispose();

        s2D.dispose();
        u2D.dispose();
        vt2D.dispose();
      });
    } finally {
      ScratchArena.reset(marker);
      aCopy.dispose();
    }

    if (out == null) {
      uMat.detachToParentScope();
      sMat.detachToParentScope();
      vtMat.detachToParentScope();
    }
    return (u: uMat, s: sMat, vh: vtMat);
  });
}

NDArray<DTypeTag> _svdVals<T extends DTypeTag>(NDArray<T> a) {
  final rank = a.shape.length;
  final m = a.shape[rank - 2];
  final n = a.shape[rank - 1];
  checkBlasIntDim(m, 'm', 'svdvals');
  checkBlasIntDim(n, 'n', 'svdvals');
  checkBlasIntStride(a.strides[rank - 2], 'lda', 'svdvals');
  checkBlasIntStride(a.strides[rank - 1], 'lda', 'svdvals');

  if (a.dtype == DType.float16 || a.dtype == DType.bfloat16) {
    return NDArray.scope(() {
      final aF64 = castNDArray<Float64>(a, DType.float64);
      final sF64 = _svdVals<Float64>(aF64);
      return sF64.detachToParentScope();
    });
  }

  final stackShape = a.shape.sublist(0, rank - 2);

  return NDArray.scope(() {
    if (m == 0 || n == 0) {
      final dtypeS = a.dtype.isComplex
          ? (a.dtype == DType.complex128 ? DType.float64 : DType.float32)
          : a.dtype;
      final sShape = [...stackShape, 0];
      final sMat = NDArray<DTypeTag>.zeros(sShape, dtypeS);
      sMat.detachToParentScope();
      return sMat;
    }

    if (m < n) {
      final axes = List<int>.generate(rank, (i) => i);
      axes[rank - 2] = rank - 1;
      axes[rank - 1] = rank - 2;
      final aT = a.dtype.isComplex
          ? conjugate(a.transpose(axes))
          : a.transpose(axes).copy();
      try {
        final resT = _svdVals<T>(aT);
        resT.detachToParentScope();
        return resT;
      } finally {
        aT.dispose();
      }
    }

    final dtypeS = a.dtype.isComplex
        ? (a.dtype == DType.complex128 ? DType.float64 : DType.float32)
        : a.dtype;

    final sShape = [...stackShape, n];
    final NDArray<DTypeTag> sMat = NDArray<DTypeTag>.zeros(sShape, dtypeS);

    final aCopy = NDArray<T>.create([m, n], a.dtype);
    final marker = ScratchArena.marker;
    try {
      final superbLen = math.max(1, n - 1);
      final ffi.Pointer<ffi.Void> superb = switch (a.dtype) {
        DType.float64 || DType.complex128 => ScratchArena.allocate<ffi.Double>(
          superbLen * ffi.sizeOf<ffi.Double>(),
        ).cast<ffi.Void>(),
        _ => ScratchArena.allocate<ffi.Float>(
          superbLen * ffi.sizeOf<ffi.Float>(),
        ).cast<ffi.Void>(),
      };

      walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
        coords,
      ) {
        var offsetA = 0;
        for (var i = 0; i < coords.length; i++) {
          offsetA += coords[i] * a.strides[i];
        }

        final sliceView = NDArray.view(
          a,
          shape: [m, n],
          strides: a.strides.sublist(rank - 2),
          offsetElements: offsetA,
        );
        sliceView.copy(out: aCopy);
        sliceView.dispose();

        final NDArray<DTypeTag> s2D =
            (a.dtype == DType.float32 || a.dtype == DType.complex64)
            ? NDArray<Float32>.zeros([n], DType.float32)
            : NDArray<DTypeTag>.zeros([n], DType.float64);

        final nf = _analyzeNonFinitePtr(aCopy.pointer, m * n, a.dtype);
        if (nf.hasNaN) {
          s2D.dispose();
          throw const IterationsExceededException(
            'SVD did not converge (input contains NaN).',
          );
        }
        if (nf.hasInf) {
          _fillPtrWithNaN(s2D.pointer, n, s2D.dtype);
        } else {
          final int info;
          final String routine;
          switch (a.dtype) {
            case DType.float64:
              routine = 'LAPACKE_dgesvd';
              info = LAPACKE_dgesvd(
                101,
                78,
                78,
                m,
                n,
                aCopy.pointer.cast<ffi.Double>(),
                n,
                s2D.pointer.cast<ffi.Double>(),
                ffi.nullptr,
                1,
                ffi.nullptr,
                1,
                superb.cast<ffi.Double>(),
              );

            case DType.float32:
              routine = 'LAPACKE_sgesvd';
              info = LAPACKE_sgesvd(
                101,
                78,
                78,
                m,
                n,
                aCopy.pointer.cast<ffi.Float>(),
                n,
                s2D.pointer.cast<ffi.Float>(),
                ffi.nullptr,
                1,
                ffi.nullptr,
                1,
                superb.cast<ffi.Float>(),
              );

            case DType.complex128:
              routine = 'LAPACKE_zgesvd';
              info = LAPACKE_zgesvd(
                101,
                78,
                78,
                m,
                n,
                aCopy.pointer.cast<ffi.Double>(),
                n,
                s2D.pointer.cast<ffi.Double>(),
                ffi.nullptr,
                1,
                ffi.nullptr,
                1,
                superb.cast<ffi.Double>(),
              );

            case DType.complex64:
              routine = 'LAPACKE_cgesvd';
              info = LAPACKE_cgesvd(
                101,
                78,
                78,
                m,
                n,
                aCopy.pointer.cast<ffi.Float>(),
                n,
                s2D.pointer.cast<ffi.Float>(),
                ffi.nullptr,
                1,
                ffi.nullptr,
                1,
                superb.cast<ffi.Float>(),
              );
            default:
              s2D.dispose();
              throw ArgumentError.value(
                a.dtype,
                'a.dtype',
                'Must be float or complex (unsupported dtype for SVD: ${a.dtype})',
              );
          }
          if (info != 0) {
            s2D.dispose();
            _checkLapackInfo(
              info,
              routine,
              positiveKind: _LapackFailureKind.iterationsExceeded,
              positiveMessage: '$routine failed to converge (info = $info).',
            );
          }
        }

        var offsetS = 0;
        for (var i = 0; i < coords.length; i++) {
          offsetS += coords[i] * sMat.strides[i];
        }

        final sSlice = NDArray<DTypeTag>.view(
          sMat,
          shape: [n],
          strides: sMat.strides.isEmpty ? [1] : [sMat.strides.last],
          offsetElements: offsetS,
        );
        s2D.copy(out: sSlice);
        sSlice.dispose();

        s2D.dispose();
      });
    } finally {
      ScratchArena.reset(marker);
      aCopy.dispose();
    }

    sMat.detachToParentScope();
    return sMat;
  });
}

/// Computes the eigenvalues and eigenvectors of a complex Hermitian (conjugate symmetric) or a real symmetric matrix.
///
/// Returns a record containing:
/// - [eigenvalues]: A 1D array containing the eigenvalues in ascending order.
/// - [eigenvectors]: A 2D matrix whose columns are the normalized eigenvectors.
///
/// **Preconditions:**
/// - [a] must be a square 2D matrix, or a stack of square 2D matrices.
/// - [a] must have a floating-point or complex dtype (`Float32`, `Float64`, `Complex64`, `Complex128`).
///   Integer types are promoted to `Float64`.
/// - If provided, [outEigenvalues] and [outEigenvectors] must have compatible shapes and dtypes.
///
/// **Throws:**
/// - Throws an [IterationsExceededException] if the eigenvalue computation does not converge.
/// - Throws a [LinAlgException] if [a] contains non-finite values or if the LAPACK routine fails.
({NDArray<F> eigenvalues, NDArray<R> eigenvectors})
eigh<F extends DTypeTag, R extends DTypeTag>(
  NDArray<
    DTypeSpec<
      DTypeTag,
      Object?,
      F,
      DTypeTag,
      R,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag
    >
  >
  a, {
  MatrixTriangle uplo = MatrixTriangle.lower,
  NDArray<F>? outEigenvalues,
  NDArray<R>? outEigenvectors,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot calculate eigh on a disposed array.');
  }
  if (a.rank < 2) {
    throw ArgumentError.value(
      a.rank,
      'a.rank',
      'Must be at least 2-dimensional (array must be at least 2-dimensional).',
    );
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m != n) {
    throw ArgumentError.value(
      [m, n],
      'a.shape',
      'Must be square in the last two dimensions (last two dimensions must be square, got $m x $n).',
    );
  }
  checkBlasIntDim(n, 'n', 'eigh');
  checkBlasIntStride(a.strides[a.rank - 2], 'lda', 'eigh');
  checkBlasIntStride(a.strides[a.rank - 1], 'lda', 'eigh');

  final bool promoted =
      a.dtype.isInteger ||
      (a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16;
  DType targetDType = a.dtype;
  if (promoted) {
    targetDType = DType.float64;
  }

  if (targetDType != DType.float64 &&
      targetDType != DType.float32 &&
      targetDType != DType.complex128 &&
      targetDType != DType.complex64) {
    throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      'Must be float or complex (unsupported dtype: ${a.dtype})',
    );
  }

  final DType<DTypeTag> eigenvalueDType = switch (targetDType) {
    DType.float32 || DType.complex64 => DType.float32,
    DType.float64 || DType.complex128 => DType.float64,
    _ => throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      'Must be float or complex (unsupported dtype: ${a.dtype})',
    ),
  };

  final stackShape = a.shape.sublist(0, a.rank - 2);

  final eigenvaluesShape = [...stackShape, n];
  final eigenvectorsShape = [...stackShape, n, n];

  if (outEigenvalues != null) {
    if (outEigenvalues.isDisposed) {
      throw StateError('outEigenvalues is disposed.');
    }
    validateOutBuffer(outEigenvalues, 'outEigenvalues');
    if (!listEquals(outEigenvalues.shape, eigenvaluesShape) ||
        (outEigenvalues.dtype != eigenvalueDType &&
            (!promoted ||
                (outEigenvalues.dtype as DType<DTypeTag>) !=
                    (a.dtype as DType<DTypeTag>)))) {
      throw ArgumentError.value(
        outEigenvalues,
        'outEigenvalues',
        'Must have compatible shape $eigenvaluesShape and dtype $eigenvalueDType (incompatible outEigenvalues, expected shape $eigenvaluesShape and dtype $eigenvalueDType, got shape ${outEigenvalues.shape} and dtype ${outEigenvalues.dtype}).',
      );
    }
  }

  if (outEigenvectors != null) {
    if (outEigenvectors.isDisposed) {
      throw StateError('outEigenvectors is disposed.');
    }
    validateOutBuffer(outEigenvectors, 'outEigenvectors');
    if (!listEquals(outEigenvectors.shape, eigenvectorsShape) ||
        (outEigenvectors.dtype != targetDType &&
            (!promoted ||
                (outEigenvectors.dtype as DType<DTypeTag>) !=
                    (a.dtype as DType<DTypeTag>)))) {
      throw ArgumentError.value(
        outEigenvectors,
        'outEigenvectors',
        'Must have compatible shape $eigenvectorsShape and dtype $targetDType (incompatible outEigenvectors, expected shape $eigenvectorsShape and dtype $targetDType, got shape ${outEigenvectors.shape} and dtype ${outEigenvectors.dtype}).',
      );
    }
  }

  if (outEigenvalues != null &&
      outEigenvectors != null &&
      sharesMemory(outEigenvalues, outEigenvectors)) {
    throw ArgumentError.value(
      outEigenvectors,
      'outEigenvectors',
      'Must not share memory with outEigenvalues.',
    );
  }

  final bool needTempVal =
      outEigenvalues != null &&
      (outEigenvalues.dtype != eigenvalueDType ||
          !outEigenvalues.isContiguous ||
          sharesMemory(a, outEigenvalues));
  final bool needTempVec =
      outEigenvectors != null &&
      (outEigenvectors.dtype != targetDType ||
          !outEigenvectors.isContiguous ||
          sharesMemory(a, outEigenvectors));
  if (needTempVal || needTempVec) {
    return NDArray.scope(() {
      final res = eigh<F, R>(
        a,
        uplo: uplo,
        outEigenvalues: needTempVal ? null : outEigenvalues,
        outEigenvectors: needTempVec ? null : outEigenvectors,
      );
      if (needTempVal) {
        if (outEigenvalues.dtype == res.eigenvalues.dtype) {
          res.eigenvalues.copy(out: outEigenvalues);
        } else {
          castNDArray(
            res.eigenvalues,
            outEigenvalues.dtype,
          ).copy(out: outEigenvalues);
        }
      }
      if (needTempVec) {
        if (outEigenvectors.dtype == res.eigenvectors.dtype) {
          res.eigenvectors.copy(out: outEigenvectors);
        } else {
          castNDArray(
            res.eigenvectors,
            outEigenvectors.dtype,
          ).copy(out: outEigenvectors);
        }
      }
      final finalVal = outEigenvalues ?? res.eigenvalues.detachToParentScope();
      final finalVec =
          outEigenvectors ?? res.eigenvectors.detachToParentScope();
      return (eigenvalues: finalVal, eigenvectors: finalVec);
    });
  }

  return NDArray.scope(() {
    final NDArray<DTypeTag> wMat;
    if (outEigenvalues != null) {
      wMat = outEigenvalues;
    } else {
      wMat = _zerosTyped(eigenvaluesShape, eigenvalueDType);
    }

    final NDArray vMat;
    if (outEigenvectors != null) {
      vMat = outEigenvectors;
    } else {
      vMat = _zerosTyped(eigenvectorsShape, targetDType);
    }

    if (n == 0) {
      NDArray<DTypeTag> finalW = wMat;
      if (outEigenvalues == null) {
        if (F == Float16 && finalW.dtype != DType.float16) {
          finalW = castNDArray<Float16>(finalW, DType.float16);
        } else if (F == BFloat16 && finalW.dtype != DType.bfloat16) {
          finalW = castNDArray<BFloat16>(finalW, DType.bfloat16);
        }
        finalW.detachToParentScope();
      }
      NDArray<DTypeTag> finalV = vMat;
      if (outEigenvectors == null) {
        if (R == Float16 && finalV.dtype != DType.float16) {
          finalV = castNDArray<Float16>(finalV, DType.float16);
        } else if (R == BFloat16 && finalV.dtype != DType.bfloat16) {
          finalV = castNDArray<BFloat16>(finalV, DType.bfloat16);
        }
        finalV.detachToParentScope();
      }
      return (
        eigenvalues: finalW as NDArray<F>,
        eigenvectors: finalV as NDArray<R>,
      );
    }

    final uploVal = uplo == MatrixTriangle.lower ? 76 : 85;
    final jobzVal = 86; // 'V'

    final aCopy2D = _createTyped2D(n, n, targetDType);
    final w2D = _zerosTyped([n], eigenvalueDType);

    final marker = ScratchArena.marker;
    try {
      walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
        coords,
      ) {
        final sliceView = a.slice([
          ...coords.map((c) => Index(c)),
          Slice.all(),
          Slice.all(),
        ]);
        if (sliceView.dtype == targetDType) {
          (sliceView as NDArray<DTypeTag>).copy(out: aCopy2D);
        } else {
          final casted = castNDArray(sliceView, targetDType);
          casted.copy(out: aCopy2D);
          casted.dispose();
        }
        sliceView.dispose();

        final nf = _analyzeNonFinitePtr(aCopy2D.pointer, n * n, targetDType);
        if (nf.hasNaN || nf.hasInf) {
          throw const LinAlgException(
            'Array must not contain infs or NaNs in eigh.',
          );
        }

        final int info;
        final String routine;
        switch (targetDType) {
          case DType.float64:
            routine = 'LAPACKE_dsyevd';
            info = LAPACKE_dsyevd(
              101,
              jobzVal,
              uploVal,
              n,
              aCopy2D.pointer.cast<ffi.Double>(),
              n,
              w2D.pointer.cast<ffi.Double>(),
            );
          case DType.float32:
            routine = 'LAPACKE_ssyevd';
            info = LAPACKE_ssyevd(
              101,
              jobzVal,
              uploVal,
              n,
              aCopy2D.pointer.cast<ffi.Float>(),
              n,
              w2D.pointer.cast<ffi.Float>(),
            );
          case DType.complex128:
            routine = 'LAPACKE_zheevd';
            info = LAPACKE_zheevd(
              101,
              jobzVal,
              uploVal,
              n,
              aCopy2D.pointer.cast<ffi.Double>(),
              n,
              w2D.pointer.cast<ffi.Double>(),
            );
          case DType.complex64:
            routine = 'LAPACKE_cheevd';
            info = LAPACKE_cheevd(
              101,
              jobzVal,
              uploVal,
              n,
              aCopy2D.pointer.cast<ffi.Float>(),
              n,
              w2D.pointer.cast<ffi.Float>(),
            );
          default:
            throw UnimplementedError();
        }
        _checkLapackInfo(
          info,
          routine,
          positiveKind: _LapackFailureKind.iterationsExceeded,
          positiveMessage: '$routine failed to converge: $info',
        );

        final wSlice = wMat.slice([
          ...coords.map((c) => Index(c)),
          Slice.all(),
        ]);
        w2D.copyToContiguous(wSlice);
        wSlice.dispose();

        final vSlice = vMat.slice([
          ...coords.map((c) => Index(c)),
          Slice.all(),
          Slice.all(),
        ]);
        aCopy2D.copyToContiguous(vSlice);
        vSlice.dispose();
      });
    } finally {
      ScratchArena.reset(marker);
      aCopy2D.dispose();
      w2D.dispose();
    }

    NDArray<DTypeTag> finalW = wMat;
    if (outEigenvalues == null) {
      if (F == Float16 && finalW.dtype != DType.float16) {
        finalW = castNDArray<Float16>(finalW, DType.float16);
      } else if (F == BFloat16 && finalW.dtype != DType.bfloat16) {
        finalW = castNDArray<BFloat16>(finalW, DType.bfloat16);
      }
      finalW.detachToParentScope();
    }
    NDArray<DTypeTag> finalV = vMat;
    if (outEigenvectors == null) {
      if (R == Float16 && finalV.dtype != DType.float16) {
        finalV = castNDArray<Float16>(finalV, DType.float16);
      } else if (R == BFloat16 && finalV.dtype != DType.bfloat16) {
        finalV = castNDArray<BFloat16>(finalV, DType.bfloat16);
      }
      finalV.detachToParentScope();
    }
    return (
      eigenvalues: finalW as NDArray<F>,
      eigenvectors: finalV as NDArray<R>,
    );
  });
}

/// Extension on [eigh] result record type to support easy disposal of both arrays.
extension EighRecordDispose<F extends DTypeTag, T extends DTypeTag>
    on ({NDArray<F> eigenvalues, NDArray<T> eigenvectors}) {
  /// Disposes both [eigenvalues] and [eigenvectors] simultaneously,
  /// freeing their underlying unmanaged C memory.
  void dispose() {
    this.eigenvalues.dispose();
    this.eigenvectors.dispose();
  }
}

/// Computes the eigenvalues of a complex Hermitian or real symmetric matrix.
///
/// Returns a 1D array containing the eigenvalues in ascending order.
///
/// **Preconditions:**
/// - [a] must be a square 2D matrix, or a stack of square 2D matrices.
/// - [a] must have a floating-point or complex dtype (`Float32`, `Float64`, `Complex64`, `Complex128`).
///   Integer types are promoted to `Float64`.
/// - If provided, [out] must have compatible shape and dtype.
///
/// **Throws:**
/// - Throws an [IterationsExceededException] if the eigenvalue computation does not converge.
/// - Throws a [LinAlgException] if [a] contains non-finite values or if the LAPACK routine fails.
NDArray<R> eigvalsh<R extends DTypeTag>(
  NDArray<
    DTypeSpec<
      DTypeTag,
      Object?,
      R,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag
    >
  >
  a, {
  MatrixTriangle uplo = MatrixTriangle.lower,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot calculate eigvalsh on a disposed array.');
  }
  if (a.rank < 2) {
    throw ArgumentError.value(
      a.rank,
      'a.rank',
      'Must be at least 2-dimensional (array must be at least 2-dimensional).',
    );
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m != n) {
    throw ArgumentError.value(
      [m, n],
      'a.shape',
      'Must be square in the last two dimensions (last two dimensions must be square, got $m x $n).',
    );
  }
  checkBlasIntDim(n, 'n', 'eigvalsh');
  checkBlasIntStride(a.strides[a.rank - 2], 'lda', 'eigvalsh');
  checkBlasIntStride(a.strides[a.rank - 1], 'lda', 'eigvalsh');

  final bool promoted =
      a.dtype.isInteger ||
      (a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16;
  DType targetDType = a.dtype;
  if (promoted) {
    targetDType = DType.float64;
  }

  if (targetDType != DType.float64 &&
      targetDType != DType.float32 &&
      targetDType != DType.complex128 &&
      targetDType != DType.complex64) {
    throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      'Must be float or complex (unsupported dtype: ${a.dtype})',
    );
  }

  final DType<DTypeTag> eigenvalueDType = switch (targetDType) {
    DType.float32 || DType.complex64 => DType.float32,
    DType.float64 || DType.complex128 => DType.float64,
    _ => throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      'Must be float or complex (unsupported dtype: ${a.dtype})',
    ),
  };

  final stackShape = a.shape.sublist(0, a.rank - 2);
  final eigenvaluesShape = [...stackShape, n];

  if (out != null) {
    if (out.isDisposed) {
      throw StateError('out is disposed.');
    }
    validateOutBuffer(out);
    if (!listEquals(out.shape, eigenvaluesShape) ||
        (out.dtype != eigenvalueDType &&
            (!promoted ||
                (out.dtype as DType<DTypeTag>) !=
                    (a.dtype as DType<DTypeTag>)))) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape $eigenvaluesShape and dtype $eigenvalueDType (incompatible out, expected shape $eigenvaluesShape and dtype $eigenvalueDType, got shape ${out.shape} and dtype ${out.dtype}).',
      );
    }
    if (out.dtype != eigenvalueDType ||
        !out.isContiguous ||
        sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = eigvalsh<R>(a, uplo: uplo);
        if (out.dtype == temp.dtype) {
          temp.copy(out: out);
        } else {
          castNDArray(temp, out.dtype).copy(out: out);
        }
        return out;
      });
    }
  }

  return NDArray.scope(() {
    final NDArray<DTypeTag> wMat;
    if (out != null) {
      wMat = out;
    } else {
      wMat = _zerosTyped(eigenvaluesShape, eigenvalueDType);
    }

    if (n == 0) {
      NDArray<DTypeTag> finalW = wMat;
      if (out == null) {
        if (R == Float16 && finalW.dtype != DType.float16) {
          finalW = castNDArray<Float16>(finalW, DType.float16);
        } else if (R == BFloat16 && finalW.dtype != DType.bfloat16) {
          finalW = castNDArray<BFloat16>(finalW, DType.bfloat16);
        }
        finalW.detachToParentScope();
      }
      return finalW as NDArray<R>;
    }

    final uploVal = uplo == MatrixTriangle.lower ? 76 : 85;
    final jobzVal = 78; // 'N'

    final aCopy2D = _createTyped2D(n, n, targetDType);
    final w2D = _zerosTyped([n], eigenvalueDType);

    final marker = ScratchArena.marker;
    try {
      walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
        coords,
      ) {
        final sliceView = a.slice([
          ...coords.map((c) => Index(c)),
          Slice.all(),
          Slice.all(),
        ]);
        if (sliceView.dtype == targetDType) {
          (sliceView as NDArray<DTypeTag>).copy(out: aCopy2D);
        } else {
          final casted = castNDArray(sliceView, targetDType);
          casted.copy(out: aCopy2D);
          casted.dispose();
        }
        sliceView.dispose();

        final nf = _analyzeNonFinitePtr(aCopy2D.pointer, n * n, targetDType);
        if (nf.hasNaN || nf.hasInf) {
          throw const LinAlgException(
            'Array must not contain infs or NaNs in eigvalsh.',
          );
        }

        final int info;
        final String routine;
        switch (targetDType) {
          case DType.float64:
            routine = 'LAPACKE_dsyevd';
            info = LAPACKE_dsyevd(
              101,
              jobzVal,
              uploVal,
              n,
              aCopy2D.pointer.cast<ffi.Double>(),
              n,
              w2D.pointer.cast<ffi.Double>(),
            );
          case DType.float32:
            routine = 'LAPACKE_ssyevd';
            info = LAPACKE_ssyevd(
              101,
              jobzVal,
              uploVal,
              n,
              aCopy2D.pointer.cast<ffi.Float>(),
              n,
              w2D.pointer.cast<ffi.Float>(),
            );
          case DType.complex128:
            routine = 'LAPACKE_zheevd';
            info = LAPACKE_zheevd(
              101,
              jobzVal,
              uploVal,
              n,
              aCopy2D.pointer.cast<ffi.Double>(),
              n,
              w2D.pointer.cast<ffi.Double>(),
            );
          case DType.complex64:
            routine = 'LAPACKE_cheevd';
            info = LAPACKE_cheevd(
              101,
              jobzVal,
              uploVal,
              n,
              aCopy2D.pointer.cast<ffi.Float>(),
              n,
              w2D.pointer.cast<ffi.Float>(),
            );
          default:
            throw UnimplementedError();
        }
        _checkLapackInfo(
          info,
          routine,
          positiveKind: _LapackFailureKind.iterationsExceeded,
          positiveMessage: '$routine failed to converge: $info',
        );

        final wSlice = wMat.slice([
          ...coords.map((c) => Index(c)),
          Slice.all(),
        ]);
        w2D.copyToContiguous(wSlice);
        wSlice.dispose();
      });
    } finally {
      ScratchArena.reset(marker);
      aCopy2D.dispose();
      w2D.dispose();
    }

    NDArray<DTypeTag> finalW = wMat;
    if (out == null) {
      if (R == Float16 && finalW.dtype != DType.float16) {
        finalW = castNDArray<Float16>(finalW, DType.float16);
      } else if (R == BFloat16 && finalW.dtype != DType.bfloat16) {
        finalW = castNDArray<BFloat16>(finalW, DType.bfloat16);
      }
      finalW.detachToParentScope();
    }
    return finalW as NDArray<R>;
  });
}

/// Computes the Schur decomposition of a matrix.
///
/// A = Z * T * Z^H
///
/// Returns a record containing:
/// - [T]: The Schur form. For real input and `output = SchurForm.real`, it is quasi-upper triangular.
///   For `output = SchurForm.complex`, it is upper triangular.
/// - [Z]: The unitary matrix of Schur vectors.
///
/// **Preconditions:**
/// - It is an error if [a], [outT], or [outZ] is disposed.
/// - It is an error if [a] has rank < 2 or the last two dimensions are not square.
/// - It is an error if [a] has an unsupported dtype.
/// - It is an error if [outT] or [outZ] is provided and has incompatible shape or dtype.
/// - [output] must be [SchurForm.real] or [SchurForm.complex].
///
/// **Throws:**
/// - Throws [LinAlgException] if the QR algorithm fails to compute eigenvalues or if eigenvalues cannot be reordered.
({NDArray<R> t, NDArray<R> z}) schur<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a, {
  SchurForm output = SchurForm.real,
  NDArray<R>? outT,
  NDArray<R>? outZ,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot calculate schur on a disposed array.');
  }
  if (a.rank < 2) {
    throw ArgumentError.value(
      a.rank,
      'a.rank',
      'Must be at least 2-dimensional (array must be at least 2-dimensional).',
    );
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m != n) {
    throw ArgumentError.value(
      [m, n],
      'a.shape',
      'Must be square in the last two dimensions (last two dimensions must be square, got $m x $n).',
    );
  }
  checkBlasIntDim(n, 'n', 'schur');
  checkBlasIntStride(a.strides[a.rank - 2], 'lda', 'schur');
  checkBlasIntStride(a.strides[a.rank - 1], 'lda', 'schur');

  final bool promoted =
      a.dtype.isInteger ||
      a.dtype == DType.float16 ||
      a.dtype == DType.bfloat16;
  DType targetDType = a.dtype;
  if (promoted) {
    targetDType = DType.float64;
  }

  if (targetDType != DType.float64 &&
      targetDType != DType.float32 &&
      targetDType != DType.complex128 &&
      targetDType != DType.complex64) {
    throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      'Must be float or complex (unsupported dtype: ${a.dtype})',
    );
  }

  if (output == SchurForm.complex && !targetDType.isComplex) {
    if (targetDType == DType.float64) {
      targetDType = DType.complex128;
    } else {
      targetDType = DType.complex64;
    }
  }

  final stackShape = a.shape.sublist(0, a.rank - 2);
  final schurShape = [...stackShape, n, n];

  if (outT != null) {
    if (outT.isDisposed) throw StateError('outT is disposed.');
    validateOutBuffer(outT, 'outT');
    if (!listEquals(outT.shape, schurShape) ||
        (outT.dtype != targetDType &&
            (!promoted || output != SchurForm.real || outT.dtype != a.dtype))) {
      throw ArgumentError.value(
        outT,
        'outT',
        'Must have compatible shape $schurShape and dtype $targetDType (incompatible outT).',
      );
    }
  }

  if (outZ != null) {
    if (outZ.isDisposed) throw StateError('outZ is disposed.');
    validateOutBuffer(outZ, 'outZ');
    if (!listEquals(outZ.shape, schurShape) ||
        (outZ.dtype != targetDType &&
            (!promoted || output != SchurForm.real || outZ.dtype != a.dtype))) {
      throw ArgumentError.value(
        outZ,
        'outZ',
        'Must have compatible shape $schurShape and dtype $targetDType (incompatible outZ).',
      );
    }
  }

  if (outT != null && outZ != null && sharesMemory(outT, outZ)) {
    throw ArgumentError.value(outZ, 'outZ', 'Must not share memory with outT.');
  }

  final bool needTempT =
      outT != null &&
      (outT.dtype != targetDType ||
          !outT.isContiguous ||
          sharesMemory(a, outT));
  final bool needTempZ =
      outZ != null &&
      (outZ.dtype != targetDType ||
          !outZ.isContiguous ||
          sharesMemory(a, outZ));
  if (needTempT || needTempZ) {
    return NDArray.scope(() {
      final res = schur<T, R>(
        a,
        output: output,
        outT: needTempT ? null : outT,
        outZ: needTempZ ? null : outZ,
      );
      if (needTempT) {
        if (outT.dtype == res.t.dtype) {
          res.t.copy(out: outT);
        } else {
          castNDArray(res.t, outT.dtype).copy(out: outT);
        }
      }
      if (needTempZ) {
        if (outZ.dtype == res.z.dtype) {
          res.z.copy(out: outZ);
        } else {
          castNDArray(res.z, outZ.dtype).copy(out: outZ);
        }
      }
      final finalT = outT ?? res.t.detachToParentScope();
      final finalZ = outZ ?? res.z.detachToParentScope();
      return (t: finalT, z: finalZ);
    });
  }

  return NDArray.scope(() {
    final NDArray<DTypeTag> tMat = outT ?? _zerosTyped(schurShape, targetDType);
    final NDArray<DTypeTag> zMat = outZ ?? _zerosTyped(schurShape, targetDType);

    if (n == 0) {
      NDArray<DTypeTag> finalT = tMat;
      if (outT == null) {
        if (R == Float16 && finalT.dtype != DType.float16) {
          finalT = castNDArray<Float16>(finalT, DType.float16);
        } else if (R == BFloat16 && finalT.dtype != DType.bfloat16) {
          finalT = castNDArray<BFloat16>(finalT, DType.bfloat16);
        }
        finalT.detachToParentScope();
      }
      NDArray<DTypeTag> finalZ = zMat;
      if (outZ == null) {
        if (R == Float16 && finalZ.dtype != DType.float16) {
          finalZ = castNDArray<Float16>(finalZ, DType.float16);
        } else if (R == BFloat16 && finalZ.dtype != DType.bfloat16) {
          finalZ = castNDArray<BFloat16>(finalZ, DType.bfloat16);
        }
        finalZ.detachToParentScope();
      }
      return (t: finalT as NDArray<R>, z: finalZ as NDArray<R>);
    }

    final jobvsVal = 86; // 'V'
    final sortVal = 78; // 'N'

    final aCopy2D = _createTyped2D(n, n, targetDType);
    final z2D = _zerosTyped([n, n], targetDType);

    final marker = ScratchArena.marker;
    try {
      final ffi.Pointer<ffi.Void> wr;
      final ffi.Pointer<ffi.Void> wi;
      final ffi.Pointer<ffi.Void> w;

      switch (targetDType) {
        case DType.float64:
          wr = ScratchArena.allocate<ffi.Double>(
            n * ffi.sizeOf<ffi.Double>(),
          ).cast<ffi.Void>();
          wi = ScratchArena.allocate<ffi.Double>(
            n * ffi.sizeOf<ffi.Double>(),
          ).cast<ffi.Void>();
          w = ffi.nullptr.cast<ffi.Void>();
        case DType.float32:
          wr = ScratchArena.allocate<ffi.Float>(
            n * ffi.sizeOf<ffi.Float>(),
          ).cast<ffi.Void>();
          wi = ScratchArena.allocate<ffi.Float>(
            n * ffi.sizeOf<ffi.Float>(),
          ).cast<ffi.Void>();
          w = ffi.nullptr.cast<ffi.Void>();
        case DType.complex128:
          wr = ffi.nullptr.cast<ffi.Void>();
          wi = ffi.nullptr.cast<ffi.Void>();
          w = ScratchArena.allocate<ffi.Double>(
            2 * n * ffi.sizeOf<ffi.Double>(),
          ).cast<ffi.Void>();
        default:
          // complex64
          wr = ffi.nullptr.cast<ffi.Void>();
          wi = ffi.nullptr.cast<ffi.Void>();
          w = ScratchArena.allocate<ffi.Float>(
            2 * n * ffi.sizeOf<ffi.Float>(),
          ).cast<ffi.Void>();
      }

      final sdimPtr = ScratchArena.allocate<lapack_int>(
        ffi.sizeOf<lapack_int>(),
      );

      walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
        coords,
      ) {
        final sliceView = a.slice([
          ...coords.map((c) => Index(c)),
          Slice.all(),
          Slice.all(),
        ]);

        if (sliceView.dtype == targetDType) {
          (sliceView as NDArray<DTypeTag>).copy(out: aCopy2D);
        } else {
          final casted = castNDArray(sliceView, targetDType);
          casted.copy(out: aCopy2D);
          casted.dispose();
        }
        sliceView.dispose();

        final nf = _analyzeNonFinitePtr(aCopy2D.pointer, n * n, targetDType);
        if (nf.hasNaN || nf.hasInf) {
          throw const LinAlgException(
            'Array must not contain infs or NaNs in schur.',
          );
        }

        final int info;
        final String routine;
        switch (targetDType) {
          case DType.float64:
            routine = 'LAPACKE_dgees';
            info = LAPACKE_dgees(
              101,
              jobvsVal,
              sortVal,
              ffi.nullptr.cast(),
              n,
              aCopy2D.pointer.cast<ffi.Double>(),
              n,
              sdimPtr,
              wr.cast<ffi.Double>(),
              wi.cast<ffi.Double>(),
              z2D.pointer.cast<ffi.Double>(),
              n,
            );
          case DType.float32:
            routine = 'LAPACKE_sgees';
            info = LAPACKE_sgees(
              101,
              jobvsVal,
              sortVal,
              ffi.nullptr.cast(),
              n,
              aCopy2D.pointer.cast<ffi.Float>(),
              n,
              sdimPtr,
              wr.cast<ffi.Float>(),
              wi.cast<ffi.Float>(),
              z2D.pointer.cast<ffi.Float>(),
              n,
            );
          case DType.complex128:
            routine = 'LAPACKE_zgees';
            info = LAPACKE_zgees(
              101,
              jobvsVal,
              sortVal,
              ffi.nullptr.cast(),
              n,
              aCopy2D.pointer.cast<ffi.Double>(),
              n,
              sdimPtr,
              w.cast<ffi.Double>(),
              z2D.pointer.cast<ffi.Double>(),
              n,
            );
          case DType.complex64:
            routine = 'LAPACKE_cgees';
            info = LAPACKE_cgees(
              101,
              jobvsVal,
              sortVal,
              ffi.nullptr.cast(),
              n,
              aCopy2D.pointer.cast<ffi.Float>(),
              n,
              sdimPtr,
              w.cast<ffi.Float>(),
              z2D.pointer.cast<ffi.Float>(),
              n,
            );
          default:
            throw UnimplementedError();
        }
        if (info > n) {
          throw LinAlgException(
            'Eigenvalues could not be reordered in $routine: $info',
          );
        }
        _checkLapackInfo(
          info,
          routine,
          positiveKind: _LapackFailureKind.iterationsExceeded,
          positiveMessage:
              'The QR algorithm failed to compute all eigenvalues in $routine: $info',
        );

        final tSlice = tMat.slice([
          ...coords.map((c) => Index(c)),
          Slice.all(),
          Slice.all(),
        ]);
        aCopy2D.copyToContiguous(tSlice);
        tSlice.dispose();

        final zSlice = zMat.slice([
          ...coords.map((c) => Index(c)),
          Slice.all(),
          Slice.all(),
        ]);
        z2D.copyToContiguous(zSlice);
        zSlice.dispose();
      });
    } finally {
      ScratchArena.reset(marker);
      aCopy2D.dispose();
      z2D.dispose();
    }

    NDArray<DTypeTag> finalT = tMat;
    if (outT == null) {
      if (R == Float16 && finalT.dtype != DType.float16) {
        finalT = castNDArray<Float16>(finalT, DType.float16);
      } else if (R == BFloat16 && finalT.dtype != DType.bfloat16) {
        finalT = castNDArray<BFloat16>(finalT, DType.bfloat16);
      }
      finalT.detachToParentScope();
    }
    NDArray<DTypeTag> finalZ = zMat;
    if (outZ == null) {
      if (R == Float16 && finalZ.dtype != DType.float16) {
        finalZ = castNDArray<Float16>(finalZ, DType.float16);
      } else if (R == BFloat16 && finalZ.dtype != DType.bfloat16) {
        finalZ = castNDArray<BFloat16>(finalZ, DType.bfloat16);
      }
      finalZ.detachToParentScope();
    }
    return (t: finalT as NDArray<R>, z: finalZ as NDArray<R>);
  });
}

/// Computes the Hessenberg decomposition of a matrix.
///
/// A = Q * H * Q^H
///
/// Returns a record containing:
/// - [h]: The Hessenberg matrix (zero below the first subdiagonal).
/// - [q]: The unitary matrix.
///
/// **Preconditions:**
/// - It is an error if [a], [outH], or [outQ] is disposed.
/// - It is an error if [a] is not square or has rank < 2.
/// - It is an error if [outH] or [outQ] is provided and incompatible.
///
/// **Throws:**
/// - Throws a [LinAlgException] if [a] contains non-finite values or if the LAPACK routine fails.
({NDArray<R> h, NDArray<R> q}) hessenberg<R extends DTypeTag>(
  NDArray<
    DTypeSpec<
      DTypeTag,
      Object?,
      DTypeTag,
      DTypeTag,
      R,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag
    >
  >
  a, {
  NDArray<R>? outH,
  NDArray<R>? outQ,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot calculate hessenberg on a disposed array.');
  }
  if (a.rank < 2) {
    throw ArgumentError.value(
      a.rank,
      'a.rank',
      'Must be at least 2-dimensional (array must be at least 2-dimensional).',
    );
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m != n) {
    throw ArgumentError.value(
      [m, n],
      'a.shape',
      'Must be square in the last two dimensions (last two dimensions must be square, got $m x $n).',
    );
  }
  checkBlasIntDim(n, 'n', 'hessenberg');
  checkBlasIntStride(a.strides[a.rank - 2], 'lda', 'hessenberg');
  checkBlasIntStride(a.strides[a.rank - 1], 'lda', 'hessenberg');

  final bool promoted =
      a.dtype.isInteger ||
      (a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16;
  DType targetDType = a.dtype;
  if (promoted) {
    targetDType = DType.float64;
  }

  if (targetDType != DType.float64 &&
      targetDType != DType.float32 &&
      targetDType != DType.complex128 &&
      targetDType != DType.complex64) {
    throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      'Must be float or complex (unsupported dtype: ${a.dtype})',
    );
  }

  final stackShape = a.shape.sublist(0, a.rank - 2);
  final hessenbergShape = [...stackShape, n, n];

  if (outH != null) {
    if (outH.isDisposed) throw StateError('outH is disposed.');
    validateOutBuffer(outH, 'outH');
    if (!listEquals(outH.shape, hessenbergShape) ||
        (outH.dtype != targetDType && (!promoted || outH.dtype != a.dtype))) {
      throw ArgumentError.value(
        outH,
        'outH',
        'Must have compatible shape $hessenbergShape and dtype $targetDType (incompatible outH).',
      );
    }
  }

  if (outQ != null) {
    if (outQ.isDisposed) throw StateError('outQ is disposed.');
    validateOutBuffer(outQ, 'outQ');
    if (!listEquals(outQ.shape, hessenbergShape) ||
        (outQ.dtype != targetDType && (!promoted || outQ.dtype != a.dtype))) {
      throw ArgumentError.value(
        outQ,
        'outQ',
        'Must have compatible shape $hessenbergShape and dtype $targetDType (incompatible outQ).',
      );
    }
  }

  if (outH != null && outQ != null && sharesMemory(outH, outQ)) {
    throw ArgumentError.value(outQ, 'outQ', 'Must not share memory with outH.');
  }

  final bool needTempH =
      outH != null &&
      (outH.dtype != targetDType ||
          !outH.isContiguous ||
          sharesMemory(a, outH));
  final bool needTempQ =
      outQ != null &&
      (outQ.dtype != targetDType ||
          !outQ.isContiguous ||
          sharesMemory(a, outQ));
  if (needTempH || needTempQ) {
    return NDArray.scope(() {
      final res = hessenberg<R>(
        a,
        outH: needTempH ? null : outH,
        outQ: needTempQ ? null : outQ,
      );
      if (needTempH) {
        if (outH.dtype == res.h.dtype) {
          res.h.copy(out: outH);
        } else {
          castNDArray(res.h, outH.dtype).copy(out: outH);
        }
      }
      if (needTempQ) {
        if (outQ.dtype == res.q.dtype) {
          res.q.copy(out: outQ);
        } else {
          castNDArray(res.q, outQ.dtype).copy(out: outQ);
        }
      }
      final finalH = outH ?? res.h.detachToParentScope();
      final finalQ = outQ ?? res.q.detachToParentScope();
      return (h: finalH, q: finalQ);
    });
  }

  return NDArray.scope(() {
    final NDArray<DTypeTag> hMat =
        outH ?? _zerosTyped(hessenbergShape, targetDType);
    final NDArray<DTypeTag> qMat =
        outQ ?? _zerosTyped(hessenbergShape, targetDType);

    if (n == 0) {
      NDArray<DTypeTag> finalH = hMat;
      if (outH == null) {
        if (R == Float16 && finalH.dtype != DType.float16) {
          finalH = castNDArray<Float16>(finalH, DType.float16);
        } else if (R == BFloat16 && finalH.dtype != DType.bfloat16) {
          finalH = castNDArray<BFloat16>(finalH, DType.bfloat16);
        }
        finalH.detachToParentScope();
      }
      NDArray<DTypeTag> finalQ = qMat;
      if (outQ == null) {
        if (R == Float16 && finalQ.dtype != DType.float16) {
          finalQ = castNDArray<Float16>(finalQ, DType.float16);
        } else if (R == BFloat16 && finalQ.dtype != DType.bfloat16) {
          finalQ = castNDArray<BFloat16>(finalQ, DType.bfloat16);
        }
        finalQ.detachToParentScope();
      }
      return (h: finalH as NDArray<R>, q: finalQ as NDArray<R>);
    }

    final aCopy2D = _createTyped2D(n, n, targetDType);
    final q2D = _zerosTyped([n, n], targetDType);

    final marker = ScratchArena.marker;
    try {
      final int elements = math.max(1, n - 1) * (targetDType.isComplex ? 2 : 1);
      final ffi.Pointer<ffi.Void> tau = switch (targetDType) {
        DType.float64 || DType.complex128 => ScratchArena.allocate<ffi.Double>(
          elements * ffi.sizeOf<ffi.Double>(),
        ).cast<ffi.Void>(),
        _ => ScratchArena.allocate<ffi.Float>(
          elements * ffi.sizeOf<ffi.Float>(),
        ).cast<ffi.Void>(),
      };

      walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
        coords,
      ) {
        final sliceView = a.slice([
          ...coords.map((c) => Index(c)),
          Slice.all(),
          Slice.all(),
        ]);
        if (sliceView.dtype == targetDType) {
          (sliceView as NDArray<DTypeTag>).copy(out: aCopy2D);
        } else {
          final casted = castNDArray(sliceView, targetDType);
          casted.copy(out: aCopy2D);
          casted.dispose();
        }
        sliceView.dispose();

        final nf = _analyzeNonFinitePtr(aCopy2D.pointer, n * n, targetDType);
        if (nf.hasNaN || nf.hasInf) {
          throw const LinAlgException(
            'Array must not contain infs or NaNs in hessenberg.',
          );
        }

        int info = 0;
        final ilo = 1;
        final ihi = n;

        switch (targetDType) {
          case DType.float64:
            info = LAPACKE_dgehrd(
              101,
              n,
              ilo,
              ihi,
              aCopy2D.pointer.cast<ffi.Double>(),
              n,
              tau.cast<ffi.Double>(),
            );
            _checkLapackInfo(info, 'LAPACKE_dgehrd');
          case DType.float32:
            info = LAPACKE_sgehrd(
              101,
              n,
              ilo,
              ihi,
              aCopy2D.pointer.cast<ffi.Float>(),
              n,
              tau.cast<ffi.Float>(),
            );
            _checkLapackInfo(info, 'LAPACKE_sgehrd');
          case DType.complex128:
            info = LAPACKE_zgehrd(
              101,
              n,
              ilo,
              ihi,
              aCopy2D.pointer.cast<ffi.Double>(),
              n,
              tau.cast<ffi.Double>(),
            );
            _checkLapackInfo(info, 'LAPACKE_zgehrd');
          case DType.complex64:
            info = LAPACKE_cgehrd(
              101,
              n,
              ilo,
              ihi,
              aCopy2D.pointer.cast<ffi.Float>(),
              n,
              tau.cast<ffi.Float>(),
            );
            _checkLapackInfo(info, 'LAPACKE_cgehrd');
          default:
            throw UnimplementedError();
        }

        aCopy2D.copy(out: q2D);

        final hSlice = hMat.slice([
          ...coords.map((c) => Index(c)),
          Slice.all(),
          Slice.all(),
        ]);
        try {
          aCopy2D.copyToContiguous(hSlice);

          // Zero out elements below the first subdiagonal in H using direct pointer access.
          switch (targetDType) {
            case DType.float64:
              final ptr = hSlice.pointer.cast<ffi.Double>();
              for (var i = 2; i < n; i++) {
                for (var j = 0; j < i - 1; j++) {
                  ptr[i * n + j] = 0.0;
                }
              }
            case DType.float32:
              final ptr = hSlice.pointer.cast<ffi.Float>();
              for (var i = 2; i < n; i++) {
                for (var j = 0; j < i - 1; j++) {
                  ptr[i * n + j] = 0.0;
                }
              }
            case DType.complex128:
              final ptr = hSlice.pointer.cast<ffi.Double>();
              for (var i = 2; i < n; i++) {
                for (var j = 0; j < i - 1; j++) {
                  ptr[2 * (i * n + j)] = 0.0;
                  ptr[2 * (i * n + j) + 1] = 0.0;
                }
              }
            case DType.complex64:
              final ptr = hSlice.pointer.cast<ffi.Float>();
              for (var i = 2; i < n; i++) {
                for (var j = 0; j < i - 1; j++) {
                  ptr[2 * (i * n + j)] = 0.0;
                  ptr[2 * (i * n + j) + 1] = 0.0;
                }
              }
            default:
              break;
          }

          switch (targetDType) {
            case DType.float64:
              info = LAPACKE_dorghr(
                101,
                n,
                ilo,
                ihi,
                q2D.pointer.cast<ffi.Double>(),
                n,
                tau.cast<ffi.Double>(),
              );
              _checkLapackInfo(info, 'LAPACKE_dorghr');
            case DType.float32:
              info = LAPACKE_sorghr(
                101,
                n,
                ilo,
                ihi,
                q2D.pointer.cast<ffi.Float>(),
                n,
                tau.cast<ffi.Float>(),
              );
              _checkLapackInfo(info, 'LAPACKE_sorghr');
            case DType.complex128:
              info = LAPACKE_zunghr(
                101,
                n,
                ilo,
                ihi,
                q2D.pointer.cast<ffi.Double>(),
                n,
                tau.cast<ffi.Double>(),
              );
              _checkLapackInfo(info, 'LAPACKE_zunghr');
            case DType.complex64:
              info = LAPACKE_cunghr(
                101,
                n,
                ilo,
                ihi,
                q2D.pointer.cast<ffi.Float>(),
                n,
                tau.cast<ffi.Float>(),
              );
              _checkLapackInfo(info, 'LAPACKE_cunghr');
            default:
              throw UnimplementedError();
          }
        } finally {
          hSlice.dispose();
        }

        final qSlice = qMat.slice([
          ...coords.map((c) => Index(c)),
          Slice.all(),
          Slice.all(),
        ]);
        q2D.copyToContiguous(qSlice);
        qSlice.dispose();
      });
    } finally {
      ScratchArena.reset(marker);
      aCopy2D.dispose();
      q2D.dispose();
    }

    NDArray<DTypeTag> finalH = hMat;
    if (outH == null) {
      if (R == Float16 && finalH.dtype != DType.float16) {
        finalH = castNDArray<Float16>(finalH, DType.float16);
      } else if (R == BFloat16 && finalH.dtype != DType.bfloat16) {
        finalH = castNDArray<BFloat16>(finalH, DType.bfloat16);
      }
      finalH.detachToParentScope();
    }
    NDArray<DTypeTag> finalQ = qMat;
    if (outQ == null) {
      if (R == Float16 && finalQ.dtype != DType.float16) {
        finalQ = castNDArray<Float16>(finalQ, DType.float16);
      } else if (R == BFloat16 && finalQ.dtype != DType.bfloat16) {
        finalQ = castNDArray<BFloat16>(finalQ, DType.bfloat16);
      }
      finalQ.detachToParentScope();
    }
    return (h: finalH as NDArray<R>, q: finalQ as NDArray<R>);
  });
}

NDArray<DTypeTag> _createTyped2D(int rows, int cols, DType<DTypeTag> dtype) {
  switch (dtype) {
    case DType.float64:
      return NDArray<Float64>.create([rows, cols], DType.float64);
    case DType.float32:
      return NDArray<Float32>.create([rows, cols], DType.float32);
    case DType.complex128:
      return NDArray<Complex128>.create([rows, cols], DType.complex128);
    case DType.complex64:
      return NDArray<Complex64>.create([rows, cols], DType.complex64);
    default:
      throw UnimplementedError('Unsupported dtype: $dtype');
  }
}

NDArray<DTypeTag> _zerosTyped(List<int> shape, DType<DTypeTag> dtype) {
  switch (dtype) {
    case DType.float64:
      return NDArray<Float64>.zeros(shape, DType.float64);
    case DType.float32:
      return NDArray<Float32>.zeros(shape, DType.float32);
    case DType.complex128:
      return NDArray<Complex128>.zeros(shape, DType.complex128);
    case DType.complex64:
      return NDArray<Complex64>.zeros(shape, DType.complex64);
    default:
      throw UnimplementedError('Unsupported dtype: $dtype');
  }
}

//

/// Computes the outer product of two vectors.
///
/// Given two input vectors [a] and [b], computes the outer product matrix:
/// `res[i, j] = a[i] * b[j]`.
/// If the input arrays are not 1-dimensional, they are flattened first.
///
/// **Preconditions:**
/// - It is an error if [a], [b], or [out] is disposed.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N_a \times N_b)$ using highly optimized native strided loops.
///
/// **Example:**
/// {@example /example/linalg_advanced_example.dart lang=dart}
///
/// Reference: [NumPy outer](https://numpy.org/doc/stable/reference/generated/numpy.outer.html)
NDArray<T> outer<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute outer() on a disposed array.');
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }

  final sizeA = a.size;
  final sizeB = b.size;
  final expectedShape = [sizeA, sizeB];
  final targetDType = a.dtype;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, expectedShape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape $expectedShape and dtype $targetDType (provided out recycler has incompatible shape or dtype, expected shape $expectedShape and dtype $targetDType).',
      );
    }
    if (!out.isContiguous || sharesMemory(a, out) || sharesMemory(b, out)) {
      return NDArray.scope(() {
        final temp = outer<DTypeTag>(a, b);
        temp.copy(out: out);
        return out;
      });
    }
  }

  return NDArray.scope(() {
    final result = out ?? NDArray<T>.create(expectedShape, targetDType);

    final flatA = a.rank == 1 ? a : a.ravel();
    final flatB = b.rank == 1 ? b : b.ravel();

    final aCast = flatA;
    final bCast = flatB;

    try {
      switch (targetDType) {
        case DType.float64:
          s_outer_double(
            aCast.pointer.cast(),
            aCast.strides.isEmpty ? 1 : aCast.strides[0],
            sizeA,
            bCast.pointer.cast(),
            bCast.strides.isEmpty ? 1 : bCast.strides[0],
            sizeB,
            result.pointer.cast(),
            result.strides[0],
            result.strides[1],
          );
        case DType.float32:
          s_outer_float(
            aCast.pointer.cast(),
            aCast.strides.isEmpty ? 1 : aCast.strides[0],
            sizeA,
            bCast.pointer.cast(),
            bCast.strides.isEmpty ? 1 : bCast.strides[0],
            sizeB,
            result.pointer.cast(),
            result.strides[0],
            result.strides[1],
          );
        case DType.int64:
        case DType.uint64:
          s_outer_int64(
            aCast.pointer.cast(),
            aCast.strides.isEmpty ? 1 : aCast.strides[0],
            sizeA,
            bCast.pointer.cast(),
            bCast.strides.isEmpty ? 1 : bCast.strides[0],
            sizeB,
            result.pointer.cast(),
            result.strides[0],
            result.strides[1],
          );
        case DType.int32:
        case DType.uint32:
          s_outer_int32(
            aCast.pointer.cast(),
            aCast.strides.isEmpty ? 1 : aCast.strides[0],
            sizeA,
            bCast.pointer.cast(),
            bCast.strides.isEmpty ? 1 : bCast.strides[0],
            sizeB,
            result.pointer.cast(),
            result.strides[0],
            result.strides[1],
          );
        case DType.uint8:
        case DType.int8:
          s_outer_uint8(
            aCast.pointer.cast(),
            aCast.strides.isEmpty ? 1 : aCast.strides[0],
            sizeA,
            bCast.pointer.cast(),
            bCast.strides.isEmpty ? 1 : bCast.strides[0],
            sizeB,
            result.pointer.cast(),
            result.strides[0],
            result.strides[1],
          );
        case DType.int16:
        case DType.uint16:
          s_outer_int16(
            aCast.pointer.cast(),
            aCast.strides.isEmpty ? 1 : aCast.strides[0],
            sizeA,
            bCast.pointer.cast(),
            bCast.strides.isEmpty ? 1 : bCast.strides[0],
            sizeB,
            result.pointer.cast(),
            result.strides[0],
            result.strides[1],
          );
        case DType.complex128:
          s_outer_complex128(
            aCast.pointer.cast(),
            aCast.strides.isEmpty ? 1 : aCast.strides[0],
            sizeA,
            bCast.pointer.cast(),
            bCast.strides.isEmpty ? 1 : bCast.strides[0],
            sizeB,
            result.pointer.cast(),
            result.strides[0],
            result.strides[1],
          );
        case DType.complex64:
          s_outer_complex64(
            aCast.pointer.cast(),
            aCast.strides.isEmpty ? 1 : aCast.strides[0],
            sizeA,
            bCast.pointer.cast(),
            bCast.strides.isEmpty ? 1 : bCast.strides[0],
            sizeB,
            result.pointer.cast(),
            result.strides[0],
            result.strides[1],
          );
        case DType.boolean:
          s_outer_boolean(
            aCast.pointer.cast(),
            aCast.strides.isEmpty ? 1 : aCast.strides[0],
            sizeA,
            bCast.pointer.cast(),
            bCast.strides.isEmpty ? 1 : bCast.strides[0],
            sizeB,
            result.pointer.cast(),
            result.strides[0],
            result.strides[1],
          );
        case DType.float16:
        case DType.bfloat16:
          final doubleA = castNDArray(flatA, DType.float64);
          final doubleB = castNDArray(flatB, DType.float64);
          final doubleRes = outer(doubleA, doubleB);
          final casted = castNDArray(doubleRes, result.dtype);
          casted.copy(out: result);
          doubleA.dispose();
          doubleB.dispose();
          doubleRes.dispose();
          casted.dispose();
      }
    } finally {
      if (!identical(flatA, a)) flatA.dispose();
      if (!identical(flatB, b)) flatB.dispose();
    }

    if (out == null) {
      return result.detachToParentScope();
    }
    return result;
  });
}

/// Computes the cross product of two (arrays of) vectors.
///
/// The cross product of two vectors is defined in 3D (and 2D, where it returns the z-component as a scalar).
/// If the inputs are multidimensional, the cross product is computed along the specified axes.
///
/// **Preconditions:**
/// - It is an error if [a], [b], or [out] is disposed.
/// - It is an error if axes sizes are not 2 or 3, or are mismatched.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Uses native C vector cross loops.
///
/// **Example:**
/// {@example /example/linalg_advanced_example.dart lang=dart}
///
/// Reference: [NumPy cross](https://numpy.org/doc/stable/reference/generated/numpy.cross.html)
NDArray<T> cross<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  int? axisa,
  int? axisb,
  int? axisc,
  int? axis,
  NDArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute cross() on a disposed array.');
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }

  final origAxisA = axis ?? axisa ?? -1;
  final origAxisB = axis ?? axisb ?? -1;
  var axisA = origAxisA;
  var axisB = origAxisB;
  var axisC = axis ?? axisc ?? -1;

  if (axisA < 0) axisA = a.rank + axisA;
  if (axisB < 0) axisB = b.rank + axisB;

  if (axisA < 0 || axisA >= a.rank) {
    throw RangeError.range(
      origAxisA,
      -a.rank,
      a.rank - 1,
      axis != null ? 'axis' : 'axisa',
      'Must be within valid rank range',
    );
  }
  if (axisB < 0 || axisB >= b.rank) {
    throw RangeError.range(
      origAxisB,
      -b.rank,
      b.rank - 1,
      axis != null ? 'axis' : 'axisb',
      'Must be within valid rank range',
    );
  }

  final lenA = a.shape[axisA];
  final lenB = b.shape[axisB];

  if ((lenA != 2 && lenA != 3) || (lenB != 2 && lenB != 3)) {
    throw ArgumentError.value(
      [lenA, lenB],
      'axisa, axisb',
      'Must have axis sizes of 2 or 3 (cross product axes sizes must be 2 or 3, got axisa size $lenA and axisb size $lenB).',
    );
  }
  if (lenA != lenB) {
    throw ArgumentError.value(
      lenB,
      'axisb',
      'Must match axisa size $lenA (mismatched cross product axes sizes: axisa size $lenA != axisb size $lenB).',
    );
  }

  final is3D = lenA == 3;

  final stackA = List<int>.from(a.shape)..removeAt(axisA);
  final stackB = List<int>.from(b.shape)..removeAt(axisB);
  final broadcastStack = broadcastStackShapes(stackA, stackB);

  final expectedShape = List<int>.from(broadcastStack);
  if (is3D) {
    final origAxisC = axisC;
    var finalAxisC = axisC;
    if (finalAxisC < 0) finalAxisC = expectedShape.length + 1 + finalAxisC;
    if (finalAxisC < 0 || finalAxisC > expectedShape.length) {
      throw RangeError.range(
        origAxisC,
        -(expectedShape.length + 1),
        expectedShape.length,
        axis != null ? 'axis' : 'axisc',
        'Must be within valid rank range',
      );
    }
    expectedShape.insert(finalAxisC, 3);
    axisC = finalAxisC;
  }

  final targetDType = a.dtype;
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, expectedShape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape $expectedShape and dtype $targetDType (provided out recycler has incompatible shape or dtype, expected shape $expectedShape and dtype $targetDType).',
      );
    }
    if (!out.isContiguous || sharesMemory(a, out) || sharesMemory(b, out)) {
      return NDArray.scope(() {
        final temp = cross<DTypeTag>(
          a,
          b,
          axisa: axisa,
          axisb: axisb,
          axisc: axisc,
          axis: axis,
        );
        temp.copy(out: out);
        return out;
      });
    }
  }

  final result = out ?? NDArray<T>.create(expectedShape, targetDType);

  if (targetDType == DType.float16 || targetDType == DType.bfloat16) {
    return NDArray.scope(() {
      try {
        final doubleA = castNDArray(a, DType.float64);
        final doubleB = castNDArray(b, DType.float64);
        final doubleRes = cross(
          doubleA,
          doubleB,
          axisa: axisa,
          axisb: axisb,
          axisc: axisc,
          axis: axis,
        );
        final casted = castNDArray(doubleRes, targetDType);
        casted.copy(out: result);
        return result;
      } catch (_) {
        if (out == null) result.dispose();
        rethrow;
      }
    });
  }

  final aCast = a;
  final bCast = b;

  final lenResult = broadcastStack.length;
  final walkStridesA = List<int>.filled(lenResult, 0);
  final walkStridesB = List<int>.filled(lenResult, 0);
  final walkStridesRes = List<int>.filled(lenResult, 0);

  for (var i = 0; i < lenResult; i++) {
    final resAxis = lenResult - 1 - i;
    final axisIdxA = stackA.length - 1 - i;
    final axisIdxB = stackB.length - 1 - i;

    var resAxisIdx = resAxis;
    if (is3D && resAxis >= axisC) {
      resAxisIdx = resAxis + 1;
    }

    if (axisIdxA >= 0) {
      final origAxisA = axisIdxA < axisA ? axisIdxA : axisIdxA + 1;
      walkStridesA[resAxis] = (stackA[axisIdxA] == broadcastStack[resAxis])
          ? aCast.strides[origAxisA]
          : 0;
    }
    if (axisIdxB >= 0) {
      final origAxisB = axisIdxB < axisB ? axisIdxB : axisIdxB + 1;
      walkStridesB[resAxis] = (stackB[axisIdxB] == broadcastStack[resAxis])
          ? bCast.strides[origAxisB]
          : 0;
    }
    walkStridesRes[resAxis] = result.strides[resAxisIdx];
  }

  final strideVecA = aCast.strides[axisA];
  final strideVecB = bCast.strides[axisB];
  final strideVecRes = is3D ? result.strides[axisC] : 0;

  void walk(int dim, int offsetA, int offsetB, int offsetRes) {
    if (dim == lenResult) {
      switch (targetDType) {
        case DType.float64:
          if (is3D) {
            s_cross_3d_double(
              aCast.pointer.cast<ffi.Double>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Double>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Double>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_double(
              aCast.pointer.cast<ffi.Double>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Double>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Double>() + offsetRes,
            );
          }
        case DType.float32:
          if (is3D) {
            s_cross_3d_float(
              aCast.pointer.cast<ffi.Float>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Float>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Float>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_float(
              aCast.pointer.cast<ffi.Float>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Float>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Float>() + offsetRes,
            );
          }
        case DType.int64:
        case DType.uint64:
          if (is3D) {
            s_cross_3d_int64(
              aCast.pointer.cast<ffi.Int64>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int64>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int64>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_int64(
              aCast.pointer.cast<ffi.Int64>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int64>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int64>() + offsetRes,
            );
          }
        case DType.int32:
        case DType.uint32:
          if (is3D) {
            s_cross_3d_int32(
              aCast.pointer.cast<ffi.Int32>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int32>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int32>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_int32(
              aCast.pointer.cast<ffi.Int32>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int32>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int32>() + offsetRes,
            );
          }
        case DType.uint8:
        case DType.int8:
          if (is3D) {
            s_cross_3d_uint8(
              aCast.pointer.cast<ffi.Uint8>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Uint8>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Uint8>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_uint8(
              aCast.pointer.cast<ffi.Uint8>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Uint8>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Uint8>() + offsetRes,
            );
          }
        case DType.int16:
        case DType.uint16:
          if (is3D) {
            s_cross_3d_int16(
              aCast.pointer.cast<ffi.Int16>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int16>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int16>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_int16(
              aCast.pointer.cast<ffi.Int16>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int16>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int16>() + offsetRes,
            );
          }
        case DType.complex128:
          if (is3D) {
            s_cross_3d_complex128(
              aCast.pointer.cast<cpx_t>() + offsetA,
              strideVecA,
              bCast.pointer.cast<cpx_t>() + offsetB,
              strideVecB,
              result.pointer.cast<cpx_t>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_complex128(
              aCast.pointer.cast<cpx_t>() + offsetA,
              strideVecA,
              bCast.pointer.cast<cpx_t>() + offsetB,
              strideVecB,
              result.pointer.cast<cpx_t>() + offsetRes,
            );
          }
        case DType.complex64:
          if (is3D) {
            s_cross_3d_complex64(
              aCast.pointer.cast<cpx_f_t>() + offsetA,
              strideVecA,
              bCast.pointer.cast<cpx_f_t>() + offsetB,
              strideVecB,
              result.pointer.cast<cpx_f_t>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_complex64(
              aCast.pointer.cast<cpx_f_t>() + offsetA,
              strideVecA,
              bCast.pointer.cast<cpx_f_t>() + offsetB,
              strideVecB,
              result.pointer.cast<cpx_f_t>() + offsetRes,
            );
          }
        case DType.boolean:
          if (is3D) {
            s_cross_3d_boolean(
              aCast.pointer.cast<ffi.Uint8>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Uint8>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Uint8>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_boolean(
              aCast.pointer.cast<ffi.Uint8>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Uint8>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Uint8>() + offsetRes,
            );
          }
        case DType.float16:
        case DType.bfloat16:
          break;
      }
      return;
    }

    final size = broadcastStack[dim];
    final strideA = walkStridesA[dim];
    final strideB = walkStridesB[dim];
    final strideRes = walkStridesRes[dim];

    for (var i = 0; i < size; i++) {
      walk(
        dim + 1,
        offsetA + i * strideA,
        offsetB + i * strideB,
        offsetRes + i * strideRes,
      );
    }
  }

  walk(0, 0, 0, 0);

  return result;
}

/// Matrix product of two arrays [a] and [b] computed into the specified target [dtype].
///
/// Casts [a] and [b] to [dtype] and computes [matmul], returning an
/// [NDArray<R>] whose static type [R] is inferred from [dtype].
///
/// **Preconditions:**
/// - It is an error if [a], [b], or [out] is disposed.
/// - Neither [a] nor [b] may be a 0-D scalar array.
/// - Inner matrix dimensions must be compatible.
/// - If [out] is provided, it must be writeable, have the output shape, and have dtype [dtype].
///
/// **Performance considerations:**
/// - Dispatches 2-D and batched floating-point and complex matrix products to OpenBLAS.
///
/// **Example:**
/// {@example /example/linalg_example.dart lang=dart}
///
/// Reference: [NumPy matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
NDArray<R> matmulAs<
  Ta extends DTypeTag,
  Tb extends DTypeTag,
  R extends DTypeTag
>(NDArray<Ta> a, NDArray<Tb> b, DType<R> dtype, {NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute matmulAs() on a disposed array.');
  }
  if ((a.dtype as DType<DTypeTag>) == dtype &&
      (b.dtype as DType<DTypeTag>) == dtype) {
    return matmul<DTypeTag>(a, b, out: out) as NDArray<R>;
  }
  return NDArray.scope(() {
    final aCast = castNDArray<R>(a, dtype);
    final bCast = castNDArray<R>(b, dtype);
    final res = matmul<R>(aCast, bCast, out: out);
    return out ?? res.detachToParentScope();
  });
}

/// Computes the outer product of two vectors [a] and [b] into the specified target [dtype].
///
/// Casts [a] and [b] to [dtype] and computes [outer], returning an
/// [NDArray<R>] whose static type [R] is inferred from [dtype].
///
/// **Preconditions:**
/// - It is an error if [a], [b], or [out] is disposed.
/// - If [out] is provided, it must be writeable, have shape `[a.size, b.size]`, and have dtype [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(M \cdot N)$ where $M = \text{a.size}$ and $N = \text{b.size}$.
///
/// **Example:**
/// {@example /example/linalg_advanced_example.dart lang=dart}
///
/// Reference: [NumPy outer](https://numpy.org/doc/stable/reference/generated/numpy.outer.html)
NDArray<R> outerAs<
  Ta extends DTypeTag,
  Tb extends DTypeTag,
  R extends DTypeTag
>(NDArray<Ta> a, NDArray<Tb> b, DType<R> dtype, {NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute outerAs() on a disposed array.');
  }
  if ((a.dtype as DType<DTypeTag>) == dtype &&
      (b.dtype as DType<DTypeTag>) == dtype) {
    return outer<DTypeTag>(a, b, out: out) as NDArray<R>;
  }
  return NDArray.scope(() {
    final aCast = castNDArray<R>(a, dtype);
    final bCast = castNDArray<R>(b, dtype);
    final res = outer<R>(aCast, bCast, out: out);
    return out ?? res.detachToParentScope();
  });
}

/// Computes the cross product of two (arrays of) vectors [a] and [b] into the specified target [dtype].
///
/// Casts [a] and [b] to [dtype] and computes [cross], returning an
/// [NDArray<R>] whose static type [R] is inferred from [dtype].
///
/// **Preconditions:**
/// - It is an error if [a], [b], or [out] is disposed.
/// - Cross product axes sizes must be 2 or 3 and match.
/// - If [out] is provided, it must be writeable, have the output shape, and have dtype [dtype].
///
/// **Performance considerations:**
/// - Uses native C vector cross loops.
///
/// **Example:**
/// {@example /example/linalg_advanced_example.dart lang=dart}
///
/// Reference: [NumPy cross](https://numpy.org/doc/stable/reference/generated/numpy.cross.html)
NDArray<R>
crossAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  DType<R> dtype, {
  int? axisa,
  int? axisb,
  int? axisc,
  int? axis,
  NDArray<R>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute crossAs() on a disposed array.');
  }
  if ((a.dtype as DType<DTypeTag>) == dtype &&
      (b.dtype as DType<DTypeTag>) == dtype) {
    return cross<DTypeTag>(
          a,
          b,
          axisa: axisa,
          axisb: axisb,
          axisc: axisc,
          axis: axis,
          out: out,
        )
        as NDArray<R>;
  }
  return NDArray.scope(() {
    final aCast = castNDArray<R>(a, dtype);
    final bCast = castNDArray<R>(b, dtype);
    final res = cross<R>(
      aCast,
      bCast,
      axisa: axisa,
      axisb: axisb,
      axisc: axisc,
      axis: axis,
      out: out,
    );
    return out ?? res.detachToParentScope();
  });
}

/// Matrix triangle selection for symmetric/Hermitian operations.
enum MatrixTriangle {
  /// Lower triangular part.
  lower,

  /// Upper triangular part.
  upper,
}

/// Representation form for Schur decomposition.
enum SchurForm {
  /// Real Schur form.
  real,

  /// Complex Schur form.
  complex,
}

/// Supported norm orders and calculation modes for vector and matrix norm computations.
enum NormKind {
  /// Frobenius norm (square root of sum of absolute squares).
  frobenius,

  /// Nuclear norm (sum of singular values).
  nuclear,

  /// 1-norm (maximum absolute column sum for matrices, sum of absolute values for vectors).
  l1,

  /// Negative 1-norm (minimum absolute column sum for matrices).
  negL1,

  /// 2-norm (largest singular value for matrices, Euclidean norm for vectors).
  l2,

  /// Negative 2-norm (smallest singular value for matrices).
  negL2,

  /// Infinity norm (maximum absolute row sum for matrices, max absolute value for vectors).
  infinity,

  /// Negative infinity norm (minimum absolute row sum for matrices, min absolute value for vectors).
  negInfinity,
}

/// Computes a vector or matrix norm.
///
/// Computes one of the standard vector or matrix norms (magnitude) along the specified axis/axes.
/// The result is always a real-valued floating-point array.
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [axis] or [ord] combinations are invalid.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Uses native vector reductions for Chebyshev, L1, and L2 vector calculations.
///
/// **Example:**
/// {@example /example/linalg_advanced_example.dart lang=dart}
///
/// Reference: [NumPy linalg.norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)
NDArray<R> norm<R extends DTypeTag>(
  NDArray<
    DTypeSpec<
      DTypeTag,
      Object?,
      R,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag
    >
  >
  a, {
  Object? ord,
  Object? axis,
  bool keepdims = false,
  NDArray<R>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute norm() on a disposed array.');
  }

  final rank = a.shape.length;
  List<int> targetAxes;
  if (axis == null) {
    if (rank > 2) {
      throw ArgumentError.value(
        rank,
        'rank',
        'Must be 1D or 2D if axis is null (improper axis specification: if axis is null, input must be 1D or 2D).',
      );
    }
    targetAxes = List<int>.generate(rank, (i) => i);
  } else if (axis is int) {
    var normAx = axis;
    if (normAx < 0) normAx = rank + normAx;
    if (normAx < 0 || normAx >= rank) {
      throw RangeError.range(
        axis,
        -rank,
        rank - 1,
        'axis',
        'Must be within valid rank range',
      );
    }
    targetAxes = [normAx];
  } else if (axis is List<int>) {
    if (axis.length != 1 && axis.length != 2) {
      throw ArgumentError.value(
        axis.length,
        'axis.length',
        'Must contain exactly 1 or 2 elements (axis list must contain exactly 1 or 2 elements).',
      );
    }
    final normAxes = List<int>.from(axis);
    for (var i = 0; i < normAxes.length; i++) {
      if (normAxes[i] < 0) normAxes[i] = rank + normAxes[i];
      if (normAxes[i] < 0 || normAxes[i] >= rank) {
        throw RangeError.range(
          axis[i],
          -rank,
          rank - 1,
          'axis',
          'Must be within valid rank range',
        );
      }
    }
    if (normAxes.length == 2 && normAxes[0] == normAxes[1]) {
      throw ArgumentError.value(axis, 'axis', 'axes must be distinct.');
    }
    targetAxes = normAxes;
  } else {
    throw ArgumentError.value(
      axis,
      'axis',
      'Must be null, int, or List<int> (axis must be null, int, or List<int>).',
    );
  }

  final isVecNorm = targetAxes.length == 1;
  final DType targetDType =
      ((a.dtype as DType<DTypeTag>) == DType.float32 ||
          (a.dtype as DType<DTypeTag>) == DType.complex64)
      ? DType.float32
      : DType.float64;

  final List<int> expectedShape;
  if (keepdims) {
    expectedShape = List<int>.from(a.shape);
    for (final ax in targetAxes) {
      expectedShape[ax] = 1;
    }
  } else {
    expectedShape = List<int>.from(a.shape);
    final sortedAxes = List<int>.from(targetAxes)
      ..sort((x, y) => y.compareTo(x));
    for (final ax in sortedAxes) {
      expectedShape.removeAt(ax);
    }
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, expectedShape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape $expectedShape and dtype $targetDType (provided out buffer has incompatible shape or dtype).',
      );
    }
    if (!out.isContiguous || sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = norm<R>(a, ord: ord, axis: axis, keepdims: keepdims);
        temp.copy(out: out);
        return out;
      });
    }
  }

  return NDArray.scope(() {
    final NDArray<R> result =
        out ?? (NDArray<R>.create(expectedShape, targetDType as DType<R>));

    if (targetAxes.length == rank && !keepdims) {
      // Global norm
      if (isVecNorm) {
        final val = _vectorNorm(a, ord, targetDType);
        if (targetDType == DType.float32) {
          result.pointer.cast<ffi.Float>()[0] = val;
        } else {
          result.pointer.cast<ffi.Double>()[0] = val;
        }
      } else {
        final val = _matrixNorm(a, ord, targetDType);
        if (targetDType == DType.float32) {
          result.pointer.cast<ffi.Float>()[0] = val;
        } else {
          result.pointer.cast<ffi.Double>()[0] = val;
        }
      }
      if (out == null) {
        return result.detachToParentScope();
      }
      return result;
    }

    // Reduction along specific axes
    final List<int> currentCoords = List<int>.filled(a.shape.length, 0);

    final List<int> stackShape = List<int>.from(a.shape);
    final sortedAxes = List<int>.from(targetAxes)
      ..sort((x, y) => y.compareTo(x));
    for (final ax in sortedAxes) {
      stackShape.removeAt(ax);
    }

    void walkStack(int dim, List<int> coords) {
      if (dim == stackShape.length) {
        // Reconstruct original coordinates for slicing
        var stackIdx = 0;
        for (var i = 0; i < a.shape.length; i++) {
          if (!targetAxes.contains(i)) {
            currentCoords[i] = coords[stackIdx++];
          }
        }

        final NDArray<DTypeTag> slice;
        if (isVecNorm) {
          final ax = targetAxes[0];
          final len = a.shape[ax];
          var offset = 0;
          for (var i = 0; i < a.shape.length; i++) {
            if (i != ax) {
              offset += currentCoords[i] * a.strides[i];
            }
          }
          slice = NDArray.view(
            a,
            shape: [len],
            strides: [a.strides[ax]],
            offsetElements: offset,
          );
        } else {
          final ax0 = targetAxes[0];
          final ax1 = targetAxes[1];
          final len0 = a.shape[ax0];
          final len1 = a.shape[ax1];
          var offset = 0;
          for (var i = 0; i < a.shape.length; i++) {
            if (i != ax0 && i != ax1) {
              offset += currentCoords[i] * a.strides[i];
            }
          }
          slice = NDArray.view(
            a,
            shape: [len0, len1],
            strides: [a.strides[ax0], a.strides[ax1]],
            offsetElements: offset,
          );
        }

        final double val;
        if (isVecNorm) {
          val = _vectorNorm(slice, ord, targetDType);
        } else {
          val = _matrixNorm(slice, ord, targetDType);
        }
        slice.dispose();

        // Calculate dest flat index
        var destOffset = 0;
        if (keepdims) {
          for (var i = 0; i < result.shape.length; i++) {
            if (!targetAxes.contains(i)) {
              destOffset += currentCoords[i] * result.strides[i];
            }
          }
        } else {
          for (var i = 0; i < result.shape.length; i++) {
            destOffset += coords[i] * result.strides[i];
          }
        }
        if (targetDType == DType.float32) {
          (result.pointer.cast<ffi.Float>() + destOffset).value = val;
        } else {
          (result.pointer.cast<ffi.Double>() + destOffset).value = val;
        }
        return;
      }

      final limit = stackShape[dim];
      for (var i = 0; i < limit; i++) {
        coords[dim] = i;
        walkStack(dim + 1, coords);
      }
    }

    walkStack(0, List<int>.filled(stackShape.length, 0));

    if (out == null) {
      return result.detachToParentScope();
    }
    return result;
  });
}

double _vectorNorm<T extends DTypeTag>(
  NDArray<T> a,
  Object? ord,
  DType<DTypeTag> targetDType,
) {
  if (ord is NormKind) {
    ord = switch (ord) {
      NormKind.l1 => 1,
      NormKind.negL1 => -1,
      NormKind.l2 => 2,
      NormKind.negL2 => -2,
      NormKind.infinity => double.infinity,
      NormKind.negInfinity => double.negativeInfinity,
      NormKind.frobenius || NormKind.nuclear => throw ArgumentError.value(
        ord,
        'ord',
        'Must be a valid vector norm (NormKind.${ord.name} is not valid for vectors)',
      ),
    };
  }
  final bool needsCast =
      (!a.dtype.isFloating && !a.dtype.isComplex) ||
      a.dtype == DType.float16 ||
      a.dtype == DType.bfloat16;
  final castedA = needsCast ? castNDArray(a, targetDType) : a;

  final size = castedA.size;
  final stride = castedA.strides.isEmpty ? 1 : castedA.strides[0];

  try {
    if (ord == null || ord == 2) {
      double sum;
      if (targetDType == DType.float32) {
        if (castedA.dtype.isComplex) {
          sum = r_norm_l2_complex64(castedA.pointer.cast(), stride, size);
        } else {
          sum = r_norm_l2_float(castedA.pointer.cast(), stride, size);
        }
      } else {
        if (castedA.dtype.isComplex) {
          sum = r_norm_l2_complex128(castedA.pointer.cast(), stride, size);
        } else {
          sum = r_norm_l2_double(castedA.pointer.cast(), stride, size);
        }
      }
      return math.sqrt(sum);
    } else if (ord == 1) {
      if (targetDType == DType.float32) {
        if (castedA.dtype.isComplex) {
          return r_norm_l1_complex64(castedA.pointer.cast(), stride, size);
        } else {
          return r_norm_l1_float(castedA.pointer.cast(), stride, size);
        }
      } else {
        if (castedA.dtype.isComplex) {
          return r_norm_l1_complex128(castedA.pointer.cast(), stride, size);
        } else {
          return r_norm_l1_double(castedA.pointer.cast(), stride, size);
        }
      }
    } else if (ord == double.infinity) {
      if (targetDType == DType.float32) {
        if (castedA.dtype.isComplex) {
          return r_norm_inf_complex64(castedA.pointer.cast(), stride, size);
        } else {
          return r_norm_inf_float(castedA.pointer.cast(), stride, size);
        }
      } else {
        if (castedA.dtype.isComplex) {
          return r_norm_inf_complex128(castedA.pointer.cast(), stride, size);
        } else {
          return r_norm_inf_double(castedA.pointer.cast(), stride, size);
        }
      }
    } else if (ord == double.negativeInfinity) {
      if (targetDType == DType.float32) {
        if (castedA.dtype.isComplex) {
          return r_norm_neg_inf_complex64(castedA.pointer.cast(), stride, size);
        } else {
          return r_norm_neg_inf_float(castedA.pointer.cast(), stride, size);
        }
      } else {
        if (castedA.dtype.isComplex) {
          return r_norm_neg_inf_complex128(
            castedA.pointer.cast(),
            stride,
            size,
          );
        } else {
          return r_norm_neg_inf_double(castedA.pointer.cast(), stride, size);
        }
      }
    } else if (ord == 0) {
      var count = 0;
      for (var i = 0; i < size; i++) {
        final val = castedA.getCell([i]);
        if (castedA.dtype.isComplex) {
          final c = val as Complex;
          if (c.real != 0.0 || c.imag != 0.0) count++;
        } else {
          if ((val as num) != 0) count++;
        }
      }
      return count.toDouble();
    } else if (ord is num) {
      double sum;
      final p = ord.toDouble();
      if (targetDType == DType.float32) {
        if (castedA.dtype.isComplex) {
          sum = r_norm_lp_complex64(castedA.pointer.cast(), stride, size, p);
        } else {
          sum = r_norm_lp_float(castedA.pointer.cast(), stride, size, p);
        }
      } else {
        if (castedA.dtype.isComplex) {
          sum = r_norm_lp_complex128(castedA.pointer.cast(), stride, size, p);
        } else {
          sum = r_norm_lp_double(castedA.pointer.cast(), stride, size, p);
        }
      }
      return math.pow(sum, 1.0 / p).toDouble();
    } else {
      throw ArgumentError.value(
        ord,
        'ord',
        'Must be a valid vector norm order (invalid vector norm order: $ord)',
      );
    }
  } finally {
    if (needsCast) castedA.dispose();
  }
}

double _matrixNorm<T extends DTypeTag>(
  NDArray<T> a,
  Object? ord,
  DType<DTypeTag> targetDType,
) {
  if (ord is String) {
    if (ord == 'fro' || ord == 'frobenius') {
      ord = NormKind.frobenius;
    } else if (ord == 'nuc' || ord == 'nuclear') {
      ord = NormKind.nuclear;
    }
  }
  if (ord is NormKind) {
    ord = switch (ord) {
      NormKind.frobenius => NormKind.frobenius,
      NormKind.nuclear => NormKind.nuclear,
      NormKind.l1 => 1,
      NormKind.negL1 => -1,
      NormKind.l2 => 2,
      NormKind.negL2 => -2,
      NormKind.infinity => double.infinity,
      NormKind.negInfinity => double.negativeInfinity,
    };
  }
  final rows = a.shape[0];
  final cols = a.shape[1];

  if (ord == null || ord == NormKind.frobenius) {
    final flat = a.ravel();
    final res = _vectorNorm(flat, 2, targetDType);
    flat.dispose();
    return res;
  } else if (ord == 1) {
    var maxColSum = 0.0;
    for (var c = 0; c < cols; c++) {
      final colSlice = NDArray.view(
        a,
        shape: [rows],
        strides: [a.strides[0]],
        offsetElements: c * a.strides[1],
      );
      final colSum = _vectorNorm(colSlice, 1, targetDType);
      colSlice.dispose();
      if (colSum > maxColSum) maxColSum = colSum;
    }
    return maxColSum;
  } else if (ord == -1) {
    var minColSum = double.infinity;
    for (var c = 0; c < cols; c++) {
      final colSlice = NDArray.view(
        a,
        shape: [rows],
        strides: [a.strides[0]],
        offsetElements: c * a.strides[1],
      );
      final colSum = _vectorNorm(colSlice, 1, targetDType);
      colSlice.dispose();
      if (colSum < minColSum) minColSum = colSum;
    }
    return minColSum;
  } else if (ord == double.infinity) {
    var maxRowSum = 0.0;
    for (var r = 0; r < rows; r++) {
      final rowSlice = NDArray.view(
        a,
        shape: [cols],
        strides: [a.strides[1]],
        offsetElements: r * a.strides[0],
      );
      final rowSum = _vectorNorm(rowSlice, 1, targetDType);
      rowSlice.dispose();
      if (rowSum > maxRowSum) maxRowSum = rowSum;
    }
    return maxRowSum;
  } else if (ord == double.negativeInfinity) {
    var minRowSum = double.infinity;
    for (var r = 0; r < rows; r++) {
      final rowSlice = NDArray.view(
        a,
        shape: [cols],
        strides: [a.strides[1]],
        offsetElements: r * a.strides[0],
      );
      final rowSum = _vectorNorm(rowSlice, 1, targetDType);
      rowSlice.dispose();
      if (rowSum < minRowSum) minRowSum = rowSum;
    }
    return minRowSum;
  } else if (ord == 2) {
    final s = _svdVals(a);
    if (s.shape[0] == 0) {
      s.dispose();
      return 0.0;
    }
    final maxS = (s.dtype == DType.float32)
        ? s.pointer.cast<ffi.Float>()[0]
        : s.pointer.cast<ffi.Double>()[0];
    s.dispose();
    return maxS;
  } else if (ord == -2) {
    final s = _svdVals(a);
    if (s.shape[0] == 0) {
      s.dispose();
      return 0.0;
    }
    final minS = (s.dtype == DType.float32)
        ? s.pointer.cast<ffi.Float>()[s.shape[0] - 1]
        : s.pointer.cast<ffi.Double>()[s.shape[0] - 1];
    s.dispose();
    return minS;
  } else if (ord == NormKind.nuclear) {
    final s = _svdVals(a);
    final sIsF32 = s.dtype == DType.float32;
    var sumS = 0.0;
    for (var i = 0; i < s.shape[0]; i++) {
      sumS += sIsF32
          ? s.pointer.cast<ffi.Float>()[i]
          : s.pointer.cast<ffi.Double>()[i];
    }
    s.dispose();
    return sumS;
  } else {
    throw ArgumentError.value(
      ord,
      'ord',
      'Must be a valid matrix norm order (invalid matrix norm order: $ord)',
    );
  }
}

extension QRRecordDispose<T extends DTypeTag>
    on ({NDArray<T> q, NDArray<T> r}) {
  void dispose() {
    this.q.dispose();
    this.r.dispose();
  }
}

extension SVDRecordDispose<T extends DTypeTag, S extends DTypeTag>
    on ({NDArray<T> u, NDArray<S> s, NDArray<T> vh}) {
  void dispose() {
    this.u.dispose();
    this.s.dispose();
    this.vh.dispose();
  }
}

extension SchurRecordDispose<T extends DTypeTag>
    on ({NDArray<T> t, NDArray<T> z}) {
  void dispose() {
    this.t.dispose();
    this.z.dispose();
  }
}

extension HessenbergRecordDispose<T extends DTypeTag>
    on ({NDArray<T> h, NDArray<T> q}) {
  void dispose() {
    this.h.dispose();
    this.q.dispose();
  }
}

/// Result record of a least-squares linear system solution from [lstsq].
typedef LstsqResult<T extends DTypeTag> = ({
  NDArray<T> x,
  NDArray<DTypeTag> residuals,
  int rank,
  NDArray<DTypeTag> s,
});

/// Extension on [LstsqResult] to support easy disposal of all returned unmanaged buffers.
extension LstsqResultDispose<T extends DTypeTag> on LstsqResult<T> {
  /// Disposes [x], [residuals], and [s] arrays simultaneously.
  void dispose() {
    this.x.dispose();
    this.residuals.dispose();
    this.s.dispose();
  }
}

/// Computes the least-squares solution to a linear matrix equation $a x = b$.
///
/// Solves the equation $a x = b$ by computing a vector/matrix $x$ that minimizes the
/// Euclidean 2-norm $\|b - a x\|_2^2$.
///
/// Natively offloads to LAPACK divide-and-conquer SVD-based least-squares solvers
/// (`dgelsd`, `sgelsd`, `zgelsd`, `cgelsd`) depending on precision.
///
/// The optional parameter [rcond] acts as the cut-off ratio for small singular values.
/// Singular values smaller than `rcond * largest_singular_value` are treated as zero.
/// If [rcond] is omitted or null, a negative value is passed to the LAPACK solver,
/// which falls back to using the machine precision to determine the effective rank.
///
/// The optional recycler parameter [out] allows reusing an existing array for the output,
/// avoiding new memory allocation.
///
/// **Preconditions:**
/// - It is an error if [a] or [b] is disposed.
/// - It is an error if [a] is not 2D, or [b] is not 1D or 2D.
/// - It is an error if [b]'s first dimension does not match [a]'s first dimension.
/// - It is an error if [a] or [b] has unsupported dtype (requires floating-point or complex).
/// - It is an error if [out] is provided and disposed, or has incompatible shape or dtype.
///
/// **Throws:**
/// - [IterationsExceededException] if the SVD algorithm in LAPACK fails to converge.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(M N \min(M, N))$ operations executed natively.
///
/// **Example:**
/// {@example /example/linalg_lstsq_example.dart lang=dart}
///
/// Reference: [NumPy linalg.lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
LstsqResult<R> lstsq<
  Ta extends DTypeTag,
  Tb extends DTypeTag,
  R extends DTypeTag
>(NDArray<Ta> a, NDArray<Tb> b, {double? rcond, NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute lstsq() on a disposed array.');
  }

  DType rawTargetDType = (a.dtype.isInteger && b.dtype.isInteger)
      ? DType.float64
      : (a.dtype.isInteger
            ? (b.dtype == DType.float32 ? DType.float32 : DType.float64)
            : (b.dtype.isInteger
                  ? (a.dtype == DType.float32 ? DType.float32 : DType.float64)
                  : resolveDType(a.dtype, b.dtype)));
  final origResolvedDType = rawTargetDType;
  final bool promotedHalf =
      rawTargetDType == DType.float16 || rawTargetDType == DType.bfloat16;
  if (promotedHalf) {
    rawTargetDType = DType.float64;
  }
  final targetDType = rawTargetDType;

  if (!targetDType.isFloating && !targetDType.isComplex) {
    throw ArgumentError.value(
      targetDType,
      'dtype',
      'Must be floating-point or complex (lstsq requires floating-point or complex inputs).',
    );
  }

  if (a.shape.length != 2) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must be 2-dimensional (input matrix a must be 2-dimensional, was shape ${a.shape}).',
    );
  }
  if (b.shape.length != 1 && b.shape.length != 2) {
    throw ArgumentError.value(
      b.shape,
      'b.shape',
      'Must be 1D or 2D (input right-hand side b must be 1D or 2D, was shape ${b.shape}).',
    );
  }
  final m = a.shape[0];
  final n = a.shape[1];
  if (b.shape[0] != m) {
    throw ArgumentError.value(
      b.shape[0],
      'b.shape[0]',
      'Must match first dimension of a ($m) (first dimension of b (${b.shape[0]}) must match first dimension of a ($m)).',
    );
  }

  final nrhs = b.shape.length > 1 ? b.shape[1] : 1;
  checkBlasIntDim(m, 'm', 'lstsq');
  checkBlasIntDim(n, 'n', 'lstsq');
  checkBlasIntDim(nrhs, 'nrhs', 'lstsq');
  checkBlasIntStride(a.strides[0], 'lda', 'lstsq');
  checkBlasIntStride(a.strides[1], 'lda', 'lstsq');
  checkBlasIntStride(b.strides[0], 'ldb', 'lstsq');
  if (b.shape.length > 1) {
    checkBlasIntStride(b.strides[1], 'ldb', 'lstsq');
  }

  if (out != null) {
    if (out.isDisposed) {
      throw StateError('Cannot write to a disposed out buffer.');
    }
    validateOutBuffer(out);
    final expectedXShape = b.shape.length > 1 ? [n, nrhs] : [n];
    if (!listEquals(out.shape, expectedXShape) ||
        (out.dtype != targetDType &&
            (!promotedHalf || out.dtype != origResolvedDType))) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape $expectedXShape and dtype $targetDType (incompatible out buffer shape or dtype).',
      );
    }
  }

  return NDArray.scope(() {
    if (m == 0 || n == 0) {
      final xShape = b.shape.length > 1 ? [n, nrhs] : [n];
      final NDArray<R> x;
      if (out != null) {
        if (out.size > 0) {
          out.fill((out.dtype.isComplex ? const Complex(0, 0) : 0.0));
        }
        x = out;
      } else {
        final DType<R> outXDType = R == Float16
            ? DType.float16 as DType<R>
            : R == BFloat16
            ? DType.bfloat16 as DType<R>
            : targetDType as DType<R>;
        x = NDArray<R>.zeros(xShape, outXDType);
        x.detachToParentScope();
      }
      final DType<DTypeTag> sDType =
          (targetDType == DType.complex64 || targetDType == DType.float32)
          ? DType.float32
          : DType.float64;
      final s = _zerosTyped([0], sDType);
      final residuals = _zerosTyped([0], sDType);
      s.detachToParentScope();
      residuals.detachToParentScope();
      return (x: x, residuals: residuals, rank: 0, s: s);
    }

    final aUse = a.dtype == targetDType ? a : castNDArray(a, targetDType);
    final bUse = b.dtype == targetDType ? b : castNDArray(b, targetDType);
    final wasACast = !identical(aUse, a);
    final wasBCast = !identical(bUse, b);

    // Create a contiguous copy of a (overwrite-safe)
    final aCopy = aUse.copy();

    // Row-major LAPACKE_gelsd requires b array size to be max(m, n) * nrhs
    final maxMN = m > n ? m : n;
    final bCopyShape = bUse.shape.length > 1 ? [maxMN, nrhs] : [maxMN];
    final bCopy = NDArray.zeros(bCopyShape, targetDType);

    // Copy b into bCopy
    final byteCount = bUse.size * targetDType.byteWidth;
    if (bUse.isContiguous) {
      custom_memcpy(bCopy.pointer, bUse.pointer, byteCount);
    } else {
      final bContig = bUse.copy();
      custom_memcpy(bCopy.pointer, bContig.pointer, byteCount);
      bContig.dispose();
    }

    final minMN = m < n ? m : n;
    // Singular values s is always real
    final DType<DTypeTag> sDType =
        (targetDType == DType.complex64 || targetDType == DType.float32)
        ? DType.float32
        : DType.float64;
    final s = _zerosTyped([minMN], sDType);
    final marker = ScratchArena.marker;
    try {
      final nfA = _analyzeNonFinitePtr(aCopy.pointer, m * n, targetDType);
      final nfB = _analyzeNonFinitePtr(bCopy.pointer, m * nrhs, targetDType);
      if (nfA.hasNaN || nfA.hasInf || nfB.hasNaN || nfB.hasInf) {
        throw const IterationsExceededException(
          'SVD did not converge in Linear Least Squares (input contains non-finite values).',
        );
      }
      if (rcond != null && rcond.isNaN) {
        throw const LinAlgException('rcond must not be NaN in lstsq.');
      }

      final rankPtr = ScratchArena.allocate<ffi.Int>(ffi.sizeOf<ffi.Int>());
      final rcondVal = rcond ?? -1.0;
      final int info;
      final String routine;
      switch (targetDType) {
        case DType.float64:
          routine = 'LAPACKE_dgelsd';
          info = LAPACKE_dgelsd(
            101, // ROW_MAJOR
            m,
            n,
            nrhs,
            aCopy.pointer.cast<ffi.Double>(),
            n,
            bCopy.pointer.cast<ffi.Double>(),
            nrhs,
            s.pointer.cast<ffi.Double>(),
            rcondVal,
            rankPtr,
          );
        case DType.float32:
          routine = 'LAPACKE_sgelsd';
          info = LAPACKE_sgelsd(
            101, // ROW_MAJOR
            m,
            n,
            nrhs,
            aCopy.pointer.cast<ffi.Float>(),
            n,
            bCopy.pointer.cast<ffi.Float>(),
            nrhs,
            s.pointer.cast<ffi.Float>(),
            rcondVal,
            rankPtr,
          );
        case DType.complex128:
          routine = 'LAPACKE_zgelsd';
          info = LAPACKE_zgelsd(
            101, // ROW_MAJOR
            m,
            n,
            nrhs,
            aCopy.pointer.cast<ffi.Double>(),
            n,
            bCopy.pointer.cast<ffi.Double>(),
            nrhs,
            s.pointer.cast<ffi.Double>(),
            rcondVal,
            rankPtr,
          );
        case DType.complex64:
          routine = 'LAPACKE_cgelsd';
          info = LAPACKE_cgelsd(
            101, // ROW_MAJOR
            m,
            n,
            nrhs,
            aCopy.pointer.cast<ffi.Float>(),
            n,
            bCopy.pointer.cast<ffi.Float>(),
            nrhs,
            s.pointer.cast<ffi.Float>(),
            rcondVal,
            rankPtr,
          );
        default:
          throw UnimplementedError(
            'Unsupported target DType for lstsq: $targetDType',
          );
      }

      _checkLapackInfo(
        info,
        routine,
        positiveKind: _LapackFailureKind.iterationsExceeded,
        positiveMessage:
            'The SVD algorithm in $routine failed to converge ($info).',
      );

      final rank = rankPtr.cast<ffi.Int32>()[0];

      // Extract solution x: first n rows of bCopy
      final xShape = bUse.shape.length > 1 ? [n, nrhs] : [n];
      final bCopySlice = NDArray.view(
        bCopy,
        shape: xShape,
        strides: bCopy.strides.sublist(bCopy.shape.length - xShape.length),
        offsetElements: 0,
      );
      final NDArray<R> x;
      if (out != null) {
        if (out.dtype == targetDType) {
          bCopySlice.copy(out: out);
        } else {
          castNDArray(bCopySlice, out.dtype).copy(out: out);
        }
        x = out;
      } else if (R == Float16 && targetDType != DType.float16) {
        x = castNDArray<Float16>(bCopySlice, DType.float16) as NDArray<R>;
      } else if (R == BFloat16 && targetDType != DType.bfloat16) {
        x = castNDArray<BFloat16>(bCopySlice, DType.bfloat16) as NDArray<R>;
      } else {
        x = NDArray<R>.zeros(xShape, targetDType as DType<R>);
        bCopySlice.copy(out: x);
      }
      bCopySlice.dispose();

      // Extract residuals: sum of squares of elements from row n to m-1 for each column
      final NDArray<DTypeTag> residuals;
      if (m > n && rank == n) {
        final resShape = bUse.shape.length > 1 ? [nrhs] : [1];
        residuals = _zerosTyped(resShape, sDType);
        if (targetDType == DType.complex128) {
          final bPtr = bCopy.pointer.cast<ffi.Double>();
          final resPtr = residuals.pointer.cast<ffi.Double>();
          for (var j = 0; j < nrhs; j++) {
            var sum = 0.0;
            for (var i = n; i < m; i++) {
              final real = bPtr[(i * nrhs + j) * 2];
              final imag = bPtr[(i * nrhs + j) * 2 + 1];
              sum += real * real + imag * imag;
            }
            resPtr[j] = sum;
          }
        } else if (targetDType == DType.complex64) {
          final bPtr = bCopy.pointer.cast<ffi.Float>();
          final resPtr = residuals.pointer.cast<ffi.Float>();
          for (var j = 0; j < nrhs; j++) {
            var sum = 0.0;
            for (var i = n; i < m; i++) {
              final real = bPtr[(i * nrhs + j) * 2];
              final imag = bPtr[(i * nrhs + j) * 2 + 1];
              sum += real * real + imag * imag;
            }
            resPtr[j] = sum;
          }
        } else if (targetDType == DType.float32) {
          final bPtr = bCopy.pointer.cast<ffi.Float>();
          final resPtr = residuals.pointer.cast<ffi.Float>();
          for (var j = 0; j < nrhs; j++) {
            var sum = 0.0;
            for (var i = n; i < m; i++) {
              final val = bPtr[i * nrhs + j];
              sum += val * val;
            }
            resPtr[j] = sum;
          }
        } else {
          final bPtr = bCopy.pointer.cast<ffi.Double>();
          final resPtr = residuals.pointer.cast<ffi.Double>();
          for (var j = 0; j < nrhs; j++) {
            var sum = 0.0;
            for (var i = n; i < m; i++) {
              final val = bPtr[i * nrhs + j];
              sum += val * val;
            }
            resPtr[j] = sum;
          }
        }
      } else {
        residuals = _zerosTyped([0], sDType);
      }

      if (out == null) {
        x.detachToParentScope();
      }
      residuals.detachToParentScope();
      s.detachToParentScope();
      return (x: x, residuals: residuals, rank: rank, s: s);
    } finally {
      ScratchArena.reset(marker);
      aCopy.dispose();
      bCopy.dispose();
      if (wasACast) aUse.dispose();
      if (wasBCast) bUse.dispose();
    }
  });
}

/// Computes the condition number of a matrix.
///
/// The condition number of [a] is defined as the norm of [a] times the norm of the
/// inverse of [a] ($\|a\|_p \cdot \|a^{-1}\|_p$); the norm can be the usual L2-norm
/// (root-of-sum-of-squares) or one of a number of other matrix norms specified by [p].
///
/// Supported values for [p]:
/// - `null` or `2` or [NormKind.l2]: 2-norm (largest singular value divided by smallest singular value).
/// - `-2` or [NormKind.negL2]: smallest singular value divided by largest singular value.
/// - `1` or [NormKind.l1]: 1-norm (maximum column sum).
/// - `-1` or [NormKind.negL1]: minimum column sum.
/// - `double.infinity` or [NormKind.infinity]: infinity-norm (maximum row sum).
/// - `-double.infinity` (`double.negativeInfinity`) or [NormKind.negInfinity]: minimum row sum.
/// - `'fro'` or [NormKind.frobenius]: Frobenius norm.
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] has rank less than 2 (`a.rank < 2`).
/// - It is an error if [a] has any zero dimension in the last two axes (empty matrix).
/// - It is an error if [p] is not `null`, `2`, or `-2` and [a] is not square in its last two dimensions.
/// - It is an error if [p] is an unsupported norm order.
/// - It is an error if [out] is provided and has incompatible shape or dtype.
///
/// **Throws:**
/// - Throws a [LinAlgException] (such as [IterationsExceededException]) when [p] is `null`, `2`, or `-2` and the SVD computation does not converge (e.g. if [a] contains `NaN`).
///
/// **Performance considerations:**
/// - When [p] is `null`, `2`, or `-2`, uses Singular Value Decomposition ([svd]) with complexity $O(M N \min(M, N))$.
/// - For other norms, computes matrix inverse ([inv]) and matrix norms ([norm]) with complexity $O(N^3)$.
///
/// **Example:**
/// {@example /example/linalg_advanced_example.dart lang=dart}
///
/// Reference: [NumPy linalg.cond](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html)
NDArray<R> cond<R extends DTypeTag>(
  NDArray<
    DTypeSpec<
      DTypeTag,
      Object?,
      R,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag
    >
  >
  a, {
  Object? p,
  NDArray<R>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute cond() on a disposed array.');
  }
  final rank = a.rank;
  if (rank < 2) {
    throw ArgumentError.value(
      rank,
      'a.rank',
      'Must be at least two-dimensional (array must be at least two-dimensional, got rank $rank).',
    );
  }

  var ord = p;
  if (ord is String) {
    if (ord == 'fro' || ord == 'frobenius') {
      ord = NormKind.frobenius;
    }
  } else if (ord is NormKind) {
    ord = switch (ord) {
      NormKind.frobenius => NormKind.frobenius,
      NormKind.l1 => 1,
      NormKind.negL1 => -1,
      NormKind.l2 => 2,
      NormKind.negL2 => -2,
      NormKind.infinity => double.infinity,
      NormKind.negInfinity => double.negativeInfinity,
      NormKind.nuclear => ord,
    };
  }

  final isSvdNorm = ord == null || ord == 2 || ord == -2;
  final isInvNorm =
      ord == 1 ||
      ord == -1 ||
      ord == double.infinity ||
      ord == double.negativeInfinity ||
      ord == NormKind.frobenius;

  if (!isSvdNorm && !isInvNorm) {
    throw ArgumentError.value(
      p,
      'p',
      'Must be a valid norm order for cond (invalid norm order for cond: $p)',
    );
  }

  final m = a.shape[rank - 2];
  final n = a.shape[rank - 1];

  if (!isSvdNorm && m != n) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must be square for p = $p (matrix must be square for p = $p, got shape ${a.shape}).',
    );
  }

  final k = math.min(m, n);
  if (k == 0) {
    throw ArgumentError.value(
      a.shape,
      'a.shape',
      'Must not be an empty matrix (cannot compute condition number of an empty matrix).',
    );
  }
  checkBlasIntDim(m, 'm', 'cond');
  checkBlasIntDim(n, 'n', 'cond');
  checkBlasIntStride(a.strides[rank - 2], 'lda', 'cond');
  checkBlasIntStride(a.strides[rank - 1], 'lda', 'cond');

  final DType<DTypeTag> resDType = switch (a.dtype) {
    DType.float32 || DType.complex64 => DType.float32,
    _ => DType.float64,
  };

  final stackShape = a.shape.sublist(0, rank - 2);

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, stackShape) || out.dtype != resDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape $stackShape and dtype $resDType (provided out buffer has incompatible shape or dtype, expected shape $stackShape and dtype $resDType, got shape ${out.shape} and dtype ${out.dtype}).',
      );
    }
  }

  return NDArray.scope(() {
    final NDArray<DTypeTag> aUse = switch (a.dtype) {
      DType.float64 ||
      DType.float32 ||
      DType.complex64 ||
      DType.complex128 => a,
      _ => castNDArray<Float64>(a, DType.float64),
    };

    final bool aliased =
        out != null && (!out.isContiguous || sharesMemory(a, out));
    final NDArray<R> result = (out != null && !aliased)
        ? out
        : (_createZeros(stackShape, resDType) as NDArray<R>);

    walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
      coords,
    ) {
      var offsetA = 0;
      var offsetRes = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetA += coords[i] * aUse.strides[i];
        offsetRes += coords[i] * result.strides[i];
      }
      final aSlice = stackShape.isEmpty
          ? aUse
          : NDArray<DTypeTag>.view(
              aUse,
              shape: [m, n],
              strides: aUse.strides.sublist(rank - 2),
              offsetElements: offsetA,
            );
      double val;
      try {
        final nf = _analyzeNonFinite(aSlice);
        if (isSvdNorm) {
          final s = _svdVals(aSlice);
          final strideK = s.strides[0];
          const offsetSMax = 0;
          final offsetSMin = (k - 1) * strideK;
          final double sMax;
          final double sMin;
          switch (resDType) {
            case DType.float32:
              final sPtr = s.pointer.cast<ffi.Float>();
              sMax = sPtr[offsetSMax];
              sMin = sPtr[offsetSMin];
            case DType.float64:
              final sPtr = s.pointer.cast<ffi.Double>();
              sMax = sPtr[offsetSMax];
              sMin = sPtr[offsetSMin];
            default:
              s.dispose();
              throw UnimplementedError('Unexpected dtype: $resDType');
          }
          s.dispose();
          if (ord == -2) {
            val = sMin / sMax;
          } else {
            val = sMax / sMin;
          }
        } else {
          if (nf.hasNaN) {
            val = double.nan;
          } else {
            final normAVal =
                norm<DTypeTag>(aSlice as NDArray<AnySpec>, ord: ord).scalar
                    as double;
            try {
              final invSliceA = inv<DTypeTag>(aSlice);
              final normInvAVal =
                  norm<DTypeTag>(invSliceA as NDArray<AnySpec>, ord: ord).scalar
                      as double;
              invSliceA.dispose();
              val = normAVal * normInvAVal;
            } on SingularMatrixException {
              val = double.infinity;
            }
          }
        }
        // Match NumPy: convert NaNs (e.g. 0/0 on zero matrix or inf*0 on ±inf)
        // to +infinity unless the original matrix slice contained NaN entries.
        if (val.isNaN && !nf.hasNaN) {
          val = double.infinity;
        }
      } finally {
        if (stackShape.isNotEmpty) {
          aSlice.dispose();
        }
      }
      switch (resDType) {
        case DType.float32:
          (result.pointer.cast<ffi.Float>() + offsetRes).value = val;
        case DType.float64:
          (result.pointer.cast<ffi.Double>() + offsetRes).value = val;
        default:
          throw UnimplementedError('Unexpected dtype: $resDType');
      }
    });

    if (out != null) {
      if (aliased) {
        result.copy(out: out);
      }
      return out;
    }
    return result.detachToParentScope();
  });
}
