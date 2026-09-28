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
import '../ndarray_extensions_bindings.dart';
import '../ndarray_bindings.dart'
    hide
        s_det_double,
        s_det_float,
        s_det_complex_double,
        s_det_complex_float,
        s_slogdet_double,
        s_slogdet_float,
        s_slogdet_complex_double,
        s_slogdet_complex_float;

// Standalone operational relative cross-imports
import 'math.dart';
import 'helpers.dart';

/// Matrix multiplication using OpenBLAS, supporting high-dimensional stack broadcasting and 1D vector promotions.
NDArray<R> matmul<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute matmul() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write matmul result to a disposed output array.');
  }
  final targetDType = resolveDType(a.dtype, b.dtype);

  NDArray? aCast;
  NDArray? bCast;
  NDArray? aCopy;
  NDArray? bCopy;
  NDArray<R>? result;
  var success = false;

  try {
    aCast = a.dtype == targetDType ? a : castNDArray(a, targetDType);
    bCast = b.dtype == targetDType ? b : castNDArray(b, targetDType);

    if (aCast.shape.length == 1 && bCast.shape.length == 1) {
      final n = aCast.shape[0];
      if (n != bCast.shape[0]) {
        throw ArgumentError(
          'Incompatible vector dimensions for 1D dot product in matmul: ${aCast.shape} and ${bCast.shape}',
        );
      }
      if (out != null) {
        if (!listEquals(out.shape, []) || out.dtype != targetDType) {
          throw ArgumentError(
            'Provided out buffer has incompatible shape or dtype (expected shape [] and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).',
          );
        }
      }
      final incA = aCast.strides[0];
      final incB = bCast.strides[0];
      switch (targetDType) {
        case DType.float64:
          final scalarRes = cblas_ddot(
            n,
            aCast.pointer.cast<ffi.Double>(),
            incA,
            bCast.pointer.cast<ffi.Double>(),
            incB,
          );
          if (out != null) {
            out.pointer.cast<ffi.Double>()[0] = scalarRes;
            result = out;
          } else {
            result =
                (NDArray.scalar(scalarRes, dtype: DType.float64) as NDArray<R>);
          }
          success = true;
          return result;
        case DType.float32:
          final scalarRes = cblas_sdot(
            n,
            aCast.pointer.cast<ffi.Float>(),
            incA,
            bCast.pointer.cast<ffi.Float>(),
            incB,
          );
          if (out != null) {
            out.pointer.cast<ffi.Float>()[0] = scalarRes;
            result = out;
          } else {
            result =
                (NDArray.scalar(scalarRes, dtype: DType.float32) as NDArray<R>);
          }
          success = true;
          return result;
        case DType.complex128:
          final aPtr = aCast.pointer.cast<ffi.Double>();
          final bPtr = bCast.pointer.cast<ffi.Double>();
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
                (NDArray.scalar(resVal, dtype: DType.complex128) as NDArray<R>);
          }
          success = true;
          return result;
        case DType.complex64:
          final aPtr = aCast.pointer.cast<ffi.Float>();
          final bPtr = bCast.pointer.cast<ffi.Float>();
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
                (NDArray.scalar(resVal, dtype: DType.complex64) as NDArray<R>);
          }
          success = true;
          return result;
        default:
          break;
      }
    }

    // Copy upfront ONLY if neither inner strides is 1 (very rare custom sliced strides)
    if (aCast.shape.length >= 2) {
      final r = aCast.shape.length;
      if (aCast.strides[r - 1] != 1 && aCast.strides[r - 2] != 1) {
        aCopy = aCast.copy();
      }
    }
    if (bCast.shape.length >= 2) {
      final r = bCast.shape.length;
      if (bCast.strides[r - 1] != 1 && bCast.strides[r - 2] != 1) {
        bCopy = bCast.copy();
      }
    }

    final aToUse = aCopy ?? aCast;
    final bToUse = bCopy ?? bCast;

    var aPromoted = false;
    var bPromoted = false;

    NDArray aView = aToUse;
    if (aToUse.shape.length == 1) {
      aView = NDArray.view(
        aToUse,
        shape: [1, aToUse.shape[0]],
        strides: [0, aToUse.strides[0]],
        offsetElements: 0,
      );
      aPromoted = true;
    }

    NDArray bView = bToUse;
    if (bToUse.shape.length == 1) {
      bView = NDArray.view(
        bToUse,
        shape: [bToUse.shape[0], 1],
        strides: [bToUse.strides[0], 0],
        offsetElements: 0,
      );
      bPromoted = true;
    }

    final rankA = aView.shape.length;
    final rankB = bView.shape.length;

    final m = aView.shape[rankA - 2];
    final kA = aView.shape[rankA - 1];
    final kB = bView.shape[rankB - 2];
    final n = bView.shape[rankB - 1];

    if (kA != kB) {
      throw ArgumentError(
        'Incompatible inner matrix dimensions for matmul: kA($kA) != kB($kB). Shapes: ${aCast.shape} and ${bCast.shape}',
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
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype (expected shape $expectedFinalShape and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).',
        );
      }
    }

    final resShape = [...broadcastStack, m, n];
    final bool canUseOutDirectly =
        out != null && out.isContiguous && listEquals(out.shape, resShape);
    result = canUseOutDirectly
        ? out
        : NDArray.zeros(resShape, targetDType as DType<R>);

    // Stride resolution logic for 100% copy-free BLAS matrix multiplication
    var transA = 111; // CblasNoTrans
    var lda = kA;
    if (!aPromoted) {
      if (aView.strides[rankA - 1] == 1) {
        transA = 111;
        lda = math.max(aView.strides[rankA - 2], kA);
      } else if (aView.strides[rankA - 2] == 1) {
        transA = 112; // CblasTrans
        lda = math.max(aView.strides[rankA - 1], m);
      }
    }

    var transB = 111; // CblasNoTrans
    var ldb = n;
    if (!bPromoted) {
      if (bView.strides[rankB - 1] == 1) {
        transB = 111;
        ldb = math.max(bView.strides[rankB - 2], n);
      } else if (bView.strides[rankB - 2] == 1) {
        transB = 112; // CblasTrans
        ldb = math.max(bView.strides[rankB - 1], kB);
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
            case DType.int64:
            case DType.int32:
            case DType.int16:
            case DType.uint8:
              final strideARow = aView.strides[rankA - 2];
              final strideACol = aView.strides[rankA - 1];

              final strideBRow = bView.strides[rankB - 2];
              final strideBCol = bView.strides[rankB - 1];

              final strideResRow = result!.strides[resShape.length - 2];
              final strideResCol = result.strides[resShape.length - 1];

              switch (targetDType) {
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
        if (out.isContiguous && result.isContiguous) {
          final byteCount = result.size * targetDType.byteWidth;
          ffi.Pointer.fromAddress(out.pointer.address)
              .cast<ffi.Uint8>()
              .asTypedList(byteCount)
              .setAll(
                0,
                ffi.Pointer.fromAddress(result.pointer.address)
                    .cast<ffi.Uint8>()
                    .asTypedList(byteCount),
              );
        } else {
          result.reshape(out.shape).copy(out: out);
        }
        result.dispose();
      }
      success = true;
      return out;
    }

    // Post-calculation 1D dummy dimensions demotions
    if (aPromoted && bPromoted) {
      final finalRes = result.reshape([]);
      success = true;
      return finalRes; // 0D scalar array for pure vector dot products
    } else if (aPromoted) {
      final newShape = List<int>.from(result.shape)
        ..removeAt(result.shape.length - 2);
      final finalRes = result.reshape(newShape);
      success = true;
      return finalRes;
    } else if (bPromoted) {
      final newShape = List<int>.from(result.shape)
        ..removeAt(result.shape.length - 1);
      final finalRes = result.reshape(newShape);
      success = true;
      return finalRes;
    }

    success = true;
    return result;
  } finally {
    if (aCast != null && aCast != a) aCast.dispose();
    if (bCast != null && bCast != b) bCast.dispose();
    aCopy?.dispose();
    bCopy?.dispose();
    if (!success) {
      if (result != null && result != out) {
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
NDArray<T> multi_dot<T>(List<NDArray<Object>> arrays, {NDArray<T>? out}) {
  for (final a in arrays) {
    if (a.isDisposed) {
      throw StateError(
        'Cannot execute multi_dot() with a disposed array in the list.',
      );
    }
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write multi_dot result to a disposed output array.',
    );
  }
  if (arrays.length < 2) {
    throw ArgumentError(
      'multi_dot requires at least 2 arrays (got ${arrays.length}).',
    );
  }

  final n = arrays.length;

  // Check dimensions & validate rank preconditions
  for (var i = 0; i < n; i++) {
    final rank = arrays[i].shape.length;
    if (i == 0 || i == n - 1) {
      if (rank != 1 && rank != 2) {
        throw ArgumentError(
          'First and last arrays in multi_dot must be 1D or 2D (array $i was shape ${arrays[i].shape}).',
        );
      }
    } else {
      if (rank != 2) {
        throw ArgumentError(
          'All intermediate arrays in multi_dot must be 2D (array $i was shape ${arrays[i].shape}).',
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
      throw ArgumentError(
        'Incompatible matrix dimensions in multi_dot: array $i first dimension (${shape[0]}) must match previous dimension (${p[i]}).',
      );
    }
    p[i + 1] = shape[1];
  }

  // Last array
  final lastIdx = n - 1;
  final lastShape = arrays[lastIdx].shape;
  if (lastShape[0] != p[lastIdx]) {
    throw ArgumentError(
      'Incompatible matrix dimensions in multi_dot: last array first dimension (${lastShape[0]}) must match previous dimension (${p[lastIdx]}).',
    );
  }
  if (lastShape.length == 1) {
    p[n] = 1;
  } else {
    p[n] = lastShape[1];
  }

  // Resolve target DType and upcasted type
  DType<dynamic> targetDType = arrays[0].dtype;
  for (var i = 1; i < n; i++) {
    targetDType = resolveDType(targetDType, arrays[i].dtype);
  }
  if (!targetDType.isFloating && !targetDType.isComplex) {
    targetDType = DType.float64;
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
      throw ArgumentError(
        'Provided out recycler has incompatible shape or dtype (expected shape $expectedFinalShape and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).',
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
        m[i][j] = 99999999999999; // large number as infinity
        for (var k = i; k < j; k++) {
          final cost = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
          if (cost < m[i][j]) {
            m[i][j] = cost;
            s[i][j] = k;
          }
        }
      }
    }

    // Helper function to recursively evaluate matrix multiplication chain
    NDArray eval(int i, int j) {
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
      final res = matmul(left, right);
      left.dispose();
      right.dispose();
      return res;
    }

    // Top-level split point evaluation
    final k = s[1][n];
    final left = eval(1, k);
    final right = eval(k + 1, n);

    final finalResult = matmul(left, right, out: out);
    left.dispose();
    right.dispose();

    if (out != null) return out;
    return finalResult.detachToParentScope();
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
/// - It is an error if [out] is provided and has incompatible shape or dtype, or is not contiguous.
/// - The matrix must be non-singular (invertible).
///
/// **Throws:**
/// - [ArgumentError] if the matrix is singular (non-invertible) during LU pivoting.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N^3)$ where $N$ is the matrix dimension length.
/// - For non-contiguous views, automatically flattens the matrix first, recycling allocation views
///   where safe to minimize heap churn.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([4.0, 7.0, 2.0, 6.0], [2, 2], DType.float64);
/// final b = inv(a);
/// print(b.toList()); // [0.6, -0.7, -0.2, 0.4]
/// ```
///
/// Reference: [Matrix Inversion](https://en.wikipedia.org/wiki/Invertible_matrix)
NDArray<T> inv<T>(NDArray<T> a, {NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute inverse of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write inverse to a disposed output array.');
  }
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 2] != a.shape[rank - 1]) {
    throw ArgumentError(
      'Matrix must be square in the last 2 dimensions and rank >= 2 (was ${a.shape})',
    );
  }

  if (a.dtype != DType.float32 &&
      a.dtype != DType.float64 &&
      a.dtype != DType.complex64 &&
      a.dtype != DType.complex128) {
    throw ArgumentError(
      'Matrix inversion only supports float or complex dtypes (got ${a.dtype}).',
    );
  }
  final n = a.shape[rank - 1];
  final stackShape = a.shape.sublist(0, rank - 2);
  final DType<T> targetDType = a.dtype;

  if (out != null) {
    if (!out.isContiguous) {
      throw ArgumentError('out buffer must be contiguous.');
    }
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for matrix inversion.',
      );
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
    final ipiv = ScratchArena.allocate<ffi.Int>(n * ffi.sizeOf<ffi.Int>());

    try {
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
            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_sgetrf: $info',
              );
            }
            if (info > 0) {
              throw SingularMatrixException(
                'Matrix is singular and cannot be inverted',
              );
            }
            final infoTri = LAPACKE_sgetri(
              101,
              n,
              sliceRes.pointer.cast<ffi.Float>(),
              n,
              ipiv,
            );
            if (infoTri < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_sgetri: $infoTri',
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
            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_dgetrf: $info',
              );
            }
            if (info > 0) {
              throw SingularMatrixException(
                'Matrix is singular and cannot be inverted',
              );
            }
            final infoTri = LAPACKE_dgetri(
              101,
              n,
              sliceRes.pointer.cast<ffi.Double>(),
              n,
              ipiv,
            );
            if (infoTri < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_dgetri: $infoTri',
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
            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_cgetrf: $info',
              );
            }
            if (info > 0) {
              throw SingularMatrixException(
                'Matrix is singular and cannot be inverted',
              );
            }
            final infoTri = LAPACKE_cgetri(
              101,
              n,
              sliceRes.pointer.cast<ffi.Float>(),
              n,
              ipiv,
            );
            if (infoTri < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_cgetri: $infoTri',
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
            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_zgetrf: $info',
              );
            }
            if (info > 0) {
              throw SingularMatrixException(
                'Matrix is singular and cannot be inverted',
              );
            }
            final infoTri = LAPACKE_zgetri(
              101,
              n,
              sliceRes.pointer.cast<ffi.Double>(),
              n,
              ipiv,
            );
            if (infoTri < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_zgetri: $infoTri',
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
/// - It is an error if [out] is provided and has incompatible shape or dtype, or is not contiguous.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N^3)$ using LAPACK linear algebra solvers.
/// - Fully vectorized and batched in native C for float64, complex64, and complex128, minimizing FFI transitions.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final d = det(a);
/// print(d.scalar); // -2.0 (0-D array)
/// ```
///
/// Refer to the [determinant](https://en.wikipedia.org/wiki/Determinant)
/// and [LAPACK LU solver](https://en.wikipedia.org/wiki/LU_decomposition) for additional details.
///
/// Returns a 0-dimensional [NDArray] if [a] is a 2D matrix, or a new [NDArray] with stack dimensions if [a] is a stack of matrices.
NDArray<T> det<T>(NDArray<T> a, {NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute determinant of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write determinant to a disposed output array.');
  }
  if (a.dtype != DType.float64 &&
      a.dtype != DType.float32 &&
      a.dtype != DType.complex128 &&
      a.dtype != DType.complex64) {
    throw ArgumentError('det only supports float and complex dtypes');
  }
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 1] != a.shape[rank - 2]) {
    throw ArgumentError(
      'Matrix must be square and at least 2D (was ${a.shape})',
    );
  }
  final stackShape = a.shape.sublist(0, rank - 2);
  final expectedDType = a.dtype;

  if (out != null) {
    if (!listEquals(out.shape, stackShape) || out.dtype != expectedDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    if (!out.isContiguous) {
      throw ArgumentError('Provided out buffer must be contiguous.');
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
            get_dgetrf_ptr(),
          );
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
          final cCopy = ScratchArena.allocate<ffi.Double>(
            2 * n * n * ffi.sizeOf<ffi.Double>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );

          s_det_complex_double(
            a.pointer.cast<ffi.Double>(),
            cStridesA,
            result.pointer.cast<ffi.Double>(),
            cStridesRes,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_zgetrf_ptr(),
          );
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
          final cCopy = ScratchArena.allocate<ffi.Float>(
            2 * n * n * ffi.sizeOf<ffi.Float>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );

          s_det_complex_float(
            a.pointer.cast<ffi.Float>(),
            cStridesA,
            result.pointer.cast<ffi.Float>(),
            cStridesRes,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_cgetrf_ptr(),
          );
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
            get_sgetrf_ptr(),
          );
        } finally {
          ScratchArena.reset(marker);
        }
        if (out == null) {
          result.detachToParentScope();
        }
        return result;
      default:
        throw ArgumentError('Unsupported dtype for determinant');
    }
  });
}

/// Computes the sign and natural logarithm of the absolute value of the determinant of a square 2D matrix or stack of matrices.
///
/// **Preconditions:**
/// - It is an error if [a], [outSign], or [outLogdet] is disposed.
/// - It is an error if [a] rank < 2, or the last two dimensions are not square.
/// - It is an error if [a] dtype is not float32, float64, complex64, or complex128.
/// - It is an error if [outSign] or [outLogdet] is provided and has incompatible shape, dtype, or is not contiguous.
///
/// **Returns:**
/// - A record `(sign, logdet)` of two NDArrays, representing the sign (or phase) and log of the absolute determinant.
///
/// Reference: [NumPy linalg.slogdet](https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html)
({NDArray<T> sign, NDArray<R> logabsdet}) slogdet<T, R extends num>(
  NDArray<T> a, {
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
  if (a.dtype != DType.float64 &&
      a.dtype != DType.float32 &&
      a.dtype != DType.complex128 &&
      a.dtype != DType.complex64) {
    throw ArgumentError('slogdet only supports float and complex dtypes');
  }
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 1] != a.shape[rank - 2]) {
    throw ArgumentError(
      'Matrix must be square and at least 2D (was ${a.shape})',
    );
  }
  final stackShape = a.shape.sublist(0, rank - 2);

  final DType<R> logdetDType =
      (a.dtype == DType.float32 || a.dtype == DType.complex64)
      ? DType.float32 as DType<R>
      : DType.float64 as DType<R>;

  if (outSign != null) {
    if (!listEquals(outSign.shape, stackShape) || outSign.dtype != a.dtype) {
      throw ArgumentError(
        'Provided outSign buffer has incompatible shape or dtype.',
      );
    }
    if (!outSign.isContiguous) {
      throw ArgumentError('Provided outSign buffer must be contiguous.');
    }
  }

  if (outLogdet != null) {
    if (!listEquals(outLogdet.shape, stackShape) ||
        outLogdet.dtype != logdetDType) {
      throw ArgumentError(
        'Provided outLogdet buffer has incompatible shape or dtype.',
      );
    }
    if (!outLogdet.isContiguous) {
      throw ArgumentError('Provided outLogdet buffer must be contiguous.');
    }
  }

  return NDArray.scope(() {
    final signResult = outSign ?? NDArray<T>.zeros(stackShape, a.dtype);
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
            get_dgetrf_ptr(),
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
            get_sgetrf_ptr(),
          );
        case DType.complex128:
          final cCopy = ScratchArena.allocate<ffi.Double>(
            2 * n * n * ffi.sizeOf<ffi.Double>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );
          s_slogdet_complex_double(
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
            get_zgetrf_ptr(),
          );
        case DType.complex64:
          final cCopy = ScratchArena.allocate<ffi.Float>(
            2 * n * n * ffi.sizeOf<ffi.Float>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );
          s_slogdet_complex_float(
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
            get_cgetrf_ptr(),
          );
        default:
          throw UnsupportedError('Unsupported dtype ${a.dtype}');
      }
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
extension SlogdetRecordDispose<T, R>
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
/// ```dart
/// final a = NDArray<double>.fromList([3.0, 1.0, 1.0, 2.0], [2, 2], DType.float64);
/// final b = NDArray<double>.fromList([9.0, 8.0], [2], DType.float64);
/// final x = solve(a, b);
/// print(x.toList()); // [2.0, 3.0]
/// ```
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
    final cBuf = ScratchArena.allocate<ffi.Int>(4);
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
  final int info;
  switch (dtype) {
    case DType.float64:
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
      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_dgesv: $info');
      }
      if (info > 0) {
        throw SingularMatrixException(
          'Matrix is singular and cannot be solved',
        );
      }
    case DType.float32:
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
      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_sgesv: $info');
      }
      if (info > 0) {
        throw SingularMatrixException(
          'Matrix is singular and cannot be solved',
        );
      }
    case DType.complex128:
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
      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_zgesv: $info');
      }
      if (info > 0) {
        throw SingularMatrixException(
          'Matrix is singular and cannot be solved',
        );
      }
    case DType.complex64:
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
      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_cgesv: $info');
      }
      if (info > 0) {
        throw SingularMatrixException(
          'Matrix is singular and cannot be solved',
        );
      }
    default:
      throw UnsupportedError('Unsupported type for solve: $dtype');
  }
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
/// - If [out] is provided, it must match the result shape, data type, and be contiguous.
///
/// It is an error if $A$ or $B$ is disposed, non-square, or has incompatible shapes/dtypes.
///
/// Throws a [SingularMatrixException] if the matrix $A$ is singular or ill-conditioned.
///
/// {@example test/examples/linalg_example_test.dart [lang=dart] [indent=keep]}
NDArray<T> solve<T extends Object>(
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
    throw ArgumentError(
      'Matrix a must be square in the last 2 dimensions and rank >= 2 (was ${a.shape})',
    );
  }
  final n = a.shape[rankA - 1];
  final rankB = b.shape.length;

  if (rankA == 2) {
    if (rankB == 1) {
      if (b.shape[0] != n) {
        throw ArgumentError(
          'Dimensions of b (${b.shape}) must match matrix dimension $n of a (${a.shape})',
        );
      }
    } else if (rankB == 2) {
      if (b.shape[0] != n) {
        throw ArgumentError(
          'Dimensions of b (${b.shape}) must match matrix dimension $n of a (${a.shape})',
        );
      }
    } else {
      throw ArgumentError(
        'Dimensions of b (${b.shape}) are incompatible with a (${a.shape}). Expected rank 1 or 2.',
      );
    }
  } else {
    final stackShapeA = a.shape.sublist(0, rankA - 2);
    if (rankB == rankA - 1) {
      if (!listEquals(b.shape.sublist(0, rankA - 2), stackShapeA) ||
          b.shape[rankB - 1] != n) {
        throw ArgumentError(
          'Dimensions of b (${b.shape}) must match stack shape $stackShapeA and matrix dimension $n of a (${a.shape})',
        );
      }
    } else if (rankB == rankA) {
      if (!listEquals(b.shape.sublist(0, rankA - 2), stackShapeA) ||
          b.shape[rankB - 2] != n) {
        throw ArgumentError(
          'Dimensions of b (${b.shape}) must match stack shape $stackShapeA and matrix dimension $n of a (${a.shape})',
        );
      }
    } else {
      throw ArgumentError(
        'Dimensions of b (${b.shape}) are incompatible with a (${a.shape}). Expected rank ${rankA - 1} or $rankA.',
      );
    }
  }

  if (a.dtype != b.dtype) {
    throw ArgumentError(
      'Mismatched dtypes for solve: a has dtype ${a.dtype}, b has dtype ${b.dtype}.',
    );
  }

  if (a.dtype != DType.float64 &&
      a.dtype != DType.float32 &&
      a.dtype != DType.complex128 &&
      a.dtype != DType.complex64) {
    throw ArgumentError(
      'solve only supports float64, float32, complex128, or complex64 dtypes (got ${a.dtype}).',
    );
  }

  if (out != null) {
    if (!listEquals(out.shape, b.shape) || out.dtype != b.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype (expected shape ${b.shape} and dtype ${b.dtype}, got shape ${out.shape} and dtype ${out.dtype}).',
      );
    }
    if (!out.isContiguous) {
      throw ArgumentError('Provided out buffer must be contiguous.');
    }
  }

  final nrhs = rankB == rankA ? b.shape[rankB - 1] : 1;

  final NDArray<T> bCopy;
  if (out != null) {
    bCopy = out;
    b.copy(out: bCopy);
  } else {
    bCopy = b.copy();
  }

  if (n == 0) {
    return bCopy;
  }

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

      final isSliceAContig =
          aStrides[rankA - 2] == n && aStrides[rankA - 1] == 1;
      if (isSliceAContig) {
        final srcSliceA = ffi.Pointer<ffi.Uint8>.fromAddress(
          a.pointer.address + offsetA * a.dtype.byteWidth,
        );
        custom_memcpy(
          aCopyPtr.cast(),
          srcSliceA.cast(),
          n * n * a.dtype.byteWidth,
        );
      } else {
        _copyStrided2DMatrix(
          a.pointer,
          aStrides,
          offsetA,
          n,
          a.dtype,
          aCopyPtr.cast(),
        );
      }

      final bSlicePtr = ffi.Pointer<ffi.Void>.fromAddress(
        bCopy.pointer.address + offsetB * bElemSize,
      );

      _lapackeSolve(a.dtype, n, nrhs, aCopyPtr.cast(), ipiv, bSlicePtr);

      for (var d = stackDims - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < stackShape[d]) break;
        coords[d] = 0;
      }
    }

    return bCopy;
  } finally {
    ScratchArena.reset(marker);
  }
}

/// Computes the eigenvalues and right eigenvectors of a square array or stack of square arrays.
///
/// Returns a record `(eigenvalues, eigenvectors)` containing:
/// - **eigenvalues**: An `NDArray<Complex>` of shape `[..., N]` containing the eigenvalues.
/// - **eigenvectors**: An `NDArray<Complex>` of shape `[..., N, N]` containing the corresponding right eigenvectors as columns.
///
/// Both are returned with `Complex` elements because eigenvalues and eigenvectors can be complex
/// even for real matrices.
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] is not square in its last two dimensions or is less than 2-dimensional.
/// - It is an error if the DType of [a] is not supported.
/// - It is an error if [out] is provided and has incompatible shape, dtype, or is not contiguous.
({NDArray<Complex> eigenvalues, NDArray<Complex> eigenvectors}) eig<T>(
  NDArray<T> a, {
  ({NDArray<Complex> eigenvalues, NDArray<Complex> eigenvectors})? out,
}) {
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 1] != a.shape[rank - 2]) {
    throw ArgumentError(
      'Matrix must be square and at least 2D (was ${a.shape})',
    );
  }
  final n = a.shape[rank - 1];
  final stackShape = a.shape.sublist(0, rank - 2);

  final compDType = (a.dtype == DType.float32 || a.dtype == DType.complex64)
      ? DType.complex64
      : DType.complex128;

  final wShape = [...stackShape, n];
  final vrShape = [...stackShape, n, n];

  return NDArray.scope(() {
    final NDArray<Complex> w;
    final NDArray<Complex> vr;

    if (out != null) {
      w = out.eigenvalues;
      vr = out.eigenvectors;
      if (!listEquals(w.shape, wShape) || w.dtype != compDType) {
        throw ArgumentError(
          'Provided out eigenvalues buffer has incompatible shape or dtype (expected shape $wShape and dtype $compDType, got shape ${w.shape} and dtype ${w.dtype}).',
        );
      }
      if (!w.isContiguous) {
        throw ArgumentError(
          'Provided out eigenvalues buffer must be contiguous.',
        );
      }
      if (!listEquals(vr.shape, vrShape) || vr.dtype != compDType) {
        throw ArgumentError(
          'Provided out eigenvectors buffer has incompatible shape or dtype (expected shape $vrShape and dtype $compDType, got shape ${vr.shape} and dtype ${vr.dtype}).',
        );
      }
      if (!vr.isContiguous) {
        throw ArgumentError(
          'Provided out eigenvectors buffer must be contiguous.',
        );
      }
    } else {
      w = NDArray<Complex>.create(wShape, compDType);
      vr = NDArray<Complex>.create(vrShape, compDType);
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

    final bool wasCast = a.dtype.isInteger;
    final NDArray src = wasCast ? castNDArray(a, DType.float64) : a;
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

        var offsetW = 0;
        for (var i = 0; i < coords.length; i++) {
          offsetW += coords[i] * w.strides[i];
        }
        var offsetVR = 0;
        for (var i = 0; i < coords.length; i++) {
          offsetVR += coords[i] * vr.strides[i];
        }

        switch (src.dtype) {
          case DType.complex128:
            final w2D = NDArray<Complex>.create([n], DType.complex128);
            final vr2D = NDArray<Complex>.create([n, n], DType.complex128);

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

            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_zgeev: $info',
              );
            }
            if (info > 0) {
              throw IterationsExceededException(
                'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
              );
            }

            final wView = NDArray<Complex>.view(
              w,
              shape: [n],
              strides: w.strides.isEmpty ? [1] : [w.strides.last],
              offsetElements: offsetW,
            );
            w2D.copy(out: wView);

            final vrView = NDArray<Complex>.view(
              vr,
              shape: [n, n],
              strides: vr.strides.sublist(rank - 2),
              offsetElements: offsetVR,
            );
            vr2D.copy(out: vrView);

            w2D.dispose();
            vr2D.dispose();
          case DType.complex64:
            final w2D = NDArray<Complex>.create([n], DType.complex64);
            final vr2D = NDArray<Complex>.create([n, n], DType.complex64);

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

            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_cgeev: $info',
              );
            }
            if (info > 0) {
              throw IterationsExceededException(
                'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
              );
            }

            final wView = NDArray<Complex>.view(
              w,
              shape: [n],
              strides: w.strides.isEmpty ? [1] : [w.strides.last],
              offsetElements: offsetW,
            );
            w2D.copy(out: wView);

            final vrView = NDArray<Complex>.view(
              vr,
              shape: [n, n],
              strides: vr.strides.sublist(rank - 2),
              offsetElements: offsetVR,
            );
            vr2D.copy(out: vrView);

            w2D.dispose();
            vr2D.dispose();
          case DType.float64:
            final wr = NDArray<double>.zeros([n], DType.float64);
            final wi = NDArray<double>.zeros([n], DType.float64);
            final vrReal = NDArray<double>.create([n, n], DType.float64);

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

            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_dgeev: $info',
              );
            }
            if (info > 0) {
              throw IterationsExceededException(
                'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
              );
            }

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

            wr.dispose();
            wi.dispose();
            vrReal.dispose();
          case DType.float32:
            final wr = NDArray<double>.zeros([n], DType.float32);
            final wi = NDArray<double>.zeros([n], DType.float32);
            final vrReal = NDArray<double>.create([n, n], DType.float32);

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

            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_sgeev: $info',
              );
            }
            if (info > 0) {
              throw IterationsExceededException(
                'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
              );
            }

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

            wr.dispose();
            wi.dispose();
            vrReal.dispose();
          default:
            throw UnimplementedError('Type ${src.dtype} not supported for eig');
        }
        sliceCopy.dispose();
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

/// Extension on eigenvalue decomposition result record type to support easy disposal of both arrays.
extension EigRecordDispose
    on ({NDArray<Complex> eigenvalues, NDArray<Complex> eigenvectors}) {
  /// Disposes both [eigenvalues] and [eigenvectors] simultaneously,
  /// freeing their underlying unmanaged C memory.
  ///
  /// Call this method when both matrices are no longer needed to avoid native memory leaks.
  void dispose() {
    this.eigenvalues.dispose();
    this.eigenvectors.dispose();
  }
}

/// Computes only the eigenvalues of a general square 2D matrix or stack of matrices.
///
/// Unlike [eig], this function does not compute eigenvectors, making it much faster.
///
/// **Preconditions:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] is not square or rank < 2.
/// - It is an error if [a] has integer dtype or an unsupported dtype.
/// - It is an error if [out] is provided and has incompatible shape, dtype, or is not contiguous.
///
/// **Returns:**
/// - A contiguous `NDArray<Complex>` containing the computed eigenvalues.
///
/// Reference: [NumPy linalg.eigvals](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html)
NDArray<Complex> eigvals<T>(NDArray<T> a, {NDArray<Complex>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute eigvals of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write eigvals result to a disposed output array.');
  }
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 1] != a.shape[rank - 2]) {
    throw ArgumentError(
      'Matrix must be square and at least 2D (was ${a.shape})',
    );
  }
  final n = a.shape[rank - 1];
  final stackShape = a.shape.sublist(0, rank - 2);

  final compDType = (a.dtype == DType.float32 || a.dtype == DType.complex64)
      ? DType.complex64
      : DType.complex128;

  final wShape = [...stackShape, n];

  return NDArray.scope(() {
    final NDArray<Complex> w;

    if (out != null) {
      w = out;
      if (!listEquals(w.shape, wShape) || w.dtype != compDType) {
        throw ArgumentError(
          'Provided out eigenvalues buffer has incompatible shape or dtype (expected shape $wShape and dtype $compDType, got shape ${w.shape} and dtype ${w.dtype}).',
        );
      }
      if (!w.isContiguous) {
        throw ArgumentError(
          'Provided out eigenvalues buffer must be contiguous.',
        );
      }
    } else {
      w = NDArray<Complex>.create(wShape, compDType);
    }

    if (n == 0) {
      if (out == null) {
        w.detachToParentScope();
      }
      return w;
    }

    final jobvl = 'N'.codeUnitAt(0);
    final jobvr = 'N'.codeUnitAt(0);

    final bool wasCast = a.dtype.isInteger;
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

        var offsetW = 0;
        for (var i = 0; i < coords.length; i++) {
          offsetW += coords[i] * w.strides[i];
        }

        switch (src.dtype) {
          case DType.complex128:
            final w2D = NDArray<Complex>.create([n], DType.complex128);

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

            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_zgeev: $info',
              );
            }
            if (info > 0) {
              throw IterationsExceededException(
                'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
              );
            }

            final wView = NDArray<Complex>.view(
              w,
              shape: [n],
              strides: w.strides.isEmpty ? [1] : [w.strides.last],
              offsetElements: offsetW,
            );
            w2D.copy(out: wView);
            w2D.dispose();

          case DType.complex64:
            final w2D = NDArray<Complex>.create([n], DType.complex64);

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

            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_cgeev: $info',
              );
            }
            if (info > 0) {
              throw IterationsExceededException(
                'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
              );
            }

            final wView = NDArray<Complex>.view(
              w,
              shape: [n],
              strides: w.strides.isEmpty ? [1] : [w.strides.last],
              offsetElements: offsetW,
            );
            w2D.copy(out: wView);
            w2D.dispose();

          case DType.float64:
            final wr = NDArray<double>.zeros([n], DType.float64);
            final wi = NDArray<double>.zeros([n], DType.float64);

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

            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_dgeev: $info',
              );
            }
            if (info > 0) {
              throw IterationsExceededException(
                'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
              );
            }

            final strideWLast = w.strides.isEmpty ? 1 : w.strides.last;
            assemble_eigenvalues_double(
              w.pointer.cast<cpx_t>() + offsetW,
              strideWLast,
              wr.pointer.cast<ffi.Double>(),
              wi.pointer.cast<ffi.Double>(),
              n,
            );

            wr.dispose();
            wi.dispose();

          case DType.float32:
            final wr = NDArray<double>.zeros([n], DType.float32);
            final wi = NDArray<double>.zeros([n], DType.float32);

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

            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_sgeev: $info',
              );
            }
            if (info > 0) {
              throw IterationsExceededException(
                'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
              );
            }

            final strideWLast = w.strides.isEmpty ? 1 : w.strides.last;
            assemble_eigenvalues_float(
              w.pointer.cast<cpx_f_t>() + offsetW,
              strideWLast,
              wr.pointer.cast<ffi.Float>(),
              wi.pointer.cast<ffi.Float>(),
              n,
            );

            wr.dispose();
            wi.dispose();
          default:
            throw UnimplementedError(
              'Type ${src.dtype} not supported for eigvals',
            );
        }
        sliceCopy.dispose();
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
NDArray<T> pinv<T extends Object>(
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
    throw ArgumentError(
      'Moore-Penrose pseudo-inverse is only defined for 2D matrices (was shape ${a.shape}).',
    );
  }
  final m = a.shape[0];
  final n = a.shape[1];

  final targetShape = [n, m];
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
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
    final svdResult = svd(a);
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

    final sPlus = NDArray.zeros([n, m], a.dtype);
    for (var i = 0; i < s.shape[0]; i++) {
      final double sVal = (s.dtype == DType.float32)
          ? s.pointer.cast<ffi.Float>()[i]
          : s.pointer.cast<ffi.Double>()[i];
      if (sVal > threshold) {
        sPlus.setCell([i, i], castValue(1.0 / sVal, a.dtype));
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
NDArray<T> matrix_power<T>(NDArray<T> a, int n, {NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute matrix_power() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write matrix_power result to a disposed output array.',
    );
  }
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError(
      'matrix_power is only defined for 2D square matrices (was shape ${a.shape}).',
    );
  }
  if (n < 0 && a.dtype.isInteger) {
    throw ArgumentError(
      'Integer matrices cannot be raised to negative powers because matrix inversion '
      'requires floating point types. Please convert the matrix to float64 or float32 first.',
    );
  }

  final size = a.shape[0];
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  return NDArray.scope(() {
    final result = out ?? NDArray<T>.create(a.shape, a.dtype);
    if (n == 0) {
      final eye = NDArray.eye(size, a.dtype);
      result.fill(normalizeScalar(0, a.dtype) as T);
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
      base = inv(a);
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
/// - If provided, the [out] destination matrix must have the same shape and dtype as [a], and must be contiguous.
///
/// **Throws:**
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if [a] has rank < 2 or trailing dimensions are not square.
/// - It is an error if [a] has an unsupported dtype (e.g. integer or boolean).
/// - It is an error if the provided [out] buffer has an incompatible shape, dtype, or is not contiguous.
/// - It is an error if LAPACK returns an error code (illegal value).
/// - Throws [NonPositiveDefiniteException] if any matrix slice is not positive-definite.
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
NDArray<T> cholesky<T extends Object>(NDArray<T> a, {NDArray<T>? out}) {
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
    throw ArgumentError(
      'Matrix must be square in the last 2 dimensions and rank >= 2 (was ${a.shape})',
    );
  }
  if (!a.dtype.isFloating && !a.dtype.isComplex) {
    throw ArgumentError(
      'Cholesky decomposition is only supported for float and complex dtypes (was ${a.dtype})',
    );
  }
  final n = a.shape[rank - 1];
  final stackShape = a.shape.sublist(0, rank - 2);
  final targetDType = a.dtype;

  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out L buffer has incompatible shape or dtype.',
      );
    }
    if (!out.isContiguous) {
      throw ArgumentError('Provided out L buffer must be contiguous.');
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

      final int info;
      switch (targetDType) {
        case DType.float64:
          info = LAPACKE_dpotrf(
            101, // ROW_MAJOR
            uploL,
            n,
            lSlice.pointer.cast<ffi.Double>(),
            n,
          );
        case DType.float32:
          info = LAPACKE_spotrf(
            101, // ROW_MAJOR
            uploL,
            n,
            lSlice.pointer.cast<ffi.Float>(),
            n,
          );
        case DType.complex128:
          info = LAPACKE_zpotrf(
            101, // ROW_MAJOR
            uploL,
            n,
            lSlice.pointer.cast<ffi.Double>(),
            n,
          );
        case DType.complex64:
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

      if (info < 0) {
        throw ArgumentError(
          'Illegal value in call to LAPACKE Cholesky solver: $info',
        );
      }
      if (info > 0) {
        throw NonPositiveDefiniteException(
          'Matrix must be positive-definite for Cholesky decomposition: the leading minor of order $info is not positive definite.',
        );
      }

      v_zero_upper_triangular(
        lSlice.pointer.cast<ffi.Void>(),
        n,
        encodeDType(targetDType),
      );
      lSlice.dispose();
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
/// - It is an error if [out] has incompatible shape, dtype, or is not contiguous.
///
/// **Example:**
/// ```dart
/// final a = NDArray<double>.fromList([12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0], [3, 3], DType.float64);
/// final res = qr(a);
/// final q = res.q;
/// final r = res.r;
/// ```
({NDArray<T> q, NDArray<T> r}) qr<T extends Object>(
  NDArray<T> a, {
  ({NDArray<T> q, NDArray<T> r})? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute qr() on a disposed array.');
  }
  final rank = a.shape.length;
  if (rank < 2) {
    throw ArgumentError('Matrix must be at least 2D (was ${a.shape})');
  }
  if (!a.dtype.isFloating && !a.dtype.isComplex) {
    throw ArgumentError(
      'QR decomposition is only supported for float and complex dtypes (was ${a.dtype})',
    );
  }
  final m = a.shape[rank - 2];
  final n = a.shape[rank - 1];
  final k = m < n ? m : n;
  final stackShape = a.shape.sublist(0, rank - 2);

  final DType<T> targetDType = a.dtype;

  final qShape = [...stackShape, m, k];
  final rShape = [...stackShape, k, n];

  return NDArray.scope(() {
    final NDArray<T> qMat;
    final NDArray<T> rMat;
    if (out != null) {
      qMat = out.q;
      rMat = out.r;
      if (!listEquals(qMat.shape, qShape) || qMat.dtype != targetDType) {
        throw ArgumentError(
          'Provided out Q buffer has incompatible shape or dtype.',
        );
      }
      if (!qMat.isContiguous) {
        throw ArgumentError('Provided out Q buffer must be contiguous.');
      }
      if (!listEquals(rMat.shape, rShape) || rMat.dtype != targetDType) {
        throw ArgumentError(
          'Provided out R buffer has incompatible shape or dtype.',
        );
      }
      if (!rMat.isContiguous) {
        throw ArgumentError('Provided out R buffer must be contiguous.');
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
          tau = ScratchArena.allocate<ffi.Double>(k * ffi.sizeOf<ffi.Double>())
              .cast<ffi.Void>();
        case DType.float32:
          tau = ScratchArena.allocate<ffi.Float>(k * ffi.sizeOf<ffi.Float>())
              .cast<ffi.Void>();
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
            if (info != 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_dgeqrf: $info',
              );
            }
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
            if (infoOrg != 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_dorgqr: $infoOrg',
              );
            }
          case DType.float32:
            final info = LAPACKE_sgeqrf(
              101, // ROW_MAJOR
              m,
              n,
              aCopy.pointer.cast<ffi.Float>(),
              n,
              tau.cast<ffi.Float>(),
            );
            if (info != 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_sgeqrf: $info',
              );
            }
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
            if (infoOrg != 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_sorgqr: $infoOrg',
              );
            }
          case DType.complex128:
            final info = LAPACKE_zgeqrf(
              101, // ROW_MAJOR
              m,
              n,
              aCopy.pointer.cast<ffi.Double>(),
              n,
              tau.cast<ffi.Double>(),
            );
            if (info != 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_zgeqrf: $info',
              );
            }
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
            if (infoOrg != 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_zungqr: $infoOrg',
              );
            }
          case DType.complex64:
            final info = LAPACKE_cgeqrf(
              101, // ROW_MAJOR
              m,
              n,
              aCopy.pointer.cast<ffi.Float>(),
              n,
              tau.cast<ffi.Float>(),
            );
            if (info != 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_cgeqrf: $info',
              );
            }
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
            if (infoOrg != 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_cungqr: $infoOrg',
              );
            }
          default:
            break;
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
/// - It is an error if any buffer in [out] has incompatible shape, dtype, or is not contiguous.
///
/// **Example:**
/// ```dart
/// final a = NDArray<double>.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], DType.float64);
/// final res = svd(a);
/// final u = res.u;
/// final s = res.s;
/// final vh = res.vh;
/// ```
({NDArray<T> u, NDArray<double> s, NDArray<T> vh}) svd<T extends Object>(
  NDArray<T> a, {
  ({NDArray<T> u, NDArray<double> s, NDArray<T> vh})? out,
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
    throw ArgumentError(
      'SVD decomposition is only supported for float and complex dtypes (was ${a.dtype})',
    );
  }
  final rank = a.shape.length;
  if (rank < 2) {
    throw ArgumentError('Matrix must be at least 2D (was ${a.shape})');
  }
  final m = a.shape[rank - 2];
  final n = a.shape[rank - 1];
  final stackShape = a.shape.sublist(0, rank - 2);

  final dtypeS = a.dtype.isComplex
      ? (a.dtype == DType.complex128 ? DType.float64 : DType.float32)
      : a.dtype;

  final uShape = [...stackShape, m, m];
  final sShape = m < n ? [...stackShape, m] : [...stackShape, n];
  final vtShape = [...stackShape, n, n];

  if (out != null) {
    if (!out.u.isContiguous || !out.s.isContiguous || !out.vh.isContiguous) {
      throw ArgumentError('Provided out buffers must be contiguous.');
    }
    if (!listEquals(out.u.shape, uShape) || out.u.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out U buffer has incompatible shape or dtype.',
      );
    }
    if (!listEquals(out.s.shape, sShape) || out.s.dtype != dtypeS) {
      throw ArgumentError(
        'Provided out S buffer has incompatible shape or dtype.',
      );
    }
    if (!listEquals(out.vh.shape, vtShape) || out.vh.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out Vh buffer has incompatible shape or dtype.',
      );
    }
  }

  return _svd<T>(a, out: out);
}

({NDArray<T> u, NDArray<double> s, NDArray<T> vh}) _svd<T extends Object>(
  NDArray<T> a, {
  ({NDArray<T> u, NDArray<double> s, NDArray<T> vh})? out,
}) {
  final rank = a.shape.length;
  final m = a.shape[rank - 2];
  final n = a.shape[rank - 1];
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
      final sMat = out?.s ?? NDArray<double>.zeros(sShape, dtypeS as dynamic);
      final vhMat = out?.vh ?? NDArray<T>.zeros(vtShape, a.dtype);

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
            uSlice.setCell([i, i], castValue(1.0, a.dtype));
          }
          uSlice.dispose();
        });
      }
      if (n > 0) {
        walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
          coords,
        ) {
          var offsetVh = 0;
          for (var i = 0; i < coords.length; i++) {
            offsetVh += coords[i] * vhMat.strides[i];
          }
          final vhSlice = NDArray<T>.view(
            vhMat,
            shape: [n, n],
            strides: vhMat.strides.sublist(rank - 2),
            offsetElements: offsetVh,
          );
          for (var i = 0; i < n; i++) {
            vhSlice.setCell([i, i], castValue(1.0, a.dtype));
          }
          vhSlice.dispose();
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
      final axes = List<int>.generate(rank, (i) => i);
      axes[rank - 2] = rank - 1;
      axes[rank - 1] = rank - 2;

      final aT = a.transpose(axes);
      try {
        // Do NOT pass out to recursive call, let it allocate contiguous buffers.
        final resT = _svd<T>(aT);
        final uNew = resT.u;
        final sNew = resT.s;
        final vhNew = resT.vh;

        final uResult = vhNew.transpose(axes);
        final vhResult = uNew.transpose(axes);

        if (out != null) {
          uResult.copy(out: out.u);
          sNew.copy(out: out.s);
          vhResult.copy(out: out.vh);

          uNew.dispose();
          sNew.dispose();
          vhNew.dispose();
          uResult.dispose();
          vhResult.dispose();

          return out;
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
    final NDArray<double> sMat =
        out?.s ?? NDArray<double>.zeros(sShape, dtypeS as DType<double>);
    final NDArray<T> vtMat = out?.vh ?? NDArray<T>.zeros(vtShape, a.dtype);

    final aCopy = NDArray<T>.create([m, n], a.dtype);
    final marker = ScratchArena.marker;

    try {
      final ffi.Pointer<ffi.Void> superb;
      final superbLen = math.max(1, n - 1);
      if (a.dtype == DType.float64 || a.dtype == DType.complex128) {
        superb = ScratchArena.allocate<ffi.Double>(
          superbLen * ffi.sizeOf<ffi.Double>(),
        ).cast<ffi.Void>();
      } else {
        superb = ScratchArena.allocate<ffi.Float>(
          superbLen * ffi.sizeOf<ffi.Float>(),
        ).cast<ffi.Void>();
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

        final NDArray<double> s2D =
            (a.dtype == DType.float32 || a.dtype == DType.complex64)
            ? NDArray<Float32>.zeros([n], DType.float32)
            : NDArray<Float64>.zeros([n], DType.float64);
        final NDArray u2D = NDArray.zeros([m, m], a.dtype);
        final NDArray vt2D = NDArray.zeros([n, n], a.dtype);

        switch (a.dtype) {
          case DType.float64:
            final info = LAPACKE_dgesvd(
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
            if (info != 0) throw ArgumentError('LAPACKE_dgesvd failed: $info');

          case DType.float32:
            final info = LAPACKE_sgesvd(
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
            if (info != 0) throw ArgumentError('LAPACKE_sgesvd failed: $info');

          case DType.complex128:
            final info = LAPACKE_zgesvd(
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
            if (info != 0) throw ArgumentError('LAPACKE_zgesvd failed: $info');

          case DType.complex64:
            final info = LAPACKE_cgesvd(
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
            if (info != 0) throw ArgumentError('LAPACKE_cgesvd failed: $info');
          default:
            throw ArgumentError('Unsupported dtype for SVD: ${a.dtype}');
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

        final sSlice = NDArray<double>.view(
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
/// - [ArgumentError] if [a] is not square or has rank < 2.
/// - [ArgumentError] if [a] has unsupported dtype.
/// - [ArgumentError] if [outEigenvalues] or [outEigenvectors] are incompatible.
/// - [StateError] if the LAPACK call fails.
({NDArray<num> eigenvalues, NDArray<R> eigenvectors})
eigh<T extends Object, R extends Object>(
  NDArray<T> a, {
  MatrixTriangle uplo = MatrixTriangle.lower,
  NDArray<num>? outEigenvalues,
  NDArray<T>? outEigenvectors,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot calculate eigh on a disposed array.');
  }
  if (a.rank < 2) {
    throw ArgumentError('Array must be at least 2-dimensional.');
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m != n) {
    throw ArgumentError('Last two dimensions must be square (got $m x $n).');
  }

  final bool promoted = a.dtype.isInteger;
  DType targetDType = a.dtype;
  if (promoted) {
    targetDType = DType.float64;
  }

  if (targetDType != DType.float64 &&
      targetDType != DType.float32 &&
      targetDType != DType.complex128 &&
      targetDType != DType.complex64) {
    throw ArgumentError('Unsupported dtype: ${a.dtype}');
  }

  final DType<num> eigenvalueDType =
      (targetDType.isComplex
              ? (targetDType == DType.complex128
                    ? DType.float64
                    : DType.float32)
              : targetDType)
          as DType<num>;

  final stackShape = a.shape.sublist(0, a.rank - 2);

  final eigenvaluesShape = [...stackShape, n];
  final eigenvectorsShape = [...stackShape, n, n];

  if (outEigenvalues != null) {
    if (outEigenvalues.isDisposed) {
      throw StateError('outEigenvalues is disposed.');
    }
    if (!outEigenvalues.isContiguous) {
      throw ArgumentError('outEigenvalues must be contiguous.');
    }
    if (!listEquals(outEigenvalues.shape, eigenvaluesShape) ||
        outEigenvalues.dtype != eigenvalueDType) {
      throw ArgumentError(
        'Incompatible outEigenvalues (expected shape $eigenvaluesShape and dtype $eigenvalueDType, got shape ${outEigenvalues.shape} and dtype ${outEigenvalues.dtype}).',
      );
    }
  }

  if (outEigenvectors != null) {
    if (outEigenvectors.isDisposed) {
      throw StateError('outEigenvectors is disposed.');
    }
    if (!outEigenvectors.isContiguous) {
      throw ArgumentError('outEigenvectors must be contiguous.');
    }
    if (!listEquals(outEigenvectors.shape, eigenvectorsShape) ||
        outEigenvectors.dtype != targetDType) {
      throw ArgumentError(
        'Incompatible outEigenvectors (expected shape $eigenvectorsShape and dtype $targetDType, got shape ${outEigenvectors.shape} and dtype ${outEigenvectors.dtype}).',
      );
    }
  }

  return NDArray.scope(() {
    final NDArray<num> wMat;
    if (outEigenvalues != null) {
      wMat = outEigenvalues;
    } else {
      wMat = _zerosTyped(eigenvaluesShape, eigenvalueDType) as NDArray<num>;
    }

    final NDArray vMat;
    if (outEigenvectors != null) {
      vMat = outEigenvectors;
    } else {
      vMat = _zerosTyped(eigenvectorsShape, targetDType);
    }

    if (n == 0) {
      if (outEigenvalues == null) wMat.detachToParentScope();
      if (outEigenvectors == null) vMat.detachToParentScope();
      return (eigenvalues: wMat, eigenvectors: vMat as NDArray<R>);
    }

    final uploVal = uplo == MatrixTriangle.lower ? 76 : 85;
    final jobzVal = 86; // 'V'

    final aCopy2D = _createTyped2D(n, n, targetDType);
    final w2D = _zerosTyped([n], eigenvalueDType) as NDArray<num>;

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
          sliceView.copy(out: aCopy2D as NDArray<T>);
        } else {
          final casted = castNDArray(sliceView, targetDType);
          casted.copy(out: aCopy2D);
          casted.dispose();
        }
        sliceView.dispose();

        int info = 0;
        switch (targetDType) {
          case DType.float64:
            info = LAPACKE_dsyevd(
              101,
              jobzVal,
              uploVal,
              n,
              aCopy2D.pointer.cast<ffi.Double>(),
              n,
              w2D.pointer.cast<ffi.Double>(),
            );
            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_dsyevd: $info',
              );
            }
            if (info > 0) {
              throw IterationsExceededException(
                'LAPACKE_dsyevd failed to converge: $info',
              );
            }
          case DType.float32:
            info = LAPACKE_ssyevd(
              101,
              jobzVal,
              uploVal,
              n,
              aCopy2D.pointer.cast<ffi.Float>(),
              n,
              w2D.pointer.cast<ffi.Float>(),
            );
            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_ssyevd: $info',
              );
            }
            if (info > 0) {
              throw IterationsExceededException(
                'LAPACKE_ssyevd failed to converge: $info',
              );
            }
          case DType.complex128:
            info = LAPACKE_zheevd(
              101,
              jobzVal,
              uploVal,
              n,
              aCopy2D.pointer.cast<ffi.Double>(),
              n,
              w2D.pointer.cast<ffi.Double>(),
            );
            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_zheevd: $info',
              );
            }
            if (info > 0) {
              throw IterationsExceededException(
                'LAPACKE_zheevd failed to converge: $info',
              );
            }
          case DType.complex64:
            info = LAPACKE_cheevd(
              101,
              jobzVal,
              uploVal,
              n,
              aCopy2D.pointer.cast<ffi.Float>(),
              n,
              w2D.pointer.cast<ffi.Float>(),
            );
            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_cheevd: $info',
              );
            }
            if (info > 0) {
              throw IterationsExceededException(
                'LAPACKE_cheevd failed to converge: $info',
              );
            }
          default:
            throw UnimplementedError();
        }

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

    if (outEigenvalues == null) wMat.detachToParentScope();
    if (outEigenvectors == null) vMat.detachToParentScope();
    return (eigenvalues: wMat, eigenvectors: vMat as NDArray<R>);
  });
}

/// Extension on [eigh] result record type to support easy disposal of both arrays.
extension EighRecordDispose<T>
    on ({NDArray<num> eigenvalues, NDArray<T> eigenvectors}) {
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
/// - [ArgumentError] if [a] is not square or has rank < 2.
/// - [ArgumentError] if [a] has unsupported dtype.
/// - [ArgumentError] if [out] is incompatible.
/// - [StateError] if the LAPACK call fails.
NDArray<num> eigvalsh<T>(
  NDArray<T> a, {
  MatrixTriangle uplo = MatrixTriangle.lower,
  NDArray<num>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot calculate eigvalsh on a disposed array.');
  }
  if (a.rank < 2) {
    throw ArgumentError('Array must be at least 2-dimensional.');
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m != n) {
    throw ArgumentError('Last two dimensions must be square (got $m x $n).');
  }

  final bool promoted = a.dtype.isInteger;
  DType targetDType = a.dtype;
  if (promoted) {
    targetDType = DType.float64;
  }

  if (targetDType != DType.float64 &&
      targetDType != DType.float32 &&
      targetDType != DType.complex128 &&
      targetDType != DType.complex64) {
    throw ArgumentError('Unsupported dtype: ${a.dtype}');
  }

  final DType<num> eigenvalueDType =
      (targetDType.isComplex
              ? (targetDType == DType.complex128
                    ? DType.float64
                    : DType.float32)
              : targetDType)
          as DType<num>;

  final stackShape = a.shape.sublist(0, a.rank - 2);
  final eigenvaluesShape = [...stackShape, n];

  if (out != null) {
    if (out.isDisposed) {
      throw StateError('out is disposed.');
    }
    if (!out.isContiguous) {
      throw ArgumentError('out must be contiguous.');
    }
    if (!listEquals(out.shape, eigenvaluesShape) ||
        out.dtype != eigenvalueDType) {
      throw ArgumentError(
        'Incompatible out (expected shape $eigenvaluesShape and dtype $eigenvalueDType, got shape ${out.shape} and dtype ${out.dtype}).',
      );
    }
  }

  return NDArray.scope(() {
    final NDArray<num> wMat;
    if (out != null) {
      wMat = out;
    } else {
      wMat = _zerosTyped(eigenvaluesShape, eigenvalueDType) as NDArray<num>;
    }

    if (n == 0) {
      if (out == null) {
        wMat.detachToParentScope();
      }
      return wMat;
    }

    final uploVal = uplo == MatrixTriangle.lower ? 76 : 85;
    final jobzVal = 78; // 'N'

    final aCopy2D = _createTyped2D(n, n, targetDType);
    final w2D = _zerosTyped([n], eigenvalueDType) as NDArray<num>;

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
          sliceView.copy(out: aCopy2D as NDArray<T>);
        } else {
          final casted = castNDArray(sliceView, targetDType);
          casted.copy(out: aCopy2D);
          casted.dispose();
        }
        sliceView.dispose();

        int info = 0;
        switch (targetDType) {
          case DType.float64:
            info = LAPACKE_dsyevd(
              101,
              jobzVal,
              uploVal,
              n,
              aCopy2D.pointer.cast<ffi.Double>(),
              n,
              w2D.pointer.cast<ffi.Double>(),
            );
            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_dsyevd: $info',
              );
            }
            if (info > 0) {
              throw IterationsExceededException(
                'LAPACKE_dsyevd failed to converge: $info',
              );
            }
          case DType.float32:
            info = LAPACKE_ssyevd(
              101,
              jobzVal,
              uploVal,
              n,
              aCopy2D.pointer.cast<ffi.Float>(),
              n,
              w2D.pointer.cast<ffi.Float>(),
            );
            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_ssyevd: $info',
              );
            }
            if (info > 0) {
              throw IterationsExceededException(
                'LAPACKE_ssyevd failed to converge: $info',
              );
            }
          case DType.complex128:
            info = LAPACKE_zheevd(
              101,
              jobzVal,
              uploVal,
              n,
              aCopy2D.pointer.cast<ffi.Double>(),
              n,
              w2D.pointer.cast<ffi.Double>(),
            );
            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_zheevd: $info',
              );
            }
            if (info > 0) {
              throw IterationsExceededException(
                'LAPACKE_zheevd failed to converge: $info',
              );
            }
          case DType.complex64:
            info = LAPACKE_cheevd(
              101,
              jobzVal,
              uploVal,
              n,
              aCopy2D.pointer.cast<ffi.Float>(),
              n,
              w2D.pointer.cast<ffi.Float>(),
            );
            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_cheevd: $info',
              );
            }
            if (info > 0) {
              throw IterationsExceededException(
                'LAPACKE_cheevd failed to converge: $info',
              );
            }
          default:
            throw UnimplementedError();
        }

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

    if (out == null) {
      wMat.detachToParentScope();
    }
    return wMat;
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
/// - It is an error if [outT] or [outZ] is provided and has incompatible shape, dtype, or is not contiguous.
/// - [output] must be [SchurForm.real] or [SchurForm.complex].
///
/// **Throws:**
/// - Throws [LinAlgException] if the QR algorithm fails to compute eigenvalues or if eigenvalues cannot be reordered.
({NDArray<R> t, NDArray<R> z}) schur<T extends Object, R extends Object>(
  NDArray<T> a, {
  SchurForm output = SchurForm.real,
  NDArray<R>? outT,
  NDArray<R>? outZ,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot calculate schur on a disposed array.');
  }
  if (a.rank < 2) {
    throw ArgumentError('Array must be at least 2-dimensional.');
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m != n) {
    throw ArgumentError('Last two dimensions must be square (got $m x $n).');
  }

  final bool promoted = a.dtype.isInteger;
  DType targetDType = a.dtype;
  if (promoted) {
    targetDType = DType.float64;
  }

  if (targetDType != DType.float64 &&
      targetDType != DType.float32 &&
      targetDType != DType.complex128 &&
      targetDType != DType.complex64) {
    throw ArgumentError('Unsupported dtype: ${a.dtype}');
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
    if (!outT.isContiguous) throw ArgumentError('outT must be contiguous.');
    if (!listEquals(outT.shape, schurShape) || outT.dtype != targetDType) {
      throw ArgumentError('Incompatible outT.');
    }
  }

  if (outZ != null) {
    if (outZ.isDisposed) throw StateError('outZ is disposed.');
    if (!outZ.isContiguous) throw ArgumentError('outZ must be contiguous.');
    if (!listEquals(outZ.shape, schurShape) || outZ.dtype != targetDType) {
      throw ArgumentError('Incompatible outZ.');
    }
  }

  return NDArray.scope(() {
    final NDArray tMat = outT ?? _zerosTyped(schurShape, targetDType);
    final NDArray zMat = outZ ?? _zerosTyped(schurShape, targetDType);

    if (n == 0) {
      if (outT == null) tMat.detachToParentScope();
      if (outZ == null) zMat.detachToParentScope();
      return (t: tMat as NDArray<R>, z: zMat as NDArray<R>);
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

      if (targetDType == DType.float64) {
        wr = ScratchArena.allocate<ffi.Double>(n * ffi.sizeOf<ffi.Double>())
            .cast<ffi.Void>();
        wi = ScratchArena.allocate<ffi.Double>(n * ffi.sizeOf<ffi.Double>())
            .cast<ffi.Void>();
        w = ffi.nullptr.cast<ffi.Void>();
      } else if (targetDType == DType.float32) {
        wr = ScratchArena.allocate<ffi.Float>(n * ffi.sizeOf<ffi.Float>())
            .cast<ffi.Void>();
        wi = ScratchArena.allocate<ffi.Float>(n * ffi.sizeOf<ffi.Float>())
            .cast<ffi.Void>();
        w = ffi.nullptr.cast<ffi.Void>();
      } else if (targetDType == DType.complex128) {
        wr = ffi.nullptr.cast<ffi.Void>();
        wi = ffi.nullptr.cast<ffi.Void>();
        w = ScratchArena.allocate<ffi.Double>(2 * n * ffi.sizeOf<ffi.Double>())
            .cast<ffi.Void>();
      } else {
        // complex64
        wr = ffi.nullptr.cast<ffi.Void>();
        wi = ffi.nullptr.cast<ffi.Void>();
        w = ScratchArena.allocate<ffi.Float>(2 * n * ffi.sizeOf<ffi.Float>())
            .cast<ffi.Void>();
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
          sliceView.copy(out: aCopy2D as NDArray<T>);
        } else {
          final casted = castNDArray(sliceView, targetDType);
          casted.copy(out: aCopy2D);
          casted.dispose();
        }
        sliceView.dispose();

        int info = 0;
        switch (targetDType) {
          case DType.float64:
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
            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_dgees: $info',
              );
            }
            if (info > 0 && info <= n) {
              throw IterationsExceededException(
                'The QR algorithm failed to compute all eigenvalues in LAPACKE_dgees: $info',
              );
            }
            if (info > n) {
              throw LinAlgException(
                'Eigenvalues could not be reordered in LAPACKE_dgees: $info',
              );
            }
          case DType.float32:
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
            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_sgees: $info',
              );
            }
            if (info > 0 && info <= n) {
              throw IterationsExceededException(
                'The QR algorithm failed to compute all eigenvalues in LAPACKE_sgees: $info',
              );
            }
            if (info > n) {
              throw LinAlgException(
                'Eigenvalues could not be reordered in LAPACKE_sgees: $info',
              );
            }
          case DType.complex128:
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
            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_zgees: $info',
              );
            }
            if (info > 0 && info <= n) {
              throw IterationsExceededException(
                'The QR algorithm failed to compute all eigenvalues in LAPACKE_zgees: $info',
              );
            }
            if (info > n) {
              throw LinAlgException(
                'Eigenvalues could not be reordered in LAPACKE_zgees: $info',
              );
            }
          case DType.complex64:
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
            if (info < 0) {
              throw ArgumentError(
                'Illegal value in call to LAPACKE_cgees: $info',
              );
            }
            if (info > 0 && info <= n) {
              throw IterationsExceededException(
                'The QR algorithm failed to compute all eigenvalues in LAPACKE_cgees: $info',
              );
            }
            if (info > n) {
              throw LinAlgException(
                'Eigenvalues could not be reordered in LAPACKE_cgees: $info',
              );
            }
          default:
            throw UnimplementedError();
        }

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

    if (outT == null) tMat.detachToParentScope();
    if (outZ == null) zMat.detachToParentScope();
    return (t: tMat as NDArray<R>, z: zMat as NDArray<R>);
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
/// - [StateError] if the LAPACK call fails.
({NDArray<T> h, NDArray<T> q}) hessenberg<T>(
  NDArray<T> a, {
  NDArray<T>? outH,
  NDArray<T>? outQ,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot calculate hessenberg on a disposed array.');
  }
  if (a.rank < 2) {
    throw ArgumentError('Array must be at least 2-dimensional.');
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m != n) {
    throw ArgumentError('Last two dimensions must be square (got $m x $n).');
  }

  final bool promoted = a.dtype.isInteger;
  DType targetDType = a.dtype;
  if (promoted) {
    targetDType = DType.float64;
  }

  if (targetDType != DType.float64 &&
      targetDType != DType.float32 &&
      targetDType != DType.complex128 &&
      targetDType != DType.complex64) {
    throw ArgumentError('Unsupported dtype: ${a.dtype}');
  }

  final stackShape = a.shape.sublist(0, a.rank - 2);
  final hessenbergShape = [...stackShape, n, n];

  if (outH != null) {
    if (outH.isDisposed) throw StateError('outH is disposed.');
    if (!outH.isContiguous) throw ArgumentError('outH must be contiguous.');
    if (!listEquals(outH.shape, hessenbergShape) || outH.dtype != targetDType) {
      throw ArgumentError('Incompatible outH.');
    }
  }

  if (outQ != null) {
    if (outQ.isDisposed) throw StateError('outQ is disposed.');
    if (!outQ.isContiguous) throw ArgumentError('outQ must be contiguous.');
    if (!listEquals(outQ.shape, hessenbergShape) || outQ.dtype != targetDType) {
      throw ArgumentError('Incompatible outQ.');
    }
  }

  return NDArray.scope(() {
    final NDArray hMat = outH ?? _zerosTyped(hessenbergShape, targetDType);
    final NDArray qMat = outQ ?? _zerosTyped(hessenbergShape, targetDType);

    if (n == 0) {
      if (outH == null) hMat.detachToParentScope();
      if (outQ == null) qMat.detachToParentScope();
      return (h: hMat as NDArray<T>, q: qMat as NDArray<T>);
    }

    final aCopy2D = _createTyped2D(n, n, targetDType);
    final q2D = _zerosTyped([n, n], targetDType);

    final marker = ScratchArena.marker;
    try {
      final ffi.Pointer<ffi.Void> tau;
      final int elements = (n - 1) * (targetDType.isComplex ? 2 : 1);
      if (targetDType == DType.float64 || targetDType == DType.complex128) {
        tau = ScratchArena.allocate<ffi.Double>(
          elements * ffi.sizeOf<ffi.Double>(),
        ).cast<ffi.Void>();
      } else {
        tau = ScratchArena.allocate<ffi.Float>(
          elements * ffi.sizeOf<ffi.Float>(),
        ).cast<ffi.Void>();
      }

      walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
        coords,
      ) {
        final sliceView = a.slice([
          ...coords.map((c) => Index(c)),
          Slice.all(),
          Slice.all(),
        ]);
        if (sliceView.dtype == targetDType) {
          sliceView.copy(out: aCopy2D as NDArray<T>);
        } else {
          final casted = castNDArray(sliceView, targetDType);
          casted.copy(out: aCopy2D);
          casted.dispose();
        }
        sliceView.dispose();

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
            if (info != 0) throw StateError('LAPACKE_dgehrd failed: $info');
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
            if (info != 0) throw StateError('LAPACKE_sgehrd failed: $info');
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
            if (info != 0) throw StateError('LAPACKE_zgehrd failed: $info');
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
            if (info != 0) throw StateError('LAPACKE_cgehrd failed: $info');
          default:
            throw UnimplementedError();
        }

        aCopy2D.copy(out: q2D);

        final hSlice = hMat.slice([
          ...coords.map((c) => Index(c)),
          Slice.all(),
          Slice.all(),
        ]);
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
            if (info != 0) throw StateError('LAPACKE_dorghr failed: $info');
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
            if (info != 0) throw StateError('LAPACKE_sorghr failed: $info');
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
            if (info != 0) throw StateError('LAPACKE_zunghr failed: $info');
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
            if (info != 0) throw StateError('LAPACKE_cunghr failed: $info');
          default:
            throw UnimplementedError();
        }

        hSlice.dispose();

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

    if (outH == null) hMat.detachToParentScope();
    if (outQ == null) qMat.detachToParentScope();
    return (h: hMat as NDArray<T>, q: qMat as NDArray<T>);
  });
}

NDArray _createTyped2D(int rows, int cols, DType dtype) {
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

NDArray _zerosTyped(List<int> shape, DType dtype) {
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
NDArray<R> outer<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute outer() on a disposed array.');
  }

  final sizeA = a.size;
  final sizeB = b.size;
  final expectedShape = [sizeA, sizeB];
  final targetDType = resolveDType(a.dtype, b.dtype);

  if (out != null) {
    if (!listEquals(out.shape, expectedShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out recycler has incompatible shape or dtype (expected shape $expectedShape and dtype $targetDType).',
      );
    }
  }

  final result =
      out ?? NDArray<R>.create(expectedShape, targetDType as DType<R>);

  final flatA = a.rank == 1 ? a : a.ravel();
  final flatB = b.rank == 1 ? b : b.ravel();

  final aCast = castNDArray(flatA, targetDType);
  final bCast = castNDArray(flatB, targetDType);

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
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
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
    if (flatA != a) flatA.dispose();
    if (flatB != b) flatB.dispose();
    if (aCast != flatA) aCast.dispose();
    if (bCast != flatB) bCast.dispose();
  }

  return result;
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
NDArray<R> cross<Ta, Tb, R>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  int? axisa,
  int? axisb,
  int? axisc,
  int? axis,
  NDArray<R>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute cross() on a disposed array.');
  }

  var axisA = axis ?? axisa ?? -1;
  var axisB = axis ?? axisb ?? -1;
  var axisC = axis ?? axisc ?? -1;

  if (axisA < 0) axisA = a.rank + axisA;
  if (axisB < 0) axisB = b.rank + axisB;

  if (axisA < 0 || axisA >= a.rank) {
    throw ArgumentError('axisa $axisA out of bounds for shape ${a.shape}');
  }
  if (axisB < 0 || axisB >= b.rank) {
    throw ArgumentError('axisb $axisB out of bounds for shape ${b.shape}');
  }

  final lenA = a.shape[axisA];
  final lenB = b.shape[axisB];

  if ((lenA != 2 && lenA != 3) || (lenB != 2 && lenB != 3)) {
    throw ArgumentError(
      'Cross product axes sizes must be 2 or 3 (got axisa size $lenA and axisb size $lenB).',
    );
  }
  if (lenA != lenB) {
    throw ArgumentError(
      'Mismatched cross product axes sizes: axisa size $lenA != axisb size $lenB.',
    );
  }

  final is3D = lenA == 3;

  final stackA = List<int>.from(a.shape)..removeAt(axisA);
  final stackB = List<int>.from(b.shape)..removeAt(axisB);
  final broadcastStack = broadcastStackShapes(stackA, stackB);

  final expectedShape = List<int>.from(broadcastStack);
  if (is3D) {
    var finalAxisC = axisC;
    if (finalAxisC < 0) finalAxisC = expectedShape.length + 1 + finalAxisC;
    if (finalAxisC < 0 || finalAxisC > expectedShape.length) {
      finalAxisC = expectedShape.length;
    }
    expectedShape.insert(finalAxisC, 3);
    axisC = finalAxisC;
  }

  final targetDType = resolveDType(a.dtype, b.dtype);
  if (out != null) {
    if (!listEquals(out.shape, expectedShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out recycler has incompatible shape or dtype (expected shape $expectedShape and dtype $targetDType).',
      );
    }
  }

  final result =
      out ?? NDArray<R>.create(expectedShape, targetDType as DType<R>);

  if (targetDType == DType.float16 ||
      targetDType == DType.bfloat16 ||
      targetDType == DType.int8 ||
      targetDType == DType.uint64 ||
      targetDType == DType.uint32 ||
      targetDType == DType.uint16) {
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
    final casted = castNDArray(doubleRes, targetDType as DType<R>);
    casted.copy(out: result);
    doubleA.dispose();
    doubleB.dispose();
    doubleRes.dispose();
    casted.dispose();
    return result;
  }

  final aCast = castNDArray(a, targetDType);
  final bCast = castNDArray(b, targetDType);

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
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
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

  if (aCast != a) aCast.dispose();
  if (bCast != b) bCast.dispose();

  return result;
}

/// Supported norm orders and calculation modes for vector and matrix norm computations.

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

enum NormKind { frobenius, nuclear, l1, l2, infinity, negInfinity }

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
NDArray<double> norm<T extends Object>(
  NDArray<T> a, {
  dynamic ord,
  dynamic axis,
  bool keepdims = false,
  NDArray<double>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute norm() on a disposed array.');
  }

  final rank = a.shape.length;
  List<int> targetAxes;
  if (axis == null) {
    if (rank > 2) {
      throw ArgumentError(
        'Improper axis specification: If axis is null, input must be 1D or 2D.',
      );
    }
    targetAxes = List<int>.generate(rank, (i) => i);
  } else if (axis is int) {
    var normAx = axis;
    if (normAx < 0) normAx = rank + normAx;
    if (normAx < 0 || normAx >= rank) {
      throw ArgumentError('axis $axis is out of bounds.');
    }
    targetAxes = [normAx];
  } else if (axis is List<int>) {
    if (axis.length != 2) {
      throw ArgumentError('axis list must contain exactly 1 or 2 elements.');
    }
    final normAxes = List<int>.from(axis);
    for (var i = 0; i < 2; i++) {
      if (normAxes[i] < 0) normAxes[i] = rank + normAxes[i];
      if (normAxes[i] < 0 || normAxes[i] >= rank) {
        throw ArgumentError('axis ${axis[i]} is out of bounds.');
      }
    }
    if (normAxes[0] == normAxes[1]) {
      throw ArgumentError('axes must be distinct.');
    }
    targetAxes = normAxes;
  } else {
    throw ArgumentError('axis must be null, int, or List<int>.');
  }

  final isVecNorm = targetAxes.length == 1;
  final DType targetDType =
      (a.dtype == DType.float32 || a.dtype == DType.complex64)
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
    if (!listEquals(out.shape, expectedShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final result =
      out ??
      NDArray<double>.create(expectedShape, targetDType as DType<double>);

  if (targetAxes.length == rank && !keepdims) {
    // Global norm
    if (isVecNorm) {
      final val = _vectorNorm<T>(a, ord, targetDType);
      if (targetDType == DType.float32) {
        result.pointer.cast<ffi.Float>()[0] = val;
      } else {
        result.pointer.cast<ffi.Double>()[0] = val;
      }
    } else {
      final val = _matrixNorm<T>(a, ord, targetDType);
      if (targetDType == DType.float32) {
        result.pointer.cast<ffi.Float>()[0] = val;
      } else {
        result.pointer.cast<ffi.Double>()[0] = val;
      }
    }
    return result;
  }

  // Reduction along specific axes
  final List<int> currentCoords = List<int>.filled(a.shape.length, 0);

  final List<int> stackShape = List<int>.from(a.shape);
  final sortedAxes = List<int>.from(targetAxes)..sort((x, y) => y.compareTo(x));
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

      final NDArray<T> slice;
      if (isVecNorm) {
        final ax = targetAxes[0];
        final len = a.shape[ax];
        var offset = a.offsetElements;
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
        var offset = a.offsetElements;
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
        val = _vectorNorm<T>(slice, ord, targetDType);
      } else {
        val = _matrixNorm<T>(slice, ord, targetDType);
      }
      slice.dispose();

      // Calculate dest flat index
      var destOffset = result.offsetElements;
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

  return result;
}

double _vectorNorm<T>(NDArray<T> a, dynamic ord, DType targetDType) {
  if (ord is NormKind) {
    ord = switch (ord) {
      NormKind.l1 => 1,
      NormKind.l2 => 2,
      NormKind.infinity => double.infinity,
      NormKind.negInfinity => double.negativeInfinity,
      NormKind.frobenius || NormKind.nuclear => throw ArgumentError(
        'NormKind.${ord.name} is not valid for vectors',
      ),
    };
  }
  final bool needsCast = !a.dtype.isFloating && !a.dtype.isComplex;
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
      throw ArgumentError('Invalid vector norm order: $ord');
    }
  } finally {
    if (needsCast) castedA.dispose();
  }
}

double _matrixNorm<T extends Object>(
  NDArray<T> a,
  dynamic ord,
  DType targetDType,
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
      NormKind.l2 => 2,
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
        offsetElements: a.offsetElements + c * a.strides[1],
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
        offsetElements: a.offsetElements + c * a.strides[1],
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
        offsetElements: a.offsetElements + r * a.strides[0],
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
        offsetElements: a.offsetElements + r * a.strides[0],
      );
      final rowSum = _vectorNorm(rowSlice, 1, targetDType);
      rowSlice.dispose();
      if (rowSum < minRowSum) minRowSum = rowSum;
    }
    return minRowSum;
  } else if (ord == 2) {
    final svdRes = svd(a);
    final maxS = (svdRes.s.dtype == DType.float32)
        ? svdRes.s.pointer.cast<ffi.Float>()[0]
        : svdRes.s.pointer.cast<ffi.Double>()[0];
    svdRes.dispose();
    return maxS;
  } else if (ord == -2) {
    final svdRes = svd(a);
    final minS = (svdRes.s.dtype == DType.float32)
        ? svdRes.s.pointer.cast<ffi.Float>()[svdRes.s.shape[0] - 1]
        : svdRes.s.pointer.cast<ffi.Double>()[svdRes.s.shape[0] - 1];
    svdRes.dispose();
    return minS;
  } else if (ord == NormKind.nuclear) {
    final svdRes = svd(a);
    var sumS = 0.0;
    for (var i = 0; i < svdRes.s.shape[0]; i++) {
      sumS += (svdRes.s.dtype == DType.float32)
          ? svdRes.s.pointer.cast<ffi.Float>()[i]
          : svdRes.s.pointer.cast<ffi.Double>()[i];
    }
    svdRes.dispose();
    return sumS;
  } else {
    throw ArgumentError('Invalid matrix norm order: $ord');
  }
}

extension QRRecordDispose<T> on ({NDArray<T> q, NDArray<T> r}) {
  void dispose() {
    this.q.dispose();
    this.r.dispose();
  }
}

extension SVDRecordDispose<T>
    on ({NDArray<T> u, NDArray<double> s, NDArray<T> vh}) {
  void dispose() {
    this.u.dispose();
    this.s.dispose();
    this.vh.dispose();
  }
}

extension SchurRecordDispose<T> on ({NDArray<T> t, NDArray<T> z}) {
  void dispose() {
    this.t.dispose();
    this.z.dispose();
  }
}

extension HessenbergRecordDispose<T> on ({NDArray<T> h, NDArray<T> q}) {
  void dispose() {
    this.h.dispose();
    this.q.dispose();
  }
}

/// Result record of a least-squares linear system solution from [lstsq].
typedef LstsqResult<T> = ({
  NDArray<T> x,
  NDArray<double> residuals,
  int rank,
  NDArray<double> s,
});

/// Extension on [LstsqResult] to support easy disposal of all returned unmanaged buffers.
extension LstsqResultDispose<T> on LstsqResult<T> {
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
LstsqResult<R> lstsq<Ta, Tb, R>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  double? rcond,
  NDArray<R>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute lstsq() on a disposed array.');
  }

  final targetDType = (a.dtype.isInteger && b.dtype.isInteger)
      ? DType.float64
      : (a.dtype.isInteger
            ? (b.dtype == DType.float32 ? DType.float32 : DType.float64)
            : (b.dtype.isInteger
                  ? (a.dtype == DType.float32 ? DType.float32 : DType.float64)
                  : resolveDType(a.dtype, b.dtype)));

  if (!targetDType.isFloating && !targetDType.isComplex) {
    throw ArgumentError('lstsq requires floating-point or complex inputs.');
  }

  final aUse = a.dtype == targetDType ? a : castNDArray(a, targetDType);
  final bUse = b.dtype == targetDType ? b : castNDArray(b, targetDType);
  final wasACast = aUse != a;
  final wasBCast = bUse != b;

  if (aUse.shape.length != 2) {
    if (wasACast) aUse.dispose();
    if (wasBCast) bUse.dispose();
    throw ArgumentError(
      'Input matrix a must be 2-dimensional (was shape ${a.shape}).',
    );
  }
  if (bUse.shape.length != 1 && bUse.shape.length != 2) {
    if (wasACast) aUse.dispose();
    if (wasBCast) bUse.dispose();
    throw ArgumentError(
      'Input right-hand side b must be 1D or 2D (was shape ${b.shape}).',
    );
  }
  final m = aUse.shape[0];
  final n = aUse.shape[1];
  if (bUse.shape[0] != m) {
    if (wasACast) aUse.dispose();
    if (wasBCast) bUse.dispose();
    throw ArgumentError(
      'First dimension of b (${bUse.shape[0]}) must match first dimension of a ($m).',
    );
  }

  final nrhs = bUse.shape.length > 1 ? bUse.shape[1] : 1;

  if (out != null) {
    if (out.isDisposed) {
      if (wasACast) aUse.dispose();
      if (wasBCast) bUse.dispose();
      throw StateError('Cannot write to a disposed out buffer.');
    }
    final expectedXShape = bUse.shape.length > 1 ? [n, nrhs] : [n];
    if (!listEquals(out.shape, expectedXShape) || out.dtype != targetDType) {
      if (wasACast) aUse.dispose();
      if (wasBCast) bUse.dispose();
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  return NDArray.scope(() {
    if (m == 0 || n == 0) {
      final xShape = bUse.shape.length > 1 ? [n, nrhs] : [n];
      final x = out ?? NDArray<R>.zeros(xShape, targetDType as DType<R>);
      final sDType =
          (targetDType == DType.complex64 || targetDType == DType.float32)
          ? DType.float32
          : DType.float64;
      final s = NDArray<double>.zeros([0], sDType as dynamic);
      final residuals = NDArray<double>.zeros([0], sDType as dynamic);
      if (out == null) x.detachToParentScope();
      s.detachToParentScope();
      residuals.detachToParentScope();
      return (x: x, residuals: residuals, rank: 0, s: s);
    }

    // Create a contiguous copy of a (overwrite-safe)
    final aCopy = aUse.copy();

    // Row-major LAPACKE_gelsd requires b array size to be max(m, n) * nrhs
    final maxMN = m > n ? m : n;
    final bCopyShape = bUse.shape.length > 1 ? [maxMN, nrhs] : [maxMN];
    final bCopy = NDArray.zeros(bCopyShape, targetDType);

    // Copy b into bCopy
    final byteCount = bUse.size * targetDType.byteWidth;
    if (bUse.isContiguous) {
      ffi.Pointer.fromAddress(bCopy.pointer.address)
          .cast<ffi.Uint8>()
          .asTypedList(byteCount)
          .setAll(
            0,
            ffi.Pointer.fromAddress(bUse.pointer.address)
                .cast<ffi.Uint8>()
                .asTypedList(byteCount),
          );
    } else {
      final bContig = bUse.copy();
      ffi.Pointer.fromAddress(bCopy.pointer.address)
          .cast<ffi.Uint8>()
          .asTypedList(byteCount)
          .setAll(
            0,
            ffi.Pointer.fromAddress(bContig.pointer.address)
                .cast<ffi.Uint8>()
                .asTypedList(byteCount),
          );
      bContig.dispose();
    }

    final minMN = m < n ? m : n;
    // Singular values s is always real
    final sDType =
        (targetDType == DType.complex64 || targetDType == DType.float32)
        ? DType.float32
        : DType.float64;
    final s = NDArray<double>.zeros([minMN], sDType as dynamic);
    final marker = ScratchArena.marker;
    final rankPtr = ScratchArena.allocate<ffi.Int>(ffi.sizeOf<ffi.Int>());
    final rcondVal = rcond ?? -1.0;

    try {
      int info;
      switch (targetDType) {
        case DType.float64:
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

      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE gelsd: $info');
      }
      if (info > 0) {
        throw IterationsExceededException(
          'The SVD algorithm in LAPACKE gelsd failed to converge ($info).',
        );
      }

      final rank = rankPtr[0];

      // Extract solution x: first n rows of bCopy
      final xShape = bUse.shape.length > 1 ? [n, nrhs] : [n];
      final NDArray<R> x =
          (out ?? NDArray<R>.zeros(xShape, targetDType as DType<R>));
      final bCopySlice = NDArray.view(
        bCopy,
        shape: xShape,
        strides: bCopy.strides.sublist(bCopy.shape.length - xShape.length),
        offsetElements: 0,
      );
      bCopySlice.copy(out: x);
      bCopySlice.dispose();

      // Extract residuals: sum of squares of elements from row n to m-1 for each column
      final NDArray<double> residuals;
      if (m > n && rank == n) {
        final resShape = bUse.shape.length > 1 ? [nrhs] : [1];
        residuals = NDArray<double>.zeros(resShape, sDType as dynamic);
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
        residuals = NDArray<double>.zeros([0], sDType as dynamic);
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
