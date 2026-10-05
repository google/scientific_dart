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

import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/linalg.dart' hide diagonal, trace;
import 'package:test/test.dart';

void main() {
  group('Linear Algebra Decompositions (F8)', () {
    test('svd returns named-field record and reconstructs matrix', () {
      final matrix = GpuArray.fromList(
        <double>[3.0, 0.0, 0.0, -2.0],
        [2, 2],
        DType.float64,
      );
      try {
        final result = svd(matrix);
        try {
          expect(result.u.shape, equals(<int>[2, 2]));
          expect(result.s.shape, equals(<int>[2]));
          expect(result.vt.shape, equals(<int>[2, 2]));
          final singularValues = result.s.toList().cast<double>();
          expect(singularValues[0], closeTo(3.0, 1e-10));
          expect(singularValues[1], closeTo(2.0, 1e-10));
        } finally {
          result.dispose();
        }
        expect(result.u.isDisposed, isTrue);
        expect(result.s.isDisposed, isTrue);
        expect(result.vt.isDisposed, isTrue);
      } finally {
        matrix.dispose();
      }
    });

    test('svdValues supports out: parameter', () {
      final matrix = GpuArray.fromList(
        <double>[4.0, 0.0, 0.0, 5.0],
        [2, 2],
        DType.float64,
      );
      final outSingularValues = GpuArray.zeros([2], DType.float64);
      try {
        final result = svdValues(matrix, out: outSingularValues);
        expect(identical(result, outSingularValues), isTrue);
        final values = result.toList().cast<double>();
        expect(values[0], closeTo(5.0, 1e-10));
        expect(values[1], closeTo(4.0, 1e-10));
      } finally {
        outSingularValues.dispose();
        matrix.dispose();
      }
    });

    test('qr supports QrMode.reduced and QrMode.complete', () {
      final matrix = GpuArray.fromList(
        <double>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [3, 2],
        DType.float64,
      );
      try {
        final reduced = qr(matrix, mode: QrMode.reduced);
        try {
          expect(reduced.q.shape, equals(<int>[3, 2]));
          expect(reduced.r.shape, equals(<int>[2, 2]));
          final reconstructed = matmul(reduced.q, reduced.r);
          try {
            final actual = reconstructed.toList().cast<double>();
            final expected = matrix.toList().cast<double>();
            for (var i = 0; i < expected.length; i++) {
              expect(actual[i], closeTo(expected[i], 1e-10));
            }
          } finally {
            reconstructed.dispose();
          }
        } finally {
          reduced.dispose();
        }

        final complete = qr(matrix, mode: QrMode.complete);
        try {
          expect(complete.q.shape, equals(<int>[3, 3]));
          expect(complete.r.shape, equals(<int>[3, 2]));
        } finally {
          complete.dispose();
        }
      } finally {
        matrix.dispose();
      }
    });

    test('cholesky supports MatrixTriangle.lower and MatrixTriangle.upper', () {
      final spd = GpuArray.fromList(
        <double>[4.0, 2.0, 2.0, 5.0],
        [2, 2],
        DType.float64,
      );
      final outFactor = GpuArray.zeros([2, 2], DType.float64);
      try {
        final lower = cholesky(spd, uplo: MatrixTriangle.lower, out: outFactor);
        expect(identical(lower, outFactor), isTrue);
        final lowerValues = lower.toList().cast<double>();
        expect(lowerValues[0], closeTo(2.0, 1e-10));
        expect(lowerValues[1], closeTo(0.0, 1e-10));
        expect(lowerValues[2], closeTo(1.0, 1e-10));
        expect(lowerValues[3], closeTo(2.0, 1e-10));

        final upper = cholesky(spd, uplo: MatrixTriangle.upper);
        try {
          final upperValues = upper.toList().cast<double>();
          expect(upperValues[0], closeTo(2.0, 1e-10));
          expect(upperValues[1], closeTo(1.0, 1e-10));
          expect(upperValues[2], closeTo(0.0, 1e-10));
          expect(upperValues[3], closeTo(2.0, 1e-10));
        } finally {
          upper.dispose();
        }
      } finally {
        outFactor.dispose();
        spd.dispose();
      }
    });

    test(
      'eigh and eigvalsh compute symmetric eigenvalues and eigenvectors',
      () {
        final symmetric = GpuArray.fromList(
          <double>[2.0, 1.0, 1.0, 2.0],
          [2, 2],
          DType.float64,
        );
        final outEigenvalues = GpuArray.zeros([2], DType.float64);
        try {
          final result = eigh(symmetric, uplo: MatrixTriangle.lower);
          try {
            final evals = result.eigenvalues.toList().cast<double>();
            expect(evals[0], closeTo(1.0, 1e-10));
            expect(evals[1], closeTo(3.0, 1e-10));
          } finally {
            result.dispose();
          }

          final vals = eigvalsh(symmetric, out: outEigenvalues);
          expect(identical(vals, outEigenvalues), isTrue);
          final valList = vals.toList().cast<double>();
          expect(valList[0], closeTo(1.0, 1e-10));
          expect(valList[1], closeTo(3.0, 1e-10));
        } finally {
          outEigenvalues.dispose();
          symmetric.dispose();
        }
      },
    );

    test('eig and eigvals compute general complex eigenvalues', () {
      final rotation = GpuArray.fromList(
        <double>[0.0, -1.0, 1.0, 0.0],
        [2, 2],
        DType.float64,
      );
      try {
        final result = eig(rotation);
        try {
          expect(result.eigenvalues.dtype, equals(DType.complex128));
          expect(result.eigenvectors.dtype, equals(DType.complex128));
          final evals = result.eigenvalues.toList().cast<Complex>();
          expect(evals[0].real, closeTo(0.0, 1e-10));
          expect(evals[0].imag.abs(), closeTo(1.0, 1e-10));
        } finally {
          result.dispose();
        }

        final vals = eigvals(rotation);
        try {
          expect(vals.shape, equals(<int>[2]));
        } finally {
          vals.dispose();
        }
      } finally {
        rotation.dispose();
      }
    });

    test(
      'lu and luFactor decompose matrix into P, L, U and packed factors',
      () {
        final matrix = GpuArray.fromList(
          <double>[2.0, 1.0, 4.0, 3.0],
          [2, 2],
          DType.float64,
        );
        try {
          final luDecomp = lu(matrix);
          try {
            final pl = matmul(luDecomp.p, luDecomp.l);
            final plu = matmul(pl, luDecomp.u);
            try {
              final actual = plu.toList().cast<double>();
              final expected = matrix.toList().cast<double>();
              for (var i = 0; i < expected.length; i++) {
                expect(actual[i], closeTo(expected[i], 1e-10));
              }
            } finally {
              pl.dispose();
              plu.dispose();
            }
          } finally {
            luDecomp.dispose();
          }

          final factor = luFactor(matrix);
          try {
            expect(factor.lu.shape, equals(<int>[2, 2]));
            expect(factor.pivots.shape, equals(<int>[2]));
          } finally {
            factor.dispose();
          }
        } finally {
          matrix.dispose();
        }
      },
    );

    test('qrR and svdValues compute R factor and singular values', () {
      final matrix = GpuArray.fromList(
        <double>[3.0, 0.0, 0.0, 4.0],
        [2, 2],
        DType.float64,
      );
      try {
        final rFactor = qrR(matrix);
        final singularVals = svdValues(matrix);
        try {
          expect(rFactor.shape, equals(<int>[2, 2]));
          final sList = singularVals.toList().cast<double>();
          expect(sList[0], closeTo(4.0, 1e-10));
          expect(sList[1], closeTo(3.0, 1e-10));
        } finally {
          rFactor.dispose();
          singularVals.dispose();
        }
      } finally {
        matrix.dispose();
      }
    });

    test(
      'Float32 decompositions preserve Float32 and Complex64 dtypes (F11)',
      () {
        final spdF32 = GpuArray.fromList(
          <double>[4.0, 2.0, 2.0, 5.0],
          [2, 2],
          DType.float32,
        );
        try {
          final svdRes = svd(spdF32);
          try {
            expect(svdRes.u.dtype, equals(DType.float32));
            expect(svdRes.s.dtype, equals(DType.float32));
            expect(svdRes.vt.dtype, equals(DType.float32));
          } finally {
            svdRes.dispose();
          }

          final sVals = svdValues(spdF32);
          try {
            expect(sVals.dtype, equals(DType.float32));
          } finally {
            sVals.dispose();
          }

          final qrRes = qr(spdF32);
          try {
            expect(qrRes.q.dtype, equals(DType.float32));
            expect(qrRes.r.dtype, equals(DType.float32));
          } finally {
            qrRes.dispose();
          }

          final rOnly = qrR(spdF32);
          try {
            expect(rOnly.dtype, equals(DType.float32));
          } finally {
            rOnly.dispose();
          }

          final cholRes = cholesky(spdF32);
          try {
            expect(cholRes.dtype, equals(DType.float32));
            final vals = cholRes.toList().cast<double>();
            expect(vals[0], closeTo(2.0, 1e-5));
            expect(vals[2], closeTo(1.0, 1e-5));
            expect(vals[3], closeTo(2.0, 1e-5));
          } finally {
            cholRes.dispose();
          }

          final eighRes = eigh(spdF32);
          try {
            expect(eighRes.eigenvalues.dtype, equals(DType.float32));
            expect(eighRes.eigenvectors.dtype, equals(DType.float32));
          } finally {
            eighRes.dispose();
          }

          final evalshRes = eigvalsh(spdF32);
          try {
            expect(evalshRes.dtype, equals(DType.float32));
          } finally {
            evalshRes.dispose();
          }

          final eigRes = eig(spdF32);
          try {
            expect(eigRes.eigenvalues.dtype, equals(DType.complex64));
            expect(eigRes.eigenvectors.dtype, equals(DType.complex64));
          } finally {
            eigRes.dispose();
          }

          final evalsRes = eigvals(spdF32);
          try {
            expect(evalsRes.dtype, equals(DType.complex64));
          } finally {
            evalsRes.dispose();
          }

          final luRes = lu(spdF32);
          try {
            expect(luRes.p.dtype, equals(DType.float32));
            expect(luRes.l.dtype, equals(DType.float32));
            expect(luRes.u.dtype, equals(DType.float32));
          } finally {
            luRes.dispose();
          }

          final luFacRes = luFactor(spdF32);
          try {
            expect(luFacRes.lu.dtype, equals(DType.float32));
            expect(luFacRes.pivots.dtype, equals(DType.int32));
          } finally {
            luFacRes.dispose();
          }
        } finally {
          spdF32.dispose();
        }
      },
    );
  });
}
