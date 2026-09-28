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

import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  group('Workstream 4 - Exceptions Hierarchy', () {
    test('LinAlgException subclasses hierarchy', () {
      expect(const SingularMatrixException('test'), isA<LinAlgException>());
      expect(const SingularMatrixException('test'), isA<NdArrayException>());
      expect(const SingularMatrixException('test'), isA<Exception>());

      expect(const IterationsExceededException('test'), isA<LinAlgException>());
      expect(
        const IterationsExceededException('test'),
        isA<NdArrayException>(),
      );

      expect(
        const NonPositiveDefiniteException('test'),
        isA<LinAlgException>(),
      );
      expect(
        const NonPositiveDefiniteException('test'),
        isA<NdArrayException>(),
      );
    });

    test(
      'cholesky throws NonPositiveDefiniteException on non-positive definite matrix',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [-1.0, 0.0, 0.0, -1.0],
            [2, 2],
            DType.float64,
          );
          expect(
            () => cholesky(a),
            throwsA(isA<NonPositiveDefiniteException>()),
          );
        });
      },
    );
  });

  group('Workstream 4 - Einsum Ellipsis Validation', () {
    test(
      'einsum throws ArgumentError when operand contains multiple ellipses',
      () {
        expect(
          () => EinsumSubscripts.parse('...i...->i'),
          throwsA(isA<ArgumentError>()),
        );
        expect(
          () => EinsumSubscripts.parse('i...j...->ij'),
          throwsA(isA<ArgumentError>()),
        );
      },
    );

    test(
      'einsum throws ArgumentError when output term contains multiple ellipses',
      () {
        expect(
          () => EinsumSubscripts.parse('...ij->...i...j'),
          throwsA(isA<ArgumentError>()),
        );
      },
    );

    test('EinsumSubscripts.fromIndices throws on multiple -1s in operand', () {
      expect(
        () => EinsumSubscripts.fromIndices(
          [
            [-1, 0, -1],
          ],
          [0],
        ),
        throwsA(isA<ArgumentError>()),
      );
      expect(
        () => EinsumSubscripts.fromIndices(
          [
            [0, 1],
          ],
          [-1, 0, -1],
        ),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('EinsumSubscripts.fromLabels throws on multiple "..." in operand', () {
      expect(
        () => EinsumSubscripts.fromLabels(
          [
            ['...', 'i', '...'],
          ],
          ['i'],
        ),
        throwsA(isA<ArgumentError>()),
      );
      expect(
        () => EinsumSubscripts.fromLabels(
          [
            ['i', 'j'],
          ],
          ['...', 'i', '...'],
        ),
        throwsA(isA<ArgumentError>()),
      );
    });
  });

  group('Workstream 4 - inner()', () {
    test('1-D vector inner product', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final b = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);
        final res = inner(a, b);
        expect(res.shape, equals([]));
        expect(res.scalar, closeTo(32.0, 1e-12));
      });
    });

    test('1-D with out argument', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final b = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);
        final out = NDArray<Float64>.zeros([], DType.float64);
        final res = inner(a, b, out: out);
        expect(identical(res, out), isTrue);
        expect(out.scalar, closeTo(32.0, 1e-12));
      });
    });

    test('2-D x 2-D inner product (contracts last axes)', () {
      NDArray.scope(() {
        // a is 2x3, b is 4x3 -> inner is 2x4 where res[i, j] = sum_k a[i, k] * b[j, k]
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final b = NDArray.fromList(
          [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0],
          [4, 3],
          DType.float64,
        );

        final res = inner(a, b);
        expect(res.shape, equals([2, 4]));
        // Row 0 of A: [1, 2, 3]
        // dot with B[0]: 1*1 + 2*0 + 3*1 = 4
        // dot with B[1]: 1*0 + 2*1 + 3*0 = 2
        // dot with B[2]: 1*2 + 2*1 + 3*0 = 4
        // dot with B[3]: 1*1 + 2*1 + 3*1 = 6
        expect(res[[0, 0]], closeTo(4.0, 1e-12));
        expect(res[[0, 1]], closeTo(2.0, 1e-12));
        expect(res[[0, 2]], closeTo(4.0, 1e-12));
        expect(res[[0, 3]], closeTo(6.0, 1e-12));

        // Row 1 of A: [4, 5, 6]
        // dot with B[0]: 4*1 + 5*0 + 6*1 = 10
        // dot with B[1]: 4*0 + 5*1 + 6*0 = 5
        // dot with B[2]: 4*2 + 5*1 + 6*0 = 13
        // dot with B[3]: 4*1 + 5*1 + 6*1 = 15
        expect(res[[1, 0]], closeTo(10.0, 1e-12));
        expect(res[[1, 1]], closeTo(5.0, 1e-12));
        expect(res[[1, 2]], closeTo(13.0, 1e-12));
        expect(res[[1, 3]], closeTo(15.0, 1e-12));
      });
    });

    test('0-D scalar inner product', () {
      NDArray.scope(() {
        final a = NDArray.fromList([3.0], [], DType.float64);
        final b = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);
        final res = inner(a, b);
        expect(res.shape, equals([3]));
        expect(res.toList(), equals([12.0, 15.0, 18.0]));
      });
    });

    test('inner throws on mismatched last dimension', () {
      NDArray.scope(() {
        final a = NDArray.zeros([2, 3], DType.float64);
        final b = NDArray.zeros([2, 4], DType.float64);
        expect(() => inner(a, b), throwsA(isA<ArgumentError>()));
      });
    });
  });

  group('Workstream 4 - vdot()', () {
    test('1-D complex vdot uses conjugate of first operand', () {
      NDArray.scope(() {
        // a = [1 + 2i, 3 + 4i], b = [1 - 2i, 3 - 4i]
        // conj(a) = [1 - 2i, 3 - 4i]
        // conj(a) . b = (1 - 2i)^2 + (3 - 4i)^2
        //             = (1 - 4 - 4i) + (9 - 16 - 24i)
        //             = (-3 - 4i) + (-7 - 24i) = -10 - 28i
        final a = NDArray.fromList(
          [Complex(1.0, 2.0), Complex(3.0, 4.0)],
          [2],
          DType.complex128,
        );
        final b = NDArray.fromList(
          [Complex(1.0, -2.0), Complex(3.0, -4.0)],
          [2],
          DType.complex128,
        );

        final res = vdot(a, b);
        expect(res.shape, equals([]));
        final cVal = res.scalar;
        expect(cVal.real, closeTo(-10.0, 1e-12));
        expect(cVal.imag, closeTo(-28.0, 1e-12));
      });
    });

    test('multidimensional array vdot flattens operands', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList([1.0, 1.0, 1.0, 1.0], [4], DType.float64);

        final res = vdot(a, b);
        expect(res.shape, equals([]));
        expect(res.scalar, closeTo(10.0, 1e-12));
      });
    });

    test('vdot with out parameter', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final b = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);
        final out = NDArray<Float64>.zeros([], DType.float64);
        final res = vdot(a, b, out: out);
        expect(identical(res, out), isTrue);
        expect(out.scalar, closeTo(32.0, 1e-12));
      });
    });

    test('vdot throws when element counts differ', () {
      NDArray.scope(() {
        final a = NDArray.zeros([2, 3], DType.float64);
        final b = NDArray.zeros([5], DType.float64);
        expect(() => vdot(a, b), throwsA(isA<ArgumentError>()));
      });
    });
  });

  group('Workstream 4 - Matrix Enums', () {
    test('eigh with MatrixTriangle', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 2.0, 3.0], [2, 2], DType.float64);

        final resLower = eigh(a, uplo: MatrixTriangle.lower);
        final resUpper = eigh(a, uplo: MatrixTriangle.upper);
        expect(resLower.eigenvalues.shape, equals([2]));
        expect(resUpper.eigenvalues.shape, equals([2]));
      });
    });

    test('eigvalsh with MatrixTriangle', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 2.0, 3.0], [2, 2], DType.float64);

        final wLower = eigvalsh(a, uplo: MatrixTriangle.lower);
        final wUpper = eigvalsh(a, uplo: MatrixTriangle.upper);
        expect(wLower.shape, equals([2]));
        expect(wUpper.shape, equals([2]));
      });
    });

    test('schur with SchurForm', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [5.0, 7.0, -2.0, -4.0],
          [2, 2],
          DType.float64,
        );

        final resReal = schur(a, output: SchurForm.real);
        expect(resReal.t.dtype, equals(DType.float64));

        final resComplex = schur(a, output: SchurForm.complex);
        expect(resComplex.t.dtype, equals(DType.complex128));
      });
    });
  });

  group('Workstream 4 - eigvals Integer Promotion', () {
    test('eigvals auto-promotes int64 and uint8', () {
      NDArray.scope(() {
        final aInt = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
        final w = eigvals(aInt);
        final reals = w.toList().map((c) => c.real).toList()..sort();
        expect(reals[0], closeTo(-0.3722813232690143, 1e-6));
        expect(reals[1], closeTo(5.372281323269014, 1e-6));
      });
    });
  });

  group('Workstream 4 - matmul out Direct Dispatch', () {
    test(
      'matmul writes directly into contiguous out buffer matching resShape',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [2.0, 0.0, 1.0, 2.0],
            [2, 2],
            DType.float64,
          );
          final out = NDArray<Float64>.zeros([2, 2], DType.float64);

          final res = matmul(a, b, out: out);
          expect(identical(res, out), isTrue);
          expect(out[[0, 0]], closeTo(4.0, 1e-12));
          expect(out[[0, 1]], closeTo(4.0, 1e-12));
          expect(out[[1, 0]], closeTo(10.0, 1e-12));
          expect(out[[1, 1]], closeTo(8.0, 1e-12));
        });
      },
    );
  });

  group('Workstream 4 - SVD Transposed View Disposal', () {
    test('svd transposed view does not leak memory or cause double free', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );
        final res = svd(a);
        expect(res.u.shape, equals([3, 3]));
        expect(res.s.shape, equals([2]));
        expect(res.vh.shape, equals([2, 2]));

        // Reconstruct A = U[:, :2] * S * Vh
        final uTrunc = res.u.slice([Slice.all(), Slice(start: 0, stop: 2)]);
        final sDiag = NDArray<Float64>.zeros([2, 2], DType.float64);
        sDiag[[0, 0]] = res.s[[0]];
        sDiag[[1, 1]] = res.s[[1]];
        final recon = matmul(matmul(uTrunc, sDiag), res.vh);

        for (var r = 0; r < 3; r++) {
          for (var c = 0; c < 2; c++) {
            expect(recon[[r, c]], closeTo(a[[r, c]], 1e-10));
          }
        }
      });
    });
  });

  group('Workstream 4 - matrix_power', () {
    test('matrix_power n = 0 is identity', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final res = matrix_power(a, 0);
        expect(res[[0, 0]], closeTo(1.0, 1e-12));
        expect(res[[0, 1]], closeTo(0.0, 1e-12));
        expect(res[[1, 0]], closeTo(0.0, 1e-12));
        expect(res[[1, 1]], closeTo(1.0, 1e-12));
      });
    });

    test('matrix_power positive powers', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final a2 = matrix_power(a, 2);
        final expectedA2 = matmul(a, a);
        for (var r = 0; r < 2; r++) {
          for (var c = 0; c < 2; c++) {
            expect(a2[[r, c]], closeTo(expectedA2[[r, c]], 1e-12));
          }
        }

        final a3 = matrix_power(a, 3);
        final expectedA3 = matmul(expectedA2, a);
        for (var r = 0; r < 2; r++) {
          for (var c = 0; c < 2; c++) {
            expect(a3[[r, c]], closeTo(expectedA3[[r, c]], 1e-12));
          }
        }
      });
    });

    test('matrix_power negative power is inverse power', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final aInv = matrix_power(a, -1);
        final eye = matmul(a, aInv);
        expect(eye[[0, 0]], closeTo(1.0, 1e-10));
        expect(eye[[0, 1]], closeTo(0.0, 1e-10));
        expect(eye[[1, 0]], closeTo(0.0, 1e-10));
        expect(eye[[1, 1]], closeTo(1.0, 1e-10));
      });
    });
  });

  group('Workstream 4 - Scope Safety for norm, outer, cross, lstsq', () {
    test('norm inside and outside scope', () {
      final a = NDArray.fromList([3.0, 4.0], [2], DType.float64);
      final n = norm(a);
      expect(n.scalar, closeTo(5.0, 1e-12));

      NDArray.scope(() {
        final nScoped = norm(a);
        expect(nScoped.scalar, closeTo(5.0, 1e-12));
      });
      a.dispose();
      n.dispose();
    });

    test('outer inside and outside scope', () {
      final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
      final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);
      final outRes = outer(a, b);
      expect(outRes.shape, equals([2, 2]));

      NDArray.scope(() {
        final outScoped = outer(a, b);
        expect(outScoped.shape, equals([2, 2]));
      });
      a.dispose();
      b.dispose();
      outRes.dispose();
    });
  });
}
