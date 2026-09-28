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

import "package:ndarray/ndarray.dart";
import "package:test/test.dart";

import "dart:typed_data";

void main() {
  group("Kronecker Product (kron)", () {
    test("computes 2D x 2D matrix Kronecker product for Float64", () {
      NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([0.0, 5.0, 6.0, 7.0]), [
          2,
          2,
        ], DType.float64);

        final res = kron(a, b);
        expect(res.shape, equals([4, 4]));
        expect(res.dtype, equals(DType.float64));
        expect(
          res.toList(),
          equals([
            0.0,
            5.0,
            0.0,
            10.0,
            6.0,
            7.0,
            12.0,
            14.0,
            0.0,
            15.0,
            0.0,
            20.0,
            18.0,
            21.0,
            24.0,
            28.0,
          ]),
        );
      });
    });

    test("handles 1D and different rank arrays", () {
      NDArray.scope(() {
        final a = NDArray.fromList(Int64List.fromList([1, 2]), [
          2,
        ], DType.int64);
        final b = NDArray.fromList(Int64List.fromList([3, 4, 5]), [
          3,
        ], DType.int64);

        final res = kron(a, b);
        expect(res.shape, equals([6]));
        expect(res.toList(), equals([3, 4, 5, 6, 8, 10]));
      });
    });

    test("supports out argument", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);
        final out = NDArray.zeros([4], DType.float64);

        final res = kron(a, b, out: out);
        expect(identical(res, out), isTrue);
        expect(out.toList(), equals([3.0, 4.0, 6.0, 8.0]));
      });
    });

    test("throws for disposed or incompatible out", () {
      final a = NDArray.fromList([1.0], [1], DType.float64);
      final b = NDArray.fromList([2.0], [1], DType.float64);
      final wrongOut = NDArray.zeros([2], DType.float64);

      expect(() => kron(a, b, out: wrongOut), throwsArgumentError);

      a.dispose();
      expect(() => kron(a, b), throwsStateError);
      b.dispose();
    });
  });

  group("Tensor Dot Product (tensordot)", () {
    test("computes tensordot over last 2 axes of A and first 2 axes of B", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList([1.0, 0.0, 0.0, 1.0], [2, 2], DType.float64);

        final res = tensordot(a, b, axes: const TensordotAxes.count(2));
        expect(res.shape, equals([]));
        expect(res.scalar, equals(5.0));
      });
    });

    test("computes matrix contraction with TensordotAxes.explicit and TensordotAxes.pair", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList([5.0, 6.0, 7.0, 8.0], [2, 2], DType.float64);

        final resExplicit = tensordot(
          a,
          b,
          axes: TensordotAxes.explicit([1], [0]),
        );
        expect(resExplicit.shape, equals([2, 2]));
        expect(resExplicit.getCell([0, 0]), equals(19.0));
        expect(resExplicit.getCell([0, 1]), equals(22.0));

        final resPair = tensordot(a, b, axes: TensordotAxes.pair(1, 0));

        expect(resPair.shape, equals([2, 2]));
        expect(resPair.getCell([0, 0]), equals(19.0));
      });
    });

    test("supports out argument for tensordot", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList([1.0, 0.0, 0.0, 1.0], [2, 2], DType.float64);
        final out = NDArray.zeros([2, 2], DType.float64);

        final res = tensordot(
          a,
          b,
          axes: TensordotAxes.explicit([1], [0]),
          out: out,
        );
        expect(identical(res, out), isTrue);
        expect(out.getCell([0, 0]), equals(1.0));
        expect(out.getCell([1, 1]), equals(4.0));
      });
    });
    test(
      "fast path when contracting all axes of two contiguous Float64 arrays",
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [2.0, 3.0, 4.0, 5.0],
            [2, 2],
            DType.float64,
          );
          final res = tensordot(a, b, axes: const TensordotAxes.count(2));
          expect(res.shape, equals([]));
          expect(
            res.scalar,
            equals(40.0),
          ); // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40

          final out = NDArray.zeros([], DType.float64);
          final resOut = tensordot(
            a,
            b,
            axes: const TensordotAxes.count(2),
            out: out,
          );
          expect(identical(resOut, out), isTrue);
          expect(out.scalar, equals(40.0));
        });
      },
    );

    test(
      "fast path when contracting all axes of two contiguous Float32 arrays",
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float32);
          final b = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float32);
          final res = tensordot(a, b, axes: const TensordotAxes.count(1));
          expect(res.shape, equals([]));
          expect(res.scalar, equals(32.0)); // 1*4 + 2*5 + 3*6 = 32

          final out = NDArray.zeros([], DType.float32);
          final resOut = tensordot(
            a,
            b,
            axes: const TensordotAxes.count(1),
            out: out,
          );
          expect(identical(resOut, out), isTrue);
          expect(out.scalar, equals(32.0));
        });
      },
    );
    test("flexible axes parameter types: int, records, and lists", () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final b = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );

        // 1. int N
        final resInt = tensordot(a, b, axes: 1);
        expect(resInt.shape, equals([2, 2]));
        expect(resInt.getCell([0, 0]), equals(22.0)); // 1*1 + 2*3 + 3*5 = 22

        // 2. (int, int) record pair
        final resRecordPair = tensordot(a, b, axes: (1, 0));
        expect(resRecordPair.shape, equals([2, 2]));
        expect(resRecordPair.getCell([0, 0]), equals(22.0));

        // 3. (List<int>, List<int>) record pair
        final resRecordListPair = tensordot(a, b, axes: ([1], [0]));
        expect(resRecordListPair.shape, equals([2, 2]));
        expect(resRecordListPair.getCell([0, 0]), equals(22.0));

        // 4. List<List<int>>
        final resListList = tensordot(
          a,
          b,
          axes: [
            [1],
            [0],
          ],
        );
        expect(resListList.shape, equals([2, 2]));
        expect(resListList.getCell([0, 0]), equals(22.0));

        // 5. Throws for invalid axes parameter format
        expect(() => tensordot(a, b, axes: "invalid"), throwsArgumentError);
        expect(() => tensordot(a, b, axes: [1, 2, 3]), throwsArgumentError);
      });
    });

    test("axes=0 outer product fast path", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([3.0, 4.0, 5.0], [3], DType.float64);

        // 1. int 0
        final res0 = tensordot(a, b, axes: 0);
        expect(res0.shape, equals([2, 3]));
        expect(res0.getCell([0, 0]), equals(3.0));
        expect(res0.getCell([0, 1]), equals(4.0));
        expect(res0.getCell([1, 2]), equals(10.0));

        // 2. axes=0 with out parameter
        final out = NDArray.zeros([2, 3], DType.float64);
        final resOut = tensordot(a, b, axes: 0, out: out);
        expect(identical(resOut, out), isTrue);
        expect(out.getCell([1, 2]), equals(10.0));

        // 3. 2D x 2D outer product (axes=0)
        final a2D = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final b2D = NDArray.fromList([5.0, 6.0], [2, 1], DType.float64);
        final res2D0 = tensordot(a2D, b2D, axes: 0);
        expect(res2D0.shape, equals([2, 2, 2, 1]));
        expect(res2D0.getCell([0, 0, 0, 0]), equals(5.0)); // 1*5
        expect(res2D0.getCell([1, 1, 1, 0]), equals(24.0)); // 4*6
      });
    });
  });

  group("Einstein Summation (einsum)", () {
    test("matrix multiplication ij,jk->ik", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList([5.0, 6.0, 7.0, 8.0], [2, 2], DType.float64);

        final res = einsum(EinsumSubscripts.parse("ij,jk->ik"), [a, b]);
        expect(res.shape, equals([2, 2]));
        expect(res.getCell([0, 0]), equals(19.0));
        expect(res.getCell([0, 1]), equals(22.0));
      });
    });

    test("implicit output einsum ij,jk", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList([5.0, 6.0, 7.0, 8.0], [2, 2], DType.float64);

        final res = einsum(EinsumSubscripts.parse("ij,jk"), [a, b]);
        expect(res.shape, equals([2, 2]));
        expect(res.getCell([0, 0]), equals(19.0));
      });
    });

    test("vector inner product i,i->", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final b = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);

        final res = einsum(EinsumSubscripts.parse("i,i->"), [a, b]);
        expect(res.shape, equals([]));
        expect(res.scalar, equals(32.0));
      });
    });

    test("vector outer product i,j->ij", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([3.0, 4.0, 5.0], [3], DType.float64);

        final res = einsum(EinsumSubscripts.parse("i,j->ij"), [a, b]);
        expect(res.shape, equals([2, 3]));
        expect(res.getCell([0, 0]), equals(3.0));
        expect(res.getCell([1, 2]), equals(10.0));
      });
    });

    test("matrix transpose ij->ji", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);

        final res = einsum(EinsumSubscripts.parse("ij->ji"), [a]);
        expect(res.shape, equals([2, 2]));
        expect(res.getCell([0, 1]), equals(3.0));
        expect(res.getCell([1, 0]), equals(2.0));
      });
    });

    test("trace ii->", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);

        final res = einsum(EinsumSubscripts.parse("ii->"), [a]);
        expect(res.shape, equals([]));
        expect(res.scalar, equals(5.0));
      });
    });

    test("diagonal extraction ii->i", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);

        final res = einsum(EinsumSubscripts.parse("ii->i"), [a]);
        expect(res.shape, equals([2]));
        expect(res.toList(), equals([1.0, 4.0]));
      });
    });

    test("ellipsis broadcasting ...ij,...jk->...ik", () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [1, 2, 2],
          DType.float64,
        );
        final b = NDArray.fromList(
          [5.0, 6.0, 7.0, 8.0],
          [1, 2, 2],
          DType.float64,
        );

        final res = einsum(EinsumSubscripts.parse("...ij,...jk->...ik"), [
          a,
          b,
        ]);
        expect(res.shape, equals([1, 2, 2]));
        expect(res.getCell([0, 0, 0]), equals(19.0));
      });
    });

    test("Complex128 einsum matrix contraction", () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [Complex(1, 2), Complex(3, 4), Complex(5, 6), Complex(7, 8)],
          [2, 2],
          DType.complex128,
        );
        final b = NDArray.fromList(
          [Complex(1, 0), Complex(0, 1), Complex(1, 0), Complex(0, 1)],
          [2, 2],
          DType.complex128,
        );
        final res = einsum(EinsumSubscripts.parse("ij,jk->ik"), [a, b]);
        expect(res.shape, equals([2, 2]));
        expect(res.dtype, equals(DType.complex128));
      });
    });

    test("supports out parameter in einsum", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList([5.0, 6.0, 7.0, 8.0], [2, 2], DType.float64);
        final out = NDArray.zeros([2, 2], DType.float64);

        final res = einsum(EinsumSubscripts.parse("ij,jk->ik"), [
          a,
          b,
        ], out: out);
        expect(identical(res, out), isTrue);
        expect(out.getCell([0, 0]), equals(19.0));
      });
    });

    test("throws for empty operands or subscript errors", () {
      expect(
        () => einsum(EinsumSubscripts.parse("ij"), <NDArray<Float64>>[]),
        throwsArgumentError,
      );
      final a = NDArray.fromList([1.0], [1], DType.float64);
      expect(
        () => einsum(EinsumSubscripts.parse("ij"), [a]),
        throwsArgumentError,
      );
      a.dispose();
    });

    test("matrix multiplication equivalence (tensordot axes=1)", () {
      NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([5.0, 6.0, 7.0, 8.0]), [
          2,
          2,
        ], DType.float64);
        final res = tensordot<double, double, double>(
          a,
          b,
          axes: const TensordotAxes.count(1),
        );

        final expectedMatmul = matmul(a, b);
        expect(res.shape, equals([2, 2]));
        expect(res.toList(), equals(expectedMatmul.toList()));
      });
    });

    test("throws for multiple arrow delimiters in einsum", () {
      expect(() => EinsumSubscripts.parse("i->j->k"), throwsArgumentError);
      expect(() => EinsumSubscripts.fromIndices([]), throwsArgumentError);
      expect(() => EinsumSubscripts.fromLabels([]), throwsArgumentError);
      expect(
        () => const TensordotAxes.count(-1).resolve(2, 2),
        throwsArgumentError,
      );
    });

    test(
      "EinsumSubscripts constructors: fromIndices, parse, and fromLabels",
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [5.0, 6.0, 7.0, 8.0],
            [2, 2],
            DType.float64,
          );

          // 1. EinsumSubscripts.fromIndices
          final specIndices = EinsumSubscripts.fromIndices(
            [
              [0, 1],
              [1, 2],
            ],
            [0, 2],
          );
          final resIndices = einsum(specIndices, [a, b]);
          expect(resIndices.shape, equals([2, 2]));
          expect(resIndices.getCell([0, 0]), equals(19.0));
          expect(
            specIndices.operandIndices,
            equals([
              [0, 1],
              [1, 2],
            ]),
          );
          expect(specIndices.outputIndices, equals([0, 2]));

          // 2. EinsumSubscripts.fromLabels
          final specLabels = EinsumSubscripts.fromLabels(
            [
              ['i', 'j'],
              ['j', 'k'],
            ],
            ['i', 'k'],
          );
          final resLabels = einsum(specLabels, [a, b]);
          expect(resLabels.shape, equals([2, 2]));
          expect(resLabels.getCell([0, 0]), equals(19.0));

          // 3. EinsumSubscripts.parse
          final specParse = EinsumSubscripts.parse("ij,jk->ik");
          final resParse = einsum(specParse, [a, b]);
          expect(resParse.shape, equals([2, 2]));
          expect(resParse.getCell([0, 0]), equals(19.0));

          // 4. EinsumSubscripts.fromLabels (implicit output)
          final specImplicit = EinsumSubscripts.fromLabels([
            ['i', 'j'],
            ['j', 'k'],
          ]);
          final resImplicit = einsum(specImplicit, [a, b]);
          expect(resImplicit.shape, equals([2, 2]));
          expect(resImplicit.getCell([0, 0]), equals(19.0));
          expect(specImplicit.outputIndices, isNull);
        });
      },
    );
    test(
      "batch matrix multiplication handler ...ij,...jk->...ik with Float64",
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
            [2, 2, 2],
            DType.float64,
          );
          final res = einsum(EinsumSubscripts.parse("...ij,...jk->...ik"), [
            a,
            b,
          ]);
          expect(res.shape, equals([2, 2, 2]));
          expect(res.getCell([0, 0, 0]), equals(1.0));
          expect(res.getCell([0, 0, 1]), equals(2.0));
          expect(res.getCell([1, 0, 0]), equals(10.0));
          expect(res.getCell([1, 1, 1]), equals(16.0));

          final out = NDArray.zeros([2, 2, 2], DType.float64);
          final resOut = einsum(EinsumSubscripts.parse("...ij,...jk->...ik"), [
            a,
            b,
          ], out: out);
          expect(identical(resOut, out), isTrue);
          expect(out.getCell([1, 0, 0]), equals(10.0));
        });
      },
    );
    test(
      "implicit output subscripts alphabetical sorting (NumPy compatibility)",
      () {
        NDArray.scope(() {
          // einsum('ji,kj', a, b) -> implicit output labels 'ik' (alphabetical order of single labels 'i', 'k')
          // a is 2x3 (j=2, i=3), b is 4x2 (k=4, j=2)
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );
          final b = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [4, 2],
            DType.float64,
          );

          final res = einsum(EinsumSubscripts.parse("ji,kj"), [a, b]);
          // Output labels: 'ik' -> shape [3, 4]
          expect(res.shape, equals([3, 4]));

          // Calculate expected manually for res[i, k]: sum_j a[j, i] * b[k, j]
          // For i=0, k=0: a[0, 0]*b[0, 0] + a[1, 0]*b[0, 1] = 1*1 + 4*2 = 9
          expect(res.getCell([0, 0]), equals(9.0));

          // Another case: einsum('ki,jk', a_ik, bJk) -> implicit output labels 'ij'
          final aKi = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final bJk = NDArray.fromList(
            [5.0, 6.0, 7.0, 8.0],
            [2, 2],
            DType.float64,
          );
          final resIj = einsum(EinsumSubscripts.parse("ki,jk"), [aKi, bJk]);
          expect(resIj.shape, equals([2, 2]));
        });
      },
    );

    test(
      "multi-operand pairwise contraction tree optimization (3 and 4 operands)",
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [5.0, 6.0, 7.0, 8.0],
            [2, 2],
            DType.float64,
          );
          final c = NDArray.fromList(
            [1.0, 0.0, 0.0, 1.0],
            [2, 2],
            DType.float64,
          );
          final d = NDArray.fromList(
            [2.0, 0.0, 0.0, 2.0],
            [2, 2],
            DType.float64,
          );

          // 3 operands: einsum('ij,jk,kl->il', A, B, C) == (A @ B) @ C
          final res3 = einsum(EinsumSubscripts.parse("ij,jk,kl->il"), [
            a,
            b,
            c,
          ]);
          expect(res3.shape, equals([2, 2]));
          expect(res3.getCell([0, 0]), equals(19.0));
          expect(res3.getCell([0, 1]), equals(22.0));
          expect(res3.getCell([1, 0]), equals(43.0));
          expect(res3.getCell([1, 1]), equals(50.0));

          // 4 operands: einsum('ij,jk,kl,lm->im', A, B, C, D) == ((A @ B) @ C) @ D
          final res4 = einsum(EinsumSubscripts.parse("ij,jk,kl,lm->im"), [
            a,
            b,
            c,
            d,
          ]);
          expect(res4.shape, equals([2, 2]));
          expect(res4.getCell([0, 0]), equals(38.0));
          expect(res4.getCell([0, 1]), equals(44.0));
          expect(res4.getCell([1, 0]), equals(86.0));
          expect(res4.getCell([1, 1]), equals(100.0));
        });
      },
    );

    test(
      "Einsum fast paths: vector dot (i,i->) and outer product (i,j->ij)",
      () {
        NDArray.scope(() {
          // Float64 Vector Dot (i,i->)
          final v1_64 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final v2_64 = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);
          final dot64 = einsum(EinsumSubscripts.parse("i,i->"), [v1_64, v2_64]);
          expect(dot64.shape, equals([]));
          expect(dot64.scalar, equals(32.0));

          final outDot64 = NDArray<Float64>.create([], DType.float64);
          einsum(EinsumSubscripts.parse("i,i->"), [
            v1_64,
            v2_64,
          ], out: outDot64);
          expect(outDot64.scalar, equals(32.0));

          // Float32 Vector Dot (i,i->)
          final v1_32 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float32);
          final v2_32 = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float32);
          final dot32 = einsum(EinsumSubscripts.parse("i,i->"), [v1_32, v2_32]);
          expect(dot32.shape, equals([]));
          expect(dot32.scalar, equals(32.0));

          // Vector Outer Product (i,j->ij)
          final outer = einsum(EinsumSubscripts.parse("i,j->ij"), [
            v1_64,
            v2_64,
          ]);
          expect(outer.shape, equals([3, 3]));
          expect(outer.getCell([0, 0]), equals(4.0));
          expect(outer.getCell([0, 2]), equals(6.0));
          expect(outer.getCell([2, 2]), equals(18.0));

          final outOuter = NDArray<Float64>.create([3, 3], DType.float64);
          einsum(EinsumSubscripts.parse("i,j->ij"), [
            v1_64,
            v2_64,
          ], out: outOuter);
          expect(outOuter.getCell([2, 2]), equals(18.0));
        });
      },
    );

    test(
      "Einsum fast paths: 4D batch matrix multiplication (abij,abjk->abik)",
      () {
        NDArray.scope(() {
          final a4d = NDArray<Float64>.fromList(
            List.generate(16, (i) => i.toDouble()),
            [2, 2, 2, 2],
            DType.float64,
          );
          final b4d = NDArray<Float64>.fromList(
            List.generate(16, (i) => (i + 1).toDouble()),
            [2, 2, 2, 2],
            DType.float64,
          );
          final res4d = einsum(EinsumSubscripts.parse("abij,abjk->abik"), [
            a4d,
            b4d,
          ]);
          expect(res4d.shape, equals([2, 2, 2, 2]));

          final out4d = NDArray<Float64>.create([2, 2, 2, 2], DType.float64);
          einsum(EinsumSubscripts.parse("abij,abjk->abik"), [
            a4d,
            b4d,
          ], out: out4d);
          expect(out4d.shape, equals([2, 2, 2, 2]));
          expect(
            out4d.getCell([0, 0, 0, 0]),
            equals(res4d.getCell([0, 0, 0, 0])),
          );
        });
      },
    );

    test(
      "Einsum multi-operand contraction with out buffer & error handling",
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [5.0, 6.0, 7.0, 8.0],
            [2, 2],
            DType.float64,
          );
          final c = NDArray.fromList(
            [1.0, 0.0, 0.0, 1.0],
            [2, 2],
            DType.float64,
          );

          final outBuffer = NDArray<Float64>.create([2, 2], DType.float64);
          einsum(EinsumSubscripts.parse("ij,jk,kl->il"), [
            a,
            b,
            c,
          ], out: outBuffer);
          expect(outBuffer.getCell([0, 0]), equals(19.0));

          final invalidOut = NDArray<Float64>.create([3, 3], DType.float64);
          expect(
            () => einsum(EinsumSubscripts.parse("ij,jk,kl->il"), [
              a,
              b,
              c,
            ], out: invalidOut),
            throwsArgumentError,
          );
        });
      },
    );

    test("Einsum 3-operand broadcast and reduction fallback path", () {
      NDArray.scope(() {
        final v1 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final v2 = NDArray.fromList([3.0, 4.0], [2], DType.float64);
        final v3 = NDArray.fromList([5.0, 6.0], [2], DType.float64);

        final res = einsum(EinsumSubscripts.parse("i,j,k->ijk"), [v1, v2, v3]);
        expect(res.shape, equals([2, 2, 2]));
        expect(res.getCell([0, 0, 0]), equals(15.0)); // 1 * 3 * 5
        expect(res.getCell([1, 1, 1]), equals(48.0)); // 2 * 4 * 6

        final outBuf = NDArray<Float64>.create([2, 2, 2], DType.float64);
        einsum(EinsumSubscripts.parse("i,j,k->ijk"), [v1, v2, v3], out: outBuf);
        expect(outBuf.getCell([1, 1, 1]), equals(48.0));
      });
    });

    test(
      "Float32 fast paths for einsum (2D GEMM, 3D GEMM, 4D GEMM, vector dot)",
      () {
        NDArray.scope(() {
          // 2D GEMM Float32
          final a2d32 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float32,
          );
          final b2d32 = NDArray.fromList(
            [5.0, 6.0, 7.0, 8.0],
            [2, 2],
            DType.float32,
          );
          final res2d32 = einsum(EinsumSubscripts.parse("ij,jk->ik"), [
            a2d32,
            b2d32,
          ]);
          expect(res2d32.dtype, equals(DType.float32));
          expect(res2d32.getCell([0, 0]), equals(19.0));

          final out2d32 = NDArray.zeros([2, 2], DType.float32);
          final res2d32Out = einsum(EinsumSubscripts.parse("ij,jk->ik"), [
            a2d32,
            b2d32,
          ], out: out2d32);
          expect(identical(res2d32Out, out2d32), isTrue);
          expect(out2d32.getCell([0, 0]), equals(19.0));

          // 3D GEMM Float32
          final a3d32 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 2, 2],
            DType.float32,
          );
          final b3d32 = NDArray.fromList(
            [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
            [2, 2, 2],
            DType.float32,
          );
          final res3d32 = einsum(EinsumSubscripts.parse("bij,bjk->bik"), [
            a3d32,
            b3d32,
          ]);
          expect(res3d32.dtype, equals(DType.float32));
          expect(res3d32.getCell([0, 0, 0]), equals(1.0));

          final out3d32 = NDArray.zeros([2, 2, 2], DType.float32);
          final res3d32Out = einsum(EinsumSubscripts.parse("bij,bjk->bik"), [
            a3d32,
            b3d32,
          ], out: out3d32);
          expect(identical(res3d32Out, out3d32), isTrue);
          expect(out3d32.getCell([0, 0, 0]), equals(1.0));

          // 4D GEMM Float32
          final a4d32 = NDArray.fromList(
            List.generate(16, (i) => i.toDouble()),
            [2, 2, 2, 2],
            DType.float32,
          );
          final b4d32 = NDArray.fromList(
            List.generate(16, (i) => (i + 1).toDouble()),
            [2, 2, 2, 2],
            DType.float32,
          );
          final res4d32 = einsum(EinsumSubscripts.parse("abij,abjk->abik"), [
            a4d32,
            b4d32,
          ]);
          expect(res4d32.dtype, equals(DType.float32));

          final out4d32 = NDArray.zeros([2, 2, 2, 2], DType.float32);
          einsum(EinsumSubscripts.parse("abij,abjk->abik"), [
            a4d32,
            b4d32,
          ], out: out4d32);
          expect(out4d32.dtype, equals(DType.float32));

          // Vector Dot Float32 with out
          final v1_32 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float32);
          final v2_32 = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float32);
          final outDot32 = NDArray.zeros([], DType.float32);
          final resDot32Out = einsum(EinsumSubscripts.parse("i,i->"), [
            v1_32,
            v2_32,
          ], out: outDot32);
          expect(identical(resDot32Out, outDot32), isTrue);
          expect(outDot32.scalar, equals(32.0));
        });
      },
    );

    test(
      "Float64 fast paths with out buffer (2D GEMM, 3D GEMM, vector dot)",
      () {
        NDArray.scope(() {
          final a2d = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b2d = NDArray.fromList(
            [5.0, 6.0, 7.0, 8.0],
            [2, 2],
            DType.float64,
          );
          final out2d = NDArray.zeros([2, 2], DType.float64);
          final res2dOut = einsum(EinsumSubscripts.parse("ij,jk->ik"), [
            a2d,
            b2d,
          ], out: out2d);
          expect(identical(res2dOut, out2d), isTrue);

          final a3d = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 2, 2],
            DType.float64,
          );
          final b3d = NDArray.fromList(
            [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
            [2, 2, 2],
            DType.float64,
          );
          final out3d = NDArray.zeros([2, 2, 2], DType.float64);
          final res3dOut = einsum(EinsumSubscripts.parse("bij,bjk->bik"), [
            a3d,
            b3d,
          ], out: out3d);
          expect(identical(res3dOut, out3d), isTrue);
        });
      },
    );

    test("Single-operand einsum operations and out buffer handling", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final out = NDArray.zeros([2, 2], DType.float64);

        final resOut = einsum(EinsumSubscripts.parse("ij->ji"), [a], out: out);
        expect(identical(resOut, out), isTrue);
        expect(out.getCell([0, 1]), equals(3.0));

        final wrongOut = NDArray.zeros([3, 3], DType.float64);
        expect(
          () => einsum(EinsumSubscripts.parse("ij->ji"), [a], out: wrongOut),
          throwsArgumentError,
        );

        // Multi-axis sum reduction
        final a3d = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 2, 2],
          DType.float64,
        );
        final resMultiSum = einsum(EinsumSubscripts.parse("ijk->i"), [a3d]);
        expect(resMultiSum.shape, equals([2]));
        expect(resMultiSum.toList(), equals([10.0, 26.0]));
      });
    });

    test("einsum error handling for disposed inputs, subscript mismatches, dimension mismatch", () {
      final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
      final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);
      final disposed = NDArray.fromList([1.0], [1], DType.float64);
      disposed.dispose();

      expect(
        () => einsum(EinsumSubscripts.parse("i,i->"), [disposed, b]),
        throwsStateError,
      );
      expect(
        () => einsum(EinsumSubscripts.parse("i,i->"), [a, b], out: disposed),
        throwsStateError,
      );

      expect(
        () => einsum(EinsumSubscripts.parse("i,j,k->"), [a, b]),
        throwsArgumentError,
      );

      final cWrongShape = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      expect(
        () => einsum(EinsumSubscripts.parse("i,i->"), [a, cWrongShape]),
        throwsArgumentError,
      );

      expect(
        () => einsum(EinsumSubscripts.parse("...i,j->ij"), [
          NDArray.scalar(1.0, dtype: DType.float64),
          b,
        ]),
        throwsArgumentError,
      );

      a.dispose();
      b.dispose();
      cWrongShape.dispose();
    });

    test("EinsumSubscripts constructors error cases and formatting", () {
      expect(() => EinsumSubscripts.fromLabels([]), throwsArgumentError);
      expect(() => EinsumSubscripts.fromIndices([]), throwsArgumentError);

      final specWithSpaces = EinsumSubscripts.parse(" i j , j k -> i k ");
      expect(specWithSpaces.operandIndices.length, equals(2));
    });

    test("tensordot error handling and additional edge cases", () {
      final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
      final b = NDArray.fromList([5.0, 6.0, 7.0, 8.0], [2, 2], DType.float64);
      final disposed = NDArray.fromList([1.0], [1], DType.float64);
      disposed.dispose();

      expect(() => tensordot(disposed, b), throwsStateError);
      expect(() => tensordot(a, disposed), throwsStateError);
      expect(() => tensordot(a, b, out: disposed), throwsStateError);

      expect(() => tensordot(a, b, axes: [1, 0] as Object), returnsNormally);

      expect(() => TensordotAxes.count(-1).resolve(2, 2), throwsArgumentError);
      expect(() => TensordotAxes.count(3).resolve(2, 2), throwsArgumentError);
      expect(
        () => TensordotAxes.explicit([0, 1], [0]).resolve(2, 2),
        throwsArgumentError,
      );

      final wrongOut = NDArray.zeros([3, 3], DType.float64);
      expect(() => tensordot(a, b, out: wrongOut), throwsArgumentError);

      expect(
        () => tensordot(a, b, axes: TensordotAxes.explicit([5], [0])),
        throwsRangeError,
      );
      expect(
        () => tensordot(a, b, axes: TensordotAxes.explicit([0], [5])),
        throwsRangeError,
      );

      final cMismatch = NDArray.fromList(
        [1.0, 2.0, 3.0],
        [3, 1],
        DType.float64,
      );
      expect(
        () => tensordot(a, cMismatch, axes: TensordotAxes.explicit([1], [0])),
        throwsArgumentError,
      );

      a.dispose();
      b.dispose();
      cMismatch.dispose();
    });

    test("tensordot type casting fast path and out buffer", () {
      NDArray.scope(() {
        final a32 = NDArray.fromList([1.0, 2.0], [2], DType.float32);
        final b64 = NDArray.fromList([3.0, 4.0], [2], DType.float64);

        final res = tensordot(a32, b64, axes: 1);
        expect(res.dtype, equals(DType.float64));
        expect(res.scalar, equals(11.0));

        final out = NDArray.zeros([], DType.float64);
        tensordot(a32, b64, axes: 1, out: out);
        expect(out.scalar, equals(11.0));
      });
    });

    test("einsum batch tensordot and fallback pathways", () {
      NDArray.scope(() {
        // Batch matrix multiplication with permutation required (bij,bjk -> ikb)
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 2, 2],
          DType.float64,
        );
        final b = NDArray.fromList(
          [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
          [2, 2, 2],
          DType.float64,
        );

        final resPerm = einsum(EinsumSubscripts.parse("bij,bjk->ikb"), [a, b]);
        expect(resPerm.shape, equals([2, 2, 2]));

        final outPerm = NDArray.zeros([2, 2, 2], DType.float64);
        einsum(EinsumSubscripts.parse("bij,bjk->ikb"), [a, b], out: outPerm);
        expect(outPerm.shape, equals([2, 2, 2]));

        // Non-fastpath 2-operand einsum fallback
        final resFallback = einsum(EinsumSubscripts.parse("ij,jk->k"), [
          a[0] as NDArray<Object>,
          b[0] as NDArray<Object>,
        ]);
        expect(resFallback.shape, equals([2]));

        final outFallback = NDArray.zeros([2], DType.float64);
        einsum(EinsumSubscripts.parse("ij,jk->k"), [
          a[0] as NDArray<Object>,
          b[0] as NDArray<Object>,
        ], out: outFallback);
        expect(outFallback.shape, equals([2]));

        // Float32 batch tensordot pathway
        final a32Batch = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 2, 2],
          DType.float32,
        );
        final b32Batch = NDArray.fromList(
          [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
          [2, 2, 2],
          DType.float32,
        );
        final res32Batch = einsum(EinsumSubscripts.parse("bij,bjk->ikb"), [
          a32Batch,
          b32Batch,
        ]);
        expect(res32Batch.shape, equals([2, 2, 2]));

        // General fallback with diagonal view and output permutation
        final resGeneralPerm = einsum(EinsumSubscripts.parse("i,j,k->kji"), [
          NDArray.fromList([1.0, 2.0], [2], DType.float64),
          NDArray.fromList([3.0, 4.0], [2], DType.float64),
          NDArray.fromList([5.0, 6.0], [2], DType.float64),
        ]);
        expect(resGeneralPerm.shape, equals([2, 2, 2]));

        final outGeneralPerm = NDArray.zeros([2, 2, 2], DType.float64);
        einsum(EinsumSubscripts.parse("i,j,k->kji"), [
          NDArray.fromList([1.0, 2.0], [2], DType.float64),
          NDArray.fromList([3.0, 4.0], [2], DType.float64),
          NDArray.fromList([5.0, 6.0], [2], DType.float64),
        ], out: outGeneralPerm);
        expect(outGeneralPerm.shape, equals([2, 2, 2]));

        // N-operand fallback with diagonal view on an operand
        final aDiag = NDArray.fromList(
          [1.0, 0.0, 0.0, 2.0],
          [2, 2],
          DType.float64,
        );
        final bVec = NDArray.fromList([3.0, 4.0], [2], DType.float64);
        final cVec = NDArray.fromList([5.0, 6.0], [2], DType.float64);
        final resDiagFallback = einsum(EinsumSubscripts.parse("ii,j,k->ijk"), [
          aDiag,
          bVec,
          cVec,
        ]);
        expect(resDiagFallback.shape, equals([2, 2, 2]));
      });
    });

    test("Einsum implicit output with ellipsis and non-contiguous batch GEMM fallback", () {
      NDArray.scope(() {
        final a3d = NDArray<Float64>.fromList(
          List.generate(8, (i) => (i + 1).toDouble()),
          [2, 2, 2],
          DType.float64,
        );
        final b3d = NDArray<Float64>.fromList(
          List.generate(8, (i) => (i + 1).toDouble()),
          [2, 2, 2],
          DType.float64,
        );

        // Implicit output subscript with ellipsis: '...ij,...jk' => '...ik'
        final resImplicitEllipsis = einsum(
          EinsumSubscripts.parse("...ij,...jk"),
          [a3d, b3d],
        );
        expect(resImplicitEllipsis.shape, equals([2, 2, 2]));

        // Non-contiguous 3D batch matmul fallback (transposed slices)
        final aTrans = a3d.transpose([0, 2, 1]);
        final bTrans = b3d.transpose([0, 2, 1]);
        final resNonContigBatch = einsum(
          EinsumSubscripts.parse("...ij,...jk->...ik"),
          [aTrans, bTrans],
        );
        expect(resNonContigBatch.shape, equals([2, 2, 2]));
      });
    });

    test(
      "Einsum Float32 2D GEMM incompatible out buffer shape and dtype errors",
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float32,
          );
          final b = NDArray.fromList(
            [5.0, 6.0, 7.0, 8.0],
            [2, 2],
            DType.float32,
          );

          final invalidShapeOut = NDArray<Float32>.create([
            3,
            3,
          ], DType.float32);
          expect(
            () => einsum(EinsumSubscripts.parse("ij,jk->ik"), [
              a,
              b,
            ], out: invalidShapeOut),
            throwsArgumentError,
          );

          final invalidDTypeOut = NDArray<Float64>.create([
            2,
            2,
          ], DType.float64);
          expect(
            () => einsum<Float32, Float64>(
              EinsumSubscripts.parse("ij,jk->ik"),
              [a, b],
              out: invalidDTypeOut,
            ),
            throwsArgumentError,
          );
        });
      },
    );

    test("Kron incompatible out buffer error handling", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);

        final invalidOut = NDArray<Float64>.create([3, 3], DType.float64);
        expect(() => kron(a, b, out: invalidOut), throwsArgumentError);
      });
    });

    test(
      "Einsum Int64 3D batch matmul fallback & batch permuted out buffer",
      () {
        NDArray.scope(() {
          // Int64 batch matmul fallback
          final aInt = NDArray<Int64>.fromList(List.generate(8, (i) => i + 1), [
            2,
            2,
            2,
          ], DType.int64);
          final bInt = NDArray<Int64>.fromList(List.generate(8, (i) => i + 1), [
            2,
            2,
            2,
          ], DType.int64);

          final resIntBatch = einsum<Int64, Int64>(
            EinsumSubscripts.parse("bij,bjk->bik"),
            [aInt, bInt],
          );
          expect(resIntBatch.shape, equals([2, 2, 2]));

          final outIntBatch = NDArray<Int64>.create([2, 2, 2], DType.int64);
          einsum<Int64, Int64>(EinsumSubscripts.parse("bij,bjk->bik"), [
            aInt,
            bInt,
          ], out: outIntBatch);
          expect(outIntBatch.shape, equals([2, 2, 2]));

          // Permuted batch output with out buffer: 'bij,bjk->bki'
          final a3d = NDArray<Float64>.fromList(
            List.generate(8, (i) => (i + 1).toDouble()),
            [2, 2, 2],
            DType.float64,
          );
          final b3d = NDArray<Float64>.fromList(
            List.generate(8, (i) => (i + 1).toDouble()),
            [2, 2, 2],
            DType.float64,
          );

          final resPerm = einsum(EinsumSubscripts.parse("bij,bjk->bki"), [
            a3d,
            b3d,
          ]);
          expect(resPerm.shape, equals([2, 2, 2]));

          final outPerm = NDArray<Float64>.create([2, 2, 2], DType.float64);
          einsum(EinsumSubscripts.parse("bij,bjk->bki"), [
            a3d,
            b3d,
          ], out: outPerm);
          expect(outPerm.shape, equals([2, 2, 2]));

          // Incompatible out buffer on fallback path
          final invalidOutFallback = NDArray<Float64>.create([
            3,
            3,
            3,
          ], DType.float64);
          expect(
            () => einsum(EinsumSubscripts.parse("i,j,k->ijk"), [
              NDArray.fromList([1.0, 2.0], [2], DType.float64),
              NDArray.fromList([3.0, 4.0], [2], DType.float64),
              NDArray.fromList([5.0, 6.0], [2], DType.float64),
            ], out: invalidOutFallback),
            throwsArgumentError,
          );

          // Mixed ellipsis and non-ellipsis operands: '...ij,jk'
          final a3dEllipsis = NDArray<Float64>.fromList(
            List.generate(8, (i) => (i + 1).toDouble()),
            [2, 2, 2],
            DType.float64,
          );
          final b2dNoEllipsis = NDArray<Float64>.fromList(
            [1.0, 0.0, 0.0, 1.0],
            [2, 2],
            DType.float64,
          );
          final resMixedEllipsis = einsum(EinsumSubscripts.parse("...ij,jk"), [
            a3dEllipsis,
            b2dNoEllipsis,
          ]);
          expect(resMixedEllipsis.shape, equals([2, 2, 2]));

          // Batch tensordot output permutation with out buffer: 'bij,bjk->ikb'
          final resBatchPerm = einsum(EinsumSubscripts.parse("bij,bjk->ikb"), [
            a3d,
            b3d,
          ]);
          expect(resBatchPerm.shape, equals([2, 2, 2]));

          final outBatchPerm = NDArray<Float64>.create([
            2,
            2,
            2,
          ], DType.float64);
          einsum(EinsumSubscripts.parse("bij,bjk->ikb"), [
            a3d,
            b3d,
          ], out: outBatchPerm);
          expect(outBatchPerm.shape, equals([2, 2, 2]));

          final invalidOutBatchPerm = NDArray<Float64>.create([
            3,
            3,
            3,
          ], DType.float64);
          expect(
            () => einsum(EinsumSubscripts.parse("bij,bjk->ikb"), [
              a3d,
              b3d,
            ], out: invalidOutBatchPerm),
            throwsArgumentError,
          );

          final invalidDTypeBatchPerm = NDArray<Float32>.create([
            2,
            2,
            2,
          ], DType.float32);
          expect(
            () => einsum<Float64, Float32>(
              EinsumSubscripts.parse("bij,bjk->ikb"),
              [a3d, b3d],
              out: invalidDTypeBatchPerm,
            ),
            throwsArgumentError,
          );
        });
      },
    );

    test("Tensordot mismatched axes count error handling", () {
      NDArray.scope(() {
        final a = NDArray.zeros([2, 2], DType.float64);
        final b = NDArray.zeros([2, 2], DType.float64);
        expect(
          () => tensordot(a, b, axes: TensordotAxes.explicit([0, 1], [0])),
          throwsArgumentError,
        );
      });
    });

    test("Kron incompatible out buffer dtype and shape error handling", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);

        final invalidDTypeOut = NDArray<Float32>.create([4], DType.float32);
        expect(
          () => kron<Float64, Float64, Float32>(a, b, out: invalidDTypeOut),
          throwsArgumentError,
        );

        final invalidShapeOut = NDArray<Float64>.create([5], DType.float64);
        expect(() => kron(a, b, out: invalidShapeOut), throwsArgumentError);
      });
    });

    test("3D diagonal view extraction (iij->ij)", () {
      NDArray.scope(() {
        final a3d = NDArray<Float64>.fromList(
          List.generate(8, (i) => (i + 1).toDouble()),
          [2, 2, 2],
          DType.float64,
        );
        final diagRes = einsum(EinsumSubscripts.parse("iij->ij"), [a3d]);
        expect(diagRes.shape, equals([2, 2]));
      });
    });

    test("Fallback einsum broadcaster output permutation for Int64 & Int32 batch tensordot", () {
      NDArray.scope(() {
        final v1 = NDArray.fromList([1, 2], [2], DType.int64);
        final v2 = NDArray.fromList([3, 4], [2], DType.int64);
        final v3 = NDArray.fromList([5, 6], [2], DType.int64);

        final resPerm = einsum(EinsumSubscripts.parse("i,j,k->kji"), [
          v1,
          v2,
          v3,
        ]);
        expect(resPerm.shape, equals([2, 2, 2]));

        // Int64 batch tensordot permutation 'bij,bjk->ikb'
        final aInt = NDArray<Int64>.fromList(List.generate(8, (i) => i + 1), [
          2,
          2,
          2,
        ], DType.int64);
        final bInt = NDArray<Int64>.fromList(List.generate(8, (i) => i + 1), [
          2,
          2,
          2,
        ], DType.int64);

        final resBatchPerm = einsum<Int64, Int64>(
          EinsumSubscripts.parse("bij,bjk->ikb"),
          [aInt, bInt],
        );
        expect(resBatchPerm.shape, equals([2, 2, 2]));

        final outBatchPerm = NDArray<Int64>.create([2, 2, 2], DType.int64);
        einsum<Int64, Int64>(EinsumSubscripts.parse("bij,bjk->ikb"), [
          aInt,
          bInt,
        ], out: outBatchPerm);
        expect(outBatchPerm.shape, equals([2, 2, 2]));

        final invalidOutBatchPerm = NDArray<Int64>.create([
          3,
          3,
          3,
        ], DType.int64);
        expect(
          () => einsum<Int64, Int64>(EinsumSubscripts.parse("bij,bjk->ikb"), [
            aInt,
            bInt,
          ], out: invalidOutBatchPerm),
          throwsArgumentError,
        );
      });
    });

    test("Targeted 100% line coverage edge case dispatches", () {
      NDArray.scope(() {
        // Untyped NDArray<Object> to Float64 _asTyped cast (lines 13-17)
        final untypedA = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        ) as NDArray<Object>;
        final untypedB = NDArray.fromList(
          [5.0, 6.0, 7.0, 8.0],
          [2, 2],
          DType.float64,
        ) as NDArray<Object>;
        final castRes = einsum<Object, Float64>(
          EinsumSubscripts.parse("ij,jk->ik"),
          [untypedA, untypedB],
        );
        expect(castRes.shape, equals([2, 2]));

        // Non-contiguous strided Int64 batch tensordot permutation 'bij,bjk->ikb' (lines 1103-1116)
        final aInt = NDArray<Int64>.fromList(List.generate(8, (i) => i + 1), [
          2,
          2,
          2,
        ], DType.int64);
        final bInt = NDArray<Int64>.fromList(List.generate(8, (i) => i + 1), [
          2,
          2,
          2,
        ], DType.int64);
        final aStridedInt = aInt.transpose([0, 2, 1]);
        final bStridedInt = bInt.transpose([0, 2, 1]);
        final resStridedBatchPerm = einsum<Int64, Int64>(
          EinsumSubscripts.parse("bij,bjk->ikb"),
          [aStridedInt, bStridedInt],
        );
        expect(resStridedBatchPerm.shape, equals([2, 2, 2]));

        final outStridedBatchPerm = NDArray<Int64>.create([
          2,
          2,
          2,
        ], DType.int64);
        einsum<Int64, Int64>(EinsumSubscripts.parse("bij,bjk->ikb"), [
          aStridedInt,
          bStridedInt,
        ], out: outStridedBatchPerm);
        expect(outStridedBatchPerm.shape, equals([2, 2, 2]));

        final invalidOutStridedBatchPerm = NDArray<Int64>.create([
          3,
          3,
          3,
        ], DType.int64);
        expect(
          () => einsum<Int64, Int64>(EinsumSubscripts.parse("bij,bjk->ikb"), [
            aStridedInt,
            bStridedInt,
          ], out: invalidOutStridedBatchPerm),
          throwsArgumentError,
        );

        // Non-contiguous strided Int64 multi-operand broadcaster permutation (lines 1457-1458)
        final v1Strided = NDArray.fromList(
          [1, 2, 3, 4],
          [4],
          DType.int64,
        ).slice([Slice(start: 0, stop: 4, step: 2)]);
        final v2Strided = NDArray.fromList(
          [5, 6, 7, 8],
          [4],
          DType.int64,
        ).slice([Slice(start: 0, stop: 4, step: 2)]);
        final v3Strided = NDArray.fromList(
          [9, 10, 11, 12],
          [4],
          DType.int64,
        ).slice([Slice(start: 0, stop: 4, step: 2)]);
        final resStridedBroadcaster = einsum<Int64, Int64>(
          EinsumSubscripts.parse("i,j,k->kji"),
          [v1Strided, v2Strided, v3Strided],
        );
        expect(resStridedBroadcaster.shape, equals([2, 2, 2]));

        // 4D Batch contraction permuted output & out buffer validation (lines 1103-1116)
        final a4d = NDArray<Float64>.fromList(
          List.generate(16, (i) => (i + 1).toDouble()),
          [2, 2, 2, 2],
          DType.float64,
        );
        final b4d = NDArray<Float64>.fromList(
          List.generate(16, (i) => (i + 1).toDouble()),
          [2, 2, 2, 2],
          DType.float64,
        );

        final res4dPerm = einsum(EinsumSubscripts.parse("bijk,bklm->bmlji"), [
          a4d,
          b4d,
        ]);
        expect(res4dPerm.shape, equals([2, 2, 2, 2, 2]));

        final out4dPerm = NDArray<Float64>.create([
          2,
          2,
          2,
          2,
          2,
        ], DType.float64);
        einsum(EinsumSubscripts.parse("bijk,bklm->bmlji"), [
          a4d,
          b4d,
        ], out: out4dPerm);
        expect(out4dPerm.shape, equals([2, 2, 2, 2, 2]));

        final invalidOut4dPerm = NDArray<Float64>.create([
          3,
          3,
          3,
          3,
          3,
        ], DType.float64);
        expect(
          () => einsum(EinsumSubscripts.parse("bijk,bklm->bmlji"), [
            a4d,
            b4d,
          ], out: invalidOut4dPerm),
          throwsArgumentError,
        );
      });
      // Additional coverage test cases for edge cases
      // 1. Multi-operand fallback einsum with output permutation (lines 1445-1446)
      final a1 = NDArray<Int64>.fromList([1, 2], [2], DType.int64);
      final b1 = NDArray<Int64>.fromList([3, 4], [2], DType.int64);
      final c1 = NDArray<Int64>.fromList([5, 6], [2], DType.int64);
      final resFallbackPerm = einsum(EinsumSubscripts.parse('i,j,k->kji'), [
        a1,
        b1,
        c1,
      ]);
      expect(resFallbackPerm.shape, equals([2, 2, 2]));

      // 2. tensordot all axes with type casting non-float (lines 305-306)
      final aInt32 = NDArray<Int32>.fromList([1, 2, 3, 4], [2, 2], DType.int32);
      final bInt64 = NDArray<Int64>.fromList([1, 1, 1, 1], [2, 2], DType.int64);
      final resTdCast = tensordot(aInt32, bInt64, axes: 2);
      expect(resTdCast.scalar, equals(10));

      // 3. tensordot non-sequential axes isSeq return false (line 254)
      final aF64 = NDArray<Float64>.fromList(
        [1.0, 2.0, 3.0, 4.0],
        [2, 2],
        DType.float64,
      );
      final bF64 = NDArray<Float64>.fromList(
        [1.0, 1.0, 1.0, 1.0],
        [2, 2],
        DType.float64,
      );
      final resTdNonSeq = tensordot(aF64, bF64, axes: ([1, 0], [1, 0]));
      expect(resTdNonSeq.scalar, equals(10.0));

      // 4. empty inStr in EinsumSubscripts (line 418)
      expect(() => EinsumSubscripts.parse('->'), throwsArgumentError);

      // 5. einsum operand ellipsis without output ellipsis (line 581)
      final aEll = NDArray<Float64>.fromList(
        [1.0, 2.0, 3.0, 4.0],
        [1, 2, 2],
        DType.float64,
      );
      final bEll = NDArray<Float64>.fromList(
        [1.0, 0.0, 0.0, 1.0],
        [2, 2],
        DType.float64,
      );
      final resEll = einsum(EinsumSubscripts.parse('...ij,jk->ik'), [
        aEll,
        bEll,
      ]);
      expect(resEll.shape, equals([2, 2]));

      // 6. matmul einsum fast path with out parameter
      final outMatmul = NDArray<Float64>.zeros([2, 2], DType.float64);
      einsum(EinsumSubscripts.parse('ij,jk->ik'), [aF64, bF64], out: outMatmul);
      expect(outMatmul.shape, equals([2, 2]));

      // 7. tensordot error handling tests
      expect(() => tensordot(aF64, bF64, axes: 10), throwsArgumentError);
      final a3x5 = NDArray<Float64>.zeros([3, 5], DType.float64);
      expect(() => tensordot(a3x5, bF64, axes: 1), throwsArgumentError);
      final invalidOut = NDArray<Float64>.zeros([5, 5], DType.float64);
      expect(
        () => tensordot(aF64, bF64, axes: 0, out: invalidOut),
        throwsArgumentError,
      );

      // 8. diagonal view dimension mismatch error
      final aMismatchDiag = NDArray<Float64>.zeros([2, 3, 4], DType.float64);
      expect(
        () => einsum(EinsumSubscripts.parse('iij->ij'), [aMismatchDiag]),
        throwsArgumentError,
      );
    });
  });
}
