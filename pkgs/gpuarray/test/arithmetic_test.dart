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
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Arithmetic & Ufuncs', () {
    test('Elementwise binary operators (+, -, *, /)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final b = GpuArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );

        final sum = a + b;
        expect(
          sum.toNestedList(),
          equals([
            [11.0, 22.0],
            [33.0, 44.0],
          ]),
        );

        final diff = b - a;
        expect(
          diff.toNestedList(),
          equals([
            [9.0, 18.0],
            [27.0, 36.0],
          ]),
        );

        final prod = a * b;
        expect(
          prod.toNestedList(),
          equals([
            [10.0, 40.0],
            [90.0, 160.0],
          ]),
        );

        final quot = b / a;
        expect(
          quot.toNestedList(),
          equals([
            [10.0, 10.0],
            [10.0, 10.0],
          ]),
        );
      });
    });

    test('Scalar operations and broadcasting', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final addedScalar = a + 10.0;
        expect(addedScalar.toNestedList(), equals([11.0, 12.0, 13.0]));

        final scaled = a * 5.0;
        expect(scaled.toNestedList(), equals([5.0, 10.0, 15.0]));

        // Multidimensional broadcasting: [2, 3] + [1, 3]
        final m1 = GpuArray.fromList(
          [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
          ],
          [2, 3],
          DType.float64,
        );
        final m2 = GpuArray.fromList(
          [
            [10.0, 20.0, 30.0],
          ],
          [1, 3],
          DType.float64,
        );

        final broadcasted = m1 + m2;
        expect(broadcasted.shape, equals([2, 3]));
        expect(
          broadcasted.toNestedList(),
          equals([
            [11.0, 22.0, 33.0],
            [14.0, 25.0, 36.0],
          ]),
        );
      });
    });

    test('Unary operations (sin, cos, exp, log, sqrt, abs, negate)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList([-1.0, 0.0, 1.0, 4.0], [4], DType.float64);

        final neg = -a;
        expect(neg.toList(), equals([1.0, -0.0, -1.0, -4.0]));

        final absolute = a.abs();
        expect(absolute.toList(), equals([1.0, 0.0, 1.0, 4.0]));

        final pos = GpuArray.fromList([0.0, 1.0, 4.0, 9.0], [4], DType.float64);
        final sq = pos.sqrt();
        expect(sq.toList(), equals([0.0, 1.0, 2.0, 3.0]));

        final zeros = GpuArray.zeros([2], DType.float64);
        final cosZeros = zeros.cos();
        expect(cosZeros.toList(), equals([1.0, 1.0]));

        final sinZeros = zeros.sin();
        expect(sinZeros.toList(), equals([0.0, 0.0]));
      });
    });

    test('Comparison operations', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList([1.0, 5.0, 10.0], [3], DType.float64);
        final b = GpuArray.fromList([2.0, 5.0, 8.0], [3], DType.float64);

        final gt = a.greater(b);
        expect(gt.dtype, equals(DType.boolean));
        expect(gt.toList(), equals([false, false, true]));

        final eq = a.equal(b);
        expect(eq.toList(), equals([false, true, false]));

        final le = a.lessEqual(b);
        expect(le.toList(), equals([true, true, false]));
      });
    });

    test(
      'In-place out: parameter for binary, unary, and comparison operations',
      () {
        ResourceScope.scope(() {
          final a = GpuArray.fromList([1.0, 4.0, 9.0], [3], DType.float64);
          final b = GpuArray.fromList([2.0, 3.0, 4.0], [3], DType.float64);
          final out = GpuArray.zeros([3], DType.float64);

          final resAdd = a.add(b, out: out);
          expect(identical(resAdd, out), isTrue);
          expect(out.toList(), equals([3.0, 7.0, 13.0]));

          final resSqrt = a.sqrt(out: out);
          expect(identical(resSqrt, out), isTrue);
          expect(out.toList(), equals([1.0, 2.0, 3.0]));

          final boolOut = GpuArray.zeros([3], DType.boolean);
          final resGt = a.greater(b, out: boolOut);
          expect(identical(resGt, boolOut), isTrue);
          expect(boolOut.toList(), equals([false, true, true]));

          // Mismatched shape or dtype on out throws ArgumentError subclass
          final wrongShape = GpuArray.zeros([2], DType.float64);
          expect(
            () => a.add(b, out: wrongShape),
            throwsA(isA<GpuShapeMismatchException>()),
          );
          expect(
            () => a.add(b, out: wrongShape),
            throwsA(isA<ArgumentError>()),
          );

          final wrongDtype = GpuArray.zeros([3], DType.float32);
          expect(
            () => (a as GpuArray<DTypeTag>).add(b, out: wrongDtype),
            throwsA(isA<ArgumentError>()),
          );
        });
      },
    );

    test('Operations on negative-stride and offset sliced views', () {
      ResourceScope.scope(() {
        final base = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float32,
        );
        final rev = base.slice([Slice(null, null, -1)]);
        final sub = base.slice([Slice(1, 4)]);

        expect(rev.toList(), equals([5.0, 4.0, 3.0, 2.0, 1.0]));
        expect((rev + base).toList(), equals([6.0, 6.0, 6.0, 6.0, 6.0]));
        expect(sub.abs().toList(), equals([2.0, 3.0, 4.0]));
      });
    });

    test(
      'F1: GpuArray<T> static typing and scalar operand dtype preservation',
      () {
        ResourceScope.scope(() {
          final f32 = GpuArray.fromList([4.0, 9.0, 16.0], [3], DType.float32);
          final f32b = GpuArray.fromList([2.0, 3.0, 4.0], [3], DType.float32);

          // Static and dynamic type preservation for binary operators & scalars
          final GpuArray<Float32> addScalar = f32 + 1.0;
          final GpuArray<Float32> subScalar = f32 - 1.0;
          final GpuArray<Float32> mulScalar = f32 * 2.0;
          final GpuArray<Float32> divScalar = f32 / 2.0;
          final GpuArray<Float32> fdivScalar = f32 ~/ 3.0;
          final GpuArray<Float32> modScalar = f32 % 5.0;
          expect(addScalar.dtype, equals(DType.float32));
          expect(subScalar.dtype, equals(DType.float32));
          expect(mulScalar.dtype, equals(DType.float32));
          expect(divScalar.dtype, equals(DType.float32));
          expect(fdivScalar.dtype, equals(DType.float32));
          expect(modScalar.dtype, equals(DType.float32));
          expect(addScalar.toList(), equals([5.0, 10.0, 17.0]));
          expect(fdivScalar.toList(), equals([1.0, 3.0, 5.0]));
          expect(modScalar.toList(), equals([4.0, 4.0, 1.0]));

          // Binary methods and top-level functions preserve GpuArray<Float32>
          final GpuArray<Float32> mPow = pow(f32b, 2.0);
          final GpuArray<Float32> mMax = maximum(f32, 10.0);
          final GpuArray<Float32> mMin = minimum(f32, 10.0);
          final GpuArray<Float32> mHypot = hypot(
            GpuArray.fromList([3.0, 5.0], [2], DType.float32),
            GpuArray.fromList([4.0, 12.0], [2], DType.float32),
          );
          final GpuArray<Float32> mAtan2 = atan2(
            GpuArray.fromList([1.0, 0.0], [2], DType.float32),
            GpuArray.fromList([1.0, 1.0], [2], DType.float32),
          );
          final GpuArray<Float32> mCopysign = copysign(f32, -1.0);
          expect(mPow.dtype, equals(DType.float32));
          expect(mPow.toList(), equals([4.0, 9.0, 16.0]));
          expect(mMax.toList(), equals([10.0, 10.0, 16.0]));
          expect(mMin.toList(), equals([4.0, 9.0, 10.0]));
          expect(mHypot.toList(), equals([5.0, 13.0]));
          expect(
            (mAtan2.toList()[0] as num).toDouble(),
            closeTo(0.785398, 1e-4),
          );
          expect(mCopysign.toList(), equals([-4.0, -9.0, -16.0]));
        });
      },
    );

    test(
      'F5: Bitwise, predicate, nanToNum, isClose/allClose, clip, and complex ufuncs (methods & top-level)',
      () {
        ResourceScope.scope(() {
          // Bitwise ops on Int32
          final a = GpuArray.fromList([6, 12, 15], [3], DType.int32);
          final b = GpuArray.fromList([3, 5, 7], [3], DType.int32);
          final GpuArray<Int32> bAnd = a & b;
          final GpuArray<Int32> bOr = a | b;
          final GpuArray<Int32> bXor = a ^ b;
          final GpuArray<Int32> bNot = ~a;
          final GpuArray<Int32> bShl = a << 1;
          final GpuArray<Int32> bShr = a >> 1;
          expect(bAnd.toList(), equals([2, 4, 7]));
          expect(bitwiseAnd(a, b).toList(), equals([2, 4, 7]));
          expect(bOr.toList(), equals([7, 13, 15]));
          expect(bitwiseOr(a, b).toList(), equals([7, 13, 15]));
          expect(bXor.toList(), equals([5, 9, 8]));
          expect(bitwiseXor(a, b).toList(), equals([5, 9, 8]));
          expect(bNot.toList(), equals([-7, -13, -16]));
          expect(invert(a).toList(), equals([-7, -13, -16]));
          expect(bShl.toList(), equals([12, 24, 30]));
          expect(leftShift(a, 1).toList(), equals([12, 24, 30]));
          expect(bShr.toList(), equals([3, 6, 7]));
          expect(rightShift(a, 1).toList(), equals([3, 6, 7]));
          expect(gcd(a, b).toList(), equals([3, 1, 1]));
          expect(lcm(a, b).toList(), equals([6, 60, 105]));

          // Sign, clip, rint, trunc, fix, square, reciprocal, cbrt
          final x = GpuArray.fromList([-2.7, 0.0, 3.2], [3], DType.float64);
          expect(sign(x).toList(), equals([-1.0, 0.0, 1.0]));
          expect(x.clip(-1.0, 2.0).toList(), equals([-1.0, 0.0, 2.0]));
          expect(clip(x, -1.0, 2.0).toList(), equals([-1.0, 0.0, 2.0]));
          expect(trunc(x).toList(), equals([-2.0, 0.0, 3.0]));
          expect(fix(x).toList(), equals([-2.0, 0.0, 3.0]));
          expect(rint(x).toList(), equals([-3.0, 0.0, 3.0]));

          // Predicates & nanToNum
          final special = GpuArray.fromList(
            [1.0, double.nan, double.infinity, double.negativeInfinity],
            [4],
            DType.float64,
          );
          expect(isnan(special).toList(), equals([false, true, false, false]));
          expect(isinf(special).toList(), equals([false, false, true, true]));
          expect(
            isfinite(special).toList(),
            equals([true, false, false, false]),
          );
          final cleaned = nanToNum(
            special,
            nan: 0.0,
            posinf: 99.0,
            neginf: -99.0,
          );
          expect(cleaned.toList(), equals([1.0, 0.0, 99.0, -99.0]));

          // isClose / isclose & allClose / allclose
          final u = GpuArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final v = GpuArray.fromList(
            [1.0, 2.0 + 1e-7, 3.1],
            [3],
            DType.float64,
          );
          expect(
            isClose(u, v, atol: 1e-5).toList(),
            equals([true, true, false]),
          );
          expect(
            isclose(u, v, atol: 1e-5).toList(),
            equals([true, true, false]),
          );
          expect(allClose(u, v, atol: 1e-5), isFalse);
          expect(allclose(u, v, atol: 0.2), isTrue);

          // Complex components: real, imag, conj, conjugate, angle
          final c64 = GpuArray.fromList(
            [Complex(1.0, 1.0), Complex(0.0, -2.0)],
            [2],
            DType.complex64,
          );
          final GpuArray<Float32> r64 = c64.real();
          final GpuArray<Float32> i64 = c64.imag();
          final GpuArray<Complex64> cj64 = c64.conjugate();
          final GpuArray<Float32> angDeg = c64.angle(deg: true);
          expect(r64.dtype, equals(DType.float32));
          expect(r64.toList(), equals([1.0, 0.0]));
          expect(i64.dtype, equals(DType.float32));
          expect(i64.toList(), equals([1.0, -2.0]));
          expect(
            cj64.toList(),
            equals([Complex(1.0, -1.0), Complex(0.0, 2.0)]),
          );
          expect((angDeg.toList()[0] as num).toDouble(), closeTo(45.0, 1e-3));
          expect((angDeg.toList()[1] as num).toDouble(), closeTo(-90.0, 1e-3));
          expect(real(c64).toList(), equals([1.0, 0.0]));
          expect(imag(c64).toList(), equals([1.0, -2.0]));
          expect(
            conj(c64).toList(),
            equals([Complex(1.0, -1.0), Complex(0.0, 2.0)]),
          );
        });
      },
    );

    test(
      'Widened GpuArray<DTypeTag> with mismatched out buffer throws ArgumentError not TypeError',
      () {
        ResourceScope.scope(() {
          final a = GpuArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float32);
          final widened = a as GpuArray<DTypeTag>;
          final wrongDtype = GpuArray.zeros([4], DType.int32);

          expect(
            () => widened.abs(out: wrongDtype),
            throwsA(
              isA<ArgumentError>().having(
                (e) => e is! TypeError,
                'not TypeError',
                isTrue,
              ),
            ),
          );
          expect(
            () => widened.copy(out: wrongDtype),
            throwsA(
              isA<ArgumentError>().having(
                (e) => e is! TypeError,
                'not TypeError',
                isTrue,
              ),
            ),
          );
          expect(
            () => widened.negate(out: wrongDtype),
            throwsA(
              isA<ArgumentError>().having(
                (e) => e is! TypeError,
                'not TypeError',
                isTrue,
              ),
            ),
          );
          expect(
            () => widened.sqrt(out: wrongDtype),
            throwsA(
              isA<ArgumentError>().having(
                (e) => e is! TypeError,
                'not TypeError',
                isTrue,
              ),
            ),
          );
        });
      },
    );
  });
}
