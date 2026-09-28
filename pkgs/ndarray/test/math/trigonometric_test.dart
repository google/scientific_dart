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

import 'package:ndarray/ndarray.dart';
import 'dart:typed_data';
import 'package:test/test.dart';
import 'dart:math' as math;

void main() {
  group('Trigonometric Tests', () {
    test('sin/cos/tan fallbacks (int types)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([0, 1, 2], [3], DType.int32);

        final s = sin(a);
        expect(s.dtype, DType.float64);
        expect(s.toList()[0], closeTo(math.sin(0), 1e-10));
        expect(s.toList()[1], closeTo(math.sin(1), 1e-10));
        expect(s.toList()[2], closeTo(math.sin(2), 1e-10));

        final c = cos(a);
        expect(c.toList()[0], closeTo(math.cos(0), 1e-10));
        expect(c.toList()[1], closeTo(math.cos(1), 1e-10));
        expect(c.toList()[2], closeTo(math.cos(2), 1e-10));

        final t = tan(a);
        expect(t.toList()[0], closeTo(math.tan(0), 1e-10));
        expect(t.toList()[1], closeTo(math.tan(1), 1e-10));
        expect(t.toList()[2], closeTo(math.tan(2), 1e-10));
      });
    });

    test('asin/acos/atan fallbacks (int types)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([0, 1], [2], DType.int32);

        final s = asin(a);
        expect(s.toList()[0], closeTo(math.asin(0), 1e-10));
        expect(s.toList()[1], closeTo(math.asin(1), 1e-10));

        final c = acos(a);
        expect(c.toList()[0], closeTo(math.acos(0), 1e-10));
        expect(c.toList()[1], closeTo(math.acos(1), 1e-10));

        final t = atan(a);
        expect(t.toList()[0], closeTo(math.atan(0), 1e-10));
        expect(t.toList()[1], closeTo(math.atan(1), 1e-10));

        // Specific check for asin/acos/atan being correct
        final a2 = NDArray.fromList([1], [1], DType.int32);
        expect(asin(a2).toList()[0], closeTo(math.asin(1), 1e-10));
        expect(acos(a2).toList()[0], closeTo(math.acos(1), 1e-10));
        expect(atan(a2).toList()[0], closeTo(math.atan(1), 1e-10));
      });
    });

    test('sin/cos strided fallbacks (int types)', () {
      NDArray.scope(() {
        // [0, 100, 1, 200, 2] -> slice [0, 1, 2]
        final a = NDArray.fromList([0, 100, 1, 200, 2], [5], DType.int32);
        final sliced = a.slice([
          const Slice(start: 0, stop: 5, step: 2),
        ]); // [0, 1, 2]
        expect(sliced.shape, [3]);
        expect(sliced.isContiguous, false);

        final s = sin(sliced);
        expect(s.shape, [3]);
        expect(s.toList()[0], closeTo(math.sin(0), 1e-10));
        expect(s.toList()[1], closeTo(math.sin(1), 1e-10));
        expect(s.toList()[2], closeTo(math.sin(2), 1e-10));

        final c = cos(sliced);
        expect(c.toList()[0], closeTo(math.cos(0), 1e-10));
        expect(c.toList()[1], closeTo(math.cos(1), 1e-10));
        expect(c.toList()[2], closeTo(math.cos(2), 1e-10));
      });
    });

    test('atan2 contiguous float64', () {
      NDArray.scope(() {
        final y = NDArray.fromList([1.0, 0.0], [2], DType.float64);
        final x = NDArray.fromList([1.0, 1.0], [2], DType.float64);
        final res = atan2(y, x);
        expect(res.toList()[0], closeTo(math.atan2(1, 1), 1e-10));
        expect(res.toList()[1], closeTo(math.atan2(0, 1), 1e-10));
      });
    });

    test('atan2 contiguous float32', () {
      NDArray.scope(() {
        final y = NDArray.fromList([1.0, 0.0], [2], DType.float32);
        final x = NDArray.fromList([1.0, 1.0], [2], DType.float32);
        final res = atan2(y, x);
        expect(res.dtype, DType.float32);
        expect(res.toList()[0], closeTo(math.atan2(1, 1), 1e-7));
        expect(res.toList()[1], closeTo(math.atan2(0, 1), 1e-7));
      });
    });

    test('atan2 strided broadcasting', () {
      NDArray.scope(() {
        // y: [1.0, 2.0] (2,)
        // x: [[1.0], [2.0]] (2, 1)
        // result: (2, 2)
        final y = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final x = NDArray.fromList([1.0, 2.0], [2, 1], DType.float64);
        final res = atan2(y, x);

        expect(res.shape, [2, 2]);
        // [atan2(1,1), atan2(2,1)]
        // [atan2(1,2), atan2(2,2)]
        expect(res[[0, 0]], closeTo(math.atan2(1, 1), 1e-10));
        expect(res[[0, 1]], closeTo(math.atan2(2, 1), 1e-10));
        expect(res[[1, 0]], closeTo(math.atan2(1, 2), 1e-10));
        expect(res[[1, 1]], closeTo(math.atan2(2, 2), 1e-10));
      });
    });

    test('atan2 int fallbacks', () {
      NDArray.scope(() {
        final y = NDArray.fromList([1, 0], [2], DType.int32);
        final x = NDArray.fromList([1, 1], [2], DType.int32);
        final res = atan2(y, x);
        expect(res.toList()[0], closeTo(math.atan2(1, 1), 1e-10));
        expect(res.toList()[1], closeTo(math.atan2(0, 1), 1e-10));
      });
    });
  });

  group('Sinh Bug Repro', () {
    test('sinh on non-contiguous int32 array', () {
      NDArray.scope(() {
        final a = NDArray.fromList([0, 1, 2, 3], [4], DType.int32);
        final sliced = a.slice([const Slice(start: 0, stop: 4, step: 2)]);
        expect(sliced.isContiguous, isFalse);
        final result = sinh(sliced);
        expect(result.shape, [2]);
        expect(result.toList()[0], 0.0);
        expect(
          result.toList()[1],
          closeTo((math.exp(2.0) - math.exp(-2.0)) / 2.0, 1e-9),
        );
      });
    });
  });

  group('Angle Converters (deg2rad, rad2deg) Tests', () {
    test(
      'deg2rad basic float64 contiguous conversion checks',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [180.0, 90.0, 0.0, -45.0],
          [4],
          DType.float64,
        );
        final res = deg2rad(a);

        expect(res.dtype, DType.float64);
        expect(res.toList()[0], closeTo(math.pi, 1e-10));
        expect(res.toList()[1], closeTo(math.pi / 2.0, 1e-10));
        expect(res.toList()[2], 0.0);
        expect(res.toList()[3], closeTo(-math.pi / 4.0, 1e-10));
      }),
    );

    test(
      'rad2deg basic float64 contiguous conversion checks',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [math.pi, math.pi / 2.0, 0.0, -math.pi / 4.0],
          [4],
          DType.float64,
        );
        final res = rad2deg(a);

        expect(res.dtype, DType.float64);
        expect(res.toList()[0], closeTo(180.0, 1e-10));
        expect(res.toList()[1], closeTo(90.0, 1e-10));
        expect(res.toList()[2], 0.0);
        expect(res.toList()[3], closeTo(-45.0, 1e-10));
      }),
    );

    test(
      'deg2rad and rad2deg support Float32',
      () => NDArray.scope(() {
        final a = NDArray.fromList([180.0, 90.0], [2], DType.float32);
        final r = deg2rad(a);
        expect(r.dtype, DType.float32);
        expect(r.toList()[0], closeTo(math.pi, 1e-5));

        final b = NDArray.fromList(
          [math.pi, math.pi / 2.0],
          [2],
          DType.float32,
        );
        final d = rad2deg(b);
        expect(d.dtype, DType.float32);
        expect(d.toList()[0], closeTo(180.0, 1e-5));
      }),
    );

    test(
      'Complex arrays throw UnsupportedError',
      () => NDArray.scope(() {
        final a = NDArray.create([2], DType.complex128);
        expect(() => deg2rad(a), throwsUnsupportedError);
        expect(() => rad2deg(a), throwsUnsupportedError);
      }),
    );

    test(
      'Disposed arrays throw StateError',
      () => NDArray.scope(() {
        final a = NDArray.fromList([180.0], [1], DType.float64);
        a.dispose();
        expect(() => deg2rad(a), throwsStateError);
        expect(() => rad2deg(a), throwsStateError);
      }),
    );
  });

  group('hypot and power ufuncs', () {
    test('hypot basic', () {
      final a = NDArray.fromList(Float64List.fromList([3.0, 5.0]), [
        2,
      ], DType.float64);
      final b = NDArray.fromList(Float64List.fromList([4.0, 12.0]), [
        2,
      ], DType.float64);
      final h = hypot(a, b);
      expect(h.toList()[0], closeTo(5.0, 1e-10));
      expect(h.toList()[1], closeTo(13.0, 1e-10));
      expect(h.dtype, DType.float64);
    });

    test('hypot broadcasting', () {
      final a = NDArray.fromList(Float64List.fromList([3.0, 8.0]), [
        2,
      ], DType.float64);
      final b = NDArray.fromList(Float64List.fromList([4.0]), [
        1,
      ], DType.float64);
      final h = hypot(a, b);
      expect(h.toList()[0], closeTo(5.0, 1e-10));
      expect(h.toList()[1], closeTo(math.sqrt(80.0), 1e-10));
    });

    test('power basic', () {
      final a = NDArray.fromList(Float64List.fromList([2.0, 3.0]), [
        2,
      ], DType.float64);
      final b = NDArray.fromList(Float64List.fromList([3.0, 2.0]), [
        2,
      ], DType.float64);
      final p = power(a, b);
      expect(p.toList()[0], 8.0);
      expect(p.toList()[1], 9.0);
    });

    test('power broadcasting', () {
      final a = NDArray.fromList(Int32List.fromList([2, 3]), [2], DType.int32);
      final b = NDArray.fromList(Int32List.fromList([2]), [1], DType.int32);
      final p = power(a, b);
      expect(p.toList()[0], 4);
      expect(p.toList()[1], 9);
      expect(p.dtype, DType.int32);
    });
  });

  group('sign ufunc', () {
    test('int32 sign', () {
      final a = NDArray.fromList(Int32List.fromList([-5, 0, 10]), [
        3,
      ], DType.int32);
      final s = sign(a);
      expect(s.toList(), [-1, 0, 1]);
      expect(s.dtype, DType.int32);
    });

    test('float64 sign', () {
      final a = NDArray.fromList(
        Float64List.fromList([-5.5, 0.0, 10.2, double.nan]),
        [4],
        DType.float64,
      );
      final s = sign(a);
      expect(s.toList()[0], -1.0);
      expect(s.toList()[1], 0.0);
      expect(s.toList()[2], 1.0);
      expect(s.toList()[3].isNaN, isTrue);
      expect(s.dtype, DType.float64);
    });

    test('complex128 sign', () {
      final a = NDArray.fromList(
        [Complex(3.0, 4.0), Complex(0.0, 0.0)],
        [2],
        DType.complex128,
      );

      final s = sign(a);
      expect(s.toList()[0].real, closeTo(0.6, 1e-10));
      expect(s.toList()[0].imag, closeTo(0.8, 1e-10));
      expect(s.toList()[1].real, 0.0);
      expect(s.toList()[1].imag, 0.0);
      expect(s.dtype, DType.complex128);
    });
    group('FFI accelerated Hyperbolic and Trigonometric Tests', () {
      test(
        'linalg.sinh(), cosh(), tanh(), asinh(), acosh(), and atanh() FFI-accelerated correctness',
        () => NDArray.scope(() {
          final a = NDArray.fromList([0.0, 1.0, -1.0], [3], DType.float64);

          // 1. sinh
          final sh = sinh(a);
          expect(sh.getCell([0]), closeTo(0.0, 1e-9));
          expect(
            sh.getCell([1]),
            closeTo((math.exp(1.0) - math.exp(-1.0)) / 2.0, 1e-9),
          );

          // 2. cosh
          final ch = cosh(a);
          expect(ch.getCell([0]), closeTo(1.0, 1e-9));
          expect(
            ch.getCell([1]),
            closeTo((math.exp(1.0) + math.exp(-1.0)) / 2.0, 1e-9),
          );

          // 3. tanh
          final th = tanh(a);
          expect(th.getCell([0]), closeTo(0.0, 1e-9));
          expect(
            th.getCell([1]),
            closeTo((math.exp(2.0) - 1) / (math.exp(2.0) + 1), 1e-9),
          );

          // 4. inverse hyperbolic loops
          final ash = asinh(sh);
          expect(allClose(ash, a, atol: 1e-9), true);

          // 5. in-place recycler out reuse
          final recycler = NDArray.zeros([3], DType.float64);
          final shRec = sinh(a, out: recycler);
          expect(identical(shRec, recycler), true);
          expect(shRec.getCell([0]), closeTo(0.0, 1e-9));
        }),
      );

      test(
        'linalg.sin(), cos(), and tan() complex FFI-accelerated correctness',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [Complex(0.0, 0.0), Complex(0.1, 0.2), Complex(0.1, -0.2)],
            [3],
            DType.complex128,
          );

          // 1. sin(z)
          final s = sin(a);
          expect(s.dtype, DType.complex128);
          expect(s.getCell([0]), Complex(0.0, 0.0));
          expect(
            s.getCell([1]).real,
            closeTo(
              math.sin(0.1) * ((math.exp(0.2) + math.exp(-0.2)) / 2.0),
              1e-9,
            ),
          );
          expect(
            s.getCell([1]).imag,
            closeTo(
              math.cos(0.1) * ((math.exp(0.2) - math.exp(-0.2)) / 2.0),
              1e-9,
            ),
          );

          // 2. cos(z)
          final c = cos(a);
          expect(c.dtype, DType.complex128);
          expect(c.getCell([0]), Complex(1.0, 0.0));
          expect(
            c.getCell([1]).real,
            closeTo(
              math.cos(0.1) * ((math.exp(0.2) + math.exp(-0.2)) / 2.0),
              1e-9,
            ),
          );
          expect(
            c.getCell([1]).imag,
            closeTo(
              -math.sin(0.1) * ((math.exp(0.2) - math.exp(-0.2)) / 2.0),
              1e-9,
            ),
          );

          // 3. tan(z)
          final t = tan(a);
          expect(t.dtype, DType.complex128);
          expect(t.getCell([0]), Complex(0.0, 0.0));
          final denom =
              math.cos(0.2) + ((math.exp(0.4) + math.exp(-0.4)) / 2.0);
          expect(t.getCell([1]).real, closeTo(math.sin(0.2) / denom, 1e-9));
          expect(
            t.getCell([1]).imag,
            closeTo(((math.exp(0.4) - math.exp(-0.4)) / 2.0) / denom, 1e-9),
          );

          // 4. in-place out recycler reuse
          final recycler = NDArray.zeros([3], DType.complex128);
          final sRec = sin(a, out: recycler);
          expect(identical(sRec, recycler), true);
          expect(sRec.getCell([0]), Complex(0.0, 0.0));

          // 5. inverse complex trig (asin, acos, atan)
          final as = asin(s);

          for (var i = 0; i < 3; i++) {
            expect(as.getCell([i]).real, closeTo(a.getCell([i]).real, 1e-8));
            expect(as.getCell([i]).imag, closeTo(a.getCell([i]).imag, 1e-8));
          }

          final ac = acos(c);
          for (var i = 0; i < 3; i++) {
            expect(ac.getCell([i]).real, closeTo(a.getCell([i]).real, 1e-8));
            expect(ac.getCell([i]).imag, closeTo(a.getCell([i]).imag, 1e-8));
          }

          final at = atan(t);
          for (var i = 0; i < 3; i++) {
            expect(at.getCell([i]).real, closeTo(a.getCell([i]).real, 1e-8));
            expect(at.getCell([i]).imag, closeTo(a.getCell([i]).imag, 1e-8));
          }
          // 6. complex atanh, hypot, power
          final ath = atanh(a);
          expect(ath.dtype, DType.complex128);
          expect(
            ath.getCell([1]).real,
            closeTo(
              0.25 * math.log(((1.1) * (1.1) + 0.04) / ((0.9) * (0.9) + 0.04)),
              1e-9,
            ),
          );
          expect(
            ath.getCell([1]).imag,
            closeTo(0.5 * math.atan2(0.4, 1.0 - 0.01 - 0.04), 1e-9),
          );

          final h = hypot(a, a);
          expect(h.dtype, DType.float64);
          expect(h.getCell([1]), closeTo(math.sqrt(0.1), 1e-9));

          final z2 = NDArray.fromList(List.filled(3, Complex(2.0, 0.0)), [
            3,
          ], DType.complex128);
          final p = power(a, z2);
          expect(p.dtype, DType.complex128);
          expect(p.getCell([1]).real, closeTo(-0.03, 1e-9));
          expect(p.getCell([1]).imag, closeTo(0.04, 1e-9));

          // 7. FFI complex conjugation (conj)
          final conjContig = conj(a);
          expect(conjContig.dtype, DType.complex128);
          expect(conjContig.getCell([0]), Complex(0.0, 0.0));
          expect(conjContig.getCell([1]), Complex(0.1, -0.2));
          expect(conjContig.getCell([2]), Complex(0.1, 0.2));

          // Strided complex view conjugation
          final sliceA = a.slice([Slice(start: 1, stop: 3)]); // shape [2]
          final conjSlice = conj(sliceA);
          expect(conjSlice.dtype, DType.complex128);
          expect(conjSlice.shape, [2]);
          expect(conjSlice.getCell([0]), Complex(0.1, -0.2));
          expect(conjSlice.getCell([1]), Complex(0.1, 0.2));

          // Real array conjugation
          final realA = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final conjReal = conj(realA);
          expect(conjReal.dtype, DType.float64);
          expect(conjReal.getCell([0]), 1.0);
          expect(conjReal.getCell([1]), 2.0);

          // Recycler out buffer reuse
          final conjRecycler = NDArray.zeros([3], DType.complex128);
          final conjRes = conj(a, out: conjRecycler);
          expect(identical(conjRes, conjRecycler), true);
          expect(conjRes.getCell([1]), Complex(0.1, -0.2));
        }),
      );
    });
  });

  group('Phase 2: Hyperbolic and Transcendental DType Coverage', () {
    // Helper to calculate expected complex cosh
    Complex expCosh(Complex z) {
      final x = z.real;
      final y = z.imag;
      final sh = (math.exp(x) - math.exp(-x)) / 2.0;
      final ch = (math.exp(x) + math.exp(-x)) / 2.0;
      return Complex(ch * math.cos(y), sh * math.sin(y));
    }

    // Helper to calculate expected complex tanh
    Complex expTanh(Complex z) {
      final x = z.real;
      final y = z.imag;
      final denom =
          ((math.exp(2 * x) + math.exp(-2 * x)) / 2.0) + math.cos(2 * y);
      final numReal = (math.exp(2 * x) - math.exp(-2 * x)) / 2.0;
      final numImag = math.sin(2 * y);
      return Complex(numReal / denom, numImag / denom);
    }

    test('acosh contiguous and strided (float64, float32)', () {
      NDArray.scope(() {
        // float64 contiguous
        final a64 = NDArray.fromList([1.0, 2.0, 5.0], [3], DType.float64);
        final res64 = acosh(a64);
        expect(res64.dtype, DType.float64);
        expect(res64.getCell([0]), closeTo(0.0, 1e-12));
        expect(
          res64.getCell([1]),
          closeTo(math.log(2.0 + math.sqrt(3.0)), 1e-12),
        );
        expect(
          res64.getCell([2]),
          closeTo(math.log(5.0 + math.sqrt(24.0)), 1e-12),
        );

        // float64 strided (step 2)
        final a64Strided = NDArray.fromList(
          [1.0, -9.9, 2.0, -9.9, 5.0],
          [5],
          DType.float64,
        ).slice([const Slice(start: 0, stop: 5, step: 2)]);
        expect(a64Strided.isContiguous, false);
        final res64Strided = acosh(a64Strided);
        expect(res64Strided.getCell([0]), closeTo(0.0, 1e-12));
        expect(
          res64Strided.getCell([1]),
          closeTo(math.log(2.0 + math.sqrt(3.0)), 1e-12),
        );
        expect(
          res64Strided.getCell([2]),
          closeTo(math.log(5.0 + math.sqrt(24.0)), 1e-12),
        );

        // float32 contiguous
        final a32 = NDArray.fromList([1.0, 2.0, 5.0], [3], DType.float32);
        final res32 = acosh(a32);
        expect(res32.dtype, DType.float32);
        expect(res32.getCell([0]), closeTo(0.0, 1e-6));
        expect(
          res32.getCell([1]),
          closeTo(math.log(2.0 + math.sqrt(3.0)), 1e-6),
        );
        expect(
          res32.getCell([2]),
          closeTo(math.log(5.0 + math.sqrt(24.0)), 1e-6),
        );

        // float32 strided (step 2)
        final a32Strided = NDArray.fromList(
          [1.0, -9.9, 2.0, -9.9, 5.0],
          [5],
          DType.float32,
        ).slice([const Slice(start: 0, stop: 5, step: 2)]);
        final res32Strided = acosh(a32Strided);
        expect(res32Strided.getCell([0]), closeTo(0.0, 1e-6));
        expect(
          res32Strided.getCell([1]),
          closeTo(math.log(2.0 + math.sqrt(3.0)), 1e-6),
        );
        expect(
          res32Strided.getCell([2]),
          closeTo(math.log(5.0 + math.sqrt(24.0)), 1e-6),
        );
      });
    });

    test(
      'tanh, cosh, asinh, atanh across float32, complex128, complex64 (contiguous & strided)',
      () {
        NDArray.scope(() {
          // --- float32 ---
          final f32Contig = NDArray.fromList(
            [0.0, 0.5, -0.5],
            [3],
            DType.float32,
          );
          final f32Strided = NDArray.fromList(
            [0.0, 9.9, 0.5, 9.9, -0.5],
            [5],
            DType.float32,
          ).slice([const Slice(start: 0, stop: 5, step: 2)]);

          for (final a in [f32Contig, f32Strided]) {
            final t = tanh(a);
            expect(t.dtype, DType.float32);
            expect(t.getCell([0]), closeTo(0.0, 1e-6));
            expect(t.getCell([1]), closeTo(0.46211715, 1e-6));
            expect(t.getCell([2]), closeTo(-0.46211715, 1e-6));

            final c = cosh(a);
            expect(c.dtype, DType.float32);
            expect(c.getCell([0]), closeTo(1.0, 1e-6));
            expect(c.getCell([1]), closeTo(1.1276259, 1e-6));
            expect(c.getCell([2]), closeTo(1.1276259, 1e-6));

            final as = asinh(a);
            expect(as.dtype, DType.float32);
            expect(as.getCell([0]), closeTo(0.0, 1e-6));
            expect(as.getCell([1]), closeTo(0.4812118, 1e-6));
            expect(as.getCell([2]), closeTo(-0.4812118, 1e-6));

            final at = atanh(a);
            expect(at.dtype, DType.float32);
            expect(at.getCell([0]), closeTo(0.0, 1e-6));
            expect(at.getCell([1]), closeTo(0.5493061, 1e-6));
            expect(at.getCell([2]), closeTo(-0.5493061, 1e-6));
          }

          // --- complex128 ---
          final c128Contig = NDArray.fromList(
            [Complex(0.2, 0.3), Complex(-0.4, 0.1)],
            [2],
            DType.complex128,
          );
          final c128Strided = NDArray.fromList(
            [
              Complex(0.2, 0.3),
              Complex(99, 99),
              Complex(-0.4, 0.1),
              Complex(99, 99),
            ],
            [4],
            DType.complex128,
          ).slice([const Slice(start: 0, stop: 4, step: 2)]);

          for (final a in [c128Contig, c128Strided]) {
            // cosh
            final c = cosh(a);
            expect(c.dtype, DType.complex128);
            for (var i = 0; i < 2; i++) {
              final expected = expCosh(a.getCell([i]));
              expect(c.getCell([i]).real, closeTo(expected.real, 1e-12));
              expect(c.getCell([i]).imag, closeTo(expected.imag, 1e-12));
            }

            // tanh
            final t = tanh(a);
            expect(t.dtype, DType.complex128);
            for (var i = 0; i < 2; i++) {
              final expected = expTanh(a.getCell([i]));
              expect(t.getCell([i]).real, closeTo(expected.real, 1e-12));
              expect(t.getCell([i]).imag, closeTo(expected.imag, 1e-12));
            }

            // sinh & asinh round-trip
            final sh = sinh(a);
            final ash = asinh(sh);
            expect(ash.dtype, DType.complex128);
            for (var i = 0; i < 2; i++) {
              expect(
                ash.getCell([i]).real,
                closeTo(a.getCell([i]).real, 1e-12),
              );
              expect(
                ash.getCell([i]).imag,
                closeTo(a.getCell([i]).imag, 1e-12),
              );
            }

            // tanh & atanh round-trip
            final ath = atanh(t);
            expect(ath.dtype, DType.complex128);
            for (var i = 0; i < 2; i++) {
              expect(
                ath.getCell([i]).real,
                closeTo(a.getCell([i]).real, 1e-12),
              );
              expect(
                ath.getCell([i]).imag,
                closeTo(a.getCell([i]).imag, 1e-12),
              );
            }
          }

          // --- complex64 ---
          final c64Contig = NDArray.fromList(
            [Complex(0.2, 0.3), Complex(-0.4, 0.1)],
            [2],
            DType.complex64,
          );
          final c64Strided = NDArray.fromList(
            [
              Complex(0.2, 0.3),
              Complex(99, 99),
              Complex(-0.4, 0.1),
              Complex(99, 99),
            ],
            [4],
            DType.complex64,
          ).slice([const Slice(start: 0, stop: 4, step: 2)]);

          for (final a in [c64Contig, c64Strided]) {
            // cosh
            final c = cosh(a);
            expect(c.dtype, DType.complex64);
            for (var i = 0; i < 2; i++) {
              final expected = expCosh(a.getCell([i]));
              expect(c.getCell([i]).real, closeTo(expected.real, 1e-6));
              expect(c.getCell([i]).imag, closeTo(expected.imag, 1e-6));
            }

            // tanh
            final t = tanh(a);
            expect(t.dtype, DType.complex64);
            for (var i = 0; i < 2; i++) {
              final expected = expTanh(a.getCell([i]));
              expect(t.getCell([i]).real, closeTo(expected.real, 1e-6));
              expect(t.getCell([i]).imag, closeTo(expected.imag, 1e-6));
            }

            // sinh & asinh round-trip
            final sh = sinh(a);
            final ash = asinh(sh);
            expect(ash.dtype, DType.complex64);
            for (var i = 0; i < 2; i++) {
              expect(ash.getCell([i]).real, closeTo(a.getCell([i]).real, 1e-6));
              expect(ash.getCell([i]).imag, closeTo(a.getCell([i]).imag, 1e-6));
            }

            // tanh & atanh round-trip
            final ath = atanh(t);
            expect(ath.dtype, DType.complex64);
            for (var i = 0; i < 2; i++) {
              expect(ath.getCell([i]).real, closeTo(a.getCell([i]).real, 1e-6));
              expect(ath.getCell([i]).imag, closeTo(a.getCell([i]).imag, 1e-6));
            }
          }
        });
      },
    );

    test('Strided Unary Operators (sin, cos, square, sqrt)', () {
      NDArray.scope(() {
        // float64 strided
        final a = NDArray.fromList(
          [1.0, -9.9, 4.0, -9.9, 9.0],
          [5],
          DType.float64,
        ).slice([const Slice(start: 0, stop: 5, step: 2)]);
        expect(a.isContiguous, false);

        final sq = square(a);
        expect(sq.dtype, DType.float64);
        expect(sq.getCell([0]), closeTo(1.0, 1e-12));
        expect(sq.getCell([1]), closeTo(16.0, 1e-12));
        expect(sq.getCell([2]), closeTo(81.0, 1e-12));

        final sqt = sqrt(sq);
        expect(sqt.dtype, DType.float64);
        expect(sqt.getCell([0]), closeTo(1.0, 1e-12));
        expect(sqt.getCell([1]), closeTo(4.0, 1e-12));
        expect(sqt.getCell([2]), closeTo(9.0, 1e-12));

        final s = sin(a);
        expect(s.getCell([0]), closeTo(math.sin(1.0), 1e-12));
        expect(s.getCell([1]), closeTo(math.sin(4.0), 1e-12));
        expect(s.getCell([2]), closeTo(math.sin(9.0), 1e-12));

        final c = cos(a);
        expect(c.getCell([0]), closeTo(math.cos(1.0), 1e-12));
        expect(c.getCell([1]), closeTo(math.cos(4.0), 1e-12));
        expect(c.getCell([2]), closeTo(math.cos(9.0), 1e-12));

        // complex128 strided
        final z = NDArray.fromList(
          [Complex(1.0, 2.0), Complex(9, 9), Complex(3.0, 4.0), Complex(9, 9)],
          [4],
          DType.complex128,
        ).slice([const Slice(start: 0, stop: 4, step: 2)]);

        final zSq = square(z);
        expect(zSq.getCell([0]), Complex(-3.0, 4.0));
        expect(zSq.getCell([1]), Complex(-7.0, 24.0));

        final zSin = sin(z);
        // sin(1 + 2i) = sin(1)*cosh(2) + i*cos(1)*sinh(2)
        // sinh(2) = 3.626860407847019
        // cosh(2) = 3.7621956910836314
        expect(
          zSin.getCell([0]).real,
          closeTo(math.sin(1.0) * 3.7621956910836314, 1e-12),
        );
        expect(
          zSin.getCell([0]).imag,
          closeTo(math.cos(1.0) * 3.626860407847019, 1e-12),
        );
      });
    });

    test('Complex exp, log, sqrt, abs, acosh (contiguous & strided)', () {
      NDArray.scope(() {
        for (final dtype in <DType<AnySpec>>[
          DType.complex128,
          DType.complex64,
        ]) {
          final isComplex128 = dtype == DType.complex128;
          final double tol = isComplex128 ? 1e-12 : 1e-6;

          // Contiguous
          final cContig = NDArray.fromList(
            [Complex(0.2, 0.3), Complex(-0.4, 0.1)],
            [2],
            dtype,
          );

          // Strided
          final cStrided = NDArray.fromList(
            [
              Complex(0.2, 0.3),
              Complex(99, 99),
              Complex(-0.4, 0.1),
              Complex(99, 99),
            ],
            [4],
            dtype,
          ).slice([const Slice(start: 0, stop: 4, step: 2)]);

          for (final a in <NDArray<AnySpec>>[cContig, cStrided]) {
            // exp & log round-trip
            final e = exp(a) as NDArray<AnySpec>;
            expect(e.dtype, dtype);
            final le = log(e);
            expect(le.dtype, dtype);
            for (var i = 0; i < 2; i++) {
              expect(le.getCell([i]).real, closeTo(a.getCell([i]).real, tol));
              expect(le.getCell([i]).imag, closeTo(a.getCell([i]).imag, tol));
            }

            // sqrt & square round-trip
            final sqt = sqrt(a);
            expect(sqt.dtype, dtype);
            final sq = square(sqt);
            expect(sq.dtype, dtype);
            for (var i = 0; i < 2; i++) {
              expect(sq.getCell([i]).real, closeTo(a.getCell([i]).real, tol));
              expect(sq.getCell([i]).imag, closeTo(a.getCell([i]).imag, tol));
            }

            // abs (magnitude)
            final ab = abs(a);
            expect(ab.dtype, isComplex128 ? DType.float64 : DType.float32);
            for (var i = 0; i < 2; i++) {
              final expectedAbs = math.sqrt(
                a.getCell([i]).real * a.getCell([i]).real +
                    a.getCell([i]).imag * a.getCell([i]).imag,
              );
              expect(ab.getCell([i]), closeTo(expectedAbs, tol));
            }

            // cosh & acosh round-trip (cosh(acosh(a)) == a)
            final ach = acosh(a) as NDArray<AnySpec>;
            expect(ach.dtype, dtype);
            final ch = cosh(ach);
            expect(ch.dtype, dtype);
            for (var i = 0; i < 2; i++) {
              expect(ch.getCell([i]).real, closeTo(a.getCell([i]).real, tol));
              expect(ch.getCell([i]).imag, closeTo(a.getCell([i]).imag, tol));
            }
          }
        }
      });
    });
  });

  group('Integer Abs Tests', () {
    test('Contiguous integer abs', () {
      NDArray.scope(() {
        final a64 = NDArray.fromList([-1, -2, 3, 0], [4], DType.int64);
        final a32 = NDArray.fromList([-1, -2, 3, 0], [4], DType.int32);
        final a16 = NDArray.fromList([-1, -2, 3, 0], [4], DType.int16);
        final u8 = NDArray.fromList([1, 2, 3, 0], [4], DType.uint8);

        final r64 = abs(a64);
        final r32 = abs(a32);
        final r16 = abs(a16);
        final ru8 = abs(u8);

        expect(r64.dtype, DType.int64);
        expect(r64.toList(), [1, 2, 3, 0]);

        expect(r32.dtype, DType.int32);
        expect(r32.toList(), [1, 2, 3, 0]);

        expect(r16.dtype, DType.int16);
        expect(r16.toList(), [1, 2, 3, 0]);

        expect(ru8.dtype, DType.uint8);
        expect(ru8.toList(), [1, 2, 3, 0]);
      });
    });

    test('Strided integer abs', () {
      NDArray.scope(() {
        final a64 = NDArray.fromList(
          [-1, 99, -2, 99, 3, 99, 0],
          [7],
          DType.int64,
        ).slice([const Slice(start: 0, stop: 7, step: 2)]);
        final a32 = NDArray.fromList(
          [-1, 99, -2, 99, 3, 99, 0],
          [7],
          DType.int32,
        ).slice([const Slice(start: 0, stop: 7, step: 2)]);
        final a16 = NDArray.fromList(
          [-1, 99, -2, 99, 3, 99, 0],
          [7],
          DType.int16,
        ).slice([const Slice(start: 0, stop: 7, step: 2)]);
        final u8 = NDArray.fromList(
          [1, 99, 2, 99, 3, 99, 0],
          [7],
          DType.uint8,
        ).slice([const Slice(start: 0, stop: 7, step: 2)]);

        expect(a64.isContiguous, false);

        final r64 = abs(a64);
        final r32 = abs(a32);
        final r16 = abs(a16);
        final ru8 = abs(u8);

        expect(r64.dtype, DType.int64);
        expect(r64.toList(), [1, 2, 3, 0]);

        expect(r32.dtype, DType.int32);
        expect(r32.toList(), [1, 2, 3, 0]);

        expect(r16.dtype, DType.int16);
        expect(r16.toList(), [1, 2, 3, 0]);

        expect(ru8.dtype, DType.uint8);
        expect(ru8.toList(), [1, 2, 3, 0]);
      });
    });

    test('Integer abs error handling', () {
      NDArray.scope(() {
        final a = NDArray.fromList([-1, -2, 3, 0], [4], DType.int32);
        a.dispose();
        expect(() => abs(a), throwsStateError);

        final aValid = NDArray.fromList([-1, -2, 3, 0], [4], DType.int32);
        final outInvalidShape = NDArray<Int32>.create([3], DType.int32);
        expect(() => abs(aValid, out: outInvalidShape), throwsArgumentError);

        final outInvalidDType = NDArray<Float64>.create([4], DType.float64);
        expect(() => abs(aValid, out: outInvalidDType), throwsArgumentError);
      });
    });
  });

  group('Cycle 8 Stream 2 Remediation Tests', () {
    test('boolean array promotion in transcendental functions', () {
      NDArray.scope(() {
        final b = NDArray.fromList([true, false], [2], DType.boolean);
        final s = sin(b);
        expect(s.dtype, DType.float64);
        expect(s.getCell([0]), closeTo(math.sin(1.0), 1e-10));
        expect(s.getCell([1]), closeTo(math.sin(0.0), 1e-10));

        final c = cos(b);
        expect(c.getCell([0]), closeTo(math.cos(1.0), 1e-10));
        expect(c.getCell([1]), closeTo(math.cos(0.0), 1e-10));

        final at2 = atan2(b, b);
        expect(at2.getCell([0]), closeTo(math.atan2(1.0, 1.0), 1e-10));
        expect(at2.getCell([1]), closeTo(math.atan2(0.0, 0.0), 1e-10));
      });
    });

    test('deg2rad and rad2deg with where mask', () {
      NDArray.scope(() {
        final a = NDArray.fromList([180.0, 90.0, 45.0], [3], DType.float64);
        final mask = NDArray.fromList([true, false, true], [3], DType.boolean);
        final out = NDArray.fromList([-1.0, -1.0, -1.0], [3], DType.float64);

        deg2rad(a, where: mask, out: out);
        expect(out.getCell([0]), closeTo(math.pi, 1e-10));
        expect(out.getCell([1]), closeTo(-1.0, 1e-10));
        expect(out.getCell([2]), closeTo(math.pi / 4.0, 1e-10));

        final rads = NDArray.fromList(
          [math.pi, math.pi / 2.0, math.pi / 4.0],
          [3],
          DType.float64,
        );
        final outDeg = NDArray.fromList([-1.0, -1.0, -1.0], [3], DType.float64);
        rad2deg(rads, where: mask, out: outDeg);
        expect(outDeg.getCell([0]), closeTo(180.0, 1e-10));
        expect(outDeg.getCell([1]), closeTo(-1.0, 1e-10));
        expect(outDeg.getCell([2]), closeTo(45.0, 1e-10));
      });
    });

    test('hypot with real operands and where mask', () {
      NDArray.scope(() {
        final a = NDArray.fromList([3.0, 5.0], [2], DType.float64);
        final b = NDArray.fromList([4.0, 12.0], [2], DType.float64);
        final mask = NDArray.fromList([true, false], [2], DType.boolean);
        final out = NDArray.fromList([-99.0, -99.0], [2], DType.float64);

        hypot(a, b, where: mask, out: out);
        expect(out.getCell([0]), closeTo(5.0, 1e-10));
        expect(out.getCell([1]), closeTo(-99.0, 1e-10));
      });
    });

    test('real, imag, and conj with where mask', () {
      NDArray.scope(() {
        final c = NDArray.fromList(
          [Complex(3.0, 4.0), Complex(-1.0, 2.0)],
          [2],
          DType.complex128,
        );
        final mask = NDArray.fromList([true, false], [2], DType.boolean);

        final outReal = NDArray.fromList([-99.0, -99.0], [2], DType.float64);
        real(c, where: mask, out: outReal);
        expect(outReal.getCell([0]), closeTo(3.0, 1e-10));
        expect(outReal.getCell([1]), closeTo(-99.0, 1e-10));

        final outImag = NDArray.fromList([-99.0, -99.0], [2], DType.float64);
        imag(c, where: mask, out: outImag);
        expect(outImag.getCell([0]), closeTo(4.0, 1e-10));
        expect(outImag.getCell([1]), closeTo(-99.0, 1e-10));

        final outConj = NDArray.fromList(
          [Complex(0, 0), Complex(0, 0)],
          [2],
          DType.complex128,
        );
        conj(c, where: mask, out: outConj);
        expect(outConj.getCell([0]), Complex(3.0, -4.0));
        expect(outConj.getCell([1]), Complex(0.0, 0.0));
      });
    });

    test('nan_to_num and ndenumerate verification without data property', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, double.nan, double.infinity, double.negativeInfinity],
          [2, 2],
          DType.float64,
        );
        final cleaned = nan_to_num(a, nan: 42.0, posinf: 100.0, neginf: -100.0);
        expect(cleaned.getCell([0, 0]), 1.0);
        expect(cleaned.getCell([0, 1]), 42.0);
        expect(cleaned.getCell([1, 0]), 100.0);
        expect(cleaned.getCell([1, 1]), -100.0);

        final coords = <List<int>>[];
        for (final (coord, _) in ndenumerate(a)) {
          coords.add(coord);
        }
        expect(coords, [
          [0, 0],
          [0, 1],
          [1, 0],
          [1, 1],
        ]);
      });
    });
  });
}
