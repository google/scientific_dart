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
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('angle', () {
    test('contiguous complex128', () {
      NDArray.scope(() {
        final a = NDArray<Complex128>.fromList(
          [
            Complex(1.0, 0.0),
            Complex(0.0, 1.0),
            Complex(-1.0, 0.0),
            Complex(0.0, -1.0),
            Complex(1.0, 1.0),
            Complex(-1.0, -1.0),
          ],
          [6],
          DType.complex128,
        );

        final res = angle(a);
        expect(res.dtype, DType.float64);
        expect(res.isContiguous, true);

        final expected = [
          0.0,
          math.pi / 2,
          math.pi,
          -math.pi / 2,
          math.pi / 4,
          -3 * math.pi / 4,
        ];

        for (var i = 0; i < res.size; i++) {
          expect(res.data[i], closeTo(expected[i], 1e-9));
        }
      });
    });

    test('contiguous complex64', () {
      NDArray.scope(() {
        final a = NDArray<Complex64>.fromList(
          [Complex(1.0, 0.0), Complex(0.0, 1.0)],
          [2],
          DType.complex64,
        );

        final res = angle(a);
        expect(res.dtype, DType.float32);
        expect(res.isContiguous, true);

        final expected = [0.0, math.pi / 2];
        for (var i = 0; i < res.size; i++) {
          expect(res.data[i], closeTo(expected[i], 1e-5));
        }
      });
    });

    test('strided complex128', () {
      NDArray.scope(() {
        final a = NDArray<Complex128>.fromList(
          [
            Complex(1.0, 0.0),
            Complex(99.0, 99.0),
            Complex(0.0, 1.0),
            Complex(99.0, 99.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final sliced = a.slice([Slice.all(), Index(0)]);
        expect(sliced.isContiguous, false);

        final res = angle(sliced);
        expect(res.dtype, DType.float64);

        final expected = [0.0, math.pi / 2];
        expect(res.size, 2);
        for (var i = 0; i < res.size; i++) {
          expect(res.getCell([i]), closeTo(expected[i], 1e-9));
        }
      });
    });

    test('out parameter', () {
      NDArray.scope(() {
        final a = NDArray<Complex128>.fromList(
          [Complex(1.0, 0.0), Complex(0.0, 1.0)],
          [2],
          DType.complex128,
        );

        final out = NDArray<Float64>.zeros([2], DType.float64);
        final res = angle(a, out: out);
        expect(identical(res, out), true);
        expect(res.getCell([0]), closeTo(0.0, 1e-9));
        expect(res.getCell([1]), closeTo(math.pi / 2, 1e-9));
      });
    });

    test('invalid type and real/integer support', () {
      NDArray.scope(() {
        final boolArray = NDArray<Boolean>.fromList(
          [true, false],
          [2],
          DType.boolean,
        );
        expect(() => angle(boolArray as dynamic), throwsArgumentError);

        final f64 = NDArray<Float64>.fromList([1.0, -2.0], [2], DType.float64);
        final resF64 = angle(f64);
        expect(resF64.dtype, DType.float64);
        expect(resF64.getCell([0]), closeTo(0.0, 1e-9));
        expect(resF64.getCell([1]), closeTo(math.pi, 1e-9));

        final i32 = NDArray<Int32>.fromList([1, -1], [2], DType.int32);
        final resI32 = angle(i32);
        expect(resI32.dtype, DType.float64);
        expect(resI32.getCell([0]), closeTo(0.0, 1e-9));
        expect(resI32.getCell([1]), closeTo(math.pi, 1e-9));
      });
    });
  });

  group('unwrap', () {
    test('1D float64 default discont', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [
            0.0,
            0.5 * math.pi,
            1.1 * math.pi,
            1.6 * math.pi,
            2.2 * math.pi,
            0.1 * math.pi,
          ],
          [6],
          DType.float64,
        );

        final res = unwrap(a);
        expect(res.dtype, DType.float64);

        final expected = [
          0.0,
          0.5 * math.pi,
          1.1 * math.pi,
          1.6 * math.pi,
          2.2 * math.pi,
          2.1 * math.pi,
        ];

        for (var i = 0; i < res.size; i++) {
          expect(res.data[i], closeTo(expected[i], 1e-9));
        }
      });
    });

    test('1D float64 custom discont', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [0.0, 1.2 * math.pi],
          [2],
          DType.float64,
        );

        final res1 = unwrap(a, discont: 1.5 * math.pi);
        expect(res1.getCell([1]), closeTo(1.2 * math.pi, 1e-9));

        final res2 = unwrap(a);
        expect(res2.getCell([1]), closeTo(-0.8 * math.pi, 1e-9));
      });
    });

    test('1D float32 custom discont', () {
      NDArray.scope(() {
        final a = NDArray<Float32>.fromList(
          [0.0, 1.2 * math.pi],
          [2],
          DType.float32,
        );

        final res1 = unwrap(a, discont: 1.5 * math.pi);
        expect(res1.getCell([1]), closeTo(1.2 * math.pi, 1e-6));

        final res2 = unwrap(a);
        expect(res2.getCell([1]), closeTo(-0.8 * math.pi, 1e-6));
      });
    });

    test('2D unwrap along axis 0', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [0.0, 0.0, 1.5 * math.pi, 1.5 * math.pi, 0.0, 0.0],
          [3, 2],
          DType.float64,
        );

        final res = unwrap(a, axis: 0);

        final expected = [0.0, 0.0, -0.5 * math.pi, -0.5 * math.pi, 0.0, 0.0];

        for (var i = 0; i < res.size; i++) {
          expect(res.data[i], closeTo(expected[i], 1e-9));
        }
      });
    });

    test('2D unwrap along axis 1', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [0.0, 1.5 * math.pi, 1.6 * math.pi, 0.0, 0.0, 0.0],
          [2, 3],
          DType.float64,
        );

        final res = unwrap(a, axis: 1);
        expect(res.shape, equals([2, 3]));

        expect(res.getCell([0, 0]), closeTo(0.0, 1e-9));
        expect(res.getCell([0, 1]), closeTo(-0.5 * math.pi, 1e-9));
        expect(res.getCell([0, 2]), closeTo(-0.4 * math.pi, 1e-9));

        expect(res.getCell([1, 0]), closeTo(0.0, 1e-9));
        expect(res.getCell([1, 1]), closeTo(0.0, 1e-9));
        expect(res.getCell([1, 2]), closeTo(0.0, 1e-9));
      });
    });

    test('unwrap boundary cases (pi and -pi)', () {
      NDArray.scope(() {
        // numpy.unwrap([0.0, pi]) -> [0.0, pi]
        // numpy.unwrap([0.0, -pi]) -> [0.0, -pi]
        final a = NDArray<Float64>.fromList([0.0, math.pi], [2], DType.float64);
        final res = unwrap(a);
        expect(res.data[1], closeTo(math.pi, 1e-9));

        final b = NDArray<Float64>.fromList(
          [0.0, -math.pi],
          [2],
          DType.float64,
        );
        final res2 = unwrap(b);
        expect(res2.data[1], closeTo(-math.pi, 1e-9));
      });
    });
  });

  group('correlate & convolve', () {
    test('1D correlate valid, full, same (Float64)', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float64,
        );
        final v = NDArray<Float64>.fromList(
          [1.0, 10.0, 100.0],
          [3],
          DType.float64,
        );

        final valid = correlate(a, v, mode: ConvMode.valid);
        expect(valid.shape, equals([3]));
        expect(valid.getCell([0]), closeTo(321.0, 1e-9));
        expect(valid.getCell([1]), closeTo(432.0, 1e-9));
        expect(valid.getCell([2]), closeTo(543.0, 1e-9));

        final full = correlate(a, v, mode: ConvMode.full);
        expect(full.shape, equals([7]));
        expect(full.getCell([0]), closeTo(100.0, 1e-9));
        expect(full.getCell([1]), closeTo(210.0, 1e-9));
        expect(full.getCell([2]), closeTo(321.0, 1e-9));

        final same = correlate(a, v, mode: ConvMode.same);
        expect(same.shape, equals([5]));
        expect(same.getCell([0]), closeTo(210.0, 1e-9));
        expect(same.getCell([1]), closeTo(321.0, 1e-9));
      });
    });

    test('correlate out.dtype validation throws ArgumentError on mismatch', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final v = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final outBad = NDArray.zeros([2], DType.float32);
        expect(
          () => correlate(a, v, out: outBad as dynamic),
          throwsArgumentError,
        );
      });
    });

    test('1D convolve valid, full, same (Float64)', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float64,
        );
        final v = NDArray<Float64>.fromList(
          [1.0, 10.0, 100.0],
          [3],
          DType.float64,
        );

        final valid = convolve(a, v, mode: ConvMode.valid);
        expect(valid.shape, equals([3]));
        expect(valid.getCell([0]), closeTo(123.0, 1e-9));
        expect(valid.getCell([1]), closeTo(234.0, 1e-9));
        expect(valid.getCell([2]), closeTo(345.0, 1e-9));

        final full = convolve(a, v, mode: ConvMode.full);
        expect(full.shape, equals([7]));
        expect(full.getCell([0]), closeTo(1.0, 1e-9));
        expect(full.getCell([1]), closeTo(12.0, 1e-9));
        expect(full.getCell([2]), closeTo(123.0, 1e-9));

        final same = convolve(a, v, mode: ConvMode.same);
        expect(same.shape, equals([5]));
        expect(same.getCell([0]), closeTo(12.0, 1e-9));
        expect(same.getCell([1]), closeTo(123.0, 1e-9));
      });
    });

    test('2D convolve2d (Float64)', () {
      NDArray.scope(() {
        final img = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
          [3, 3],
          DType.float64,
        );
        final k = NDArray<Float64>.fromList(
          [1.0, 0.0, 0.0, 1.0],
          [2, 2],
          DType.float64,
        );

        final valid = convolve2d(img, k, mode: ConvMode.valid);
        expect(valid.shape, equals([2, 2]));
        // (1*1 + 5*1) = 6, (2*1 + 6*1) = 8
        expect(valid.getCell([0, 0]), closeTo(6.0, 1e-9));
        expect(valid.getCell([0, 1]), closeTo(8.0, 1e-9));
        // (4*1 + 8*1) = 12, (5*1 + 9*1) = 14
        expect(valid.getCell([1, 0]), closeTo(12.0, 1e-9));
        expect(valid.getCell([1, 1]), closeTo(14.0, 1e-9));
      });
    });

    test('1D correlate (Int32)', () {
      NDArray.scope(() {
        final a = NDArray<Int32>.fromList([1, 2, 3, 4], [4], DType.int32);
        final v = NDArray<Int32>.fromList([1, 2], [2], DType.int32);

        final valid = correlate(a, v, mode: ConvMode.valid);
        expect(valid.dtype, DType.int32);
        expect(valid.shape, equals([3]));
        // 1*1 + 2*2 = 5; 2*1 + 3*2 = 8; 3*1 + 4*2 = 11
        expect(valid.getCell([0]), 5);
        expect(valid.getCell([1]), 8);
        expect(valid.getCell([2]), 11);
      });
    });

    test('Complex correlation', () {
      NDArray.scope(() {
        final a = NDArray<Complex128>.fromList(
          [Complex(1.0, 2.0), Complex(3.0, 4.0)],
          [2],
          DType.complex128,
        );
        final v = NDArray<Complex128>.fromList(
          [Complex(1.0, 1.0), Complex(2.0, 2.0)],
          [2],
          DType.complex128,
        );

        final res = correlate<Complex128>(a, v, mode: ConvMode.valid);
        expect(res.dtype, DType.complex128);
        expect(res.shape, equals([1]));
        final val = res.getCell([0]);
        expect(val.real, closeTo(17.0, 1e-9));
        expect(val.imag, closeTo(3.0, 1e-9));
      });
    });

    test('Out argument & error handling', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );
        final v = NDArray<Float64>.fromList([1.0, 1.0], [2], DType.float64);
        final out = NDArray<Float64>.zeros([2], DType.float64);

        final res = correlate(a, v, mode: ConvMode.valid, out: out);
        expect(identical(res, out), true);
        expect(res.getCell([0]), closeTo(3.0, 1e-9));
        expect(res.getCell([1]), closeTo(5.0, 1e-9));
      });
    });
    test("1D correlate valid, full, same (Float32)", () {
      NDArray.scope(() {
        final a = NDArray<Float32>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float32,
        );
        final v = NDArray<Float32>.fromList(
          [1.0, 10.0, 100.0],
          [3],
          DType.float32,
        );

        final valid = correlate(a, v, mode: ConvMode.valid);
        expect(valid.shape, equals([3]));
        expect(valid.getCell([0]), closeTo(321.0, 1e-4));

        final full = correlate(a, v, mode: ConvMode.full);
        expect(full.shape, equals([7]));
        expect(full.getCell([0]), closeTo(100.0, 1e-4));
        expect(full.getCell([1]), closeTo(210.0, 1e-4));
        expect(full.getCell([2]), closeTo(321.0, 1e-4));

        final same = correlate(a, v, mode: ConvMode.same);
        expect(same.shape, equals([5]));
        expect(same.getCell([0]), closeTo(210.0, 1e-4));
      });
    });

    test("1D convolve valid, full, same (Float32)", () {
      NDArray.scope(() {
        final a = NDArray<Float32>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float32,
        );
        final v = NDArray<Float32>.fromList(
          [1.0, 10.0, 100.0],
          [3],
          DType.float32,
        );

        final full = convolve(a, v, mode: ConvMode.full);
        expect(full.shape, equals([7]));
        expect(full.getCell([0]), closeTo(1.0, 1e-4));
        expect(full.getCell([1]), closeTo(12.0, 1e-4));
        expect(full.getCell([2]), closeTo(123.0, 1e-4));
      });
    });
  });
}
