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
import 'package:test/test.dart';
import 'dart:typed_data';

void main() {
  group('NDArray Status Checkers (isnan, isinf, isfinite) Tests', () {
    test(
      'isnan, isinf, isfinite basic double (Float64)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          Float64List.fromList([
            1.0,
            double.nan,
            double.infinity,
            double.negativeInfinity,
            5.0,
          ]),
          [5],
          DType.float64,
        );

        final nanMask = isnan(a);
        expect(nanMask.dtype, DType.boolean);
        expect(nanMask.toList(), [false, true, false, false, false]);

        final infMask = isinf(a);
        expect(infMask.dtype, DType.boolean);
        expect(infMask.toList(), [false, false, true, true, false]);

        final finiteMask = isfinite(a);
        expect(finiteMask.dtype, DType.boolean);
        expect(finiteMask.toList(), [true, false, false, false, true]);
      }),
    );

    test(
      'isnan, isinf, isfinite float32',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          Float32List.fromList([1.0, double.nan, double.infinity, 5.0]),
          [4],
          DType.float32,
        );

        final resNan = isnan(a);
        expect(resNan.toList(), [false, true, false, false]);

        final resInf = isinf(a);
        expect(resInf.toList(), [false, false, true, false]);

        final resFinite = isfinite(a);
        expect(resFinite.toList(), [true, false, false, true]);
      }),
    );

    test(
      'isnan, isinf, isfinite integer arrays (Int32 and Int64)',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final b = NDArray.fromList([10, 20, 30], [3], DType.int64);

        // Integers are never NaN or Infinite, and always finite
        final nanA = isnan(a);
        expect(nanA.toList(), [false, false, false]);

        final infA = isinf(a);
        expect(infA.toList(), [false, false, false]);

        final finiteA = isfinite(a);
        expect(finiteA.toList(), [true, true, true]);

        final nanB = isnan(b);
        expect(nanB.toList(), [false, false, false]);

        final infB = isinf(b);
        expect(infB.toList(), [false, false, false]);

        final finiteB = isfinite(b);
        expect(finiteB.toList(), [true, true, true]);
      }),
    );

    test(
      'isnan, isinf, isfinite complex arrays (Complex128 and Complex64)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [
            Complex(1.0, 0.0),
            Complex(double.nan, 1.0),
            Complex(1.0, double.infinity),
          ],
          [3],
          DType.complex128,
        );

        final nanA = isnan(a);
        expect(nanA.toList(), [false, true, false]);

        final infA = isinf(a);
        expect(infA.toList(), [false, false, true]);

        final finiteA = isfinite(a);
        expect(finiteA.toList(), [true, false, false]);

        final b = NDArray.fromList(
          [
            Complex(1.0, 0.0),
            Complex(double.nan, 1.0),
            Complex(1.0, double.infinity),
          ],
          [3],
          DType.complex64,
        );

        final nanB = isnan(b);
        expect(nanB.toList(), [false, true, false]);

        final infB = isinf(b);
        expect(infB.toList(), [false, false, true]);

        final finiteB = isfinite(b);
        expect(finiteB.toList(), [true, false, false]);
      }),
    );

    test(
      'isnan, isinf, isfinite on transposed/sliced views',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          Float64List.fromList([1.0, double.nan, double.infinity, 4.0]),
          [2, 2],
          DType.float64,
        );

        final view = parent.transposed;

        final resNan = isnan(view);
        expect(resNan.toList(), [
          false,
          false,
          true,
          false,
        ]); // transposed order: [1.0, inf, nan, 4.0]

        final resInf = isinf(view);
        expect(resInf.toList(), [false, true, false, false]);

        final resFinite = isfinite(view);
        expect(resFinite.toList(), [true, false, false, true]);
      }),
    );

    test(
      'Disposed array checks throw StateError',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        a.dispose();

        expect(() => isnan(a), throwsStateError);
        expect(() => isinf(a), throwsStateError);
        expect(() => isfinite(a), throwsStateError);
      }),
    );

    test(
      'SIMD boundary sizes, misaligned views, subnormals, -0.0, and NaN payloads',
      () => NDArray.scope(() {
        const sizes = [
          0,
          1,
          3,
          4,
          7,
          8,
          15,
          16,
          17,
          31,
          32,
          33,
          63,
          64,
          65,
          129,
          257,
        ];
        for (final n in sizes) {
          final f64List = Float64List(n + 3);
          final f32List = Float32List(n + 3);
          final bd64 = f64List.buffer.asByteData();
          final bd32 = f32List.buffer.asByteData();
          for (var i = 0; i < n + 3; i++) {
            switch (i % 8) {
              case 0:
                f64List[i] = 0.0;
                f32List[i] = 0.0;
              case 1:
                f64List[i] = -0.0;
                f32List[i] = -0.0;
              case 2:
                f64List[i] = 5e-324; // subnormal double
                f32List[i] = 1.4e-45; // subnormal float
              case 3:
                f64List[i] = -5e-324;
                f32List[i] = -1.4e-45;
              case 4:
                f64List[i] = double.infinity;
                f32List[i] = double.infinity;
              case 5:
                f64List[i] = double.negativeInfinity;
                f32List[i] = double.negativeInfinity;
              case 6:
                // Quiet NaN with custom payload
                bd64.setUint64(i * 8, 0x7FF8000000123456, Endian.host);
                bd32.setUint32(i * 4, 0x7FC01234, Endian.host);
              default:
                // Negative NaN with payload
                bd64.setUint64(i * 8, 0xFFF8000000ABCDEF, Endian.host);
                bd32.setUint32(i * 4, 0xFFC0ABCD, Endian.host);
            }
          }
          final a64 = NDArray.fromList(f64List, [n + 3], DType.float64);
          final a32 = NDArray.fromList(f32List, [n + 3], DType.float32);
          for (var off = 0; off < 3; off++) {
            final v64 = a64.slice([Slice(start: off, stop: off + n)]);
            final v32 = a32.slice([Slice(start: off, stop: off + n)]);
            final expNan = List<bool>.generate(n, (i) => (off + i) % 8 >= 6);
            final expInf = List<bool>.generate(
              n,
              (i) => (off + i) % 8 == 4 || (off + i) % 8 == 5,
            );
            final expFin = List<bool>.generate(n, (i) => (off + i) % 8 < 4);
            expect(isnan(v64).toList(), expNan);
            expect(isinf(v64).toList(), expInf);
            expect(isfinite(v64).toList(), expFin);
            expect(isnan(v32).toList(), expNan);
            expect(isinf(v32).toList(), expInf);
            expect(isfinite(v32).toList(), expFin);
          }
        }
      }),
    );

    test(
      'broadcast-row unary ops match materialized contiguous arrays bit-for-bit',
      () => NDArray.scope(() {
        final col = NDArray.fromList(
          [
            -10.5,
            -1.23456789012345,
            -0.0,
            0.0,
            0.3333333333333333,
            2.718281828,
            42.125,
          ],
          [7, 1],
          DType.float64,
        );
        final bcast = broadcastTo(col, [7, 32]);
        final materialized = bcast.copy();
        expect(bcast.strides, [1, 0]);
        expect(materialized.isContiguous, isTrue);

        expect(exp(bcast).toList(), exp(materialized).toList());
        expect(sin(bcast).toList(), sin(materialized).toList());
        expect(cos(bcast).toList(), cos(materialized).toList());
        expect(tan(bcast).toList(), tan(materialized).toList());
        expect(isnan(bcast).toList(), isnan(materialized).toList());
        expect(isinf(bcast).toList(), isinf(materialized).toList());
        expect(isfinite(bcast).toList(), isfinite(materialized).toList());
      }),
    );
  });
}
