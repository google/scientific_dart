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
  });
}
