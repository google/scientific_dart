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

import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/ndarray_bindings.dart' as bindings;
import 'package:test/test.dart';

void main() {
  group('Review Cycle 2 Native Fixes', () {
    test('sqrt(a, out: out) where out is a reversed strided view of a', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 4.0, 9.0, 16.0, 25.0],
          [5],
          DType.float64,
        );
        final out = a.slice([const Slice(step: -1)]);

        sqrt(a, out: out);

        // out[i] should be sqrt(original_a[i]): [1.0, 2.0, 3.0, 4.0, 5.0]
        expect(out.getCell([0]), closeTo(1.0, 1e-12));
        expect(out.getCell([1]), closeTo(2.0, 1e-12));
        expect(out.getCell([2]), closeTo(3.0, 1e-12));
        expect(out.getCell([3]), closeTo(4.0, 1e-12));
        expect(out.getCell([4]), closeTo(5.0, 1e-12));

        // Since out is a reversed view of a, a should now hold [5.0, 4.0, 3.0, 2.0, 1.0]
        expect(a.getCell([0]), closeTo(5.0, 1e-12));
        expect(a.getCell([1]), closeTo(4.0, 1e-12));
        expect(a.getCell([2]), closeTo(3.0, 1e-12));
        expect(a.getCell([3]), closeTo(2.0, 1e-12));
        expect(a.getCell([4]), closeTo(1.0, 1e-12));
      });
    });

    test('add(a, b, out: out) where out is a reversed strided view of a', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float64,
        );
        final b = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0, 50.0],
          [5],
          DType.float64,
        );
        final out = a.slice([const Slice(step: -1)]);

        add(a, b, out: out);

        // out[i] should be original_a[i] + b[i]: [11.0, 22.0, 33.0, 44.0, 55.0]
        expect(out.getCell([0]), closeTo(11.0, 1e-12));
        expect(out.getCell([1]), closeTo(22.0, 1e-12));
        expect(out.getCell([2]), closeTo(33.0, 1e-12));
        expect(out.getCell([3]), closeTo(44.0, 1e-12));
        expect(out.getCell([4]), closeTo(55.0, 1e-12));

        // Underlying array a should hold [55.0, 44.0, 33.0, 22.0, 11.0]
        expect(a.getCell([0]), closeTo(55.0, 1e-12));
        expect(a.getCell([1]), closeTo(44.0, 1e-12));
        expect(a.getCell([2]), closeTo(33.0, 1e-12));
        expect(a.getCell([3]), closeTo(22.0, 1e-12));
        expect(a.getCell([4]), closeTo(11.0, 1e-12));
      });
    });

    test('add(a, b, out: out) where out is a transposed 2D view of a', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final b = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );
        final out = a.transpose();

        add(a, b, out: out);

        // out[0, 0] = 1 + 10 = 11
        // out[0, 1] = 2 + 20 = 22
        // out[1, 0] = 3 + 30 = 33
        // out[1, 1] = 4 + 40 = 44
        expect(out.getCell([0, 0]), closeTo(11.0, 1e-12));
        expect(out.getCell([0, 1]), closeTo(22.0, 1e-12));
        expect(out.getCell([1, 0]), closeTo(33.0, 1e-12));
        expect(out.getCell([1, 1]), closeTo(44.0, 1e-12));
      });
    });

    test('0D scalar s_interp_double native binding works with rank == 0', () {
      using((arena) {
        final x = arena<ffi.Double>()..value = 2.5;
        final xp = arena<ffi.Double>(3);
        xp[0] = 1.0;
        xp[1] = 2.0;
        xp[2] = 3.0;
        final fp = arena<ffi.Double>(3);
        fp[0] = 10.0;
        fp[1] = 20.0;
        fp[2] = 30.0;
        final res = arena<ffi.Double>()..value = -999.0;

        bindings.s_interp_double(
          x,
          ffi.nullptr,
          xp,
          1,
          3,
          fp,
          1,
          res,
          ffi.nullptr,
          ffi.nullptr,
          0,
          ffi.nullptr,
          ffi.nullptr,
        );

        expect(res.value, closeTo(25.0, 1e-12));
      });
    });

    test('0D scalar s_polyval_double native binding works with rank == 0', () {
      using((arena) {
        // p(x) = 2*x^2 + 3*x + 5 -> coeffs in descending order: [2.0, 3.0, 5.0]
        final c = arena<ffi.Double>(3);
        c[0] = 2.0;
        c[1] = 3.0;
        c[2] = 5.0;
        final x = arena<ffi.Double>()..value = 4.0;
        final res = arena<ffi.Double>()..value = -999.0;

        bindings.s_polyval_double(
          c,
          1,
          3,
          x,
          ffi.nullptr,
          res,
          ffi.nullptr,
          ffi.nullptr,
          0,
        );

        // 2*(16) + 3*(4) + 5 = 32 + 12 + 5 = 49.0
        expect(res.value, closeTo(49.0, 1e-12));
      });
    });
  });
}
