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
import 'package:ndarray/src/ndarray_bindings.dart';
import 'dart:typed_data';

void main() {
  group('Stream 1 Cycle 11 Remediation Tests', () {
    test(
      'detachToParentScope() outside active NDArray scope throws StateError',
      () {
        final arr = NDArray.fromList([1, 2, 3], [3], DType.int32);
        expect(
          () => arr.detachToParentScope(),
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              contains(
                'detachToParentScope() is only valid inside an active NDArray scope',
              ),
            ),
          ),
        );
      },
    );

    test('detachToParentScope() inside active NDArray scope succeeds', () {
      NDArray<AnySpec>? result;
      NDArray.scope(() {
        final arr = NDArray.fromList([10, 20, 30], [3], DType.int32);
        result = arr.detachToParentScope();
      });
      expect(result!.isDisposed, isFalse);
      expect(result!.data[1], equals(20));
      result!.dispose();
    });

    test('NDArray.fill() strided dispatch for all DTypes', () {
      void checkStridedFill<T extends DTypeTag>(
        DType<T> dtype,
        dynamic fillVal,
        dynamic expectedVal,
      ) {
        final shape = [6];
        final arr = NDArray<T>.zeros(shape, dtype);
        final slice = arr.slice([
          const Slice(start: 0, stop: 6, step: 2),
        ]); // indices 0, 2, 4
        expect(slice.isContiguous, isFalse);

        slice.fill(fillVal);

        final list = arr.toList();
        expect(list[0], equals(expectedVal));
        expect(
          list[1],
          equals(
            dtype == DType.boolean
                ? false
                : (dtype == DType.complex128 || dtype == DType.complex64
                      ? Complex(0, 0)
                      : 0),
          ),
        );
        expect(list[2], equals(expectedVal));
        expect(
          list[3],
          equals(
            dtype == DType.boolean
                ? false
                : (dtype == DType.complex128 || dtype == DType.complex64
                      ? Complex(0, 0)
                      : 0),
          ),
        );
        expect(list[4], equals(expectedVal));
        expect(
          list[5],
          equals(
            dtype == DType.boolean
                ? false
                : (dtype == DType.complex128 || dtype == DType.complex64
                      ? Complex(0, 0)
                      : 0),
          ),
        );
      }

      checkStridedFill(DType.float64, 5.5, 5.5);
      checkStridedFill(DType.float32, 3.25, 3.25);
      checkStridedFill(DType.int64, -42, -42);
      checkStridedFill(DType.int32, 100, 100);
      checkStridedFill(DType.int16, 50, 50);
      checkStridedFill(DType.uint8, 255, 255);
      checkStridedFill(
        DType.complex128,
        Complex(1.5, -2.5),
        Complex(1.5, -2.5),
      );
      checkStridedFill(DType.complex64, Complex(1.5, -2.5), Complex(1.5, -2.5));
      checkStridedFill(DType.boolean, true, true);
    });

    test('Complex and Float32 contiguous ufuncs respect mask pointer', () {
      final aFloat = NDArray.fromList(Float32List.fromList([1.0, 2.0, 3.0]), [
        3,
      ], DType.float32);
      final bFloat = NDArray.fromList(
        Float32List.fromList([10.0, 20.0, 30.0]),
        [3],
        DType.float32,
      );
      final resFloat = NDArray.zeros([3], DType.float32);
      final mask = NDArray.fromList([true, false, true], [3], DType.boolean);

      v_add_float(
        aFloat.pointer.cast(),
        bFloat.pointer.cast(),
        resFloat.pointer.cast(),
        3,
        mask.pointer.cast(),
      );

      final listFloat = resFloat.toList();
      expect(listFloat[0], closeTo(11.0, 1e-5));
      expect(listFloat[1], equals(0.0)); // masked out in output
      expect(listFloat[2], closeTo(33.0, 1e-5));

      final aCpx = NDArray.fromList(
        [Complex(1, 2), Complex(3, 4)],
        [2],
        DType.complex128,
      );
      final bCpx = NDArray.fromList(
        [Complex(5, 6), Complex(7, 8)],
        [2],
        DType.complex128,
      );
      final resCpx = NDArray.zeros([2], DType.complex128);
      final maskCpx = NDArray.fromList([false, true], [2], DType.boolean);

      v_add_complex(
        aCpx.pointer.cast(),
        bCpx.pointer.cast(),
        resCpx.pointer.cast(),
        2,
        maskCpx.pointer.cast(),
      );

      final listCpx = resCpx.toList();
      expect(listCpx[0], equals(Complex(0, 0)));
      expect(listCpx[1], equals(Complex(10, 12)));
    });
  });
}
