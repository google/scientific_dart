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

import 'dart:isolate';
import 'dart:math' as math;
import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void _divisionWorker(SendPort sendPort) {
  NDArray.scope(() {
    try {
      final a = NDArray.scalar(1, dtype: DType.int64);
      final b = NDArray.scalar(0, dtype: DType.int64);
      floorDivide(a, b); // Use floorDivide to trigger integer division by zero
      sendPort.send('ERROR: Did not throw');
    } catch (e) {
      if (e is UnsupportedError &&
          e.toString().contains('Integer division by zero')) {
        sendPort.send('OK');
      } else {
        sendPort.send('ERROR: Threw wrong exception: $e');
      }
    }
  });
}

void _normalWorker(SendPort sendPort) {
  NDArray.scope(() {
    try {
      final a = NDArray.scalar(4, dtype: DType.int64);
      final b = NDArray.scalar(2, dtype: DType.int64);
      final res = floorDivide(a, b);
      if (res.scalar == 2) {
        sendPort.send('OK');
      } else {
        sendPort.send('ERROR: Wrong result: ${res.scalar}');
      }
    } catch (e) {
      sendPort.send('ERROR: Threw exception: $e');
    }
  });
}

const bool _isWasm = bool.fromEnvironment('dart.tool.dart2wasm');

void main() {
  group('Regression Tests for Native Bugs', () {
    test(
      'Thread-safe division error tracking in concurrent isolates',
      () async {
        const numWorkers = 20;
        final receivePorts = List.generate(numWorkers, (_) => ReceivePort());
        final futures = <Future<dynamic>>[];

        for (var i = 0; i < numWorkers; i++) {
          futures.add(receivePorts[i].first);
          if (i.isEven) {
            await Isolate.spawn(_divisionWorker, receivePorts[i].sendPort);
          } else {
            await Isolate.spawn(_normalWorker, receivePorts[i].sendPort);
          }
        }

        final results = await Future.wait(futures);
        for (var i = 0; i < numWorkers; i++) {
          expect(results[i], 'OK', reason: 'Worker $i failed');
          receivePorts[i].close();
        }
      },
      skip: _isWasm ? 'Isolate.spawn is not supported on Wasm' : false,
    );

    test('Complex NaN sorting strict weak ordering', () {
      NDArray.scope(() {
        final nan = double.nan;
        final data = [
          Complex(nan, 1.0),
          Complex(2.0, nan),
          Complex(nan, nan),
          Complex(1.0, 2.0),
          Complex(3.0, 4.0),
          Complex(1.0, 1.0),
        ];
        final a = NDArray<Complex128>.fromList(
          data.map((e) => Complex(e.real, e.imag)).toList(),
          [6],
          DType.complex128,
        );

        // This should not crash or hang.
        final sorted = sort(a);

        expect(sorted.shape, [6]);
        // Lexicographical sort order:
        // 1. (1.0, 1.0)
        // 2. (1.0, 2.0)
        // 3. (2.0, nan)  -- 2.0 real, nan imag goes to end of 2.0s
        // 4. (3.0, 4.0)
        // 5. (nan, 1.0)  -- nan real goes to end
        // 6. (nan, nan)  -- nan real, nan imag goes to very end

        final expected = [
          Complex(1.0, 1.0),
          Complex(1.0, 2.0),
          Complex(2.0, nan),
          Complex(3.0, 4.0),
          Complex(nan, 1.0),
          Complex(nan, nan),
        ];

        final actual = sorted.toList();
        for (var i = 0; i < 6; i++) {
          final act = actual[i];
          final exp = expected[i];
          if (exp.real.isNaN) {
            expect(
              act.real.isNaN,
              isTrue,
              reason: 'Element $i real should be NaN',
            );
          } else {
            expect(act.real, exp.real, reason: 'Element $i real mismatch');
          }
          if (exp.imag.isNaN) {
            expect(
              act.imag.isNaN,
              isTrue,
              reason: 'Element $i imag should be NaN',
            );
          } else {
            expect(act.imag, exp.imag, reason: 'Element $i imag mismatch');
          }
        }
      });
    });

    test('Stable NaN segregation for double (stable sort)', () {
      NDArray.scope(() {
        final nan = double.nan;
        final a = NDArray<Float64>.fromList(
          [nan, 0.0, -0.0],
          [3],
          DType.float64,
        );

        final sorted = sort(a, kind: SortKind.mergesort);

        expect(sorted.toList()[0], 0.0);
        expect(sorted.toList()[1], -0.0);
        expect(sorted.toList()[2].isNaN, isTrue);

        expect(1.0 / (sorted.toList()[0]), double.infinity);
        expect(1.0 / (sorted.toList()[1]), double.negativeInfinity);
      });
    });

    test('Stable NaN segregation for float (stable sort)', () {
      NDArray.scope(() {
        final nan = double.nan;
        final a = NDArray<Float32>.fromList(
          [nan, 0.0, -0.0],
          [3],
          DType.float32,
        );

        final sorted = sort(a, kind: SortKind.mergesort);

        expect(sorted.toList()[0], 0.0);
        expect(sorted.toList()[1], -0.0);
        expect(sorted.toList()[2].isNaN, isTrue);

        expect(1.0 / (sorted.toList()[0]), double.infinity);
        expect(1.0 / (sorted.toList()[1]), double.negativeInfinity);
      });
    });
  });

  group('Phase 3 Type Safety Tests', () {
    test('floorDivide with uint8/int16', () {
      final a = NDArray<Uint8>.fromList([4, 5, 6], [3], DType.uint8);
      final b = NDArray<Int16>.fromList([2, 2, 2], [3], DType.int16);

      expect(() => floorDivide<DTypeTag>(a, b), throwsArgumentError);
      final aCast = castNDArray(a, DType.int16);
      final c = floorDivide(aCast, b);
      expect(c.toList(), [2, 2, 3]);
      expect(c.dtype, DType.int16);
      aCast.dispose();
    });

    test('floorDivide with uint8/float64', () {
      final a = NDArray<Uint8>.fromList([5, 6, 7], [3], DType.uint8);
      final b = NDArray<Float64>.fromList([2.0, 2.0, 2.0], [3], DType.float64);

      expect(() => floorDivide<DTypeTag>(a, b), throwsArgumentError);
      final aCast = castNDArray(a, DType.float64);
      final c = floorDivide(aCast, b);
      expect(c.toList(), [2.0, 3.0, 3.0]);
      expect(c.dtype, DType.float64);
      aCast.dispose();
    });

    test('remainder with uint8/int16', () {
      final a = NDArray<Uint8>.fromList([5, 6, 7], [3], DType.uint8);
      final b = NDArray<Int16>.fromList([3, 3, 3], [3], DType.int16);

      expect(() => remainder<DTypeTag>(a, b), throwsArgumentError);
      final aCast = castNDArray(a, DType.int16);
      final c = remainder(aCast, b);
      expect(c.toList(), [2, 0, 1]);
      expect(c.dtype, DType.int16);
      aCast.dispose();
    });

    test('remainder with uint8/float64', () {
      final a = NDArray<Uint8>.fromList([5, 6, 7], [3], DType.uint8);
      final b = NDArray<Float64>.fromList([3.0, 3.0, 3.0], [3], DType.float64);

      expect(() => remainder<DTypeTag>(a, b), throwsArgumentError);
      final aCast = castNDArray(a, DType.float64);
      final c = remainder(aCast, b);
      expect(c.toList(), [2.0, 0.0, 1.0]);
      expect(c.dtype, DType.float64);
      aCast.dispose();
    });

    test('sin with uint8/int16', () {
      final a = NDArray<Uint8>.fromList([0, 30, 90], [3], DType.uint8);
      // This should not crash
      final c = sin(a);
      expect(c.dtype, DType.float64); // default float type for sin on int
    });

    test('abs with uint8/int16', () {
      final a = NDArray<Int16>.fromList([-1, -2, 3], [3], DType.int16);
      final c = abs(a);
      expect(c.toList(), [1, 2, 3]);
      expect(c.dtype, DType.int16);
    });

    test('negative with uint8/int16', () {
      final a = NDArray<Int16>.fromList([1, 2, 3], [3], DType.int16);
      final b = negative(a);
      expect(b.toList(), [-1, -2, -3]);
      expect(b.dtype, DType.int16);

      final c = NDArray<Uint8>.fromList([1, 2, 3], [3], DType.uint8);
      final d = negative(c);
      expect(d.toList(), [255, 254, 253]); // wrap around for uint8
      expect(d.dtype, DType.uint8);
    });

    test('det with float32 preserves type', () {
      final a = NDArray<Float32>.fromList(
        [1.0, 2.0, 3.0, 4.0],
        [2, 2],
        DType.float32,
      );
      final d = det(a);
      expect(d.dtype, DType.float32);
      expect(d.scalar, closeTo(-2.0, 1e-5));
    });

    test('svd and qr throw ArgumentError for integer inputs', () {
      final a = NDArray<Int32>.fromList([1, 2, 3, 4], [2, 2], DType.int32);
      expect(() => svd(a), throwsArgumentError);
      expect(() => qr(a), throwsArgumentError);
    });

    test('complex SVD (complex128)', () {
      final a = NDArray<Complex128>.fromList(
        [
          Complex(2.0, 1.0),
          Complex(0.0, 0.0),
          Complex(0.0, 0.0),
          Complex(3.0, -1.0),
        ],
        [2, 2],
        DType.complex128,
      );

      final res = svd(a);
      expect(res.s.dtype, DType.float64);
      expect(res.s.toList()[0], closeTo(math.sqrt(10), 1e-5));
      expect(res.s.toList()[1], closeTo(math.sqrt(5), 1e-5));
    });

    test('complex pinv (complex128)', () {
      final a = NDArray<Complex128>.fromList(
        [
          Complex(2.0, 1.0),
          Complex(0.0, 0.0),
          Complex(0.0, 0.0),
          Complex(3.0, -1.0),
        ],
        [2, 2],
        DType.complex128,
      );

      final invA = pinv(a);
      expect(invA.dtype, DType.complex128);
      expect((invA.getCell([0, 0])).real, closeTo(0.4, 1e-5));
      expect((invA.getCell([0, 0])).imag, closeTo(-0.2, 1e-5));
      expect((invA.getCell([1, 1])).real, closeTo(0.3, 1e-5));
      expect((invA.getCell([1, 1])).imag, closeTo(0.1, 1e-5));
    });
  });

  group('Empty Sort Bug Repro', () {
    test('sort on empty 1D array', () {
      NDArray.scope(() {
        final a = NDArray.create([0], DType.float64);
        final sorted = sort(a);
        expect(sorted.shape, [0]);
        expect(sorted.toList(), []);
        final indices = argsort(a);
        expect(indices.shape, [0]);
        expect(indices.toList(), []);
      });
    });
  });
}
