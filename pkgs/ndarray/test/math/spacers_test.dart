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
  group('Spacers Tests', () {
    group('linspace', () {
      test(
        'standard linspace',
        () => NDArray.scope(() {
          final a = linspace(0.0, 1.0, 5, dtype: DType.float64);
          expect(a.shape, [5]);
          for (var i = 0; i < 5; i++) {
            expect(a.data[i], closeTo(i * 0.25, 1e-10));
          }
        }),
      );

      test(
        'linspace without endpoint',
        () => NDArray.scope(() {
          final a = linspace(
            0.0,
            1.0,
            5,
            endpoint: false,
            dtype: DType.float64,
          );
          expect(a.shape, [5]);
          for (var i = 0; i < 5; i++) {
            expect(a.data[i], closeTo(i * 0.2, 1e-10));
          }
        }),
      );

      test(
        'integer linspace',
        () => NDArray.scope(() {
          final a = linspace(0, 10, 5, dtype: DType.int64);
          expect(a.dtype, DType.int64);
          expect(a.data, [
            0,
            2,
            5,
            7,
            10,
          ]); // interpolated: 0, 2.5, 5, 7.5, 10 -> truncated to int
        }),
      );
    });

    group('logspace', () {
      test(
        'standard logspace (base 10)',
        () => NDArray.scope(() {
          final a = logspace(
            0.0,
            2.0,
            3,
            dtype: DType.float64,
          ); // 10^0, 10^1, 10^2
          expect(a.shape, [3]);
          expect(a.data[0], closeTo(1.0, 1e-10));
          expect(a.data[1], closeTo(10.0, 1e-10));
          expect(a.data[2], closeTo(100.0, 1e-10));
        }),
      );

      test(
        'logspace base 2',
        () => NDArray.scope(() {
          final a = logspace(
            0.0,
            4.0,
            5,
            base: 2.0,
            dtype: DType.float64,
          ); // 2^0, 2^1, 2^2, 2^3, 2^4
          expect(a.data, [1.0, 2.0, 4.0, 8.0, 16.0]);
        }),
      );

      test(
        'logspace without endpoint',
        () => NDArray.scope(() {
          final a = logspace(
            0.0,
            3.0,
            3,
            endpoint: false,
            dtype: DType.float64,
          ); // 10^0, 10^1, 10^2
          expect(a.data[0], closeTo(1.0, 1e-10));
          expect(a.data[1], closeTo(10.0, 1e-10));
          expect(a.data[2], closeTo(100.0, 1e-10));
        }),
      );
    });

    group('geomspace', () {
      test(
        'standard geomspace',
        () => NDArray.scope(() {
          final a = geomspace(1.0, 1000.0, 4, dtype: DType.float64);
          expect(a.data[0], closeTo(1.0, 1e-10));
          expect(a.data[1], closeTo(10.0, 1e-10));
          expect(a.data[2], closeTo(100.0, 1e-10));
          expect(a.data[3], closeTo(1000.0, 1e-10));
        }),
      );

      test(
        'negative geomspace',
        () => NDArray.scope(() {
          final a = geomspace(-1.0, -1000.0, 4, dtype: DType.float64);
          expect(a.data[0], closeTo(-1.0, 1e-10));
          expect(a.data[1], closeTo(-10.0, 1e-10));
          expect(a.data[2], closeTo(-100.0, 1e-10));
          expect(a.data[3], closeTo(-1000.0, 1e-10));
        }),
      );

      test(
        'geomspace without endpoint',
        () => NDArray.scope(() {
          final a = geomspace(
            1.0,
            1000.0,
            3,
            endpoint: false,
            dtype: DType.float64,
          );
          expect(a.data[0], closeTo(1.0, 1e-10));
          expect(a.data[1], closeTo(10.0, 1e-10));
          expect(a.data[2], closeTo(100.0, 1e-10));
        }),
      );
    });

    group('Complex spacers', () {
      test(
        'linspace complex',
        () => NDArray.scope(() {
          final a = linspace(
            Complex(0, 0),
            Complex(1, 1),
            3,
            dtype: DType.complex128,
          );
          expect(a.dtype, DType.complex128);
          expect(a.data[0], Complex(0, 0));
          expect(a.data[1], Complex(0.5, 0.5));
          expect(a.data[2], Complex(1, 1));
        }),
      );

      test(
        'logspace complex',
        () => NDArray.scope(() {
          final a = logspace(
            Complex(0, 0),
            Complex(0, 2),
            3,
            base: 10.0,
            dtype: DType.complex128,
          );
          expect(a.data[0], Complex(1, 0));
          final expected1 = Complex(
            math.cos(math.log(10)),
            math.sin(math.log(10)),
          );
          final val1 = a.data[1];
          expect(val1.real, closeTo(expected1.real, 1e-10));
          expect(val1.imag, closeTo(expected1.imag, 1e-10));
        }),
      );

      test(
        'geomspace complex',
        () => NDArray.scope(() {
          final a = geomspace(
            Complex(1, 0),
            Complex(-1, 0),
            3,
            dtype: DType.complex128,
          );
          expect(a.data[0], Complex(1, 0));
          final val1 = a.data[1];
          expect(val1.real, closeTo(0, 1e-10));
          expect(val1.imag, closeTo(1, 1e-10));
          final val2 = a.data[2];
          expect(val2.real, closeTo(-1, 1e-10));
          expect(val2.imag, closeTo(0, 1e-10));
        }),
      );
    });

    group('Broadcasting spacers', () {
      test(
        'linspace broadcasting',
        () => NDArray.scope(() {
          final start = NDArray.fromList([0.0, 10.0], [2], DType.float64);
          final stop = NDArray.fromList([1.0, 11.0], [2], DType.float64);
          // axis=0 (default) -> [numSamples, 2]
          final a = linspaceGrid(start, stop, 3);
          expect(a.shape, [3, 2]);
          expect(a.toList(), [0.0, 10.0, 0.5, 10.5, 1.0, 11.0]);
        }),
      );

      test(
        'linspace axis support',
        () => NDArray.scope(() {
          final start = NDArray.fromList([0.0, 10.0], [2], DType.float64);
          final stop = NDArray.fromList([1.0, 11.0], [2], DType.float64);
          // axis=1 -> [2, numSamples]
          final a = linspaceGrid(start, stop, 3, axis: 1);
          expect(a.shape, [2, 3]);
          expect(a.toList(), [0.0, 0.5, 1.0, 10.0, 10.5, 11.0]);
        }),
      );

      test(
        'linspaceGridWithStep',
        () => NDArray.scope(() {
          final start = NDArray.fromList([0.0, 10.0], [2], DType.float64);
          final stop = NDArray.fromList([1.0, 12.0], [2], DType.float64);
          final (samples: a, step: step) = linspaceGridWithStep(start, stop, 3);
          expect(a.shape, [3, 2]);
          expect(step.shape, [2]);
          expect(step.toList(), [0.5, 1.0]);
        }),
      );
    });

    group('Step retrieval', () {
      test(
        'linspaceWithStep',
        () => NDArray.scope(() {
          final (samples: a, step: step) = linspaceWithStep(
            0.0,
            10.0,
            5,
            dtype: DType.float64,
          );
          expect(step, 2.5);
          expect(a.toList(), [0.0, 2.5, 5.0, 7.5, 10.0]);
        }),
      );
    });

    group('Exhaustive DType Grid Coverage', () {
      void testDTypeGrid(
        DType<AnySpec> dtype,
        List<Object?> startList,
        List<Object?> stopList,
        List<Object?> expectedGridList,
        List<Object?> expectedStepList,
      ) {
        test(
          'linspaceGrid and linspaceGridWithStep for ${dtype.name}',
          () => NDArray.scope(() {
            final start = NDArray.fromList(startList, [2], dtype);
            final stop = NDArray.fromList(stopList, [2], dtype);

            final grid = linspaceGrid(start, stop, 3, dtype: dtype);
            expect(grid.dtype, dtype);
            expect(grid.shape, [3, 2]);
            expect(grid.toList(), expectedGridList);

            final (samples: grid2, step: step) = linspaceGridWithStep(
              start,
              stop,
              3,
              dtype: dtype,
            );
            expect(grid2.dtype, dtype);
            expect(grid2.shape, [3, 2]);
            expect(grid2.toList(), expectedGridList);
            expect(step.dtype, dtype);
            expect(step.shape, [2]);
            expect(step.toList(), expectedStepList);
          }),
        );
      }

      // Float64
      testDTypeGrid(
        DType.float64,
        [0.0, 10.0],
        [10.0, 30.0],
        [0.0, 10.0, 5.0, 20.0, 10.0, 30.0],
        [5.0, 10.0],
      );
      // Float32
      testDTypeGrid(
        DType.float32,
        [0.0, 10.0],
        [10.0, 30.0],
        [0.0, 10.0, 5.0, 20.0, 10.0, 30.0],
        [5.0, 10.0],
      );
      // Complex128
      testDTypeGrid(
        DType.complex128,
        [Complex(0, 0), Complex(10, 10)],
        [Complex(10, 10), Complex(30, 30)],
        [
          Complex(0, 0),
          Complex(10, 10),
          Complex(5, 5),
          Complex(20, 20),
          Complex(10, 10),
          Complex(30, 30),
        ],
        [Complex(5, 5), Complex(10, 10)],
      );
      // Complex64
      testDTypeGrid(
        DType.complex64,
        [Complex(0, 0), Complex(10, 10)],
        [Complex(10, 10), Complex(30, 30)],
        [
          Complex(0, 0),
          Complex(10, 10),
          Complex(5, 5),
          Complex(20, 20),
          Complex(10, 10),
          Complex(30, 30),
        ],
        [Complex(5, 5), Complex(10, 10)],
      );
      // Int64
      testDTypeGrid(
        DType.int64,
        [0, 10],
        [10, 30],
        [0, 10, 5, 20, 10, 30],
        [5, 10],
      );
      // Int32
      testDTypeGrid(
        DType.int32,
        [0, 10],
        [10, 30],
        [0, 10, 5, 20, 10, 30],
        [5, 10],
      );
      // Int16
      testDTypeGrid(
        DType.int16,
        [0, 10],
        [10, 30],
        [0, 10, 5, 20, 10, 30],
        [5, 10],
      );
      // Uint8
      testDTypeGrid(
        DType.uint8,
        [0, 10],
        [10, 30],
        [0, 10, 5, 20, 10, 30],
        [5, 10],
      );
    });

    group('Unsupported DType', () {
      test(
        'linspaceGrid with DType.boolean throws UnsupportedError',
        () => NDArray.scope(() {
          final start = NDArray.fromList([true], [1], DType.boolean);
          final stop = NDArray.fromList([false], [1], DType.boolean);
          expect(() => linspaceGrid(start, stop, 3), throwsUnsupportedError);
        }),
      );

      test(
        'linspaceGridWithStep with DType.boolean throws UnsupportedError',
        () => NDArray.scope(() {
          final start = NDArray.fromList([true], [1], DType.boolean);
          final stop = NDArray.fromList([false], [1], DType.boolean);
          expect(
            () => linspaceGridWithStep(start, stop, 3),
            throwsUnsupportedError,
          );
        }),
      );

      test(
        'logspace with integer/boolean dtype throws UnsupportedError',
        () => NDArray.scope(() {
          expect(
            () => logspace(0, 10, 5, dtype: DType.int64),
            throwsUnsupportedError,
          );
          expect(
            () => logspace(true, false, 5, dtype: DType.boolean),
            throwsUnsupportedError,
          );
        }),
      );

      test(
        'geomspace with integer/boolean dtype throws UnsupportedError',
        () => NDArray.scope(() {
          expect(
            () => geomspace(1, 100, 5, dtype: DType.int64),
            throwsUnsupportedError,
          );
          expect(
            () => geomspace(true, false, 5, dtype: DType.boolean),
            throwsUnsupportedError,
          );
        }),
      );

      test(
        'geomspaceGrid with integer/boolean dtype throws UnsupportedError',
        () => NDArray.scope(() {
          final start = NDArray.fromList([1], [1], DType.int32);
          final stop = NDArray.fromList([100], [1], DType.int32);
          expect(() => geomspaceGrid(start, stop, 5), throwsUnsupportedError);
        }),
      );
    });

    group('New Grid Generators', () {
      test(
        'logspaceGrid with 2D shape [2, 2]',
        () => NDArray.scope(() {
          final start = NDArray.fromList(
            [0.0, 1.0, 2.0, 3.0],
            [2, 2],
            DType.float64,
          );
          final stop = NDArray.fromList(
            [2.0, 3.0, 4.0, 5.0],
            [2, 2],
            DType.float64,
          );

          final grid = logspaceGrid(start, stop, 3);
          expect(grid.shape, [3, 2, 2]);
          expect(grid.dtype, DType.float64);

          expect(grid.toList(), [
            // Sample 0 (10^start)
            1.0, 10.0,
            100.0, 1000.0,
            // Sample 1 (10^mid)
            10.0, 100.0,
            1000.0, 10000.0,
            // Sample 2 (10^stop)
            100.0, 1000.0,
            10000.0, 100000.0,
          ]);
        }),
      );

      test(
        'logspaceGrid with custom base of shape [2]',
        () => NDArray.scope(() {
          final start = NDArray.fromList([0.0, 1.0], [2], DType.float64);
          final stop = NDArray.fromList([2.0, 3.0], [2], DType.float64);
          final base = NDArray.fromList([2.0, 10.0], [2], DType.float64);

          final grid = logspaceGrid(start, stop, 3, base: base);
          expect(grid.shape, [3, 2]);
          expect(grid.dtype, DType.float64);

          // Expected:
          // grid[0] = [2^0, 10^1] = [1.0, 10.0]
          // grid[1] = [2^1, 10^2] = [2.0, 100.0]
          // grid[2] = [2^2, 10^3] = [4.0, 1000.0]
          expect(grid.toList(), [1.0, 10.0, 2.0, 100.0, 4.0, 1000.0]);
        }),
      );

      test(
        'geomspaceGrid with 2D shape [2, 2]',
        () => NDArray.scope(() {
          final start = NDArray.fromList(
            [1.0, 10.0, 100.0, 1000.0],
            [2, 2],
            DType.float64,
          );
          final stop = NDArray.fromList(
            [100.0, 1000.0, 10000.0, 100000.0],
            [2, 2],
            DType.float64,
          );

          final grid = geomspaceGrid(start, stop, 3);
          expect(grid.shape, [3, 2, 2]);
          expect(grid.dtype, DType.float64);

          // Expected:
          // grid[0] = start
          // grid[1] = sqrt(start * stop)
          // grid[2] = stop
          // Using closeTo for double comparisons
          final list = grid.toList();
          final expected = [
            1.0,
            10.0,
            100.0,
            1000.0,

            10.0,
            100.0,
            1000.0,
            10000.0,

            100.0,
            1000.0,
            10000.0,
            100000.0,
          ];
          expect(list.length, expected.length);
          for (var i = 0; i < list.length; i++) {
            expect(list[i], closeTo(expected[i], 1e-9));
          }
        }),
      );
    });

    group('Precondition & State Validation', () {
      test(
        'Disposed input arrays throw StateError',
        () => NDArray.scope(() {
          final start = NDArray.fromList([0.0], [1], DType.float64);
          final stop = NDArray.fromList([10.0], [1], DType.float64);

          // Dispose start
          start.dispose();
          expect(() => linspaceGrid(start, stop, 3), throwsStateError);
          expect(() => linspaceGridWithStep(start, stop, 3), throwsStateError);
          expect(() => logspaceGrid(start, stop, 3), throwsStateError);
          expect(() => geomspaceGrid(start, stop, 3), throwsStateError);

          // Re-create start for base dispose test
          final start2 = NDArray.fromList([0.0], [1], DType.float64);
          final base = NDArray.fromList([10.0], [1], DType.float64);
          base.dispose();
          expect(
            () => logspaceGrid(start2, stop, 3, base: base),
            throwsStateError,
          );
        }),
      );

      test(
        'linspaceGrid and linspaceGridWithStep with numSamples < 0 throw ArgumentError, numSamples == 0 is empty',
        () => NDArray.scope(() {
          final start = NDArray.fromList([0.0], [1], DType.float64);
          final stop = NDArray.fromList([10.0], [1], DType.float64);

          expect(() => linspaceGrid(start, stop, -5), throwsArgumentError);
          expect(
            () => linspaceGridWithStep(start, stop, -5),
            throwsArgumentError,
          );
          final g0 = linspaceGrid(start, stop, 0);
          expect(g0.shape, [0, 1]);
          final (samples: g0Step, step: step0) = linspaceGridWithStep(
            start,
            stop,
            0,
          );
          expect(g0Step.shape, [0, 1]);
          expect(step0.getCell([0]).isNaN, isTrue);
        }),
      );

      test(
        'linspaceGrid and linspaceGridWithStep with out-of-bounds axis throw ArgumentError',
        () => NDArray.scope(() {
          final start = NDArray.fromList([0.0], [1], DType.float64);
          final stop = NDArray.fromList([10.0], [1], DType.float64);

          // Valid axes are 0, 1, -1, -2 for rank 1 start/stop (result rank 2)
          expect(
            () => linspaceGrid(start, stop, 3, axis: 2),
            throwsArgumentError,
          );
          expect(
            () => linspaceGrid(start, stop, 3, axis: -3),
            throwsArgumentError,
          );
          expect(
            () => linspaceGridWithStep(start, stop, 3, axis: 2),
            throwsArgumentError,
          );
          expect(
            () => linspaceGridWithStep(start, stop, 3, axis: -3),
            throwsArgumentError,
          );
        }),
      );

      test(
        'geomspace with 0.0 or mismatched signs throws ArgumentError',
        () => NDArray.scope(() {
          expect(
            () => geomspace(0.0, 10.0, 5, dtype: DType.float64),
            throwsArgumentError,
          );
          expect(
            () => geomspace(1.0, 0.0, 5, dtype: DType.float64),
            throwsArgumentError,
          );
          expect(
            () => geomspace(-1.0, 10.0, 5, dtype: DType.float64),
            throwsArgumentError,
          );
          expect(
            () => geomspace(1.0, -10.0, 5, dtype: DType.float64),
            throwsArgumentError,
          );

          // Complex geomspace with 0.0 throws ArgumentError
          expect(
            () => geomspace(
              Complex(0, 0),
              Complex(1, 1),
              5,
              dtype: DType.complex128,
            ),
            throwsArgumentError,
          );
          expect(
            () => geomspace(
              Complex(1, 1),
              Complex(0, 0),
              5,
              dtype: DType.complex128,
            ),
            throwsArgumentError,
          );
        }),
      );

      test(
        'geomspaceGrid with 0.0 or mismatched signs throws ArgumentError',
        () => NDArray.scope(() {
          final startZero = NDArray.fromList([0.0, 1.0], [2], DType.float64);
          final stop = NDArray.fromList([10.0, 10.0], [2], DType.float64);
          expect(() => geomspaceGrid(startZero, stop, 5), throwsArgumentError);

          final startNeg = NDArray.fromList([-1.0, 2.0], [2], DType.float64);
          final stopPos = NDArray.fromList([10.0, 10.0], [2], DType.float64);
          expect(
            () => geomspaceGrid(startNeg, stopPos, 5),
            throwsArgumentError,
          );

          // Complex geomspaceGrid with 0.0 throws ArgumentError
          final startComplexZero = NDArray.fromList(
            [Complex(0, 0), Complex(1, 1)],
            [2],
            DType.complex128,
          );
          final stopComplex = NDArray.fromList(
            [Complex(10, 10), Complex(10, 10)],
            [2],
            DType.complex128,
          );
          expect(
            () => geomspaceGrid(startComplexZero, stopComplex, 5),
            throwsArgumentError,
          );
        }),
      );
    });
  });
}
