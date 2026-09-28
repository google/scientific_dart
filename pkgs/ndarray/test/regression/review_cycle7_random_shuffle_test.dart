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

void main() {
  group('Review Cycle 7: random & shuffle regressions', () {
    test(
      '1. multivariateNormal and choice return root arrays (!isView) when out == null and dispose properly',
      () {
        final mean = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
        final cov = NDArray<Float64>.fromList(
          [1.0, 0.0, 0.0, 1.0],
          [2, 2],
          DType.float64,
        );
        final a = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [4],
          DType.float64,
        );
        addTearDown(() {
          mean.dispose();
          cov.dispose();
          a.dispose();
        });

        final mvn0 = multivariateNormal(mean, cov, seed: 42);
        expect(mvn0.shape, equals([2]));
        expect(mvn0.isView, isFalse);
        expect(mvn0.isDisposed, isFalse);
        mvn0.dispose();
        expect(mvn0.isDisposed, isTrue);

        final mvn1 = multivariateNormal(mean, cov, size: [4], seed: 42);
        expect(mvn1.shape, equals([4, 2]));
        expect(mvn1.isView, isFalse);
        expect(mvn1.isDisposed, isFalse);
        mvn1.dispose();
        expect(mvn1.isDisposed, isTrue);

        final mvn2 = multivariateNormal(mean, cov, size: [2, 3], seed: 42);
        expect(mvn2.shape, equals([2, 3, 2]));
        expect(mvn2.isView, isFalse);
        expect(mvn2.isDisposed, isFalse);
        mvn2.dispose();
        expect(mvn2.isDisposed, isTrue);

        final c0 = choice(a, seed: 42);
        expect(c0.shape, equals(<int>[]));
        expect(c0.isView, isFalse);
        expect(c0.isDisposed, isFalse);
        c0.dispose();
        expect(c0.isDisposed, isTrue);

        final c1 = choice(a, size: [4], seed: 42);
        expect(c1.shape, equals([4]));
        expect(c1.isView, isFalse);
        expect(c1.isDisposed, isFalse);
        c1.dispose();
        expect(c1.isDisposed, isTrue);

        final c2 = choice(a, size: [2, 3], seed: 42);
        expect(c2.shape, equals([2, 3]));
        expect(c2.isView, isFalse);
        expect(c2.isDisposed, isFalse);
        c2.dispose();
        expect(c2.isDisposed, isTrue);
      },
    );

    test(
      '2. choice with out aliasing a or p does not corrupt inputs during sampling',
      () {
        NDArray.scope(() {
          final aOrig = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [6],
            DType.float64,
          );
          final aAlias = aOrig.copy();
          final expectedA = choice(
            aOrig,
            size: [6],
            replace: false,
            seed: 12345,
          );
          final actualA = choice(
            aAlias,
            size: [6],
            replace: false,
            seed: 12345,
            out: aAlias,
          );

          expect(identical(actualA, aAlias), isTrue);
          for (var i = 0; i < 6; i++) {
            expect(aAlias.getCellFlat(i), equals(expectedA.getCellFlat(i)));
          }

          // Also verify weighted choice where out aliases p
          final pOrig = NDArray<Float64>.fromList(
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.25],
            [6],
            DType.float64,
          );
          final pAlias = pOrig.copy();
          final expectedP = choice(
            aOrig,
            size: [6],
            replace: false,
            p: pOrig,
            seed: 67890,
          );
          final actualP = choice(
            aOrig,
            size: [6],
            replace: false,
            p: pAlias,
            seed: 67890,
            out: pAlias,
          );

          expect(identical(actualP, pAlias), isTrue);
          for (var i = 0; i < 6; i++) {
            expect(pAlias.getCellFlat(i), equals(expectedP.getCellFlat(i)));
          }
        });
      },
    );

    test('3. choice validates out and p even when sampleCount == 0', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );

        final disposedOut = NDArray<Float64>.create([0], DType.float64);
        disposedOut.dispose();
        expect(() => choice(a, size: [0], out: disposedOut), throwsStateError);

        final wrongShapeOut = NDArray<Float64>.create([2], DType.float64);
        expect(
          () => choice(a, size: [0], out: wrongShapeOut),
          throwsArgumentError,
        );

        final negP = NDArray<Float64>.fromList(
          [0.5, -0.2, 0.7],
          [3],
          DType.float64,
        );
        expect(() => choice(a, size: [0], p: negP), throwsArgumentError);

        final zeroSumP = NDArray<Float64>.fromList(
          [0.0, 0.0, 0.0],
          [3],
          DType.float64,
        );
        expect(() => choice(a, size: [0], p: zeroSumP), throwsArgumentError);
      });
    });

    test(
      '4. multivariateNormal with out aliasing mean or cov computes valid samples without corruption',
      () {
        NDArray.scope(() {
          final meanOrig = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0],
            [3],
            DType.float64,
          );
          final covOrig = NDArray<Float64>.fromList(
            [
              1.0, 0.5, 0.2, //
              0.5, 2.0, 0.3, //
              0.2, 0.3, 1.5, //
            ],
            [3, 3],
            DType.float64,
          );

          // Test out aliasing mean (when size is null, output shape is [3])
          final meanAlias = meanOrig.copy();
          final expectedMean = multivariateNormal(meanOrig, covOrig, seed: 777);
          final actualMean = multivariateNormal(
            meanAlias,
            covOrig,
            seed: 777,
            out: meanAlias,
          );
          expect(identical(actualMean, meanAlias), isTrue);
          for (var i = 0; i < 3; i++) {
            expect(
              meanAlias.getCellFlat(i),
              closeTo(expectedMean.getCellFlat(i), 1e-12),
            );
          }

          // Test out aliasing a view sharing memory with mean across multiple samples (size: [2])
          final backing = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0, 0.0, 0.0, 0.0],
            [2, 3],
            DType.float64,
          );
          final meanSlice = backing
              .slice([Slice(start: 0, stop: 1), Slice.all()])
              .reshape([3]);
          final expectedMulti = multivariateNormal(
            meanOrig,
            covOrig,
            size: [2],
            seed: 888,
          );
          final actualMulti = multivariateNormal(
            meanSlice,
            covOrig,
            size: [2],
            seed: 888,
            out: backing,
          );
          expect(identical(actualMulti, backing), isTrue);
          for (var i = 0; i < 6; i++) {
            expect(
              backing.getCellFlat(i),
              closeTo(expectedMulti.getCellFlat(i), 1e-12),
            );
          }

          // Test out aliasing cov (when size is [3], output shape is [3, 3])
          final covAlias = covOrig.copy();
          final expectedCov = multivariateNormal(
            meanOrig,
            covOrig,
            size: [3],
            seed: 999,
          );
          final actualCov = multivariateNormal(
            meanOrig,
            covAlias,
            size: [3],
            seed: 999,
            out: covAlias,
          );
          expect(identical(actualCov, covAlias), isTrue);
          for (var i = 0; i < 9; i++) {
            expect(
              covAlias.getCellFlat(i),
              closeTo(expectedCov.getCellFlat(i), 1e-12),
            );
          }
        });
      },
    );

    test(
      '5. shuffle on a non-contiguous strided rank = 34 array shuffles along axis 0 without no-opping',
      () {
        NDArray.scope(() {
          // Base shape: [6, 1, ..., 1, 4] (34 dimensions)
          // Take a strided view on the last axis (step 2) so shape is [6, 1, ..., 1, 2]
          // and slice_is_contiguous is false!
          final baseShape = <int>[6, ...List<int>.filled(32, 1), 4];
          expect(baseShape.length, equals(34));
          final base = NDArray<Float64>.create(baseShape, DType.float64);
          for (var i = 0; i < 6; i++) {
            for (var k = 0; k < 4; k++) {
              final coords = <int>[i, ...List<int>.filled(32, 0), k];
              base.setCell(coords, (i + 1) * 100.0 + k);
            }
          }

          final slices = <Selector>[
            const Slice.all(),
            ...List<Selector>.filled(32, const Slice.all()),
            const Slice(start: 0, stop: 4, step: 2),
          ];
          final view = base.slice(slices);
          expect(view.rank, equals(34));
          expect(view.shape.first, equals(6));
          expect(view.shape.last, equals(2));
          expect(view.isContiguous, isFalse);

          shuffle(view, seed: 42);

          final seenRows = <int>[];
          for (var i = 0; i < 6; i++) {
            final c0 = <int>[i, ...List<int>.filled(32, 0), 0];
            final c1 = <int>[i, ...List<int>.filled(32, 0), 1];
            final v0 = view.getCell(c0);
            final v1 = view.getCell(c1);
            final rowId = (v0 / 100.0).round();
            expect(v0, equals(rowId * 100.0 + 0));
            expect(v1, equals(rowId * 100.0 + 2));
            seenRows.add(rowId);

            // Odd indices in base (1 and 3) should remain untouched
            final cOdd1 = <int>[i, ...List<int>.filled(32, 0), 1];
            final cOdd3 = <int>[i, ...List<int>.filled(32, 0), 3];
            expect(base.getCell(cOdd1), equals((i + 1) * 100.0 + 1));
            expect(base.getCell(cOdd3), equals((i + 1) * 100.0 + 3));
          }

          // Must be a valid permutation of [1, 2, 3, 4, 5, 6] and not identity
          expect(seenRows..sort(), equals([1, 2, 3, 4, 5, 6]));
          final unshuffledRows = <int>[];
          for (var i = 0; i < 6; i++) {
            final c0 = <int>[i, ...List<int>.filled(32, 0), 0];
            unshuffledRows.add((view.getCell(c0) / 100.0).round());
          }
          expect(unshuffledRows, isNot(equals([1, 2, 3, 4, 5, 6])));
        });
      },
    );
  });
}
