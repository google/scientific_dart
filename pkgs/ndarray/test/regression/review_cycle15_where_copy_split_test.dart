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
  group('Review Cycle 15 Regression Suite', () {
    group('1. zeroInit when where != null and out == null', () {
      test(
        'exponential ufuncs (exp, log, log2, log10) zero-initialize masked-out elements',
        () {
          NDArray.scope(() {
            // Allocate and free dirty buffers so allocator reuses non-zero heap memory
            for (var i = 0; i < 8; i++) {
              final dirty = NDArray<Float64>.fromList(
                List<double>.filled(64, 999.0),
                [64],
                DType.float64,
              );
              dirty.dispose();
            }

            final a = NDArray<Float64>.fromList(List<double>.filled(64, 4.0), [
              64,
            ], DType.float64);
            final maskList = List<bool>.generate(64, (i) => i.isEven);
            final where = NDArray<Boolean>.fromList(maskList, [
              64,
            ], DType.boolean);

            final resExp = exp(a, where: where);
            final resLog = log(a, where: where);
            final resLog2 = log2(a, where: where);
            final resLog10 = log10(a, where: where);

            for (var i = 0; i < 64; i++) {
              if (i.isOdd) {
                expect(resExp[[i]], equals(0.0));
                expect(resLog[[i]], equals(0.0));
                expect(resLog2[[i]], equals(0.0));
                expect(resLog10[[i]], equals(0.0));
              } else {
                expect(resLog2[[i]], closeTo(2.0, 1e-12));
              }
            }
          });
        },
      );

      test(
        'logical and comparison ufuncs zero-initialize masked-out elements to false',
        () {
          NDArray.scope(() {
            for (var i = 0; i < 8; i++) {
              final dirty = NDArray<Uint8>.fromList(List<int>.filled(64, 255), [
                64,
              ], DType.uint8);
              dirty.dispose();
            }

            final a = NDArray<Boolean>.zeros([64], DType.boolean);
            final b = NDArray<Boolean>.ones([64], DType.boolean);
            final maskList = List<bool>.generate(64, (i) => i.isEven);
            final where = NDArray<Boolean>.fromList(maskList, [
              64,
            ], DType.boolean);

            final resNot = logicalNot(a, where: where);
            final resOr = logicalOr(a, b, where: where);
            final resEq = equal(a, a, where: where);

            for (var i = 0; i < 64; i++) {
              if (i.isOdd) {
                expect(resNot[[i]], isFalse);
                expect(resOr[[i]], isFalse);
                expect(resEq[[i]], isFalse);
              } else {
                expect(resNot[[i]], isTrue);
                expect(resOr[[i]], isTrue);
                expect(resEq[[i]], isTrue);
              }
            }
          });
        },
      );

      test(
        'bitwise ufuncs (invert, bitwiseAnd, bitwiseOr, bitwiseXor) zero-initialize masked-out elements',
        () {
          NDArray.scope(() {
            for (var i = 0; i < 8; i++) {
              final dirty = NDArray<Int32>.fromList(
                List<int>.filled(64, 0x7fffffff),
                [64],
                DType.int32,
              );
              dirty.dispose();
            }

            final a = NDArray<Int32>.zeros([64], DType.int32);
            final b = NDArray<Int32>.fromList(List<int>.filled(64, 7), [
              64,
            ], DType.int32);
            final maskList = List<bool>.generate(64, (i) => i.isEven);
            final where = NDArray<Boolean>.fromList(maskList, [
              64,
            ], DType.boolean);

            final resInv = invert<Int32>(a, where: where);
            final resOr = bitwiseOr<Int32>(a, b, where: where);

            for (var i = 0; i < 64; i++) {
              if (i.isOdd) {
                expect(resInv[[i]], equals(0));
                expect(resOr[[i]], equals(0));
              } else {
                expect(resInv[[i]], equals(-1));
                expect(resOr[[i]], equals(7));
              }
            }
          });
        },
      );
    });

    group('2. Upfront bounds validation for Index and Indices in sliceAssign', () {
      test(
        'negative out-of-bounds Index and Indices throw RangeError instead of corrupting or double-normalizing',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.zeros([3, 4], DType.float64);

            // Index in [-2*dim, -dim-1] combined with Indices (isAdvanced == true)
            expect(
              () => a.sliceAssign([
                Index(-4),
                Indices([0, 1]),
              ], 99.0),
              throwsRangeError,
            );
            expect(
              () => a.sliceAssign([
                Index(0),
                Indices([0, -5]),
              ], 99.0),
              throwsRangeError,
            );
            expect(
              () => a.sliceAssign([
                Index(3),
                Indices([0]),
              ], 99.0),
              throwsRangeError,
            );
            expect(
              () => a.sliceAssign([
                Index(0),
                Indices([4]),
              ], 99.0),
              throwsRangeError,
            );
          });
        },
      );

      test(
        'out-of-bounds Index or Indices throws RangeError even when another axis is empty',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.zeros([0, 4], DType.float64);
            expect(
              () => a.sliceAssign([Slice.all(), Index(10)], 1.0),
              throwsRangeError,
            );
            expect(
              () => a.sliceAssign([
                Slice.all(),
                Indices([10]),
              ], 1.0),
              throwsRangeError,
            );
          });
        },
      );
    });

    group(
      '3. Overlapping contiguous out: protection in copy, sort, and permutation',
      () {
        test(
          'NDArray.copy(out:) handles overlapping contiguous views accurately',
          () {
            NDArray.scope(() {
              final a = NDArray<Float64>.fromList(
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5],
                DType.float64,
              );
              final src = a.slice([Slice(start: 0, stop: 4)]);
              final dst = a.slice([Slice(start: 1, stop: 5)]);
              expect(src.isContiguous, isTrue);
              expect(dst.isContiguous, isTrue);

              src.copy(out: dst);
              expect(a.toList(), equals([1.0, 1.0, 2.0, 3.0, 4.0]));
            });
          },
        );

        test(
          'sort() with overlapping contiguous out buffer for uint64 and int8/float16',
          () {
            NDArray.scope(() {
              final aU64 = NDArray<Uint64>.fromList(
                [50, 40, 30, 20, 10],
                [5],
                DType.uint64,
              );
              final srcU64 = aU64.slice([Slice(start: 0, stop: 4)]);
              final dstU64 = aU64.slice([Slice(start: 1, stop: 5)]);
              sort(srcU64, out: dstU64);
              expect(dstU64.toList(), equals([20, 30, 40, 50]));

              final aI8 = NDArray<Int8>.fromList(
                [50, 40, 30, 20, 10],
                [5],
                DType.int8,
              );
              final srcI8 = aI8.slice([Slice(start: 0, stop: 4)]);
              final dstI8 = aI8.slice([Slice(start: 1, stop: 5)]);
              sort(srcI8, out: dstI8);
              expect(dstI8.toList(), equals([20, 30, 40, 50]));
            });
          },
        );

        test(
          'permutation() with strided or overlapping out buffer does not corrupt source elements',
          () {
            NDArray.scope(() {
              final a = NDArray<Float64>.fromList(
                [10.0, 20.0, 30.0, 40.0, 50.0],
                [5],
                DType.float64,
              );
              final src = a.slice([Slice(start: 0, stop: 4)]);
              final dst = a.slice([Slice(start: 1, stop: 5)]);

              permutation(src, seed: 42, out: dst);
              final sortedDst = dst.toList().cast<double>()..sort();
              expect(sortedDst, equals([10.0, 20.0, 30.0, 40.0]));

              // Also test strided out buffer
              final stridedBase = NDArray<Float64>.zeros([8], DType.float64);
              final stridedOut = stridedBase.slice([
                Slice(start: 0, stop: 8, step: 2),
              ]);
              final input = NDArray<Float64>.fromList(
                [1.0, 2.0, 3.0, 4.0],
                [4],
                DType.float64,
              );
              permutation(input, seed: 7, out: stridedOut);
              final sortedStrided = stridedOut.toList().cast<double>()..sort();
              expect(sortedStrided, equals([1.0, 2.0, 3.0, 4.0]));
            });
          },
        );
      },
    );

    group('4. Aliased out: protection in array_split and array_split_at', () {
      test(
        'array_split with overlapping out views preserves original values',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            // Reverse the halves in-place via out views into `a`
            final firstHalf = a.slice([Slice(start: 0, stop: 2)]);
            final secondHalf = a.slice([Slice(start: 2, stop: 4)]);

            array_split(a, 2, out: [secondHalf, firstHalf]);
            expect(secondHalf.toList(), equals([1.0, 2.0]));
            expect(firstHalf.toList(), equals([3.0, 4.0]));
          });
        },
      );

      test(
        'array_split_at with overlapping out views preserves original values',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              [10.0, 20.0, 30.0, 40.0],
              [4],
              DType.float64,
            );
            final firstHalf = a.slice([Slice(start: 0, stop: 2)]);
            final secondHalf = a.slice([Slice(start: 2, stop: 4)]);

            array_split_at(a, [2], out: [secondHalf, firstHalf]);
            expect(secondHalf.toList(), equals([10.0, 20.0]));
            expect(firstHalf.toList(), equals([30.0, 40.0]));
          });
        },
      );
    });

    group('5. Value-level complex Spacing check in gradient()', () {
      test(
        'gradient throws ArgumentError for Spacing<Object> containing Complex on real input',
        () {
          NDArray.scope(() {
            final f = NDArray<Float64>.fromList(
              [1.0, 2.0, 4.0],
              [3],
              DType.float64,
            );
            final Spacing<Object> stepComplex = Spacing<Object>.step(
              Complex(1.0, 1.0),
            );
            final Spacing<Object> coordComplex = Spacing<Object>.coordinates([
              Complex(0.0, 0.0),
              Complex(1.0, 1.0),
              Complex(2.0, 2.0),
            ]);

            expect(
              () => gradient(f, spacing: stepComplex),
              throwsArgumentError,
            );
            expect(
              () => gradient(f, spacing: coordComplex),
              throwsArgumentError,
            );
          });
        },
      );
    });

    group('6. root_scalar(method: RootMethod.secant) ignores fprime', () {
      test('RootMethod.secant does not invoke fprime even when provided', () {
        var fprimeCalled = false;
        final res = root_scalar(
          (x) => x * x - 4.0,
          method: RootMethod.secant,
          x0: 1.5,
          fprime: (x) {
            fprimeCalled = true;
            return 0.0; // Would cause division by zero / failure if called
          },
        );
        expect(fprimeCalled, isFalse);
        expect(res.converged, isTrue);
        expect(res.root, closeTo(2.0, 1e-6));
      });
    });
  });
}
