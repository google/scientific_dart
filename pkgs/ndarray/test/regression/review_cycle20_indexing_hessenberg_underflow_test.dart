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
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Cycle 20 Regression Tests', () {
    group(
      'Finding 1: take_along_axis and put_along_axis with all integer index dtypes',
      () {
        test(
          'take_along_axis supports int8, uint16, uint32, uint64 indices (1D, 2D, 3D)',
          () {
            NDArray.scope(() {
              final arr1d = NDArray<Float64>.fromList(
                [10.0, 20.0, 30.0, 40.0],
                [4],
                DType.float64,
              );
              for (final idxDType in [
                DType.int8,
                DType.uint16,
                DType.uint32,
                DType.uint64,
              ]) {
                final idx = NDArray.fromList([3, 1, 0], [3], idxDType);
                final res = take_along_axis(arr1d, idx, 0);
                expect(res[[0]], equals(40.0));
                expect(res[[1]], equals(20.0));
                expect(res[[2]], equals(10.0));
              }

              // Signed negative indices in int8
              final idxNegInt8 = NDArray<Int8>.fromList(
                [-1, -3],
                [2],
                DType.int8,
              );
              final resNeg = take_along_axis(arr1d, idxNegInt8, 0);
              expect(resNeg[[0]], equals(40.0));
              expect(resNeg[[1]], equals(20.0));

              // 2D test across axis 0 and axis 1
              final arr2d = NDArray<Float64>.fromList(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [2, 3],
                DType.float64,
              );
              final idx2d = NDArray<Uint32>.fromList(
                [2, 0, 1, 1],
                [2, 2],
                DType.uint32,
              );
              final res2d = take_along_axis(arr2d, idx2d, 1);
              expect(res2d[[0, 0]], equals(3.0));
              expect(res2d[[0, 1]], equals(1.0));
              expect(res2d[[1, 0]], equals(5.0));
              expect(res2d[[1, 1]], equals(5.0));
            });
          },
        );

        test(
          'put_along_axis supports int8, uint16, uint32, uint64 indices (1D, 2D)',
          () {
            NDArray.scope(() {
              for (final idxDType in [
                DType.int8,
                DType.uint16,
                DType.uint32,
                DType.uint64,
              ]) {
                final arr = NDArray<Float64>.fromList(
                  [10.0, 20.0, 30.0, 40.0],
                  [4],
                  DType.float64,
                );
                final idx = NDArray.fromList([3, 1], [2], idxDType);
                final vals = NDArray<Float64>.fromList(
                  [99.0, 88.0],
                  [2],
                  DType.float64,
                );
                put_along_axis(arr, idx, vals, 0);
                expect(arr[[0]], equals(10.0));
                expect(arr[[1]], equals(88.0));
                expect(arr[[2]], equals(30.0));
                expect(arr[[3]], equals(99.0));
              }
            });
          },
        );

        test(
          'uint64 indices >= 2^63 throw RangeError instead of wrapping as negative',
          () {
            NDArray.scope(() {
              final arr = NDArray<Float64>.fromList(
                [10.0, 20.0, 30.0, 40.0],
                [4],
                DType.float64,
              );
              final idx = NDArray<Uint64>.zeros([1], DType.uint64);
              // Write 0xFFFFFFFFFFFFFFFE (-2 as signed int64, which would wrap to 4 - 2 = 2 if signed)
              idx.pointer.cast<ffi.Uint64>()[0] = -2;

              expect(
                () => take_along_axis(arr, idx, 0),
                throwsA(isA<RangeError>()),
              );

              final vals = NDArray<Float64>.fromList(
                [99.0],
                [1],
                DType.float64,
              );
              expect(
                () => put_along_axis(arr, idx, vals, 0),
                throwsA(isA<RangeError>()),
              );
            });
          },
        );
      },
    );

    group('Finding 2: hessenberg on strongly-typed integer NDArray inputs', () {
      test(
        'hessenberg succeeds on NDArray<Int32>, NDArray<Int64>, NDArray<Uint8>',
        () {
          NDArray.scope(() {
            final aInt32 = NDArray<Int32>.fromList(
              [1, 2, 3, 4, 5, 6, 7, 8, 9],
              [3, 3],
              DType.int32,
            );
            final res32 = hessenberg(aInt32);
            expect(res32.h.dtype, equals(DType.float64));
            expect(res32.q.dtype, equals(DType.float64));
            expect(res32.h[[2, 0]], closeTo(0.0, 1e-12));

            // Verify Q * H * Q^T == A
            final q = res32.q;
            final h = res32.h;
            final qt = q.transpose();
            final reconstructed = matmul(matmul(q, h), qt);
            expect(reconstructed[[0, 0]], closeTo(1.0, 1e-10));
            expect(reconstructed[[1, 1]], closeTo(5.0, 1e-10));
            expect(reconstructed[[2, 2]], closeTo(9.0, 1e-10));

            final aInt64 = NDArray<Int64>.fromList(
              [2, 1, 0, 1, 3, 1, 0, 1, 2],
              [3, 3],
              DType.int64,
            );
            final outH = NDArray<Float64>.zeros([3, 3], DType.float64);
            final outQ = NDArray<Float64>.zeros([3, 3], DType.float64);
            final res64 = hessenberg(aInt64, outH: outH, outQ: outQ);
            expect(identical(res64.h, outH), isTrue);
            expect(identical(res64.q, outQ), isTrue);
            expect(outH[[2, 0]], closeTo(0.0, 1e-12));

            final aUint8 = NDArray<Uint8>.fromList(
              [1, 0, 0, 0, 2, 0, 0, 0, 3],
              [3, 3],
              DType.uint8,
            );
            final outHU8 = NDArray<Float64>.zeros([3, 3], DType.float64);
            final outQU8 = NDArray<Float64>.zeros([3, 3], DType.float64);
            final resU8 = hessenberg(aUint8, outH: outHU8, outQ: outQU8);
            expect(identical(resU8.h, outHU8), isTrue);
            expect(identical(resU8.q, outQU8), isTrue);
            expect(outHU8[[0, 0]], closeTo(1.0, 1e-12));
            expect(outHU8[[1, 1]], closeTo(2.0, 1e-12));
            expect(outHU8[[2, 2]], closeTo(3.0, 1e-12));
          });
        },
      );
    });

    group(
      'Finding 3: put_along_axis does not mutate out before validating values shape',
      () {
        test(
          'out buffer remains untouched when values has incompatible shape',
          () {
            NDArray.scope(() {
              final arr = NDArray<Float64>.fromList(
                [1.0, 2.0, 3.0, 4.0],
                [2, 2],
                DType.float64,
              );
              final out = NDArray<Float64>.fromList(
                [100.0, 200.0, 300.0, 400.0],
                [2, 2],
                DType.float64,
              );
              final indices = NDArray<Int32>.fromList(
                [0, 1, 1, 0],
                [2, 2],
                DType.int32,
              );
              // Incompatible values shape [2, 3] vs indices shape [2, 2]
              final badValues = NDArray<Float64>.zeros([2, 3], DType.float64);

              expect(
                () => put_along_axis(arr, indices, badValues, 1, out: out),
                throwsArgumentError,
              );

              // Verify out was NOT overwritten with arr's contents
              expect(out[[0, 0]], equals(100.0));
              expect(out[[0, 1]], equals(200.0));
              expect(out[[1, 0]], equals(300.0));
              expect(out[[1, 1]], equals(400.0));
            });
          },
        );
      },
    );

    group(
      'Finding 4: geomspace and brentq sub-1e-162 float underflow in sign check',
      () {
        test(
          'geomspace succeeds for sub-1e-162 start and stop of same sign',
          () {
            NDArray.scope(() {
              final pos = geomspace(1e-200, 1e-180, 5, dtype: DType.float64);
              expect(pos.shape, equals([5]));
              expect(pos[[0]], closeTo(1e-200, 1e-212));
              expect(pos[[2]], closeTo(1e-190, 1e-202));
              expect(pos[[4]], closeTo(1e-180, 1e-192));

              final neg = geomspace(-1e-200, -1e-180, 5, dtype: DType.float64);
              expect(neg[[0]], closeTo(-1e-200, 1e-212));
              expect(neg[[4]], closeTo(-1e-180, 1e-192));

              expect(
                () => geomspace(1e-200, -1e-200, 5, dtype: DType.float64),
                throwsArgumentError,
              );
            });
          },
        );

        test(
          'brentq rejects same-sign sub-1e-162 endpoints and converges on opposite-sign sub-1e-162',
          () {
            // Same sign at endpoints (1e-200 and 1e-200): product underflows to 0.0, must throw ArgumentError
            expect(() => brentq((x) => 1e-200, 1.0, 2.0), throwsArgumentError);
            expect(() => brentq((x) => -1e-200, 1.0, 2.0), throwsArgumentError);

            // Opposite sign at endpoints: f(x) = 1e-200 * (x - 1.5) on [1.0, 2.0]
            final res = brentq((x) => 1e-200 * (x - 1.5), 1.0, 2.0);
            expect(res.converged, isTrue);
            expect(res.root, closeTo(1.5, 1e-10));
          },
        );
      },
    );
  });
}
