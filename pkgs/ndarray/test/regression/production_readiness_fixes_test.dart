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

import 'dart:async' show Zone;
import 'dart:io';
import 'dart:typed_data';
import 'package:archive/archive.dart';
import 'package:ndarray/ndarray.dart';
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

void main() {
  group('Workstream 1: Memory, Scope, & Stride Safety Fixes', () {
    test(
      'C1: _s_stat_strided_fallback does not overflow buffer when out is strided',
      () {
        NDArray.scope(() {
          // Create an input array with float16 (which uses fallback path in stats.dart)
          final a = NDArray<Float16>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float16,
          );

          // Create a larger backing array and take a non-contiguous slice as out
          final base = NDArray<Float16>.zeros([3, 3], DType.float16);
          // Base elements are 0. We'll use a strided slice: rows 0, 1, 2 at col 1 (strides = [3])
          final outView = NDArray<Float16>.view(
            base,
            shape: [3],
            strides: [3],
            offsetElements: 1,
          );
          expect(outView.isContiguous, isFalse);

          // Sum along axis 0 (reducing 2x3 -> 3) into outView
          final res = sum<Float16>(a, axis: 0, out: outView);
          expect(identical(res, outView), isTrue);

          // Column sums: [1+4, 2+5, 3+6] = [5.0, 7.0, 9.0]
          expect(outView.getCell([0]), equals(5.0));
          expect(outView.getCell([1]), equals(7.0));
          expect(outView.getCell([2]), equals(9.0));

          // Check that untouched base elements remain 0.0
          expect(base.getCell([0, 0]), equals(0.0));
          expect(base.getCell([0, 2]), equals(0.0));
          expect(base.getCell([1, 0]), equals(0.0));
          expect(base.getCell([1, 2]), equals(0.0));
          expect(base.getCell([2, 0]), equals(0.0));
          expect(base.getCell([2, 2]), equals(0.0));
        });
      },
    );

    test('H6: sort and argsort write correctly into non-contiguous out', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [4.0, 2.0, 5.0, 1.0, 3.0, 6.0],
          [2, 3],
          DType.float64,
        );

        // Backing array for out
        final baseSort = NDArray<Float64>.zeros([2, 5], DType.float64);
        final outSort = NDArray<Float64>.view(
          baseSort,
          shape: [2, 3],
          strides: [5, 1],
          offsetElements: 1,
        );
        expect(outSort.isContiguous, isFalse);

        sort(a, axis: -1, out: outSort);
        // Row 0 sorted: [2.0, 4.0, 5.0]
        // Row 1 sorted: [1.0, 3.0, 6.0]
        expect(outSort.getCell([0, 0]), equals(2.0));
        expect(outSort.getCell([0, 1]), equals(4.0));
        expect(outSort.getCell([0, 2]), equals(5.0));
        expect(outSort.getCell([1, 0]), equals(1.0));
        expect(outSort.getCell([1, 1]), equals(3.0));
        expect(outSort.getCell([1, 2]), equals(6.0));

        // argsort with non-contiguous out
        final baseArgsort = NDArray.zeros([2, 5], DType.int32);
        final outArgsort = NDArray<Int32>.view(
          baseArgsort,
          shape: [2, 3],
          strides: [5, 1],
          offsetElements: 1,
        );
        expect(outArgsort.isContiguous, isFalse);

        argsort(a, axis: -1, out: outArgsort);
        expect(outArgsort.getCell([0, 0]), equals(1));
        expect(outArgsort.getCell([0, 1]), equals(0));
        expect(outArgsort.getCell([0, 2]), equals(2));
        expect(outArgsort.getCell([1, 0]), equals(0));
        expect(outArgsort.getCell([1, 1]), equals(1));
        expect(outArgsort.getCell([1, 2]), equals(2));
      });
    });

    test(
      'H6: partition and argpartition write correctly into non-contiguous out',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [9.0, 1.0, 7.0, 3.0, 5.0, 2.0],
            [2, 3],
            DType.float64,
          );

          final basePart = NDArray<Float64>.zeros([2, 6], DType.float64);
          final outPart = NDArray<Float64>.view(
            basePart,
            shape: [2, 3],
            strides: [6, 1],
            offsetElements: 2,
          );
          expect(outPart.isContiguous, isFalse);

          partition(a, 1, axis: -1, out: outPart);
          // For row 0: elements are 9, 1, 7. kth=1 means index 1 should have 7.0, index 0 <= 7.0, index 2 >= 7.0
          expect(outPart.getCell([0, 1]), equals(7.0));
          expect(outPart.getCell([0, 0]), lessThanOrEqualTo(7.0));
          expect(outPart.getCell([0, 2]), greaterThanOrEqualTo(7.0));

          final baseArgpart = NDArray.zeros([2, 6], DType.int32);
          final outArgpart = NDArray<Int32>.view(
            baseArgpart,
            shape: [2, 3],
            strides: [6, 1],
            offsetElements: 2,
          );
          expect(outArgpart.isContiguous, isFalse);

          argpartition(a, 1, axis: -1, out: outArgpart);
          final kthIdx = outArgpart.getCell([0, 1]);
          expect(a.getCell([0, kthIdx]), equals(7.0));
        });
      },
    );

    test(
      'C4: mean on empty axis returns NaN instead of uninitialized heap garbage',
      () {
        NDArray.scope(() {
          // Shape [0, 4], axis: 0 -> targetShape is [4]
          final emptyArr = NDArray<Float64>.zeros([0, 4], DType.float64);

          final m0 = mean(emptyArr, axis: 0);
          expect(m0.shape, equals([4]));
          for (var i = 0; i < 4; i++) {
            expect(m0.getCell([i]).toDouble().isNaN, isTrue);
          }

          // Test with explicit out buffer
          final outBuf = NDArray<Float64>.zeros([4], DType.float64);
          mean(emptyArr, axis: 0, out: outBuf);
          for (var i = 0; i < 4; i++) {
            expect(outBuf.getCell([i]).isNaN, isTrue);
          }

          // Test std and var on empty axis
          final s0 = std(emptyArr, axis: 0);
          expect(s0.shape, equals([4]));
          for (var i = 0; i < 4; i++) {
            expect(s0.getCell([i]).isNaN, isTrue);
          }

          final v0 = variance(emptyArr, axis: 0);
          expect(v0.shape, equals([4]));
          for (var i = 0; i < 4; i++) {
            expect(v0.getCell([i]).isNaN, isTrue);
          }
        });
      },
    );

    test('H2: matmul(x, x, out: x) handles in-place aliasing safely', () {
      NDArray.scope(() {
        final x = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        // x * x = [[7, 10], [15, 22]]
        final res = matmul(x, x, out: x);
        expect(identical(res, x), isTrue);

        expect(x.getCell([0, 0]), equals(7.0));
        expect(x.getCell([0, 1]), equals(10.0));
        expect(x.getCell([1, 0]), equals(15.0));
        expect(x.getCell([1, 1]), equals(22.0));
      });
    });

    test(
      'H2: matmul(a, b, out: a) handles in-place aliasing with first operand',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 0.0, 0.0, 2.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray<Float64>.fromList(
            [3.0, 4.0, 5.0, 6.0],
            [2, 2],
            DType.float64,
          );

          // [[1, 0], [0, 2]] * [[3, 4], [5, 6]] = [[3, 4], [10, 12]]
          matmul(a, b, out: a);
          expect(a.getCell([0, 0]), equals(3.0));
          expect(a.getCell([0, 1]), equals(4.0));
          expect(a.getCell([1, 0]), equals(10.0));
          expect(a.getCell([1, 1]), equals(12.0));
        });
      },
    );

    test('H7: put_along_axis does not adopt caller array into local scope', () {
      late NDArray<Float64> callerArr;

      NDArray.scope(() {
        callerArr = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );

        // Execute put_along_axis inside an inner scope
        NDArray.scope(() {
          final indices = NDArray.fromList([1, 0], [2, 1], DType.int32);
          final values = NDArray<Float64>.fromList(
            [99.0, 88.0],
            [2, 1],
            DType.float64,
          );
          put_along_axis(callerArr, indices, values, 1);
        });

        // After inner scope exits, callerArr must NOT be disposed!
        expect(callerArr.isDisposed, isFalse);
        expect(callerArr.getCell([0, 1]), equals(99.0));
        expect(callerArr.getCell([1, 0]), equals(88.0));
      });

      // Now outer scope has exited, so callerArr is disposed
      expect(callerArr.isDisposed, isTrue);
    });

    test(
      'H7: take_along_axis does not dispose caller out array across scopes',
      () {
        late NDArray<Float64> callerOut;

        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [2, 2],
            DType.float64,
          );
          callerOut = NDArray<Float64>.zeros([2, 1], DType.float64);

          NDArray.scope(() {
            final indices = NDArray.fromList([1, 0], [2, 1], DType.int32);
            take_along_axis(a, indices, 1, out: callerOut);
          });

          expect(callerOut.isDisposed, isFalse);
          expect(callerOut.getCell([0, 0]), equals(20.0));
          expect(callerOut.getCell([1, 0]), equals(30.0));
        });

        expect(callerOut.isDisposed, isTrue);
      },
    );

    test(
      'H8: nanvar and nanstd return owned arrays that can be explicitly disposed',
      () {
        final a = NDArray<Float64>.fromList(
          [1.0, double.nan, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );

        final v = nanvar(a, axis: 0);
        expect(v.isDisposed, isFalse);
        expect(v.shape, equals([3]));
        v.dispose();
        expect(v.isDisposed, isTrue);

        final s = nanstd(a, axis: 0);
        expect(s.isDisposed, isFalse);
        expect(s.shape, equals([3]));
        s.dispose();
        expect(s.isDisposed, isTrue);

        a.dispose();
        expect(a.isDisposed, isTrue);
      },
    );

    test(
      'H14: Closed ResourceScope rejects late tracking and disposes resource immediately',
      () {
        late dynamic capturedZone;
        NDArray.scope(() {
          capturedZone = Zone.current;
        });

        NDArray? lateArr;
        expect(
          () {
            capturedZone.run(() {
              lateArr = NDArray<Float64>.zeros([2, 2], DType.float64);
            });
          },
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              contains('closed scope'),
            ),
          ),
        );

        // Even though constructor threw, if lateArr was assigned or tracked, it was disposed
        expect(lateArr, isNull);
      },
    );
  });

  group('Workstream 4 - DX & Core APIs (M2)', () {
    test('M2: astype() converts dtypes and respects copy parameter', () {
      final a = NDArray.fromList([1, 2, 3], [3], DType.int32);

      // copy: false with matching dtype returns the identical instance
      final same = a.astype(DType.int32, copy: false);
      expect(identical(same, a), isTrue);

      // copy: true with matching dtype returns a new copy
      final copied = a.astype(DType.int32, copy: true);
      expect(identical(copied, a), isFalse);
      expect(copied.dtype, DType.int32);
      expect(copied.toList(), [1, 2, 3]);

      // Convert Int32 -> Float64
      final asFloat = a.astype(DType.float64);
      expect(asFloat.dtype, DType.float64);
      expect(asFloat.toList(), [1.0, 2.0, 3.0]);

      // Convert Float64 -> Int32 (truncation)
      final f = NDArray.fromList([1.7, 2.4, -3.9], [3], DType.float64);
      final truncated = f.astype(DType.int32);
      expect(truncated.dtype, DType.int32);
      expect(truncated.toList(), [1, 2, -3]);

      // Convert Int64 -> Boolean (non-zero -> true)
      final ints = NDArray.fromList([0, 5, -1, 0], [4], DType.int64);
      final bools = ints.astype(DType.boolean);
      expect(bools.dtype, DType.boolean);
      expect(bools.toList(), [false, true, true, false]);

      // Convert Boolean -> Int32 (true -> 1, false -> 0)
      final boolInts = bools.astype(DType.int32);
      expect(boolInts.dtype, DType.int32);
      expect(boolInts.toList(), [0, 1, 1, 0]);

      // Top-level astype function
      final topLevel = astype(a, DType.float32);
      expect(topLevel.dtype, DType.float32);
      expect(topLevel.toList(), [1.0, 2.0, 3.0]);

      // Disposed array throws StateError
      copied.dispose();
      expect(() => copied.astype(DType.float64), throwsStateError);
      expect(() => astype(copied, DType.float64), throwsStateError);
    });

    test('M2: toString() formats 0D, 1D, 2D, and large truncated arrays', () {
      // 0-D scalars
      final sInt = NDArray.scalar(42, dtype: DType.int32);
      expect(sInt.toString(), '42, dtype=int32');

      final sFloat = NDArray.scalar(3.14, dtype: DType.float64);
      expect(sFloat.toString(), '3.14, dtype=float64');

      final sBool = NDArray.scalar(true, dtype: DType.boolean);
      expect(sBool.toString(), 'true, dtype=boolean');

      // 1-D small float64 (omits dtype=float64 per NumPy conventions)
      final aF64 = NDArray.fromList([1.0, 2.0, 3.5], [3], DType.float64);
      expect(aF64.toString(), '[1., 2., 3.5]');

      // 1-D small integer (includes dtype)
      final aI32 = NDArray.fromList([1, 2, 3], [3], DType.int32);
      expect(aI32.toString(), '[1, 2, 3], dtype=int32');

      // 1-D large array truncation (threshold > 6 elements)
      final large1D = NDArray.arange(0.0, 10.0, dtype: DType.float64);
      expect(large1D.toString(), '[0., 1., 2., ..., 7., 8., 9.]');

      // 2-D matrix formatting with column alignment
      final mat = NDArray.arange(
        0.0,
        12.0,
        dtype: DType.float64,
      ).reshape([3, 4]);
      final matStr = mat.toString();
      expect(matStr, contains('[[0., 1.,  2.,  3.]'));
      expect(matStr, contains('[4., 5.,  6.,  7.]'));
      expect(matStr, contains('[8., 9., 10., 11.]]'));

      // 2-D large matrix truncation
      final large2D = NDArray.arange(
        0.0,
        100.0,
        dtype: DType.float64,
      ).reshape([10, 10]);
      final large2DStr = large2D.toString();
      expect(large2DStr, contains('...'));
      expect(large2DStr, contains('[[ 0.,  1.,  2., ...,  7.,  8.,  9.]'));
      expect(large2DStr, contains('[90., 91., 92., ..., 97., 98., 99.]]'));

      // Empty arrays
      final empty1D = NDArray<Float64>.create([0], DType.float64);
      expect(empty1D.toString(), '[]');

      final empty2D = NDArray<Int32>.create([0, 3], DType.int32);
      expect(empty2D.toString(), '[], shape=[0, 3], dtype=int32');

      // Disposed array
      empty1D.dispose();
      expect(empty1D.toString(), '<disposed NDArray<float64>>');
    });
  });

  group('Numerical Correctness & Input Validation Regressions', () {
    test('C2: Array with size > 2^31 elements throws UnsupportedError', () {
      expect(
        () => NDArray<Uint8>.create([2147483648], DType.uint8),
        throwsA(isA<UnsupportedError>()),
      );
    });

    test(
      'C3: Malformed .npz where dataLen does not match shape/dtype throws FormatException',
      () {
        final tempDir = Directory.systemTemp.createTempSync('npz_test_');
        final npzPath = '${tempDir.path}/malformed.npz';
        try {
          final archive = Archive();
          final header =
              "{'descr': '<f8', 'fortran_order': False, 'shape': (1,)}";
          final padCount = 64 - 10 - header.length - 1;
          final padded = "$header${' ' * padCount}\n";
          final bytes = BytesBuilder();
          bytes.add([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00]);
          final lenBytes = Uint8List(2);
          ByteData.view(
            lenBytes.buffer,
          ).setUint16(0, padded.length, Endian.little);
          bytes.add(lenBytes);
          bytes.add(padded.codeUnits);
          bytes.add(Uint8List(200)); // 200 bytes payload instead of 8 bytes!

          final fileBytes = bytes.toBytes();
          archive.addFile(ArchiveFile('test.npy', fileBytes.length, fileBytes));
          final zipData = ZipEncoder().encode(archive)!;
          File(npzPath).writeAsBytesSync(zipData);

          expect(() => loadz(npzPath), throwsA(isA<FormatException>()));
        } finally {
          tempDir.deleteSync(recursive: true);
        }
      },
    );

    test('H3: _wrapScalar preserves target dtype for float32 and int8', () {
      final f32 = NDArray.fromList([1.0, 2.0], [2], DType.float32);
      final resF32 = f32 + 1.0;
      expect(resF32.dtype, DType.float32);

      final i8 = NDArray.fromList([10, 20], [2], DType.int8);
      final resI8 = i8 + 1;
      expect(resI8.dtype, DType.int8);

      final i32 = NDArray.fromList([5, 10], [2], DType.int32);
      final resI32 = i32 * 2;
      expect(resI32.dtype, DType.int32);
    });

    test('H4: min, max, argmax, and median propagate NaN correctly', () {
      final a = NDArray.fromList([1.0, double.nan, 3.0], [3], DType.float64);
      expect(min(a).scalar.isNaN, isTrue, reason: 'min must propagate NaN');
      expect(max(a).scalar.isNaN, isTrue, reason: 'max must propagate NaN');
      expect(
        argmax(a).scalar,
        1,
        reason: 'argmax must return first NaN index per NumPy',
      );
      expect(
        median(a).scalar.isNaN,
        isTrue,
        reason: 'median must propagate NaN',
      );
      a.dispose();
    });

    test(
      'H5: sum of boolean returns int64 count of trues and sumAs(boolean) preserves Boolean',
      () {
        final b = NDArray.fromList(
          [true, true, false, true],
          [4],
          DType.boolean,
        );
        final sSame = sumAs(b, DType.boolean);
        expect(sSame.dtype, DType.boolean);
        expect(sSame.scalar, true);
        final s = sum(b);
        expect(s.dtype, DType.int64);
        expect(s.scalar, 3);
        b.dispose();
      },
    );

    test(
      'H10: searchsorted rejects out-of-bounds sorter indices and wrong dtype',
      () {
        final arr = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final v = NDArray.fromList([1.5], [1], DType.float64);

        // Wrong sorter dtype (float64 instead of int32)
        final wrongDType = NDArray.fromList(
          [0.0, 1.0, 2.0],
          [3],
          DType.float64,
        );
        expect(
          () => searchsorted(arr, v, sorter: wrongDType as dynamic),
          throwsA(anyOf(isA<ArgumentError>(), isA<TypeError>())),
        );

        // Out of bounds index
        final oobSorter = NDArray.fromList([0, 1, 999], [3], DType.int32);
        expect(() => searchsorted(arr, v, sorter: oobSorter), throwsRangeError);

        // Negative index
        final negSorter = NDArray.fromList([-1, 0, 1], [3], DType.int32);
        expect(() => searchsorted(arr, v, sorter: negSorter), throwsRangeError);

        arr.dispose();
        v.dispose();
        wrongDType.dispose();
        oobSorter.dispose();
        negSorter.dispose();
      },
    );

    group('Review Gate Regressions and Hardening Tests', () {
      test(
        'matmul vector promotion inside ResourceScope.returning produces owned valid array',
        () {
          final res = ResourceScope.returning(() {
            final v = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            final m = NDArray.fromList(
              [3.0, 4.0, 5.0, 6.0],
              [2, 2],
              DType.float64,
            );
            return matmul(v, m);
          });

          // Verify result is valid after scope exit
          expect(res.shape, [2]);
          expect(res.toList(), equals([13.0, 16.0]));
          // Verify it is an owned array that can be explicitly disposed
          expect(res.isDisposed, isFalse);
          res.dispose();
          expect(res.isDisposed, isTrue);
        },
      );

      test(
        'nanstd with axis: null and keepdims: true returns 1D/scalar array',
        () {
          final a = NDArray.fromList(
            [1.0, 2.0, double.nan, 3.0],
            [4],
            DType.float64,
          );
          final s = nanstd(a, keepdims: true);
          expect(s.shape, [1]);
          expect(s.getCellFlat(0), closeTo(0.816496, 1e-4));
          a.dispose();
          s.dispose();
        },
      );

      test('argsort with int64 out buffer on fallback dtype', () {
        final a = NDArray.fromList([10, -5, 20], [3], DType.int8);
        final out64 = NDArray.zeros([3], DType.int64);
        final res = argsortAs(a, DType.int64, out: out64);
        expect(identical(res, out64), isTrue);
        expect(res.dtype, DType.int64);
        expect(res.toList(), equals([1, 0, 2]));
        a.dispose();
        out64.dispose();
      });

      test('searchsorted with uint64 and int64 out buffer', () {
        // -1 as uint64 is 2^64 - 1
        final a = NDArray.fromList([10, 20, -1], [3], DType.uint64);
        final v = NDArray.fromList([15, -1], [2], DType.uint64);
        final out64 = NDArray.zeros([2], DType.int64);
        final res = searchsortedAs(a, v, DType.int64, out: out64);
        expect(res.dtype, DType.int64);
        expect(res.toList(), equals([1, 2]));
        a.dispose();
        v.dispose();
        out64.dispose();
      });

      test('argmax and argmin with uint64 values >= 2^63', () {
        // 100 vs -1 (where -1 is 2^64 - 1 in unsigned 64-bit)
        final a = NDArray.fromList([100, -1], [2], DType.uint64);
        expect(argmax(a).scalar, 1);
        expect(argmin(a).scalar, 0);

        // 2D along axis
        final a2D = NDArray.fromList([100, -1, -1, 100], [2, 2], DType.uint64);
        expect(argmax(a2D, axis: 1).toList(), equals([1, 0]));
        expect(argmin(a2D, axis: 1).toList(), equals([0, 1]));

        a.dispose();
        a2D.dispose();
      });

      test('argpartition with uint64 values >= 2^63', () {
        final a = NDArray.fromList([100, -1, 50], [3], DType.uint64);
        final part = argpartition(a, 1);
        // Smallest is 50 (index 2), then 100 (index 0), largest is -1 (index 1)
        expect(part.getCellFlat(1), 0);
        a.dispose();
        part.dispose();
      });

      test('complex64 + real scalar maintains Complex64', () {
        final c = NDArray.fromList([Complex(1.0, 2.0)], [1], DType.complex64);
        final res = c + 5.0;
        expect(res.dtype, DType.complex64);
        expect(res.getCellFlat(0), equals(Complex(6.0, 2.0)));
        c.dispose();
        res.dispose();
      });

      test(
        'shape with dim > 2^31 throws UnsupportedError regardless of zero dimension',
        () {
          expect(
            () => NDArray<Float64>.create([0, 2147483648], DType.float64),
            throwsUnsupportedError,
          );
        },
      );

      test(
        'B4: view with stride > 2^31 - 1 or < -2^31 throws UnsupportedError',
        () {
          final a = NDArray<Float64>.create([1], DType.float64);
          expect(
            () => NDArray<Float64>.view(a, shape: [1], strides: [0x80000000]),
            throwsUnsupportedError,
          );
          expect(
            () => NDArray<Float64>.view(a, shape: [1], strides: [-0x80000001]),
            throwsUnsupportedError,
          );
          a.dispose();
        },
      );

      test(
        'B4: broadcastBinaryStrides guards commonShape exceeding 32-bit limit',
        () {
          expect(
            () =>
                broadcastBinaryStrides([1, 100000], [0, 1], [30000, 1], [1, 0]),
            throwsUnsupportedError,
          );
        },
      );

      test('B4: ScratchArena copyInts and copyInt32s guard 32-bit bounds', () {
        expect(
          () => ScratchArena.copyInts([0x80000000]),
          throwsUnsupportedError,
        );
        expect(
          () => ScratchArena.copyInt32s([0x80000000]),
          throwsUnsupportedError,
        );
      });

      test('Instance arithmetic operators +, -, *, ~/, % on NDArray', () {
        final a = NDArray<Int32>.fromList([10, 20], [2], DType.int32);
        final b = NDArray<Int32>.fromList([2, 3], [2], DType.int32);
        final sum = a + b;
        final diff = a - b;
        final prod = a * b;
        final div = a ~/ b;
        final rem = a % b;

        expect(sum.toList(), equals([12, 23]));
        expect(diff.toList(), equals([8, 17]));
        expect(prod.toList(), equals([20, 60]));
        expect(div.toList(), equals([5, 6]));
        expect(rem.toList(), equals([0, 2]));

        a.dispose();
        b.dispose();
        sum.dispose();
        diff.dispose();
        prod.dispose();
        div.dispose();
        rem.dispose();
      });
    });
  });
}
