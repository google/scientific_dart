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

import 'dart:io';
import 'dart:typed_data';
import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/operations/io.dart' show parseNpyHeader;
import 'package:test/test.dart';

void main() {
  group('Reductions, Stats, Linalg & IO NumPy Parity Regressions', () {
    test('P1 & S2: sum, prod, cumsum, cumprod integer & boolean widening', () {
      NDArray.scope(() {
        // Signed narrow integers -> Int64
        for (final dt in <DType<AnySpec>>[
          DType.int8,
          DType.int16,
          DType.int32,
        ]) {
          final a = NDArray.fromList([100, 100, 100], [3], dt);
          final s = sum(a);
          expect(s.dtype, DType.int64);
          expect(s.scalar, 300);

          final p = prod(a);
          expect(p.dtype, DType.int64);
          expect(p.scalar, 1000000);

          final cs = cumsum(a);
          expect(cs.dtype, DType.int64);
          expect(cs.toList(), [100, 200, 300]);

          final cp = cumprod(a);
          expect(cp.dtype, DType.int64);
          expect(cp.toList(), [100, 10000, 1000000]);
        }

        // Unsigned narrow integers -> Uint64
        for (final dt in <DType<AnySpec>>[
          DType.uint8,
          DType.uint16,
          DType.uint32,
        ]) {
          final a = NDArray.fromList([200, 200, 200], [3], dt);
          final s = sum(a);
          expect(s.dtype, DType.uint64);
          expect(s.scalar, 600);

          final p = prod(a);
          expect(p.dtype, DType.uint64);
          expect(p.scalar, 8000000);

          final cs = cumsum(a);
          expect(cs.dtype, DType.uint64);
          expect(cs.toList(), [200, 400, 600]);

          final cp = cumprod(a);
          expect(cp.dtype, DType.uint64);
          expect(cp.toList(), [200, 40000, 8000000]);
        }

        // Boolean -> Int64
        final b = NDArray<Boolean>.fromList(
          [true, true, false, true],
          [4],
          DType.boolean,
        );
        final sb = sum(b);
        expect(sb.dtype, DType.int64);
        expect(sb.scalar, 3);

        final pb = prod(b);
        expect(pb.dtype, DType.int64);
        expect(pb.scalar, 0);

        final bAllTrue = NDArray<Boolean>.fromList(
          [true, true, true],
          [3],
          DType.boolean,
        );
        expect(prod(bAllTrue).dtype, DType.int64);
        expect(prod(bAllTrue).scalar, 1);

        final csb = cumsum(b);
        expect(csb.dtype, DType.int64);
        expect(csb.toList(), [1, 2, 2, 3]);

        final cpb = cumprod(b);
        expect(cpb.dtype, DType.int64);
        expect(cpb.toList(), [1, 1, 0, 0]);
      });
    });

    test('B6 & B9: float16/bfloat16/float32 sum precision & int64 wrap', () {
      NDArray.scope(() {
        // In float16 (10-bit mantissa), 2048 + 1 == 2048 if accumulated in fp16.
        // With float32 accumulation, 4096 ones sum to 4096.0 (exact in fp16).
        final f16Ones = NDArray.ones([4096], DType.float16);
        final s16 = sum(f16Ones);
        expect(s16.dtype, DType.float16);
        expect(s16.scalar, 4096.0);

        // In bfloat16 (7-bit mantissa), 256 + 1 == 256 if accumulated in bf16.
        // With float32 accumulation, 512 ones sum to 512.0 (exact in bf16).
        final bf16Ones = NDArray.ones([512], DType.bfloat16);
        final sbf16 = sum(bf16Ones);
        expect(sbf16.dtype, DType.bfloat16);
        expect(sbf16.scalar, 512.0);

        // B9: float32 sum and nansum accumulate in float64
        // In float32 (24-bit mantissa), 16777216.0 + 1.0 == 16777216.0.
        final f32Arr = NDArray<Float32>.ones([2049], DType.float32);
        f32Arr.setCell([0], 16777216.0);
        // 16777216.0 + 2048 * 1.0 = 16779264.0 (exact in float32)
        expect(sum(f32Arr).scalar, 16779264.0);
        expect(nansum(f32Arr).scalar, 16779264.0);
      });
    });

    test('S3: NDArray.fromList rejects out-of-range integers', () {
      NDArray.scope(() {
        expect(
          () => NDArray.fromList([256], [1], DType.uint8),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([-1], [1], DType.uint8),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([128], [1], DType.int8),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([-129], [1], DType.int8),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([32768], [1], DType.int16),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([65536], [1], DType.uint16),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([2147483648], [1], DType.int32),
          throwsRangeError,
        );
        expect(
          () => NDArray.fromList([4294967296], [1], DType.uint32),
          throwsRangeError,
        );
      });
    });

    test('S4: var, std, nanvar, nanstd handle ddof >= count and complex', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
        // ddof >= N with non-zero ssd -> Infinity (divisor clamped to 0.0 per NumPy)
        expect(variance(a, ddof: 2).scalar, double.infinity);
        expect(std(a, ddof: 2).scalar, double.infinity);
        expect(variance(a, ddof: 3).scalar, double.infinity);
        expect(std(a, ddof: 3).scalar, double.infinity);

        // ddof >= N with zero ssd -> NaN (0.0 / 0.0 per NumPy)
        final eq = NDArray<Float64>.fromList([2.0, 2.0], [2], DType.float64);
        expect(variance(eq, ddof: 2).scalar.isNaN, isTrue);
        expect(std(eq, ddof: 2).scalar.isNaN, isTrue);
        expect(variance(eq, ddof: 3).scalar.isNaN, isTrue);
        expect(std(eq, ddof: 3).scalar.isNaN, isTrue);

        // Complex variance & std
        final c = NDArray<Complex128>.fromList(
          [Complex(1.0, 2.0), Complex(3.0, 4.0)],
          [2],
          DType.complex128,
        );
        expect(variance(c).scalar, closeTo(2.0, 1e-12));
        expect(std(c).scalar, closeTo(1.4142135623730951, 1e-12));

        // nanvar / nanstd when count <= ddof -> NaN
        final nanArr = NDArray<Float64>.fromList(
          [double.nan, 1.0],
          [2],
          DType.float64,
        );
        expect(nanvar(nanArr, ddof: 1).scalar.isNaN, isTrue);
        expect(nanstd(nanArr, ddof: 1).scalar.isNaN, isTrue);
      });
    });

    test('B2 & H8: interp complex fp support, validation, and Inf/NaN', () {
      NDArray.scope(() {
        final xp = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );
        final fp = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0],
          [3],
          DType.float64,
        );

        // Non-strictly increasing xp throws ArgumentError
        final badXp = NDArray<Float64>.fromList(
          [1.0, 2.0, 2.0],
          [3],
          DType.float64,
        );
        expect(() => interp(xp, badXp, fp), throwsArgumentError);

        // Complex x or xp throws ArgumentError
        final cArr = NDArray<Complex128>.fromList(
          [Complex(1.0, 0.0), Complex(2.0, 0.0), Complex(3.0, 0.0)],
          [3],
          DType.complex128,
        );
        expect(() => interp(cArr, xp, fp), throwsArgumentError);
        expect(() => interp(xp, cArr, fp), throwsArgumentError);

        // Complex fp interpolates both real and imaginary parts
        final cFp = NDArray<Complex128>.fromList(
          [Complex(0.0, 0.0), Complex(10.0, 20.0), Complex(20.0, 40.0)],
          [3],
          DType.complex128,
        );
        final xQuery = NDArray<Float64>.fromList([1.5], [1], DType.float64);
        final cRes = interp(xQuery, xp, cFp);
        expect(cRes.dtype, DType.complex128);
        final cVal = cRes.toList().first;
        expect(cVal.real, closeTo(5.0, 1e-12));
        expect(cVal.imag, closeTo(10.0, 1e-12));

        // NaN and Inf in x
        final x = NDArray<Float64>.fromList(
          [double.negativeInfinity, double.nan, double.infinity],
          [3],
          DType.float64,
        );
        final res = interp(x, xp, fp, left: -99.0, right: 99.0).toList();
        expect(res[0], -99.0);
        expect(res[1].isNaN, isTrue);
        expect(res[2], 99.0);
      });
    });

    test(
      'P2 & S8: median float64 promotion for ints and complex sorting/NaN',
      () {
        NDArray.scope(() {
          // Integer median returns Float64 (1.5 for [1, 2])
          final intArr = NDArray<Int32>.fromList([1, 2], [2], DType.int32);
          final mInt = median(intArr);
          expect(mInt.dtype, DType.float64);
          expect(mInt.scalar, closeTo(1.5, 1e-12));

          // Float32 median preserves Float32
          final f32Arr = NDArray<Float32>.fromList(
            [1.0, 2.0],
            [2],
            DType.float32,
          );
          final mF32 = median(f32Arr);
          expect(mF32.dtype, DType.float32);
          expect(mF32.scalar, closeTo(1.5, 1e-6));

          // S8: Complex median lexicographical sort and NaN propagation
          final cArr = NDArray<Complex128>.fromList(
            [Complex(2.0, 1.0), Complex(1.0, 5.0), Complex(1.0, 1.0)],
            [3],
            DType.complex128,
          );
          final mComplex = median(cArr);
          expect(mComplex.dtype, DType.complex128);
          expect(mComplex.scalar, Complex(1.0, 5.0));

          final cNan = NDArray<Complex128>.fromList(
            [Complex(1.0, 2.0), Complex(0.0, double.nan), Complex(3.0, 4.0)],
            [3],
            DType.complex128,
          );
          final mNan = median(cNan).scalar;
          expect(mNan.real.isNaN, isTrue);
          expect(mNan.imag.isNaN, isTrue);
        });
      },
    );

    test(
      'S7 & B12: quantile and percentile reject complex arrays and invalid q',
      () {
        NDArray.scope(() {
          final c = NDArray<Complex128>.fromList(
            [Complex(1.0, 2.0), Complex(3.0, 4.0)],
            [2],
            DType.complex128,
          );
          expect(() => quantile(c, 0.5), throwsUnsupportedError);
          expect(() => percentile(c, 50.0), throwsUnsupportedError);

          final a = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );
          expect(() => quantile(a, double.nan), throwsArgumentError);
          expect(() => quantile(a, -0.1), throwsArgumentError);
          expect(() => quantile(a, 1.1), throwsArgumentError);
          expect(() => percentile(a, double.nan), throwsArgumentError);
          expect(() => percentile(a, -1.0), throwsArgumentError);
          expect(() => percentile(a, 101.0), throwsArgumentError);
        });
      },
    );

    test('B8: cummin and cummax propagate NaN sticky along axis', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [3.0, 1.0, double.nan, 0.5, 10.0],
          [5],
          DType.float64,
        );
        final cmin = cummin(a).toList().cast<double>();
        expect(cmin[0], 3.0);
        expect(cmin[1], 1.0);
        expect(cmin[2].isNaN, isTrue);
        expect(cmin[3].isNaN, isTrue);
        expect(cmin[4].isNaN, isTrue);

        final cmax = cummax(a).toList().cast<double>();
        expect(cmax[0], 3.0);
        expect(cmax[1], 3.0);
        expect(cmax[2].isNaN, isTrue);
        expect(cmax[3].isNaN, isTrue);
        expect(cmax[4].isNaN, isTrue);
      });
    });

    test(
      'B11: histogram returns Int64 counts, preserves Float32 weights, and density',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float64,
          );
          final res = histogram(a, bins: 2);
          expect(res.hist.dtype, DType.int64);
          expect(res.hist.toList(), [2, 2]);

          final w32 = NDArray<Float32>.fromList(
            [0.5, 1.5, 2.0, 3.0],
            [4],
            DType.float32,
          );
          final resW = histogram(a, bins: 2, weights: w32);
          expect(resW.hist.dtype, DType.float32);
          expect(resW.hist.toList(), [2.0, 5.0]);

          final resDensity = histogram(a, bins: 2, density: true);
          expect(resDensity.hist.dtype, DType.float64);
        });
      },
    );

    test('P10 & B10: linspace exact endpoint and integer floor rounding', () {
      NDArray.scope(() {
        // Exact endpoint for float64
        final l = linspace(0.0, 0.3, 4, endpoint: true, dtype: DType.float64);
        expect(l.toList()[3], 0.3);

        // Integer linspace uses floor rounding
        final lInt = linspace(0, 5, 4, endpoint: true, dtype: DType.int32);
        // 0.0, 1.666 -> 1, 3.333 -> 3, 5.0 -> 5
        expect(lInt.toList(), [0, 1, 3, 5]);

        // linspaceGrid exact endpoint and integer floor
        final startG = NDArray<Int32>.fromList([0], [1], DType.int32);
        final stopG = NDArray<Int32>.fromList([5], [1], DType.int32);
        final gridInt = linspaceGrid(startG, stopG, 4, endpoint: true);
        expect(gridInt.toList(), [0, 1, 3, 5]);
      });
    });

    test(
      'S1: pad on narrow integer arrays avoids overflow in mean/linearRamp',
      () {
        NDArray.scope(() {
          final a = NDArray<Int8>.fromList([100, 100], [2], DType.int8);
          final padded = pad(a, PadWidth.all(1), mode: PadMode.mean);
          expect(padded.dtype, DType.int8);
          expect(padded.toList(), [100, 100, 100, 100]);

          final u16 = NDArray<Uint16>.fromList(
            [50000, 50000],
            [2],
            DType.uint16,
          );
          final paddedU16 = pad(u16, PadWidth.all(1), mode: PadMode.mean);
          expect(paddedU16.dtype, DType.uint16);
          expect(paddedU16.toList(), [50000, 50000, 50000, 50000]);
        });
      },
    );

    test('S5: intersect1d and isin treat NaN != NaN per IEEE-754', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [double.nan, 2.0, 1.0],
          [3],
          DType.float64,
        );
        final b = NDArray<Float64>.fromList(
          [2.0, double.nan, 3.0],
          [3],
          DType.float64,
        );
        final inter = intersect1d(a, b).toList().cast<double>();
        expect(inter, [2.0]);

        final mask = isin(
          a,
          NDArray<Float64>.fromList([double.nan, 2.0], [2], DType.float64),
        ).toList();
        expect(mask, [false, true, false]);
      });
    });

    test('B3: convolve does not conjugate second complex argument', () {
      NDArray.scope(() {
        final a = NDArray<Complex128>.fromList(
          [Complex(1.0, 1.0)],
          [1],
          DType.complex128,
        );
        final v = NDArray<Complex128>.fromList(
          [Complex(1.0, 2.0)],
          [1],
          DType.complex128,
        );
        // (1 + i) * (1 + 2i) = -1 + 3i
        final conv = convolve(a, v).toList().first;
        expect(conv.real, closeTo(-1.0, 1e-12));
        expect(conv.imag, closeTo(3.0, 1e-12));

        // correlate conjugates v: (1 + i) * (1 - 2i) = 3 - i
        final corr = correlate(a, v).toList().first;
        expect(corr.real, closeTo(3.0, 1e-12));
        expect(corr.imag, closeTo(-1.0, 1e-12));
      });
    });

    test(
      'P9 & B18: solve NumPy 2 batched 1D rhs rule and norm 1-element axis list',
      () {
        NDArray.scope(() {
          // 3D A (2, 2, 2) and 1D b (2,) -> broadcasts b across batch -> (2, 2)
          final a3d = NDArray<Float64>.fromList(
            [
              2.0, 0.0, 0.0, 4.0, // batch 0: diag(2, 4)
              5.0, 0.0, 0.0, 10.0, // batch 1: diag(5, 10)
            ],
            [2, 2, 2],
            DType.float64,
          );
          final b1d = NDArray<Float64>.fromList(
            [20.0, 40.0],
            [2],
            DType.float64,
          );
          final xBroadcast = solve(a3d, b1d);
          expect(xBroadcast.shape, [2, 2]);
          expect(xBroadcast.toList(), [10.0, 10.0, 4.0, 4.0]);

          // Under NumPy 2.0 rules, 3D A (2, 2, 2) and 2D b (2, 2) is ambiguous and throws ArgumentError
          final b2d = NDArray<Float64>.fromList(
            [2.0, 8.0, 15.0, 30.0],
            [2, 2],
            DType.float64,
          );
          expect(() => solve(a3d, b2d), throwsArgumentError);

          // B18: norm with 1-element axis list [0]
          final v = NDArray<Float64>.fromList([3.0, 4.0], [2], DType.float64);
          final n = norm(v, axis: [0]);
          expect(n.scalar, closeTo(5.0, 1e-12));
        });
      },
    );

    test(
      'S9: parseNpyHeader Python 2 L suffix and load version validation',
      () {
        final parsed = parseNpyHeader(
          "{'descr': '<f8', 'fortran_order': False, 'shape': (3L, 4L)}",
        );
        expect(parsed.dtype, DType.float64);
        expect(parsed.fortranOrder, isFalse);
        expect(parsed.shape, [3, 4]);

        if (!const bool.fromEnvironment('dart.tool.dart2wasm')) {
          final tmpDir = Directory.systemTemp.createTempSync(
            'npy_parity_test_',
          );
          try {
            NDArray.scope(() {
              final badMagicPath = '${tmpDir.path}/bad_magic.npy';
              File(
                badMagicPath,
              ).writeAsBytesSync([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
              expect(() => load(badMagicPath), throwsFormatException);

              // Save valid array then corrupt version or truncate payload
              final validPath = '${tmpDir.path}/valid.npy';
              final valid = NDArray<Float64>.fromList(
                [1.0, 2.0, 3.0],
                [3],
                DType.float64,
              );
              save(validPath, valid);
              final bytes = File(validPath).readAsBytesSync();

              // Unsupported major version 99
              final badVerBytes = Uint8List.fromList(bytes);
              badVerBytes[6] = 99;
              final badVerPath = '${tmpDir.path}/bad_ver.npy';
              File(badVerPath).writeAsBytesSync(badVerBytes);
              expect(() => load(badVerPath), throwsFormatException);

              // Truncated payload
              final truncPath = '${tmpDir.path}/trunc.npy';
              File(
                truncPath,
              ).writeAsBytesSync(bytes.sublist(0, bytes.length - 4));
              expect(() => load(truncPath), throwsFormatException);
            });
          } finally {
            tmpDir.deleteSync(recursive: true);
          }
        }
      },
    );
  });
}
