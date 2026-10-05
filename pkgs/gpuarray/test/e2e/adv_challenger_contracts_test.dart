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

import 'package:gpuarray/fft.dart' as gpu_fft;
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/linalg.dart' as gpu_linalg;
import 'package:gpuarray/nn.dart' as gpu_nn;
import 'package:ndarray/ndarray.dart' as nd;
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

void _expectMatchNDArray(
  GpuArray<DTypeTag> actual,
  nd.NDArray<nd.DTypeTag> expected, {
  double tol = 1e-4,
  String? reason,
}) {
  expect(actual.shape, equals(expected.shape), reason: reason);
  final actualNd = actual.toNDArray();
  final flatActual = actualNd.reshape([actualNd.size]);
  final flatExpected = expected.reshape([expected.size]);
  try {
    for (var i = 0; i < actual.size; i++) {
      final a = flatActual.getCell([i]);
      final e = flatExpected.getCell([i]);
      if (a is num && e is num) {
        if (a.isNaN && e.isNaN) continue;
        if (a.isInfinite && e.isInfinite && a.sign == e.sign) continue;
        expect(
          a.toDouble(),
          closeTo(e.toDouble(), tol),
          reason: '$reason [$i] actual=$a expected=$e',
        );
      } else if (a is nd.Complex && e is nd.Complex) {
        expect(a.real, closeTo(e.real, tol), reason: '$reason [$i].real');
        expect(a.imag, closeTo(e.imag, tol), reason: '$reason [$i].imag');
      } else {
        expect(a, equals(e), reason: '$reason [$i]');
      }
    }
  } finally {
    flatActual.dispose();
    actualNd.dispose();
    flatExpected.dispose();
  }
}

void main() {
  late GpuDevice device;

  setUpAll(() {
    device = GpuDevice.defaultDevice;
  });

  group('Adversarial R1 Audit: Static Typing & Precision Invariants', () {
    test(
      'R1.1: Static typing preservation across standard nn layers on Float32',
      () {
        ResourceScope.scope(() {
          final x = GpuArray<Float32>.fromList(
            List<double>.generate(8, (i) => (i - 4) * 0.5),
            [2, 4],
            DType.float32,
            device: device,
          );

          final linear = gpu_nn.Linear(
            4,
            3,
            dtype: DType.float32,
            device: device,
          );
          final GpuArray<Float32> linOut = linear.forward(x);
          expect(linOut.dtype, equals(DType.float32));
          expect(linOut.shape, equals([2, 3]));

          final ln = gpu_nn.LayerNorm(
            [4],
            dtype: DType.float32,
            device: device,
          );
          final GpuArray<Float32> lnOut = ln.forward(x);
          expect(lnOut.dtype, equals(DType.float32));

          final rms = gpu_nn.RMSNorm([4], dtype: DType.float32, device: device);
          final GpuArray<Float32> rmsOut = rms.forward(x);
          expect(rmsOut.dtype, equals(DType.float32));

          final dropout = gpu_nn.Dropout(p: 0.2);
          final GpuArray<Float32> dropOut = dropout.forward(x);
          expect(dropOut.dtype, equals(DType.float32));

          final mha = gpu_nn.MultiheadAttention(
            4,
            2,
            dtype: DType.float32,
            device: device,
          );
          final x3d = x.reshape([1, 2, 4]);
          final GpuArray<Float32> mhaOut = mha.forward(x3d);
          expect(mhaOut.dtype, equals(DType.float32));
          expect(mhaOut.shape, equals([1, 2, 4]));

          final bn = gpu_nn.BatchNorm1d(
            4,
            dtype: DType.float32,
            device: device,
          );
          final GpuArray<Float32> bnOut = bn.forward(x);
          expect(bnOut.dtype, equals(DType.float32));

          final xImg = GpuArray<Float32>.ones(
            [1, 2, 4, 4],
            DType.float32,
            device: device,
          );
          final conv = gpu_nn.Conv2d(
            2,
            3,
            3,
            dtype: DType.float32,
            device: device,
          );
          final GpuArray<Float32> convOut = conv.forward(xImg);
          expect(convOut.dtype, equals(DType.float32));
        });
      },
    );

    test(
      'R1.2: mean() & nanmean() dtype preservation and all-NaN handling',
      () {
        ResourceScope.scope(() {
          final f32 = GpuArray<Float32>.fromList(
            [2.0, 4.0],
            [2],
            DType.float32,
            device: device,
          );
          final f16 = GpuArray<Float16>.fromList(
            [2.0, 4.0],
            [2],
            DType.float16,
            device: device,
          );
          final bf16 = GpuArray<BFloat16>.fromList(
            [2.0, 4.0],
            [2],
            DType.bfloat16,
            device: device,
          );
          final f64 = GpuArray<Float64>.fromList(
            [2.0, 4.0],
            [2],
            DType.float64,
            device: device,
          );
          final i32 = GpuArray<Int32>.fromList(
            [2, 4],
            [2],
            DType.int32,
            device: device,
          );
          final bArr = GpuArray<Boolean>.fromList(
            [true, false],
            [2],
            DType.boolean,
            device: device,
          );

          expect(f32.mean().dtype, equals(DType.float32));
          expect(f16.mean().dtype, equals(DType.float16));
          expect(bf16.mean().dtype, equals(DType.bfloat16));
          expect(f64.mean().dtype, equals(DType.float64));
          expect(i32.mean().dtype, equals(DType.float64));
          expect(bArr.mean().dtype, equals(DType.float64));

          final allNan = GpuArray<Float32>.fromList(
            [double.nan, double.nan],
            [2],
            DType.float32,
            device: device,
          );
          final nanMeanVal = (allNan.nanmean().scalar as num).toDouble();
          expect(nanMeanVal.isNaN, isTrue);

          final nanSumVal = (nansum(allNan).scalar as num).toDouble();
          expect(nanSumVal, equals(0.0));
        });
      },
    );

    test('R1.3: Int64 return types on all index and count operations', () {
      ResourceScope.scope(() {
        final data = GpuArray<Float32>.fromList(
          [3.0, 1.0, 4.0, 1.0, 5.0, 9.0],
          [2, 3],
          DType.float32,
          device: device,
        );

        expect(argmin(data, axis: 0).dtype, equals(DType.int64));
        expect(argmax(data, axis: 1).dtype, equals(DType.int64));
        expect(argmin(data).dtype, equals(DType.int64));
        expect(argmax(data).dtype, equals(DType.int64));

        final nz = nonzero(data);
        for (final arr in nz) {
          expect(arr.dtype, equals(DType.int64));
        }
        expect(flatnonzero(data).dtype, equals(DType.int64));
        expect(argwhere(data).dtype, equals(DType.int64));

        expect(argsort(data, axis: 1).dtype, equals(DType.int64));
        expect(argpartition(data, 1, axis: 1).dtype, equals(DType.int64));
        expect(countNonzero(data, axis: 1).dtype, equals(DType.int64));
        expect(countNonzero(data).dtype, equals(DType.int64));

        final top = topk(data, 2, axis: 1);
        expect(top.indices.dtype, equals(DType.int64));
        expect(top.values.dtype, equals(DType.float32));

        final uAll = uniqueAll(data);
        expect(uAll.indices.dtype, equals(DType.int64));
        expect(uAll.inverse.dtype, equals(DType.int64));
        expect(uAll.counts.dtype, equals(DType.int64));

        final intData = GpuArray<Int32>.fromList(
          [1, 2, 2, 4],
          [4],
          DType.int32,
          device: device,
        );
        expect(bincount(intData).dtype, equals(DType.int64));
      });
    });

    test('R1.4: Native Float32 and Complex64 execution on linalg and fft', () {
      nd.NDArray.scope(() {
        ResourceScope.scope(() {
          final ndSpd = nd.NDArray<Float32>.fromList(
            [5.0, 2.0, 2.0, 4.0],
            [2, 2],
            nd.DType.float32,
          );
          final ndB = nd.NDArray<Float32>.fromList(
            [1.0, 2.0],
            [2],
            nd.DType.float32,
          );
          final gpuSpd = GpuArray<Float32>.fromNDArray(ndSpd, device: device);
          final gpuB = GpuArray<Float32>.fromNDArray(ndB, device: device);

          final GpuArray<Float32> s = gpu_linalg.svdValues(gpuSpd);
          final GpuArray<Float32> chol = gpu_linalg.cholesky(gpuSpd);
          final GpuArray<Float32> sol = gpu_linalg.solve(gpuSpd, gpuB);
          final GpuArray<Float32> inv = gpu_linalg.inv(gpuSpd);
          final GpuArray<Float32> det = gpu_linalg.det(gpuSpd);
          final GpuArray<Float32> norm = gpu_linalg.norm(gpuSpd);
          final GpuArray<Float32> cond = gpu_linalg.cond(gpuSpd);

          expect(s.dtype, equals(DType.float32));
          expect(chol.dtype, equals(DType.float32));
          expect(sol.dtype, equals(DType.float32));
          expect(inv.dtype, equals(DType.float32));
          expect(det.dtype, equals(DType.float32));
          expect(norm.dtype, equals(DType.float32));
          expect(cond.dtype, equals(DType.float32));

          _expectMatchNDArray(
            s,
            nd.svd(ndSpd).s,
            tol: 1e-4,
            reason: 'svdValues',
          );
          _expectMatchNDArray(
            chol,
            nd.cholesky(ndSpd),
            tol: 1e-4,
            reason: 'cholesky',
          );
          _expectMatchNDArray(
            sol,
            nd.solve(ndSpd, ndB),
            tol: 1e-4,
            reason: 'solve',
          );
          _expectMatchNDArray(inv, nd.inv(ndSpd), tol: 1e-4, reason: 'inv');
          _expectMatchNDArray(det, nd.det(ndSpd), tol: 1e-4, reason: 'det');
          _expectMatchNDArray(norm, nd.norm(ndSpd), tol: 1e-4, reason: 'norm');
          _expectMatchNDArray(cond, nd.cond(ndSpd), tol: 1e-3, reason: 'cond');

          final GpuArray<Complex64> spec = gpu_fft.rfft(gpuB);
          expect(spec.dtype, equals(DType.complex64));
          final GpuArray<Float32> rec = gpu_fft.irfft(spec, n: 2);
          expect(rec.dtype, equals(DType.float32));
          _expectMatchNDArray(
            rec,
            ndB,
            tol: 1e-4,
            reason: 'rfft/irfft round-trip',
          );
        });
      });
    });
  });

  group('Adversarial R4 Audit: Stressing Sorting, Searching, Scans & Ufuncs', () {
    test(
      'R4.1: Sorting & searching with large arrays (> 512) and duplicate elements',
      () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            // Array of 600 elements to stress workgroup boundary (> 512)
            final rnd = math.Random(42);
            final rawVals = List<double>.generate(
              600,
              (_) => (rnd.nextInt(200) - 100).toDouble(),
            );
            final ndBig = nd.NDArray<Float32>.fromList(rawVals, [
              600,
            ], nd.DType.float32);
            final gpuBig = GpuArray<Float32>.fromNDArray(ndBig, device: device);

            final GpuArray<Float32> gSorted = sort(gpuBig);
            final GpuArray<Int64> gArgsorted = argsort(gpuBig);
            final ndSorted = nd.sort(ndBig);

            expect(gSorted.shape, equals([600]));
            expect(gArgsorted.dtype, equals(DType.int64));
            _expectMatchNDArray(
              gSorted,
              ndSorted,
              tol: 1e-5,
              reason: 'large sort 600 elements',
            );

            final argsortedIdxList = gArgsorted.toList();
            final sortedValList = ndSorted.toList();
            for (var i = 0; i < 600; i++) {
              final origIdx = (argsortedIdxList[i] as num).toInt();
              expect(rawVals[origIdx], equals(sortedValList[i]));
            }

            final topRes = topk(gpuBig, 10);
            expect(topRes.values.shape, equals([10]));
            expect(topRes.indices.dtype, equals(DType.int64));
            // Check top 10 values match descending end of sorted ndArray
            final topNDVals = ndSorted.toList().reversed.take(10).toList();
            final topValsList = topRes.values.toList();
            for (var i = 0; i < 10; i++) {
              expect(
                (topValsList[i] as num).toDouble(),
                closeTo(topNDVals[i], 1e-5),
              );
            }

            // Array with identical elements
            final identical = GpuArray<Float32>.filled(
              [300],
              7.0,
              DType.float32,
              device: device,
            );
            final sortedIdentical = sort(identical);
            expect(
              sortedIdentical.toList(),
              equals(List<double>.filled(300, 7.0)),
            );

            // searchsorted out of bounds and exact match
            final sortedArr = GpuArray<Int32>.fromList(
              [10, 20, 30, 40, 50],
              [5],
              DType.int32,
              device: device,
            );
            final queries = GpuArray<Int32>.fromList(
              [5, 10, 25, 50, 60],
              [5],
              DType.int32,
              device: device,
            );
            final ndSortedArr = nd.NDArray<Int32>.fromList(
              [10, 20, 30, 40, 50],
              [5],
              nd.DType.int32,
            );
            final ndQueries = nd.NDArray<Int32>.fromList(
              [5, 10, 25, 50, 60],
              [5],
              nd.DType.int32,
            );

            final ssL = searchsorted(sortedArr, queries, side: SearchSide.left);
            final ssR = searchsorted(
              sortedArr,
              queries,
              side: SearchSide.right,
            );
            _expectMatchNDArray(
              ssL,
              nd.searchsorted(ndSortedArr, ndQueries, side: nd.SearchSide.left),
              reason: 'ss left',
            );
            _expectMatchNDArray(
              ssR,
              nd.searchsorted(
                ndSortedArr,
                ndQueries,
                side: nd.SearchSide.right,
              ),
              reason: 'ss right',
            );

            // bincount with weights and negative value validation
            final binX = GpuArray<Int32>.fromList(
              [0, 1, 1, 2, 2, 2],
              [6],
              DType.int32,
              device: device,
            );
            final binW = GpuArray<Float64>.fromList(
              [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
              [6],
              DType.float64,
              device: device,
            );
            final ndBinX = nd.NDArray<Int32>.fromList(
              [0, 1, 1, 2, 2, 2],
              [6],
              nd.DType.int32,
            );
            final ndBinW = nd.NDArray<Float64>.fromList(
              [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
              [6],
              nd.DType.float64,
            );
            final weightedBin = bincount(binX, weights: binW);
            expect(weightedBin.dtype, equals(DType.float64));
            _expectMatchNDArray(
              weightedBin,
              nd.bincount(ndBinX, weights: ndBinW),
              reason: 'bincount with weights',
            );

            final negBinX = GpuArray<Int32>.fromList(
              [-1, 2],
              [2],
              DType.int32,
              device: device,
            );
            expect(() => bincount(negBinX), throwsArgumentError);

            // partition and argpartition with negative kth
            final partArr = GpuArray<Float32>.fromList(
              [5.0, 2.0, 9.0, 1.0, 7.0],
              [5],
              DType.float32,
              device: device,
            );
            final pNeg = partition(partArr, -1);
            final apNeg = argpartition(partArr, -1);
            expect(apNeg.dtype, equals(DType.int64));
            expect((pNeg.toList().last as num).toDouble(), equals(9.0));
          });
        });
      },
    );

    test(
      'R4.2: Scans & differences across workgroup boundary (> 256 elements)',
      () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            // Array of 600 elements: exercises multi-block prefix scan across 256-element blocks!
            final rawVals = List<int>.generate(600, (i) => (i % 5) + 1);
            final ndScan = nd.NDArray<Int32>.fromList(rawVals, [
              600,
            ], nd.DType.int32);
            final gpuScan = GpuArray<Int32>.fromNDArray(ndScan, device: device);

            final gCumsum = cumsum(gpuScan);
            expect(gCumsum.dtype, equals(DType.int64));
            _expectMatchNDArray(
              gCumsum,
              nd.cumsum(ndScan),
              reason: 'cumsum > 256 elements',
            );

            final smallVals = List<int>.generate(20, (i) => 2);
            final ndProd = nd.NDArray<Int32>.fromList(smallVals, [
              20,
            ], nd.DType.int32);
            final gpuProd = GpuArray<Int32>.fromNDArray(ndProd, device: device);
            final gCumprod = cumprod(gpuProd);
            _expectMatchNDArray(
              gCumprod,
              nd.cumprod(ndProd),
              reason: 'cumprod',
            );

            // diff with n = 0, n = 2, and prepend/append
            final gDiff0 = diff(gpuScan, n: 0);
            expect(gDiff0.toList(), equals(gpuScan.toList()));

            final gDiff2 = diff(gpuScan, n: 2);
            _expectMatchNDArray(
              gDiff2,
              nd.diff(ndScan, n: 2),
              reason: 'diff n=2',
            );

            final gDiffPre = diff(gpuScan, prepend: 0);
            expect(gDiffPre.shape, equals([600]));
            expect(gDiffPre.toList().first, equals(gpuScan.toList().first - 0));
          });
        });
      },
    );

    test('R4.3: Statistical & NaN-aware reductions edge cases', () {
      nd.NDArray.scope(() {
        ResourceScope.scope(() {
          // Large offset with small variance: tests numerical stability
          final raw = [1000000.0, 1000001.0, 1000002.0];
          final ndOffset = nd.NDArray<Float32>.fromList(raw, [
            3,
          ], nd.DType.float32);
          final gpuOffset = GpuArray<Float32>.fromNDArray(
            ndOffset,
            device: device,
          );

          _expectMatchNDArray(
            variance(gpuOffset),
            nd.var_(ndOffset),
            tol: 1e-4,
            reason: 'variance large offset',
          );
          _expectMatchNDArray(
            std(gpuOffset),
            nd.std(ndOffset),
            tol: 1e-4,
            reason: 'std large offset',
          );

          // ptp with negative values
          final negVals = [-50.0, -10.0, 20.0];
          final ndNeg = nd.NDArray<Float32>.fromList(negVals, [
            3,
          ], nd.DType.float32);
          final gpuNeg = GpuArray<Float32>.fromNDArray(ndNeg, device: device);
          _expectMatchNDArray(
            ptp(gpuNeg),
            nd.ptp(ndNeg),
            tol: 1e-5,
            reason: 'ptp neg',
          );

          // NaN reductions with mixed values
          final nanRaw = [double.nan, 3.0, double.nan, 1.0, 5.0, double.nan];
          final ndNan = nd.NDArray<Float32>.fromList(nanRaw, [
            6,
          ], nd.DType.float32);
          final gpuNan = GpuArray<Float32>.fromNDArray(ndNan, device: device);

          _expectMatchNDArray(
            nanmin(gpuNan),
            nd.nanmin(ndNan),
            reason: 'nanmin',
          );
          _expectMatchNDArray(
            nanmax(gpuNan),
            nd.nanmax(ndNan),
            reason: 'nanmax',
          );
          _expectMatchNDArray(
            nansum(gpuNan),
            nd.nansum(ndNan),
            reason: 'nansum',
          );
          _expectMatchNDArray(
            nanmean(gpuNan),
            nd.nanmean(ndNan),
            reason: 'nanmean',
          );
        });
      });
    });

    test('R4.4: Ufuncs, 4-quadrant trigonometry, bitwise & complex ufuncs', () {
      nd.NDArray.scope(() {
        ResourceScope.scope(() {
          final xVals = [1.0, -1.0, -1.0, 1.0];
          final yVals = [1.0, 1.0, -1.0, -1.0];
          final ndX = nd.NDArray<Float32>.fromList(xVals, [
            4,
          ], nd.DType.float32);
          final ndY = nd.NDArray<Float32>.fromList(yVals, [
            4,
          ], nd.DType.float32);
          final gpuX = GpuArray<Float32>.fromNDArray(ndX, device: device);
          final gpuY = GpuArray<Float32>.fromNDArray(ndY, device: device);

          _expectMatchNDArray(
            atan2(gpuY, gpuX),
            nd.atan2(ndY, ndX),
            tol: 1e-4,
            reason: 'atan2 4 quadrants',
          );
          _expectMatchNDArray(
            hypot(gpuY, gpuX),
            nd.hypot(ndY, ndX),
            tol: 1e-4,
            reason: 'hypot 4 quadrants',
          );

          // nanToNum with custom values
          final specVals = [
            double.nan,
            double.infinity,
            double.negativeInfinity,
            42.0,
          ];
          final ndSpec = nd.NDArray<Float32>.fromList(specVals, [
            4,
          ], nd.DType.float32);
          final gpuSpec = GpuArray<Float32>.fromNDArray(ndSpec, device: device);
          _expectMatchNDArray(
            nanToNum(gpuSpec, nan: 99.0, posinf: 888.0, neginf: -888.0),
            nd.nan_to_num(ndSpec, nan: 99.0, posinf: 888.0, neginf: -888.0),
            reason: 'nanToNum custom',
          );

          // Bitwise operations on Int64
          final bitValsA = [0x0F0F0F0F, 0x55555555];
          final bitValsB = [0x00FF00FF, 0xAAAAAAAA];
          final ndBitA = nd.NDArray<Int64>.fromList(bitValsA, [
            2,
          ], nd.DType.int64);
          final ndBitB = nd.NDArray<Int64>.fromList(bitValsB, [
            2,
          ], nd.DType.int64);
          final gpuBitA = GpuArray<Int64>.fromNDArray(ndBitA, device: device);
          final gpuBitB = GpuArray<Int64>.fromNDArray(ndBitB, device: device);

          _expectMatchNDArray(
            gpuBitA & gpuBitB,
            ndBitA & ndBitB,
            reason: 'Int64 bitwise &',
          );
          _expectMatchNDArray(
            gpuBitA | gpuBitB,
            ndBitA | ndBitB,
            reason: 'Int64 bitwise |',
          );
          _expectMatchNDArray(
            gpuBitA ^ gpuBitB,
            ndBitA ^ ndBitB,
            reason: 'Int64 bitwise ^',
          );
          _expectMatchNDArray(~gpuBitA, ~ndBitA, reason: 'Int64 bitwise ~');

          // Complex ufuncs across 4 quadrants on Complex64
          final cVals = [
            nd.Complex(1.0, 1.0),
            nd.Complex(-1.0, 1.0),
            nd.Complex(-1.0, -1.0),
            nd.Complex(1.0, -1.0),
          ];
          final ndC = nd.NDArray<Complex64>.fromList(cVals, [
            4,
          ], nd.DType.complex64);
          final gpuC = GpuArray<Complex64>.fromNDArray(ndC, device: device);

          _expectMatchNDArray(gpuC.real(), nd.real(ndC), reason: 'real');
          _expectMatchNDArray(gpuC.imag(), nd.imag(ndC), reason: 'imag');
          _expectMatchNDArray(conj(gpuC), nd.conj(ndC), reason: 'conj');
          _expectMatchNDArray(
            gpuC.angle(),
            nd.angle(ndC),
            tol: 1e-4,
            reason: 'angle',
          );
        });
      });
    });
  });
}
