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

import 'dart:typed_data';

import 'package:gpuarray/fft.dart' as gpu_fft;
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/jit.dart';
import 'package:gpuarray/linalg.dart' as gpu_linalg;
import 'package:gpuarray/safetensors.dart';
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
    expect(actualNd.shape, equals(expected.shape), reason: reason);
    for (var i = 0; i < actual.size; i++) {
      final a = flatActual.getCell([i]);
      final e = flatExpected.getCell([i]);
      if (a is num && e is num) {
        if (a.isNaN && e.isNaN) continue;
        if (a.isInfinite && e.isInfinite && a.sign == e.sign) continue;
        expect(
          a.toDouble(),
          closeTo(e.toDouble(), tol),
          reason: '$reason [$i]',
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

void _expectArgumentErrorMust(void Function() action, {String? reason}) {
  expect(
    action,
    throwsA(
      isA<ArgumentError>().having(
        (e) => e.message?.toString() ?? '',
        'message',
        startsWith('Must '),
      ),
    ),
    reason: reason,
  );
}

void main() {
  late GpuDevice device;

  setUpAll(() {
    device = GpuDevice.defaultDevice;
  });

  group('Table-driven NDArray oracle parity', () {
    test(
      'Elementwise binary operations across contiguous, transposed, and broadcast views',
      () {
        ResourceScope.scope(() {
          final ndA = nd.NDArray<Float32>.fromList(
            [1.5, 2.0, 3.5, 4.0, 5.5, 6.0],
            [2, 3],
            nd.DType.float32,
          );
          final ndB = nd.NDArray<Float32>.fromList(
            [0.5, 1.5, 2.0, 2.5, 1.0, 3.0],
            [2, 3],
            nd.DType.float32,
          );
          try {
            final gpuA = GpuArray<Float32>.fromNDArray(ndA, device: device);
            final gpuB = GpuArray<Float32>.fromNDArray(ndB, device: device);

            final ndAdd = nd.add(ndA, ndB);
            final ndSub = nd.subtract(ndA, ndB);
            final ndMul = nd.multiply(ndA, ndB);
            final ndDiv = nd.divide(ndA, ndB);
            final ndPow = nd.power(ndA, ndB);
            try {
              _expectMatchNDArray(gpuA.add(gpuB), ndAdd, reason: 'add');
              _expectMatchNDArray(gpuA.subtract(gpuB), ndSub, reason: 'sub');
              _expectMatchNDArray(gpuA.multiply(gpuB), ndMul, reason: 'mul');
              _expectMatchNDArray(gpuA.divide(gpuB), ndDiv, reason: 'div');
              _expectMatchNDArray(gpuA.pow(gpuB), ndPow, reason: 'pow');
            } finally {
              ndAdd.dispose();
              ndSub.dispose();
              ndMul.dispose();
              ndDiv.dispose();
              ndPow.dispose();
            }

            // Transposed views
            final ndTransAdd = nd.add(ndA.transpose(), ndB.transpose());
            try {
              _expectMatchNDArray(
                gpuA.transpose().add(gpuB.transpose()),
                ndTransAdd,
                reason: 'transposed add',
              );
            } finally {
              ndTransAdd.dispose();
            }

            // Broadcast [2, 3] + [1, 3]
            final ndRow = nd.NDArray<Float32>.fromList(
              [10.0, 20.0, 30.0],
              [1, 3],
              nd.DType.float32,
            );
            final ndBroadcastAdd = nd.add(ndA, ndRow);
            try {
              final gpuRow = GpuArray<Float32>.fromNDArray(
                ndRow,
                device: device,
              );
              _expectMatchNDArray(
                gpuA.add(gpuRow),
                ndBroadcastAdd,
                reason: 'broadcast add',
              );
            } finally {
              ndRow.dispose();
              ndBroadcastAdd.dispose();
            }
          } finally {
            ndA.dispose();
            ndB.dispose();
          }
        });
      },
    );

    test('Elementwise unary operations across contiguous and sliced views', () {
      ResourceScope.scope(() {
        final ndA = nd.NDArray<Float32>.fromList(
          [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
          [2, 4],
          nd.DType.float32,
        );
        try {
          final gpuA = GpuArray<Float32>.fromNDArray(ndA, device: device);
          final ndNeg = nd.negative(ndA);
          final ndAbs = nd.abs(ndA);
          final ndExp = nd.exp(ndA);
          final ndLog = nd.log(ndA);
          final ndSqrt = nd.sqrt(ndA);
          final ndSin = nd.sin(ndA);
          final ndCos = nd.cos(ndA);
          final ndTan = nd.tan(ndA);
          final ndTanh = nd.tanh(ndA);
          try {
            _expectMatchNDArray(gpuA.negate(), ndNeg, reason: 'negate');
            _expectMatchNDArray(gpuA.abs(), ndAbs, reason: 'abs');
            _expectMatchNDArray(gpuA.exp(), ndExp, reason: 'exp');
            _expectMatchNDArray(gpuA.log(), ndLog, reason: 'log');
            _expectMatchNDArray(gpuA.sqrt(), ndSqrt, reason: 'sqrt');
            _expectMatchNDArray(gpuA.sin(), ndSin, reason: 'sin');
            _expectMatchNDArray(gpuA.cos(), ndCos, reason: 'cos');
            _expectMatchNDArray(gpuA.tan(), ndTan, reason: 'tan');
            _expectMatchNDArray(gpuA.tanh(), ndTanh, reason: 'tanh');
          } finally {
            ndNeg.dispose();
            ndAbs.dispose();
            ndExp.dispose();
            ndLog.dispose();
            ndSqrt.dispose();
            ndSin.dispose();
            ndCos.dispose();
            ndTan.dispose();
            ndTanh.dispose();
          }

          // Negative-step strided slice
          final gpuFlipped = gpuA.slice([
            const Slice.all(),
            const Slice(3, null, -1),
          ]);
          final ndFlipped = ndA.slice([
            nd.Slice.all(),
            nd.Slice(start: 3, step: -1),
          ]);
          final ndFlippedExp = nd.exp(ndFlipped);
          try {
            _expectMatchNDArray(
              gpuFlipped.exp(),
              ndFlippedExp,
              reason: 'negative-stride exp',
            );
          } finally {
            ndFlippedExp.dispose();
          }
        } finally {
          ndA.dispose();
        }
      });
    });

    test('Reductions across axes and keepDims', () {
      ResourceScope.scope(() {
        final ndA = nd.NDArray<Float64>.fromList(
          [1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
          [2, 3],
          nd.DType.float64,
        );
        try {
          final gpuA = GpuArray<Float64>.fromNDArray(ndA, device: device);
          final sAll = nd.sum(ndA);
          final s0 = nd.sum(ndA, axis: 0);
          final s1Keep = nd.sum(ndA, axis: 1, keepdims: true);
          final mAll = nd.mean(ndA);
          final m0 = nd.mean(ndA, axis: 0);
          final min1 = nd.min(ndA, axis: 1);
          final max0 = nd.max(ndA, axis: 0);
          try {
            _expectMatchNDArray(gpuA.sum(), sAll, reason: 'sum all');
            _expectMatchNDArray(gpuA.sum(axis: 0), s0, reason: 'sum axis 0');
            _expectMatchNDArray(
              gpuA.sum(axis: 1, keepDims: true),
              s1Keep,
              reason: 'sum axis 1 keepDims',
            );
            _expectMatchNDArray(gpuA.mean(), mAll, reason: 'mean all');
            _expectMatchNDArray(gpuA.mean(axis: 0), m0, reason: 'mean axis 0');
            _expectMatchNDArray(gpuA.min(axis: 1), min1, reason: 'min axis 1');
            _expectMatchNDArray(gpuA.max(axis: 0), max0, reason: 'max axis 0');
          } finally {
            sAll.dispose();
            s0.dispose();
            s1Keep.dispose();
            mAll.dispose();
            m0.dispose();
            min1.dispose();
            max0.dispose();
          }
        } finally {
          ndA.dispose();
        }
      });
    });

    test(
      'R4.2: Indexing, manipulation, linalg, and FFT NDArray oracle parity across contiguous and strided views',
      () {
        ResourceScope.scope(() {
          // 1. Indexing (where, take, clip) on contiguous & strided views
          final ndCond = nd.NDArray<Boolean>.fromList(
            [true, false, false, true, true, false],
            [2, 3],
            nd.DType.boolean,
          );
          final ndX = nd.NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [2, 3],
            nd.DType.float64,
          );
          final ndY = nd.NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            nd.DType.float64,
          );
          final ndIdx = nd.NDArray<Int32>.fromList([2, 0], [2], nd.DType.int32);
          final ndWhere = nd.where(ndCond, ndX, ndY) as nd.NDArray<Float64>;
          final ndWhereT =
              nd.where(ndCond.transpose(), ndX.transpose(), ndY.transpose())
                  as nd.NDArray<Float64>;
          final ndTake = nd.NDArray<Float64>.fromList(
            [30.0, 10.0, 60.0, 40.0],
            [2, 2],
            nd.DType.float64,
          );
          try {
            final gpuCond = GpuArray<Boolean>.fromNDArray(
              ndCond,
              device: device,
            );
            final gpuX = GpuArray<Float64>.fromNDArray(ndX, device: device);
            final gpuY = GpuArray<Float64>.fromNDArray(ndY, device: device);
            final gpuIdx = GpuArray<Int32>.fromNDArray(ndIdx, device: device);
            _expectMatchNDArray(
              where<Float64>(gpuCond, gpuX, gpuY),
              ndWhere,
              reason: 'indexing where',
            );
            _expectMatchNDArray(
              where<Float64>(
                gpuCond.transpose(),
                gpuX.transpose(),
                gpuY.transpose(),
              ),
              ndWhereT,
              reason: 'indexing where transposed',
            );
            _expectMatchNDArray(
              take<Float64>(gpuX, gpuIdx, axis: 1),
              ndTake,
              reason: 'indexing take axis 1',
            );
            _expectMatchNDArray(
              select<Float64>([gpuCond], [gpuX], defaultValue: gpuY),
              ndWhere,
              reason: 'indexing select',
            );

            // 2. Manipulation (concatenate, tile, repeat, pad, roll)
            final ndCat = nd.concatenate([ndX, ndY], axis: 0);
            final ndCatT = nd.concatenate([
              ndX.transpose(),
              ndY.transpose(),
            ], axis: 1);
            final ndTile = nd.tile(ndX, [2, 2]);
            final ndRep = nd.repeat(ndX, 2, axis: 1);
            final ndRoll = nd.roll(ndX, 1, axis: 1);
            try {
              _expectMatchNDArray(
                concatenate<Float64>([gpuX, gpuY], axis: 0),
                ndCat,
                reason: 'manipulation concatenate',
              );
              _expectMatchNDArray(
                concatenate<Float64>([
                  gpuX.transpose(),
                  gpuY.transpose(),
                ], axis: 1),
                ndCatT,
                reason: 'manipulation concatenate transposed',
              );
              _expectMatchNDArray(
                tile<Float64>(gpuX, [2, 2]),
                ndTile,
                reason: 'manipulation tile',
              );
              _expectMatchNDArray(
                repeat<Float64>(gpuX, 2, axis: 1),
                ndRep,
                reason: 'manipulation repeat',
              );
              _expectMatchNDArray(
                roll<Float64>(gpuX, 1, axis: 1),
                ndRoll,
                reason: 'manipulation roll',
              );
            } finally {
              ndCat.dispose();
              ndCatT.dispose();
              ndTile.dispose();
              ndRep.dispose();
              ndRoll.dispose();
            }
          } finally {
            ndCond.dispose();
            ndX.dispose();
            ndY.dispose();
            ndIdx.dispose();
            ndWhere.dispose();
            ndWhereT.dispose();
            ndTake.dispose();
          }

          // 3. Linalg (matmul, solve, inv, det, cholesky, svdvals) & 4. FFT (fft, ifft, rfft, irfft)
          final ndSpd = nd.NDArray<Float64>.fromList(
            [4.0, 1.0, 1.0, 3.0],
            [2, 2],
            nd.DType.float64,
          );
          final ndRhs = nd.NDArray<Float64>.fromList(
            [1.0, 2.0],
            [2],
            nd.DType.float64,
          );
          final ndSignal = nd.NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.5, -1.0],
            [2, 4],
            nd.DType.float64,
          );
          final ndSolve = nd.solve(ndSpd, ndRhs);
          final ndInv = nd.inv(ndSpd);
          final ndDet = nd.det(ndSpd);
          final ndChol = nd.cholesky(ndSpd);
          final ndSvd = nd.svd(ndSpd);
          final ndFft = nd.fft(ndSignal);
          final ndRfft = nd.rfft(ndSignal);
          try {
            final gpuSpd = GpuArray<Float64>.fromNDArray(ndSpd, device: device);
            final gpuRhs = GpuArray<Float64>.fromNDArray(ndRhs, device: device);
            final gpuSig = GpuArray<Float64>.fromNDArray(
              ndSignal,
              device: device,
            );
            _expectMatchNDArray(
              gpu_linalg.solve(gpuSpd, gpuRhs),
              ndSolve,
              tol: 1e-5,
              reason: 'linalg solve',
            );
            _expectMatchNDArray(
              gpu_linalg.inv(gpuSpd.transpose()),
              ndInv,
              tol: 1e-5,
              reason: 'linalg inv on transposed SPD view',
            );
            _expectMatchNDArray(
              gpu_linalg.det(gpuSpd),
              ndDet,
              tol: 1e-5,
              reason: 'linalg det',
            );
            _expectMatchNDArray(
              gpu_linalg.cholesky(gpuSpd),
              ndChol,
              tol: 1e-5,
              reason: 'linalg cholesky',
            );
            _expectMatchNDArray(
              gpu_linalg.svdValues(gpuSpd),
              ndSvd.s,
              tol: 1e-5,
              reason: 'linalg svdValues',
            );
            final gpuSpec = gpu_fft.fft(gpuSig);
            _expectMatchNDArray(gpuSpec, ndFft, tol: 1e-4, reason: 'fft');
            _expectMatchNDArray(
              gpu_fft.ifft(gpuSpec),
              ndSignal.astype(nd.DType.complex128),
              tol: 1e-4,
              reason: 'ifft round-trip',
            );
            final gpuRspec = gpu_fft.rfft(gpuSig);
            _expectMatchNDArray(gpuRspec, ndRfft, tol: 1e-4, reason: 'rfft');
            _expectMatchNDArray(
              gpu_fft.irfft(gpuRspec, n: 4),
              ndSignal,
              tol: 1e-4,
              reason: 'irfft round-trip',
            );
          } finally {
            ndSpd.dispose();
            ndRhs.dispose();
            ndSignal.dispose();
            ndSolve.dispose();
            ndInv.dispose();
            ndDet.dispose();
            ndChol.dispose();
            ndSvd.u.dispose();
            ndSvd.s.dispose();
            ndSvd.vh.dispose();
            ndFft.dispose();
            ndRfft.dispose();
          }
        });
      },
    );

    test(
      'R1 & R4: DType preservation, Int64 index/count ops, scans, statistical/NaN reductions, and ufuncs vs NDArray oracle',
      () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            // 1. Unary/binary arithmetic & scalar ops preserve Float32
            final ndF32A = nd.NDArray<Float32>.fromList(
              [4.0, 9.0, 16.0, 25.0],
              [2, 2],
              nd.DType.float32,
            );
            final ndF32B = nd.NDArray<Float32>.fromList(
              [2.0, 3.0, 4.0, 5.0],
              [2, 2],
              nd.DType.float32,
            );
            final gpuF32A = GpuArray<Float32>.fromNDArray(
              ndF32A,
              device: device,
            );
            final gpuF32B = GpuArray<Float32>.fromNDArray(
              ndF32B,
              device: device,
            );

            final GpuArray<Float32> opAdd = gpuF32A + gpuF32B;
            final GpuArray<Float32> opSub = gpuF32A - gpuF32B;
            final GpuArray<Float32> opMul = gpuF32A * gpuF32B;
            final GpuArray<Float32> opDiv = gpuF32A / gpuF32B;
            final GpuArray<Float32> opMod = gpuF32A % gpuF32B;
            final GpuArray<Float32> opFDiv = gpuF32A ~/ gpuF32B;
            final GpuArray<Float32> opScalar = gpuF32A + 1.0;
            final GpuArray<Float32> mMax = maximum(gpuF32A, gpuF32B);
            final GpuArray<Float32> mMin = minimum(gpuF32A, gpuF32B);
            final GpuArray<Float32> mRem = remainder(gpuF32A, gpuF32B);
            final GpuArray<Float32> mFloorDiv = floorDivide(gpuF32A, gpuF32B);
            expect(opAdd.dtype, equals(DType.float32));
            expect(opSub.dtype, equals(DType.float32));
            expect(opMul.dtype, equals(DType.float32));
            expect(opDiv.dtype, equals(DType.float32));
            expect(opMod.dtype, equals(DType.float32));
            expect(opFDiv.dtype, equals(DType.float32));
            expect(opScalar.dtype, equals(DType.float32));
            _expectMatchNDArray(opAdd, nd.add(ndF32A, ndF32B), reason: 'op +');
            _expectMatchNDArray(
              opSub,
              nd.subtract(ndF32A, ndF32B),
              reason: 'op -',
            );
            _expectMatchNDArray(
              opMul,
              nd.multiply(ndF32A, ndF32B),
              reason: 'op *',
            );
            _expectMatchNDArray(
              opDiv,
              nd.divide(ndF32A, ndF32B),
              reason: 'op /',
            );
            _expectMatchNDArray(
              opMod,
              nd.remainder(ndF32A, ndF32B),
              reason: 'op %',
            );
            _expectMatchNDArray(opFDiv, ndF32A ~/ ndF32B, reason: 'op ~/');
            _expectMatchNDArray(
              mRem,
              nd.remainder(ndF32A, ndF32B),
              reason: 'remainder',
            );
            _expectMatchNDArray(
              mFloorDiv,
              ndF32A ~/ ndF32B,
              reason: 'floorDivide',
            );
            expect(mMax.toList(), equals([4.0, 9.0, 16.0, 25.0]));
            expect(mMin.toList(), equals([2.0, 3.0, 4.0, 5.0]));

            // 2. mean() floating-point preservation & integer/bool promotion
            final GpuArray<Float32> meanF32 = gpuF32A.mean();
            final GpuArray<Float16> meanF16 = GpuArray<Float16>.fromList(
              [2.0, 4.0],
              [2],
              DType.float16,
              device: device,
            ).mean();
            final GpuArray<BFloat16> meanBF16 = GpuArray<BFloat16>.fromList(
              [2.0, 6.0],
              [2],
              DType.bfloat16,
              device: device,
            ).mean();
            final GpuArray<Float64> meanF64 = GpuArray<Float64>.fromList(
              [2.0, 6.0],
              [2],
              DType.float64,
              device: device,
            ).mean();
            final meanI32 = GpuArray<Int32>.fromList(
              [1, 3],
              [2],
              DType.int32,
              device: device,
            ).mean();
            final meanBool = GpuArray<Boolean>.fromList(
              [true, false],
              [2],
              DType.boolean,
              device: device,
            ).mean();
            expect(meanF32.dtype, equals(DType.float32));
            expect(meanF16.dtype, equals(DType.float16));
            expect(meanBF16.dtype, equals(DType.bfloat16));
            expect(meanF64.dtype, equals(DType.float64));
            expect(meanI32.dtype, equals(DType.float64));
            expect(meanBool.dtype, equals(DType.float64));
            _expectMatchNDArray(meanF32, nd.mean(ndF32A), reason: 'mean f32');

            // 3. Index/count operations returning GpuArray<Int64>
            final ndSortIn = nd.NDArray<Float32>.fromList(
              [3.0, 0.0, 2.0, 1.0, 5.0, 0.0],
              [2, 3],
              nd.DType.float32,
            );
            final gpuSortIn = GpuArray<Float32>.fromNDArray(
              ndSortIn,
              device: device,
            );
            final GpuArray<Int64> gArgmin = argmin(gpuSortIn, axis: 1);
            final GpuArray<Int64> gArgmax = argmax(gpuSortIn, axis: 1);
            final List<GpuArray<Int64>> gNonzero = nonzero(gpuSortIn);
            final GpuArray<Int64> gFlatnz = flatnonzero(gpuSortIn);
            final GpuArray<Int64> gArgwhere = argwhere(gpuSortIn);
            final GpuArray<Float32> gSort = sort(gpuSortIn, axis: 1);
            final GpuArray<Int64> gArgsort = argsort(gpuSortIn, axis: 1);
            final GpuArray<Float32> gPart = partition(gpuSortIn, 1, axis: 1);
            final GpuArray<Int64> gArgpart = argpartition(
              gpuSortIn,
              1,
              axis: 1,
            );
            final GpuArray<Int64> gCnz = countNonzero(gpuSortIn, axis: 1);
            final topkRes = topk(gpuSortIn, 2, axis: 1);

            expect(gArgmin.dtype, equals(DType.int64));
            expect(gArgmax.dtype, equals(DType.int64));
            expect(gNonzero.first.dtype, equals(DType.int64));
            expect(gFlatnz.dtype, equals(DType.int64));
            expect(gArgwhere.dtype, equals(DType.int64));
            expect(gArgsort.dtype, equals(DType.int64));
            expect(gArgpart.dtype, equals(DType.int64));
            expect(gCnz.dtype, equals(DType.int64));
            expect(topkRes.values.dtype, equals(DType.float32));
            expect(topkRes.indices.dtype, equals(DType.int64));

            _expectMatchNDArray(
              gArgmin,
              nd.argmin(ndSortIn, axis: 1),
              reason: 'argmin',
            );
            _expectMatchNDArray(
              gArgmax,
              nd.argmax(ndSortIn, axis: 1),
              reason: 'argmax',
            );
            final ndNz = nd.nonzero(ndSortIn);
            _expectMatchNDArray(gNonzero[0], ndNz[0], reason: 'nonzero[0]');
            _expectMatchNDArray(gNonzero[1], ndNz[1], reason: 'nonzero[1]');
            _expectMatchNDArray(
              gFlatnz,
              nd.flatnonzero(ndSortIn),
              reason: 'flatnonzero',
            );
            _expectMatchNDArray(
              gArgwhere,
              nd.argwhere(ndSortIn),
              reason: 'argwhere',
            );
            _expectMatchNDArray(
              gSort,
              nd.sort(ndSortIn, axis: 1),
              reason: 'sort',
            );
            _expectMatchNDArray(
              gArgsort,
              nd.argsort(ndSortIn, axis: 1),
              reason: 'argsort',
            );
            _expectMatchNDArray(
              gPart,
              nd.partition(ndSortIn, [1], axis: 1),
              reason: 'partition',
            );
            _expectMatchNDArray(
              gCnz,
              nd.count_nonzero(ndSortIn, axis: 1),
              reason: 'countNonzero',
            );

            // searchsorted, uniqueAll, bincount
            final ndSorted1D = nd.NDArray<Int32>.fromList(
              [10, 20, 20, 30, 40],
              [5],
              nd.DType.int32,
            );
            final ndQuery1D = nd.NDArray<Int32>.fromList(
              [5, 20, 35],
              [3],
              nd.DType.int32,
            );
            final gpuSorted1D = GpuArray<Int32>.fromNDArray(
              ndSorted1D,
              device: device,
            );
            final gpuQuery1D = GpuArray<Int32>.fromNDArray(
              ndQuery1D,
              device: device,
            );
            final GpuArray<Int64> gSearchL = searchsorted(
              gpuSorted1D,
              gpuQuery1D,
              side: SearchSide.left,
            );
            final GpuArray<Int64> gSearchR = searchsorted(
              gpuSorted1D,
              gpuQuery1D,
              side: SearchSide.right,
            );
            expect(gSearchL.dtype, equals(DType.int64));
            expect(gSearchR.dtype, equals(DType.int64));
            _expectMatchNDArray(
              gSearchL,
              nd.searchsorted(ndSorted1D, ndQuery1D, side: nd.SearchSide.left),
              reason: 'searchsorted left',
            );
            _expectMatchNDArray(
              gSearchR,
              nd.searchsorted(ndSorted1D, ndQuery1D, side: nd.SearchSide.right),
              reason: 'searchsorted right',
            );

            final ndUniqIn = nd.NDArray<Int32>.fromList(
              [3, 1, 2, 1, 3, 0, 2],
              [7],
              nd.DType.int32,
            );
            final gpuUniqIn = GpuArray<Int32>.fromNDArray(
              ndUniqIn,
              device: device,
            );
            final gUniqAll = uniqueAll(gpuUniqIn);
            final ndUniqAll = nd.uniqueAll(ndUniqIn);
            expect(gUniqAll.values.dtype, equals(DType.int32));
            expect(gUniqAll.indices.dtype, equals(DType.int64));
            expect(gUniqAll.inverse.dtype, equals(DType.int64));
            expect(gUniqAll.counts.dtype, equals(DType.int64));
            _expectMatchNDArray(
              gUniqAll.values,
              ndUniqAll.values,
              reason: 'uniqueAll.values',
            );
            _expectMatchNDArray(
              gUniqAll.indices,
              ndUniqAll.index,
              reason: 'uniqueAll.indices',
            );
            _expectMatchNDArray(
              gUniqAll.inverse,
              ndUniqAll.inverse,
              reason: 'uniqueAll.inverse',
            );
            _expectMatchNDArray(
              gUniqAll.counts,
              ndUniqAll.counts,
              reason: 'uniqueAll.counts',
            );

            final gBincount = bincount(gpuUniqIn);
            expect(gBincount.dtype, equals(DType.int64));
            _expectMatchNDArray(
              gBincount,
              nd.bincount(ndUniqIn),
              reason: 'bincount',
            );

            // 4. Cumulative scans & differences (cumsum, cumprod, diff)
            final ndScanIn = nd.NDArray<Int32>.fromList(
              [1, 2, 3, 4, 5, 6],
              [2, 3],
              nd.DType.int32,
            );
            final gpuScanIn = GpuArray<Int32>.fromNDArray(
              ndScanIn,
              device: device,
            );
            final gCumsum = cumsum(gpuScanIn, axis: 1);
            final gCumprod = cumprod(gpuScanIn, axis: 1);
            final gDiff = diff(gpuScanIn, n: 1, axis: 1);
            expect(gCumsum.dtype, equals(DType.int64));
            expect(gCumprod.dtype, equals(DType.int64));
            expect(gDiff.dtype, equals(DType.int32));
            _expectMatchNDArray(
              gCumsum,
              nd.cumsum(ndScanIn, axis: 1),
              reason: 'cumsum',
            );
            _expectMatchNDArray(
              gCumprod,
              nd.cumprod(ndScanIn, axis: 1),
              reason: 'cumprod',
            );
            _expectMatchNDArray(
              gDiff,
              nd.diff(ndScanIn, n: 1, axis: 1),
              reason: 'diff',
            );

            // 5. Statistical & NaN-aware reductions
            _expectMatchNDArray(
              variance(gpuF32A, axis: 1),
              nd.var_(ndF32A, axis: 1),
              reason: 'variance',
            );
            _expectMatchNDArray(
              std(gpuF32A, axis: 1),
              nd.std(ndF32A, axis: 1),
              reason: 'std',
            );
            _expectMatchNDArray(
              ptp(gpuF32A, axis: 1),
              nd.ptp(ndF32A, axis: 1),
              reason: 'ptp',
            );

            final ndNanArr = nd.NDArray<Float32>.fromList(
              [1.0, double.nan, 3.0, 4.0, 2.0, double.nan],
              [2, 3],
              nd.DType.float32,
            );
            final gpuNanArr = GpuArray<Float32>.fromNDArray(
              ndNanArr,
              device: device,
            );
            _expectMatchNDArray(
              nansum(gpuNanArr, axis: 1),
              nd.nansum(ndNanArr, axis: 1),
              reason: 'nansum',
            );
            _expectMatchNDArray(
              nanmean(gpuNanArr, axis: 1),
              nd.nanmean(ndNanArr, axis: 1),
              reason: 'nanmean',
            );
            _expectMatchNDArray(
              nanmin(gpuNanArr, axis: 1),
              nd.nanmin(ndNanArr, axis: 1),
              reason: 'nanmin',
            );
            _expectMatchNDArray(
              nanmax(gpuNanArr, axis: 1),
              nd.nanmax(ndNanArr, axis: 1),
              reason: 'nanmax',
            );

            // 6. Elementwise, bitwise & complex ufuncs
            _expectMatchNDArray(
              clip(gpuF32A, 5.0, 20.0),
              nd.clip(ndF32A, min: 5.0, max: 20.0),
              reason: 'clip',
            );
            _expectMatchNDArray(
              sign(gpuF32A - 10.0),
              nd.sign(ndF32A - 10.0),
              reason: 'sign',
            );
            _expectMatchNDArray(
              atan2(gpuF32A, gpuF32B),
              nd.atan2(ndF32A, ndF32B),
              reason: 'atan2',
            );
            _expectMatchNDArray(
              hypot(gpuF32A, gpuF32B),
              nd.hypot(ndF32A, ndF32B),
              reason: 'hypot',
            );

            final ndSpec = nd.NDArray<Float32>.fromList(
              [1.0, double.nan, double.infinity, double.negativeInfinity],
              [4],
              nd.DType.float32,
            );
            final gpuSpec = GpuArray<Float32>.fromNDArray(
              ndSpec,
              device: device,
            );
            _expectMatchNDArray(
              isnan(gpuSpec),
              nd.isnan(ndSpec),
              reason: 'isnan',
            );
            _expectMatchNDArray(
              isinf(gpuSpec),
              nd.isinf(ndSpec),
              reason: 'isinf',
            );
            _expectMatchNDArray(
              isfinite(gpuSpec),
              nd.isfinite(ndSpec),
              reason: 'isfinite',
            );
            _expectMatchNDArray(
              nanToNum(gpuSpec, nan: 0.0, posinf: 99.0, neginf: -99.0),
              nd.nan_to_num(ndSpec, nan: 0.0, posinf: 99.0, neginf: -99.0),
              reason: 'nanToNum',
            );
            expect(
              isClose(gpuF32A, gpuF32A + 1e-6, atol: 1e-4).toList(),
              equals([true, true, true, true]),
            );
            expect(allClose(gpuF32A, gpuF32A + 1e-6, atol: 1e-4), isTrue);

            // Bitwise ufuncs
            final ndBitA = nd.NDArray<Int32>.fromList(
              [6, 12, 15],
              [3],
              nd.DType.int32,
            );
            final ndBitB = nd.NDArray<Int32>.fromList(
              [3, 5, 7],
              [3],
              nd.DType.int32,
            );
            final gpuBitA = GpuArray<Int32>.fromNDArray(ndBitA, device: device);
            final gpuBitB = GpuArray<Int32>.fromNDArray(ndBitB, device: device);
            _expectMatchNDArray(
              gpuBitA & gpuBitB,
              ndBitA & ndBitB,
              reason: 'bitwise &',
            );
            _expectMatchNDArray(
              gpuBitA | gpuBitB,
              ndBitA | ndBitB,
              reason: 'bitwise |',
            );
            _expectMatchNDArray(
              gpuBitA ^ gpuBitB,
              ndBitA ^ ndBitB,
              reason: 'bitwise ^',
            );
            _expectMatchNDArray(~gpuBitA, ~ndBitA, reason: 'bitwise ~');
            _expectMatchNDArray(
              gpuBitA << 1,
              ndBitA << 1,
              reason: 'bitwise <<',
            );
            _expectMatchNDArray(
              gpuBitA >> 1,
              ndBitA >> 1,
              reason: 'bitwise >>',
            );

            // Complex ufuncs (real, imag, conj, angle) on Complex64
            final ndC64 = nd.NDArray<Complex64>.fromList(
              [nd.Complex(1.0, 1.0), nd.Complex(0.0, -2.0)],
              [2],
              nd.DType.complex64,
            );
            final gpuC64 = GpuArray<Complex64>.fromNDArray(
              ndC64,
              device: device,
            );
            final GpuArray<Float32> gReal = gpuC64.real();
            final GpuArray<Float32> gImag = gpuC64.imag();
            final GpuArray<Complex64> gConj = conj(gpuC64);
            final GpuArray<Float32> gAngle = gpuC64.angle();
            expect(gReal.dtype, equals(DType.float32));
            expect(gImag.dtype, equals(DType.float32));
            expect(gConj.dtype, equals(DType.complex64));
            expect(gAngle.dtype, equals(DType.float32));
            _expectMatchNDArray(gReal, nd.real(ndC64), reason: 'real');
            _expectMatchNDArray(gImag, nd.imag(ndC64), reason: 'imag');
            _expectMatchNDArray(gConj, nd.conj(ndC64), reason: 'conj');
            _expectMatchNDArray(
              gAngle,
              nd.angle(ndC64),
              tol: 1e-3,
              reason: 'angle',
            );
          });
        });
      },
    );

    test(
      'R1.4: Native Float32 and Complex64 linalg and FFT DTypeSpec projections vs NDArray oracle',
      () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final ndSpd32 = nd.NDArray<Float32>.fromList(
              [4.0, 1.0, 1.0, 3.0],
              [2, 2],
              nd.DType.float32,
            );
            final ndRhs32 = nd.NDArray<Float32>.fromList(
              [1.0, 2.0],
              [2],
              nd.DType.float32,
            );
            final gpuSpd32 = GpuArray<Float32>.fromNDArray(
              ndSpd32,
              device: device,
            );
            final gpuRhs32 = GpuArray<Float32>.fromNDArray(
              ndRhs32,
              device: device,
            );

            // Linalg decompositions & solvers on Float32
            final svdRes = gpu_linalg.svd(gpuSpd32);
            final GpuArray<Float32> sVals = gpu_linalg.svdValues(gpuSpd32);
            final qrRes = gpu_linalg.qr(gpuSpd32);
            final GpuArray<Float32> cholRes = gpu_linalg.cholesky(gpuSpd32);
            final eighRes = gpu_linalg.eigh(gpuSpd32);
            final GpuArray<Float32> eigvalshRes = gpu_linalg.eigvalsh(gpuSpd32);
            final luRes = gpu_linalg.lu(gpuSpd32);
            final luFac = gpu_linalg.luFactor(gpuSpd32);
            final GpuArray<Float32> luSol = gpu_linalg.luSolve(
              luFac.lu,
              luFac.pivots,
              gpuRhs32,
            );
            final GpuArray<Float32> sol = gpu_linalg.solve(gpuSpd32, gpuRhs32);
            final GpuArray<Float32> invRes = gpu_linalg.inv(gpuSpd32);
            final GpuArray<Float32> pinvRes = gpu_linalg.pinv(gpuSpd32);
            final lstsqRes = gpu_linalg.lstsq(gpuSpd32, gpuRhs32);
            final GpuArray<Float32> detRes = gpu_linalg.det(gpuSpd32);
            final slogdetRes = gpu_linalg.slogdet(gpuSpd32);
            final GpuArray<Float32> normRes = gpu_linalg.norm(gpuSpd32);
            final GpuArray<Float32> condRes = gpu_linalg.cond(gpuSpd32);

            expect(svdRes.u.dtype, equals(DType.float32));
            expect(svdRes.s.dtype, equals(DType.float32));
            expect(svdRes.vt.dtype, equals(DType.float32));
            expect(sVals.dtype, equals(DType.float32));
            expect(qrRes.q.dtype, equals(DType.float32));
            expect(qrRes.r.dtype, equals(DType.float32));
            expect(cholRes.dtype, equals(DType.float32));
            expect(eighRes.eigenvalues.dtype, equals(DType.float32));
            expect(eighRes.eigenvectors.dtype, equals(DType.float32));
            expect(eigvalshRes.dtype, equals(DType.float32));
            expect(luRes.p.dtype, equals(DType.float32));
            expect(luRes.l.dtype, equals(DType.float32));
            expect(luRes.u.dtype, equals(DType.float32));
            expect(luFac.lu.dtype, equals(DType.float32));
            expect(luSol.dtype, equals(DType.float32));
            expect(sol.dtype, equals(DType.float32));
            expect(invRes.dtype, equals(DType.float32));
            expect(pinvRes.dtype, equals(DType.float32));
            expect(lstsqRes.solution.dtype, equals(DType.float32));
            expect(detRes.dtype, equals(DType.float32));
            expect(slogdetRes.sign.dtype, equals(DType.float32));
            expect(slogdetRes.logabsdet.dtype, equals(DType.float32));
            expect(normRes.dtype, equals(DType.float32));
            expect(condRes.dtype, equals(DType.float32));

            _expectMatchNDArray(
              sVals,
              nd.svd(ndSpd32).s,
              tol: 1e-4,
              reason: 'f32 svdValues',
            );
            _expectMatchNDArray(
              cholRes,
              nd.cholesky(ndSpd32),
              tol: 1e-4,
              reason: 'f32 cholesky',
            );
            _expectMatchNDArray(
              eigvalshRes,
              nd.eigvalsh(ndSpd32),
              tol: 1e-4,
              reason: 'f32 eigvalsh',
            );
            _expectMatchNDArray(
              sol,
              nd.solve(ndSpd32, ndRhs32),
              tol: 1e-4,
              reason: 'f32 solve',
            );
            _expectMatchNDArray(
              luSol,
              nd.solve(ndSpd32, ndRhs32),
              tol: 1e-4,
              reason: 'f32 luSolve',
            );
            _expectMatchNDArray(
              invRes,
              nd.inv(ndSpd32),
              tol: 1e-4,
              reason: 'f32 inv',
            );
            _expectMatchNDArray(
              pinvRes,
              nd.pinv(ndSpd32),
              tol: 1e-4,
              reason: 'f32 pinv',
            );
            _expectMatchNDArray(
              lstsqRes.solution,
              nd.lstsq(ndSpd32, ndRhs32).x,
              tol: 1e-4,
              reason: 'f32 lstsq',
            );
            _expectMatchNDArray(
              detRes,
              nd.det(ndSpd32),
              tol: 1e-4,
              reason: 'f32 det',
            );
            _expectMatchNDArray(
              normRes,
              nd.norm(ndSpd32),
              tol: 1e-4,
              reason: 'f32 norm',
            );
            _expectMatchNDArray(
              condRes,
              nd.cond(ndSpd32),
              tol: 1e-3,
              reason: 'f32 cond',
            );

            // FFT on Float32 & Complex64 (fft, ifft, rfft, irfft, fft2, ifft2, fftn, ifftn)
            final ndSig32 = nd.NDArray<Float32>.fromList(
              [1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.5, -1.0],
              [2, 4],
              nd.DType.float32,
            );
            final gpuSig32 = GpuArray<Float32>.fromNDArray(
              ndSig32,
              device: device,
            );
            final GpuArray<Complex64> gFft = gpu_fft.fft(gpuSig32);
            final GpuArray<Complex64> gIfft = gpu_fft.ifft(gFft);
            final GpuArray<Complex64> gRfft = gpu_fft.rfft(gpuSig32);
            final GpuArray<Float32> gIrfft = gpu_fft.irfft(gRfft, n: 4);
            final GpuArray<Complex64> gFft2 = gpu_fft.fft2(gpuSig32);
            final GpuArray<Complex64> gIfft2 = gpu_fft.ifft2(gFft2);
            final GpuArray<Complex64> gFftn = gpu_fft.fftn(gpuSig32);
            final GpuArray<Complex64> gIfftn = gpu_fft.ifftn(gFftn);

            expect(gFft.dtype, equals(DType.complex64));
            expect(gIfft.dtype, equals(DType.complex64));
            expect(gRfft.dtype, equals(DType.complex64));
            expect(gIrfft.dtype, equals(DType.float32));
            expect(gFft2.dtype, equals(DType.complex64));
            expect(gIfft2.dtype, equals(DType.complex64));
            expect(gFftn.dtype, equals(DType.complex64));
            expect(gIfftn.dtype, equals(DType.complex64));

            _expectMatchNDArray(
              gFft,
              nd.fft(ndSig32),
              tol: 1e-4,
              reason: 'f32 fft',
            );
            _expectMatchNDArray(
              gRfft,
              nd.rfft(ndSig32),
              tol: 1e-4,
              reason: 'f32 rfft',
            );
            _expectMatchNDArray(
              gIrfft,
              ndSig32,
              tol: 1e-4,
              reason: 'f32 irfft',
            );
            _expectMatchNDArray(
              gFft2,
              nd.fft2(ndSig32),
              tol: 1e-4,
              reason: 'f32 fft2',
            );
            _expectMatchNDArray(
              gFftn,
              nd.fftn(ndSig32),
              tol: 1e-4,
              reason: 'f32 fftn',
            );
            _expectMatchNDArray(
              gIfft.real(),
              ndSig32,
              tol: 1e-4,
              reason: 'f32 ifft real',
            );
            _expectMatchNDArray(
              gIfft2.real(),
              ndSig32,
              tol: 1e-4,
              reason: 'f32 ifft2 real',
            );
            _expectMatchNDArray(
              gIfftn.real(),
              ndSig32,
              tol: 1e-4,
              reason: 'f32 ifftn real',
            );
          });
        });
      },
    );

    test(
      'Cross-cutting behavioral invariants: 0-D scalars, empty arrays, strided/negative-stride views, out: aliasing, and ResourceScope zero-leak',
      () {
        final baselineActive = device.activeBufferCount;
        ResourceScope.scope(() {
          // 1. 0-D scalars
          final s1 = GpuArray<Float32>.fromList(
            [3.0],
            const [],
            DType.float32,
            device: device,
          );
          final s2 = GpuArray<Float32>.fromList(
            [4.0],
            const [],
            DType.float32,
            device: device,
          );
          final sAdd = s1 + s2;
          final sHyp = hypot(s1, s2);
          final sMean = s1.mean();
          expect(sAdd.shape, isEmpty);
          expect(sAdd.scalar, closeTo(7.0, 1e-5));
          expect(sHyp.shape, isEmpty);
          expect(sHyp.scalar, closeTo(5.0, 1e-5));
          expect(sMean.shape, isEmpty);
          expect(sMean.scalar, closeTo(3.0, 1e-5));

          // 2. Empty arrays
          final empty1D = GpuArray<Float32>.zeros(
            [0],
            DType.float32,
            device: device,
          );
          final emptySum = empty1D.sum();
          final emptySort = sort(empty1D);
          final emptyArgsort = argsort(empty1D);
          final emptyCumsum = cumsum(empty1D);
          final emptyUniq = unique(empty1D);
          final emptyInt = GpuArray<Int32>.zeros(
            [0],
            DType.int32,
            device: device,
          );
          final emptyBin = bincount(emptyInt, minlength: 3);
          expect(emptySum.scalar, equals(0.0));
          expect(emptySort.shape, equals([0]));
          expect(emptyArgsort.shape, equals([0]));
          expect(emptyCumsum.shape, equals([0]));
          expect(emptyUniq.shape, equals([0]));
          expect(emptyBin.toList(), equals([0, 0, 0]));

          // 3. Non-contiguous views (transposed, step-sliced, negative-stride)
          final ndBase = nd.NDArray<Float32>.fromList(
            [3.0, -1.0, 4.0, 2.0, 5.0, 0.0, -2.0, 6.0],
            [2, 4],
            nd.DType.float32,
          );
          final gpuBase = GpuArray<Float32>.fromNDArray(ndBase, device: device);
          try {
            final gpuFlip = gpuBase.slice([
              const Slice.all(),
              const Slice(3, null, -1),
            ]);
            final ndFlip = ndBase.slice([
              nd.Slice.all(),
              nd.Slice(start: 3, step: -1),
            ]);
            _expectMatchNDArray(
              sort(gpuFlip, axis: 1),
              nd.sort(ndFlip, axis: 1),
              reason: 'negative-stride sort',
            );
            _expectMatchNDArray(
              cumsum(gpuFlip, axis: 1),
              nd.cumsum(ndFlip, axis: 1),
              reason: 'negative-stride cumsum',
            );
            _expectMatchNDArray(
              diff(gpuFlip, axis: 1),
              nd.diff(ndFlip, axis: 1),
              reason: 'negative-stride diff',
            );
            _expectMatchNDArray(
              variance(gpuFlip, axis: 1),
              nd.var_(ndFlip, axis: 1),
              reason: 'negative-stride variance',
            );
            _expectMatchNDArray(
              clip(gpuFlip, 0.0, 4.0),
              nd.clip(ndFlip, min: 0.0, max: 4.0),
              reason: 'negative-stride clip',
            );
          } finally {
            ndBase.dispose();
          }

          // 4. out: aliasing & non-contiguous out: on R4 ops
          final aliasArr = GpuArray<Float32>.fromList(
            [-2.0, 0.5, 4.0],
            [3],
            DType.float32,
            device: device,
          );
          expect(
            identical(clip(aliasArr, 0.0, 2.0, out: aliasArr), aliasArr),
            isTrue,
          );
          expect(aliasArr.toList(), equals([0.0, 0.5, 2.0]));

          final fullOut = GpuArray<Float32>.filled(
            [6],
            -1.0,
            DType.float32,
            device: device,
          );
          final stridedOut = fullOut.slice([const Slice(0, 6, 2)]);
          final scanSrc = GpuArray<Float32>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float32,
            device: device,
          );
          expect(
            identical(cumsum(scanSrc, out: stridedOut), stridedOut),
            isTrue,
          );
          expect(fullOut.toList(), equals([1.0, -1.0, 3.0, -1.0, 6.0, -1.0]));
        });
        expect(device.activeBufferCount, equals(baselineActive));
      },
    );
  });

  group('Universal out: parameter contracts', () {
    test('Contract 1: out parameter returns identical instance', () {
      ResourceScope.scope(() {
        final a = GpuArray<Float32>.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float32,
          device: device,
        );
        final b = GpuArray<Float32>.fromList(
          [4.0, 5.0, 6.0],
          [3],
          DType.float32,
          device: device,
        );
        final out = GpuArray<Float32>.zeros([3], DType.float32, device: device);

        expect(identical(a.add(b, out: out), out), isTrue);
        expect(identical(a.subtract(b, out: out), out), isTrue);
        expect(identical(a.multiply(b, out: out), out), isTrue);
        expect(identical(a.divide(b, out: out), out), isTrue);
        expect(identical(a.exp(out: out), out), isTrue);
        expect(identical(a.sin(out: out), out), isTrue);

        final sumOut = GpuArray<Float32>.zeros(
          [],
          DType.float32,
          device: device,
        );
        expect(identical(a.sum(out: sumOut), sumOut), isTrue);
      });
    });

    test(
      'Contract 2: writing into strided out view preserves non-selected elements',
      () {
        ResourceScope.scope(() {
          final fullOut = GpuArray<Float32>.filled(
            [3, 4],
            -99.0,
            DType.float32,
            device: device,
          );
          // Select every other column of row 1 -> shape [2]
          final stridedOut = fullOut.slice([1, const Slice(0, 4, 2)]);
          final a = GpuArray<Float32>.fromList(
            [10.0, 20.0],
            [2],
            DType.float32,
            device: device,
          );
          final b = GpuArray<Float32>.fromList(
            [5.0, 7.0],
            [2],
            DType.float32,
            device: device,
          );

          final res = a.add(b, out: stridedOut);
          expect(identical(res, stridedOut), isTrue);
          expect(stridedOut.toList(), equals([15.0, 27.0]));

          // Verify untouched elements remain -99.0
          final fullNested = fullOut.toNestedList();
          expect(
            fullNested,
            equals([
              [-99.0, -99.0, -99.0, -99.0],
              [15.0, -99.0, 27.0, -99.0],
              [-99.0, -99.0, -99.0, -99.0],
            ]),
          );
        });
      },
    );

    test(
      'Contract 3: in-place execution (out: a) produces accurate results',
      () {
        ResourceScope.scope(() {
          final a = GpuArray<Float32>.fromList(
            [2.0, 4.0, 6.0],
            [3],
            DType.float32,
            device: device,
          );
          final b = GpuArray<Float32>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float32,
            device: device,
          );
          a.add(b, out: a);
          expect(a.toList(), equals([3.0, 6.0, 9.0]));
          a.multiply(b, out: a);
          expect(a.toList(), equals([3.0, 12.0, 27.0]));
        });
      },
    );

    test(
      'Contract 4: StateError on disposed array/out precedes ArgumentError on shape mismatch',
      () {
        ResourceScope.scope(() {
          final valid = GpuArray<Float32>.zeros(
            [3],
            DType.float32,
            device: device,
          );
          final wrongShapeDisposed = GpuArray<Float32>.zeros(
            [5],
            DType.float32,
            device: device,
          )..dispose();

          // Even though shape [5] mismatches [3], StateError must be thrown first!
          expect(
            () => valid.add(valid, out: wrongShapeDisposed),
            throwsStateError,
          );
          expect(() => valid.exp(out: wrongShapeDisposed), throwsStateError);
          expect(() => valid.sum(out: wrongShapeDisposed), throwsStateError);
          expect(() => wrongShapeDisposed.add(valid), throwsStateError);
        });
      },
    );

    test('Contract 5: read-only broadcast view rejected as out:', () {
      ResourceScope.scope(() {
        final base = GpuArray<Float32>.fromList(
          [1.0],
          [1],
          DType.float32,
          device: device,
        );
        final broadcasted = base.broadcastTo([4]);
        final input = GpuArray<Float32>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float32,
          device: device,
        );
        expect(
          () => input.exp(out: broadcasted),
          throwsA(
            anyOf(
              isA<UnsupportedError>(),
              isA<ArgumentError>(),
              isA<StateError>(),
            ),
          ),
        );
      });
    });

    test(
      'R4.2: 5-rule out:, strided view, error, and ResourceScope contracts across all 5 domains (core, indexing, manipulation, linalg, fft)',
      () {
        final otherDevice = GpuDevice.create(name: 'R42-OtherDevice');
        otherDevice.detachFromScope();
        try {
          ResourceScope.scope(() {
            final base2x4 = GpuArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
              [2, 4],
              DType.float64,
              device: device,
            );
            // Non-contiguous strided view with step=2, offset=1 -> shape [2, 2]
            // Elements: [[2.0, 4.0], [6.0, 8.0]]
            final stridedView = base2x4.slice([
              const Slice.all(),
              const Slice(1, 4, 2),
            ]);
            final contiguousCopy = GpuArray<Float64>.fromList(
              [2.0, 4.0, 6.0, 8.0],
              [2, 2],
              DType.float64,
              device: device,
            );
            final condView = base2x4
                .slice([const Slice.all(), const Slice(0, 3, 2)])
                .greater(2.0);
            final condContig = condView.copy();

            // --- Rule 1 (identity & mutation), Rule 4 (strided view parity), & Rule 5 (ResourceScope tracking vs outer out:) ---
            final outCore = GpuArray<Float64>.zeros(
              [2, 2],
              DType.float64,
              device: device,
            );
            final outWhere = GpuArray<Float64>.zeros(
              [2, 2],
              DType.float64,
              device: device,
            );
            final outTile = GpuArray<Float64>.zeros(
              [2, 4],
              DType.float64,
              device: device,
            );
            final outInv = GpuArray<Float64>.zeros(
              [2, 2],
              DType.float64,
              device: device,
            );
            final outFft = GpuArray<Complex128>.zeros(
              [2, 2],
              DType.complex128,
              device: device,
            );

            late GpuArray<DTypeTag> innerCoreAlloc;
            late GpuArray<Float64> innerWhereAlloc;
            late GpuArray<Float64> innerTileAlloc;
            late GpuArray<Float64> innerInvAlloc;
            late GpuArray<Complex128> innerFftAlloc;

            ResourceScope.scope(() {
              // Allocate inside inner scope (Rule 5: should dispose on scope exit)
              innerCoreAlloc = contiguousCopy.add(contiguousCopy);
              innerWhereAlloc = where<Float64>(
                condContig,
                contiguousCopy,
                contiguousCopy,
              );
              innerTileAlloc = tile<Float64>(contiguousCopy, [1, 2]);
              innerInvAlloc = gpu_linalg.inv(contiguousCopy);
              innerFftAlloc = gpu_fft.fft(contiguousCopy);

              // Pass outer out: into inner scope (Rule 1 & Rule 5: identical & NOT disposed by inner scope)
              expect(
                identical(stridedView.add(stridedView, out: outCore), outCore),
                isTrue,
              );
              expect(
                identical(
                  where<Float64>(
                    condView,
                    stridedView,
                    stridedView,
                    out: outWhere,
                  ),
                  outWhere,
                ),
                isTrue,
              );
              expect(
                identical(
                  tile<Float64>(stridedView, [1, 2], out: outTile),
                  outTile,
                ),
                isTrue,
              );
              expect(
                identical(gpu_linalg.inv(stridedView, out: outInv), outInv),
                isTrue,
              );
              expect(
                identical(gpu_fft.fft(stridedView, out: outFft), outFft),
                isTrue,
              );

              // Rule 4: Strided view outputs match contiguous outputs
              expect(outCore.toList(), equals(innerCoreAlloc.toList()));
              expect(outWhere.toList(), equals(innerWhereAlloc.toList()));
              expect(outTile.toList(), equals(innerTileAlloc.toList()));
              for (var i = 0; i < 4; i++) {
                expect(
                  (outInv.toList()[i] as num).toDouble(),
                  closeTo((innerInvAlloc.toList()[i] as num).toDouble(), 1e-6),
                );
              }
              for (var i = 0; i < 4; i++) {
                final a = outFft.toList()[i] as nd.Complex;
                final b = innerFftAlloc.toList()[i] as nd.Complex;
                expect(a.real, closeTo(b.real, 1e-6));
                expect(a.imag, closeTo(b.imag, 1e-6));
              }
            });

            // Rule 5 verification after inner scope exits:
            expect(innerCoreAlloc.isDisposed, isTrue);
            expect(innerWhereAlloc.isDisposed, isTrue);
            expect(innerTileAlloc.isDisposed, isTrue);
            expect(innerInvAlloc.isDisposed, isTrue);
            expect(innerFftAlloc.isDisposed, isTrue);
            expect(outCore.isDisposed, isFalse);
            expect(outWhere.isDisposed, isFalse);
            expect(outTile.isDisposed, isFalse);
            expect(outInv.isDisposed, isFalse);
            expect(outFft.isDisposed, isFalse);

            // --- Rule 2: out: shape/dtype/device mismatch throws ArgumentError starting with 'Must ' ---
            final wrongShapeF64 = GpuArray<Float64>.zeros(
              [3, 3],
              DType.float64,
              device: device,
            );
            final wrongShapeC128 = GpuArray<Complex128>.zeros(
              [3, 3],
              DType.complex128,
              device: device,
            );
            final wrongDTypeI32 = GpuArray<Int32>.zeros(
              [2, 2],
              DType.int32,
              device: device,
            );
            final wrongDeviceF64 = GpuArray<Float64>.zeros(
              [2, 2],
              DType.float64,
              device: otherDevice,
            );
            final wrongDeviceC128 = GpuArray<Complex128>.zeros(
              [2, 2],
              DType.complex128,
              device: otherDevice,
            );

            // Shape mismatch across all 5 domains
            _expectArgumentErrorMust(
              () => contiguousCopy.add(contiguousCopy, out: wrongShapeF64),
              reason: 'core shape mismatch',
            );
            _expectArgumentErrorMust(
              () => where<Float64>(
                condContig,
                contiguousCopy,
                contiguousCopy,
                out: wrongShapeF64,
              ),
              reason: 'indexing shape mismatch',
            );
            _expectArgumentErrorMust(
              () => tile<Float64>(contiguousCopy, [1, 2], out: wrongShapeF64),
              reason: 'manipulation shape mismatch',
            );
            _expectArgumentErrorMust(
              () => gpu_linalg.inv(contiguousCopy, out: wrongShapeF64),
              reason: 'linalg shape mismatch',
            );
            _expectArgumentErrorMust(
              () => gpu_fft.fft(contiguousCopy, out: wrongShapeC128),
              reason: 'fft shape mismatch',
            );

            // DType mismatch & Device mismatch across all 5 domains
            _expectArgumentErrorMust(
              () => (contiguousCopy as GpuArray<DTypeTag>).add(
                contiguousCopy,
                out: wrongDTypeI32,
              ),
              reason: 'core dtype mismatch',
            );
            _expectArgumentErrorMust(
              () => contiguousCopy.add(contiguousCopy, out: wrongDeviceF64),
              reason: 'core device mismatch',
            );
            _expectArgumentErrorMust(
              () => where<Float64>(
                condContig,
                contiguousCopy,
                contiguousCopy,
                out: wrongDeviceF64,
              ),
              reason: 'indexing device mismatch',
            );
            _expectArgumentErrorMust(
              () => tile<Float64>(contiguousCopy, [1, 1], out: wrongDeviceF64),
              reason: 'manipulation device mismatch',
            );
            _expectArgumentErrorMust(
              () => gpu_linalg.inv(contiguousCopy, out: wrongDeviceF64),
              reason: 'linalg device mismatch',
            );
            _expectArgumentErrorMust(
              () => gpu_fft.fft(contiguousCopy, out: wrongDeviceC128),
              reason: 'fft device mismatch',
            );

            // --- Rule 3: Disposed input or disposed out: throws StateError across all 5 domains ---
            final disposedF64 = GpuArray<Float64>.zeros(
              [2, 2],
              DType.float64,
              device: device,
            )..dispose();
            final disposedC128 = GpuArray<Complex128>.zeros(
              [2, 2],
              DType.complex128,
              device: device,
            )..dispose();

            expect(() => disposedF64.add(contiguousCopy), throwsStateError);
            expect(
              () => contiguousCopy.add(contiguousCopy, out: disposedF64),
              throwsStateError,
            );
            expect(
              () => where<Float64>(condContig, disposedF64, contiguousCopy),
              throwsStateError,
            );
            expect(
              () => where<Float64>(
                condContig,
                contiguousCopy,
                contiguousCopy,
                out: disposedF64,
              ),
              throwsStateError,
            );
            expect(() => tile<Float64>(disposedF64, [1, 2]), throwsStateError);
            expect(
              () => tile<Float64>(contiguousCopy, [1, 1], out: disposedF64),
              throwsStateError,
            );
            expect(() => gpu_linalg.inv(disposedF64), throwsStateError);
            expect(
              () => gpu_linalg.inv(contiguousCopy, out: disposedF64),
              throwsStateError,
            );
            expect(() => gpu_fft.fft(disposedF64), throwsStateError);
            expect(
              () => gpu_fft.fft(contiguousCopy, out: disposedC128),
              throwsStateError,
            );
          });
        } finally {
          otherDevice.dispose();
        }
      },
    );
  });

  group('15-DType sweep & reified generics', () {
    test(
      'All 15 DType.values round-trip with NDArray and preserve reified generic type',
      () {
        for (final dtype in DType.values) {
          ResourceScope.scope(() {
            final arr = GpuArray.zeros([2, 2], dtype, device: device);
            expect(arr.dtype, equals(dtype));
            expect(arr.shape, equals([2, 2]));
            expect(
              arr.runtimeType.toString(),
              isNot(equals('GpuArray<DTypeTag>')),
            );

            final ndArr = arr.toNDArray();
            try {
              final roundTrip = GpuArray.fromNDArray(ndArr, device: device);
              expect(roundTrip.dtype, equals(dtype));
              expect(roundTrip.shape, equals([2, 2]));
              expect(
                roundTrip.runtimeType.toString(),
                equals(arr.runtimeType.toString()),
              );
            } finally {
              ndArr.dispose();
            }
          });
        }
      },
    );
  });

  group('Error hierarchy contracts', () {
    test(
      'StateError, RangeError, ArgumentError, and FormatException are thrown appropriately',
      () {
        ResourceScope.scope(() {
          final arr = GpuArray<Float32>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float32,
            device: device,
          );

          // StateError on non-0D .scalar
          expect(() => arr.scalar, throwsStateError);

          // RangeError on out-of-bounds slice or axis
          expect(() => arr.slice([5, 0]), throwsRangeError);
          expect(() => arr.sum(axis: 3), throwsRangeError);
          expect(() => gpu_fft.fft(arr, axis: 5), throwsRangeError);

          // ArgumentError on incompatible shapes or parameters
          expect(() => arr.reshape([5]), throwsArgumentError);
          final vec3 = GpuArray<Float32>.zeros(
            [3],
            DType.float32,
            device: device,
          );
          expect(() => arr.add(vec3), throwsArgumentError);
          expect(
            () => gpu_linalg.inv(arr.reshape([1, 4])),
            throwsArgumentError,
          );

          // StateError on GpuArray.backward() when requiresGrad is false;
          // ArgumentError on non-scalar GpuArray.backward() without gradient
          expect(() => arr.backward(), throwsStateError);
          final v = GpuArray<Float32>.fromList(
            [1.0, 2.0],
            [2],
            DType.float32,
            device: device,
            requiresGrad: true,
          );
          expect(() => v.backward(), throwsArgumentError);

          // FormatException on malformed external data (Safetensors, Pipeline JSON)
          // and structured validation errors on invalid WGSL
          expect(
            () =>
                loadSafetensors(Uint8List.fromList([1, 2, 3]), device: device),
            throwsFormatException,
          );
          expect(
            WgslSyntaxValidator.validate('invalid wgsl code').isValid,
            isFalse,
          );
          expect(
            () => GpuComputePipelinePackage.fromJson({'invalid': 'json'}),
            throwsFormatException,
          );
        });
      },
    );
  });
}
