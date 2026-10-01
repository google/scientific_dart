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
import 'package:gpuarray/linalg.dart' as gpu_linalg;
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
              gpu_linalg.svdvals(gpuSpd),
              ndSvd.s,
              tol: 1e-5,
              reason: 'linalg svdvals',
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
              () => contiguousCopy.add(contiguousCopy, out: wrongDTypeI32),
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
