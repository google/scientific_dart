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
  final actualList = actual.toList();
  final flatExpected = expected.reshape([expected.size]);
  for (var i = 0; i < actual.size; i++) {
    final a = actualList[i];
    final e = flatExpected.getCell([i]);
    if (a is num && e is num) {
      if (a.isNaN && e.isNaN) continue;
      if (a.isInfinite && e.isInfinite && a.sign == e.sign) continue;
      expect(a.toDouble(), closeTo(e.toDouble(), tol), reason: '$reason [$i]');
    } else if (a is nd.Complex && e is nd.Complex) {
      expect(a.real, closeTo(e.real, tol), reason: '$reason [$i].real');
      expect(a.imag, closeTo(e.imag, tol), reason: '$reason [$i].imag');
    } else {
      expect(a, equals(e), reason: '$reason [$i]');
    }
  }
}

void main() {
  late GpuDevice device;

  setUpAll(() {
    device = GpuDevice.cpu();
    device.detachFromScope();
  });

  tearDownAll(() {
    device.dispose();
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
