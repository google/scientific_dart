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
import 'package:ndarray/src/operations/io.dart' show parseNpyHeader;
import 'package:test/test.dart';

/// Descriptor for a unary operation `f(a, {out})` on `NDArray<Float64>`.
final class UnaryOpSpec {
  final String name;
  final NDArray<Float64> Function(NDArray<Float64> a, {NDArray<Float64>? out})
  call;
  final bool supportsInPlace;
  final bool supports0DAndEmpty;

  const UnaryOpSpec(
    this.name,
    this.call, {
    this.supportsInPlace = true,
    this.supports0DAndEmpty = true,
  });
}

/// Descriptor for a binary operation `f(a, b, {out})` on `NDArray<Float64>`.
final class BinaryOpSpec {
  final String name;
  final NDArray<Float64> Function(
    NDArray<Float64> a,
    NDArray<Float64> b, {
    NDArray<Float64>? out,
  })
  call;

  const BinaryOpSpec(this.name, this.call);
}

/// Descriptor for an axis reduction `f(a, {axis, keepdims, out})` on `NDArray<Float64>`.
final class ReductionOpSpec {
  final String name;
  final NDArray<Float64> Function(
    NDArray<Float64> a, {
    int? axis,
    bool keepdims,
    NDArray<Float64>? out,
  })
  call;

  const ReductionOpSpec(this.name, this.call);
}

/// Descriptor for a boolean-producing operation `f(a, {out})` on `NDArray<Float64>` -> `NDArray<Boolean>`.
final class BooleanResultOpSpec {
  final String name;
  final NDArray<Boolean> Function(NDArray<Float64> a, {NDArray<Boolean>? out})
  call;

  const BooleanResultOpSpec(this.name, this.call);
}

/// Descriptor for an index/count reduction `f(a, {axis, out})` -> `NDArray<Int64>`.
final class IndexReductionOpSpec {
  final String name;
  final NDArray<Int64> Function(
    NDArray<Float64> a, {
    int? axis,
    NDArray<Int64>? out,
  })
  call;

  const IndexReductionOpSpec(this.name, this.call);
}

/// Descriptor for a shape/view/copy operation `f(a)` asserting `sharesMemory(a, res) == expectedIsView`.
final class ShapeViewOrCopyOpSpec {
  final String name;
  final NDArray<Float64> Function(NDArray<Float64> a) call;
  final bool expectedIsView;

  const ShapeViewOrCopyOpSpec(
    this.name,
    this.call, {
    required this.expectedIsView,
  });
}

void main() {
  final unaryOps = <UnaryOpSpec>[
    UnaryOpSpec('sin', (a, {out}) => sin(a, out: out)),
    UnaryOpSpec('cos', (a, {out}) => cos(a, out: out)),
    UnaryOpSpec('tan', (a, {out}) => tan(a, out: out)),
    UnaryOpSpec('asin', (a, {out}) => asin(a * 0.25, out: out)),
    UnaryOpSpec('acos', (a, {out}) => acos(a * 0.25, out: out)),
    UnaryOpSpec('atan', (a, {out}) => atan(a, out: out)),
    UnaryOpSpec('sinh', (a, {out}) => sinh(a, out: out)),
    UnaryOpSpec('cosh', (a, {out}) => cosh(a, out: out)),
    UnaryOpSpec('tanh', (a, {out}) => tanh(a, out: out)),
    UnaryOpSpec('asinh', (a, {out}) => asinh(a, out: out)),
    UnaryOpSpec('acosh', (a, {out}) => acosh(a + 1.0, out: out)),
    UnaryOpSpec('atanh', (a, {out}) => atanh(a * 0.25, out: out)),
    UnaryOpSpec('exp', (a, {out}) => exp(a, out: out)),
    UnaryOpSpec('expm1', (a, {out}) => expm1(a, out: out)),
    UnaryOpSpec('log', (a, {out}) => log(a, out: out)),
    UnaryOpSpec('log2', (a, {out}) => log2(a, out: out)),
    UnaryOpSpec('log10', (a, {out}) => log10(a, out: out)),
    UnaryOpSpec('log1p', (a, {out}) => log1p(a, out: out)),
    UnaryOpSpec('sqrt', (a, {out}) => sqrt(a, out: out)),
    UnaryOpSpec('square', (a, {out}) => square(a, out: out)),
    UnaryOpSpec('reciprocal', (a, {out}) => reciprocal(a, out: out)),
    UnaryOpSpec('abs', (a, {out}) => abs(a, out: out)),
    UnaryOpSpec('negative', (a, {out}) => negative(a, out: out)),
    UnaryOpSpec('sign', (a, {out}) => sign(a, out: out)),
    UnaryOpSpec('floor', (a, {out}) => floor(a, out: out)),
    UnaryOpSpec('ceil', (a, {out}) => ceil(a, out: out)),
    UnaryOpSpec('round', (a, {out}) => round(a, out: out)),
    UnaryOpSpec('trunc', (a, {out}) => trunc(a, out: out)),
    UnaryOpSpec('rint', (a, {out}) => rint(a, out: out)),
    UnaryOpSpec('sinc', (a, {out}) => sinc(a, out: out)),
    UnaryOpSpec('i0', (a, {out}) => i0(a, out: out)),
    UnaryOpSpec('gamma', (a, {out}) => gamma(a, out: out)),
    UnaryOpSpec('erf', (a, {out}) => erf(a, out: out)),
    UnaryOpSpec('clip', (a, {out}) => clip(a, min: 0.8, max: 2.5, out: out)),
    UnaryOpSpec('nan_to_num', (a, {out}) => nan_to_num(a, out: out)),
    UnaryOpSpec(
      'cumsum',
      (a, {out}) => cumsum(a, axis: -1, out: out),
      supports0DAndEmpty: false,
    ),
    UnaryOpSpec(
      'cumprod',
      (a, {out}) => cumprod(a, axis: -1, out: out),
      supports0DAndEmpty: false,
    ),
    UnaryOpSpec('sort', (a, {out}) => sort(a, out: out), supportsInPlace: true),
    UnaryOpSpec(
      'roll(shift: 2)',
      (a, {out}) => roll(a, 2, out: out),
      supportsInPlace: false,
    ),
  ];

  final binaryOps = <BinaryOpSpec>[
    BinaryOpSpec('add', (a, b, {out}) => add(a, b, out: out)),
    BinaryOpSpec('subtract', (a, b, {out}) => subtract(a, b, out: out)),
    BinaryOpSpec('multiply', (a, b, {out}) => multiply(a, b, out: out)),
    BinaryOpSpec('divide', (a, b, {out}) => divide(a, b, out: out)),
    BinaryOpSpec('floorDivide', (a, b, {out}) => floorDivide(a, b, out: out)),
    BinaryOpSpec('power', (a, b, {out}) => power(a, b, out: out)),
    BinaryOpSpec('atan2', (a, b, {out}) => atan2(a, b, out: out)),
    BinaryOpSpec('hypot', (a, b, {out}) => hypot(a, b, out: out)),
    BinaryOpSpec('copysign', (a, b, {out}) => copysign(a, b, out: out)),
    BinaryOpSpec('fmod', (a, b, {out}) => fmod(a, b, out: out)),
    BinaryOpSpec('logaddexp', (a, b, {out}) => logaddexp(a, b, out: out)),
    BinaryOpSpec('logaddexp2', (a, b, {out}) => logaddexp2(a, b, out: out)),
    BinaryOpSpec(
      'minimum',
      (a, b, {out}) => binaryUfunc(a, b, op: BinaryOp.minimum, out: out),
    ),
    BinaryOpSpec(
      'maximum',
      (a, b, {out}) => binaryUfunc(a, b, op: BinaryOp.maximum, out: out),
    ),
    BinaryOpSpec(
      'fmin',
      (a, b, {out}) => binaryUfunc(a, b, op: BinaryOp.fmin, out: out),
    ),
    BinaryOpSpec(
      'fmax',
      (a, b, {out}) => binaryUfunc(a, b, op: BinaryOp.fmax, out: out),
    ),
  ];

  final reductionOps = <ReductionOpSpec>[
    ReductionOpSpec(
      'sum',
      (a, {axis, keepdims = false, out}) =>
          sum(a, axis: axis, keepdims: keepdims, out: out),
    ),
    ReductionOpSpec(
      'prod',
      (a, {axis, keepdims = false, out}) =>
          prod(a, axis: axis, keepdims: keepdims, out: out),
    ),
    ReductionOpSpec(
      'mean',
      (a, {axis, keepdims = false, out}) =>
          mean(a, axis: axis, keepdims: keepdims, out: out),
    ),
    ReductionOpSpec(
      'min',
      (a, {axis, keepdims = false, out}) =>
          min(a, axis: axis, keepdims: keepdims, out: out),
    ),
    ReductionOpSpec(
      'max',
      (a, {axis, keepdims = false, out}) =>
          max(a, axis: axis, keepdims: keepdims, out: out),
    ),
    ReductionOpSpec(
      'ptp',
      (a, {axis, keepdims = false, out}) =>
          ptp(a, axis: axis, keepdims: keepdims, out: out),
    ),
    ReductionOpSpec(
      'std',
      (a, {axis, keepdims = false, out}) =>
          std(a, axis: axis, keepdims: keepdims, out: out),
    ),
    ReductionOpSpec(
      'var_',
      (a, {axis, keepdims = false, out}) =>
          var_(a, axis: axis, keepdims: keepdims, out: out),
    ),
    ReductionOpSpec(
      'nansum',
      (a, {axis, keepdims = false, out}) =>
          nansum(a, axis: axis, keepdims: keepdims, out: out),
    ),
    ReductionOpSpec(
      'nanmean',
      (a, {axis, keepdims = false, out}) =>
          nanmean(a, axis: axis, keepdims: keepdims, out: out),
    ),
    ReductionOpSpec(
      'nanmin',
      (a, {axis, keepdims = false, out}) =>
          nanmin(a, axis: axis, keepdims: keepdims, out: out),
    ),
    ReductionOpSpec(
      'nanmax',
      (a, {axis, keepdims = false, out}) =>
          nanmax(a, axis: axis, keepdims: keepdims, out: out),
    ),
    ReductionOpSpec(
      'nanstd',
      (a, {axis, keepdims = false, out}) =>
          nanstd(a, axis: axis, keepdims: keepdims, out: out),
    ),
    ReductionOpSpec(
      'nanvar',
      (a, {axis, keepdims = false, out}) =>
          nanvar(a, axis: axis, keepdims: keepdims, out: out),
    ),
  ];

  final booleanResultOps = <BooleanResultOpSpec>[
    BooleanResultOpSpec('isnan', (a, {out}) => isnan(a, out: out)),
    BooleanResultOpSpec('isinf', (a, {out}) => isinf(a, out: out)),
    BooleanResultOpSpec('isfinite', (a, {out}) => isfinite(a, out: out)),
    BooleanResultOpSpec('logicalNot', (a, {out}) => logicalNot(a, out: out)),
    BooleanResultOpSpec('logicalAnd', (a, {out}) => logicalAnd(a, a, out: out)),
    BooleanResultOpSpec('logicalOr', (a, {out}) => logicalOr(a, a, out: out)),
    BooleanResultOpSpec('logicalXor', (a, {out}) => logicalXor(a, a, out: out)),
    BooleanResultOpSpec('equal', (a, {out}) => equal(a, a, out: out)),
    BooleanResultOpSpec('greater', (a, {out}) => greater(a, a * 0.5, out: out)),
    BooleanResultOpSpec('less', (a, {out}) => less(a * 0.5, a, out: out)),
  ];

  final indexReductionOps = <IndexReductionOpSpec>[
    IndexReductionOpSpec(
      'argmax',
      (a, {axis, out}) => argmax(a, axis: axis, out: out),
    ),
    IndexReductionOpSpec(
      'argmin',
      (a, {axis, out}) => argmin(a, axis: axis, out: out),
    ),
    IndexReductionOpSpec(
      'count_nonzero',
      (a, {axis, out}) => count_nonzero(a, axis: axis, out: out),
    ),
  ];

  final shapeViewOrCopyOps = <ShapeViewOrCopyOpSpec>[
    ShapeViewOrCopyOpSpec(
      'reshape (contiguous)',
      (a) => a.reshape([2, 6]),
      expectedIsView: true,
    ),
    ShapeViewOrCopyOpSpec(
      'transpose',
      (a) => a.transpose(),
      expectedIsView: true,
    ),
    ShapeViewOrCopyOpSpec('fliplr', (a) => fliplr(a), expectedIsView: true),
    ShapeViewOrCopyOpSpec(
      'expand_dims',
      (a) => expand_dims(a, 0),
      expectedIsView: true,
    ),
    ShapeViewOrCopyOpSpec(
      'squeeze',
      (a) => squeeze(expand_dims(a, 0), axis: [0]),
      expectedIsView: true,
    ),
    ShapeViewOrCopyOpSpec(
      'broadcastTo',
      (a) => broadcastTo(a, [2, 3, 4]),
      expectedIsView: true,
    ),
    ShapeViewOrCopyOpSpec('flip', (a) => flip(a), expectedIsView: true),
    ShapeViewOrCopyOpSpec('rot90', (a) => rot90(a), expectedIsView: true),
    ShapeViewOrCopyOpSpec(
      'ravel (contiguous)',
      (a) => a.ravel(),
      expectedIsView: true,
    ),
    ShapeViewOrCopyOpSpec('flatten', (a) => a.flatten(), expectedIsView: false),
    ShapeViewOrCopyOpSpec('copy', (a) => a.copy(), expectedIsView: false),
    ShapeViewOrCopyOpSpec('repeat', (a) => repeat(a, 2), expectedIsView: false),
    ShapeViewOrCopyOpSpec(
      'tile',
      (a) => tile(a, [2, 1]),
      expectedIsView: false,
    ),
  ];

  group('Unary Operation Contracts', () {
    for (final op in unaryOps) {
      group(op.name, () {
        test(
          'strided, negative-stride, interior offset, and transposed equivalence',
          () {
            NDArray.scope(() {
              final base = linspace(
                0.5,
                3.0,
                12,
                dtype: DType.float64,
              ).reshape([3, 4]);

              // 1. Reversed 1D view (negative stride)
              final flat = base.reshape([12]);
              final revView = flat.slice([Slice(step: -1)]);
              final revContig = revView.copy();
              expect(
                allClose(op.call(revView), op.call(revContig)),
                isTrue,
                reason: '${op.name} failed negative-stride equivalence',
              );

              // 2. Interior positive-stride subview (offsetElements > 0, stride > 0)
              final interiorView = base.slice([
                Slice(start: 1, stop: 3),
                Slice(start: 1, stop: 3),
              ]);
              expect(
                allClose(op.call(interiorView), op.call(interiorView.copy())),
                isTrue,
                reason: '${op.name} failed interior offset subview equivalence',
              );

              // 3. Transposed 2D view (non-contiguous strides)
              final tView = base.transpose();
              expect(
                allClose(op.call(tView), op.call(tView.copy())),
                isTrue,
                reason: '${op.name} failed transposed 2D equivalence',
              );
            });
          },
        );

        test(
          'non-contiguous out view, transposed out view, in-place out, and read-only broadcast out rejection',
          () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                <double>[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                [6],
                DType.float64,
              );
              final expected = op.call(a);

              // 1. Non-contiguous out view (step: 2)
              final carrier = NDArray.full([12], 99.0, dtype: DType.float64);
              final outSlice = carrier.slice([
                Slice(start: 0, stop: 12, step: 2),
              ]);
              final returned = op.call(a, out: outSlice);
              expect(sameId(returned, outSlice), isTrue);
              expect(allClose(outSlice, expected), isTrue);
              final untouched = carrier.slice([
                Slice(start: 1, stop: 12, step: 2),
              ]);
              for (var i = 0; i < 6; i++) {
                expect(
                  untouched[[i]],
                  equals(99.0),
                  reason:
                      '${op.name} corrupted interstitial elements of strided out',
                );
              }

              // 2. 2D transposed out view
              final a2d = a.reshape([2, 3]);
              final expected2d = op.call(a2d);
              final tOutCarrier = NDArray.zeros([3, 2], DType.float64);
              final tOutView = tOutCarrier.transpose(); // shape [2, 3]
              op.call(a2d, out: tOutView);
              expect(
                allClose(tOutView, expected2d),
                isTrue,
                reason: '${op.name} failed writing to transposed 2D out view',
              );

              // 3. Read-only broadcast out view (stride == 0) must throw ArgumentError
              final bcastSource = NDArray.fromList([99.0], [1], DType.float64);
              final bcastOut = broadcastTo(bcastSource, [6]);
              expect(
                () => op.call(a, out: bcastOut),
                throwsArgumentError,
                reason: '${op.name} must reject read-only broadcast out view',
              );
              expect(bcastSource[[0]], equals(99.0));

              // Read-only out buffer rejection (isWriteable = false)
              final roOut = NDArray.zeros([6], DType.float64)
                ..isWriteable = false;
              expect(
                () => op.call(a, out: roOut),
                throwsA(anyOf(isA<StateError>(), isA<ArgumentError>())),
                reason: '${op.name} must reject read-only out buffer',
              );

              // Aliased non-contiguous out safety: reversed view
              final revCarrier = linspace(0.5, 3.0, 6, dtype: DType.float64);
              final revView = revCarrier.slice([Slice(step: -1)]);
              final revExpected = op.call(revView.copy());
              op.call(revView, out: revView);
              expect(
                allClose(revView, revExpected),
                isTrue,
                reason: '${op.name} failed aliased reversed out view safety',
              );

              // 4. In-place out: a
              if (op.supportsInPlace) {
                final inPlace = a.copy();
                op.call(inPlace, out: inPlace);
                expect(allClose(inPlace, expected), isTrue);
              }
            });
          },
        );

        test(
          'disposed input or disposed out throws StateError and preserves ScratchArena',
          () {
            final markerBefore = ScratchArena.marker;
            final disposed = NDArray.zeros([4], DType.float64)..dispose();
            final valid = NDArray.ones([4], DType.float64);
            try {
              expect(() => op.call(disposed), throwsStateError);
              expect(() => op.call(valid, out: disposed), throwsStateError);
            } finally {
              valid.dispose();
            }
            expect(ScratchArena.marker, equals(markerBefore));
          },
        );
      });
    }
  });

  group('Binary Operation Contracts', () {
    for (final op in binaryOps) {
      group(op.name, () {
        test(
          'validation happens BEFORE mutating out buffer and rejects read-only broadcast out',
          () {
            NDArray.scope(() {
              final a = NDArray.ones([4], DType.float64);
              final badB = NDArray.ones([3], DType.float64);
              final out = NDArray.full([4], 99.0, dtype: DType.float64);
              final markerBefore = ScratchArena.marker;

              expect(() => op.call(a, badB, out: out), throwsArgumentError);
              expect(ScratchArena.marker, equals(markerBefore));

              for (var i = 0; i < 4; i++) {
                expect(
                  out[[i]],
                  equals(99.0),
                  reason:
                      '${op.name} mutated `out` before shape validation failed',
                );
              }

              // Read-only broadcast out view
              final bcastSrc = NDArray.fromList([99.0], [1], DType.float64);
              final bcastOut = broadcastTo(bcastSrc, [4]);
              expect(
                () => op.call(a, a, out: bcastOut),
                throwsArgumentError,
                reason: '${op.name} must reject read-only broadcast out view',
              );
              expect(bcastSrc[[0]], equals(99.0));

              // Read-only out buffer rejection (isWriteable = false)
              final roOut = NDArray.zeros([4], DType.float64)
                ..isWriteable = false;
              expect(
                () => op.call(a, a, out: roOut),
                throwsA(anyOf(isA<StateError>(), isA<ArgumentError>())),
                reason: '${op.name} must reject read-only out buffer',
              );
            });
          },
        );

        test(
          'strided & negative-stride equivalence and non-contiguous out',
          () {
            NDArray.scope(() {
              final a = linspace(1.0, 6.0, 6, dtype: DType.float64);
              final b = linspace(0.5, 3.0, 6, dtype: DType.float64);
              final revA = a.slice([Slice(step: -1)]);
              final revB = b.slice([Slice(step: -1)]);

              final expected = op.call(revA.copy(), revB.copy());
              final actual = op.call(revA, revB);
              expect(allClose(actual, expected), isTrue);

              final carrier = NDArray.full([12], 99.0, dtype: DType.float64);
              final stridedOut = carrier.slice([
                Slice(start: 0, stop: 12, step: 2),
              ]);
              op.call(revA, revB, out: stridedOut);
              expect(allClose(stridedOut, expected), isTrue);

              // Aliased non-contiguous out safety: reversed view
              final revCarrier = linspace(1.0, 4.0, 4, dtype: DType.float64);
              final revView = revCarrier.slice([Slice(step: -1)]);
              final revExpected = op.call(revView.copy(), revView.copy());
              op.call(revView, revView, out: revView);
              expect(
                allClose(revView, revExpected),
                isTrue,
                reason: '${op.name} failed aliased reversed out view safety',
              );
            });
          },
        );

        test('disposed input or out throws StateError', () {
          final disposed = NDArray.ones([4], DType.float64)..dispose();
          final valid = NDArray.ones([4], DType.float64);
          try {
            expect(() => op.call(disposed, valid), throwsStateError);
            expect(() => op.call(valid, disposed), throwsStateError);
            expect(
              () => op.call(valid, valid, out: disposed),
              throwsStateError,
            );
          } finally {
            valid.dispose();
          }
        });
      });
    }
  });

  group('Reduction Operation Contracts', () {
    for (final op in reductionOps) {
      group(op.name, () {
        test(
          'invalid axis or read-only broadcast out does not mutate out buffer',
          () {
            NDArray.scope(() {
              final a = NDArray.ones([3, 4], DType.float64);
              final out = NDArray.full([3], 99.0, dtype: DType.float64);
              final markerBefore = ScratchArena.marker;

              expect(() => op.call(a, axis: 5, out: out), throwsRangeError);
              expect(ScratchArena.marker, equals(markerBefore));
              for (var i = 0; i < 3; i++) {
                expect(
                  out[[i]],
                  equals(99.0),
                  reason:
                      '${op.name} mutated `out` before axis validation failed',
                );
              }

              final bcastSrc = NDArray.fromList([99.0], [1], DType.float64);
              final bcastOut = broadcastTo(bcastSrc, [3]);
              expect(
                () => op.call(a, axis: 1, out: bcastOut),
                throwsArgumentError,
                reason: '${op.name} must reject read-only broadcast out view',
              );
              expect(bcastSrc[[0]], equals(99.0));
            });
          },
        );

        test('transposed and negative-stride reduction equivalence', () {
          NDArray.scope(() {
            final a = NDArray.arange(
              1.0,
              13.0,
              dtype: DType.float64,
            ).reshape([3, 4]);
            final tView = a.transpose();
            final tCopy = tView.copy();

            expect(
              allClose(op.call(tView, axis: 0), op.call(tCopy, axis: 0)),
              isTrue,
            );
            expect(
              allClose(op.call(tView, axis: 1), op.call(tCopy, axis: 1)),
              isTrue,
            );
          });
        });

        test('read-only out rejection throws Error', () {
          NDArray.scope(() {
            final a = NDArray.ones([3, 4], DType.float64);
            final roOut = NDArray.zeros([3], DType.float64)
              ..isWriteable = false;
            expect(
              () => op.call(a, axis: 1, out: roOut),
              throwsA(anyOf(isA<StateError>(), isA<ArgumentError>())),
              reason: '${op.name} must reject read-only out buffer',
            );
          });
        });

        test('keepdims: true with and without out: buffer', () {
          NDArray.scope(() {
            final a = NDArray.arange(
              1.0,
              13.0,
              dtype: DType.float64,
            ).reshape([3, 4]);

            // axis 1 reduction with keepdims: true -> shape [3, 1]
            final res1 = op.call(a, axis: 1, keepdims: true);
            expect(res1.shape, equals([3, 1]));

            final out1 = NDArray.zeros([3, 1], DType.float64);
            final ret1 = op.call(a, axis: 1, keepdims: true, out: out1);
            expect(sameId(ret1, out1), isTrue);
            expect(allClose(out1, res1), isTrue);

            // axis 0 reduction with keepdims: true -> shape [1, 4]
            final res0 = op.call(a, axis: 0, keepdims: true);
            expect(res0.shape, equals([1, 4]));

            final out0 = NDArray.zeros([1, 4], DType.float64);
            final ret0 = op.call(a, axis: 0, keepdims: true, out: out0);
            expect(sameId(ret0, out0), isTrue);
            expect(allClose(out0, res0), isTrue);
          });
        });

        test('2D empty [0, 3] axis reduction along non-empty axis 1', () {
          NDArray.scope(() {
            final empty2d = NDArray.zeros([0, 3], DType.float64);
            final res = op.call(empty2d, axis: 1, keepdims: false);
            expect(res.shape, equals([0]));

            final resKeep = op.call(empty2d, axis: 1, keepdims: true);
            expect(resKeep.shape, equals([0, 1]));
          });
        });
      });
    }
  });

  group('Boolean Result & Index Reduction & View/Copy Contracts', () {
    for (final op in booleanResultOps) {
      test('${op.name} strided equivalence and strided out view', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [0.0, 1.5, double.nan, double.infinity, -2.0, 3.0],
            [6],
            DType.float64,
          );
          final rev = a.slice([Slice(step: -1)]);
          final expected = op.call(rev.copy());
          final actual = op.call(rev);
          expect(actual.toList(), equals(expected.toList()));

          final carrier = NDArray.full([12], true, dtype: DType.boolean);
          final outSlice = carrier.slice([Slice(start: 0, stop: 12, step: 2)]);
          op.call(rev, out: outSlice);
          expect(outSlice.toList(), equals(expected.toList()));
        });
      });
    }

    for (final op in indexReductionOps) {
      test('${op.name} transposed axis equivalence and strided out', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [3.0, 0.0, 5.0, 1.0, 4.0, 2.0],
            [2, 3],
            DType.float64,
          );
          final tView = a.transpose();
          final tCopy = tView.copy();
          final res0 = op.call(tView, axis: 0);
          final expected0 = op.call(tCopy, axis: 0);
          expect(res0.dtype, equals(DType.int64));
          expect(res0.toList(), equals(expected0.toList()));

          final res1 = op.call(tView, axis: 1);
          final expected1 = op.call(tCopy, axis: 1);
          expect(res1.dtype, equals(DType.int64));
          expect(res1.toList(), equals(expected1.toList()));

          // Contiguous out buffer reuse
          final outContig = NDArray<Int64>.zeros([2], DType.int64);
          final retContig = op.call(tView, axis: 0, out: outContig);
          expect(identical(retContig, outContig), isTrue);
          expect(outContig.toList(), equals(expected0.toList()));

          // Non-contiguous strided out buffer reuse
          final outBase = NDArray<Int64>.zeros([3, 2], DType.int64);
          final outStrided = outBase.slice([Slice.all(), Index(0)]);
          final retStrided = op.call(tView, axis: 1, out: outStrided);
          expect(identical(retStrided, outStrided), isTrue);
          expect(outStrided.toList(), equals(expected1.toList()));
        });
      });
    }

    for (final op in shapeViewOrCopyOps) {
      test('${op.name} enforces sharesMemory == ${op.expectedIsView}', () {
        NDArray.scope(() {
          final a = NDArray.arange(
            1.0,
            13.0,
            dtype: DType.float64,
          ).reshape([3, 4]);
          final res = op.call(a);
          expect(
            sharesMemory(a, res),
            equals(op.expectedIsView),
            reason:
                '${op.name} expected sharesMemory == ${op.expectedIsView}, got ${sharesMemory(a, res)}',
          );
        });
      });
    }
  });

  group('Cross-Cutting Semantic & Edge-Case Contracts', () {
    test(
      'R1: Overlapping slice aliasing (a[1:] -> a[:-1] and a[:-1] -> a[1:]) and where-mask aliasing out',
      () {
        NDArray.scope(() {
          // Forward overlap: out = a[1:], input = a[:-1]
          final a1 = NDArray.arange(1.0, 7.0, dtype: DType.float64);
          final expected1 = sin(a1.slice([Slice(stop: 5)]).copy());
          sin(a1.slice([Slice(stop: 5)]), out: a1.slice([Slice(start: 1)]));
          expect(allClose(a1.slice([Slice(start: 1)]), expected1), isTrue);

          // Backward overlap: out = a[:-1], input = a[1:]
          final a2 = NDArray.arange(1.0, 7.0, dtype: DType.float64);
          final expected2 = sin(a2.slice([Slice(start: 1)]).copy());
          sin(a2.slice([Slice(start: 1)]), out: a2.slice([Slice(stop: 5)]));
          expect(allClose(a2.slice([Slice(stop: 5)]), expected2), isTrue);

          // Binary overlap: add(a[:-1], a[1:], out: a[1:])
          final a3 = NDArray.arange(1.0, 7.0, dtype: DType.float64);
          final expected3 = add(
            a3.slice([Slice(stop: 5)]).copy(),
            a3.slice([Slice(start: 1)]).copy(),
          );
          add(
            a3.slice([Slice(stop: 5)]),
            a3.slice([Slice(start: 1)]),
            out: a3.slice([Slice(start: 1)]),
          );
          expect(allClose(a3.slice([Slice(start: 1)]), expected3), isTrue);

          // where-mask aliasing out: equal(a, b, out: mask, where: mask)
          final x = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final y = NDArray.fromList([1.0, 9.0, 3.0, 9.0], [4], DType.float64);
          final mask = NDArray.fromList(
            [true, true, false, false],
            [4],
            DType.boolean,
          );
          equal(x, y, out: mask, where: mask);
          expect(mask.toList(), equals([true, false, false, false]));
        });
      },
    );

    test(
      'R6: SIMD vector-length & remainder tail sweep (N in [1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 1031])',
      () {
        const lengths = [1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 1031];
        for (final n in lengths) {
          NDArray.scope(() {
            // Build carrier of length 2*n so step:2 slice exercises strided s_* loop
            // while contiguous copy exercises SIMD v_* loop + scalar tail
            final carrier = linspace(0.5, 2.5, 2 * n, dtype: DType.float64);
            final strided = carrier.slice([
              Slice(start: 0, stop: 2 * n, step: 2),
            ]);
            final contig = strided.copy();

            expect(
              allClose(sin(contig), sin(strided)),
              isTrue,
              reason: 'sin N=$n',
            );
            expect(
              allClose(add(contig, contig), add(strided, strided)),
              isTrue,
              reason: 'add N=$n',
            );
            expect(
              allClose(sum(contig), sum(strided)),
              isTrue,
              reason: 'sum N=$n',
            );
          });
        }
      },
    );

    test(
      'R7: Disposed-first exception precedence (StateError precedes shape/axis ArgumentError/RangeError)',
      () {
        final disposed = NDArray.ones([4], DType.float64)..dispose();
        final badShape = NDArray.ones([3], DType.float64);
        try {
          expect(
            () => add(disposed, badShape),
            throwsStateError,
            reason:
                'Disposed input must throw StateError before shape mismatch ArgumentError',
          );
          expect(
            () => sum(disposed, axis: 99),
            throwsStateError,
            reason:
                'Disposed input must throw StateError before invalid axis RangeError',
          );
        } finally {
          badShape.dispose();
        }
      },
    );

    test(
      'R8: NaN sort ordering, argsort + take_along_axis consistency, and strided even/odd FFT round-trips',
      () {
        NDArray.scope(() {
          // 1. NaNs sort to the end and argsort indices reconstruct sort(a)
          final withNan = NDArray.fromList(
            [double.nan, 3.0, -1.0, double.nan, 0.0, 2.0],
            [6],
            DType.float64,
          );
          final sorted = sort(withNan);
          final idx = argsort(withNan);
          final gathered = take_along_axis(withNan, idx, 0);
          expect(sorted[[0]], equals(-1.0));
          expect(sorted[[1]], equals(0.0));
          expect(sorted[[2]], equals(2.0));
          expect(sorted[[3]], equals(3.0));
          expect(sorted[[4]].isNaN, isTrue);
          expect(sorted[[5]].isNaN, isTrue);
          for (var i = 0; i < 4; i++) {
            expect(gathered[[i]], equals(sorted[[i]]));
          }
          expect(gathered[[4]].isNaN, isTrue);
          expect(gathered[[5]].isNaN, isTrue);

          // 2. Strided even (n=8) and odd (n=9) fft/ifft and rfft/irfft round-trip
          for (final n in [8, 9]) {
            final full = linspace(1.0, 10.0, 2 * n, dtype: DType.float64);
            final stridedSignal = full.slice([
              Slice(start: 0, stop: 2 * n, step: 2),
            ]);
            final cSignal = astype(stridedSignal, DType.complex128);
            final reconstructed = ifft(fft(cSignal));
            expect(
              allClose(real(reconstructed), stridedSignal),
              isTrue,
              reason: 'fft/ifft round-trip failed for strided n=$n',
            );

            final rReconstructed = irfft(rfft(stridedSignal), n: n);
            expect(
              allClose(rReconstructed, stridedSignal),
              isTrue,
              reason: 'rfft/irfft round-trip failed for strided n=$n',
            );
          }
        });
      },
    );

    test(
      'where: mask contract (out == null zero-init, out != null preservation, Float16/BFloat16 & aliasing)',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [double.nan, double.infinity, 4.0, 9.0],
            [4],
            DType.float64,
          );
          final mask = NDArray.fromList(
            [false, false, true, true],
            [4],
            DType.boolean,
          );

          // 1. out == null with where != null must zero-initialize unmasked positions
          final nanRes = nan_to_num(a, where: mask);
          expect(nanRes[[0]], equals(0.0));
          expect(nanRes[[1]], equals(0.0));
          expect(nanRes[[2]], equals(4.0));
          expect(nanRes[[3]], equals(9.0));

          // 2. out != null with where != null must preserve existing out values at where == false
          final outBuf = NDArray.full([4], 77.0, dtype: DType.float64);
          sqrt(
            NDArray.fromList([16.0, 25.0, 4.0, 9.0], [4], DType.float64),
            where: mask,
            out: outBuf,
          );
          expect(outBuf.toList(), equals([77.0, 77.0, 2.0, 3.0]));

          // 3. Float16 / BFloat16 isnan / isinf / isfinite with where != null and out != null
          final f16 = NDArray.fromList(
            [double.nan, 1.0, double.infinity, 2.0],
            [4],
            DType.float16,
          );
          final boolOut = NDArray.fromList(
            [true, true, true, true],
            [4],
            DType.boolean,
          );
          isnan(f16, where: mask, out: boolOut);
          // Indices 0 and 1 have mask == false so boolOut[0..1] must stay true!
          expect(boolOut.toList(), equals([true, true, false, false]));
        });
      },
    );

    test(
      'degenerate shapes ([] 0-D scalar, [0] empty 1-D, [0, 3] empty 2-D) and rank-33 fallback',
      () {
        NDArray.scope(() {
          // 1. Empty integer power([0]) must not throw RangeError in min(x2)
          final emptyI64 = NDArray<Int64>.zeros([0], DType.int64);
          final powEmpty = power(emptyI64, emptyI64);
          expect(powEmpty.shape, equals([0]));

          // 2. 0-D scalar and [0, 3] empty 2-D across unary ops
          final scalar = NDArray.scalar(1.5, dtype: DType.float64);
          final empty2d = NDArray.zeros([0, 3], DType.float64);
          for (final op in unaryOps.where((u) => u.supports0DAndEmpty)) {
            final res0d = op.call(scalar);
            expect(res0d.shape, isEmpty, reason: '${op.name} 0-D shape');
            final resEmpty = op.call(empty2d);
            expect(
              resEmpty.shape,
              equals([0, 3]),
              reason: '${op.name} [0, 3] shape',
            );
          }

          // 3. Rank-33 non-contiguous view fallback (> MAX_DIMS 32)
          final shape33 = List<int>.filled(33, 1);
          final base33 = NDArray.ones(shape33, DType.float64);
          final view33 = broadcastTo(base33, shape33);
          final added33 = add(view33, view33);
          expect(added33.shape.length, equals(33));
          expect(added33.getCellFlat(0), equals(2.0));
        });
      },
    );

    test(
      'Uint64 >= 2^63 unsigned ordering and pure-imaginary Complex128 (0 + 2i) truthiness',
      () {
        NDArray.scope(() {
          // -2 in two's complement int64 is 2^64 - 2 (18446744073709551614 > 1)
          final u64 = NDArray<Uint64>.fromList([1, -2, 5], [3], DType.uint64);
          expect(min(u64).scalar, equals(1));
          expect(max(u64).scalar, equals(-2));
          final sortedU64 = sort(u64);
          expect(sortedU64.toList(), equals([1, 5, -2]));

          // Pure-imaginary Complex128 (0.0 + 2.0i) is non-zero (truthy)
          final c128 = NDArray<Complex128>.fromList(
            [Complex(0.0, 0.0), Complex(0.0, 2.0)],
            [2],
            DType.complex128,
          );
          expect(any(c128).scalar, isTrue);
          expect(all(c128).scalar, isFalse);
          expect(count_nonzero(c128).scalar, equals(1));
        });
      },
    );

    test(
      'NDArray.scope ownership: fresh results are scoped, caller-provided out survives inner scope',
      () {
        final outerOut = NDArray.zeros([4], DType.float64);
        late NDArray<Float64> leakedRef;
        try {
          NDArray.scope(() {
            final a = NDArray.ones([4], DType.float64);
            leakedRef = sin(a);
            final returnedOut = sin(a, out: outerOut);
            expect(sameId(returnedOut, outerOut), isTrue);
          });
          expect(leakedRef.isDisposed, isTrue);
          expect(outerOut.isDisposed, isFalse);
        } finally {
          outerOut.dispose();
        }
      },
    );

    group(
      '15-DType × {Contiguous (v_*), Strided (s_*)} Dispatch Matrix Sweep',
      () {
        void verifyContiguousMatchesStrided<T extends AnySpec>(
          DType<T> dtype,
          List<Object> valuesA,
          List<Object> valuesB,
        ) {
          NDArray.scope(() {
            final n = valuesA.length;
            final aContig = NDArray.fromList(valuesA.cast<dynamic>(), [
              n,
            ], dtype);
            final bContig = NDArray.fromList(valuesB.cast<dynamic>(), [
              n,
            ], dtype);

            // Create strided 1D views (step=2) with identical logical elements
            final aInterleaved = <dynamic>[];
            final bInterleaved = <dynamic>[];
            for (var i = 0; i < n; i++) {
              aInterleaved
                ..add(valuesA[i])
                ..add(valuesA[i]);
              bInterleaved
                ..add(valuesB[i])
                ..add(valuesB[i]);
            }
            final aStrided = NDArray.fromList(aInterleaved, [
              n * 2,
            ], dtype).slice([Slice(start: 0, stop: n * 2, step: 2)]);
            final bStrided = NDArray.fromList(bInterleaved, [
              n * 2,
            ], dtype).slice([Slice(start: 0, stop: n * 2, step: 2)]);

            expect(aContig.isContiguous, isTrue);
            expect(aStrided.isContiguous, isFalse);

            if (dtype != DType.boolean) {
              // Core arithmetic (v_* vs s_*)
              for (final op
                  in <NDArray<DTypeTag> Function(NDArray<T>, NDArray<T>)>[
                    (x, y) => add(x, y),
                    (x, y) => subtract(x, y),
                    (x, y) => multiply(x, y),
                  ]) {
                final cRes = op(aContig, bContig);
                final sRes = op(aStrided, bStrided);
                for (var i = 0; i < n; i++) {
                  expect(sRes[[i]], equals(cRes[[i]]));
                }
              }
            }

            // Comparisons across all DTypes (v_* vs s_*)
            for (final cmp
                in <NDArray<Boolean> Function(NDArray<T>, NDArray<T>)>[
                  (x, y) => equal(x, y),
                  (x, y) => notEqual(x, y),
                ]) {
              final cCmp = cmp(aContig, bContig);
              final sCmp = cmp(aStrided, bStrided);
              for (var i = 0; i < n; i++) {
                expect(sCmp[[i]], equals(cCmp[[i]]));
              }
            }

            if (dtype != DType.boolean &&
                dtype != DType.complex128 &&
                dtype != DType.complex64) {
              for (final cmp
                  in <NDArray<Boolean> Function(NDArray<T>, NDArray<T>)>[
                    (x, y) => greater(x, y),
                    (x, y) => greaterEqual(x, y),
                    (x, y) => less(x, y),
                    (x, y) => lessEqual(x, y),
                  ]) {
                final cCmp = cmp(aContig, bContig);
                final sCmp = cmp(aStrided, bStrided);
                for (var i = 0; i < n; i++) {
                  expect(sCmp[[i]], equals(cCmp[[i]]));
                }
              }
            }

            if (dtype.isInteger) {
              for (final intOp in <NDArray<T> Function(NDArray<T>, NDArray<T>)>[
                (x, y) => gcd(x, y),
                (x, y) => lcm(x, y),
                (x, y) => bitwiseAnd(x, y),
                (x, y) => bitwiseOr(x, y),
                (x, y) => bitwiseXor(x, y),
                (x, y) => leftShift(x, y),
                (x, y) => rightShift(x, y),
              ]) {
                final cRes = intOp(aContig, bContig);
                final sRes = intOp(aStrided, bStrided);
                for (var i = 0; i < n; i++) {
                  expect(sRes[[i]], equals(cRes[[i]]));
                }
              }

              final cInv = invert(aContig);
              final sInv = invert(aStrided);
              for (var i = 0; i < n; i++) {
                expect(sInv[[i]], equals(cInv[[i]]));
              }
            }

            if (dtype == DType.float64) {
              final f64A = aContig as NDArray<Float64>;
              final f64B = bContig as NDArray<Float64>;
              final f64AStrided = aStrided as NDArray<Float64>;
              final f64BStrided = bStrided as NDArray<Float64>;

              for (final fltOp
                  in <
                    NDArray<Float64> Function(
                      NDArray<Float64>,
                      NDArray<Float64>,
                    )
                  >[
                    (x, y) => divide(x, y),
                    (x, y) => mod(x, y),
                    (x, y) => heaviside(x, y),
                    (x, y) => hypot(x, y),
                    (x, y) => atan2(x, y),
                    (x, y) => copysign(x, y),
                  ]) {
                final cRes = fltOp(f64A, f64B);
                final sRes = fltOp(f64AStrided, f64BStrided);
                for (var i = 0; i < n; i++) {
                  expect(sRes[[i]], closeTo(cRes[[i]], 1e-5));
                }
              }

              for (final fltUnary
                  in <NDArray<Float64> Function(NDArray<Float64>)>[
                    (x) => deg2rad(x),
                    (x) => rad2deg(x),
                    (x) => rint(x),
                    (x) => trunc(x),
                  ]) {
                final cRes = fltUnary(f64A);
                final sRes = fltUnary(f64AStrided);
                for (var i = 0; i < n; i++) {
                  expect(sRes[[i]], closeTo(cRes[[i]], 1e-5));
                }
              }
            }

            if (dtype == DType.complex128 || dtype == DType.complex64) {
              final cConj = conj(aContig);
              final sConj = conj(aStrided);
              final cReal = real(aContig);
              final sReal = real(aStrided);
              final cImag = imag(aContig);
              final sImag = imag(aStrided);
              final cAngle = angle(aContig);
              final sAngle = angle(aStrided);
              for (var i = 0; i < n; i++) {
                final sc = sConj[[i]] as Complex;
                final cc = cConj[[i]] as Complex;
                expect(sc.real, closeTo(cc.real, 1e-5));
                expect(sc.imag, closeTo(cc.imag, 1e-5));
                expect(
                  sReal[[i]] as double,
                  closeTo(cReal[[i]] as double, 1e-5),
                );
                expect(
                  sImag[[i]] as double,
                  closeTo(cImag[[i]] as double, 1e-5),
                );
                expect(
                  sAngle[[i]] as double,
                  closeTo(cAngle[[i]] as double, 1e-5),
                );
              }
            }
          });
        }

        test('sweeps all 15 DTypes across contiguous and strided kernels', () {
          verifyContiguousMatchesStrided(
            DType.float64,
            [1.5, 2.5, 3.5, 4.5],
            [0.5, 1.5, 2.0, 1.0],
          );
          verifyContiguousMatchesStrided(
            DType.float32,
            [1.5, 2.5, 3.5, 4.5],
            [0.5, 1.5, 2.0, 1.0],
          );
          verifyContiguousMatchesStrided(
            DType.float16,
            [1.5, 2.5, 3.5, 4.5],
            [0.5, 1.5, 2.0, 1.0],
          );
          verifyContiguousMatchesStrided(
            DType.bfloat16,
            [1.5, 2.5, 3.5, 4.5],
            [0.5, 1.5, 2.0, 1.0],
          );
          verifyContiguousMatchesStrided(
            DType.int64,
            [12, 18, 24, 30],
            [1, 2, 3, 2],
          );
          verifyContiguousMatchesStrided(
            DType.int32,
            [12, 18, 24, 30],
            [1, 2, 3, 2],
          );
          verifyContiguousMatchesStrided(
            DType.int16,
            [12, 18, 24, 30],
            [1, 2, 3, 2],
          );
          verifyContiguousMatchesStrided(
            DType.int8,
            [12, 18, 24, 30],
            [1, 2, 3, 2],
          );
          verifyContiguousMatchesStrided(
            DType.uint64,
            [12, 18, 24, 30],
            [1, 2, 3, 2],
          );
          verifyContiguousMatchesStrided(
            DType.uint32,
            [12, 18, 24, 30],
            [1, 2, 3, 2],
          );
          verifyContiguousMatchesStrided(
            DType.uint16,
            [12, 18, 24, 30],
            [1, 2, 3, 2],
          );
          verifyContiguousMatchesStrided(
            DType.uint8,
            [12, 18, 24, 30],
            [1, 2, 3, 2],
          );
          verifyContiguousMatchesStrided(
            DType.complex128,
            [Complex(1.0, 2.0), Complex(3.0, -1.0), Complex(2.0, 4.0)],
            [Complex(0.5, 1.0), Complex(1.0, 2.0), Complex(-1.0, 0.5)],
          );
          verifyContiguousMatchesStrided(
            DType.complex64,
            [Complex(1.0, 2.0), Complex(3.0, -1.0), Complex(2.0, 4.0)],
            [Complex(0.5, 1.0), Complex(1.0, 2.0), Complex(-1.0, 0.5)],
          );
          verifyContiguousMatchesStrided(
            DType.boolean,
            [true, false, true, false],
            [true, true, false, false],
          );
        });

        test(
          'exercises multi-array and linalg out: operations (matmul, inv, solve, cholesky, det, convolve, correlate, concatenate, stack, outer, kron, tensordot)',
          () {
            NDArray.scope(() {
              final m1 = NDArray.fromList(
                [4.0, 1.0, 1.0, 3.0],
                [2, 2],
                DType.float64,
              );
              final m2 = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0],
                [2, 2],
                DType.float64,
              );
              final out2x2 = NDArray.zeros([2, 2], DType.float64);

              expect(sameId(matmul(m1, m2, out: out2x2), out2x2), isTrue);
              expect(sameId(inv(m1, out: out2x2), out2x2), isTrue);
              expect(sameId(solve(m1, m2, out: out2x2), out2x2), isTrue);
              expect(sameId(cholesky(m1, out: out2x2), out2x2), isTrue);

              final outScalar = NDArray.zeros([], DType.float64);
              expect(sameId(det(m1, out: outScalar), outScalar), isTrue);

              final v1 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
              final v2 = NDArray.fromList([0.5, 1.5], [2], DType.float64);
              final outConv = NDArray.zeros([4], DType.float64);
              final outCorr = NDArray.zeros([2], DType.float64);
              expect(sameId(convolve(v1, v2, out: outConv), outConv), isTrue);
              expect(sameId(correlate(v1, v2, out: outCorr), outCorr), isTrue);

              final outConcat = NDArray.zeros([5], DType.float64);
              expect(
                sameId(
                  concatenate<Float64>([v1, v2], out: outConcat),
                  outConcat,
                ),
                isTrue,
              );

              final outStack = NDArray.zeros([2, 3], DType.float64);
              expect(
                sameId(stack<Float64>([v1, v1], out: outStack), outStack),
                isTrue,
              );

              final outOuter = NDArray.zeros([3, 2], DType.float64);
              expect(sameId(outer(v1, v2, out: outOuter), outOuter), isTrue);

              final outKron = NDArray.zeros([6], DType.float64);
              expect(sameId(kron(v1, v2, out: outKron), outKron), isTrue);

              final outTensorDot = NDArray.zeros([2, 2], DType.float64);
              expect(
                sameId(
                  tensordot(m1, m2, axes: 1, out: outTensorDot),
                  outTensorDot,
                ),
                isTrue,
              );
            });
          },
        );
      },
    );
  });

  group('Same-dtype binary kernel contracts', () {
    // Binary arithmetic operations require matching operand dtypes, while *As
    // variants require matching operand dtypes and match cast-then-compute.
    const realDTypes = <DType>[
      DType.float64,
      DType.float32,
      DType.float16,
      DType.int64,
      DType.int32,
      DType.int16,
      DType.int8,
      DType.uint64,
      DType.uint32,
      DType.uint16,
      DType.uint8,
      DType.boolean,
    ];
    final ops = <String, NDArray Function(NDArray, NDArray)>{
      'add': (a, b) => add<DTypeTag>(a, b),
      'subtract': (a, b) => subtract<DTypeTag>(a, b),
      'multiply': (a, b) => multiply<DTypeTag>(a, b),
      'divide': (a, b) => divide(a, b),
    };

    NDArray make(DType dtype, List<int> values) => switch (dtype) {
      DType.boolean => NDArray.fromList(values.map((v) => v.isOdd).toList(), [
        values.length,
      ], dtype),
      _ when dtype.isFloating => NDArray.fromList(
        values.map((v) => v.toDouble()).toList(),
        [values.length],
        dtype,
      ),
      _ => NDArray.fromList(values, [values.length], dtype),
    };

    for (final MapEntry(key: name, value: op) in ops.entries) {
      for (final dtypeA in realDTypes) {
        for (final dtypeB in realDTypes) {
          test('$name($dtypeA, $dtypeB) enforces same-dtype contract', () {
            NDArray.scope(() {
              // Contiguous and strided operands exercise both v_ and s_
              // kernels.
              final a = make(dtypeA, [7, 5, 9, 3]);
              final b = make(dtypeB, [1, 2, 3, 1]);
              final strided = [const Slice(start: 0, stop: 4, step: 2)];
              for (final (x, y) in [
                (a, b),
                (a.slice(strided), b.slice(strided)),
              ]) {
                if (dtypeA != dtypeB) {
                  expect(() => op(x, y), throwsArgumentError);
                } else if (name == 'subtract' && dtypeA == DType.boolean) {
                  expect(() => op(x, y), throwsUnsupportedError);
                } else {
                  final actual = op(x, y);
                  final expected = op(
                    castNDArray(x, actual.dtype),
                    castNDArray(y, actual.dtype),
                  );
                  expect(actual.toList(), expected.toList());
                }
              }
            });
          });
        }
      }
    }
  });

  group('Read-only broadcast view mutability contracts', () {
    test(
      'Every in-place mutator rejects read-only broadcastTo views across all 15 DTypes and preserves source memory',
      () {
        for (final dtype in DType.values) {
          NDArray.scope(() {
            final Object sampleScalar = switch (dtype) {
              DType.boolean => true,
              DType.complex128 || DType.complex64 => Complex(3.0, 4.0),
              _ when dtype.isFloating => 5.0,
              _ => 5,
            };
            final src = NDArray.full([1, 2], sampleScalar, dtype: dtype);
            final expectedSnapshot = src.toList();
            final view = broadcastTo(src, [2, 2]);
            expect(view.isWriteable, isFalse);

            final mask = NDArray<Boolean>.fromList(
              [true, false, true, false],
              [2, 2],
              DType.boolean,
            );
            final indices1D = NDArray<Int32>.fromList([0], [1], DType.int32);
            final indices2D = NDArray<Int32>.fromList(
              [0, 1, 0, 1],
              [2, 2],
              DType.int32,
            );
            final replacementRow = NDArray.full(
              [1, 2],
              sampleScalar,
              dtype: dtype,
            );

            expect(() => view.fill(sampleScalar), throwsStateError);
            expect(() => view.setCell([0, 0], sampleScalar), throwsStateError);
            expect(() => view.setCellFlat(0, sampleScalar), throwsStateError);
            expect(() => view.setCellRaw(0, sampleScalar), throwsStateError);
            expect(
              () => view.setByMask(mask, replacementRow),
              throwsStateError,
            );
            expect(
              () => view.setByMaskScalar(mask, sampleScalar),
              throwsStateError,
            );
            expect(
              () => view.setIndices(indices1D, replacementRow),
              throwsStateError,
            );
            expect(
              () => view.setIndicesScalar(indices1D, sampleScalar),
              throwsStateError,
            );
            expect(
              () => view.sliceAssign([const Slice.all()], sampleScalar),
              throwsStateError,
            );
            expect(() => view[0] = sampleScalar, throwsStateError);
            expect(() => view[mask] = sampleScalar, throwsStateError);
            expect(
              () => put_along_axis(view, indices2D, replacementRow, 1),
              throwsArgumentError,
            );
            expect(
              () => atUfunc(view, indices1D, replacementRow, op: BinaryOp.add),
              throwsArgumentError,
            );

            // Source array must remain completely untouched
            expect(src.toList(), equals(expectedSnapshot));
          });
        }
      },
    );
  });

  group('Operator dtype & scalar bounds contracts', () {
    test(
      'All same-dtype binary operators throw ArgumentError (never TypeError) on mismatched array dtype or out-of-range scalar',
      () {
        NDArray.scope(() {
          final aInt8 = NDArray<Int8>.fromList([1, 2, 3], [3], DType.int8);
          final bInt32 = NDArray<Int32>.fromList([1, 2, 3], [3], DType.int32);

          final sameDTypeOps =
              <String, Object? Function(NDArray<Int8>, Object?)>{
                '+': (a, o) => a + o,
                '-': (a, o) => a - o,
                '*': (a, o) => a * o,
                '~/': (a, o) => a ~/ o,
                '%': (a, o) => a % o,
                '&': (a, o) => a & o,
                '|': (a, o) => a | o,
                '^': (a, o) => a ^ o,
                '<<': (a, o) => a << o,
                '>>': (a, o) => a >> o,
              };

          for (final MapEntry(key: opName, value: invoke)
              in sameDTypeOps.entries) {
            expect(
              () => invoke(aInt8, bInt32),
              throwsArgumentError,
              reason:
                  'Operator $opName with mismatched NDArray<Int32> must throw ArgumentError',
            );
            expect(
              () => invoke(aInt8, 255),
              throwsArgumentError,
              reason:
                  'Operator $opName with out-of-range scalar 255 on Int8 must throw ArgumentError',
            );
          }
        });
      },
    );
  });

  group('64-bit shapes, strides, overflow guards, and BLAS boundary contracts', () {
    const dim64 =
        0x80000000; // 2^31 (2,147,483,648) — exceeds 32-bit signed int

    test(
      'NDArray views, broadcasting, empty creation, and NPY headers support > 2^31 - 1 dimensions and strides',
      () {
        NDArray.scope(() {
          final base = NDArray<Float64>.fromList([42.0], [1], DType.float64);

          // 0-stride view with > 2^31 - 1 elements
          final view1D = NDArray<Float64>.view(
            base,
            shape: [dim64],
            strides: [0],
          );
          expect(view1D.shape, equals([dim64]));
          expect(view1D.size, equals(dim64));
          expect(view1D.getCell([dim64 - 1]), equals(42.0));

          // asStrided with > 2^31 - 1 dimension and > 2^31 - 1 stride on length-1 axis
          final strided = asStrided<Float64>(
            base,
            shape: [1, dim64],
            strides: [dim64, 0],
          );
          expect(strided.shape, equals([1, dim64]));
          expect(strided.strides, equals([dim64, 0]));
          expect(strided.size, equals(dim64));
          expect(strided.getCell([0, dim64 - 1]), equals(42.0));

          // broadcastTo with > 2^31 - 1 target shape
          final broadcasted = broadcastTo<Float64>(base, [2, dim64]);
          expect(broadcasted.shape, equals([2, dim64]));
          expect(broadcasted.size, equals(2 * dim64));
          expect(broadcasted.getCell([1, dim64 - 1]), equals(42.0));

          // Empty array with > 2^31 - 1 dimension allocates 0 elements safely
          final emptyLarge = NDArray<Float64>.create([0, dim64], DType.float64);
          expect(emptyLarge.shape, equals([0, dim64]));
          expect(emptyLarge.size, equals(0));

          // NPY header parser accepts > 2^31 - 1 dimensions
          final dict =
              "{'descr': '<f8', 'fortran_order': False, 'shape': ($dim64,), }";
          final parsed = parseNpyHeader(dict);
          expect(parsed.shape, equals([dim64]));
        });
      },
    );

    test(
      '64-bit signed multiplication, byte-size, and span overflows throw ArgumentError',
      () {
        NDArray.scope(() {
          final base = NDArray<Float64>.fromList(
            [1.0, 2.0],
            [2],
            DType.float64,
          );

          // Product 3037000500 * 3037000500 > 0x7fffffffffffffff wraps 64-bit signed int
          expect(
            () => NDArray<Float64>.create([
              3037000500,
              3037000500,
            ], DType.float64),
            throwsArgumentError,
          );
          expect(
            () => broadcastTo<Float64>(base, [3037000500, 3037000500]),
            throwsArgumentError,
          );

          // Byte-size overflow: totalSize * 8 > 0x7fffffffffffffff
          expect(
            () => NDArray<Float64>.create([0x2000000000000000], DType.float64),
            throwsArgumentError,
          );

          // Span overflow: (dim - 1) * stride > 0x7fffffffffffffff
          expect(
            () => NDArray<Float64>.view(
              base,
              shape: [3],
              strides: [0x4000000000000000],
            ),
            throwsArgumentError,
          );
        });
      },
    );

    test(
      'OpenBLAS and LAPACK entrypoints throw UnsupportedError when dimensions or strides exceed 32-bit blasint (0x7fffffff)',
      () {
        NDArray.scope(() {
          final base = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float64,
          );
          // 2D view with dimension > 0x7fffffff using 0-stride (no extra memory allocated)
          final tall = NDArray<Float64>.view(
            base,
            shape: [dim64, 1],
            strides: [0, 1],
          );
          final wide = NDArray<Float64>.view(
            base,
            shape: [1, dim64],
            strides: [1, 0],
          );
          final squareBig = NDArray<Float64>.view(
            base,
            shape: [dim64, dim64],
            strides: [0, 0],
          );
          final vecBig = NDArray<Float64>.view(
            base,
            shape: [dim64],
            strides: [0],
          );
          // 2x2 view with leading stride > 0x7fffffff (dim=1 along axis 0 so span is 0, or test via 1x2 view)
          final bigStride1x2 = NDArray<Float64>.view(
            base,
            shape: [1, 2],
            strides: [dim64, 1],
          );
          final small2x2 = NDArray<Float64>.fromList(
            [1.0, 0.0, 0.0, 1.0],
            [2, 2],
            DType.float64,
          );

          expect(() => matmul(tall, wide), throwsUnsupportedError);
          expect(() => matmul(bigStride1x2, small2x2), throwsUnsupportedError);
          expect(() => tensordot(tall, wide, axes: 1), throwsUnsupportedError);
          expect(() => inner(vecBig, vecBig), throwsUnsupportedError);
          expect(() => vdot(vecBig, vecBig), throwsUnsupportedError);
          expect(() => multi_dot([tall, wide]), throwsUnsupportedError);
          expect(() => cholesky(squareBig), throwsUnsupportedError);
          expect(() => qr(tall), throwsUnsupportedError);
          expect(() => svd(tall), throwsUnsupportedError);
          expect(() => det(squareBig), throwsUnsupportedError);
          expect(() => slogdet(squareBig), throwsUnsupportedError);
          expect(() => inv(squareBig), throwsUnsupportedError);
          expect(() => solve(squareBig, tall), throwsUnsupportedError);
          expect(() => lstsq(tall, vecBig), throwsUnsupportedError);
          expect(() => pinv(tall), throwsUnsupportedError);
          expect(() => matrix_power(squareBig, 2), throwsUnsupportedError);
          expect(() => eigh(squareBig), throwsUnsupportedError);
          expect(() => eigvalsh(squareBig), throwsUnsupportedError);
          expect(() => eig(squareBig), throwsUnsupportedError);
          expect(() => eigvals(squareBig), throwsUnsupportedError);
          expect(() => schur(squareBig), throwsUnsupportedError);
          expect(() => hessenberg(squareBig), throwsUnsupportedError);
          expect(() => norm(tall, ord: 2), throwsUnsupportedError);
          expect(() => cond(tall), throwsUnsupportedError);
          expect(() => polyfit(vecBig, vecBig, 1), throwsUnsupportedError);
          expect(() => roots(vecBig), throwsUnsupportedError);
          expect(() => nelder_mead((x) => 0.0, vecBig), throwsUnsupportedError);
          expect(() => lbfgs((x) => 0.0, vecBig), throwsUnsupportedError);
        });
      },
    );

    test(
      'flatnonzero returns NDArray<Int64> and supports out: and strided views',
      () {
        NDArray.scope(() {
          final a = NDArray<Int32>.fromList(
            [0, 5, 0, -3, 0, 7],
            [2, 3],
            DType.int32,
          );
          final idx = flatnonzero(a);
          expect(idx.dtype, equals(DType.int64));
          expect(idx.toList(), equals([1, 3, 5]));

          final out = NDArray<Int64>.zeros([3], DType.int64);
          final res = flatnonzero(a, out: out);
          expect(sameId(res, out), isTrue);
          expect(out.toList(), equals([1, 3, 5]));

          // Strided transposed view: [[0, -3], [5, 0], [0, 7]] -> nonzero at flat indices 1, 2, 5
          final idxT = flatnonzero(a.transpose());
          expect(idxT.dtype, equals(DType.int64));
          expect(idxT.toList(), equals([1, 2, 5]));
        });
      },
    );

    test(
      '64-bit index utilities (unravel_index, ravel_multi_index, indices, sparse_indices, diag_indices, tril/triu_indices, mask_indices) satisfy algebraic identities',
      () {
        NDArray.scope(() {
          // 1. unravel_index <-> ravel_multi_index round-trip in C and F orders
          final dims = [4, 5, 3];
          final flat = NDArray<Int64>.arange(0, 60, dtype: DType.int64);
          for (final order in IndexOrder.values) {
            final coords = unravel_index(flat, dims, order: order);
            for (final c in coords) {
              expect(c.dtype, equals(DType.int64));
              expect(c.shape, equals([60]));
            }
            final reconstructed = ravel_multi_index(coords, dims, order: order);
            expect(reconstructed.dtype, equals(DType.int64));
            expect(reconstructed.toList(), equals(flat.toList()));
          }

          // 2. tril_indices(n, m: m, k: k - 1) and triu_indices(n, m: m, k: k) partition all n * m cells
          for (final (n, m, k) in [(4, 4, 0), (3, 5, 1), (5, 3, -1)]) {
            final (row: lRow, col: lCol) = tril_indices(n, m: m, k: k - 1);
            final (row: uRow, col: uCol) = triu_indices(n, m: m, k: k);
            expect(lRow.dtype, equals(DType.int64));
            expect(uRow.dtype, equals(DType.int64));
            expect(lRow.size + uRow.size, equals(n * m));
            final visited = NDArray<Int64>.zeros([n, m], DType.int64);
            for (var i = 0; i < lRow.size; i++) {
              final r = lRow.getCell([i]);
              final c = lCol.getCell([i]);
              visited.setCell([r, c], visited.getCell([r, c]) + 1);
            }
            for (var i = 0; i < uRow.size; i++) {
              final r = uRow.getCell([i]);
              final c = uCol.getCell([i]);
              visited.setCell([r, c], visited.getCell([r, c]) + 1);
            }
            expect(
              visited.toList(),
              equals(NDArray<Int64>.ones([n, m], DType.int64).toList()),
            );
          }

          // 3. mask_indices(n, triu, k: k) == triu_indices(n, k: k)
          final (row: mRow, col: mCol) = mask_indices(4, triu, k: -1);
          final (row: tRow, col: tCol) = triu_indices(4, k: -1);
          expect(mRow.toList(), equals(tRow.toList()));
          expect(mCol.toList(), equals(tCol.toList()));

          // 4. select and multinomial default to DType.int64
          final cond = NDArray<Boolean>.fromList(
            [true, false],
            [2],
            DType.boolean,
          );
          expect(
            select([cond], [1], defaultValue: 0).dtype,
            equals(DType.int64),
          );
          final pvals = NDArray<Float64>.fromList(
            [0.5, 0.5],
            [2],
            DType.float64,
          );
          expect(multinomial(5, pvals, seed: 42).dtype, equals(DType.int64));
        });
      },
    );

    group('Linear Algebra Contracts (inv, solve, cholesky, det)', () {
      test('inv, solve, cholesky, det contracts and edge cases', () {
        NDArray.scope(() {
          // 1. inv: computes 2x2 matrix inverse, identity property A @ inv(A) ≈ I
          final a = NDArray<Float64>.fromList(
            [4.0, 7.0, 2.0, 6.0],
            [2, 2],
            DType.float64,
          );
          final aInv = inv(a);
          final identity = matmul(a, aInv);
          expect(allClose(identity, NDArray.eye(2, DType.float64)), isTrue);

          // 2. solve: solves A @ X = B
          final b = NDArray<Float64>.fromList(
            [1.0, 0.0],
            [2, 1],
            DType.float64,
          );
          final x = solve(a, b);
          final ax = matmul(a, x);
          expect(allClose(ax, b), isTrue);

          // 3. cholesky: positive definite symmetric matrix decomposition
          final spd = NDArray<Float64>.fromList(
            [4.0, 2.0, 2.0, 5.0],
            [2, 2],
            DType.float64,
          );
          final l = cholesky(spd);
          final recon = matmul(l, l.transpose());
          expect(allClose(recon, spd), isTrue);

          // 4. det: matrix determinant
          final d = det(a);
          expect((d.scalar - 10.0).abs() < 1e-12, isTrue);

          // 5. Batched operations [2, 2, 2]
          final batchA = stack([a, spd], axis: 0);
          final batchInv = inv(batchA);
          expect(batchInv.shape, equals([2, 2, 2]));
          final batchDet = det(batchA);
          expect(batchDet.shape, equals([2]));
        });
      });
    });

    group(
      'Sorting & Searching Contracts (sort, argsort, partition, argpartition, searchsorted)',
      () {
        test(
          'sort, argsort, partition, argpartition, searchsorted contracts',
          () {
            NDArray.scope(() {
              final arr = NDArray<Float64>.fromList(
                [3.0, 1.0, 4.0, 1.5, 9.0, 2.0],
                [6],
                DType.float64,
              );

              // sort
              final sorted = sort(arr);
              expect(sorted.toList(), equals([1.0, 1.5, 2.0, 3.0, 4.0, 9.0]));

              // argsort with out:
              final outIdx = NDArray<Int64>.zeros([6], DType.int64);
              final sortedIdx = argsort(arr, out: outIdx);
              expect(sameId(sortedIdx, outIdx), isTrue);
              expect(sortedIdx.dtype, equals(DType.int64));
              expect(sortedIdx.toList(), equals([1, 3, 5, 0, 2, 4]));

              // partition
              final part = partition(arr, 2);
              expect(part[[2]], equals(2.0));

              // argpartition with out:
              final outPartIdx = NDArray<Int64>.zeros([6], DType.int64);
              final partIdx = argpartition(arr, 2, out: outPartIdx);
              expect(sameId(partIdx, outPartIdx), isTrue);
              expect(partIdx.dtype, equals(DType.int64));

              // searchsorted
              final needles = NDArray<Float64>.fromList(
                [0.0, 1.5, 2.5, 10.0],
                [4],
                DType.float64,
              );
              final outSearch = NDArray<Int64>.zeros([4], DType.int64);
              final ins = searchsorted(sorted, needles, out: outSearch);
              expect(sameId(ins, outSearch), isTrue);
              expect(ins.dtype, equals(DType.int64));
              expect(ins.toList(), equals([0, 1, 3, 6]));

              // Read-only out rejection
              final roOut = NDArray<Int64>.zeros([4], DType.int64)
                ..isWriteable = false;
              expect(
                () => searchsorted(sorted, needles, out: roOut),
                throwsA(anyOf(isA<StateError>(), isA<ArgumentError>())),
              );
            });
          },
        );
      },
    );

    group(
      'Percentiles & Quantiles Contracts (median, quantile, percentile, nanmedian, nanquantile, nanpercentile, ptp)',
      () {
        test('percentiles and quantiles contracts', () {
          NDArray.scope(() {
            final data = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
              [6],
              DType.float64,
            );

            // median
            expect(median(data).scalar, equals(3.5));

            // quantile
            expect(quantile(data, 0.5).scalar, equals(3.5));

            // percentile
            expect(percentile(data, 50.0).scalar, equals(3.5));

            // with NaNs
            final dataWithNan = NDArray<Float64>.fromList(
              [double.nan, 1.0, 2.0, 3.0, 4.0, 5.0],
              [6],
              DType.float64,
            );
            expect(nanmin(dataWithNan).scalar, equals(1.0));
            expect(nanmax(dataWithNan).scalar, equals(5.0));
            expect(median(dataWithNan).scalar.isNaN, isTrue);
            expect(quantile(dataWithNan, 0.5).scalar.isNaN, isTrue);
            expect(percentile(dataWithNan, 50.0).scalar.isNaN, isTrue);
            expect(nanmedian(dataWithNan).scalar, equals(3.0));
            expect(nanquantile(dataWithNan, 0.5).scalar, equals(3.0));
            expect(nanpercentile(dataWithNan, 50.0).scalar, equals(3.0));

            // 2D nanmedian / nanquantile / nanpercentile with axis, keepdims, out
            final mat2dNan = NDArray<Float64>.fromList(
              [1.0, double.nan, 3.0, 4.0, 5.0, double.nan],
              [2, 3],
              DType.float64,
            );
            final outNanMed = NDArray<Float64>.zeros([2, 1], DType.float64);
            final resNanMed = nanmedian(
              mat2dNan,
              axis: 1,
              keepdims: true,
              out: outNanMed,
            );
            expect(sameId(resNanMed, outNanMed), isTrue);
            expect(outNanMed.toList(), equals([2.0, 4.5]));

            final outNanQuant = NDArray<Float64>.zeros([2, 1], DType.float64);
            final resNanQuant = nanquantile(
              mat2dNan,
              0.5,
              axis: 1,
              keepdims: true,
              out: outNanQuant,
            );
            expect(sameId(resNanQuant, outNanQuant), isTrue);
            expect(outNanQuant.toList(), equals([2.0, 4.5]));

            final outNanPct = NDArray<Float64>.zeros([2, 1], DType.float64);
            final resNanPct = nanpercentile(
              mat2dNan,
              50.0,
              axis: 1,
              keepdims: true,
              out: outNanPct,
            );
            expect(sameId(resNanPct, outNanPct), isTrue);
            expect(outNanPct.toList(), equals([2.0, 4.5]));

            // ptp (peak to peak) with keepdims and out:
            final ptp2d = NDArray<Float64>.arange(
              1.0,
              7.0,
              dtype: DType.float64,
            ).reshape([2, 3]);
            final ptpOut = NDArray<Float64>.zeros([1, 3], DType.float64);
            final ptpRes = ptp(ptp2d, axis: 0, keepdims: true, out: ptpOut);
            expect(sameId(ptpRes, ptpOut), isTrue);
            expect(ptpOut.shape, equals([1, 3]));
            expect(ptpOut.toList(), equals([3.0, 3.0, 3.0]));
          });
        });
      },
    );

    group(
      'Signal & Discrete Transforms Contracts (diff, bincount, digitize, correlate, convolve, interp, pad)',
      () {
        test('signal and discrete transforms contracts', () {
          NDArray.scope(() {
            // diff
            final a = NDArray<Float64>.fromList(
              [1.0, 2.0, 4.0, 7.0, 11.0],
              [5],
              DType.float64,
            );
            expect(diff(a).toList(), equals([1.0, 2.0, 3.0, 4.0]));
            expect(diff(a, n: 2).toList(), equals([1.0, 1.0, 1.0]));

            // bincount
            final counts = NDArray<Int64>.fromList(
              [0, 1, 1, 3, 2, 1, 7],
              [7],
              DType.int64,
            );
            final bc = bincount(counts);
            expect(bc.dtype, equals(DType.int64));
            expect(bc.toList(), equals([1, 3, 1, 1, 0, 0, 0, 1]));

            // digitize
            final x = NDArray<Float64>.fromList(
              [0.2, 6.4, 3.0, 1.6],
              [4],
              DType.float64,
            );
            final bins = NDArray<Float64>.fromList(
              [0.0, 1.0, 2.5, 4.0, 10.0],
              [5],
              DType.float64,
            );
            final binIdx = digitize(x, bins);
            expect(binIdx.dtype, equals(DType.int64));
            expect(binIdx.toList(), equals([1, 4, 3, 2]));

            // correlate and convolve
            final sig = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0],
              [3],
              DType.float64,
            );
            final kernel = NDArray<Float64>.fromList(
              [0.0, 1.0, 0.5],
              [3],
              DType.float64,
            );
            final conv = convolve(sig, kernel);
            expect(conv.shape, equals([5]));
            final corr = correlate(sig, kernel, mode: ConvMode.full);
            expect(corr.shape, equals([5]));

            // interp
            final xPoints = NDArray<Float64>.fromList(
              [2.5],
              [1],
              DType.float64,
            );
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
            final interpVal = interp(xPoints, xp, fp);
            expect(interpVal[[0]], equals(25.0));

            // pad
            final toPad = NDArray<Float64>.fromList(
              [1.0, 2.0],
              [2],
              DType.float64,
            );
            final padded = pad(
              toPad,
              PadWidth.axes([(1, 2)]),
              mode: PadMode.constant,
              constantValues: PadValues.all(0.0),
            );
            expect(padded.toList(), equals([0.0, 1.0, 2.0, 0.0, 0.0]));
          });
        });
      },
    );

    group('Reshaping & Grid Contracts (repeat, tile, roll, flip, rot90)', () {
      test('reshaping and grid operation contracts', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );

          // repeat
          final rep = repeat(a, 2);
          expect(rep.toList(), equals([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]));

          // tile
          final tl = tile(a, [2]);
          expect(tl.toList(), equals([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]));

          // roll
          final rl = roll(a, 1);
          expect(rl.toList(), equals([3.0, 1.0, 2.0]));

          // flip
          final flp = flip(a);
          expect(flp.toList(), equals([3.0, 2.0, 1.0]));

          // rot90 on 2D
          final mat = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final rot = rot90(mat);
          expect(rot.toList(), equals([2.0, 4.0, 1.0, 3.0]));
        });
      });
    });

    group('Targeted Point Regressions (F1-F12, D1-1..D1-11, D3-1..D3-7, D5-1)', () {
      test(
        'F2: reduceat on fmin/fmax with slice length > 16 starting with NaN ignores leading NaN',
        () {
          NDArray.scope(() {
            final elements = <double>[
              double.nan,
              15.0,
              12.0,
              3.0,
              18.0,
              7.0,
              22.0,
              1.5,
              9.0,
              4.0,
              11.0,
              14.0,
              6.0,
              8.0,
              13.0,
              17.0,
              25.0,
              19.0,
              21.0,
              2.0,
            ];
            final src = NDArray<Float64>.fromList(elements, [
              20,
            ], DType.float64);
            final indices = NDArray<Int64>.fromList([0], [1], DType.int64);

            final minRes = reduceatUfunc(src, indices, op: BinaryOp.fmin);
            expect(
              minRes[[0]],
              equals(1.5),
              reason: 'fmin must ignore leading NaN in slice > 16',
            );

            final maxRes = reduceatUfunc(src, indices, op: BinaryOp.fmax);
            expect(
              maxRes[[0]],
              equals(25.0),
              reason: 'fmax must ignore leading NaN in slice > 16',
            );
          });
        },
      );

      test(
        'F3: at and reduceat integer division/remainder by zero throws UnsupportedError',
        () {
          NDArray.scope(() {
            final intArr = NDArray<Int64>.fromList(
              [10, 20, 30],
              [3],
              DType.int64,
            );
            final zeroArr = NDArray<Int64>.fromList([0], [1], DType.int64);
            final idx = NDArray<Int64>.fromList([0], [1], DType.int64);

            expect(
              () => atUfunc(intArr, idx, zeroArr, op: BinaryOp.divide),
              throwsUnsupportedError,
              reason: 'at integer divide by zero must throw UnsupportedError',
            );
            expect(
              () => atUfunc(intArr, idx, zeroArr, op: BinaryOp.remainder),
              throwsUnsupportedError,
              reason:
                  'at integer remainder by zero must throw UnsupportedError',
            );

            final divArr = NDArray<Int64>.fromList(
              [10, 0, 30],
              [3],
              DType.int64,
            );
            final reduceIdx = NDArray<Int64>.fromList([0], [1], DType.int64);
            expect(
              () => reduceatUfunc(divArr, reduceIdx, op: BinaryOp.divide),
              throwsArgumentError,
              reason:
                  'reduceat with non-reducible op divide must throw ArgumentError',
            );
          });
        },
      );

      test(
        'F1: extreme signed int64/int32 values in diff, correlate, and reduceat',
        () {
          NDArray.scope(() {
            const maxInt64 = 9223372036854775807;
            const minInt64 = -9223372036854775808;
            final arr64 = NDArray<Int64>.fromList(
              [minInt64, 0, maxInt64],
              [3],
              DType.int64,
            );

            // diff handles extreme int64 values
            final d = diff(arr64);
            expect(d.shape, equals([2]));
            expect(d.dtype, equals(DType.int64));

            // correlate handles int64 values
            final kernel = NDArray<Int64>.fromList([1, -1], [2], DType.int64);
            final corr = correlate(arr64, kernel);
            expect(corr.dtype, equals(DType.int64));

            // reduceat handles int64
            final indices = NDArray<Int64>.fromList([0, 2], [2], DType.int64);
            final redAt = reduceatUfunc(arr64, indices, op: BinaryOp.maximum);
            expect(redAt[[0]], equals(0));
            expect(redAt[[1]], equals(maxInt64));
          });
        },
      );

      test(
        'F4: histogram with Float64 input containing 1e100 / -1e100 finite outliers',
        () {
          NDArray.scope(() {
            final outliers = NDArray<Float64>.fromList(
              [-1e100, 0.2, 0.8, 1e100],
              [4],
              DType.float64,
            );
            final (:hist, :binEdges) = histogram(
              outliers,
              bins: 2,
              range: (0.0, 1.0),
            );
            expect(hist.shape, equals([2]));
            expect(binEdges.shape, equals([3]));
            expect(hist.toList(), equals([1, 1]));
          });
        },
      );

      test('F9: unique on Int64, Int32, Int16, Uint8 with >1000 elements', () {
        NDArray.scope(() {
          for (final dt in [
            DType.int64,
            DType.int32,
            DType.int16,
            DType.uint8,
          ]) {
            final list = List<int>.generate(
              1200,
              (i) => (i % 25) - (dt == DType.uint8 ? 0 : 10),
            );
            final arr = NDArray.fromList(list, [1200], dt);
            final u = unique(arr);
            expect(u.dtype, equals(dt));
            expect(u.size, equals(25));
            for (var i = 0; i < u.size - 1; i++) {
              final a = u.getCell([i]) as num;
              final b = u.getCell([i + 1]) as num;
              expect(a < b, isTrue);
            }
          }
        });
      });

      test(
        'binaryUfunc minimum, maximum, fmin, fmax with where: null on contiguous, strided, and NaN',
        () {
          NDArray.scope(() {
            final aContig = NDArray<Float64>.fromList(
              [double.nan, 2.0, 3.0, double.nan],
              [4],
              DType.float64,
            );
            final bContig = NDArray<Float64>.fromList(
              [1.0, double.nan, 0.0, double.nan],
              [4],
              DType.float64,
            );

            // contiguous
            expect(
              binaryUfunc(aContig, bContig, op: BinaryOp.fmin)[[0]],
              equals(1.0),
            );
            expect(
              binaryUfunc(aContig, bContig, op: BinaryOp.fmax)[[1]],
              equals(2.0),
            );
            expect(
              binaryUfunc(aContig, bContig, op: BinaryOp.fmin)[[3]].isNaN,
              isTrue,
            );

            // strided (transposed 2D)
            final aStrided = aContig.reshape([2, 2]).transpose();
            final bStrided = bContig.reshape([2, 2]).transpose();
            final fminStrided = binaryUfunc(
              aStrided,
              bStrided,
              op: BinaryOp.fmin,
            );
            expect(fminStrided.shape, equals([2, 2]));
          });
        },
      );

      test('F12: contiguous and strided integer matmul (Int32, Int64)', () {
        NDArray.scope(() {
          final a32 = NDArray<Int32>.fromList(
            [1, 2, 3, 4],
            [2, 2],
            DType.int32,
          );
          final b32 = NDArray<Int32>.fromList(
            [5, 6, 7, 8],
            [2, 2],
            DType.int32,
          );
          final c32 = matmul(a32, b32);
          expect(c32.dtype, equals(DType.int32));
          expect(c32.toList(), equals([19, 22, 43, 50]));

          final c32Strided = matmul(a32.transpose(), b32);
          expect(c32Strided.toList(), equals([26, 30, 38, 44]));

          final a64 = NDArray<Int64>.fromList(
            [1, 2, 3, 4],
            [2, 2],
            DType.int64,
          );
          final b64 = NDArray<Int64>.fromList(
            [5, 6, 7, 8],
            [2, 2],
            DType.int64,
          );
          final c64 = matmul(a64, b64);
          expect(c64.dtype, equals(DType.int64));
          expect(c64.toList(), equals([19, 22, 43, 50]));
        });
      });

      test(
        'F6: advanced indexing slice assign with broadcast value and self-aliasing',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.zeros([4, 4], DType.float64);
            final val = NDArray<Float64>.fromList(
              [10.0, 20.0],
              [2],
              DType.float64,
            );

            // slice assign with broadcast: target [2, 2], val [2] broadcasts to [2, 2]
            a[[Slice(start: 0, stop: 2), Slice(start: 0, stop: 2)]] = val;
            expect(a[[0, 0]], equals(10.0));
            expect(a[[0, 1]], equals(20.0));
            expect(a[[1, 0]], equals(10.0));
            expect(a[[1, 1]], equals(20.0));

            // self-aliasing slice assignment
            a[[Slice(start: 2, stop: 4)]] = a[[Slice(start: 0, stop: 2)]];
            expect(a[[2, 0]], equals(10.0));
            expect(a[[2, 1]], equals(20.0));
          });
        },
      );

      test(
        'D1-3: inv() on singular matrices throws LinAlgError across real and complex',
        () {
          NDArray.scope(() {
            final singularReal = NDArray<Float64>.fromList(
              [1.0, 2.0, 2.0, 4.0],
              [2, 2],
              DType.float64,
            );
            expect(() => inv(singularReal), throwsA(isA<LinAlgError>()));

            final singularComplex = NDArray<Complex128>.fromList(
              [
                const Complex(1.0, 0.0),
                const Complex(2.0, 0.0),
                const Complex(2.0, 0.0),
                const Complex(4.0, 0.0),
              ],
              [2, 2],
              DType.complex128,
            );
            expect(() => inv(singularComplex), throwsA(isA<LinAlgError>()));
          });
        },
      );

      test(
        'D1-5 & D1-7: const Complex, const Index, and unmodifiable list fields',
        () {
          const c = Complex(1.5, -2.5);
          expect(c.real, equals(1.5));
          expect(c.imag, equals(-2.5));

          const idx = Index(42);
          expect(idx.value, equals(42));

          final indicesObj = Indices([1, 2, 3]);
          expect(
            () => (indicesObj.values as dynamic).add(4),
            throwsUnsupportedError,
          );

          final coord = CoordinateSpacing([0.5, 1.0]);
          expect(
            () => (coord.values as dynamic).add(1.5),
            throwsUnsupportedError,
          );

          final tensorAxes = TensordotAxes.explicit([0], [1]);
          expect(
            () => (tensorAxes.explicitAxesA as dynamic).add(2),
            throwsUnsupportedError,
          );

          final bcastRes = BroadcastResult([2], [1], [0]);
          expect(
            () => (bcastRes.shape as dynamic).add(3),
            throwsUnsupportedError,
          );
        },
      );

      test(
        'D3-2: argsort, argpartition, searchsorted with out: aliased to an Int64 input array',
        () {
          NDArray.scope(() {
            final inputArr = NDArray<Int64>.fromList(
              [40, 10, 30, 20],
              [4],
              DType.int64,
            );
            final sortedIdx = argsort(inputArr, out: inputArr);
            expect(sameId(sortedIdx, inputArr), isTrue);
            expect(inputArr.toList(), equals([1, 3, 2, 0]));

            final partArr = NDArray<Int64>.fromList(
              [50, 10, 40, 20, 30],
              [5],
              DType.int64,
            );
            final partIdx = argpartition(partArr, 2, out: partArr);
            expect(sameId(partIdx, partArr), isTrue);

            final searchArr = NDArray<Int64>.fromList(
              [10, 20, 30, 40],
              [4],
              DType.int64,
            );
            final needles = NDArray<Int64>.fromList([15, 35], [2], DType.int64);
            final outSearch = needles;
            final resSearch = searchsorted(searchArr, needles, out: outSearch);
            expect(sameId(resSearch, outSearch), isTrue);
            expect(outSearch.toList(), equals([1, 3]));
          });
        },
      );

      test('D3-4: ptp with keepdims: true and axis: 0', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 5.0, 10.0, 2.0, 8.0, 3.0],
            [2, 3],
            DType.float64,
          );
          final p = ptp(a, axis: 0, keepdims: true);
          expect(p.shape, equals([1, 3]));
          expect(p.toList(), equals([1.0, 3.0, 7.0]));
        });
      });

      test('D3-7: bincount rejecting float inputs and negative values', () {
        NDArray.scope(() {
          final negative = NDArray<Int64>.fromList(
            [-1, 2, 3],
            [3],
            DType.int64,
          );
          expect(() => bincount(negative), throwsArgumentError);
        });
      });
    });

    group('Cross-Cutting Class-of-Bug Contracts (P1-1..P1-5, P2-1..P2-5)', () {
      test(
        'P2-5: Universal out-of-bounds axis RangeError contract across all axis-accepting operations',
        () {
          NDArray.scope(() {
            final a2d = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
              [2, 3],
              DType.float64,
            );
            final a3d = NDArray<Float64>.ones([2, 3, 3], DType.float64);
            final idx2d = NDArray<Int64>.fromList(
              [0, 1, 0, 1, 0, 1],
              [2, 3],
              DType.int64,
            );
            final axisOps = <String, void Function(int badAxis)>{
              'sum': (ax) => sum(a2d, axis: ax),
              'prod': (ax) => prod(a2d, axis: ax),
              'mean': (ax) => mean(a2d, axis: ax),
              'std': (ax) => std(a2d, axis: ax),
              'var_': (ax) => var_(a2d, axis: ax),
              'min': (ax) => min(a2d, axis: ax),
              'max': (ax) => max(a2d, axis: ax),
              'ptp': (ax) => ptp(a2d, axis: ax),
              'all': (ax) => all(a2d, axis: ax),
              'any': (ax) => any(a2d, axis: ax),
              'nansum': (ax) => nansum(a2d, axis: ax),
              'nanmean': (ax) => nanmean(a2d, axis: ax),
              'nanstd': (ax) => nanstd(a2d, axis: ax),
              'nanvar': (ax) => nanvar(a2d, axis: ax),
              'nanmin': (ax) => nanmin(a2d, axis: ax),
              'nanmax': (ax) => nanmax(a2d, axis: ax),
              'median': (ax) => median(a2d, axis: ax),
              'nanmedian': (ax) => nanmedian(a2d, axis: ax),
              'quantile': (ax) => quantile(a2d, 0.5, axis: ax),
              'nanquantile': (ax) => nanquantile(a2d, 0.5, axis: ax),
              'percentile': (ax) => percentile(a2d, 50.0, axis: ax),
              'nanpercentile': (ax) => nanpercentile(a2d, 50.0, axis: ax),
              'cumsum': (ax) => cumsum(a2d, axis: ax),
              'cumprod': (ax) => cumprod(a2d, axis: ax),
              'argmax': (ax) => argmax(a2d, axis: ax),
              'argmin': (ax) => argmin(a2d, axis: ax),
              'count_nonzero': (ax) => count_nonzero(a2d, axis: ax),
              'sort': (ax) => sort(a2d, axis: ax),
              'argsort': (ax) => argsort(a2d, axis: ax),
              'partition': (ax) => partition(a2d, 0, axis: ax),
              'argpartition': (ax) => argpartition(a2d, 0, axis: ax),
              'flip': (ax) => flip(a2d, axis: ax),
              'roll': (ax) => roll(a2d, 1, axis: ax),
              'rot90': (ax) => rot90(a2d, 1, [ax, 1]),
              'NDArray.take': (ax) => a2d.take([0, 1], axis: ax),
              'take_along_axis': (ax) => take_along_axis(a2d, idx2d, ax),
              'repeat': (ax) => repeat(a2d, 2, axis: ax),
              'concatenate': (ax) => concatenate([a2d, a2d], axis: ax),
              'stack': (ax) => stack([a2d, a2d], axis: ax),
              'split': (ax) => split(a2d, 1, axis: ax),
              'array_split': (ax) => array_split(a2d, 2, axis: ax),
              'array_split_at': (ax) => array_split_at(a2d, [1], axis: ax),
              'diff': (ax) => diff(a2d, axis: ax),
              'trapz': (ax) => trapz(a2d, axis: ax),
              'gradient': (ax) => gradient(a2d, axis: ax),
              'gradientArray': (ax) => gradientArray(a2d, axis: [ax]),
              'unwrap': (ax) => unwrap(a2d, axis: ax),
              'linspaceGrid': (ax) => linspaceGrid(a2d, a2d, 4, axis: ax),
              'norm': (ax) => norm(a2d, axis: [ax]),
              'cross': (ax) => cross(a3d, a3d, axis: ax),
              'tensordot': (ax) =>
                  tensordot(a2d, a2d, axes: TensordotAxes.explicit([ax], [0])),
              'fft': (ax) => fft(a2d, axis: ax),
              'ifft': (ax) => ifft(a2d, axis: ax),
              'rfft': (ax) => rfft(a2d, axis: ax),
              'irfft': (ax) => irfft(a2d, axis: ax),
              'fftshift': (ax) => fftshift(a2d, axes: [ax]),
              'ifftshift': (ax) => ifftshift(a2d, axes: [ax]),
            };

            for (final entry in axisOps.entries) {
              expect(
                () => entry.value(5),
                throwsRangeError,
                reason:
                    '${entry.key}(axis: 5) must throw RangeError on out-of-bounds positive axis',
              );
              expect(
                () => entry.value(-5),
                throwsRangeError,
                reason:
                    '${entry.key}(axis: -5) must throw RangeError on out-of-bounds negative axis',
              );
            }
          });
        },
      );

      test(
        'P1-1 & P2-1: All-DType sweep for unique, uniqueWithIndex, uniqueWithInverse, uniqueWithCounts, and uniqueAll',
        () {
          NDArray.scope(() {
            for (final dt in [
              DType.float64,
              DType.float32,
              DType.float16,
              DType.bfloat16,
              DType.int64,
              DType.int32,
              DType.int16,
              DType.int8,
              DType.uint64,
              DType.uint32,
              DType.uint16,
              DType.uint8,
            ]) {
              final raw = dt.isFloating
                  ? <double>[3.0, 1.0, 2.0, 1.0, 3.0, 2.0]
                  : <int>[3, 1, 2, 1, 3, 2];
              final arr = NDArray.fromList(raw, [6], dt);
              final u = unique(arr);
              expect(u.dtype, equals(dt), reason: 'unique($dt) dtype mismatch');
              expect(u.shape, equals([3]), reason: 'unique($dt) shape');

              final (:values, :index, :inverse, :counts) = uniqueAll(arr);
              expect(values.dtype, equals(dt));
              expect(index.dtype, equals(DType.int64));
              expect(inverse.dtype, equals(DType.int64));
              expect(counts.dtype, equals(DType.int64));
              expect(counts.toList(), equals([2, 2, 2]));
              expect(index.toList(), equals([1, 2, 0]));
              expect(inverse.toList(), equals([2, 0, 1, 0, 2, 1]));
            }

            // Multiple NaNs in Float32 & Float64 coalesce to a single trailing NaN
            final f64NaN = NDArray<Float64>.fromList(
              [double.nan, 2.0, double.nan, 1.0, 2.0],
              [5],
              DType.float64,
            );
            final u64 = uniqueWithCounts(f64NaN);
            expect(u64.values.size, equals(3));
            expect(u64.values[[0]], equals(1.0));
            expect(u64.values[[1]], equals(2.0));
            expect(u64.values[[2]].isNaN, isTrue);
            expect(u64.counts.toList(), equals([1, 2, 2]));
          });
        },
      );

      test(
        'P1-4 & P2-4: All-DType sweep for nansum, all, and any (including narrow integer/boolean accumulation widening)',
        () {
          NDArray.scope(() {
            expect(
              nansum(NDArray<Int8>.fromList([100, 100], [2], DType.int8)).dtype,
              equals(DType.int64),
            );
            expect(
              nansum(
                NDArray<Int16>.fromList([30000, 30000], [2], DType.int16),
              ).dtype,
              equals(DType.int64),
            );
            expect(
              nansum(NDArray<Int32>.fromList([1, 2], [2], DType.int32)).dtype,
              equals(DType.int64),
            );
            expect(
              nansum(
                NDArray<Uint8>.fromList([200, 200], [2], DType.uint8),
              ).dtype,
              equals(DType.uint64),
            );
            expect(
              nansum(
                NDArray<Uint16>.fromList([50000, 50000], [2], DType.uint16),
              ).dtype,
              equals(DType.uint64),
            );
            expect(
              nansum(NDArray<Uint32>.fromList([1, 2], [2], DType.uint32)).dtype,
              equals(DType.uint64),
            );
            expect(
              nansum(
                NDArray<Boolean>.fromList([true, true], [2], DType.boolean),
              ).dtype,
              equals(DType.int64),
            );

            // all & any axis reductions across all numeric/bool DTypes
            for (final dt in [
              DType.float64,
              DType.float32,
              DType.float16,
              DType.bfloat16,
              DType.int64,
              DType.int32,
              DType.int16,
              DType.int8,
              DType.uint64,
              DType.uint32,
              DType.uint16,
              DType.uint8,
            ]) {
              final raw = dt.isFloating
                  ? <double>[1.0, 0.0, 2.0, 3.0]
                  : <int>[1, 0, 2, 3];
              final arr = NDArray.fromList(raw, [2, 2], dt);
              final allAx0 = all(arr, axis: 0);
              final anyAx1 = any(arr, axis: 1);
              expect(allAx0.toList(), equals([true, false]));
              expect(anyAx1.toList(), equals([true, true]));
            }
          });
        },
      );

      test(
        'P1-3: accumulateUfunc and outerUfunc native C dispatch across DTypes',
        () {
          NDArray.scope(() {
            final i16 = NDArray<Int16>.fromList(
              [1, 2, 3, 4],
              [2, 2],
              DType.int16,
            );
            expect(
              accumulateUfunc(i16, axis: 1, op: BinaryOp.add).toList(),
              equals([1, 3, 3, 7]),
            );
            expect(
              accumulateUfunc(i16, axis: 1, op: BinaryOp.multiply).toList(),
              equals([1, 2, 3, 12]),
            );

            final u8 = NDArray<Uint8>.fromList([2, 3], [2], DType.uint8);
            final outer = outerUfunc(u8, u8, op: BinaryOp.multiply);
            expect(outer.shape, equals([2, 2]));
            expect(outer.toList(), equals([4, 6, 6, 9]));
          });
        },
      );

      test(
        'Hermitian complex cov & corrcoef, angle(NaN), and NDIter.broadcast defensive copy',
        () {
          NDArray.scope(() {
            // Complex Hermitian cov and corrcoef matching NumPy
            final cMat = NDArray<Complex128>.fromList(
              [
                const Complex(1.0, 2.0),
                const Complex(3.0, -1.0),
                const Complex(2.0, 0.0),
                const Complex(0.0, 4.0),
              ],
              [2, 2],
              DType.complex128,
            );
            final cCov = cov(cMat);
            expect(cCov.dtype, equals(DType.complex128));
            expect(cCov.shape, equals([2, 2]));
            // Diagonal of Hermitian covariance must be real and non-negative
            expect(cCov[[0, 0]].imag, closeTo(0.0, 1e-12));
            expect(cCov[[1, 1]].imag, closeTo(0.0, 1e-12));
            expect(cCov[[0, 0]].real, greaterThanOrEqualTo(0.0));
            expect(cCov[[1, 1]].real, greaterThanOrEqualTo(0.0));
            // Off-diagonals must be complex conjugates: C[0, 1] == conj(C[1, 0])
            expect(cCov[[0, 1]].real, closeTo(cCov[[1, 0]].real, 1e-12));
            expect(cCov[[0, 1]].imag, closeTo(-cCov[[1, 0]].imag, 1e-12));

            final cCorr = corrcoef(cMat);
            expect(cCorr.dtype, equals(DType.complex128));
            expect(cCorr[[0, 0]].real, closeTo(1.0, 1e-12));
            expect(cCorr[[0, 0]].imag, closeTo(0.0, 1e-12));
            expect(cCorr[[1, 1]].real, closeTo(1.0, 1e-12));
            expect(cCorr[[1, 1]].imag, closeTo(0.0, 1e-12));
            expect(cCorr[[0, 1]].real, closeTo(cCorr[[1, 0]].real, 1e-12));
            expect(cCorr[[0, 1]].imag, closeTo(-cCorr[[1, 0]].imag, 1e-12));

            // angle(NaN) preserves NaN on real inputs
            final nanReal = NDArray<Float64>.fromList(
              [double.nan, -1.0, 1.0],
              [3],
              DType.float64,
            );
            final ang = angle(nanReal);
            expect(ang[[0]].isNaN, isTrue);
            expect(ang[[1]], closeTo(3.141592653589793, 1e-12));
            expect(ang[[2]], closeTo(0.0, 1e-12));

            // NDIter.broadcast defensively copies input list
            final a = NDArray<Float64>.ones([2], DType.float64);
            final b = NDArray<Float64>.ones([2], DType.float64);
            final list = <NDArray>[a, b];
            final iter = NDIter.broadcast(list);
            list.clear();
            expect(iter.numArrays, equals(2));
          });
        },
      );
    });

    group('Round 2 Production-Readiness Contracts & Regressions', () {
      test(
        '1.1 & 1.8: Native strided out: aliasing safety (diff, var_, gradient, reduceat, copyto, binaryUfunc, sin, exp)',
        () {
          NDArray.scope(() {
            // diff with overlapping out slice
            final dSrc = NDArray<Float64>.fromList(
              [1.0, 4.0, 9.0, 16.0, 25.0],
              [5],
              DType.float64,
            );
            final dOut = dSrc.slice([Slice(start: 1)]);
            diff(dSrc, out: dOut);
            expect(dOut.toList(), equals([3.0, 5.0, 7.0, 9.0]));

            // var_ with overlapping out slice
            final vSrc = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 5.0],
              [2, 2],
              DType.float64,
            );
            final vExpected = var_(vSrc, axis: 0);
            final vOut = vSrc.slice([const Index(0), Slice.all()]);
            var_(vSrc, axis: 0, out: vOut);
            expect(vOut.toList(), equals(vExpected.toList()));

            // gradient with reversed out view
            final gSrc = NDArray<Float64>.fromList(
              [1.0, 4.0, 9.0, 16.0],
              [4],
              DType.float64,
            );
            final gExpected = gradient(gSrc);
            final gRev = gSrc.slice([Slice(step: -1)]);
            gradient(gSrc, out: gRev);
            expect(gRev.toList(), equals(gExpected.toList()));

            // reduceatUfunc with overlapping out slice
            final rSrc = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            final rIdx = NDArray<Int64>.fromList([0, 2], [2], DType.int64);
            final rExpected = reduceatUfunc(rSrc, rIdx, op: BinaryOp.add);
            final rOut = rSrc.slice([Slice(start: 1, stop: 3)]);
            reduceatUfunc(rSrc, rIdx, op: BinaryOp.add, out: rOut);
            expect(rOut.toList(), equals(rExpected.toList()));

            // copy with transposed self-view
            final cMat = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            cMat.copy(out: cMat.transpose());
            expect(cMat.toList(), equals([1.0, 3.0, 2.0, 4.0]));

            // sin and exp with transposed out view
            final sMat = NDArray<Float64>.fromList(
              [0.1, 0.2, 0.3, 0.4],
              [2, 2],
              DType.float64,
            );
            final sExpected = sin(sMat);
            sin(sMat, out: sMat.transpose());
            expect(sMat.transpose().toList(), equals(sExpected.toList()));

            final eMat = NDArray<Float64>.fromList(
              [0.1, 0.2, 0.3, 0.4],
              [2, 2],
              DType.float64,
            );
            final eExpected = exp(eMat);
            exp(eMat, out: eMat.transpose());
            expect(eMat.transpose().toList(), equals(eExpected.toList()));
          });
        },
      );

      test(
        '1.2: SendableNDArray.unsafeBorrow preserves negative-stride view offsets',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              [10.0, 20.0, 30.0, 40.0],
              [4],
              DType.float64,
            );
            final rev = a.slice([Slice(step: -1)]);
            final token = SendableNDArray.unsafeBorrow(rev);
            final view = token.materializeView();
            expect(view.toList(), equals([40.0, 30.0, 20.0, 10.0]));
          });
        },
      );

      test(
        '1.3: pad with reflect/symmetric on size-1 dimension does not divide by zero',
        () {
          NDArray.scope(() {
            final single = NDArray<Float64>.fromList(
              [42.0],
              [1],
              DType.float64,
            );
            final pReflect = pad(
              single,
              PadWidth.all(3),
              mode: PadMode.reflect,
            );
            expect(pReflect.shape, equals([7]));
            expect(pReflect.toList(), equals(List<double>.filled(7, 42.0)));

            final pSym = pad(single, PadWidth.all(2), mode: PadMode.symmetric);
            expect(pSym.shape, equals([5]));
            expect(pSym.toList(), equals(List<double>.filled(5, 42.0)));
          });
        },
      );

      test(
        '1.4: diff preserves exact 64-bit integer precision and native integer/boolean semantics',
        () {
          NDArray.scope(() {
            // Large int64 beyond 2^53 float64 mantissa limit
            const base = 1 << 60;
            final i64 = NDArray<Int64>.fromList(
              [base + 3, base + 10, base + 25],
              [3],
              DType.int64,
            );
            final d64 = diff(i64);
            expect(d64.dtype, equals(DType.int64));
            expect(d64.toList(), equals([7, 15]));

            // uint8 modular wrap-around
            final u8 = NDArray<Uint8>.fromList([10, 20, 5], [3], DType.uint8);
            final du8 = diff(u8);
            expect(du8.dtype, equals(DType.uint8));
            expect(du8.toList(), equals([10, 241]));

            // boolean XOR difference
            final bArr = NDArray<Boolean>.fromList(
              [true, true, false, true],
              [4],
              DType.boolean,
            );
            final db = diff(bArr);
            expect(db.dtype, equals(DType.boolean));
            expect(db.toList(), equals([false, true, true]));
          });
        },
      );

      test(
        '1.5 & 1.7: Empty sorting kernels and 1x1 hessenberg do not crash',
        () {
          NDArray.scope(() {
            final empty = NDArray<Float64>.zeros([0, 3], DType.float64);
            expect(sort(empty, axis: 0).shape, equals([0, 3]));
            expect(argsort(empty, axis: 0).shape, equals([0, 3]));
            expect(partition(empty, 0, axis: 1).shape, equals([0, 3]));
            expect(argpartition(empty, 0, axis: 1).shape, equals([0, 3]));

            final m1x1 = NDArray<Float64>.fromList(
              [7.5],
              [1, 1],
              DType.float64,
            );
            final (:h, :q) = hessenberg(m1x1);
            expect(h.shape, equals([1, 1]));
            expect(q.shape, equals([1, 1]));
            expect(h[[0, 0]], closeTo(7.5, 1e-12));
            expect(q[[0, 0]], closeTo(1.0, 1e-12));
          });
        },
      );

      test(
        '2.1 & 2.2: isWriteable propagation through view chains, setCell guard, and multi-axis List<int> indexing',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
              [2, 3],
              DType.float64,
            );
            a.isWriteable = false;
            final v = a.slice([Slice.all(), Slice.all()]);
            expect(v.isWriteable, isFalse);
            expect(() => v.isWriteable = true, throwsArgumentError);
            expect(() => v.setCell([0, 0], 99.0), throwsStateError);
            expect(() => v.setCellUntyped([0, 0], 99.0), throwsStateError);

            final bcast = broadcastTo(
              NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64),
              [3, 2],
            );
            expect(() => bcast.isWriteable = true, throwsArgumentError);

            // Multi-axis List<int> indexing in operator [] and operator []=
            final m = NDArray<Float64>.fromList(
              [10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0],
              [3, 3],
              DType.float64,
            );
            final picked =
                m[<Object>[
                      <int>[0, 2],
                      1,
                    ]]
                    as NDArray<Float64>;
            expect(picked.shape, equals([2]));
            expect(picked.toList(), equals([11.0, 31.0]));

            m[<Object>[
              <int>[0, 2],
              1,
            ]] = NDArray<Float64>.fromList(
              [111.0, 331.0],
              [2],
              DType.float64,
            );
            expect(m[[0, 1]], equals(111.0));
            expect(m[[2, 1]], equals(331.0));
          });
        },
      );

      test(
        '2.3 & 2.4: uint64 high-bit index bounds and tile on non-contiguous views',
        () {
          NDArray.scope(() {
            final arr = NDArray<Float64>.fromList(
              [10.0, 20.0, 30.0, 40.0, 50.0],
              [5],
              DType.float64,
            );
            final u64High = NDArray<Uint64>.fromList([-1], [1], DType.uint64);
            expect(() => arr[u64High], throwsRangeError);
            expect(
              () => ravel_multi_index([u64High], [5], mode: ChooseMode.raise),
              throwsRangeError,
            );
            expect(
              ravel_multi_index([u64High], [5], mode: ChooseMode.clip).toList(),
              equals([4]),
            );
            // 2^64 - 1 = 18446744073709551615, which is divisible by 5 (rem 0)
            expect(
              ravel_multi_index([u64High], [5], mode: ChooseMode.wrap).toList(),
              equals([0]),
            );

            final mat = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            final tiled = tile(mat.transpose(), [2, 1, 1]);
            expect(tiled.shape, equals([2, 2, 2]));
            expect(
              tiled.toList(),
              equals([1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0]),
            );
          });
        },
      );

      test(
        '3.1–3.11: Stats, Complex math, Einsum, Linalg, Sorting, DSP, Financial & CSPRNG regressions',
        () {
          NDArray.scope(() {
            // 3.1: corrcoef 1-D NaN propagation
            final nan1d = NDArray<Float64>.fromList(
              [1.0, double.nan, 3.0],
              [3],
              DType.float64,
            );
            expect(corrcoef(nan1d).scalar.isNaN, isTrue);

            // 3.2: cov rowvar: false with 2-D shape (1, k)
            final x2x1 = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0],
              [3, 1],
              DType.float64,
            );
            final y3x1 = NDArray<Float64>.fromList(
              [4.0, 5.0, 6.0],
              [3, 1],
              DType.float64,
            );
            final covRes = cov(x2x1, y: y3x1, rowvar: false);
            expect(covRes.shape, equals([2, 2]));

            // 3.3: ptp modular wrap-around for int8 when axis == null vs axis == 0
            final i8 = NDArray<Int8>.fromList([-100, 100], [2], DType.int8);
            expect(ptp(i8).scalar, equals(ptp(i8, axis: 0).scalar));

            // 3.4: Complex tan/tanh large imaginary/real and hypot overflow immunity
            final cLargeIm = NDArray<Complex128>.fromList(
              [const Complex(0.0, 1000.0)],
              [1],
              DType.complex128,
            );
            final cTan = tan(cLargeIm);
            expect(cTan[[0]].real, closeTo(0.0, 1e-12));
            expect(cTan[[0]].imag, closeTo(1.0, 1e-12));

            final cBig = NDArray<Complex128>.fromList(
              [const Complex(3e200, 4e200)],
              [1],
              DType.complex128,
            );
            final absBig = abs(cBig);
            expect(absBig[[0]], closeTo(5e200, 1e188));

            final rBig1 = NDArray<Float64>.fromList(
              [3e200],
              [1],
              DType.float64,
            );
            final rBig2 = NDArray<Float64>.fromList(
              [4e200],
              [1],
              DType.float64,
            );
            final hBig = hypot(rBig1, rBig2);
            expect(hBig[[0]], closeTo(5e200, 1e188));

            // 3.5: norm NaN propagation for ord: inf, -inf, 1
            final nanVec = NDArray<Float64>.fromList(
              [1.0, double.nan, 2.0],
              [3],
              DType.float64,
            );
            expect(norm(nanVec, ord: double.infinity).scalar.isNaN, isTrue);
            expect(
              norm(nanVec, ord: double.negativeInfinity).scalar.isNaN,
              isTrue,
            );
            expect(norm(nanVec, ord: 1).scalar.isNaN, isTrue);

            // 3.6: einsum 3-operand contraction + outer product and invalid output labels
            final ea = NDArray<Float64>.ones([2, 3], DType.float64);
            final eb = NDArray<Float64>.ones([3, 2], DType.float64);
            final ec = NDArray<Float64>.ones([2, 2], DType.float64);
            final eRes = einsum(EinsumSubscripts.parse('ij,jk,lm->iklm'), [
              ea,
              eb,
              ec,
            ]);
            expect(eRes.shape, equals([2, 2, 2, 2]));
            expect(eRes[[0, 0, 0, 0]], closeTo(3.0, 1e-12));
            expect(
              () => einsum(EinsumSubscripts.parse('ij->ijk'), [ea]),
              throwsArgumentError,
            );
            expect(
              () => einsum(EinsumSubscripts.parse('ij->ii'), [ea]),
              throwsArgumentError,
            );

            // 3.7: multi_dot dtype preservation and batched solve with 1-D RHS
            final mi = NDArray<Int32>.fromList(
              [1, 2, 3, 4],
              [2, 2],
              DType.int32,
            );
            final md = multi_dot([mi, mi]);
            expect(md.dtype, equals(DType.int32));
            expect(md.toList(), equals([7, 10, 15, 22]));

            final batchA = NDArray<Float64>.fromList(
              [2.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 4.0],
              [2, 2, 2],
              DType.float64,
            );
            final batchB1d = NDArray<Float64>.fromList(
              [6.0, 8.0],
              [2],
              DType.float64,
            );
            final solved = solve(batchA, batchB1d);
            expect(solved.shape, equals([2, 2]));
            expect(solved.toList(), equals([3.0, 2.0, 6.0, 2.0]));

            // 3.8: findIndex on float16 with non-representable target
            final f16 = NDArray<Float16>.fromList(
              [1.0, 2.0],
              [2],
              DType.float16,
            );
            expect(findIndex(f16, CompareOp.equal, 1.0001), isNull);
            expect(findIndex(f16, CompareOp.lessEqual, 1.0001), equals([0]));

            // 3.10: OS CSPRNG distributions (secure: true)
            final pSec = poisson([8], lam: 5.0, secure: true);
            expect(pSec.shape, equals([8]));
            final bSec = binomial([8], n: 10, p: 0.5, secure: true);
            expect(bSec.shape, equals([8]));
            final popArr = NDArray<Int64>.arange(0, 10, dtype: DType.int64);
            final cSec = choice(
              popArr,
              size: [5],
              replace: false,
              secure: true,
            );
            expect(cSec.shape, equals([5]));
            final permSec = permutation(popArr, secure: true);
            expect(permSec.shape, equals([10]));

            // 3.11: correlate on uint64 and npv preserving 1-D [1] shape
            final u64Corr = NDArray<Uint64>.fromList(
              [2, 3, 4],
              [3],
              DType.uint64,
            );
            final u64K = NDArray<Uint64>.fromList([1, 2], [2], DType.uint64);
            expect(correlate(u64Corr, u64K).toList(), equals([8, 11]));

            final rate1d = NDArray<Float64>.fromList([0.1], [1], DType.float64);
            final cf = NDArray<Float64>.fromList(
              [-100.0, 110.0],
              [2],
              DType.float64,
            );
            expect(npv(rate1d, cf).shape, equals([1]));
          });
        },
      );
    });

    group('Round 3 Production-Readiness Contracts & Regressions', () {
      test(
        '1.2–1.5 & 3.7: Native C++ UB edge cases (gcd/lcm INT64_MIN, cumprod overflow, pow uint64, atUfunc shifts, divmod INT64_MIN, searchsorted uint64 NaN)',
        () {
          NDArray.scope(() {
            const int64Min = -9223372036854775808;
            final aMin = NDArray<Int64>.fromList(
              [int64Min, int64Min],
              [2],
              DType.int64,
            );
            final bGcd = NDArray<Int64>.fromList(
              [int64Min, 4],
              [2],
              DType.int64,
            );
            final g = gcd(aMin, bGcd);
            expect(g[[0]], equals(int64Min));
            expect(g[[1]], equals(4));

            final l = lcm(
              NDArray<Int64>.fromList([int64Min], [1], DType.int64),
              NDArray<Int64>.fromList([-2], [1], DType.int64),
            );
            expect(l[[0]], equals(int64Min));

            // cumprod int64 signed overflow wraps cleanly in two's complement
            final bigI64 = NDArray<Int64>.fromList(
              [0x4000000000000000, 4],
              [2],
              DType.int64,
            );
            expect(cumprod(bigI64).toList(), equals([0x4000000000000000, 0]));

            // uint64 power exact beyond 2^53 (3^35 = 50031545098999707)
            final uBase = NDArray<Uint64>.fromList([3], [1], DType.uint64);
            final uExp = NDArray<Uint64>.fromList([35], [1], DType.uint64);
            expect(power(uBase, uExp)[[0]], equals(50031545098999707));

            // atUfunc shift >= 64 or < 0 returns 0 without C++ shift UB
            final shiftTarget = NDArray<Int64>.fromList(
              [16, 16],
              [2],
              DType.int64,
            );
            final shiftIdx = NDArray<Int64>.fromList([0, 1], [2], DType.int64);
            final shiftAmt = NDArray<Int64>.fromList(
              [64, -1],
              [2],
              DType.int64,
            );
            atUfunc(shiftTarget, shiftIdx, shiftAmt, op: BinaryOp.leftShift);
            expect(shiftTarget.toList(), equals([0, 0]));

            // divmod(INT64_MIN, -1) does not trap with SIGFPE
            final dmA = NDArray<Int64>.fromList([int64Min], [1], DType.int64);
            final dmB = NDArray<Int64>.fromList([-1], [1], DType.int64);
            final (:quotient, :remainder) = divmod(dmA, dmB);
            expect(quotient[[0]], equals(int64Min));
            expect(remainder[[0]], equals(0));

            // searchsorted on uint64 with high-bit uint64 needle (>= 2^63) and float64 NaN needle
            final uSorted = NDArray<Uint64>.fromList(
              [10, 20, 30],
              [3],
              DType.uint64,
            );
            final highBitNeedle = NDArray<Uint64>.fromList(
              [-1],
              [1],
              DType.uint64,
            );
            expect(searchsorted(uSorted, highBitNeedle)[[0]], equals(3));
            final fSorted = NDArray<Float64>.fromList(
              [10.0, 20.0, 30.0],
              [3],
              DType.float64,
            );
            final nanNeedle = NDArray<Float64>.fromList(
              [double.nan],
              [1],
              DType.float64,
            );
            expect(searchsorted(fSorted, nanNeedle)[[0]], equals(3));
          });
        },
      );

      test(
        '2.1–2.4 & 3.3: Native & Dart out: aliasing and overlap safety (gradient, trapz, diff, bincount, angle, nan_to_num, trace, setByMask, setIndices)',
        () {
          NDArray.scope(() {
            // 1. In-place gradient(a, out: a)
            final gArr = NDArray<Float64>.fromList(
              [1.0, 4.0, 9.0, 16.0, 25.0],
              [5],
              DType.float64,
            );
            final gExpected = gradient(gArr);
            gradient(gArr, out: gArr);
            expect(allClose(gArr, gExpected), isTrue);

            // 2. trapz with out aliasing first row of input
            final tArr = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
              [2, 3],
              DType.float64,
            );
            final tExpected = trapz(tArr, axis: 0);
            final tOut = tArr.slice([const Index(0), Slice.all()]);
            trapz(tArr, axis: 0, out: tOut);
            expect(allClose(tOut, tExpected), isTrue);

            // 3. diff with positive-offset overlapping out slice (a[1:])
            final dArr = NDArray<Float64>.fromList(
              [1.0, 3.0, 7.0, 15.0, 31.0],
              [5],
              DType.float64,
            );
            final dExpected = diff(dArr);
            final dOut = dArr.slice([Slice(start: 1)]);
            diff(dArr, out: dOut);
            expect(allClose(dOut, dExpected), isTrue);

            // 4. bincount with out == weights
            final bins = NDArray<Int64>.fromList(
              [0, 1, 1, 2],
              [4],
              DType.int64,
            );
            final w = NDArray<Float64>.fromList(
              [1.5, 2.5, 3.5, 4.5],
              [4],
              DType.float64,
            );
            final wOut = w.slice([Slice(start: 0, stop: 3)]);
            final bExpected = bincount(bins, weights: w);
            bincount(bins, weights: w, out: wOut);
            expect(allClose(wOut, bExpected), isTrue);

            // 5. angle(z, out: zReal) where out shares memory with z
            final z = NDArray<Complex128>.fromList(
              [const Complex(1.0, 1.0), const Complex(0.0, 1.0)],
              [2],
              DType.complex128,
            );
            final angExpected = angle(z);
            final zRealView = real(z);
            angle(z, out: zRealView);
            expect(allClose(zRealView, angExpected), isTrue);

            // 6. nan_to_num with transposed overlapping out view
            final nMat = NDArray<Float64>.fromList(
              [double.nan, 2.0, 3.0, double.infinity],
              [2, 2],
              DType.float64,
            );
            final nExpected = nan_to_num(nMat);
            nan_to_num(nMat, out: nMat.transpose());
            expect(allClose(nMat.transpose(), nExpected), isTrue);

            // 7. trace with out aliasing a[0, 0]
            final trMat = NDArray<Float64>.fromList(
              [2.0, 3.0, 4.0, 5.0],
              [2, 2],
              DType.float64,
            );
            final trOut = trMat.slice([const Index(0), const Index(0)]);
            einsum(EinsumSubscripts.parse('ii->'), [trMat], out: trOut);
            expect(trOut.scalar, closeTo(7.0, 1e-12));

            // 8. setByMask & setIndices when values view overlaps target
            final mArr = NDArray<Float64>.fromList(
              [10.0, 20.0, 30.0, 40.0],
              [4],
              DType.float64,
            );
            final mask = NDArray<Boolean>.fromList(
              [false, true, true, false],
              [4],
              DType.boolean,
            );
            mArr.setByMask(mask, mArr.slice([Slice(start: 0, stop: 2)]));
            expect(mArr.toList(), equals([10.0, 10.0, 20.0, 40.0]));

            final iArr = NDArray<Float64>.fromList(
              [10.0, 20.0, 30.0, 40.0],
              [4],
              DType.float64,
            );
            final idx = NDArray<Int64>.fromList([1, 2], [2], DType.int64);
            iArr.setIndices(idx, iArr.slice([Slice(start: 0, stop: 2)]));
            expect(iArr.toList(), equals([10.0, 10.0, 20.0, 40.0]));
          });
        },
      );

      test(
        '3.1, 3.2 & 3.4–3.6: Boolean where/divide, ufunc scope safety, uint64 isClose/linspace, PSD multivariateNormal, pad uint32 & SendableNDArray',
        () {
          NDArray.scope(() {
            // where(cond, bool, bool) preserves DType.boolean; divide(bool, bool) promotes to Float64
            final bCond = NDArray<Boolean>.fromList(
              [true, false],
              [2],
              DType.boolean,
            );
            final bX = NDArray<Boolean>.fromList(
              [true, true],
              [2],
              DType.boolean,
            );
            final bY = NDArray<Boolean>.fromList(
              [false, false],
              [2],
              DType.boolean,
            );
            final wBool = where(bCond, bX, bY) as NDArray<Boolean>;
            expect(wBool.dtype, equals(DType.boolean));
            expect(wBool.toList(), equals([true, false]));

            final divBool = divide(bX, bX);
            expect(divBool.dtype, equals(DType.float64));
            expect(divBool.toList(), equals([1.0, 1.0]));

            // unaryUfunc(UnaryOp.spacing) returns an undisposed array
            final spInput = NDArray<Float64>.fromList(
              [1.0],
              [1],
              DType.float64,
            );
            final spOut = unaryUfunc(spInput, op: UnaryOp.spacing);
            expect(spOut.isDisposed, isFalse);
            expect(spOut[[0]], greaterThan(0.0));

            // uint64 >= 2^63 in isClose, allClose, arrayEqual, linspace, linspaceGrid
            final uHigh1 = NDArray<Uint64>.fromList([-2, 5], [2], DType.uint64);
            final uHigh2 = NDArray<Uint64>.fromList([-1, 5], [2], DType.uint64);
            expect(
              isClose(uHigh1, uHigh2, rtol: 0.0, atol: 1.0).toList(),
              equals([true, true]),
            );
            expect(allClose(uHigh1, uHigh2, rtol: 0.0, atol: 1.0), isTrue);
            expect(arrayEqual(uHigh1, uHigh1), isTrue);

            final uLin = linspace(0, 10, 3, dtype: DType.uint64);
            expect(uLin.toList(), equals([0, 5, 10]));

            final gStart = NDArray<Uint16>.fromList([0, 10], [2], DType.uint16);
            final gStop = NDArray<Uint16>.fromList([4, 20], [2], DType.uint16);
            final uGrid = linspaceGrid(gStart, gStop, 3);
            expect(uGrid.dtype, equals(DType.uint16));
            expect(uGrid.shape, equals([3, 2]));

            // multivariateNormal supports singular positive-semidefinite cov and rejects non-symmetric cov
            final mean0 = NDArray<Float64>.zeros([2], DType.float64);
            final singularCov = NDArray<Float64>.fromList(
              [1.0, 1.0, 1.0, 1.0],
              [2, 2],
              DType.float64,
            );
            final mvn = multivariateNormal(
              mean0,
              singularCov,
              size: [4],
              seed: 7,
            );
            expect(mvn.shape, equals([4, 2]));
            for (var i = 0; i < 4; i++) {
              expect(mvn[[i, 0]], closeTo(mvn[[i, 1]], 1e-6));
            }
            final nonSymCov = NDArray<Float64>.fromList(
              [1.0, 2.0, 0.0, 1.0],
              [2, 2],
              DType.float64,
            );
            expect(
              () => multivariateNormal(mean0, nonSymCov),
              throwsArgumentError,
            );

            // pad on uint32 and large reflect/symmetric pad_before > 2 * period
            final u32 = NDArray<Uint32>.fromList([1, 2, 3], [3], DType.uint32);
            final pU32 = pad(u32, PadWidth.all(1));
            expect(pU32.dtype, equals(DType.uint32));
            expect(pU32.toList(), equals([0, 1, 2, 3, 0]));

            final pLargeReflect = pad(
              u32,
              PadWidth.all(10),
              mode: PadMode.reflect,
            );
            expect(pLargeReflect.shape, equals([23]));

            // SendableNDArray preserves isWriteable == false and marks isConsumed on materializeView
            final ro = NDArray<Float64>.ones([3], DType.float64)
              ..isWriteable = false;
            final tok = SendableNDArray.unsafeBorrow(ro);
            expect(tok.isConsumed, isFalse);
            final roView = tok.materializeView();
            expect(tok.isConsumed, isTrue);
            expect(roView.isWriteable, isFalse);
          });
        },
      );

      test(
        '4.3: Static DTypeSpec projection contracts for polynomial, interpolation, dsp, and math APIs',
        () {
          NDArray.scope(() {
            final i32 = NDArray<Int32>.fromList([1, 2, 3], [3], DType.int32);
            final f32 = NDArray<Float32>.fromList(
              [1.0, -3.0, 2.0],
              [3],
              DType.float32,
            );
            final f32Inc = NDArray<Float32>.fromList(
              [1.0, 2.0, 3.0],
              [3],
              DType.float32,
            );

            final NDArray<Float64> pv64 = polyval(i32, i32);
            final NDArray<Float32> pv32 = polyval(f32, f32);
            final NDArray<Float64> pf64 = polyfit(i32, i32, 1);
            final NDArray<Float32> cv32 = chebval(f32, f32);
            final NDArray<Complex128> r128 = roots(i32);
            final NDArray<Complex64> r64 = roots(f32);
            final NDArray<Complex64> cr64 = chebroots(f32);
            final NDArray<Float64> uw64 = unwrap(i32);
            final NDArray<Float32> uw32 = unwrap(f32);
            final NDArray<Float64> ip64 = interp(i32, i32, i32);
            final NDArray<Float64> ip32 = interp(f32Inc, f32Inc, f32Inc);
            final NDArray<Int32> dg32 = diag(i32);

            expect(pv64.dtype, equals(DType.float64));
            expect(pv32.dtype, equals(DType.float32));
            expect(pf64.dtype, equals(DType.float64));
            expect(cv32.dtype, equals(DType.float32));
            expect(r128.dtype, equals(DType.complex128));
            expect(r64.dtype, equals(DType.complex64));
            expect(cr64.dtype, equals(DType.complex64));
            expect(uw64.dtype, equals(DType.float64));
            expect(uw32.dtype, equals(DType.float32));
            expect(ip64.dtype, equals(DType.float64));
            expect(ip32.dtype, equals(DType.float64));
            expect(dg32.dtype, equals(DType.int32));
          });
        },
      );

      test(
        '1.1–1.6, 2.1–2.4 & 3.3–3.7: Full Round 3 audit verification (randint full-width, std 3D strided out, count_nonzero strided, pad_nd strided out, atUfunc broadcast, unaryUfunc/binaryUfunc all-DType, masked overlap, linalg output pair aliasing, indexing & scientific edge cases)',
        () {
          NDArray.scope(() {
            // 1.1: Full-width uint8 [0, 256), int16 [-32768, 32768), and wide int64 randint (normal & secure)
            final rU8Sec = randint(
              [16],
              low: 0,
              high: 256,
              dtype: DType.uint8,
              secure: true,
            );
            expect(rU8Sec.dtype, equals(DType.uint8));
            expect(rU8Sec.shape, equals([16]));
            final rI16Sec = randint(
              [16],
              low: -32768,
              high: 32768,
              dtype: DType.int16,
              secure: true,
            );
            expect(rI16Sec.dtype, equals(DType.int16));
            final rI64Wide = randint(
              [8],
              low: -5000000000000000000,
              high: 5000000000000000000,
              dtype: DType.int64,
              seed: 42,
            );
            expect(rI64Wide.dtype, equals(DType.int64));
            final rI64WideSec = randint(
              [8],
              low: -5000000000000000000,
              high: 5000000000000000000,
              dtype: DType.int64,
              secure: true,
            );
            expect(rI64WideSec.dtype, equals(DType.int64));

            // 1.2: std on 3-D array with non-contiguous 2-D out view whose last stride == 1
            final a3d = NDArray<Float64>.arange(
              0.0,
              24.0,
              dtype: DType.float64,
            ).reshape([2, 3, 4]);
            final stdExpected = std(a3d, axis: 2);
            final stdCarrier = NDArray<Float64>.full(
              [4, 3],
              999.0,
              dtype: DType.float64,
            );
            final stdOutView = stdCarrier.slice([
              const Slice(start: 0, stop: 4, step: 2),
              const Slice.all(),
            ]);
            std(a3d, axis: 2, out: stdOutView);
            expect(allClose(stdOutView, stdExpected), isTrue);
            expect(stdCarrier[[1, 0]], equals(999.0));
            expect(stdCarrier[[3, 2]], equals(999.0));

            // 1.3: count_nonzero on non-contiguous view with axis == null
            final nzBase = NDArray<Float64>.fromList(
              [0.0, 1.0, 2.0, 0.0, 3.0, 4.0],
              [2, 3],
              DType.float64,
            );
            final nzTrans = nzBase.transpose();
            expect(count_nonzero(nzTrans).scalar, equals(4));

            // 1.4: pad 3-D with PadMode.constant and non-contiguous out view
            final padSrc3d = NDArray<Float64>.ones([2, 2, 2], DType.float64);
            final padExpected3d = pad(
              padSrc3d,
              PadWidth.all(1),
              mode: PadMode.constant,
              constantValues: PadValues.all(5.0),
            );
            final padCarrier = NDArray<Float64>.full(
              [8, 4, 4],
              -77.0,
              dtype: DType.float64,
            );
            final padOutView = padCarrier.slice([
              const Slice(start: 0, stop: 8, step: 2),
              const Slice.all(),
              const Slice.all(),
            ]);
            pad(
              padSrc3d,
              PadWidth.all(1),
              mode: PadMode.constant,
              constantValues: PadValues.all(5.0),
              out: padOutView,
            );
            expect(allClose(padOutView, padExpected3d), isTrue);
            expect(padCarrier[[1, 0, 0]], equals(-77.0));

            // 1.5: atUfunc when b is broadcast with 0 < rankB < rankA - 1
            final atA = NDArray<Float64>.zeros([4, 3, 2], DType.float64);
            final atIdx = NDArray<Int64>.fromList([1, 3], [2], DType.int64);
            final atB = NDArray<Float64>.fromList(
              [10.0, 20.0],
              [2],
              DType.float64,
            );
            atUfunc(atA, atIdx, atB, op: BinaryOp.add);
            expect(atA[[1, 0, 0]], equals(10.0));
            expect(atA[[1, 2, 1]], equals(20.0));
            expect(atA[[3, 1, 0]], equals(10.0));
            expect(atA[[3, 2, 1]], equals(20.0));
            expect(atA[[0, 0, 0]], equals(0.0));

            // 1.6: unaryUfunc & binaryUfunc across Complex128, Boolean, Float32 out:, cbrt(-8), signbit(-0.0), spacing(Float32)
            final cUfuncIn = NDArray<Complex128>.fromList(
              [const Complex(3.0, 4.0)],
              [1],
              DType.complex128,
            );
            final cNeg = unaryUfunc(cUfuncIn, op: UnaryOp.negative);
            expect(cNeg[[0]], equals(const Complex(-3.0, -4.0)));
            final cConj = unaryUfunc(cUfuncIn, op: UnaryOp.conj);
            expect(cConj[[0]], equals(const Complex(3.0, -4.0)));
            final bUfuncIn = NDArray<Boolean>.fromList(
              [true, false],
              [2],
              DType.boolean,
            );
            expect(
              unaryUfunc(bUfuncIn, op: UnaryOp.logicalNot).toList(),
              equals([false, true]),
            );
            final f32In = NDArray<Float32>.fromList(
              [1.0, 2.0],
              [2],
              DType.float32,
            );
            final f32Out = NDArray<Float32>.zeros([2], DType.float32);
            unaryUfunc(f32In, op: UnaryOp.sin, out: f32Out);
            expect(allClose(f32Out, sin(f32In)), isTrue);
            binaryUfunc(f32In, f32In, op: BinaryOp.divide, out: f32Out);
            expect(f32Out.toList(), equals([1.0, 1.0]));
            final sp32 = unaryUfunc(f32In, op: UnaryOp.spacing);
            expect(sp32.dtype, equals(DType.float32));

            final negEight = NDArray<Float64>.fromList(
              [-8.0],
              [1],
              DType.float64,
            );
            expect(
              unaryUfunc(negEight, op: UnaryOp.cbrt)[[0]],
              closeTo(-2.0, 1e-12),
            );
            final negZero = NDArray<Float64>.fromList(
              [-0.0, 0.0],
              [2],
              DType.float64,
            );
            expect(
              unaryUfunc(negZero, op: UnaryOp.signbit).toList(),
              equals([true, false]),
            );

            // 2.1: v_binary_impl overlap with where != null
            final ovA = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            final ovB = NDArray<Float64>.fromList(
              [10.0, 20.0, 30.0, 40.0],
              [4],
              DType.float64,
            );
            final ovMask = NDArray<Boolean>.fromList(
              [true, false, true, false],
              [4],
              DType.boolean,
            );
            add(ovA, ovB, out: ovA, where: ovMask);
            expect(ovA.toList(), equals([11.0, 2.0, 33.0, 4.0]));

            // 2.2: reciprocal in-place with masked-out zero does not throw IntegerDivisionByZeroException
            final recInt = NDArray<Int64>.fromList([0, 1, 2], [3], DType.int64);
            final recMask = NDArray<Boolean>.fromList(
              [false, true, true],
              [3],
              DType.boolean,
            );
            reciprocal(recInt, out: recInt, where: recMask);
            expect(recInt.toList(), equals([0, 1, 0]));

            // 2.4: real(int32Arr, out: int32Arr) and linalg aliased output-pair rejection
            final rInt32 = NDArray<Int32>.fromList([5, 6, 7], [3], DType.int32);
            real(rInt32, out: rInt32);
            expect(rInt32.toList(), equals([5, 6, 7]));

            final mat2x2 = NDArray<Float64>.fromList(
              [4.0, 1.0, 1.0, 3.0],
              [2, 2],
              DType.float64,
            );
            final outSc = NDArray<Float64>.zeros([], DType.float64);
            expect(
              () => slogdet(mat2x2, outSign: outSc, outLogdet: outSc),
              throwsArgumentError,
            );
            final out2x2 = NDArray<Float64>.zeros([2, 2], DType.float64);
            expect(
              () => schur(mat2x2, outT: out2x2, outZ: out2x2),
              throwsArgumentError,
            );
            expect(
              () => hessenberg(mat2x2, outH: out2x2, outQ: out2x2),
              throwsArgumentError,
            );

            // 3.3: multivariateNormal on Float32 without explicit dtype
            final mean32 = NDArray<Float32>.zeros([2], DType.float32);
            final cov32 = NDArray<Float32>.eye(2, DType.float32);
            final mvn32 = multivariateNormal(
              mean32,
              cov32,
              size: [3],
              seed: 11,
            );
            expect(mvn32.dtype, equals(DType.float32));
            expect(mvn32.shape, equals([3, 2]));

            // 3.4: pad on uint32 with constant >= 2^31
            final u32 = NDArray<Uint32>.fromList([1, 2, 3], [3], DType.uint32);
            final u32HighPad = pad(
              u32,
              PadWidth.all(1),
              mode: PadMode.constant,
              constantValues: PadValues.all(0x80000000),
            );
            expect(
              u32HighPad.toList(),
              equals([0x80000000, 1, 2, 3, 0x80000000]),
            );

            // 3.5: 2-D single-cell coordinate assignment with non-scalar NDArray throws ArgumentError
            final cellMat = NDArray<Float64>.zeros([2, 2], DType.float64);
            expect(
              () => cellMat[[0, 1]] = NDArray<Float64>.ones([
                2,
                2,
              ], DType.float64),
              throwsArgumentError,
            );

            // 3.7: cov ddof > N returns NaN, cosine pdist >= 0, irr trailing 0s, diag empty view, det/slogdet [0, 0]
            final covSmall = NDArray<Float64>.fromList(
              [1.0, 2.0],
              [2],
              DType.float64,
            );
            expect(cov(covSmall, ddof: 5).scalar.isNaN, isTrue);

            final cosVecs = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
              [2, 3],
              DType.float64,
            );
            expect(
              pdist(cosVecs, metric: DistanceMetric.cosine)[[0]],
              greaterThanOrEqualTo(0.0),
            );

            final cfTrailingZero = NDArray<Float64>.fromList(
              [-100.0, 110.0, 0.0],
              [3],
              DType.float64,
            );
            expect(irr(cfTrailingZero).scalar, closeTo(0.1, 1e-6));

            final emptyDiag = diag(mat2x2, k: 5);
            expect(emptyDiag.shape, equals([0]));
            expect(emptyDiag.isView, isTrue);

            final empty0x0 = NDArray<Float64>.zeros([0, 0], DType.float64);
            expect(det(empty0x0).scalar, equals(1.0));
            final (:sign, :logabsdet) = slogdet(empty0x0);
            expect(sign.scalar, equals(1.0));
            expect(logabsdet.scalar, equals(0.0));
          });
        },
      );

      test(
        'Round 4 audit verification: boolean subtract UnsupportedError, Float16/BFloat16 modf/frexp/unwrap/polyfit promotion, orthogonal polynomial (x, c) order, argsortAs stability, bincount Float16 weights, spacing/signbit, and non-contiguous indexing assignment',
        () {
          NDArray.scope(() {
            // 1. Boolean subtract throws UnsupportedError (matching NumPy)
            final b1 = NDArray<Boolean>.fromList(
              [true, false],
              [2],
              DType.boolean,
            );
            final b2 = NDArray<Boolean>.fromList(
              [false, true],
              [2],
              DType.boolean,
            );
            expect(() => subtract(b1, b2), throwsUnsupportedError);
            expect(() => b1 - b2, throwsUnsupportedError);
            expect(
              () => binaryUfunc(b1, b2, op: BinaryOp.subtract),
              throwsUnsupportedError,
            );
            expect(
              () => reduceUfunc(b1, op: BinaryOp.subtract),
              throwsUnsupportedError,
            );
            expect(
              () => accumulateUfunc(b1, op: BinaryOp.subtract),
              throwsUnsupportedError,
            );

            // 2. Float16 / BFloat16 default promotion to Float64 in modf, frexp, unwrap, polyfit
            final f16 = NDArray<Float16>.fromList(
              [1.5, 2.5, 3.5],
              [3],
              DType.float16,
            );
            final (:fractional, :integral) = modf(f16);
            expect(fractional.dtype, equals(DType.float64));
            expect(integral.dtype, equals(DType.float64));
            expect(fractional[[0]], closeTo(0.5, 1e-3));
            expect(integral[[0]], closeTo(1.0, 1e-3));

            final (:mantissa, :exponent) = frexp(f16);
            expect(mantissa.dtype, equals(DType.float64));
            expect(exponent.dtype, equals(DType.int32));

            final uw16 = unwrap(f16);
            expect(uw16.dtype, equals(DType.float64));

            final pf16 = polyfit(f16, f16, 1);
            expect(pf16.dtype, equals(DType.float64));

            // 3. Orthogonal polynomial (x, c) order even when x.length < c.length
            final xShort = NDArray<Float64>.fromList(
              [0.5, 1.0],
              [2],
              DType.float64,
            );
            final cLong = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0],
              [3],
              DType.float64,
            );
            final chebRes = chebval(xShort, cLong);
            expect(chebRes.shape, equals([2]));
            expect(legval(xShort, cLong).shape, equals([2]));
            expect(hermval(xShort, cLong).shape, equals([2]));
            expect(lagval(xShort, cLong).shape, equals([2]));

            // 4. argsortAs stability on equal elements
            final eqArr = NDArray<Float64>.fromList(
              [2.0, 1.0, 2.0, 1.0],
              [4],
              DType.float64,
            );
            final stableIdx = argsortAs(
              eqArr,
              DType.int32,
              kind: SortKind.stable,
            );
            expect(stableIdx.toList(), equals([1, 3, 0, 2]));

            // 5. bincount with Float16 weights does not overflow
            final bins = NDArray<Int64>.fromList(
              [0, 1, 1, 2],
              [4],
              DType.int64,
            );
            final w16 = NDArray<Float16>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float16,
            );
            final bc = bincount(bins, weights: w16);
            expect(bc.toList(), equals([1.0, 5.0, 4.0]));

            // 6. spacing on 0.0 / Inf / NaN and signbit on Boolean / Uint64
            final spIn = NDArray<Float64>.fromList(
              [0.0, double.infinity, double.nan],
              [3],
              DType.float64,
            );
            final spOut = unaryUfunc(spIn, op: UnaryOp.spacing);
            expect(spOut[[0]], equals(double.minPositive));
            expect(spOut[[1]].isNaN, isTrue);
            expect(spOut[[2]].isNaN, isTrue);
            expect(
              unaryUfunc(b1, op: UnaryOp.signbit).toList(),
              equals([false, false]),
            );

            // 7. Non-contiguous view setByMask / setIndices / slice assignment
            final carrier = NDArray<Float64>.zeros([6], DType.float64);
            final view = carrier.slice([
              const Slice(start: 1, stop: 6, step: 2),
            ]);
            final mask3 = NDArray<Boolean>.fromList(
              [true, false, true],
              [3],
              DType.boolean,
            );
            final vals2 = NDArray<Float64>.fromList(
              [10.0, 30.0],
              [2],
              DType.float64,
            );
            view.setByMask(mask3, vals2);
            expect(carrier.toList(), equals([0.0, 10.0, 0.0, 0.0, 0.0, 30.0]));

            final idx2 = NDArray<Int64>.fromList([0, 2], [2], DType.int64);
            view.setIndicesScalar(idx2, 77.0);
            expect(carrier.toList(), equals([0.0, 77.0, 0.0, 0.0, 0.0, 77.0]));
          });
        },
      );
    });
  });
}

bool sameId(Object a, Object b) => identical(a, b);
