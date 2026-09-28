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

/// Descriptor for an axis reduction `f(a, {axis, out})` on `NDArray<Float64>`.
final class ReductionOpSpec {
  final String name;
  final NDArray<Float64> Function(
    NDArray<Float64> a, {
    int? axis,
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
    UnaryOpSpec(
      'nan_to_num',
      (a, {out}) => nan_to_num(a, out: out) as NDArray<Float64>,
    ),
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
    BinaryOpSpec(
      'atan2',
      (a, b, {out}) =>
          atan2<Float64, Float64>(a, b, out: out) as NDArray<Float64>,
    ),
    BinaryOpSpec(
      'hypot',
      (a, b, {out}) => hypot<Float64, Float64, Float64>(a, b, out: out),
    ),
    BinaryOpSpec('copysign', (a, b, {out}) => copysign(a, b, out: out)),
    BinaryOpSpec('fmod', (a, b, {out}) => fmod(a, b, out: out)),
    BinaryOpSpec(
      'logaddexp',
      (a, b, {out}) => logaddexp(a, b, out: out) as NDArray<Float64>,
    ),
    BinaryOpSpec(
      'logaddexp2',
      (a, b, {out}) => logaddexp2(a, b, out: out) as NDArray<Float64>,
    ),
  ];

  final reductionOps = <ReductionOpSpec>[
    ReductionOpSpec('sum', (a, {axis, out}) => sum(a, axis: axis, out: out)),
    ReductionOpSpec('prod', (a, {axis, out}) => prod(a, axis: axis, out: out)),
    ReductionOpSpec('mean', (a, {axis, out}) => mean(a, axis: axis, out: out)),
    ReductionOpSpec('min', (a, {axis, out}) => min(a, axis: axis, out: out)),
    ReductionOpSpec('max', (a, {axis, out}) => max(a, axis: axis, out: out)),
    ReductionOpSpec('ptp', (a, {axis, out}) => ptp(a, axis: axis, out: out)),
    ReductionOpSpec('std', (a, {axis, out}) => std(a, axis: axis, out: out)),
    ReductionOpSpec('var_', (a, {axis, out}) => var_(a, axis: axis, out: out)),
    ReductionOpSpec(
      'nansum',
      (a, {axis, out}) => nansum(a, axis: axis, out: out),
    ),
    ReductionOpSpec(
      'nanmean',
      (a, {axis, out}) => nanmean(a, axis: axis, out: out),
    ),
    ReductionOpSpec(
      'nanmin',
      (a, {axis, out}) => nanmin(a, axis: axis, out: out),
    ),
    ReductionOpSpec(
      'nanmax',
      (a, {axis, out}) => nanmax(a, axis: axis, out: out),
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

              expect(
                () => op.call(a, axis: 5, out: out),
                throwsA(anyOf(isA<RangeError>(), isA<ArgumentError>())),
              );
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
                    (x, y) => atan2(x, y) as NDArray<Float64>,
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

            expect(() => view.fill(sampleScalar), throwsArgumentError);
            expect(
              () => view.setCell([0, 0], sampleScalar),
              throwsArgumentError,
            );
            expect(
              () => view.setCellFlat(0, sampleScalar),
              throwsArgumentError,
            );
            expect(() => view.setCellRaw(0, sampleScalar), throwsArgumentError);
            expect(
              () => view.setByMask(mask, replacementRow),
              throwsArgumentError,
            );
            expect(
              () => view.setByMaskScalar(mask, sampleScalar),
              throwsArgumentError,
            );
            expect(
              () => view.setIndices(indices1D, replacementRow),
              throwsArgumentError,
            );
            expect(
              () => view.setIndicesScalar(indices1D, sampleScalar),
              throwsArgumentError,
            );
            expect(
              () => view.sliceAssign([const Slice.all()], sampleScalar),
              throwsArgumentError,
            );
            expect(() => view[0] = sampleScalar, throwsArgumentError);
            expect(() => view[mask] = sampleScalar, throwsArgumentError);
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
  });
}

bool sameId(Object a, Object b) => identical(a, b);
