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

/// Compares two NDArrays element-by-element across any dtype, treating
/// `NaN == NaN` as equal and checking signed zero bits on float/complex when
/// [checkSignedZero] is true.
void expectArraysEquivalent<T extends DTypeTag>(
  NDArray<T> actual,
  NDArray<T> expected, {
  required String reason,
  double rtol = 1e-5,
  double atol = 1e-6,
  bool checkSignedZero = false,
}) {
  expect(
    actual.dtype,
    equals(expected.dtype),
    reason: '$reason (dtype mismatch)',
  );
  expect(
    actual.shape,
    equals(expected.shape),
    reason: '$reason (shape mismatch)',
  );

  final actualFlat = actual.copy().ravel();
  final expectedFlat = expected.copy().ravel();
  final n = actualFlat.size;

  for (var i = 0; i < n; i++) {
    final Object? a = actualFlat[[i]];
    final Object? e = expectedFlat[[i]];

    if (a is double && e is double) {
      if (a.isNaN && e.isNaN) continue;
      if (a.isInfinite || e.isInfinite) {
        expect(a, equals(e), reason: '$reason at flat index $i: $a vs $e');
        continue;
      }
      if (checkSignedZero && a == 0.0 && e == 0.0) {
        expect(
          a.isNegative,
          equals(e.isNegative),
          reason: '$reason signed zero mismatch at flat index $i: $a vs $e',
        );
        continue;
      }
      final tol =
          (actual.dtype == DType.float16 || actual.dtype == DType.bfloat16)
          ? 2e-2 + 2e-2 * e.abs()
          : (actual.dtype == DType.float32
                ? 1e-4 + 1e-4 * e.abs()
                : atol + rtol * e.abs());
      expect(
        (a - e).abs(),
        lessThanOrEqualTo(tol),
        reason: '$reason at flat index $i: $a vs $e (tol=$tol)',
      );
    } else if (a is Complex && e is Complex) {
      final aRe = a.real;
      final aIm = a.imag;
      final eRe = e.real;
      final eIm = e.imag;
      if (aRe.isNaN && eRe.isNaN && aIm.isNaN && eIm.isNaN) continue;
      if (aRe.isNaN || eRe.isNaN) {
        expect(
          aRe.isNaN,
          equals(eRe.isNaN),
          reason: '$reason real NaN at $i: $a vs $e',
        );
      } else if (aRe.isInfinite || eRe.isInfinite) {
        expect(aRe, equals(eRe), reason: '$reason real Inf at $i: $a vs $e');
      } else {
        final tol = actual.dtype == DType.complex64
            ? 1e-4 + 1e-4 * eRe.abs()
            : atol + rtol * eRe.abs();
        expect(
          (aRe - eRe).abs(),
          lessThanOrEqualTo(tol),
          reason: '$reason real at $i: $a vs $e',
        );
      }
      if (aIm.isNaN || eIm.isNaN) {
        expect(
          aIm.isNaN,
          equals(eIm.isNaN),
          reason: '$reason imag NaN at $i: $a vs $e',
        );
      } else if (aIm.isInfinite || eIm.isInfinite) {
        expect(aIm, equals(eIm), reason: '$reason imag Inf at $i: $a vs $e');
      } else {
        final tol = actual.dtype == DType.complex64
            ? 1e-4 + 1e-4 * eIm.abs()
            : atol + rtol * eIm.abs();
        expect(
          (aIm - eIm).abs(),
          lessThanOrEqualTo(tol),
          reason: '$reason imag at $i: $a vs $e',
        );
      }
    } else {
      expect(a, equals(e), reason: '$reason at flat index $i: $a vs $e');
    }
  }
}

/// Creates a strided view of [contiguous] with `step: 2` along axis 0,
/// so that `.copy()` yields the exact same logical elements as [contiguous].
NDArray<T> makeStep2View<T extends DTypeTag>(NDArray<T> contiguous) {
  final shape = contiguous.shape;
  final n0 = shape[0];
  final restShape = shape.sublist(1);
  final expandedShape = [n0 * 2, ...restShape];
  final buf = NDArray<T>.zeros(expandedShape, contiguous.dtype);
  final view = buf.slice([
    Slice(start: 0, stop: n0 * 2, step: 2),
    for (var d = 1; d < shape.length; d++) const Slice.all(),
  ]);
  contiguous.copy(out: view);
  return view;
}

/// Creates a negative-stride (`step: -1`) view along axis 0 whose logical
/// elements equal [contiguous].
NDArray<T> makeNegativeStrideView<T extends DTypeTag>(NDArray<T> contiguous) {
  final reversedCopy = flip(contiguous, axis: 0).copy();
  return flip(reversedCopy, axis: 0);
}

/// Creates a rank-3 `[1, 3, N]` strided view from a 1-D `[N]` array
/// to exercise the general N-D strided loop.
NDArray<T> makeRank3StridedView<T extends DTypeTag>(NDArray<T> contiguous1D) {
  final n = contiguous1D.shape[0];
  final base = NDArray<T>.zeros([2, 3, n * 2], contiguous1D.dtype);
  final view = base.slice([
    const Slice(start: 0, stop: 1),
    const Slice(start: 0, stop: 3),
    Slice(start: 0, stop: n * 2, step: 2),
  ]);
  for (var r = 0; r < 3; r++) {
    final rowSlice = view.slice([
      const Slice(start: 0, stop: 1),
      Slice(start: r, stop: r + 1),
      const Slice.all(),
    ]);
    contiguous1D.reshape([1, 1, n]).copy(out: rowSlice);
  }
  return view;
}

/// Generates a deterministic 1-D contiguous test array of length [length]
/// (default 25 to cover SIMD vector widths + scalar tail) for [dtype].
NDArray<T> makeSample1D<T extends DTypeTag>(
  DType<T> dtype, {
  int length = 25,
  int seedOffset = 0,
  bool positiveNonZero = false,
  bool smallShiftAmounts = false,
}) {
  switch (dtype) {
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
      final list = List<double>.generate(length, (i) {
        final v = ((i + seedOffset) % 11) - 5;
        if (positiveNonZero) return (v.abs() + 1).toDouble() * 0.5;
        return v == 0 ? 0.5 : v.toDouble() * 0.5;
      });
      return NDArray<T>.fromList(list.cast<dynamic>(), [length], dtype);
    case DType.complex128:
    case DType.complex64:
      final list = List<Complex>.generate(length, (i) {
        final re = (((i + seedOffset) % 7) + 1) * 0.5;
        final im = (((i + seedOffset * 2) % 5) - 2) * 0.25;
        return Complex(re, positiveNonZero ? im.abs() + 0.25 : im);
      });
      return NDArray<T>.fromList(list.cast<dynamic>(), [length], dtype);
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.int8:
      final list = List<int>.generate(length, (i) {
        if (smallShiftAmounts) return (i + seedOffset) % 5;
        final v = ((i + seedOffset) % 9) - 4;
        if (positiveNonZero) return v.abs() + 1;
        return v == 0 ? 2 : v;
      });
      return NDArray<T>.fromList(list.cast<dynamic>(), [length], dtype);
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
      final list = List<int>.generate(length, (i) {
        if (smallShiftAmounts) return (i + seedOffset) % 5;
        return ((i + seedOffset) % 9) + 1;
      });
      return NDArray<T>.fromList(list.cast<dynamic>(), [length], dtype);
    case DType.boolean:
      final list = List<bool>.generate(
        length,
        (i) => ((i + seedOffset) % 3) != 0,
      );
      return NDArray<T>.fromList(list.cast<dynamic>(), [length], dtype);
  }
}

void main() {
  const all15DTypes = <DType<AnySpec>>[
    DType.float64,
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.complex128,
    DType.complex64,
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

  const integerDTypes = <DType<AnySpec>>[
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
  ];

  const realNumericDTypes = <DType<AnySpec>>[
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
  ];

  const numericDTypes = <DType<AnySpec>>[
    DType.float64,
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.complex128,
    DType.complex64,
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint64,
    DType.uint32,
    DType.uint16,
    DType.uint8,
  ];

  const floatDTypes = <DType<AnySpec>>[
    DType.float64,
    DType.float32,
    DType.float16,
    DType.bfloat16,
  ];

  group(
    '1. Contiguous Fast-Path vs. Strided / Negative-Stride / Rank-3 Equivalence',
    () {
      void verifyUnaryVariants<T extends AnySpec, R extends DTypeTag>(
        String opName,
        DType<T> dtype,
        NDArray<R> Function(NDArray<T> x) op, {
        bool positiveNonZero = false,
      }) {
        NDArray.scope(() {
          // Length 25 exercises SIMD loop (4/8/16 elements) + scalar tail.
          final aContig = makeSample1D(
            dtype,
            length: 25,
            positiveNonZero: positiveNonZero,
          );
          final aStep2 = makeStep2View(aContig);
          final aNeg = makeNegativeStrideView(aContig);

          final expected = op(aContig);
          final resStep2 = op(aStep2);
          final resNeg = op(aNeg);

          expectArraysEquivalent(
            resStep2,
            expected,
            reason: '$opName($dtype) step:2 vs contiguous',
          );
          expectArraysEquivalent(
            resNeg,
            expected,
            reason: '$opName($dtype) step:-1 vs contiguous',
          );

          // Rank-3 [1, 3, 25] strided vs contiguous
          final r3Strided = makeRank3StridedView(aContig);
          final r3Contig = r3Strided.copy();
          expectArraysEquivalent(
            op(r3Strided),
            op(r3Contig),
            reason: '$opName($dtype) rank-3 [1,3,25] strided vs contiguous',
          );
        });
      }

      void verifyBinaryVariants<T extends AnySpec, R extends DTypeTag>(
        String opName,
        DType<T> dtype,
        NDArray<R> Function(NDArray<T> a, NDArray<T> b) op, {
        bool positiveNonZero = false,
        bool smallShiftB = false,
      }) {
        NDArray.scope(() {
          final aContig = makeSample1D(
            dtype,
            length: 25,
            seedOffset: 0,
            positiveNonZero: positiveNonZero,
          );
          final bContig = makeSample1D(
            dtype,
            length: 25,
            seedOffset: 3,
            positiveNonZero: positiveNonZero,
            smallShiftAmounts: smallShiftB,
          );

          final aStep2 = makeStep2View(aContig);
          final bStep2 = makeStep2View(bContig);
          final aNeg = makeNegativeStrideView(aContig);
          final bNeg = makeNegativeStrideView(bContig);

          final expected = op(aContig, bContig);
          expectArraysEquivalent(
            op(aStep2, bStep2),
            expected,
            reason: '$opName($dtype) step:2/step:2 vs contiguous',
          );
          expectArraysEquivalent(
            op(aNeg, bNeg),
            expected,
            reason: '$opName($dtype) step:-1/step:-1 vs contiguous',
          );
          expectArraysEquivalent(
            op(aStep2, bNeg),
            expected,
            reason: '$opName($dtype) step:2/step:-1 vs contiguous',
          );

          // Rank-3 [1, 3, 25] strided vs contiguous
          final r3AStrided = makeRank3StridedView(aContig);
          final r3BStrided = makeRank3StridedView(bContig);
          final r3AContig = r3AStrided.copy();
          final r3BContig = r3BStrided.copy();
          expectArraysEquivalent(
            op(r3AStrided, r3BStrided),
            op(r3AContig, r3BContig),
            reason: '$opName($dtype) rank-3 [1,3,25] strided vs contiguous',
          );
        });
      }

      test('Unary ufuncs across all applicable DTypes', () {
        for (final dtype in numericDTypes) {
          verifyUnaryVariants('abs', dtype, (x) => abs(x));
          verifyUnaryVariants('negative', dtype, (x) => negative(x));
          verifyUnaryVariants('sign', dtype, (x) => sign(x));
          verifyUnaryVariants('square', dtype, (x) => square(x));
          verifyUnaryVariants(
            'sqrt',
            dtype,
            (x) => sqrt(x),
            positiveNonZero: true,
          );
          verifyUnaryVariants('exp', dtype, (x) => exp(x));
          verifyUnaryVariants(
            'log',
            dtype,
            (x) => log(x),
            positiveNonZero: true,
          );
          verifyUnaryVariants('sin', dtype, (x) => sin(x));
          verifyUnaryVariants('cos', dtype, (x) => cos(x));
          verifyUnaryVariants('isnan', dtype, (x) => isnan(x));
          verifyUnaryVariants('isinf', dtype, (x) => isinf(x));
          verifyUnaryVariants('isfinite', dtype, (x) => isfinite(x));
        }

        // bitwise_not (invert) across integer + boolean dtypes
        for (final dtype in integerDTypes) {
          verifyUnaryVariants('bitwise_not', dtype, (x) => invert(x));
        }
        verifyUnaryVariants('bitwise_not', DType.boolean, (x) => invert(x));

        // logical_not across all 15 dtypes
        for (final dtype in all15DTypes) {
          verifyUnaryVariants('logical_not', dtype, (x) => logicalNot(x));
        }
      });

      test('Binary ufuncs across all applicable DTypes', () {
        for (final dtype in numericDTypes) {
          verifyBinaryVariants('add', dtype, (a, b) => add(a, b));
          verifyBinaryVariants('subtract', dtype, (a, b) => subtract(a, b));
          verifyBinaryVariants('multiply', dtype, (a, b) => multiply(a, b));
          verifyBinaryVariants(
            'divide',
            dtype,
            (a, b) => divide(a, b),
            positiveNonZero: true,
          );
          verifyBinaryVariants(
            'power',
            dtype,
            (a, b) => power(a, b),
            positiveNonZero: true,
            smallShiftB: true,
          );
          verifyBinaryVariants(
            'minimum',
            dtype,
            (a, b) => binaryUfunc(a, b, op: BinaryOp.minimum),
          );
          verifyBinaryVariants(
            'maximum',
            dtype,
            (a, b) => binaryUfunc(a, b, op: BinaryOp.maximum),
          );
          verifyBinaryVariants(
            'fmin',
            dtype,
            (a, b) => binaryUfunc(a, b, op: BinaryOp.fmin),
          );
          verifyBinaryVariants(
            'fmax',
            dtype,
            (a, b) => binaryUfunc(a, b, op: BinaryOp.fmax),
          );
        }

        for (final dtype in realNumericDTypes) {
          verifyBinaryVariants(
            'floor_divide',
            dtype,
            (a, b) => floorDivide(a, b),
            positiveNonZero: true,
          );
          verifyBinaryVariants(
            'remainder',
            dtype,
            (a, b) => remainder(a, b),
            positiveNonZero: true,
          );
          verifyBinaryVariants(
            'fmod',
            dtype,
            (a, b) => fmod(a, b),
            positiveNonZero: true,
          );
          verifyBinaryVariants('less', dtype, (a, b) => less(a, b));
          verifyBinaryVariants('greater', dtype, (a, b) => greater(a, b));
        }

        for (final dtype in integerDTypes) {
          verifyBinaryVariants(
            'gcd',
            dtype,
            (a, b) => gcd(a, b),
            positiveNonZero: true,
          );
          verifyBinaryVariants(
            'lcm',
            dtype,
            (a, b) => lcm(a, b),
            positiveNonZero: true,
          );
          verifyBinaryVariants(
            'bitwise_and',
            dtype,
            (a, b) => bitwiseAnd(a, b),
          );
          verifyBinaryVariants('bitwise_or', dtype, (a, b) => bitwiseOr(a, b));
          verifyBinaryVariants(
            'bitwise_xor',
            dtype,
            (a, b) => bitwiseXor(a, b),
          );
          verifyBinaryVariants(
            'left_shift',
            dtype,
            (a, b) => leftShift(a, b),
            positiveNonZero: true,
            smallShiftB: true,
          );
          verifyBinaryVariants(
            'right_shift',
            dtype,
            (a, b) => rightShift(a, b),
            positiveNonZero: true,
            smallShiftB: true,
          );
        }

        verifyBinaryVariants(
          'logical_and',
          DType.boolean,
          (a, b) => logicalAnd(a, b),
        );
        verifyBinaryVariants(
          'logical_or',
          DType.boolean,
          (a, b) => logicalOr(a, b),
        );
        verifyBinaryVariants(
          'logical_xor',
          DType.boolean,
          (a, b) => logicalXor(a, b),
        );

        for (final dtype in all15DTypes) {
          verifyBinaryVariants('equal', dtype, (a, b) => equal(a, b));
        }
      });

      test('Reductions across all 15 DTypes (full and axis reductions)', () {
        void verifyReduction<T extends AnySpec, R extends DTypeTag>(
          String opName,
          DType<T> dtype,
          NDArray<R> Function(NDArray<T> x, {int? axis}) op,
        ) {
          NDArray.scope(() {
            final aContig = makeSample1D(
              dtype,
              length: 20,
              positiveNonZero: true,
              smallShiftAmounts: true,
            );
            final aStep2 = makeStep2View(aContig);
            final aNeg = makeNegativeStrideView(aContig);

            expectArraysEquivalent(
              op(aStep2),
              op(aContig),
              reason: '$opName($dtype) full reduction step:2 vs contiguous',
            );
            expectArraysEquivalent(
              op(aNeg),
              op(aContig),
              reason: '$opName($dtype) full reduction step:-1 vs contiguous',
            );

            // Rank-3 [1, 3, 20] strided reductions along axis 0, 1, 2 and full
            final r3Strided = makeRank3StridedView(aContig);
            final r3Contig = r3Strided.copy();
            for (final axis in <int?>[null, 0, 1, 2]) {
              expectArraysEquivalent(
                op(r3Strided, axis: axis),
                op(r3Contig, axis: axis),
                reason: '$opName($dtype) rank-3 axis=$axis strided vs contig',
              );
            }
          });
        }

        for (final dtype in all15DTypes) {
          verifyReduction('sum', dtype, (x, {axis}) => sum(x, axis: axis));
          verifyReduction('prod', dtype, (x, {axis}) => prod(x, axis: axis));
          verifyReduction('all', dtype, (x, {axis}) => all(x, axis: axis));
          verifyReduction('any', dtype, (x, {axis}) => any(x, axis: axis));
        }

        for (final dtype in numericDTypes) {
          verifyReduction(
            'nansum',
            dtype,
            (x, {axis}) => nansum(x, axis: axis),
          );
          verifyReduction('mean', dtype, (x, {axis}) => mean(x, axis: axis));
          verifyReduction(
            'nanmean',
            dtype,
            (x, {axis}) => nanmean(x, axis: axis),
          );
          verifyReduction('var_', dtype, (x, {axis}) => var_(x, axis: axis));
          verifyReduction(
            'nanvar',
            dtype,
            (x, {axis}) => nanvar(x, axis: axis),
          );
          verifyUnaryVariants('cumsum', dtype, (x) => cumsum(x, axis: -1));
          verifyUnaryVariants(
            'cumprod',
            dtype,
            (x) => cumprod(x, axis: -1),
            positiveNonZero: true,
          );
        }

        for (final dtype in realNumericDTypes) {
          verifyReduction('min', dtype, (x, {axis}) => min(x, axis: axis));
          verifyReduction('max', dtype, (x, {axis}) => max(x, axis: axis));
          verifyReduction(
            'nanmin',
            dtype,
            (x, {axis}) => nanmin(x, axis: axis),
          );
          verifyReduction(
            'nanmax',
            dtype,
            (x, {axis}) => nanmax(x, axis: axis),
          );
          verifyReduction('ptp', dtype, (x, {axis}) => ptp(x, axis: axis));
        }
      });

      test('Indexing, padding, sorting & searching across all 15 DTypes', () {
        for (final dtype in all15DTypes) {
          NDArray.scope(() {
            final aContig = makeSample1D(dtype, length: 18);
            final aStep2 = makeStep2View(aContig);
            final aNeg = makeNegativeStrideView(aContig);

            // take_along_axis
            final indices = NDArray<Int64>.fromList(
              [0, 5, 17, 3, 12, 8],
              [6],
              DType.int64,
            );
            final idxStep2 = makeStep2View(indices);
            expectArraysEquivalent(
              take_along_axis(aStep2, idxStep2, 0),
              take_along_axis(aContig, indices, 0),
              reason: 'take_along_axis($dtype) strided vs contiguous',
            );
            expectArraysEquivalent(
              take_along_axis(aNeg, indices, 0),
              take_along_axis(aContig, indices, 0),
              reason: 'take_along_axis($dtype) neg stride vs contiguous',
            );

            // put_along_axis
            final vals = makeSample1D(dtype, length: 6, seedOffset: 2);
            final dstContig = aContig.copy();
            final dstStep2 = makeStep2View(aContig);
            put_along_axis(dstContig, indices, vals, 0);
            put_along_axis(dstStep2, idxStep2, makeStep2View(vals), 0);
            expectArraysEquivalent(
              dstStep2,
              dstContig,
              reason: 'put_along_axis($dtype) strided vs contiguous',
            );

            // pad modes: constant, edge, reflect, symmetric, wrap
            for (final mode in [
              PadMode.constant,
              PadMode.edge,
              PadMode.reflect,
              PadMode.symmetric,
              PadMode.wrap,
            ]) {
              expectArraysEquivalent(
                pad(aStep2, PadWidth.axes([(2, 3)]), mode: mode),
                pad(aContig, PadWidth.axes([(2, 3)]), mode: mode),
                reason: 'pad($dtype, $mode) step:2 vs contiguous',
              );
              expectArraysEquivalent(
                pad(aNeg, PadWidth.axes([(2, 3)]), mode: mode),
                pad(aContig, PadWidth.axes([(2, 3)]), mode: mode),
                reason: 'pad($dtype, $mode) step:-1 vs contiguous',
              );
            }

            // roll
            expectArraysEquivalent(
              roll(aStep2, 4),
              roll(aContig, 4),
              reason: 'roll($dtype) step:2 vs contiguous',
            );
            expectArraysEquivalent(
              roll(aNeg, -3),
              roll(aContig, -3),
              reason: 'roll($dtype) step:-1 vs contiguous',
            );

            // nonzero and count_nonzero
            final nzContig = nonzero(aContig);
            final nzStep2 = nonzero(aStep2);
            final nzNeg = nonzero(aNeg);
            expectArraysEquivalent(
              nzStep2[0],
              nzContig[0],
              reason: 'nonzero($dtype) step:2 vs contiguous',
            );
            expectArraysEquivalent(
              nzNeg[0],
              nzContig[0],
              reason: 'nonzero($dtype) step:-1 vs contiguous',
            );
            expectArraysEquivalent(
              count_nonzero(aStep2),
              count_nonzero(aContig),
              reason: 'count_nonzero($dtype) step:2 vs contiguous',
            );
            expectArraysEquivalent(
              count_nonzero(aNeg),
              count_nonzero(aContig),
              reason: 'count_nonzero($dtype) step:-1 vs contiguous',
            );
          });
        }

        // diff across numeric dtypes
        for (final dtype in numericDTypes) {
          verifyUnaryVariants('diff', dtype, (x) => diff(x, axis: -1));
        }

        // sort, argsort, partition, argpartition, uniqueAll, searchsorted, argmax, argmin
        for (final dtype in realNumericDTypes) {
          NDArray.scope(() {
            final aContig = makeSample1D(dtype, length: 25);
            final aStep2 = makeStep2View(aContig);
            final aNeg = makeNegativeStrideView(aContig);

            expectArraysEquivalent(
              sort(aStep2),
              sort(aContig),
              reason: 'sort($dtype) step:2 vs contiguous',
            );
            expectArraysEquivalent(
              sort(aNeg),
              sort(aContig),
              reason: 'sort($dtype) step:-1 vs contiguous',
            );
            expectArraysEquivalent(
              argsort(aStep2),
              argsort(aContig),
              reason: 'argsort($dtype) step:2 vs contiguous',
            );
            expectArraysEquivalent(
              argsort(aNeg),
              argsort(aContig),
              reason: 'argsort($dtype) step:-1 vs contiguous',
            );

            // partition & argpartition: kth element must equal sorted[kth]
            const kth = 10;
            final sortedRef = sort(aContig);
            final Object? kthVal = sortedRef[[kth]];
            for (final variant in [aContig, aStep2, aNeg]) {
              final part = partition(variant, kth);
              expect(
                part[[kth]],
                equals(kthVal),
                reason: 'partition($dtype) kth value',
              );
              final argpart = argpartition(variant, kth);
              final gathered = take_along_axis(variant, argpart, 0);
              expect(
                gathered[[kth]],
                equals(kthVal),
                reason: 'argpartition($dtype) kth value',
              );
            }

            // uniqueAll
            final uContig = uniqueAll(aContig);
            final uStep2 = uniqueAll(aStep2);
            final uNeg = uniqueAll(aNeg);
            expectArraysEquivalent(
              uStep2.values,
              uContig.values,
              reason: 'uniqueAll($dtype).values step:2',
            );
            expectArraysEquivalent(
              uStep2.index,
              uContig.index,
              reason: 'uniqueAll($dtype).index step:2',
            );
            expectArraysEquivalent(
              uStep2.inverse,
              uContig.inverse,
              reason: 'uniqueAll($dtype).inverse step:2',
            );
            expectArraysEquivalent(
              uStep2.counts,
              uContig.counts,
              reason: 'uniqueAll($dtype).counts step:2',
            );
            expectArraysEquivalent(
              uNeg.values,
              uContig.values,
              reason: 'uniqueAll($dtype).values step:-1',
            );

            // searchsorted
            final sortedStep2 = makeStep2View(sortedRef);
            final queries = makeSample1D(dtype, length: 9, seedOffset: 4);
            final queriesStep2 = makeStep2View(queries);
            final queriesNeg = makeNegativeStrideView(queries);
            expectArraysEquivalent(
              searchsorted(sortedStep2, queriesStep2),
              searchsorted(sortedRef, queries),
              reason: 'searchsorted($dtype) left strided vs contiguous',
            );
            expectArraysEquivalent(
              searchsorted(sortedStep2, queriesNeg, side: SearchSide.right),
              searchsorted(sortedRef, queries, side: SearchSide.right),
              reason: 'searchsorted($dtype) right strided vs contiguous',
            );

            // argmax & argmin (1-D and Rank-3)
            expectArraysEquivalent(
              argmax(aStep2),
              argmax(aContig),
              reason: 'argmax($dtype) step:2 vs contiguous',
            );
            expectArraysEquivalent(
              argmax(aNeg),
              argmax(aContig),
              reason: 'argmax($dtype) step:-1 vs contiguous',
            );
            expectArraysEquivalent(
              argmin(aStep2),
              argmin(aContig),
              reason: 'argmin($dtype) step:2 vs contiguous',
            );
            expectArraysEquivalent(
              argmin(aNeg),
              argmin(aContig),
              reason: 'argmin($dtype) step:-1 vs contiguous',
            );

            final r3Strided = makeRank3StridedView(aContig);
            final r3Contig = r3Strided.copy();
            expectArraysEquivalent(
              argmax(r3Strided, axis: 2),
              argmax(r3Contig, axis: 2),
              reason: 'argmax($dtype) rank-3 axis=2 strided vs contiguous',
            );
            expectArraysEquivalent(
              argmin(r3Strided, axis: 2),
              argmin(r3Contig, axis: 2),
              reason: 'argmin($dtype) rank-3 axis=2 strided vs contiguous',
            );
          });
        }
      });
    },
  );

  group('2. Adversarial Arithmetic & IEEE-754 Boundary Values', () {
    test(
      'Integer boundary values [0, 1, -1, minVal, maxVal], INT_MIN / -1, and div-by-zero',
      () {
        final signedSpecs = <(DType<AnySpec>, int, int)>[
          (DType.int8, -128, 127),
          (DType.int16, -32768, 32767),
          (DType.int32, -2147483648, 2147483647),
          (DType.int64, -9223372036854775808, 9223372036854775807),
        ];

        for (final (dtype, minVal, maxVal) in signedSpecs) {
          NDArray.scope(() {
            // Test signed minVal ~/ -1 and minVal % -1 (must not SIGFPE!)
            // Test both contiguous (length 25 for SIMD + tail) and strided views.
            final numerators = NDArray.fromList(
              List<int>.generate(
                25,
                (i) => i.isEven ? minVal : maxVal,
              ).cast<dynamic>(),
              [25],
              dtype,
            );
            final negOnes = NDArray.fromList(
              List<int>.filled(25, -1).cast<dynamic>(),
              [25],
              dtype,
            );
            final ones = NDArray.fromList(
              List<int>.filled(25, 1).cast<dynamic>(),
              [25],
              dtype,
            );

            final fdContig = floorDivide(numerators, negOnes);
            final fdStrided = floorDivide(
              makeStep2View(numerators),
              makeStep2View(negOnes),
            );
            expectArraysEquivalent(
              fdStrided,
              fdContig,
              reason: 'floorDivide($dtype) minVal ~/ -1 strided vs contiguous',
            );

            final remContig = remainder(numerators, negOnes);
            final remStrided = remainder(
              makeStep2View(numerators),
              makeStep2View(negOnes),
            );
            expectArraysEquivalent(
              remStrided,
              remContig,
              reason: 'remainder($dtype) minVal % -1 strided vs contiguous',
            );
            // minVal % -1 must be 0 everywhere
            for (var i = 0; i < 25; i++) {
              expect(
                remContig[[i]],
                equals(0),
                reason: 'remainder($dtype) at $i must be 0',
              );
            }

            final fmodContig = fmod(numerators, negOnes);
            final fmodStrided = fmod(
              makeStep2View(numerators),
              makeStep2View(negOnes),
            );
            expectArraysEquivalent(
              fmodStrided,
              fmodContig,
              reason: 'fmod($dtype) minVal % -1 strided vs contiguous',
            );
            for (var i = 0; i < 25; i++) {
              expect(
                fmodContig[[i]],
                equals(0),
                reason: 'fmod($dtype) at $i must be 0',
              );
            }

            // Boundary values [0, 1, -1, minVal, maxVal] with non-zero denominators
            final bounds = NDArray.fromList(
              [
                0,
                1,
                -1,
                minVal,
                maxVal,
                0,
                1,
                -1,
                minVal,
                maxVal,
              ].cast<dynamic>(),
              [10],
              dtype,
            );
            final denoms = NDArray.fromList(
              [1, -1, 1, -1, 1, -1, 1, -1, 1, -1].cast<dynamic>(),
              [10],
              dtype,
            );
            expectArraysEquivalent(
              add(makeStep2View(bounds), makeStep2View(denoms)),
              add(bounds, denoms),
              reason: 'add($dtype) boundaries strided vs contiguous',
            );
            expectArraysEquivalent(
              subtract(makeStep2View(bounds), makeStep2View(denoms)),
              subtract(bounds, denoms),
              reason: 'subtract($dtype) boundaries strided vs contiguous',
            );
            expectArraysEquivalent(
              multiply(makeStep2View(bounds), makeStep2View(denoms)),
              multiply(bounds, denoms),
              reason: 'multiply($dtype) boundaries strided vs contiguous',
            );
            expectArraysEquivalent(
              floorDivide(makeStep2View(bounds), makeStep2View(denoms)),
              floorDivide(bounds, denoms),
              reason: 'floorDivide($dtype) boundaries strided vs contiguous',
            );
            expectArraysEquivalent(
              remainder(makeStep2View(bounds), makeStep2View(denoms)),
              remainder(bounds, denoms),
              reason: 'remainder($dtype) boundaries strided vs contiguous',
            );
            expectArraysEquivalent(
              fmod(makeStep2View(bounds), makeStep2View(denoms)),
              fmod(bounds, denoms),
              reason: 'fmod($dtype) boundaries strided vs contiguous',
            );
            final ones10 = ones.slice([const Slice(start: 0, stop: 10)]);
            expectArraysEquivalent(
              gcd(makeStep2View(bounds), makeStep2View(ones10)),
              gcd(bounds, ones10),
              reason: 'gcd($dtype) boundaries strided vs contiguous',
            );
            expectArraysEquivalent(
              lcm(makeStep2View(bounds), makeStep2View(ones10)),
              lcm(bounds, ones10),
              reason: 'lcm($dtype) boundaries strided vs contiguous',
            );
          });
        }

        final unsignedSpecs = <(DType<AnySpec>, int)>[
          (DType.uint8, 255),
          (DType.uint16, 65535),
          (DType.uint32, 4294967295),
          (
            DType.uint64,
            -1,
          ), // 0xFFFFFFFFFFFFFFFF in Dart 64-bit two's complement
        ];

        for (final (dtype, maxVal) in unsignedSpecs) {
          NDArray.scope(() {
            final bounds = NDArray.fromList(
              [0, 1, maxVal, 0, 1, maxVal, 2, maxVal - 1].cast<dynamic>(),
              [8],
              dtype,
            );
            final denoms = NDArray.fromList(
              [1, 1, 1, 2, maxVal, maxVal, 1, 2].cast<dynamic>(),
              [8],
              dtype,
            );
            expectArraysEquivalent(
              add(makeStep2View(bounds), makeStep2View(denoms)),
              add(bounds, denoms),
              reason: 'add($dtype) unsigned boundaries',
            );
            expectArraysEquivalent(
              subtract(makeStep2View(bounds), makeStep2View(denoms)),
              subtract(bounds, denoms),
              reason: 'subtract($dtype) unsigned boundaries',
            );
            expectArraysEquivalent(
              multiply(makeStep2View(bounds), makeStep2View(denoms)),
              multiply(bounds, denoms),
              reason: 'multiply($dtype) unsigned boundaries',
            );
            expectArraysEquivalent(
              floorDivide(makeStep2View(bounds), makeStep2View(denoms)),
              floorDivide(bounds, denoms),
              reason: 'floorDivide($dtype) unsigned boundaries',
            );
            expectArraysEquivalent(
              remainder(makeStep2View(bounds), makeStep2View(denoms)),
              remainder(bounds, denoms),
              reason: 'remainder($dtype) unsigned boundaries',
            );
            expectArraysEquivalent(
              fmod(makeStep2View(bounds), makeStep2View(denoms)),
              fmod(bounds, denoms),
              reason: 'fmod($dtype) unsigned boundaries',
            );
          });
        }

        // Division / modulo by 0 must throw UnsupportedError across all 8 integer dtypes
        for (final dtype in integerDTypes) {
          NDArray.scope(() {
            final a = NDArray.fromList([10, 20, 30, 40].cast<dynamic>(), [
              4,
            ], dtype);
            final z = NDArray.fromList([1, 2, 0, 4].cast<dynamic>(), [
              4,
            ], dtype);
            expect(
              () => floorDivide(a, z),
              throwsUnsupportedError,
              reason: 'floorDivide($dtype) by 0 must throw UnsupportedError',
            );
            expect(
              () => remainder(a, z),
              throwsUnsupportedError,
              reason: 'remainder($dtype) by 0 must throw UnsupportedError',
            );
            expect(
              () => fmod(a, z),
              throwsUnsupportedError,
              reason: 'fmod($dtype) by 0 must throw UnsupportedError',
            );
            for (final divOp in [
              BinaryOp.floorDivide,
              BinaryOp.remainder,
              BinaryOp.fmod,
            ]) {
              expect(
                () => atUfunc(
                  a.copy(),
                  NDArray<Int64>.fromList([0, 2], [2], DType.int64),
                  NDArray.fromList([1, 0].cast<dynamic>(), [2], dtype),
                  op: divOp,
                ),
                throwsUnsupportedError,
                reason: 'atUfunc($dtype, $divOp) by 0 must throw',
              );
            }
            expect(
              () => reduceatUfunc(
                NDArray.fromList([10, 0, 5].cast<dynamic>(), [3], dtype),
                NDArray<Int64>.fromList([0], [1], DType.int64),
                op: BinaryOp.floorDivide,
              ),
              throwsArgumentError,
              reason:
                  'reduceatUfunc($dtype, floorDivide) must reject non-reducible op',
            );
          });
        }
      },
    );

    test(
      'atUfunc and reduceatUfunc across all 8 integer dtypes and all integer BinaryOps',
      () {
        const integerOps = <BinaryOp>[
          BinaryOp.add,
          BinaryOp.subtract,
          BinaryOp.multiply,
          BinaryOp.minimum,
          BinaryOp.maximum,
          BinaryOp.bitwiseAnd,
          BinaryOp.bitwiseOr,
          BinaryOp.bitwiseXor,
          BinaryOp.leftShift,
          BinaryOp.rightShift,
          BinaryOp.gcd,
          BinaryOp.lcm,
        ];

        for (final dtype in integerDTypes) {
          for (final op in integerOps) {
            NDArray.scope(() {
              // Use small positive integers so shifts/gcd/lcm/multiply are well-behaved
              final base = NDArray.fromList(
                [1, 2, 3, 4, 5, 6, 2, 3].cast<dynamic>(),
                [8],
                dtype,
              );
              final indices = NDArray<Int64>.fromList(
                [1, 3, 1, 5],
                [4],
                DType.int64,
              );
              final bVals = NDArray.fromList([1, 2, 2, 1].cast<dynamic>(), [
                4,
              ], dtype);

              // Reference sequential execution using 1-element binaryUfunc
              final expectedAt = base.copy();
              for (var k = 0; k < indices.size; k++) {
                final idx = indices[[k]];
                final cur = expectedAt.slice([
                  Slice(start: idx, stop: idx + 1),
                ]);
                final rhs = bVals.slice([Slice(start: k, stop: k + 1)]);
                final updated = binaryUfunc(cur, rhs, op: op);
                updated.copy(out: cur);
              }

              final actualAt = base.copy();
              atUfunc(actualAt, indices, bVals, op: op);
              expectArraysEquivalent(
                actualAt,
                expectedAt,
                reason: 'atUfunc($dtype, $op) vs sequential binaryUfunc',
              );

              // Also test strided target view for atUfunc
              final actualAtStrided = makeStep2View(base);
              atUfunc(
                actualAtStrided,
                makeStep2View(indices),
                makeStep2View(bVals),
                op: op,
              );
              expectArraysEquivalent(
                actualAtStrided,
                expectedAt,
                reason: 'atUfunc($dtype, $op) strided vs sequential',
              );

              // reduceatUfunc reference vs actual (for reducible ops)
              if (op.isReducible) {
                final reduceSplits = NDArray<Int64>.fromList(
                  [0, 3, 2, 5, 7],
                  [5],
                  DType.int64,
                );
                final actualReduceat = reduceatUfunc(
                  base,
                  reduceSplits,
                  op: op,
                );
                final stridedReduceat = reduceatUfunc(
                  makeStep2View(base),
                  reduceSplits,
                  op: op,
                );
                expectArraysEquivalent(
                  stridedReduceat,
                  actualReduceat,
                  reason: 'reduceatUfunc($dtype, $op) strided vs contiguous',
                );

                // Verify segment [0..3) matches manual left-fold of binaryUfunc
                var acc = base.slice([const Slice(start: 0, stop: 1)]).copy();
                for (var j = 1; j < 3; j++) {
                  acc = binaryUfunc(
                    acc,
                    base.slice([Slice(start: j, stop: j + 1)]),
                    op: op,
                  );
                }
                expect(
                  actualReduceat[[0]],
                  equals(acc[[0]]),
                  reason:
                      'reduceatUfunc($dtype, $op) segment [0..3) vs manual fold',
                );
              }

              // Also test with signed negative values (for int8..int64) and
              // MSB-set unsigned values (for uint8..uint64) on add, subtract,
              // multiply, minimum, maximum, bitwiseAnd, bitwiseOr, bitwiseXor.
              if (op != BinaryOp.leftShift &&
                  op != BinaryOp.rightShift &&
                  op != BinaryOp.gcd &&
                  op != BinaryOp.lcm) {
                final List<int> extremeRaw = switch (dtype) {
                  DType.int8 => [-100, -50, 0, 50, 100, -128, 127, -1],
                  DType.int16 => [
                    -20000,
                    -100,
                    0,
                    100,
                    20000,
                    -32768,
                    32767,
                    -1,
                  ],
                  DType.int32 => [
                    -1000000,
                    -100,
                    0,
                    100,
                    1000000,
                    -2147483648,
                    2147483647,
                    -1,
                  ],
                  DType.int64 => [
                    -10000000000,
                    -100,
                    0,
                    100,
                    10000000000,
                    -9223372036854775808,
                    9223372036854775807,
                    -1,
                  ],
                  DType.uint8 => [200, 150, 0, 50, 250, 255, 128, 1],
                  DType.uint16 => [
                    50000,
                    40000,
                    0,
                    100,
                    60000,
                    65535,
                    32768,
                    1,
                  ],
                  DType.uint32 => [
                    3000000000,
                    2500000000,
                    0,
                    100,
                    4000000000,
                    4294967295,
                    2147483648,
                    1,
                  ],
                  DType.uint64 => [
                    -100,
                    -50,
                    0,
                    50,
                    -10,
                    -1,
                    -9223372036854775808,
                    1,
                  ],
                  _ => [1, 2, 3, 4, 5, 6, 7, 8],
                };
                final extBase = NDArray.fromList(extremeRaw.cast<dynamic>(), [
                  8,
                ], dtype);
                final extB = NDArray.fromList(
                  [
                    extremeRaw[0],
                    extremeRaw[1],
                    extremeRaw[6],
                    extremeRaw[7],
                  ].cast<dynamic>(),
                  [4],
                  dtype,
                );
                final extExpectedAt = extBase.copy();
                for (var k = 0; k < indices.size; k++) {
                  final idx = indices[[k]];
                  final cur = extExpectedAt.slice([
                    Slice(start: idx, stop: idx + 1),
                  ]);
                  final rhs = extB.slice([Slice(start: k, stop: k + 1)]);
                  binaryUfunc(cur, rhs, op: op).copy(out: cur);
                }
                final extActualAt = extBase.copy();
                atUfunc(extActualAt, indices, extB, op: op);
                expectArraysEquivalent(
                  extActualAt,
                  extExpectedAt,
                  reason: 'atUfunc($dtype, $op) MSB/negative values',
                );
              }
            });
          }
        }
      },
    );

    test(
      'Float & Complex IEEE-754 boundaries: -0.0, +0.0, NaN, Inf, -Inf & length-25 leading NaN',
      () {
        for (final dtype in floatDTypes) {
          NDArray.scope(() {
            // Length-25 array with NaN at index 0 to catch SIMD fast-paths
            // that initialize best_val = a[0] and never update because x > NaN is false!
            final leadingNanList = List<double>.generate(
              25,
              (i) => i == 0 ? double.nan : ((i % 7) - 3).toDouble(),
            );
            final leadingNan = NDArray.fromList(
              leadingNanList.cast<dynamic>(),
              [25],
              dtype,
            );
            final leadingNanStep2 = makeStep2View(leadingNan);
            final leadingNanNeg = makeNegativeStrideView(leadingNan);

            // min / max must propagate NaN
            expect(
              (min(leadingNan).scalar as double).isNaN,
              isTrue,
              reason: 'min($dtype) with leading NaN must return NaN',
            );
            expect(
              (max(leadingNan).scalar as double).isNaN,
              isTrue,
              reason: 'max($dtype) with leading NaN must return NaN',
            );

            // nanmin / nanmax must ignore leading NaN at index 0 and match strided
            expect(
              nanmin(leadingNan).scalar,
              equals(-3.0),
              reason: 'nanmin($dtype) with leading NaN at index 0 must be -3.0',
            );
            expect(
              nanmax(leadingNan).scalar,
              equals(3.0),
              reason: 'nanmax($dtype) with leading NaN at index 0 must be 3.0',
            );
            expectArraysEquivalent(
              nanmin(leadingNanStep2),
              nanmin(leadingNan),
              reason: 'nanmin($dtype) leading NaN step:2 vs contiguous',
            );
            expectArraysEquivalent(
              nanmax(leadingNanStep2),
              nanmax(leadingNan),
              reason: 'nanmax($dtype) leading NaN step:2 vs contiguous',
            );
            expectArraysEquivalent(
              nanmin(leadingNanNeg),
              nanmin(leadingNan),
              reason: 'nanmin($dtype) leading NaN step:-1 vs contiguous',
            );
            expectArraysEquivalent(
              nanmax(leadingNanNeg),
              nanmax(leadingNan),
              reason: 'nanmax($dtype) leading NaN step:-1 vs contiguous',
            );

            // argmax / argmin with leading NaN at index 0
            expectArraysEquivalent(
              argmax(leadingNanStep2),
              argmax(leadingNan),
              reason: 'argmax($dtype) leading NaN step:2 vs contiguous',
            );
            expectArraysEquivalent(
              argmin(leadingNanStep2),
              argmin(leadingNan),
              reason: 'argmin($dtype) leading NaN step:2 vs contiguous',
            );

            // reduceat with leading NaN at index 0 across length-25 array
            final reduceIdx = NDArray<Int64>.fromList(
              [0, 5, 15],
              [3],
              DType.int64,
            );
            for (final op in [
              BinaryOp.minimum,
              BinaryOp.maximum,
              BinaryOp.fmin,
              BinaryOp.fmax,
              BinaryOp.add,
            ]) {
              final rContig = reduceatUfunc(leadingNan, reduceIdx, op: op);
              final rStep2 = reduceatUfunc(leadingNanStep2, reduceIdx, op: op);
              expectArraysEquivalent(
                rStep2,
                rContig,
                reason:
                    'reduceatUfunc($dtype, $op) leading NaN step:2 vs contig',
              );
              if (op == BinaryOp.fmin) {
                expect(
                  (rContig[[0]] as double).isNaN,
                  isFalse,
                  reason: 'reduceatUfunc($dtype, fmin) must ignore leading NaN',
                );
              }
              if (op == BinaryOp.fmax) {
                expect(
                  (rContig[[0]] as double).isNaN,
                  isFalse,
                  reason: 'reduceatUfunc($dtype, fmax) must ignore leading NaN',
                );
              }
            }

            // Pairwise combinations of [-0.0, +0.0, NaN, Inf, -Inf, 1.0, -1.0]
            // padded to length 25 so SIMD lanes are exercised
            final specialVals = <double>[
              double.nan,
              -0.0,
              0.0,
              double.infinity,
              double.negativeInfinity,
              1.0,
              -1.0,
            ];
            final aList = List<double>.generate(
              25,
              (i) => specialVals[i % specialVals.length],
            );
            final bList = List<double>.generate(
              25,
              (i) => specialVals[(i * 3 + 1) % specialVals.length],
            );
            final aSpec = NDArray.fromList(aList.cast<dynamic>(), [25], dtype);
            final bSpec = NDArray.fromList(bList.cast<dynamic>(), [25], dtype);
            final aSpecStep2 = makeStep2View(aSpec);
            final bSpecStep2 = makeStep2View(bSpec);
            final aSpecNeg = makeNegativeStrideView(aSpec);
            final bSpecNeg = makeNegativeStrideView(bSpec);

            for (final op in [
              BinaryOp.minimum,
              BinaryOp.maximum,
              BinaryOp.fmin,
              BinaryOp.fmax,
            ]) {
              final c = binaryUfunc(aSpec, bSpec, op: op);
              final s2 = binaryUfunc(aSpecStep2, bSpecStep2, op: op);
              final neg = binaryUfunc(aSpecNeg, bSpecNeg, op: op);
              expectArraysEquivalent(
                s2,
                c,
                reason: '$op($dtype) IEEE-754 special values step:2 vs contig',
                checkSignedZero: true,
              );
              expectArraysEquivalent(
                neg,
                c,
                reason: '$op($dtype) IEEE-754 special values step:-1 vs contig',
                checkSignedZero: true,
              );
            }

            // Explicit signed zero check: minimum(-0.0, +0.0) == -0.0, maximum(-0.0, +0.0) == +0.0
            final negZeroArr = NDArray.fromList(
              List<double>.filled(25, -0.0).cast<dynamic>(),
              [25],
              dtype,
            );
            final posZeroArr = NDArray.fromList(
              List<double>.filled(25, 0.0).cast<dynamic>(),
              [25],
              dtype,
            );
            for (final op in [BinaryOp.minimum, BinaryOp.fmin]) {
              final res1 = binaryUfunc(negZeroArr, posZeroArr, op: op);
              final res2 = binaryUfunc(posZeroArr, negZeroArr, op: op);
              for (var i = 0; i < 25; i++) {
                expect(
                  (res1[[i]] as double).isNegative,
                  isTrue,
                  reason: '$op($dtype)(-0.0, +0.0) at $i must be -0.0',
                );
                expect(
                  (res2[[i]] as double).isNegative,
                  isTrue,
                  reason: '$op($dtype)(+0.0, -0.0) at $i must be -0.0',
                );
              }
            }
            for (final op in [BinaryOp.maximum, BinaryOp.fmax]) {
              final res1 = binaryUfunc(negZeroArr, posZeroArr, op: op);
              final res2 = binaryUfunc(posZeroArr, negZeroArr, op: op);
              for (var i = 0; i < 25; i++) {
                expect(
                  (res1[[i]] as double).isNegative,
                  isFalse,
                  reason: '$op($dtype)(-0.0, +0.0) at $i must be +0.0',
                );
                expect(
                  (res2[[i]] as double).isNegative,
                  isFalse,
                  reason: '$op($dtype)(+0.0, -0.0) at $i must be +0.0',
                );
              }
            }

            // sort, argsort, unique with NaN, -Inf, +Inf, -0.0, +0.0
            final sortContig = sort(aSpec);
            final sortStep2 = sort(aSpecStep2);
            final sortNeg = sort(aSpecNeg);
            expectArraysEquivalent(
              sortStep2,
              sortContig,
              reason: 'sort($dtype) IEEE-754 special values step:2 vs contig',
            );
            expectArraysEquivalent(
              sortNeg,
              sortContig,
              reason: 'sort($dtype) IEEE-754 special values step:-1 vs contig',
            );
            // NaNs must sort to the end
            expect(
              (sortContig[[24]] as double).isNaN,
              isTrue,
              reason: 'sort($dtype) must place NaNs at the end',
            );
            expect(
              sortContig[[0]],
              equals(double.negativeInfinity),
              reason: 'sort($dtype) must place -Inf first',
            );

            final uSpecContig = unique(aSpec);
            final uSpecStep2 = unique(aSpecStep2);
            expectArraysEquivalent(
              uSpecStep2,
              uSpecContig,
              reason: 'unique($dtype) IEEE-754 special values step:2 vs contig',
            );
          });
        }

        // Complex IEEE-754 boundary values
        for (final dtype in <DType<AnySpec>>[
          DType.complex128,
          DType.complex64,
        ]) {
          NDArray.scope(() {
            final cSpecial = <Complex>[
              Complex(double.nan, 0.0),
              Complex(0.0, double.nan),
              Complex(-0.0, 0.0),
              Complex(0.0, -0.0),
              Complex(double.infinity, -1.0),
              Complex(double.negativeInfinity, 1.0),
              Complex(1.0, -2.0),
            ];
            final cAList = List<Complex>.generate(
              25,
              (i) => cSpecial[i % cSpecial.length],
            );
            final cBList = List<Complex>.generate(
              25,
              (i) => cSpecial[(i * 2 + 1) % cSpecial.length],
            );
            final cA = NDArray.fromList(cAList.cast<dynamic>(), [25], dtype);
            final cB = NDArray.fromList(cBList.cast<dynamic>(), [25], dtype);
            final cAStep2 = makeStep2View(cA);
            final cBStep2 = makeStep2View(cB);

            for (final op in [
              BinaryOp.minimum,
              BinaryOp.maximum,
              BinaryOp.fmin,
              BinaryOp.fmax,
            ]) {
              expectArraysEquivalent(
                binaryUfunc(cAStep2, cBStep2, op: op),
                binaryUfunc(cA, cB, op: op),
                reason: 'complex $op($dtype) special values step:2 vs contig',
              );
            }
          });
        }
      },
    );
  });

  group(
    '3. In-Place (out:) Aliasing, Partial Slice Overlap (+1/-1), & Non-Contiguous out: Views',
    () {
      test('Unary ufuncs: self-aliasing, +1/-1 slice overlap, strided out:', () {
        for (final dtype in numericDTypes) {
          NDArray.scope(() {
            const n = 26;
            final base = makeSample1D(dtype, length: n, positiveNonZero: true);

            // 1. Exact self-aliasing: out: a
            final selfA = base.copy();
            final expectedSelf = negative(base);
            negative(selfA, out: selfA);
            expectArraysEquivalent(
              selfA,
              expectedSelf,
              reason: 'negative($dtype) exact self-aliasing out: a',
            );

            // 2. Partial slice overlap (+1 shift: read [0..n-1), write [1..n))
            final bufForward = base.copy();
            final srcForward = bufForward.slice([
              const Slice(start: 0, stop: n - 1),
            ]);
            final dstForward = bufForward.slice([
              const Slice(start: 1, stop: n),
            ]);
            final expectedForward = square(srcForward.copy());
            square(srcForward, out: dstForward);
            expectArraysEquivalent(
              dstForward,
              expectedForward,
              reason: 'square($dtype) partial overlap +1 shift',
            );

            // 3. Partial slice overlap (-1 shift: read [1..n), write [0..n-1))
            final bufBackward = base.copy();
            final srcBackward = bufBackward.slice([
              const Slice(start: 1, stop: n),
            ]);
            final dstBackward = bufBackward.slice([
              const Slice(start: 0, stop: n - 1),
            ]);
            final expectedBackward = square(srcBackward.copy());
            square(srcBackward, out: dstBackward);
            expectArraysEquivalent(
              dstBackward,
              expectedBackward,
              reason: 'square($dtype) partial overlap -1 shift',
            );

            // 4. Non-contiguous step: 2 and negative-stride out: views on real dtypes
            if (dtype != DType.complex128 && dtype != DType.complex64) {
              final outStep2Backing = NDArray.zeros([n * 2], dtype);
              final outStep2 = outStep2Backing.slice([
                const Slice(start: 0, stop: n * 2, step: 2),
              ]);
              square(base, out: outStep2);
              expectArraysEquivalent(
                outStep2,
                square(base),
                reason: 'square($dtype) into step:2 out: view',
              );
              // Verify untouched odd slots remain zero
              final oddSlots = outStep2Backing.slice([
                const Slice(start: 1, stop: n * 2, step: 2),
              ]);
              expectArraysEquivalent(
                oddSlots,
                NDArray.zeros([n], dtype),
                reason: 'square($dtype) step:2 out: must not clobber odd slots',
              );

              final outNegBacking = NDArray.zeros([n], dtype);
              final outNeg = flip(outNegBacking, axis: 0);
              square(base, out: outNeg);
              expectArraysEquivalent(
                outNeg,
                square(base),
                reason: 'square($dtype) into negative-stride out: view',
              );
            }
          });
        }
      });

      test('Binary ufuncs: self-aliasing, +1/-1 slice overlap, strided out:', () {
        for (final dtype in numericDTypes) {
          NDArray.scope(() {
            const n = 26;
            final aBase = makeSample1D(
              dtype,
              length: n,
              seedOffset: 1,
              positiveNonZero: true,
            );
            final bBase = makeSample1D(
              dtype,
              length: n,
              seedOffset: 4,
              positiveNonZero: true,
            );

            // 1. Self-aliasing: out: a, out: b, and out: a with b == a
            final a1 = aBase.copy();
            add(a1, bBase, out: a1);
            expectArraysEquivalent(
              a1,
              add(aBase, bBase),
              reason: 'add($dtype) self-aliasing out: a',
            );

            final b1 = bBase.copy();
            subtract(aBase, b1, out: b1);
            expectArraysEquivalent(
              b1,
              subtract(aBase, bBase),
              reason: 'subtract($dtype) self-aliasing out: b',
            );

            final aa = aBase.copy();
            multiply(aa, aa, out: aa);
            expectArraysEquivalent(
              aa,
              multiply(aBase, aBase),
              reason: 'multiply($dtype) self-aliasing a * a into a',
            );

            // 2. Partial slice overlap (+1 offset): out = buf[1..n], a = buf[0..n-1]
            final bufPlus = aBase.copy();
            final slice0 = bufPlus.slice([const Slice(start: 0, stop: n - 1)]);
            final slice1 = bufPlus.slice([const Slice(start: 1, stop: n)]);
            final expectedPlus = add(slice0.copy(), slice1.copy());
            add(slice0, slice1, out: slice1);
            expectArraysEquivalent(
              slice1,
              expectedPlus,
              reason: 'add($dtype) partial overlap +1 offset (out: slice1)',
            );

            // 3. Partial slice overlap (-1 offset): out = buf[0..n-1], a = buf[1..n]
            final bufMinus = aBase.copy();
            final mSlice0 = bufMinus.slice([
              const Slice(start: 0, stop: n - 1),
            ]);
            final mSlice1 = bufMinus.slice([const Slice(start: 1, stop: n)]);
            final expectedMinus = subtract(mSlice1.copy(), mSlice0.copy());
            subtract(mSlice1, mSlice0, out: mSlice0);
            expectArraysEquivalent(
              mSlice0,
              expectedMinus,
              reason:
                  'subtract($dtype) partial overlap -1 offset (out: slice0)',
            );

            // 4. Non-contiguous step: 2 and negative-stride out: views
            final outStep2Backing = NDArray.zeros([n * 2], dtype);
            final outStep2 = outStep2Backing.slice([
              const Slice(start: 0, stop: n * 2, step: 2),
            ]);
            multiply(
              makeStep2View(aBase),
              makeNegativeStrideView(bBase),
              out: outStep2,
            );
            expectArraysEquivalent(
              outStep2,
              multiply(aBase, bBase),
              reason: 'multiply($dtype) strided inputs into step:2 out:',
            );

            final outNegBacking = NDArray.zeros([n], dtype);
            final outNeg = flip(outNegBacking, axis: 0);
            add(aBase, bBase, out: outNeg);
            expectArraysEquivalent(
              outNeg,
              add(aBase, bBase),
              reason: 'add($dtype) into negative-stride out:',
            );
          });
        }
      });

      test(
        'cumsum, cumprod, sort, & reductions with aliased and strided out:',
        () {
          for (final dtype in realNumericDTypes) {
            NDArray.scope(() {
              const n = 25;
              final aBase = makeSample1D(
                dtype,
                length: n,
                positiveNonZero: true,
                smallShiftAmounts: true,
              );

              // cumsum & cumprod in-place & partial overlap
              final csSelf = aBase.copy();
              cumsumAs(csSelf, dtype, axis: 0, out: csSelf);
              expectArraysEquivalent(
                csSelf,
                cumsumAs(aBase, dtype, axis: 0),
                reason: 'cumsumAs($dtype) self-aliasing out: a',
              );

              final cpSelf = aBase.copy();
              cumprodAs(cpSelf, dtype, axis: 0, out: cpSelf);
              expectArraysEquivalent(
                cpSelf,
                cumprodAs(aBase, dtype, axis: 0),
                reason: 'cumprodAs($dtype) self-aliasing out: a',
              );

              final csOverlap = aBase.copy();
              final csSrc = csOverlap.slice([
                const Slice(start: 0, stop: n - 1),
              ]);
              final csDst = csOverlap.slice([const Slice(start: 1, stop: n)]);
              final csExpected = cumsumAs(csSrc.copy(), dtype, axis: 0);
              cumsumAs(csSrc, dtype, axis: 0, out: csDst);
              expectArraysEquivalent(
                csDst,
                csExpected,
                reason: 'cumsumAs($dtype) partial overlap +1',
              );

              // sort in-place, partial overlap, and strided out:
              final sortSelf = aBase.copy();
              sort(sortSelf, out: sortSelf);
              expectArraysEquivalent(
                sortSelf,
                sort(aBase),
                reason: 'sort($dtype) self-aliasing out: a',
              );

              final sortOverlap = aBase.copy();
              final sSrc = sortOverlap.slice([const Slice(start: 1, stop: n)]);
              final sDst = sortOverlap.slice([
                const Slice(start: 0, stop: n - 1),
              ]);
              final sExpected = sort(sSrc.copy());
              sort(sSrc, out: sDst);
              expectArraysEquivalent(
                sDst,
                sExpected,
                reason: 'sort($dtype) partial overlap -1',
              );

              final sortOutStep2 = NDArray.zeros([
                n * 2,
              ], dtype).slice([const Slice(start: 0, stop: n * 2, step: 2)]);
              sort(aBase, out: sortOutStep2);
              expectArraysEquivalent(
                sortOutStep2,
                sort(aBase),
                reason: 'sort($dtype) step:2 out:',
              );

              // Axis reduction into a slice of the input array (aliasing) and strided out:
              final mat = aBase.reshape([5, 5]).copy();
              final expectedRowMax = max(mat.copy(), axis: 1);
              final row0Out = mat
                  .slice([const Slice(start: 0, stop: 1), const Slice.all()])
                  .reshape([5]);
              max(mat, axis: 1, out: row0Out);
              expectArraysEquivalent(
                row0Out,
                expectedRowMax,
                reason:
                    'max($dtype, axis:1) aliased into row 0 of input matrix',
              );
            });
          }
        },
      );
    },
  );

  group(
    '4. Independent Dart Oracle: Unsigned 64-bit (MSB-set) Ordering Kernels',
    () {
      // Bit patterns >= 2^63 are negative Dart ints. The oracle orders them
      // with uint64Compare, so it cannot share a signed-comparison bug with
      // the native scalar fallback that the other groups compare against.
      const raw = <int>[
        5,
        -1, // 2^64 - 1
        0,
        -9223372036854775808, // 2^63
        42,
        -2, // 2^64 - 2
        9223372036854775807, // 2^63 - 1
        7,
        -100,
        1,
        3,
        -9223372036854775807, // 2^63 + 1
        11,
        2,
        9,
        -50,
        6,
        8,
        4,
        10,
        0,
        12,
        -3,
        13,
        14,
      ];
      const n = 25;

      int ucmp(int a, int b) => uint64Compare(a, b);
      int umax(Iterable<int> xs) =>
          xs.reduce((a, b) => ucmp(a, b) >= 0 ? a : b);
      int umin(Iterable<int> xs) =>
          xs.reduce((a, b) => ucmp(a, b) <= 0 ? a : b);
      int uargmax(List<int> xs) {
        var best = 0;
        for (var i = 1; i < xs.length; i++) {
          if (ucmp(xs[i], xs[best]) > 0) best = i;
        }
        return best;
      }

      int uargmin(List<int> xs) {
        var best = 0;
        for (var i = 1; i < xs.length; i++) {
          if (ucmp(xs[i], xs[best]) < 0) best = i;
        }
        return best;
      }

      final expectedSorted = List<int>.of(raw)..sort(ucmp);
      final expectedUnique = expectedSorted.toSet().toList()..sort(ucmp);

      NDArray<Uint64> contiguous() =>
          NDArray<Uint64>.fromList(raw, [n], DType.uint64);

      Map<String, NDArray<Uint64>> layouts(NDArray<Uint64> c) => {
        'contiguous': c,
        'step:2': makeStep2View(c),
        'step:-1': makeNegativeStrideView(c),
        'rank-3 [1,3,N] row 0': makeRank3StridedView(
          c,
        ).slice([const Index(0), const Index(0), const Slice.all()]),
      };

      test(
        'sort / argsort / partition / argpartition / unique / searchsorted',
        () {
          NDArray.scope(() {
            final c = contiguous();
            final ref = NDArray<Uint64>.fromList(expectedUnique, [
              expectedUnique.length,
            ], DType.uint64);
            for (final entry in layouts(c).entries) {
              final label = entry.key;
              final x = entry.value;

              expect(
                sort(x).toList(),
                equals(expectedSorted),
                reason: 'sort(uint64) $label',
              );

              final order = argsort(x).toList().cast<int>();
              expect(
                order.toSet().length,
                equals(n),
                reason: 'argsort(uint64) $label must be a permutation',
              );
              expect(
                [for (final i in order) raw[i]],
                equals(expectedSorted),
                reason: 'argsort(uint64) $label gathers into sorted order',
              );

              for (final kth in [0, 12, 24]) {
                final p = partition(x, kth).toList().cast<int>();
                expect(
                  p[kth],
                  equals(expectedSorted[kth]),
                  reason: 'partition(uint64, kth: $kth) $label pivot',
                );
                for (var i = 0; i < kth; i++) {
                  expect(
                    ucmp(p[i], p[kth]) <= 0,
                    isTrue,
                    reason:
                        'partition(uint64, kth: $kth) $label: p[$i] > pivot',
                  );
                }
                for (var i = kth + 1; i < n; i++) {
                  expect(
                    ucmp(p[i], p[kth]) >= 0,
                    isTrue,
                    reason:
                        'partition(uint64, kth: $kth) $label: p[$i] < pivot',
                  );
                }
                final ap = argpartition(x, kth).toList().cast<int>();
                expect(
                  raw[ap[kth]],
                  equals(expectedSorted[kth]),
                  reason: 'argpartition(uint64, kth: $kth) $label pivot',
                );
              }

              expect(
                unique(x).toList(),
                equals(expectedUnique),
                reason: 'unique(uint64) $label',
              );

              final left = searchsorted(ref, x).toList().cast<int>();
              final right = searchsorted(
                ref,
                x,
                side: SearchSide.right,
              ).toList().cast<int>();
              for (var i = 0; i < n; i++) {
                final v = raw[i];
                var expLeft = expectedUnique.indexWhere((u) => ucmp(u, v) >= 0);
                var expRight = expectedUnique.indexWhere((u) => ucmp(u, v) > 0);
                if (expLeft < 0) expLeft = expectedUnique.length;
                if (expRight < 0) expRight = expectedUnique.length;
                expect(
                  left[i],
                  equals(expLeft),
                  reason: 'searchsorted(uint64, left) $label at $i',
                );
                expect(
                  right[i],
                  equals(expRight),
                  reason: 'searchsorted(uint64, right) $label at $i',
                );
              }
            }
          });
        },
      );

      test(
        'max / min / argmax / argmin / ptp / cummax / cummin / maximum / minimum / greater / less',
        () {
          NDArray.scope(() {
            final c = contiguous();
            final rawRev = raw.reversed.toList();
            final expectedCummax = <int>[];
            final expectedCummin = <int>[];
            var runMax = raw[0];
            var runMin = raw[0];
            for (final v in raw) {
              if (ucmp(v, runMax) > 0) runMax = v;
              if (ucmp(v, runMin) < 0) runMin = v;
              expectedCummax.add(runMax);
              expectedCummin.add(runMin);
            }

            for (final entry in layouts(c).entries) {
              final label = entry.key;
              final x = entry.value;

              expect(
                max(x).scalar,
                equals(umax(raw)),
                reason: 'max(uint64) $label',
              );
              expect(
                min(x).scalar,
                equals(umin(raw)),
                reason: 'min(uint64) $label',
              );
              expect(
                argmax(x).scalar,
                equals(uargmax(raw)),
                reason: 'argmax(uint64) $label',
              );
              expect(
                argmin(x).scalar,
                equals(uargmin(raw)),
                reason: 'argmin(uint64) $label',
              );
              // Unsigned max - min wraps modulo 2^64 exactly like Dart ints.
              expect(
                ptp(x).scalar,
                equals(umax(raw) - umin(raw)),
                reason: 'ptp(uint64) $label',
              );
              expect(
                cummax(x).toList(),
                equals(expectedCummax),
                reason: 'cummax(uint64) $label',
              );
              expect(
                cummin(x).toList(),
                equals(expectedCummin),
                reason: 'cummin(uint64) $label',
              );

              final y = flip(x, axis: 0).copy(); // logical rawRev, contiguous
              expect(
                binaryUfunc(x, y, op: BinaryOp.maximum).toList(),
                equals([
                  for (var i = 0; i < n; i++)
                    ucmp(raw[i], rawRev[i]) >= 0 ? raw[i] : rawRev[i],
                ]),
                reason: 'maximum(uint64) $label',
              );
              expect(
                binaryUfunc(x, y, op: BinaryOp.minimum).toList(),
                equals([
                  for (var i = 0; i < n; i++)
                    ucmp(raw[i], rawRev[i]) <= 0 ? raw[i] : rawRev[i],
                ]),
                reason: 'minimum(uint64) $label',
              );
              expect(
                greater(x, y).toList(),
                equals([
                  for (var i = 0; i < n; i++) ucmp(raw[i], rawRev[i]) > 0,
                ]),
                reason: 'greater(uint64) $label',
              );
              expect(
                less(x, y).toList(),
                equals([
                  for (var i = 0; i < n; i++) ucmp(raw[i], rawRev[i]) < 0,
                ]),
                reason: 'less(uint64) $label',
              );
            }
          });
        },
      );

      test('axis reductions and axis sort on a transposed [5, 5] view', () {
        NDArray.scope(() {
          // tView[i][j] = raw[j * 5 + i].
          final tView = contiguous().reshape([5, 5]).transpose();
          List<int> column(int j) => [
            for (var i = 0; i < 5; i++) raw[j * 5 + i],
          ];
          List<int> row(int i) => [for (var j = 0; j < 5; j++) raw[j * 5 + i]];

          // axis 0 reduces over i (result index j); axis 1 over j (index i).
          expect(
            max(tView, axis: 0).toList(),
            equals([for (var j = 0; j < 5; j++) umax(column(j))]),
            reason: 'max(uint64, axis: 0) transposed',
          );
          expect(
            max(tView, axis: 1).toList(),
            equals([for (var i = 0; i < 5; i++) umax(row(i))]),
            reason: 'max(uint64, axis: 1) transposed',
          );
          expect(
            min(tView, axis: 0).toList(),
            equals([for (var j = 0; j < 5; j++) umin(column(j))]),
            reason: 'min(uint64, axis: 0) transposed',
          );
          expect(
            min(tView, axis: 1).toList(),
            equals([for (var i = 0; i < 5; i++) umin(row(i))]),
            reason: 'min(uint64, axis: 1) transposed',
          );
          expect(
            argmax(tView, axis: 0).toList(),
            equals([for (var j = 0; j < 5; j++) uargmax(column(j))]),
            reason: 'argmax(uint64, axis: 0) transposed',
          );
          expect(
            argmin(tView, axis: 1).toList(),
            equals([for (var i = 0; i < 5; i++) uargmin(row(i))]),
            reason: 'argmin(uint64, axis: 1) transposed',
          );

          // sort along axis 0: column j of the result is column(j) sorted.
          final sorted0 = sort(tView, axis: 0);
          for (var j = 0; j < 5; j++) {
            final col = column(j)..sort(ucmp);
            expect(
              [
                for (var i = 0; i < 5; i++) sorted0[[i, j]],
              ],
              equals(col),
              reason: 'sort(uint64, axis: 0) transposed column $j',
            );
          }
          // sort along axis 1: row i of the result is row(i) sorted.
          final sorted1 = sort(tView, axis: 1);
          for (var i = 0; i < 5; i++) {
            final r = row(i)..sort(ucmp);
            expect(
              [
                for (var j = 0; j < 5; j++) sorted1[[i, j]],
              ],
              equals(r),
              reason: 'sort(uint64, axis: 1) transposed row $i',
            );
          }
        });
      });
    },
  );
}
