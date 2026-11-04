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

NDArray<AnySpec> _sample1D(DType dtype) {
  switch (dtype) {
    case DType.boolean:
      return NDArray<Boolean>.fromList(
        [true, false, true, true],
        [4],
        DType.boolean,
      ).asAnySpec;
    case DType.complex128:
      return NDArray<Complex128>.fromList(
        [
          Complex(1.0, 2.0),
          Complex(3.0, -1.0),
          Complex(2.0, 4.0),
          Complex(4.0, 0.0),
        ],
        [4],
        DType.complex128,
      ).asAnySpec;
    case DType.complex64:
      return NDArray<Complex64>.fromList(
        [
          Complex(1.0, 2.0),
          Complex(3.0, -1.0),
          Complex(2.0, 4.0),
          Complex(4.0, 0.0),
        ],
        [4],
        DType.complex64,
      ).asAnySpec;
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
      return NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], dtype).asAnySpec;
    default:
      return NDArray.fromList([1, 2, 3, 4], [4], dtype).asAnySpec;
  }
}

NDArray<AnySpec> _sample2D(DType dtype) {
  switch (dtype) {
    case DType.boolean:
      return NDArray<Boolean>.fromList(
        [true, false, true, true],
        [2, 2],
        DType.boolean,
      ).asAnySpec;
    case DType.complex128:
      return NDArray<Complex128>.fromList(
        [
          Complex(1.0, 2.0),
          Complex(3.0, 1.0),
          Complex(2.0, -1.0),
          Complex(4.0, 2.0),
        ],
        [2, 2],
        DType.complex128,
      ).asAnySpec;
    case DType.complex64:
      return NDArray<Complex64>.fromList(
        [
          Complex(1.0, 2.0),
          Complex(3.0, 1.0),
          Complex(2.0, -1.0),
          Complex(4.0, 2.0),
        ],
        [2, 2],
        DType.complex64,
      ).asAnySpec;
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
      return NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], dtype).asAnySpec;
    default:
      return NDArray.fromList([1, 2, 3, 4], [2, 2], dtype).asAnySpec;
  }
}

void main() {
  group('stats branch coverage - sumAs and prodAs across all dtypes', () {
    test(
      'sumAs and prodAs axis: null (contiguous and strided) and axis: 0',
      () {
        for (final dt in DType.values) {
          NDArray.scope(() {
            final a1 = _sample1D(dt);
            final s1 = sumAs(a1, dt);
            expect(s1.dtype, equals(dt));
            final p1 = prodAs(a1, dt);
            expect(p1.dtype, equals(dt));

            final a1Strided = a1.slice([const Slice(step: 2)]);
            final s1Strided = sumAs(a1Strided, dt);
            expect(s1Strided.dtype, equals(dt));
            final p1Strided = prodAs(a1Strided, dt);
            expect(p1Strided.dtype, equals(dt));

            final a2 = _sample2D(dt);
            final s2 = sumAs(a2, dt, axis: 0);
            expect(s2.shape, equals([2]));
            final p2 = prodAs(a2, dt, axis: 0);
            expect(p2.shape, equals([2]));

            final outS = NDArray.zeros([2], dt);
            sumAs(a2, dt, axis: 0, out: outS);
            final outP = NDArray.zeros([2], dt);
            prodAs(a2, dt, axis: 0, out: outP);
          });
        }
      },
    );

    test('sumAs and prodAs on empty arrays, cast path, and out aliasing', () {
      NDArray.scope(() {
        final empty1D = NDArray<Float64>.zeros([0], DType.float64);
        final outScalar = NDArray<Float64>.zeros([], DType.float64);
        sumAs(empty1D, DType.float64, out: outScalar);
        expect(outScalar.scalar, equals(0.0));
        prodAs(empty1D, DType.float64, out: outScalar);
        expect(outScalar.scalar, equals(1.0));

        final emptyAxis = NDArray<Float64>.zeros([0, 2], DType.float64);
        final outVec = NDArray<Float64>.zeros([2], DType.float64);
        sumAs(emptyAxis, DType.float64, axis: 0, out: outVec);
        expect(outVec.toList(), equals([0.0, 0.0]));
        prodAs(emptyAxis, DType.float64, axis: 0, out: outVec);
        expect(outVec.toList(), equals([1.0, 1.0]));

        final emptyResult = NDArray<Float64>.zeros([2, 0], DType.float64);
        expect(sumAs(emptyResult, DType.float64, axis: 0).shape, equals([0]));
        expect(prodAs(emptyResult, DType.float64, axis: 0).shape, equals([0]));

        final i32 = NDArray<Int32>.fromList([1, 2, 3], [3], DType.int32);
        expect(sumAs(i32, DType.float64).scalar, equals(6.0));
        expect(prodAs(i32, DType.float64).scalar, equals(6.0));

        // out aliasing (sharesMemory(a, out))
        final a2 = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final outAlias = a2.slice([const Index(0), const Slice.all()]);
        sumAs(a2, DType.float64, axis: 0, out: outAlias);
        expect(outAlias.toList(), equals([4.0, 6.0]));

        final a2p = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final outAliasP = a2p.slice([const Index(0), const Slice.all()]);
        prodAs(a2p, DType.float64, axis: 0, out: outAliasP);
        expect(outAliasP.toList(), equals([3.0, 8.0]));
      });
    });
  });

  group('stats branch coverage - all and any across all dtypes', () {
    test('all and any contiguous, strided, axis, and empty', () {
      for (final dt in DType.values) {
        NDArray.scope(() {
          final a1 = _sample1D(dt);
          expect(all(a1).dtype, equals(DType.boolean));
          expect(any(a1).dtype, equals(DType.boolean));

          final a1Strided = a1.slice([const Slice(step: 2)]);
          expect(all(a1Strided).dtype, equals(DType.boolean));
          expect(any(a1Strided).dtype, equals(DType.boolean));

          final a2 = _sample2D(dt);
          expect(all(a2, axis: 0).shape, equals([2]));
          expect(any(a2, axis: 0).shape, equals([2]));

          final z = NDArray.zeros([3], dt);
          expect(all(z).scalar, isFalse);
          expect(any(z).scalar, isFalse);
        });
      }

      NDArray.scope(() {
        final cImag = NDArray<Complex128>.fromList(
          [Complex(0.0, 1.0), Complex(0.0, 2.0)],
          [2],
          DType.complex128,
        );
        expect(all(cImag).scalar, isTrue);
        expect(any(cImag).scalar, isTrue);

        final cImag64 = NDArray<Complex64>.fromList(
          [Complex(0.0, 1.0), Complex(0.0, 0.0)],
          [2],
          DType.complex64,
        );
        expect(all(cImag64).scalar, isFalse);
        expect(any(cImag64).scalar, isTrue);

        final emptyAxis = NDArray<Float64>.zeros([0, 2], DType.float64);
        final outAll = NDArray<Boolean>.zeros([2], DType.boolean);
        all(emptyAxis, axis: 0, out: outAll);
        expect(outAll.toList(), equals([true, true]));

        final outAny = NDArray<Boolean>.ones([2], DType.boolean);
        any(emptyAxis, axis: 0, out: outAny);
        expect(outAny.toList(), equals([false, false]));

        final emptyRes = NDArray<Boolean>.zeros([2, 0], DType.boolean);
        expect(all(emptyRes, axis: 0).shape, equals([0]));
        expect(any(emptyRes, axis: 0).shape, equals([0]));

        // out aliasing on boolean all/any
        final b2 = NDArray<Boolean>.fromList(
          [true, false, true, true],
          [2, 2],
          DType.boolean,
        );
        final bAlias = b2.slice([const Index(0), const Slice.all()]);
        all(b2, axis: 0, out: bAlias);
        expect(bAlias.toList(), equals([true, false]));

        final b2Any = NDArray<Boolean>.fromList(
          [false, false, true, false],
          [2, 2],
          DType.boolean,
        );
        final bAliasAny = b2Any.slice([const Index(0), const Slice.all()]);
        any(b2Any, axis: 0, out: bAliasAny);
        expect(bAliasAny.toList(), equals([true, false]));
      });
    });
  });

  group('stats branch coverage - mean, std, variance, nanmean, nanstd, nanvar', () {
    test('mean, std, variance across dtypes (contiguous, strided, axis)', () {
      for (final dt in DType.values) {
        NDArray.scope(() {
          final a1 = _sample1D(dt);
          expect(mean(a1).rank, equals(0));
          expect(std(a1).rank, equals(0));
          expect(variance(a1).rank, equals(0));

          final a1Strided = a1.slice([const Slice(step: 2)]);
          expect(mean(a1Strided).rank, equals(0));
          expect(std(a1Strided).rank, equals(0));
          expect(variance(a1Strided).rank, equals(0));

          final a2 = _sample2D(dt);
          expect(mean(a2, axis: 0).shape, equals([2]));
          expect(std(a2, axis: 0).shape, equals([2]));
          expect(variance(a2, axis: 0).shape, equals([2]));
        });
      }
    });

    test('mean, std, variance empty arrays, high ddof, and out aliasing', () {
      NDArray.scope(() {
        final emptyF = NDArray<Float64>.zeros([0], DType.float64);
        final outF = NDArray<Float64>.zeros([], DType.float64);
        mean(emptyF, out: outF);
        expect(outF.scalar.isNaN, isTrue);
        std(emptyF, out: outF);
        expect(outF.scalar.isNaN, isTrue);
        variance(emptyF, out: outF);
        expect(outF.scalar.isNaN, isTrue);

        final emptyC = NDArray<Complex128>.zeros([0], DType.complex128);
        final outC = NDArray<Complex128>.zeros([], DType.complex128);
        mean(emptyC, out: outC);
        expect(outC.scalar.real.isNaN, isTrue);
        std(emptyC, out: outF);
        expect(outF.scalar.isNaN, isTrue);
        variance(emptyC, out: outF);
        expect(outF.scalar.isNaN, isTrue);

        final emptyAxisF = NDArray<Float64>.zeros([0, 2], DType.float64);
        final outVecF = NDArray<Float64>.zeros([2], DType.float64);
        mean(emptyAxisF, axis: 0, out: outVecF);
        expect(outVecF.getCell([0]).isNaN, isTrue);
        std(emptyAxisF, axis: 0, out: outVecF);
        expect(outVecF.getCell([0]).isNaN, isTrue);
        variance(emptyAxisF, axis: 0, out: outVecF);
        expect(outVecF.getCell([0]).isNaN, isTrue);

        final emptyAxisC = NDArray<Complex128>.zeros([0, 2], DType.complex128);
        final outVecC = NDArray<Complex128>.zeros([2], DType.complex128);
        mean(emptyAxisC, axis: 0, out: outVecC);
        expect(outVecC.getCell([0]).real.isNaN, isTrue);
        std(emptyAxisC, axis: 0, out: outVecF);
        expect(outVecF.getCell([0]).isNaN, isTrue);
        variance(emptyAxisC, axis: 0, out: outVecF);
        expect(outVecF.getCell([0]).isNaN, isTrue);

        final emptyRes = NDArray<Float64>.zeros([2, 0], DType.float64);
        expect(mean(emptyRes, axis: 0).shape, equals([0]));
        expect(std(emptyRes, axis: 0).shape, equals([0]));
        expect(variance(emptyRes, axis: 0).shape, equals([0]));

        // ddof >= count
        final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
        expect(std(a, ddof: 2).scalar.isInfinite, isTrue);
        expect(variance(a, ddof: 2).scalar.isInfinite, isTrue);
        final a2 = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        expect(std(a2, axis: 0, ddof: 2).getCell([0]).isInfinite, isTrue);
        expect(variance(a2, axis: 0, ddof: 2).getCell([0]).isInfinite, isTrue);

        final c2 = NDArray<Complex128>.fromList(
          [Complex(1, 1), Complex(2, 2), Complex(3, 3), Complex(4, 4)],
          [2, 2],
          DType.complex128,
        );
        expect(std(c2, ddof: 5).scalar, equals(double.infinity));
        expect(variance(c2, ddof: 5).scalar, equals(double.infinity));
        expect(std(c2, axis: 0, ddof: 2).getCell([0]), equals(double.infinity));
        expect(
          variance(c2, axis: 0, ddof: 2).getCell([0]),
          equals(double.infinity),
        );

        // out aliasing on mean, std, variance
        final aAliasM = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        mean(
          aAliasM,
          axis: 0,
          out: aAliasM.slice([const Index(0), const Slice.all()]),
        );
        final aAliasS = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        std(
          aAliasS,
          axis: 0,
          out: aAliasS.slice([const Index(0), const Slice.all()]),
        );
        final aAliasV = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        variance(
          aAliasV,
          axis: 0,
          out: aAliasV.slice([const Index(0), const Slice.all()]),
        );
      });
    });

    test('nanmean, nanstd, nanvar across dtypes, complex, NaNs, and empty', () {
      for (final dt in [
        DType.float64,
        DType.float32,
        DType.float16,
        DType.int64,
        DType.int32,
        DType.boolean,
        DType.complex128,
        DType.complex64,
      ]) {
        NDArray.scope(() {
          final a1 = _sample1D(dt);
          expect(nanmean(a1).rank, equals(0));
          expect(nanstd(a1).rank, equals(0));
          expect(nanvar(a1).rank, equals(0));
          expect(nanvar(a1, ddof: 1).rank, equals(0));

          final a1Strided = a1.slice([const Slice(step: 2)]);
          expect(nanmean(a1Strided).rank, equals(0));
          expect(nanstd(a1Strided).rank, equals(0));
          expect(nanvar(a1Strided).rank, equals(0));
          expect(nanvar(a1Strided, ddof: 1).rank, equals(0));

          final a2 = _sample2D(dt);
          expect(nanmean(a2, axis: 0).shape, equals([2]));
          expect(nanstd(a2, axis: 0).shape, equals([2]));
          expect(nanvar(a2, axis: 0).shape, equals([2]));
          expect(nanvar(a2, axis: 0, ddof: 1).shape, equals([2]));
        });
      }

      NDArray.scope(() {
        for (final cdt in [DType.complex128, DType.complex64]) {
          final cNan = NDArray.fromList(
            [Complex(1.0, 2.0), Complex(double.nan, 1.0), Complex(3.0, 4.0)],
            [3],
            cdt,
          ).asAnySpec;
          final nm = nanmean(cNan);
          expect((nm.scalar as Complex).real, closeTo(2.0, 1e-5));
          final nv = nanvar(cNan);
          expect(nv.scalar, closeTo(2.0, 1e-5));
          final ns = nanstd(cNan);
          expect(ns.scalar, closeTo(1.41421356, 1e-5));

          final cAllNan = NDArray.fromList(
            [Complex(double.nan, 0.0), Complex(0.0, double.nan)],
            [2],
            cdt,
          ).asAnySpec;
          expect((nanmean(cAllNan).scalar as Complex).real.isNaN, isTrue);
          expect(nanvar(cAllNan).scalar.isNaN, isTrue);

          final c2Nan = NDArray.fromList(
            [
              Complex(1.0, 2.0),
              Complex(double.nan, 0.0),
              Complex(3.0, 4.0),
              Complex(0.0, double.nan),
            ],
            [2, 2],
            cdt,
          ).asAnySpec;
          final outM = NDArray<Complex128>.zeros([2], DType.complex128);
          nanmean(c2Nan, axis: 0, out: outM.asAnySpec);
          expect(outM.getCell([0]).real, closeTo(2.0, 1e-5));
          expect(outM.getCell([1]).real.isNaN, isTrue);

          final outV = NDArray<Float64>.zeros([2], DType.float64);
          nanvar(c2Nan, axis: 0, out: outV);
          expect(outV.getCell([0]), closeTo(2.0, 1e-5));
          expect(outV.getCell([1]).isNaN, isTrue);
        }

        // Real nanvar with ddof: 1 and NaNs (exercises _computeNanvarGeneral real branch)
        final fNan1D = NDArray<Float64>.fromList(
          [1.0, double.nan, 3.0],
          [3],
          DType.float64,
        );
        expect(nanvar(fNan1D, ddof: 1).scalar, closeTo(2.0, 1e-5));
        expect(nanvar(fNan1D, ddof: 2).scalar.isNaN, isTrue);

        final fNan2D = NDArray<Float64>.fromList(
          [1.0, double.nan, 3.0, double.nan],
          [2, 2],
          DType.float64,
        );
        final nv2D = nanvar(fNan2D, axis: 0, ddof: 1);
        expect(nv2D.getCell([0]), closeTo(2.0, 1e-5));
        expect(nv2D.getCell([1]).isNaN, isTrue);

        final emptyF = NDArray<Float64>.zeros([0], DType.float64);
        final outF = NDArray<Float64>.zeros([], DType.float64);
        nanmean(emptyF, out: outF);
        expect(outF.scalar.isNaN, isTrue);
        nanvar(emptyF, out: outF);
        expect(outF.scalar.isNaN, isTrue);
        nanstd(emptyF, out: outF);
        expect(outF.scalar.isNaN, isTrue);

        final emptyC = NDArray<Complex128>.zeros([0], DType.complex128);
        final outC = NDArray<Complex128>.zeros([], DType.complex128);
        nanmean(emptyC, out: outC);
        expect(outC.scalar.real.isNaN, isTrue);

        final emptyAxisF = NDArray<Float64>.zeros([0, 2], DType.float64);
        final outVecF = NDArray<Float64>.zeros([2], DType.float64);
        nanmean(emptyAxisF, axis: 0, out: outVecF);
        expect(outVecF.getCell([0]).isNaN, isTrue);
        nanvar(emptyAxisF, axis: 0, out: outVecF);
        expect(outVecF.getCell([0]).isNaN, isTrue);
        nanvar(emptyAxisF, axis: 0, ddof: 1, out: outVecF);
        expect(outVecF.getCell([0]).isNaN, isTrue);

        final emptyAxisC = NDArray<Complex128>.zeros([0, 2], DType.complex128);
        final outVecC = NDArray<Complex128>.zeros([2], DType.complex128);
        nanmean(emptyAxisC, axis: 0, out: outVecC);
        expect(outVecC.getCell([0]).real.isNaN, isTrue);

        final emptyRes = NDArray<Float64>.zeros([2, 0], DType.float64);
        expect(nanmean(emptyRes, axis: 0).shape, equals([0]));
        expect(nanvar(emptyRes, axis: 0).shape, equals([0]));

        // out aliasing on nanmean, nanvar, nanstd
        final aAliasNM = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        nanmean(
          aAliasNM,
          axis: 0,
          out: aAliasNM.slice([const Index(0), const Slice.all()]),
        );
        final aAliasNV = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        nanvar(
          aAliasNV,
          axis: 0,
          out: aAliasNV.slice([const Index(0), const Slice.all()]),
        );
        final aAliasNS = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        nanstd(
          aAliasNS,
          axis: 0,
          out: aAliasNS.slice([const Index(0), const Slice.all()]),
        );
      });
    });
  });

  group('stats branch coverage - min, max, nanmin, nanmax, ptp', () {
    test('min, max, nanmin, nanmax across all dtypes', () {
      for (final dt in DType.values) {
        NDArray.scope(() {
          final a1 = _sample1D(dt);
          expect(min(a1).dtype, equals(dt));
          expect(max(a1).dtype, equals(dt));

          final a1Strided = a1.slice([const Slice(step: 2)]);
          expect(min(a1Strided).dtype, equals(dt));
          expect(max(a1Strided).dtype, equals(dt));

          final a2 = _sample2D(dt);
          expect(min(a2, axis: 0).shape, equals([2]));
          expect(max(a2, axis: 0).shape, equals([2]));

          if (dt.isComplex) {
            expect(() => nanmin(a1), throwsUnsupportedError);
            expect(() => nanmax(a1), throwsUnsupportedError);
          } else {
            expect(nanmin(a1).dtype, equals(dt));
            expect(nanmax(a1).dtype, equals(dt));
            expect(nanmin(a1Strided).dtype, equals(dt));
            expect(nanmax(a1Strided).dtype, equals(dt));
            expect(nanmin(a2, axis: 0).shape, equals([2]));
            expect(nanmax(a2, axis: 0).shape, equals([2]));
            expect(ptp(a1).dtype, equals(dt));
            expect(ptp(a1Strided).dtype, equals(dt));
            expect(ptp(a2, axis: 0).shape, equals([2]));
          }
        });
      }
    });

    test('complex min/max tie-breaking and NaN positions', () {
      for (final cdt in [DType.complex128, DType.complex64]) {
        NDArray.scope(() {
          final cTie = NDArray.fromList(
            [Complex(2.0, 5.0), Complex(2.0, 1.0), Complex(2.0, 8.0)],
            [3],
            cdt,
          ).asAnySpec;
          expect((min(cTie).scalar as Complex).imag, closeTo(1.0, 1e-5));
          expect((max(cTie).scalar as Complex).imag, closeTo(8.0, 1e-5));

          final cFirstNan = NDArray.fromList(
            [Complex(double.nan, 1.0), Complex(2.0, 3.0), Complex(1.0, 4.0)],
            [3],
            cdt,
          ).asAnySpec;
          expect((min(cFirstNan).scalar as Complex).real.isNaN, isTrue);
          expect((max(cFirstNan).scalar as Complex).real.isNaN, isTrue);

          final cLaterNan = NDArray.fromList(
            [Complex(2.0, 3.0), Complex(1.0, double.nan), Complex(4.0, 1.0)],
            [3],
            cdt,
          ).asAnySpec;
          expect((min(cLaterNan).scalar as Complex).imag.isNaN, isTrue);
          expect((max(cLaterNan).scalar as Complex).imag.isNaN, isTrue);

          final c2 = NDArray.fromList(
            [
              Complex(2.0, 5.0),
              Complex(double.nan, 0.0),
              Complex(2.0, 1.0),
              Complex(3.0, 4.0),
              Complex(2.0, 8.0),
              Complex(1.0, double.nan),
            ],
            [3, 2],
            cdt,
          ).asAnySpec;
          expect(min(c2, axis: 0).shape, equals([2]));
          expect(max(c2, axis: 0).shape, equals([2]));
        });
      }
    });

    test(
      'uint64 and generic reduce ordering, out aliasing, and empty output [0, 3] on axis 1',
      () {
        NDArray.scope(() {
          final u64 = NDArray<Uint64>.fromList(
            [10, 2, 3, 20],
            [2, 2],
            DType.uint64,
          );
          expect(min(u64, axis: 0).toList(), equals([3, 2]));
          expect(max(u64, axis: 0).toList(), equals([10, 20]));
          expect(nanmin(u64, axis: 0).toList(), equals([3, 2]));
          expect(nanmax(u64, axis: 0).toList(), equals([10, 20]));

          final f16 = NDArray<Float16>.fromList(
            [10.0, 2.0, 3.0, 20.0],
            [2, 2],
            DType.float16,
          );
          expect(min(f16, axis: 0).toList(), equals([3.0, 2.0]));
          expect(max(f16, axis: 0).toList(), equals([10.0, 20.0]));

          // Empty output shape [0] when reducing [0, 3] along axis 1
          final empty03 = NDArray<Float64>.zeros([0, 3], DType.float64);
          final out0 = NDArray<Float64>.zeros([0], DType.float64);
          expect(min(empty03, axis: 1).shape, equals([0]));
          expect(min(empty03, axis: 1, out: out0).shape, equals([0]));
          expect(max(empty03, axis: 1).shape, equals([0]));
          expect(max(empty03, axis: 1, out: out0).shape, equals([0]));
          expect(nanmin(empty03, axis: 1).shape, equals([0]));
          expect(nanmin(empty03, axis: 1, out: out0).shape, equals([0]));
          expect(nanmax(empty03, axis: 1).shape, equals([0]));
          expect(nanmax(empty03, axis: 1, out: out0).shape, equals([0]));
          expect(ptp(empty03, axis: 1).shape, equals([0]));
          expect(ptp(empty03, axis: 1, out: out0).shape, equals([0]));

          // out aliasing on min, max, nanmin, nanmax, ptp
          for (final op in [
            (NDArray<Float64> x, NDArray<Float64> o) => min(x, axis: 0, out: o),
            (NDArray<Float64> x, NDArray<Float64> o) => max(x, axis: 0, out: o),
            (NDArray<Float64> x, NDArray<Float64> o) =>
                nanmin(x, axis: 0, out: o),
            (NDArray<Float64> x, NDArray<Float64> o) =>
                nanmax(x, axis: 0, out: o),
            (NDArray<Float64> x, NDArray<Float64> o) => ptp(x, axis: 0, out: o),
          ]) {
            final arr = NDArray<Float64>.fromList(
              [4.0, 1.0, 2.0, 3.0],
              [2, 2],
              DType.float64,
            );
            op(arr, arr.slice([const Index(0), const Slice.all()]));
          }

          final c = NDArray<Complex128>.fromList(
            [Complex(1, 2)],
            [1],
            DType.complex128,
          );
          expect(() => ptp(c), throwsArgumentError);
        });
      },
    );
  });

  group('stats branch coverage - cumulative operations and nansum', () {
    test('cumsumAs, cumprodAs, cummin, cummax across dtypes and aliasing', () {
      for (final dt in DType.values) {
        NDArray.scope(() {
          final a2 = _sample2D(dt);
          expect(cumsumAs(a2, dt, axis: 0).shape, equals([2, 2]));
          expect(cumprodAs(a2, dt, axis: 0).shape, equals([2, 2]));
          if (dt.isComplex) {
            expect(() => cummin(a2, axis: 0), throwsArgumentError);
            expect(() => cummax(a2, axis: 0), throwsArgumentError);
          } else {
            expect(cummin(a2, axis: 0).shape, equals([2, 2]));
            expect(cummax(a2, axis: 0).shape, equals([2, 2]));
          }
        });
      }

      NDArray.scope(() {
        // Negative axis and out aliasing for cumsumAs, cumprodAs, cummin, cummax
        for (final op in [
          (NDArray<Float64> x, int? ax, NDArray<Float64> o) =>
              cumsumAs(x, DType.float64, axis: ax, out: o),
          (NDArray<Float64> x, int? ax, NDArray<Float64> o) =>
              cumprodAs(x, DType.float64, axis: ax, out: o),
          (NDArray<Float64> x, int? ax, NDArray<Float64> o) =>
              cummin(x, axis: ax, out: o),
          (NDArray<Float64> x, int? ax, NDArray<Float64> o) =>
              cummax(x, axis: ax, out: o),
        ]) {
          final a1 = NDArray<Float64>.fromList(
            [3.0, 1.0, 4.0, 2.0],
            [4],
            DType.float64,
          );
          op(a1, null, a1);
          final a2 = NDArray<Float64>.fromList(
            [3.0, 1.0, 4.0, 2.0],
            [2, 2],
            DType.float64,
          );
          op(a2, -1, a2);
        }
      });
    });

    test(
      'nansum across all dtypes (axis: null and axis: 0, strided, empty)',
      () {
        for (final dt in DType.values) {
          NDArray.scope(() {
            final a1 = _sample1D(dt);
            expect(nansum(a1).rank, equals(0));
            expect(nansum(a1.slice([const Slice(step: 2)])).rank, equals(0));
            final a2 = _sample2D(dt);
            expect(nansum(a2, axis: 0).shape, equals([2]));
          });
        }

        NDArray.scope(() {
          for (final cdt in [DType.complex128, DType.complex64]) {
            final cNan = NDArray.fromList(
              [
                Complex(1.0, 2.0),
                Complex(double.nan, 5.0),
                Complex(3.0, double.nan),
              ],
              [3],
              cdt,
            ).asAnySpec;
            expect(nansum(cNan).scalar, equals(Complex(1.0, 2.0)));

            final c2Nan = NDArray.fromList(
              [
                Complex(1.0, 2.0),
                Complex(double.nan, 1.0),
                Complex(3.0, 4.0),
                Complex(2.0, 3.0),
              ],
              [2, 2],
              cdt,
            ).asAnySpec;
            expect(nansum(c2Nan, axis: 0).shape, equals([2]));
          }

          final emptyAxis = NDArray<Float64>.zeros([0, 2], DType.float64);
          final outVec = NDArray<Float64>.ones([2], DType.float64);
          nansum(emptyAxis, axis: 0, out: outVec);
          expect(outVec.toList(), equals([0.0, 0.0]));

          final emptyRes = NDArray<Float64>.zeros([2, 0], DType.float64);
          expect(nansum(emptyRes, axis: 0).shape, equals([0]));

          final aAlias = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          nansum(
            aAlias,
            axis: 0,
            out: aAlias.slice([const Index(0), const Slice.all()]),
          );
        });
      },
    );
  });

  group('stats branch coverage - quantile, nanquantile, median, nanmedian', () {
    test('quantile, nanquantile, median, nanmedian across all real dtypes', () {
      for (final dt in DType.values) {
        NDArray.scope(() {
          final a1 = _sample1D(dt);
          final a1Strided = a1.slice([const Slice(step: 2)]);
          final a2 = _sample2D(dt);
          if (dt.isComplex) {
            expect(() => quantile(a1, 0.5), throwsUnsupportedError);
            expect(() => nanquantile(a1, 0.5), throwsUnsupportedError);
            expect(() => percentile(a1, 50.0), throwsUnsupportedError);
            expect(() => nanpercentile(a1, 50.0), throwsUnsupportedError);
            expect(median(a1).dtype, equals(dt));
            expect(median(a1Strided).dtype, equals(dt));
            expect(median(a2, axis: 0).shape, equals([2]));
            expect(nanmedian(a1).dtype, equals(dt));
            expect(nanmedian(a1Strided).dtype, equals(dt));
            expect(nanmedian(a2, axis: 0).shape, equals([2]));
          } else {
            expect(quantile(a1, 0.5).rank, equals(0));
            expect(quantile(a1Strided, 0.5).rank, equals(0));
            expect(quantile(a2, 0.5, axis: 0).shape, equals([2]));
            expect(nanquantile(a1, 0.5).rank, equals(0));
            expect(nanquantile(a1Strided, 0.5).rank, equals(0));
            expect(nanquantile(a2, 0.5, axis: 0).shape, equals([2]));
            expect(nanpercentile(a1, 50.0).rank, equals(0));
            expect(median(a1).rank, equals(0));
            expect(median(a1Strided).rank, equals(0));
            expect(median(a2, axis: 0).shape, equals([2]));
            expect(nanmedian(a1).rank, equals(0));
            expect(nanmedian(a1Strided).rank, equals(0));
            expect(nanmedian(a2, axis: 0).shape, equals([2]));
          }
        });
      }
    });

    test('complex median/nanmedian odd lengths, NaNs, and out aliasing', () {
      for (final cdt in [DType.complex128, DType.complex64]) {
        NDArray.scope(() {
          final cOdd = NDArray.fromList(
            [Complex(3.0, 1.0), Complex(1.0, 2.0), Complex(2.0, 5.0)],
            [3],
            cdt,
          ).asAnySpec;
          expect((median(cOdd).scalar as Complex).real, closeTo(2.0, 1e-5));
          expect((nanmedian(cOdd).scalar as Complex).real, closeTo(2.0, 1e-5));

          final cNan = NDArray.fromList(
            [Complex(3.0, 1.0), Complex(double.nan, 2.0), Complex(1.0, 5.0)],
            [3],
            cdt,
          ).asAnySpec;
          expect((median(cNan).scalar as Complex).real.isNaN, isTrue);
          expect((nanmedian(cNan).scalar as Complex).real, closeTo(2.0, 1e-5));

          final cAllNan = NDArray.fromList(
            [Complex(double.nan, 1.0), Complex(1.0, double.nan)],
            [2],
            cdt,
          ).asAnySpec;
          expect((nanmedian(cAllNan).scalar as Complex).real.isNaN, isTrue);

          final c2Nan = NDArray.fromList(
            [
              Complex(1.0, 0.0),
              Complex(double.nan, 0.0),
              Complex(3.0, 0.0),
              Complex(2.0, 0.0),
              Complex(2.0, 0.0),
              Complex(4.0, 0.0),
            ],
            [3, 2],
            cdt,
          ).asAnySpec;
          expect(median(c2Nan, axis: 0).shape, equals([2]));
          expect(nanmedian(c2Nan, axis: 0).shape, equals([2]));
        });
      }
    });

    test(
      'quantile/nanquantile/median/nanmedian out aliasing and empty [0, 3] axis 1',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [4.0, 1.0, 2.0, 3.0],
            [2, 2],
            DType.float64,
          );
          final outAlias = a.slice([const Index(0), const Slice.all()]);
          quantile(a, 0.5, axis: 0, out: outAlias);
          expect(outAlias.toList(), equals([3.0, 2.0]));

          final a2 = NDArray<Float64>.fromList(
            [4.0, 1.0, 2.0, 3.0],
            [2, 2],
            DType.float64,
          );
          final outAlias2 = a2.slice([const Index(0), const Slice.all()]);
          nanquantile(a2, 0.5, axis: 0, out: outAlias2);
          expect(outAlias2.toList(), equals([3.0, 2.0]));

          final a3 = NDArray<Float64>.fromList(
            [4.0, 1.0, 2.0, 3.0],
            [2, 2],
            DType.float64,
          );
          final outAlias3 = a3.slice([const Index(0), const Slice.all()]);
          median(a3, axis: 0, out: outAlias3);
          expect(outAlias3.toList(), equals([3.0, 2.0]));

          final a4 = NDArray<Float64>.fromList(
            [4.0, 1.0, 2.0, 3.0],
            [2, 2],
            DType.float64,
          );
          final outAlias4 = a4.slice([const Index(0), const Slice.all()]);
          nanmedian(a4, axis: 0, out: outAlias4);
          expect(outAlias4.toList(), equals([3.0, 2.0]));

          // Empty output [0] when reducing [0, 3] along axis 1
          final empty03 = NDArray<Float64>.zeros([0, 3], DType.float64);
          final out0 = NDArray<Float64>.zeros([0], DType.float64);
          expect(quantile(empty03, 0.5, axis: 1).shape, equals([0]));
          expect(quantile(empty03, 0.5, axis: 1, out: out0).shape, equals([0]));
          expect(nanquantile(empty03, 0.5, axis: 1).shape, equals([0]));
          expect(
            nanquantile(empty03, 0.5, axis: 1, out: out0).shape,
            equals([0]),
          );
          expect(median(empty03, axis: 1).shape, equals([0]));
          expect(median(empty03, axis: 1, out: out0).shape, equals([0]));
          expect(nanmedian(empty03, axis: 1).shape, equals([0]));
          expect(nanmedian(empty03, axis: 1, out: out0).shape, equals([0]));
        });
      },
    );
  });

  group('stats branch coverage - average, cov, corrcoef', () {
    test(
      'average across dtypes, weights, returned, out aliasing, and errors',
      () {
        for (final dt in [
          DType.float64,
          DType.float32,
          DType.float16,
          DType.bfloat16,
          DType.int32,
          DType.boolean,
          DType.complex128,
          DType.complex64,
        ]) {
          NDArray.scope(() {
            final a1 = _sample1D(dt);
            final w1 = NDArray<Float64>.fromList(
              [1.0, 2.0, 1.0, 2.0],
              [4],
              DType.float64,
            );
            final res = average(a1, weights: w1);
            expect(res.average.rank, equals(0));
            expect(res.sumOfWeights, isNull);

            final resRet = average(a1, weights: w1, returned: true);
            expect(resRet.sumOfWeights, isNotNull);

            final resNoW = average(a1, returned: true);
            expect(resNoW.sumOfWeights, isNotNull);

            final a2 = _sample2D(dt);
            final wAxis = NDArray<Float64>.fromList(
              [1.0, 3.0],
              [2],
              DType.float64,
            );
            final resAxis = average(
              a2,
              axis: 0,
              weights: wAxis,
              returned: true,
            );
            expect(resAxis.average.shape, equals([2]));
            expect(resAxis.sumOfWeights!.shape, equals([2]));
          });
        }

        NDArray.scope(() {
          final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
          final wZero = NDArray<Float64>.fromList(
            [1.0, -1.0],
            [2],
            DType.float64,
          );
          expect(average(a, weights: wZero).average.scalar.isInfinite, isTrue);

          final a2 = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          expect(() => average(a2, weights: wZero), throwsArgumentError);

          // out aliasing with weights
          final w2 = NDArray<Float64>.fromList(
            [1.0, 1.0, 1.0, 1.0],
            [2, 2],
            DType.float64,
          );
          final wAlias = w2.slice([const Index(0), const Slice.all()]);
          average(a2, axis: 0, weights: w2, returned: true, out: wAlias);
        });
      },
    );

    test(
      'empty all/any, nanvar mixed NaN & int edge cases, nanquantile/nanmedian errors',
      () {
        NDArray.scope(() {
          final emptyF = NDArray<Float64>.zeros([0], DType.float64);
          expect(all(emptyF).scalar, isTrue);
          expect(any(emptyF).scalar, isFalse);

          // _computeNanvarGeneral 2D with mixed valid and NaN in same slice
          final f2d = NDArray<Float64>.fromList(
            [1.0, 2.0, double.nan, 4.0, 3.0, double.nan],
            [3, 2],
            DType.float64,
          );
          final nvF = nanvar(f2d, axis: 0, ddof: 1);
          expect(nvF.getCell([0]), closeTo(2.0, 1e-6));

          for (final cdt in [DType.complex128, DType.complex64]) {
            final c2d = NDArray.fromList(
              [
                Complex(1.0, 1.0),
                Complex(2.0, 0.0),
                Complex(double.nan, 1.0),
                Complex(4.0, 2.0),
                Complex(3.0, 1.0),
                Complex(1.0, double.nan),
              ],
              [3, 2],
              cdt,
            ).asAnySpec;
            final nvC = nanvar(c2d, axis: 0, ddof: 1);
            expect(nvC.shape, equals([2]));
          }

          // Integer and complex nanvar with empty array / ddof >= size and out
          expect(
            nanvar(
              NDArray<Complex128>.zeros([0], DType.complex128),
            ).scalar.isNaN,
            isTrue,
          );
          final emptyInt = NDArray<Int32>.zeros([0], DType.int32);
          final outScalarF = NDArray<Float64>.zeros([], DType.float64);
          expect(nanvar(emptyInt).scalar.isNaN, isTrue);
          nanvar(emptyInt, out: outScalarF);
          expect(outScalarF.scalar.isNaN, isTrue);

          final emptyIntAxis = NDArray<Int32>.zeros([0, 2], DType.int32);
          final outVecF = NDArray<Float64>.zeros([2], DType.float64);
          expect(nanvar(emptyIntAxis, axis: 0).getCell([0]).isNaN, isTrue);
          nanvar(emptyIntAxis, axis: 0, out: outVecF);
          expect(outVecF.getCell([0]).isNaN, isTrue);

          final emptyIntRes = NDArray<Int32>.zeros([2, 0], DType.int32);
          expect(nanvar(emptyIntRes, axis: 0).shape, equals([0]));

          // nanquantile, nanpercentile, nanmedian error preconditions and empty
          final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
          final disp = NDArray<Float64>.fromList([1.0], [1], DType.float64)
            ..dispose();
          expect(() => nanquantile(disp, 0.5), throwsStateError);
          expect(() => nanpercentile(disp, 50.0), throwsStateError);
          expect(() => nanmedian(disp), throwsStateError);

          expect(() => nanquantile(a, -0.1), throwsArgumentError);
          expect(() => nanquantile(a, 1.1), throwsArgumentError);
          expect(() => nanquantile(a, double.nan), throwsArgumentError);
          expect(() => nanpercentile(a, -1.0), throwsArgumentError);
          expect(() => nanpercentile(a, 101.0), throwsArgumentError);
          expect(() => nanpercentile(a, double.nan), throwsArgumentError);

          expect(() => nanquantile(a, 0.5, axis: 5), throwsRangeError);
          expect(() => nanmedian(a, axis: 5), throwsRangeError);
          expect(nanquantile(a, 0.5, axis: -1).scalar, closeTo(1.5, 1e-6));
          expect(nanmedian(a, axis: -1).scalar, closeTo(1.5, 1e-6));

          final empty2x0 = NDArray<Float64>.zeros([2, 0], DType.float64);
          final outEmpty0 = NDArray<Float64>.zeros([0], DType.float64);
          expect(nanquantile(empty2x0, 0.5, axis: 0).shape, equals([0]));
          expect(
            nanquantile(empty2x0, 0.5, axis: 0, out: outEmpty0).shape,
            equals([0]),
          );
          expect(nanmedian(empty2x0, axis: 0).shape, equals([0]));
          expect(
            nanmedian(empty2x0, axis: 0, out: outEmpty0).shape,
            equals([0]),
          );

          final cOutBad = NDArray<Complex128>.zeros([2, 2], DType.complex128);
          expect(
            () => cov(a.asAnySpec, y: a.asAnySpec, out: cOutBad.asAnySpec),
            throwsArgumentError,
          );
        });
      },
    );

    test(
      'cov and corrcoef on complex128/complex64, weights, ddof, and errors',
      () {
        for (final cdt in [DType.complex128, DType.complex64]) {
          NDArray.scope(() {
            final m1 = NDArray.fromList(
              [Complex(1.0, 2.0), Complex(2.0, 1.0), Complex(3.0, 4.0)],
              [3],
              cdt,
            ).asAnySpec;
            final y1 = NDArray.fromList(
              [Complex(2.0, 0.0), Complex(1.0, 3.0), Complex(4.0, 2.0)],
              [3],
              cdt,
            ).asAnySpec;
            final c1 = cov(m1, y: y1);
            expect(c1.shape, equals([2, 2]));
            expect(c1.dtype, equals(DType.complex128));

            final cc1 = corrcoef(m1, y: y1);
            expect(cc1.shape, equals([2, 2]));
            expect(cc1.dtype, equals(DType.complex128));

            // 0D scalar corrcoef when m1 is 1D and y is null
            final ccScalar = corrcoef(m1);
            expect(ccScalar.shape, isEmpty);
            final outScalar = NDArray<Complex128>.zeros([], DType.complex128);
            corrcoef(m1, out: outScalar.asAnySpec);
            expect(outScalar.scalar.real, closeTo(1.0, 1e-5));

            final m1Const = NDArray.fromList(
              [Complex(1.0, 1.0), Complex(1.0, 1.0)],
              [2],
              cdt,
            ).asAnySpec;
            expect((corrcoef(m1Const).scalar as Complex).real.isNaN, isTrue);

            // 2D with rowvar: false, fweights, aweights, out
            final m2 = NDArray.fromList(
              [
                Complex(1.0, 0.0),
                Complex(2.0, 1.0),
                Complex(2.0, 1.0),
                Complex(1.0, 2.0),
                Complex(3.0, 2.0),
                Complex(4.0, 0.0),
              ],
              [3, 2],
              cdt,
            ).asAnySpec;
            final y2 = NDArray.fromList(
              [
                Complex(1.0, 1.0),
                Complex(0.0, 2.0),
                Complex(2.0, 0.0),
                Complex(3.0, 1.0),
                Complex(0.0, 3.0),
                Complex(2.0, 2.0),
              ],
              [3, 2],
              cdt,
            ).asAnySpec;
            final fw = NDArray<Int64>.fromList([1, 2, 1], [3], DType.int64);
            final aw = NDArray<Float64>.fromList(
              [0.5, 1.5, 1.0],
              [3],
              DType.float64,
            );
            final outCov = NDArray<Complex128>.zeros([4, 4], DType.complex128);
            cov(
              m2,
              y: y2,
              rowvar: false,
              fweights: fw,
              aweights: aw,
              out: outCov.asAnySpec,
            );
            expect(outCov.shape, equals([4, 4]));

            final outCc = NDArray<Complex128>.zeros([4, 4], DType.complex128);
            corrcoef(m2, y: y2, rowvar: false, out: outCc.asAnySpec);
            expect(outCc.shape, equals([4, 4]));

            // High ddof where denominator <= 0 on complex cov
            final cHighDdof = cov(m1, ddof: 5);
            expect((cHighDdof.scalar as Complex).real.isNaN, isTrue);

            // Zero variance row in complex corrcoef
            final mConst = NDArray.fromList(
              [
                Complex(1.0, 1.0),
                Complex(1.0, 1.0),
                Complex(1.0, 1.0),
                Complex(1.0, 0.0),
                Complex(2.0, 1.0),
                Complex(3.0, 2.0),
              ],
              [2, 3],
              cdt,
            ).asAnySpec;
            final ccConst = corrcoef(mConst);
            expect((ccConst.getCell([0, 0]) as Complex).real.isNaN, isTrue);

            // Complex cov fweights/aweights errors
            final fwNeg = NDArray<Int64>.fromList([1, -1, 1], [3], DType.int64);
            final fwShort = NDArray<Int64>.fromList([1, 1], [2], DType.int64);
            final awNeg = NDArray<Float64>.fromList(
              [1.0, -0.5, 1.0],
              [3],
              DType.float64,
            );
            final awShort = NDArray<Float64>.fromList(
              [1.0, 1.0],
              [2],
              DType.float64,
            );
            expect(() => cov(m1, fweights: m1), throwsArgumentError);
            expect(() => cov(m1, fweights: fwShort), throwsArgumentError);
            expect(() => cov(m1, fweights: fwNeg), throwsArgumentError);
            expect(() => cov(m1, aweights: m1), throwsArgumentError);
            expect(() => cov(m1, aweights: awShort), throwsArgumentError);
            expect(() => cov(m1, aweights: awNeg), throwsArgumentError);

            // Complex cov/corrcoef bad out dtype/shape
            final badOutF = NDArray<Float64>.zeros([], DType.float64);
            expect(() => cov(m1, out: badOutF.asAnySpec), throwsArgumentError);
            expect(
              () => corrcoef(m1, out: badOutF.asAnySpec),
              throwsArgumentError,
            );
          });
        }

        NDArray.scope(() {
          final m = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );
          final outScalarF = NDArray<Float64>.zeros([], DType.float64);
          corrcoef(m, out: outScalarF);
          expect(outScalarF.scalar, closeTo(1.0, 1e-5));

          final mConst1D = NDArray<Float64>.fromList(
            [2.0, 2.0, 2.0],
            [3],
            DType.float64,
          );
          expect(corrcoef(mConst1D).scalar.isNaN, isTrue);

          // Real 2D with y 2D and rowvar: false
          final m2R = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
            DType.float64,
          );
          final y2R = NDArray<Float64>.fromList(
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [3, 2],
            DType.float64,
          );
          expect(cov(m2R, y: y2R, rowvar: false).shape, equals([4, 4]));

          final cY = NDArray<Complex128>.fromList(
            [Complex(1, 0), Complex(2, 0), Complex(3, 0)],
            [3],
            DType.complex128,
          );
          expect(() => cov(m.asAnySpec, y: cY.asAnySpec), throwsArgumentError);
          expect(
            () => corrcoef(m.asAnySpec, y: cY.asAnySpec),
            throwsArgumentError,
          );

          final s0 = NDArray<Float64>.scalar(1.0, dtype: DType.float64);
          final s3 = NDArray<Float64>.zeros([1, 1, 1], DType.float64);
          expect(() => cov(s0), throwsArgumentError);
          expect(() => cov(s3), throwsArgumentError);
          expect(() => cov(m, y: s0), throwsArgumentError);
          expect(() => cov(m, y: s3), throwsArgumentError);
          expect(() => corrcoef(s0), throwsArgumentError);
          expect(() => corrcoef(s3), throwsArgumentError);
          expect(() => corrcoef(m, y: s0), throwsArgumentError);
          expect(() => corrcoef(m, y: s3), throwsArgumentError);

          final fw2D = NDArray<Int64>.fromList([1, 2, 3], [1, 3], DType.int64);
          final fwShort = NDArray<Int64>.fromList([1, 2], [2], DType.int64);
          final fwNeg = NDArray<Int64>.fromList([1, -1, 2], [3], DType.int64);
          expect(() => cov(m, fweights: fw2D), throwsArgumentError);
          expect(() => cov(m, fweights: fwShort), throwsArgumentError);
          expect(() => cov(m, fweights: cY), throwsArgumentError);
          expect(() => cov(m, fweights: fwNeg), throwsArgumentError);

          final aw2D = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0],
            [1, 3],
            DType.float64,
          );
          final awShort = NDArray<Float64>.fromList(
            [1.0, 2.0],
            [2],
            DType.float64,
          );
          final awComplex = cY;
          final awNeg = NDArray<Float64>.fromList(
            [1.0, -0.5, 2.0],
            [3],
            DType.float64,
          );
          expect(() => cov(m, aweights: aw2D), throwsArgumentError);
          expect(() => cov(m, aweights: awShort), throwsArgumentError);
          expect(() => cov(m, aweights: awComplex), throwsArgumentError);
          expect(() => cov(m, aweights: awNeg), throwsArgumentError);
        });
      },
    );
  });
}
