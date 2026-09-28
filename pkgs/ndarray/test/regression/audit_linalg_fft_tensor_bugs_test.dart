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

import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Bug 1: Double pointer offsetting in norm() and _matrixNorm()', () {
    test('_matrixNorm (ord: 1, -1, inf, -inf) on reversed/sliced 2D views', () {
      NDArray.scope(() {
        final base = NDArray.fromList(
          Float64List.fromList([
            1.0,
            -5.0,
            3.0,
            -4.0,
            2.0,
            -6.0,
            7.0,
            -8.0,
            9.0,
          ]),
          [3, 3],
          DType.float64,
        );

        final revRows = base.slice([const Slice(step: -1), const Slice.all()]);
        final revCols = base.slice([const Slice.all(), const Slice(step: -1)]);
        final revBoth = base.slice([
          const Slice(step: -1),
          const Slice(step: -1),
        ]);

        for (final view in [revRows, revCols, revBoth]) {
          expect(view.offsetElements, greaterThan(0));
          final contig = view.copy();

          for (final ord in [1, -1, double.infinity, double.negativeInfinity]) {
            final resView = norm(view, ord: ord);
            final resContig = norm(contig, ord: ord);
            expect(
              resView.getCell([]),
              closeTo(resContig.getCell([]), 1e-12),
              reason: 'Mismatch for matrix norm ord=$ord on reversed 2D view',
            );
          }
        }
      });
    });

    test('_matrixNorm on 3D batched strided tensor with axis and keepdims', () {
      NDArray.scope(() {
        final data = Float64List.fromList(
          List<double>.generate(24, (i) => (i.isEven ? 1.0 : -1.0) * (i + 1)),
        );
        final batch = NDArray.fromList(data, [2, 3, 4], DType.float64);

        final view = batch.slice([
          const Slice(start: 1),
          const Slice(step: -1),
          const Slice(step: -1),
        ]);
        final contig = view.copy();

        for (final ord in [1, -1, double.infinity, double.negativeInfinity]) {
          final resView = norm(view, ord: ord, axis: [1, 2], keepdims: true);
          final resContig = norm(
            contig,
            ord: ord,
            axis: [1, 2],
            keepdims: true,
          );
          expect(resView.shape, resContig.shape);
          expect(
            resView.getCell([0, 0, 0]),
            closeTo(resContig.getCell([0, 0, 0]), 1e-12),
          );
        }
      });
    });

    test('vector norm and multi-axis reduction on reversed/sliced views', () {
      NDArray.scope(() {
        final vBase = NDArray.fromList(
          Float64List.fromList([10.0, -3.0, 4.0, -12.0, 5.0]),
          [5],
          DType.float64,
        );
        final vRev = vBase.slice([const Slice(start: 3, stop: 0, step: -1)]);
        expect(vRev.offsetElements, greaterThan(0));
        final vContig = vRev.copy();

        for (final ord in [2, 1, double.infinity, 3]) {
          expect(
            norm(vRev, ord: ord).getCell([]),
            closeTo(norm(vContig, ord: ord).getCell([]), 1e-12),
            reason: 'Vector norm ord=$ord mismatch on reversed slice',
          );
        }

        final tBase = NDArray.fromList(
          Float64List.fromList(
            List<double>.generate(18, (i) => (i - 8).toDouble()),
          ),
          [3, 2, 3],
          DType.float64,
        );
        final tView = tBase.slice([
          const Slice(step: -1),
          const Slice(start: 0, stop: 2),
          const Slice(step: -1),
        ]);
        final tContig = tView.copy();

        final nView = norm(tView, ord: 2, axis: [0, 2]);
        final nContig = norm(tContig, ord: 2, axis: [0, 2]);
        expect(nView.shape, [2]);
        for (var i = 0; i < 2; i++) {
          expect(nView.getCell([i]), closeTo(nContig.getCell([i]), 1e-12));
        }
      });
    });
  });

  group('Bug 2: matmul stride validation, pointer aliasing, and cleanup', () {
    test('1D dot product with negative strides (float64 and float32)', () {
      NDArray.scope(() {
        final a64 = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
          [4],
          DType.float64,
        );
        final b64 = NDArray.fromList(
          Float64List.fromList([10.0, 20.0, 30.0, 40.0]),
          [4],
          DType.float64,
        );

        final aRev = a64.slice([const Slice(step: -1)]);
        final bRev = b64.slice([const Slice(step: -1)]);

        expect(
          matmul(aRev, b64).getCell([]),
          closeTo(matmul(aRev.copy(), b64).getCell([]), 1e-12),
        );
        expect(
          matmul(a64, bRev).getCell([]),
          closeTo(matmul(a64, bRev.copy()).getCell([]), 1e-12),
        );
        expect(
          matmul(aRev, bRev).getCell([]),
          closeTo(matmul(aRev.copy(), bRev.copy()).getCell([]), 1e-12),
        );

        final a32 = NDArray.fromList(
          Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
          [4],
          DType.float32,
        );
        final b32 = NDArray.fromList(
          Float32List.fromList([5.0, 6.0, 7.0, 8.0]),
          [4],
          DType.float32,
        );
        final a32Rev = a32.slice([const Slice(step: -1)]);
        expect(
          matmul(a32Rev, b32).getCell([]),
          closeTo(matmul(a32Rev.copy(), b32).getCell([]), 1e-5),
        );
      });
    });

    test(
      '1D dot product with zero stride (broadcasted vector) including mixed dtypes',
      () {
        NDArray.scope(() {
          final scalar1D = NDArray.fromList(Float64List.fromList([3.0]), [
            1,
          ], DType.float64);
          final bcast = broadcastTo(scalar1D, [4]);
          expect(bcast.strides[0], 0);

          final vec = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [4],
            DType.float64,
          );
          final res = matmul(bcast, vec);
          expect(res.getCell([]), closeTo(30.0, 1e-12));

          // Mixed dtype (int32 + float64) throws ArgumentError; explicit astype works
          final scalarInt = NDArray.fromList(Int32List.fromList([3]), [
            1,
          ], DType.int32);
          final bcastInt = broadcastTo(scalarInt, [4]);
          expect(bcastInt.strides[0], 0);
          expect(() => matmul<DTypeTag>(bcastInt, vec), throwsArgumentError);
          final resMixed = matmul(bcastInt.astype(DType.float64), vec);
          expect(resMixed.getCell([]), closeTo(30.0, 1e-12));
          expect(scalarInt.getCell([0]), 3);
        });
      },
    );

    test(
      '2D matmul with reversed rows (negative outer stride) and zero strides',
      () {
        NDArray.scope(() {
          final baseA = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [3, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            Float64List.fromList([10.0, 20.0, 30.0, 40.0]),
            [2, 2],
            DType.float64,
          );

          final aRevRows = baseA.slice([const Slice(step: -1), Slice.all()]);
          expect(aRevRows.strides[0], lessThan(0));
          expect(aRevRows.strides[1], 1);

          final res = matmul(aRevRows, b);
          final expected = matmul(aRevRows.copy(), b);
          for (var i = 0; i < 3; i++) {
            for (var j = 0; j < 2; j++) {
              expect(
                res.getCell([i, j]),
                closeTo(expected.getCell([i, j]), 1e-12),
              );
            }
          }

          final singleRow = NDArray.fromList(Float64List.fromList([2.0, 3.0]), [
            1,
            2,
          ], DType.float64);
          final aBcast = broadcastTo(singleRow, [3, 2]);
          expect(aBcast.strides[0], 0);

          final resBcast = matmul(aBcast, b);
          final expectedBcast = matmul(aBcast.copy(), b);
          for (var i = 0; i < 3; i++) {
            for (var j = 0; j < 2; j++) {
              expect(
                resBcast.getCell([i, j]),
                closeTo(expectedBcast.getCell([i, j]), 1e-12),
              );
            }
          }
        });
      },
    );

    test(
      'matmul with aliased out slice or transposed view does not corrupt output',
      () {
        NDArray.scope(() {
          final buf = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0]),
            [3, 3],
            DType.float64,
          );
          final aSlice = buf.slice([
            const Slice(start: 0, stop: 2),
            const Slice(start: 0, stop: 2),
          ]);
          final b = NDArray.fromList(
            Float64List.fromList([2.0, 1.0, 1.0, 2.0]),
            [2, 2],
            DType.float64,
          );

          final expected = matmul(aSlice.copy(), b);

          final outSlice = buf.slice([
            const Slice(start: 0, stop: 2),
            const Slice(start: 0, stop: 2),
          ]);
          matmul(aSlice, b, out: outSlice);

          for (var i = 0; i < 2; i++) {
            for (var j = 0; j < 2; j++) {
              expect(
                outSlice.getCell([i, j]),
                closeTo(expected.getCell([i, j]), 1e-12),
              );
            }
          }

          // Also test writing into a.transpose() sharing underlying memory
          final a2 = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b2 = NDArray<Float64>.fromList(
            [5.0, 6.0, 7.0, 8.0],
            [2, 2],
            DType.float64,
          );
          final expected2 = matmul(a2, b2);
          final a2T = a2.transpose();
          matmul(a2, b2, out: a2T);
          for (var i = 0; i < 2; i++) {
            for (var j = 0; j < 2; j++) {
              expect(a2T[[i, j]], closeTo(expected2[[i, j]], 1e-12));
            }
          }
        });
      },
    );

    test('matmul dtype promotion cleanup does not dispose caller input', () {
      NDArray.scope(() {
        final aInt = NDArray.fromList(Int32List.fromList([1, 2, 3, 4]), [
          2,
          2,
        ], DType.int32);
        final bFloat = NDArray.fromList(
          Float64List.fromList([1.0, 0.0, 0.0, 1.0]),
          [2, 2],
          DType.float64,
        );

        expect(() => matmul<DTypeTag>(aInt, bFloat), throwsArgumentError);
        final res = matmul(aInt.astype(DType.float64), bFloat);
        expect(res.getCell([0, 0]), closeTo(1.0, 1e-12));
        expect(aInt.getCell([1, 1]), 4);
      });
    });

    test('matmul rejects 0D scalar inputs with ArgumentError', () {
      NDArray.scope(() {
        final s = NDArray.scalar(2.0, dtype: DType.float64);
        final v = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          2,
        ], DType.float64);
        expect(() => matmul(s, v), throwsArgumentError);
        expect(() => matmul(v, s), throwsArgumentError);
      });
    });
  });

  group(
    'Bug 3: tensor_contractions memory management and single-operand views',
    () {
      test(
        'tensordot, einsum, and vdot return non-view root arrays that dispose cleanly',
        () {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            Float64List.fromList([5.0, 6.0, 7.0, 8.0]),
            [2, 2],
            DType.float64,
          );

          final td = tensordot(a, b, axes: 1);
          expect(td.isView, isFalse);
          expect(td.getCell([0, 0]), closeTo(19.0, 1e-12));
          td.dispose();
          expect(td.isDisposed, isTrue);

          final es = einsum(EinsumSubscripts.parse('ij,jk->ik'), [a, b]);
          expect(es.isView, isFalse);
          expect(es.getCell([0, 0]), closeTo(19.0, 1e-12));
          es.dispose();
          expect(es.isDisposed, isTrue);

          final einSum = einsum<Float64>(EinsumSubscripts.parse('ij->i'), [a]);
          expect(einSum.isView, isFalse);
          einSum.dispose();
          expect(einSum.isDisposed, isTrue);

          final vd = vdot(a, b);
          expect(vd.isView, isFalse);
          expect(vd.getCell([]), closeTo(70.0, 1e-12));
          vd.dispose();
          expect(vd.isDisposed, isTrue);

          a.dispose();
          b.dispose();
        },
      );

      test(
        'single-operand einsum view does not detach caller input from outer scope',
        () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
              ]),
              [3, 3],
              DType.float64,
            );

            NDArray.scope(() {
              final diag = einsum(EinsumSubscripts.parse('ii->i'), [a]);
              expect(diag.isView, isTrue);
              expect(diag.getCell([0]), 1.0);
              expect(diag.getCell([1]), 5.0);
              expect(diag.getCell([2]), 9.0);

              final trans = einsum(EinsumSubscripts.parse('ij->ji'), [a]);
              expect(trans.isView, isTrue);
              expect(trans.getCell([0, 1]), 4.0);
            });

            // After nested scope exits, `a` must still be valid and readable!
            expect(a.isDisposed, isFalse);
            expect(a.getCell([0, 0]), 1.0);
            expect(a.getCell([2, 2]), 9.0);
          });
        },
      );

      test(
        'single-operand einsum on reversed view does not double-offset pointer',
        () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
              ]),
              [3, 3],
              DType.float64,
            );
            final rev = a.slice([const Slice(step: -1), const Slice(step: -1)]);
            expect(rev.offsetElements, greaterThan(0));

            final diagRev = einsum(EinsumSubscripts.parse('ii->i'), [rev]);
            final diagContig = einsum(EinsumSubscripts.parse('ii->i'), [
              rev.copy(),
            ]);
            for (var i = 0; i < 3; i++) {
              expect(
                diagRev.getCell([i]) as num,
                closeTo(diagContig.getCell([i]) as num, 1e-12),
              );
            }

            final sumRev = einsum<Float64>(EinsumSubscripts.parse('ij->i'), [
              rev,
            ]);
            final sumContig = einsum<Float64>(EinsumSubscripts.parse('ij->i'), [
              rev.copy(),
            ]);
            for (var i = 0; i < 3; i++) {
              expect(sumRev[[i]], closeTo(sumContig[[i]], 1e-12));
            }
          });
        },
      );
    },
  );

  group('Bug 4: kron stale strides after copy and unsafe cleanup', () {
    test(
      'kron on negative-stride view with dtype casting matches contiguous copy',
      () {
        NDArray.scope(() {
          final aInt = NDArray.fromList(Int32List.fromList([1, 2, 3, 4]), [
            2,
            2,
          ], DType.int32);
          final aRev = aInt.slice([
            const Slice(step: -1),
            const Slice(step: -1),
          ]);
          expect(aRev.strides[0], lessThan(0));

          final bFloat = NDArray.fromList(
            Float64List.fromList([10.0, 20.0, 30.0, 40.0]),
            [2, 2],
            DType.float64,
          );
          final bRev = bFloat.slice([
            const Slice(step: -1),
            const Slice(step: -1),
          ]);

          expect(() => kron<DTypeTag>(aRev, bRev), throwsArgumentError);
          final aRevFloat = aInt.astype(DType.float64).slice([
            const Slice(step: -1),
            const Slice(step: -1),
          ]);
          final res = kron(aRevFloat, bRev);
          final expected = kron(aRevFloat.copy(), bRev.copy());

          expect(res.shape, expected.shape);
          for (var i = 0; i < res.shape[0]; i++) {
            for (var j = 0; j < res.shape[1]; j++) {
              expect(
                res.getCell([i, j]),
                closeTo(expected.getCell([i, j]), 1e-12),
              );
            }
          }

          expect(aRev.getCell([0, 0]), 4);
        });
      },
    );
  });

  group('Bug 5: irfft odd-length fallback crashes and out-of-bounds access', () {
    test(
      'irfft odd target length n with real/integer inputs and round-trip',
      () {
        NDArray.scope(() {
          final signal = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0]),
            [5],
            DType.float64,
          );
          final freq = rfft(signal);
          final recon = irfft(freq, n: 5);
          expect(recon.shape, [5]);
          for (var i = 0; i < 5; i++) {
            expect(recon.getCell([i]), closeTo(signal.getCell([i]), 1e-9));
          }

          final realInput = NDArray.fromList(
            Float64List.fromList([10.0, -2.0, 1.0]),
            [3],
            DType.float64,
          );
          final irReal = irfft(realInput, n: 5);
          expect(irReal.shape, [5]);

          final intInput = NDArray.fromList(Int32List.fromList([10, -2, 1]), [
            3,
          ], DType.int32);
          final irInt = irfft(intInput, n: 5);
          expect(irInt.shape, [5]);
          for (var i = 0; i < 5; i++) {
            expect(irInt.getCell([i]), closeTo(irReal.getCell([i]), 1e-9));
          }
        });
      },
    );

    test(
      'irfft odd target length with M > targetLen (truncation) and M < half (padding)',
      () {
        NDArray.scope(() {
          final longInput = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            [8],
            DType.float64,
          );
          final resTrunc = irfft(longInput, n: 5);
          final resTruncExact = irfft(
            longInput.slice([const Slice(start: 0, stop: 3)]),
            n: 5,
          );
          expect(resTrunc.shape, [5]);
          for (var i = 0; i < 5; i++) {
            expect(
              resTrunc.getCell([i]),
              closeTo(resTruncExact.getCell([i]), 1e-12),
            );
          }

          final shortInput = NDArray.fromList(
            Float64List.fromList([4.0, 1.0]),
            [2],
            DType.float64,
          );
          final resPad = irfft(shortInput, n: 7);
          expect(resPad.shape, [7]);

          final explicitPadded = NDArray.fromList(
            Float64List.fromList([4.0, 1.0, 0.0, 0.0]),
            [4],
            DType.float64,
          );
          final resExplicit = irfft(explicitPadded, n: 7);
          for (var i = 0; i < 7; i++) {
            expect(
              resPad.getCell([i]),
              closeTo(resExplicit.getCell([i]), 1e-9),
            );
          }
        });
      },
    );
  });

  group('Bug 6: fft unhandled DTypes, uint64 overflow, and empty axis crash', () {
    test(
      'fft, ifft, rfft, irfft, fftn handle empty axis (size == 0) with explicit n',
      () {
        NDArray.scope(() {
          final empty2D = NDArray<Float64>.create([2, 0], DType.float64);

          final f = fft(empty2D, n: 4, axis: 1);
          expect(f.shape, [2, 4]);
          expect(f.getCell([0, 0]).real, 0.0);

          final inv = ifft(empty2D, n: 4, axis: 1);
          expect(inv.shape, [2, 4]);

          final rf = rfft(empty2D, n: 4, axis: 1);
          expect(rf.shape, [2, 3]);

          final irf = irfft(empty2D, n: 4, axis: 1);
          expect(irf.shape, [2, 4]);

          final fn = fftn(empty2D, s: [2, 4]);
          expect(fn.shape, [2, 4]);
        });
      },
    );

    test(
      'fft2/fftn supports int8, uint16, uint32, uint64, float16, bfloat16, boolean',
      () {
        NDArray.scope(() {
          final refFloat = NDArray.fromList(
            Float64List.fromList([1.0, 0.0, 0.0, 1.0]),
            [2, 2],
            DType.float64,
          );
          final expected = fft2(refFloat);

          final dtypesToTest = <DType<AnySpec>>[
            DType.int8,
            DType.uint16,
            DType.uint32,
            DType.uint64,
            DType.float16,
            DType.bfloat16,
            DType.boolean,
          ];

          for (final dt in dtypesToTest) {
            final arr = refFloat.astype(dt);
            final res = fft2(arr);
            expect(res.shape, [2, 2], reason: 'fft2 shape failed for $dt');
            expect(
              res.getCell([0, 0]).real,
              closeTo(expected.getCell([0, 0]).real, 1e-2),
              reason: 'fft2 DC component failed for $dt',
            );
          }
        });
      },
    );

    test(
      'uint64 values >= 2^63 are converted as unsigned positive values in fft and fftn',
      () {
        NDArray.scope(() {
          final u64 = NDArray<Uint64>.create([2], DType.uint64);
          u64.pointer.cast<ffi.Uint64>()[0] = -9223372036854775808;
          u64.pointer.cast<ffi.Uint64>()[1] = 0;

          final res1D = fft(u64);
          final dc1D = res1D.getCell([0]).real;
          expect(dc1D, greaterThan(0.0));
          expect(dc1D, closeTo(9223372036854775808.0, 1e5));

          final u64_2D = u64.reshape([1, 2]);
          final res2D = fft2(u64_2D);
          final dc2D = res2D.getCell([0, 0]).real;
          expect(dc2D, greaterThan(0.0));
          expect(dc2D, closeTo(9223372036854775808.0, 1e5));
        });
      },
    );
  });

  group(
    'Bug 7: optimize (nelder_mead, lbfgs) with non-contiguous inputs and gradients',
    () {
      test(
        'nelder_mead with non-contiguous strided x0 converges to true minimum',
        () {
          NDArray.scope(() {
            final buf = NDArray.fromList(
              Float64List.fromList([999.0, 4.0, -999.0, -3.0]),
              [4],
              DType.float64,
            );
            final x0 = buf.slice([const Slice(start: 1, stop: 4, step: 2)]);
            expect(x0.isContiguous, isFalse);
            expect(x0.getCell([0]), 4.0);
            expect(x0.getCell([1]), -3.0);

            double quad(NDArray<Float64> x) {
              final dx = x.getCell([0]) - 1.0;
              final dy = x.getCell([1]) - 2.0;
              return dx * dx + dy * dy;
            }

            final result = minimize(
              quad,
              x0,
              method: MinimizeMethod.nelderMead,
            );
            expect(result.success, isTrue);
            expect(result.x.getCell([0]), closeTo(1.0, 1e-3));
            expect(result.x.getCell([1]), closeTo(2.0, 1e-3));
          });
        },
      );

      test(
        'lbfgs with non-contiguous x0 and non-contiguous gradient callback',
        () {
          NDArray.scope(() {
            final buf = NDArray.fromList(
              Float64List.fromList([-5.0, 3.5, 100.0]),
              [3],
              DType.float64,
            );
            final x0 = buf.slice([const Slice(start: 1, step: -1)]);
            expect(x0.isContiguous, isFalse);

            double quad(NDArray<Float64> x) {
              final dx = x.getCell([0]) - 1.0;
              final dy = x.getCell([1]) - 2.0;
              return dx * dx + dy * dy;
            }

            NDArray<Float64> nonContigJac(NDArray<Float64> x) {
              final g0 = 2.0 * (x.getCell([0]) - 1.0);
              final g1 = 2.0 * (x.getCell([1]) - 2.0);
              final gBuf = NDArray.fromList(Float64List.fromList([g1, g0]), [
                2,
              ], DType.float64);
              return gBuf.slice([const Slice(step: -1)]);
            }

            final result = minimize(
              quad,
              x0,
              method: MinimizeMethod.lbfgs,
              jac: nonContigJac,
            );
            expect(result.success, isTrue);
            expect(result.x.getCell([0]), closeTo(1.0, 1e-5));
            expect(result.x.getCell([1]), closeTo(2.0, 1e-5));
          });
        },
      );
    },
  );

  group('L2 Synthesis Additional Regression Tests', () {
    test(
      'matmul complex 1D dot product with stride 0 (broadcasted vector) does not read out of bounds',
      () {
        NDArray.scope(() {
          final scalarVec = NDArray.fromList(
            [Complex(2.0, 1.0)],
            [1],
            DType.complex128,
          );
          final a = broadcastTo(scalarVec, [4]);
          expect(a.strides, [0]);
          final b = NDArray.fromList(
            [
              Complex(1.0, 0.0),
              Complex(0.0, 1.0),
              Complex(2.0, -1.0),
              Complex(-1.0, 2.0),
            ],
            [4],
            DType.complex128,
          );

          final res = matmul(a, b);
          // Expected dot product: (2+i)*(1 + i + 2-i -1+2i) = (2+i)*(2+2i) = 4 + 4i + 2i - 2 = 2 + 6i
          expect(res.shape, isEmpty);
          final val = res.getCell([]);
          expect(val.real, closeTo(2.0, 1e-12));
          expect(val.imag, closeTo(6.0, 1e-12));
        });
      },
    );

    test(
      'matmul 2D GEMM with 1D strided vector (step > 1) computes accurate dot products',
      () {
        NDArray.scope(() {
          final vecBuf = NDArray.fromList(
            Float64List.fromList([1.0, 999.0, 2.0, 888.0, 3.0, 777.0]),
            [6],
            DType.float64,
          );
          final stridedVec = vecBuf.slice([
            const Slice(start: 0, stop: 6, step: 2),
          ]);
          expect(stridedVec.shape, [3]);
          expect(stridedVec.strides, [2]);

          final mat = NDArray.fromList(
            Float64List.fromList([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]),
            [3, 3],
            DType.float64,
          );

          // 1D @ 2D: [1, 2, 3] @ diag(1, 2, 3) = [1, 4, 9]
          final res1 = matmul(stridedVec, mat);
          expect(res1.shape, [3]);
          expect(res1.getCell([0]), closeTo(1.0, 1e-12));
          expect(res1.getCell([1]), closeTo(4.0, 1e-12));
          expect(res1.getCell([2]), closeTo(9.0, 1e-12));

          // 2D @ 1D: diag(1, 2, 3) @ [1, 2, 3]^T = [1, 4, 9]
          final res2 = matmul(mat, stridedVec);
          expect(res2.shape, [3]);
          expect(res2.getCell([0]), closeTo(1.0, 1e-12));
          expect(res2.getCell([1]), closeTo(4.0, 1e-12));
          expect(res2.getCell([2]), closeTo(9.0, 1e-12));
        });
      },
    );

    test(
      'einsum and tensordot without explicit <R> type parameter do not throw AssertionError on R=Object',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [2, 3],
            DType.float64,
          );
          final b = NDArray.fromList(
            Float64List.fromList([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            [3, 2],
            DType.float64,
          );

          // Call without <Float64> type argument so R is inferred as Object
          final tdRes = tensordot(
            a.transpose(),
            b.transpose(),
            axes: ([0], [1]),
          );
          expect(tdRes.shape, [2, 2]);

          final einRes = einsum(EinsumSubscripts.parse('ij,jk->ik'), [a, b]);
          expect(einRes.shape, [2, 2]);
          expect(einRes.getCell([0, 0]), closeTo(4.0, 1e-12));
        });
      },
    );

    test(
      'einsum CBLAS fast path handles aliased and non-contiguous out buffer safely',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            Float64List.fromList([0.0, 1.0, 1.0, 0.0]),
            [2, 2],
            DType.float64,
          );

          // Aliased out == a
          einsum(EinsumSubscripts.parse('ij,jk->ik'), [a, b], out: a);
          expect(a.getCell([0, 0]), closeTo(2.0, 1e-12));
          expect(a.getCell([0, 1]), closeTo(1.0, 1e-12));
          expect(a.getCell([1, 0]), closeTo(4.0, 1e-12));
          expect(a.getCell([1, 1]), closeTo(3.0, 1e-12));

          // Non-contiguous out slice
          final bigOut = NDArray<Float64>.zeros([2, 4], DType.float64);
          final outSlice = bigOut.slice([
            const Slice.all(),
            const Slice(start: 0, stop: 4, step: 2),
          ]);
          expect(outSlice.isContiguous, isFalse);
          einsum(EinsumSubscripts.parse('ij,jk->ik'), [a, b], out: outSlice);
          expect(outSlice.getCell([0, 0]), closeTo(1.0, 1e-12));
          expect(outSlice.getCell([0, 1]), closeTo(2.0, 1e-12));
        });
      },
    );
  });
}
