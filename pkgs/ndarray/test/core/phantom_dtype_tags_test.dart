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

void main() {
  group('Reified Phantom DTypeTag Hierarchy', () {
    test('runtime type checks distinguish all 15 dtypes and DTypeTag', () {
      NDArray.scope(() {
        final f64 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final f32 = NDArray.fromList([1.0, 2.0], [2], DType.float32);
        final f16 = NDArray.fromList([1.0, 2.0], [2], DType.float16);
        final i32 = NDArray.fromList([1, 2], [2], DType.int32);
        final i64 = NDArray.fromList([1, 2], [2], DType.int64);
        final c128 = NDArray.fromList(
          [Complex(1.0, 0.0)],
          [1],
          DType.complex128,
        );
        final b = NDArray.fromList([true, false], [2], DType.boolean);

        final Object erasedF64 = f64;
        expect(erasedF64 is NDArray<Float64>, isTrue);
        expect(erasedF64 is NDArray<Float32>, isFalse);
        expect(erasedF64 is NDArray<Float16>, isFalse);
        expect(erasedF64 is NDArray<AnySpec>, isTrue);

        expect((f32 as Object) is NDArray<Float32>, isTrue);
        expect((f32 as Object) is NDArray<Float64>, isFalse);
        expect((f16 as Object) is NDArray<Float16>, isTrue);
        expect((i32 as Object) is NDArray<Int32>, isTrue);
        expect((i32 as Object) is NDArray<Int64>, isFalse);
        expect((i64 as Object) is NDArray<Int64>, isTrue);
        expect((i64 as Object) is NDArray<AnySpec>, isTrue);
        expect((c128 as Object) is NDArray<Complex128>, isTrue);
        expect((c128 as Object) is NDArray<Complex64>, isFalse);
        expect((c128 as Object) is NDArray<AnySpec>, isTrue);
        expect((b as Object) is NDArray<Boolean>, isTrue);
        expect((b as Object) is NDArray<AnySpec>, isTrue);

        // Unannotated binary and unary operations statically infer concrete NDArray<T>:
        final NDArray<Float64> sumF64 = add(f64, f64);
        final NDArray<Float32> prodF32 = multiply(f32, f32);
        final NDArray<Int32> diffI32 = subtract(i32, i32);
        final NDArray<Float64> divI32 = i32 / i32;
        final NDArray<Float32> divF32 = f32 / f32;
        expect(sumF64.toList(), [2.0, 4.0]);
        expect(prodF32.toList(), [1.0, 4.0]);
        expect(diffI32.toList(), [0, 0]);
        expect(divI32.toList(), [1.0, 1.0]);
        expect(divF32.toList(), [1.0, 1.0]);

        // Mixed-dtype / target-dtype operations via *As<Ta, Tb, R>(a, b, DType<R>)
        // infer all 3 type parameters from (a, b, dtype) without explicit <...>:
        final NDArray<Float32> mixedAddF32 = addAs(i32, f64, DType.float32);
        final NDArray<Float64> mixedSubF64 = subtractAs(
          f32,
          i32,
          DType.float64,
        );
        final NDArray<Int64> mixedMulI64 = multiplyAs(i32, i64, DType.int64);
        final NDArray<Float32> mixedDivF32 = divideAs(i32, i64, DType.float32);
        expect(mixedAddF32.dtype, DType.float32);
        expect(mixedAddF32.toList(), [2.0, 4.0]);
        expect(mixedSubF64.dtype, DType.float64);
        expect(mixedSubF64.toList(), [0.0, 0.0]);
        expect(mixedMulI64.dtype, DType.int64);
        expect(mixedMulI64.toList(), [1, 4]);
        expect(mixedDivF32.dtype, DType.float32);
        expect(mixedDivF32.toList(), [1.0, 1.0]);
      });
    });

    test(
      'Bucket A: float16 linalg ops preserve Float16 runtime and static tag',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [4.0, 1.0, 1.0, 3.0],
            [2, 2],
            DType.float16,
          );
          final b = NDArray.fromList([1.0, 2.0], [2], DType.float16);

          final invA = inv(a);
          expect(invA.dtype, DType.float16);
          expect((invA as Object) is NDArray<Float16>, isTrue);

          final detA = det(a);
          expect(detA.dtype, DType.float16);
          expect((detA as Object) is NDArray<Float16>, isTrue);

          final slog = slogdet(a);
          expect(slog.sign.dtype, DType.float16);
          expect(slog.logabsdet.dtype, DType.float16);

          final x = solve(a, b);
          expect(x.dtype, DType.float16);
          expect((x as Object) is NDArray<Float16>, isTrue);

          final p = pinv(a);
          expect(p.dtype, DType.float16);
          expect((p as Object) is NDArray<Float16>, isTrue);

          final l = cholesky(a);
          expect(l.dtype, DType.float16);
          expect((l as Object) is NDArray<Float16>, isTrue);

          final qrRes = qr(a);
          expect(qrRes.q.dtype, DType.float16);
          expect(qrRes.r.dtype, DType.float16);

          final svdRes = svd(a);
          expect(svdRes.u.dtype, DType.float16);
          expect(svdRes.s.dtype, DType.float16);
          expect(svdRes.vh.dtype, DType.float16);
        });
      },
    );

    test(
      'Bucket B: index-returning ops accept Int64 out and preserve runtime tag',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([30.0, 10.0, 20.0], [3], DType.float64);
          final out64 = NDArray.zeros([3], DType.int64);

          final res = argsortAs(a, DType.int64, out: out64);
          expect(res.dtype, DType.int64);
          expect((res as Object) is NDArray<Int64>, isTrue);
          expect(res.toList(), [1, 2, 0]);
        });
      },
    );

    test(
      'Bucket C: sin/fft/argsort/sum infer concrete return types and reject wrong casts at runtime',
      () {
        NDArray.scope(() {
          final f64 = NDArray.fromList([0.0, 1.0], [2], DType.float64);
          final NDArray<Float64> s = sin(f64);
          final NDArray<Complex128> f = fft(f64);
          final NDArray<Int32> idx = argsort(f64);
          final NDArray<Float64> total = sum(f64);
          expect(s.dtype, DType.float64);
          expect(f.dtype, DType.complex128);
          expect(idx.dtype, DType.int32);
          expect(total.dtype, DType.float64);

          final NDArray<AnySpec> erased = f64;
          expect(() {
            final _ = sin(erased) as NDArray<Float32>;
          }, throwsA(isA<TypeError>()));
        });
      },
    );
  });
}
