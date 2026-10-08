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
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('NDArray Arithmetic Extension Methods Tests', () {
    test(
      'Generic add, subtract, multiply, divide on contiguous double arrays',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [2.0, 2.0, 2.0, 2.0],
            [2, 2],
            DType.float64,
          );

          final addRes = add(a, b);
          expect(addRes.dtype, DType.float64);
          expect(addRes.toList(), [3.0, 4.0, 5.0, 6.0]);

          final subRes = subtract(a, b);
          expect(subRes.toList(), [-1.0, 0.0, 1.0, 2.0]);

          final mulRes = multiply(a, b);
          expect(mulRes.toList(), [2.0, 4.0, 6.0, 8.0]);

          final divRes = divide(a, b);
          expect(divRes.toList(), [0.5, 1.0, 1.5, 2.0]);
        });
      },
    );

    test('Contiguous Float32 SIMD lanes and C vector sweeps', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float32);
        final b = NDArray.fromList([1.0, 1.0, 1.0, 1.0], [4], DType.float32);

        final addRes = add(a, b);
        expect(addRes.dtype, DType.float32);
        expect(addRes.toList(), [2.0, 3.0, 4.0, 5.0]);
      });
    });

    test('Multidimensional strided broadcasting with FFI', () {
      NDArray.scope(() {
        // Shape [2, 2]
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        // Shape [2, 1] (broadcast column)
        final b = NDArray.fromList([10.0, 20.0], [2, 1], DType.float64);

        final res = add(a, b);
        expect(res.shape, [2, 2]);
        expect(res.toList(), [11.0, 12.0, 23.0, 24.0]);
      });
    });

    test('Complex128 and Float64 upcasting promotions', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [Complex(1.0, 2.0), Complex(3.0, 4.0)],
          [2],
          DType.complex128,
        );
        final b = NDArray.fromList([10.0, 20.0], [2], DType.float64);

        expect(() => add(a.asAnySpec, b.asAnySpec), throwsArgumentError);
        final bCast = castNDArray(b, DType.complex128);
        final res = add(a, bCast);
        expect(res.dtype, DType.complex128);
        final resList = res.toList();
        expect(resList[0].real, 11.0);
        expect(resList[0].imag, 2.0);
        expect(resList[1].real, 23.0);
        expect(resList[1].imag, 4.0);
      });
    });

    test('out parameter validation and success/error paths', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([2.0, 2.0], [2], DType.float64);
        final outValid = NDArray<Float64>.create([2], DType.float64);

        final res = add(a, b, out: outValid);
        expect(identical(res, outValid), true);
        expect(outValid.toList(), [3.0, 4.0]);

        final outInvalidShape = NDArray<Float64>.create([3], DType.float64);
        expect(() => add(a, b, out: outInvalidShape), throwsArgumentError);

        final outInvalidDType = NDArray<Int32>.create([2], DType.int32);
        expect(
          () => add(a.asAnySpec, b.asAnySpec, out: outInvalidDType.asAnySpec),
          throwsArgumentError,
        );
      });
    });
  });

  group('NDArray Operator Overloading Tests', () {
    test(
      'Basic Arithmetic Operators',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );

        final sum = a + b;
        expect(sum.toList(), [11.0, 22.0, 33.0, 44.0]);

        final diff = b - a;
        expect(diff.toList(), [9.0, 18.0, 27.0, 36.0]);

        final prod = a * b;
        expect(prod.toList(), [10.0, 40.0, 90.0, 160.0]);

        final quot = b / a;
        expect(quot.toList(), [10.0, 10.0, 10.0, 10.0]);
      }),
    );

    test(
      'Scalar Broadcasting Operators',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);

        final b = a + 10.0;
        expect(b.toList(), [11.0, 12.0]);

        final c = a * 2;
        expect(c.toList(), [2.0, 4.0]);

        final d = a - 1;
        expect(d.toList(), [0.0, 1.0]);

        final e = a / 0.5;
        expect(e.toList(), [2.0, 4.0]);
      }),
    );

    test(
      'Unary Negation',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, -2.0, 0.0], [3], DType.float64);
        final b = -a;
        expect(b.toList(), [-1.0, 2.0, 0.0]);
      }),
    );

    test(
      'Integer Division and Remainder',
      () => NDArray.scope(() {
        final a = NDArray.fromList([7, 10], [2], DType.int32);
        final b = NDArray.fromList([3, 4], [2], DType.int32);

        final floorDiv = a ~/ b;
        expect(floorDiv.toList(), [2, 2]);

        final rem = a % b;
        expect(rem.toList(), [1, 2]);

        final floorDivScalar = a ~/ 2;
        expect(floorDivScalar.toList(), [3, 5]);
      }),
    );

    test(
      'Bitwise Operators',
      () => NDArray.scope(() {
        final a = NDArray.fromList([0x0F, 0xF0], [2], DType.int32);
        final b = NDArray.fromList([0xFF, 0x00], [2], DType.int32);

        expect((a & b).toList(), [0x0F, 0x00]);
        expect((a | b).toList(), [0xFF, 0xF0]);
        expect((a ^ b).toList(), [0xF0, 0xF0]);
        expect((~a).toList().map((x) => x & 0xFF).toList(), [0xF0, 0x0F]);

        final c = NDArray.fromList([1, 2], [2], DType.int32);
        expect((c << 1).toList(), [2, 4]);
        expect((c >> 1).toList(), [0, 1]);
      }),
    );

    test(
      'Complex Arithmetic',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [Complex(1, 2), Complex(3, 4)],
          [2],
          DType.complex128,
        );
        final b = NDArray.fromList(
          [Complex(10, 20), Complex(30, 40)],
          [2],
          DType.complex128,
        );

        final sum = a + b;
        final sumList = sum.toList();
        expect(sumList[0].real, 11.0);
        expect(sumList[0].imag, 22.0);

        final neg = -a;
        final negList = neg.toList();
        expect(negList[0].real, -1.0);
        expect(negList[0].imag, -2.0);

        final scalarSum = a + Complex(10, 10);
        final scalarSumList = scalarSum.toList();
        expect(scalarSumList[0].real, 11.0);
        expect(scalarSumList[0].imag, 12.0);
      }),
    );

    test(
      'Mixed Type Arithmetic',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2], [2], DType.int32);
        final b = NDArray.fromList([0.5, 1.5], [2], DType.float64);

        final mixed = addAs(a, b, DType.float64);
        expect(mixed.dtype, DType.float64);
        expect(mixed.toList(), [1.5, 3.5]);

        final c = addAs(a, a, DType.float64);
        expect(c.dtype, DType.float64);
        expect(c.toList(), [2.0, 4.0]);

        expect(() => add(a.asAnySpec, b.asAnySpec), throwsArgumentError);
        final NDArray erased = a;
        expect(() => erased + b, throwsArgumentError);
      }),
    );

    test(
      'Arithmetic Error Handling (Division by Zero & Overflow)',
      () => NDArray.scope(() {
        // 1. True Division (operator /) by zero
        final doubleArr = NDArray.fromList(
          [5.0, -5.0, 0.0],
          [3],
          DType.float64,
        );
        final zeroDoubleArr = NDArray.zeros([3], DType.float64);
        final divDouble = doubleArr / zeroDoubleArr;
        final divDoubleList = divDouble.toList();
        expect(divDoubleList[0], double.infinity);
        expect(divDoubleList[1], double.negativeInfinity);
        expect(divDoubleList[2].isNaN, true);

        // True Division on integer arrays upcasts to float64 and handles division by zero identically
        final intArr = NDArray.fromList([5, -5, 0], [3], DType.int32);
        final zeroIntArr = NDArray.zeros([3], DType.int32);
        final divInt = intArr / zeroIntArr;
        expect(divInt.dtype, DType.float64);
        final divIntList = divInt.toList();
        expect(divIntList[0], double.infinity);
        expect(divIntList[1], double.negativeInfinity);
        expect(divIntList[2].isNaN, true);

        // 2. Floor Division (operator ~/) and Remainder (operator %) by zero on integer throws UnsupportedError
        expect(() => intArr ~/ zeroIntArr, throwsA(isA<UnsupportedError>()));
        expect(() => intArr % zeroIntArr, throwsA(isA<UnsupportedError>()));

        final minIntArr = NDArray.fromList([-2147483648], [1], DType.int32);
        final minusOneArr = NDArray.fromList([-1], [1], DType.int32);
        expect(
          (minIntArr ~/ minusOneArr).toList()[0],
          -2147483648,
        ); // or wraps/throws? Let's see!

        // Floor Division on floats with zero divisor follows IEEE-754 (±Inf for non-zero, NaN for 0/0)
        final divFloorDouble = doubleArr ~/ zeroDoubleArr;
        final divFloorDoubleList = divFloorDouble.toList();
        expect(divFloorDoubleList[0], double.infinity);
        expect(divFloorDoubleList[1], double.negativeInfinity);
        expect(divFloorDoubleList[2].isNaN, true);

        // 3. Integer Multiplication Overflow wraps around (two's complement)
        // Max 32-bit signed int is 2147483647
        final overflowArr32 = NDArray.fromList([2147483647], [1], DType.int32);
        final scaleArr32 = NDArray.fromList([2], [1], DType.int32);
        final prod32 = overflowArr32 * scaleArr32;
        expect(prod32.dtype, DType.int32);
        expect(
          prod32.toList()[0],
          -2,
        ); // 2147483647 * 2 = 4294967294 -> wraps to -2

        // Max 64-bit signed int is 9223372036854775807
        final overflowArr64 = NDArray.fromList(
          [9223372036854775807],
          [1],
          DType.int64,
        );
        final scaleArr64 = NDArray.fromList([2], [1], DType.int64);
        final prod64 = overflowArr64 * scaleArr64;
        expect(prod64.dtype, DType.int64);
        expect(
          prod64.toList()[0],
          -2,
        ); // 9223372036854775807 * 2 wraps to -2 in 64-bit two's complement

        // 4. Float Multiplication Overflow yields infinity
        final overflowArrFloat = NDArray.fromList([1e308], [1], DType.float64);
        final prodFloat = overflowArrFloat * 10.0;
        expect(prodFloat.toList()[0], double.infinity);
      }),
    );
  });

  group('Type Safety Fixes (Phase 3)', () {
    test('floorDivide and remainder with uint8 and int16', () {
      NDArray.scope(() {
        final a = NDArray.fromList([10, 20, 30, 40], [4], DType.uint8);
        final b = NDArray.fromList([3, 3, 3, 3], [4], DType.uint8);

        // floorDivide
        final resDiv = floorDivide(a, b);
        expect(resDiv.dtype, DType.uint8);
        expect(resDiv.toList(), [3, 6, 10, 13]);

        // remainder
        final resRem = remainder(a, b);
        expect(resRem.dtype, DType.uint8);
        expect(resRem.toList(), [1, 2, 0, 1]);

        // int16
        final a16 = NDArray.fromList([10, -20, 30, -40], [4], DType.int16);
        final b16 = NDArray.fromList([3, 3, 3, 3], [4], DType.int16);

        final resDiv16 = floorDivide(a16, b16);
        expect(resDiv16.dtype, DType.int16);
        expect(resDiv16.toList(), [3, -7, 10, -14]);

        final resRem16 = remainder(a16, b16);
        expect(resRem16.dtype, DType.int16);
        expect(resRem16.toList(), [1, 1, 0, 2]);
      });
    });

    test('sin and abs with uint8 and int16', () {
      NDArray.scope(() {
        final a = NDArray.fromList([0, 30, 90], [3], DType.uint8);

        final resSin = sin(a);
        expect(resSin.dtype, DType.float64);
        expect(resSin.toList()[0], closeTo(0.0, 1e-5));

        // abs
        final a16 = NDArray.fromList([-10, 0, 20], [3], DType.int16);
        final resAbs = abs(a16);
        expect(resAbs.dtype, DType.int16);
        expect(resAbs.toList(), [10, 0, 20]);
      });
    });

    test('negative with uint8 and int16 (no silent fallthrough)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([-10, 0, 20], [3], DType.int16);
        final resNeg = negative(a);
        expect(resNeg.dtype, DType.int16);
        expect(resNeg.toList(), [10, 0, -20]);

        final a8 = NDArray.fromList([10, 0, 20], [3], DType.uint8);
        final resNeg8 = negative(a8);
        expect(resNeg8.dtype, DType.uint8);
        expect(resNeg8.toList(), [246, 0, 236]);
      });
    });

    test('det type consistency for float32', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float32);
        final d = det(a);
        expect(d.dtype, DType.float32);
        expect(d.toList()[0], closeTo(-2.0, 1e-5));
      });
    });

    test('svd and qr throw ArgumentError for integer inputs', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        expect(() => svd(a), throwsArgumentError);
        expect(() => qr(a), throwsArgumentError);

        final a8 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.uint8);
        expect(() => svd(a8), throwsArgumentError);
        expect(() => qr(a8), throwsArgumentError);
      });
    });

    test('complex SVD and pinv', () {
      NDArray.scope(() {
        // complex128
        final a = NDArray.fromList(
          [Complex(1, 2), Complex(3, 4), Complex(5, 6), Complex(7, 8)],
          [2, 2],
          DType.complex128,
        );

        final svdRes = svd(a);
        expect(svdRes.u.dtype, DType.complex128);
        expect(svdRes.s.dtype, DType.float64);
        expect(svdRes.vh.dtype, DType.complex128);

        final sDiag = castNDArray(diag(svdRes.s), DType.complex128);
        final NDArray<AnySpec> uS = matmul(svdRes.u, sDiag);
        final reconstructed = matmul(uS, svdRes.vh);

        expect(allClose(reconstructed, a, rtol: 1e-5, atol: 1e-5), isTrue);

        // complex64
        final a64 = NDArray.fromList(
          [Complex(1, 2), Complex(3, 4), Complex(5, 6), Complex(7, 8)],
          [2, 2],
          DType.complex64,
        );

        final svdRes64 = svd(a64);
        expect(svdRes64.u.dtype, DType.complex64);
        expect(svdRes64.s.dtype, DType.float32);
        expect(svdRes64.vh.dtype, DType.complex64);

        final sDiag64 = castNDArray(diag(svdRes64.s), DType.complex64);
        final NDArray<AnySpec> uS64 = matmul(svdRes64.u, sDiag64);
        final reconstructed64 = matmul(uS64, svdRes64.vh);
        expect(
          allClose(reconstructed64, a64, rtol: 1e-3, atol: 1e-3),
          isTrue,
        ); // Lower tolerance for float32

        // complex pinv
        final aPinv = pinv(a);
        expect(aPinv.dtype, DType.complex128);

        final aPinvA = matmul(a, aPinv);
        final aPinvAA = matmul(aPinvA, a);
        expect(allClose(aPinvAA, a, rtol: 1e-5, atol: 1e-5), isTrue);

        final aPinv64 = pinv(a64);
        expect(aPinv64.dtype, DType.complex64);
      });
    });
  });

  group('Out Parameter Tests', () {
    test(
      'det with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final outBuffer = NDArray<Float64>.zeros([], DType.float64);
        final d = det(a, out: outBuffer);
        expect(identical(d, outBuffer), true);
        expect(outBuffer.toList()[0], closeTo(-2.0, 1e-9));
      }),
    );

    test(
      'min with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList([4.0, 2.0, 5.0, 1.0], [2, 2], DType.float64);
        final outBuffer = NDArray<Float64>.zeros([2], DType.float64);
        final res = min(a, axis: 1, out: outBuffer);
        expect(identical(res, outBuffer), true);
        expect(outBuffer.toList(), [2.0, 1.0]);
      }),
    );

    test(
      'max with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList([4.0, 2.0, 5.0, 1.0], [2, 2], DType.float64);
        final outBuffer = NDArray<Float64>.zeros([2], DType.float64);
        final res = max(a, axis: 1, out: outBuffer);
        expect(identical(res, outBuffer), true);
        expect(outBuffer.toList(), [4.0, 5.0]);
      }),
    );

    test(
      'nanmin with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [4.0, double.nan, 5.0, 1.0],
          [2, 2],
          DType.float64,
        );
        final outBuffer = NDArray<Float64>.zeros([2], DType.float64);
        final res = nanmin(a, axis: 1, out: outBuffer);
        expect(identical(res, outBuffer), true);
        expect(outBuffer.toList(), [4.0, 1.0]);
      }),
    );

    test(
      'nanmax with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [4.0, double.nan, 5.0, 1.0],
          [2, 2],
          DType.float64,
        );
        final outBuffer = NDArray<Float64>.zeros([2], DType.float64);
        final res = nanmax(a, axis: 1, out: outBuffer);
        expect(identical(res, outBuffer), true);
        expect(outBuffer.toList(), [4.0, 5.0]);
      }),
    );

    test(
      'fft with out parameter (no transpose)',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final outBuffer = NDArray.zeros([4], DType.complex128);
        final res = fft(a, out: outBuffer);
        expect(identical(res, outBuffer), true);
        // Expected FFT result: [10, -2+2i, -2, -2-2i]
        final outBufferList = outBuffer.toList();
        expect(outBufferList[0].real, closeTo(10.0, 1e-9));
        expect(outBufferList[0].imag, closeTo(0.0, 1e-9));
        expect(outBufferList[1].real, closeTo(-2.0, 1e-9));
        expect(outBufferList[1].imag, closeTo(2.0, 1e-9));
      }),
    );

    test(
      'fft with out parameter (with transpose)',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final outBuffer = NDArray.zeros([2, 2], DType.complex128);
        // FFT along axis 0
        final res = fft(a, axis: 0, out: outBuffer);
        expect(identical(res, outBuffer), true);
        // a is:
        // [[1, 2],
        //  [3, 4]]
        // Along axis 0:
        // Col 0: [1, 3] -> FFT: [4, -2]
        // Col 1: [2, 4] -> FFT: [6, -2]
        // Result:
        // [[4, 6],
        //  [-2, -2]]
        final outBufferList = outBuffer.toList();
        expect(outBufferList[0].real, closeTo(4.0, 1e-9));
        expect(outBufferList[1].real, closeTo(6.0, 1e-9));
        expect(outBufferList[2].real, closeTo(-2.0, 1e-9));
        expect(outBufferList[3].real, closeTo(-2.0, 1e-9));
      }),
    );

    test(
      'ifft with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [
            Complex(10.0, 0.0),
            Complex(-2.0, 2.0),
            Complex(-2.0, 0.0),
            Complex(-2.0, -2.0),
          ],
          [4],
          DType.complex128,
        );
        final outBuffer = NDArray.zeros([4], DType.complex128);
        final res = ifft(a, out: outBuffer);
        expect(identical(res, outBuffer), true);
        final outBufferList = outBuffer.toList();
        expect(outBufferList[0].real, closeTo(1.0, 1e-9));
        expect(outBufferList[1].real, closeTo(2.0, 1e-9));
        expect(outBufferList[2].real, closeTo(3.0, 1e-9));
        expect(outBufferList[3].real, closeTo(4.0, 1e-9));
      }),
    );

    test(
      'gradientArray with out parameter',
      () => NDArray.scope(() {
        final f = NDArray.fromList([1.0, 2.0, 4.0, 8.0], [2, 2], DType.float64);
        final out1 = NDArray<Float64>.zeros([2, 2], DType.float64);
        final out2 = NDArray<Float64>.zeros([2, 2], DType.float64);
        final res = gradientArray(f, out: [out1, out2]);
        expect(identical(res[0], out1), true);
        expect(identical(res[1], out2), true);
        // f is:
        // [[1, 2],
        //  [4, 8]]
        // Grad axis 0 (rows):
        // Col 0: [1, 4] -> [3, 3]
        // Col 1: [2, 8] -> [6, 6]
        // Grad 0:
        // [[3, 6],
        //  [3, 6]]
        // Grad axis 1 (cols):
        // Row 0: [1, 2] -> [1, 1]
        // Row 1: [4, 8] -> [4, 4]
        // Grad 1:
        // [[1, 1],
        //  [4, 4]]
        expect(out1.toList(), [3.0, 6.0, 3.0, 6.0]);
        expect(out2.toList(), [1.0, 1.0, 4.0, 4.0]);
      }),
    );

    test(
      'svd with out parameter (m >= n)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );
        final outU = NDArray<Float64>.zeros([3, 3], DType.float64);
        final outS = NDArray<Float64>.zeros([2], DType.float64);
        final outVh = NDArray<Float64>.zeros([2, 2], DType.float64);

        final res = svd(a, out: (u: outU, s: outS, vh: outVh));
        expect(identical(res.u, outU), true);
        expect(identical(res.s, outS), true);
        expect(identical(res.vh, outVh), true);

        // SVD values should be populated
        expect(outS.toList()[0], greaterThan(0.0));
        expect(outS.toList()[1], greaterThan(0.0));
      }),
    );

    test(
      'svd with out parameter (m < n)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final outU = NDArray<Float64>.zeros([2, 2], DType.float64);
        final outS = NDArray<Float64>.zeros([2], DType.float64);
        final outVh = NDArray<Float64>.zeros([3, 3], DType.float64);

        final res = svd(a, out: (u: outU, s: outS, vh: outVh));
        expect(identical(res.u, outU), true);
        expect(identical(res.s, outS), true);
        expect(identical(res.vh, outVh), true);

        expect(outS.toList()[0], greaterThan(0.0));
        expect(outS.toList()[1], greaterThan(0.0));
      }),
    );
  });

  group('Extensions', () {
    test(
      'Float64NDArrayOperations Contiguous & Strided ufuncs and Recycler',
      () {
        NDArray.scope(() {
          // Contiguous same-shape
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [2, 2],
            DType.float64,
          );

          final resAdd = add(a, b);
          expect(resAdd.toList(), [11.0, 22.0, 33.0, 44.0]);

          final resSub = subtract(a, b);
          expect(resSub.toList(), [-9.0, -18.0, -27.0, -36.0]);

          final resMul = multiply(a, b);
          expect(resMul.toList(), [10.0, 40.0, 90.0, 160.0]);

          final resDiv = divide(b, a);
          expect(resDiv.toList(), [10.0, 10.0, 10.0, 10.0]);

          // Strided non-contiguous views and broadcasting
          final aView = a.transpose();
          final bView = b.transpose();
          final resAddView = add(aView, bView);
          expect(resAddView.toList(), [11.0, 33.0, 22.0, 44.0]);

          // out Recycler parameter
          final intoBuf = NDArray<Float64>.create([2, 2], DType.float64);
          final resInto = add(a, b, out: intoBuf);
          expect(resInto, intoBuf);
          expect(resInto.toList(), [11.0, 22.0, 33.0, 44.0]);

          // out incompatible shape/dtype throws ArgumentError
          expect(
            () => add(a, b, out: NDArray<Float64>.create([3], DType.float64)),
            throwsArgumentError,
          );

          // matmul
          final resMatmul = matmul(a, b);
          expect(resMatmul.toList(), [70.0, 100.0, 150.0, 220.0]);

          // Mixed Float32 throws ArgumentError; explicit astype works
          final f32 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float32,
          );
          expect(() => add(a.asAnySpec, f32.asAnySpec), throwsArgumentError);
          expect(
            () => subtract(a.asAnySpec, f32.asAnySpec),
            throwsArgumentError,
          );
          expect(
            () => multiply(a.asAnySpec, f32.asAnySpec),
            throwsArgumentError,
          );
          expect(() => divide(a.asAnySpec, f32.asAnySpec), throwsArgumentError);
          final f32As64 = f32.astype(DType.float64);
          expect(add(a, f32As64).toList(), [2.0, 4.0, 6.0, 8.0]);
          expect(subtract(a, f32As64).toList(), [0.0, 0.0, 0.0, 0.0]);
          expect(multiply(a, f32As64).toList(), [1.0, 4.0, 9.0, 16.0]);
          expect(divide(a, f32As64).toList(), [1.0, 1.0, 1.0, 1.0]);

          // Mixed Int64 throws ArgumentError; explicit astype works
          final i64 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
          expect(() => add(a.asAnySpec, i64.asAnySpec), throwsArgumentError);
          expect(
            () => subtract(a.asAnySpec, i64.asAnySpec),
            throwsArgumentError,
          );
          expect(
            () => multiply(a.asAnySpec, i64.asAnySpec),
            throwsArgumentError,
          );
          expect(() => divide(a.asAnySpec, i64.asAnySpec), throwsArgumentError);
          final i64As64 = i64.astype(DType.float64);
          expect(add(a, i64As64).toList(), [2.0, 4.0, 6.0, 8.0]);
          expect(subtract(a, i64As64).toList(), [0.0, 0.0, 0.0, 0.0]);
          expect(multiply(a, i64As64).toList(), [1.0, 4.0, 9.0, 16.0]);
          expect(divide(a, i64As64).toList(), [1.0, 1.0, 1.0, 1.0]);

          // Mixed Scalar
          expect(
            add(a, NDArray.fromList([10.0], [1], DType.float64)).toList(),
            [11.0, 12.0, 13.0, 14.0],
          );
          expect(
            subtract(a, NDArray.fromList([1.0], [1], DType.float64)).toList(),
            [0.0, 1.0, 2.0, 3.0],
          );
          expect(
            multiply(a, NDArray.fromList([2.0], [1], DType.float64)).toList(),
            [2.0, 4.0, 6.0, 8.0],
          );
          expect(
            divide(a, NDArray.fromList([0.5], [1], DType.float64)).toList(),
            [2.0, 4.0, 6.0, 8.0],
          );
        });
      },
    );

    test(
      'Float32NDArrayOperations Contiguous & Strided ufuncs and Mixed Scalar',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float32,
          );
          final b = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [2, 2],
            DType.float32,
          );

          final resAdd = add(a, b);
          expect(resAdd.toList(), [11.0, 22.0, 33.0, 44.0]);

          final resSub = subtract(a, b);
          expect(resSub.toList(), [-9.0, -18.0, -27.0, -36.0]);

          final resMul = multiply(a, b);
          expect(resMul.toList(), [10.0, 40.0, 90.0, 160.0]);

          final resDiv = divide(b, a);
          expect(resDiv.toList(), [10.0, 10.0, 10.0, 10.0]);

          // Strided view
          final aView = a.transpose();
          final bView = b.transpose();
          expect(add(aView, bView).toList(), [11.0, 33.0, 22.0, 44.0]);

          // Recycler
          final intoBuf = NDArray<Float32>.create([2, 2], DType.float32);
          expect(add(a, b, out: intoBuf), intoBuf);

          // Incompatible recycler
          expect(
            () => add(a, b, out: NDArray<Float32>.create([3], DType.float32)),
            throwsArgumentError,
          );

          // Mixed Float64 throws ArgumentError; explicit astype works
          final f64 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          expect(() => add(a.asAnySpec, f64.asAnySpec), throwsArgumentError);
          expect(
            () => subtract(a.asAnySpec, f64.asAnySpec),
            throwsArgumentError,
          );
          expect(
            () => multiply(a.asAnySpec, f64.asAnySpec),
            throwsArgumentError,
          );
          expect(() => divide(a.asAnySpec, f64.asAnySpec), throwsArgumentError);
          final aAs64 = a.astype(DType.float64);
          expect(add(aAs64, f64).toList(), [2.0, 4.0, 6.0, 8.0]);
          expect(subtract(aAs64, f64).toList(), [0.0, 0.0, 0.0, 0.0]);
          expect(multiply(aAs64, f64).toList(), [1.0, 4.0, 9.0, 16.0]);
          expect(divide(aAs64, f64).toList(), [1.0, 1.0, 1.0, 1.0]);

          // Mixed Scalar
          expect(
            add(a, NDArray.fromList([10.0], [1], DType.float32)).toList(),
            [11.0, 12.0, 13.0, 14.0],
          );
          expect(
            subtract(a, NDArray.fromList([1.0], [1], DType.float32)).toList(),
            [0.0, 1.0, 2.0, 3.0],
          );
          expect(
            multiply(a, NDArray.fromList([2.0], [1], DType.float32)).toList(),
            [2.0, 4.0, 6.0, 8.0],
          );
          expect(
            divide(a, NDArray.fromList([0.5], [1], DType.float32)).toList(),
            [2.0, 4.0, 6.0, 8.0],
          );
        });
      },
    );

    test(
      'Int64NDArrayOperations Contiguous & Strided ufuncs and Mixed Double',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
          final b = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int64);

          final resAdd = add(a, b);
          expect(resAdd.toList(), [11, 22, 33, 44]);

          final resSub = subtract(a, b);
          expect(resSub.toList(), [-9, -18, -27, -36]);

          final resMul = multiply(a, b);
          expect(resMul.toList(), [10, 40, 90, 160]);

          final resDiv = divide(b, a);
          expect(resDiv.toList(), [10.0, 10.0, 10.0, 10.0]);

          // Strided view
          final aView = a.transpose();
          final bView = b.transpose();
          expect(add(aView, bView).toList(), [11, 33, 22, 44]);

          // Recycler
          final intoBuf = NDArray<Int64>.create([2, 2], DType.int64);
          expect(add(a, b, out: intoBuf), intoBuf);

          final intoDoubleBuf = NDArray<Float64>.create([2, 2], DType.float64);
          expect(divide(b, a, out: intoDoubleBuf), intoDoubleBuf);

          // Incompatible recycler
          expect(
            () => add(a, b, out: NDArray<Int64>.create([3], DType.int64)),
            throwsArgumentError,
          );
          expect(
            () =>
                divide(b, a, out: NDArray<Float64>.create([3], DType.float64)),
            throwsArgumentError,
          );

          // Mixed Double throws ArgumentError; explicit astype works
          final f64 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          expect(() => add(a.asAnySpec, f64.asAnySpec), throwsArgumentError);
          expect(
            () => subtract(a.asAnySpec, f64.asAnySpec),
            throwsArgumentError,
          );
          expect(
            () => multiply(a.asAnySpec, f64.asAnySpec),
            throwsArgumentError,
          );
          final aAs64 = a.astype(DType.float64);
          expect(add(aAs64, f64).toList(), [2.0, 4.0, 6.0, 8.0]);
          expect(subtract(aAs64, f64).toList(), [0.0, 0.0, 0.0, 0.0]);
          expect(multiply(aAs64, f64).toList(), [1.0, 4.0, 9.0, 16.0]);

          // Mixed Scalar
          expect(add(a, NDArray.fromList([10], [1], DType.int64)).toList(), [
            11,
            12,
            13,
            14,
          ]);
          expect(
            subtract(a, NDArray.fromList([1], [1], DType.int64)).toList(),
            [0, 1, 2, 3],
          );
          expect(
            multiply(a, NDArray.fromList([2], [1], DType.int64)).toList(),
            [2, 4, 6, 8],
          );
        });
      },
    );

    test(
      'Int32NDArrayOperations Contiguous & Strided ufuncs and Mixed Int64',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final b = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int32);

          final resAdd = add(a, b);
          expect(resAdd.toList(), [11, 22, 33, 44]);

          final resSub = subtract(a, b);
          expect(resSub.toList(), [-9, -18, -27, -36]);

          final resMul = multiply(a, b);
          expect(resMul.toList(), [10, 40, 90, 160]);

          final resDiv = divide(b, a);
          expect(resDiv.toList(), [10.0, 10.0, 10.0, 10.0]);

          // Strided view
          final aView = a.transpose();
          final bView = b.transpose();
          expect(add(aView, bView).toList(), [11, 33, 22, 44]);

          // Recycler
          final intoBuf = NDArray<Int32>.create([2, 2], DType.int32);
          expect(add(a, b, out: intoBuf), intoBuf);

          final intoDoubleBuf = NDArray<Float64>.create([2, 2], DType.float64);
          expect(divide(b, a, out: intoDoubleBuf), intoDoubleBuf);

          // Incompatible recycler
          expect(
            () => add(a, b, out: NDArray<Int32>.create([3], DType.int32)),
            throwsArgumentError,
          );
          expect(
            () =>
                divide(b, a, out: NDArray<Float64>.create([3], DType.float64)),
            throwsArgumentError,
          );

          // Mixed Int64 throws ArgumentError; explicit astype works
          final i64 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
          expect(() => add(a.asAnySpec, i64.asAnySpec), throwsArgumentError);
          expect(
            () => subtract(a.asAnySpec, i64.asAnySpec),
            throwsArgumentError,
          );
          expect(
            () => multiply(a.asAnySpec, i64.asAnySpec),
            throwsArgumentError,
          );
          final aAsI64 = a.astype(DType.int64);
          expect(add(aAsI64, i64).toList(), [2, 4, 6, 8]);
          expect(subtract(aAsI64, i64).toList(), [0, 0, 0, 0]);
          expect(multiply(aAsI64, i64).toList(), [1, 4, 9, 16]);

          // Mixed Scalar
          expect(add(a, NDArray.fromList([10], [1], DType.int32)).toList(), [
            11,
            12,
            13,
            14,
          ]);
          expect(
            subtract(a, NDArray.fromList([1], [1], DType.int32)).toList(),
            [0, 1, 2, 3],
          );
          expect(
            multiply(a, NDArray.fromList([2], [1], DType.int32)).toList(),
            [2, 4, 6, 8],
          );
        });
      },
    );

    test(
      'ComplexNDArrayOperations Contiguous & Strided ufuncs and Mixed Scalar',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [Complex(1, 2), Complex(3, 4)],
            [2],
            DType.complex128,
          );
          final b = NDArray.fromList(
            [Complex(10, 20), Complex(30, 40)],
            [2],
            DType.complex128,
          );

          final resAdd = add(a, b);
          expect(resAdd.toList()[0].real, 11.0);
          expect(resAdd.toList()[0].imag, 22.0);
          expect(resAdd.toList()[1].real, 33.0);
          expect(resAdd.toList()[1].imag, 44.0);

          final resSub = subtract(a, b);
          expect(resSub.toList()[0].real, -9.0);
          expect(resSub.toList()[0].imag, -18.0);

          final resMul = multiply(a, b);
          // (1+2i)*(10+20i) = 10 + 20i + 20i - 40 = -30 + 40i
          expect(resMul.toList()[0].real, -30.0);
          expect(resMul.toList()[0].imag, 40.0);

          final resDiv = divide(b, a);
          // (10+20i)/(1+2i) = 10*(1+2i)/(1+2i) = 10
          expect(resDiv.toList()[0].real, 10.0);
          expect(resDiv.toList()[0].imag, 0.0);

          // Strided view and recycler
          final aView = a.slice([const Slice(start: 0, stop: 2, step: 1)]);
          final bView = b.slice([const Slice(start: 0, stop: 2, step: 1)]);
          final intoBuf = NDArray.create([2], DType.complex128);
          expect(add(aView, bView, out: intoBuf), intoBuf);

          // Incompatible recycler
          expect(
            () => add(a, b, out: NDArray.create([3], DType.complex128)),
            throwsArgumentError,
          );

          // Mixed Float64 throws ArgumentError; explicit astype works
          final f64 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          expect(() => add(a.asAnySpec, f64.asAnySpec), throwsArgumentError);
          expect(
            () => subtract(a.asAnySpec, f64.asAnySpec),
            throwsArgumentError,
          );
          expect(
            () => multiply(a.asAnySpec, f64.asAnySpec),
            throwsArgumentError,
          );
          expect(() => divide(a.asAnySpec, f64.asAnySpec), throwsArgumentError);
          final f64AsC128 = f64.astype(DType.complex128);
          final resAddF64 = add(a, f64AsC128);
          expect(resAddF64.toList()[0].real, 2.0);
          expect(resAddF64.toList()[0].imag, 2.0);

          final resSubF64 = subtract(a, f64AsC128);
          expect(resSubF64.toList()[0].real, 0.0);
          expect(resSubF64.toList()[0].imag, 2.0);

          final resMulF64 = multiply(a, f64AsC128);
          expect(resMulF64.toList()[0].real, 1.0);
          expect(resMulF64.toList()[0].imag, 2.0);

          final resDivF64 = divide(a, f64AsC128);
          expect(resDivF64.toList()[0].real, 1.0);
          expect(resDivF64.toList()[0].imag, 2.0);

          // Mixed Int64 throws ArgumentError; explicit astype works
          final i64 = NDArray.fromList([1, 2], [2], DType.int64);
          expect(() => add(a.asAnySpec, i64.asAnySpec), throwsArgumentError);
          expect(
            () => subtract(a.asAnySpec, i64.asAnySpec),
            throwsArgumentError,
          );
          expect(
            () => multiply(a.asAnySpec, i64.asAnySpec),
            throwsArgumentError,
          );
          expect(() => divide(a.asAnySpec, i64.asAnySpec), throwsArgumentError);
          final i64AsC128 = i64.astype(DType.complex128);
          final resAddI64 = add(a, i64AsC128);
          expect(resAddI64.toList()[0].real, 2.0);
          expect(resAddI64.toList()[0].imag, 2.0);

          final resSubI64 = subtract(a, i64AsC128);
          expect(resSubI64.toList()[0].real, 0.0);

          final resMulI64 = multiply(a, i64AsC128);
          expect(resMulI64.toList()[0].real, 1.0);

          final resDivI64 = divide(a, i64AsC128);
          expect(resDivI64.toList()[0].real, 1.0);

          // Mixed Scalar
          expect(
            add(
              a,
              NDArray.fromList([Complex(10, 10)], [1], DType.complex128),
            ).toList()[0].real,
            11.0,
          );
          expect(
            subtract(
              a,
              NDArray.fromList([Complex(1, 1)], [1], DType.complex128),
            ).toList()[0].real,
            0.0,
          );
          expect(
            multiply(
              a,
              NDArray.fromList([Complex(2.0, 0.0)], [1], DType.complex128),
            ).toList()[0].real,
            2.0,
          );
          expect(
            divide(
              a,
              NDArray.fromList([Complex(0.5, 0.0)], [1], DType.complex128),
            ).toList()[0].real,
            2.0,
          );
        });
      },
    );
  });

  group("Math Strided Out Bug", () {
    test("add with strided out and offsets", () {
      final baseA = NDArray.fromList(
        Float64List.fromList([0.0, 1.0, 2.0, 0.0]),
        [4],
        DType.float64,
      );
      final baseB = NDArray.fromList(
        Float64List.fromList([0.0, 3.0, 4.0, 0.0]),
        [4],
        DType.float64,
      );

      final a = baseA.slice([Slice(start: 1, stop: 3)]);
      final b = baseB.slice([Slice(start: 1, stop: 3)]);

      final baseOut = NDArray.zeros([4], DType.float64);
      final out = baseOut.slice([Slice(start: 0, stop: 4, step: 2)]);

      add(a, b, out: out);

      expect(baseOut.toList(), [4.0, 0.0, 6.0, 0.0]);
    });

    test("abs with strided out and offsets (real)", () {
      final baseA = NDArray.fromList(
        Float64List.fromList([0.0, -1.5, -2.0, 0.0]),
        [4],
        DType.float64,
      );
      final a = baseA.slice([Slice(start: 1, stop: 3)]);

      final baseOut = NDArray.zeros([4], DType.float64);
      final out = baseOut.slice([Slice(start: 0, stop: 4, step: 2)]);

      abs(a, out: out);

      expect(baseOut.toList(), [1.5, 0.0, 2.0, 0.0]);
    });

    test("abs with strided out (complex -> real)", () {
      final baseA = NDArray.fromList(
        [Complex(0, 0), Complex(-3, 4), Complex(5, -12), Complex(0, 0)],
        [4],
        DType.complex128,
      );
      final a = baseA.slice([Slice(start: 1, stop: 3)]);

      final baseOut = NDArray.zeros([4], DType.float64);
      final out = baseOut.slice([Slice(start: 0, stop: 4, step: 2)]);

      abs(a, out: out);

      expect(baseOut.toList(), [5.0, 0.0, 13.0, 0.0]);
    });

    test("conj with strided out (real)", () {
      final baseA = NDArray.fromList(
        Float64List.fromList([0.0, 1.0, 2.0, 0.0]),
        [4],
        DType.float64,
      );
      final a = baseA.slice([Slice(start: 1, stop: 3)]);

      final baseOut = NDArray.zeros([4], DType.float64);
      final out = baseOut.slice([Slice(start: 0, stop: 4, step: 2)]);

      conj(a, out: out);

      expect(baseOut.toList(), [1.0, 0.0, 2.0, 0.0]);
    });

    test("conj with strided out (complex)", () {
      final baseA = NDArray.fromList(
        [Complex(0, 0), Complex(1, -2), Complex(3, -4), Complex(0, 0)],
        [4],
        DType.complex128,
      );
      final a = baseA.slice([Slice(start: 1, stop: 3)]);

      final baseOut = NDArray.fromList(
        [Complex(0, 0), Complex(0, 0), Complex(0, 0), Complex(0, 0)],
        [4],
        DType.complex128,
      );
      final out = baseOut.slice([Slice(start: 0, stop: 4, step: 2)]);

      conj(a, out: out);

      expect(baseOut.toList(), [
        Complex(1, 2),
        Complex(0, 0),
        Complex(3, 4),
        Complex(0, 0),
      ]);
    });

    test("hypot with strided out and offsets", () {
      final baseA = NDArray.fromList(
        Float64List.fromList([0.0, 3.0, 5.0, 0.0]),
        [4],
        DType.float64,
      );
      final baseB = NDArray.fromList(
        Float64List.fromList([0.0, 4.0, 12.0, 0.0]),
        [4],
        DType.float64,
      );
      final a = baseA.slice([Slice(start: 1, stop: 3)]);
      final b = baseB.slice([Slice(start: 1, stop: 3)]);

      final baseOut = NDArray.zeros([4], DType.float64);
      final out = baseOut.slice([Slice(start: 0, stop: 4, step: 2)]);

      hypot(a, b, out: out);

      expect(baseOut.toList(), [5.0, 0.0, 13.0, 0.0]);
    });

    test("Unary op (sin) with strided out and non-zero offset input", () {
      final aBacking = Int32List.fromList([99, 0, 99, 1]);
      final aParent = NDArray.fromList(aBacking, [4], DType.int32);
      final a = aParent.slice([
        Slice(start: 1, stop: 4, step: 2),
      ]); // shape [2], strides [2], offset 1

      final outBacking = Float64List.fromList([99.0, 99.0, 99.0, 99.0, 99.0]);
      final outParent = NDArray.fromList(outBacking, [5], DType.float64);
      final out = outParent.slice([
        Slice(start: 1, stop: 5, step: 2),
      ]); // shape [2], strides [2], offset 1

      sin(a, out: out);

      final outList = outParent.toList();
      expect(outList[0], 99.0);
      expect(outList[1], closeTo(0.0, 1e-7));
      expect(outList[2], 99.0);
      expect(outList[3], closeTo(0.84147098, 1e-7));
      expect(outList[4], 99.0);
    });

    test("Binary op (add) with strided out and non-zero offset inputs", () {
      final aBacking = Float64List.fromList([99.0, 1.0, 99.0, 2.0]);
      final a = NDArray.fromList(
        aBacking,
        [4],
        DType.float64,
      ).slice([Slice(start: 1, stop: 4, step: 2)]); // [1.0, 2.0]

      final bBacking = Float64List.fromList([99.0, 10.0, 99.0, 20.0]);
      final b = NDArray.fromList(
        bBacking,
        [4],
        DType.float64,
      ).slice([Slice(start: 1, stop: 4, step: 2)]); // [10.0, 20.0]

      final outBacking = Float64List.fromList([99.0, 99.0, 99.0, 99.0, 99.0]);
      final outParent = NDArray.fromList(outBacking, [5], DType.float64);
      final out = outParent.slice([Slice(start: 1, stop: 5, step: 2)]);

      add(a, b, out: out);

      expect(outParent.toList(), [99.0, 11.0, 99.0, 22.0, 99.0]);
    });

    test("Real-number conj with strided out (Worker 2 version)", () {
      final aBacking = Float64List.fromList([1.0, 2.0]);
      final a = NDArray.fromList(aBacking, [2], DType.float64);

      final outBacking = Float64List.fromList([99.0, 99.0, 99.0, 99.0, 99.0]);
      final outParent = NDArray.fromList(outBacking, [5], DType.float64);
      final out = outParent.slice([Slice(start: 1, stop: 5, step: 2)]);

      conj(a, out: out);

      expect(outParent.toList(), [99.0, 1.0, 99.0, 2.0, 99.0]);
    });

    test("Optimized abs with contiguous input/output", () {
      final a = NDArray.fromList(Float64List.fromList([-1.0, -2.0]), [
        2,
      ], DType.float64);
      final out = NDArray.create([2], DType.float64);
      abs(a, out: out);
      expect(out.toList(), [1.0, 2.0]);
    });

    test(
      "Optimized abs with contiguous sliced view (length != data.length)",
      () {
        final base = NDArray.fromList(
          Float64List.fromList([-9.0, -1.0, -2.0, -9.0]),
          [4],
          DType.float64,
        );
        final a = base.slice([
          Slice(start: 1, stop: 3),
        ]); // shape [2], contiguous, offset 1, data.length is 4
        final outBase = NDArray.zeros([4], DType.float64);
        final out = outBase.slice([
          Slice(start: 1, stop: 3),
        ]); // shape [2], contiguous

        abs(a, out: out);

        expect(outBase.toList(), [0.0, 1.0, 2.0, 0.0]);
      },
    );

    test("sin with contiguous input and strided out", () {
      final a = NDArray.fromList(
        Float64List.fromList([0.0, 1.5707963267948966]),
        [2],
        DType.float64,
      ); // contiguous

      final baseOut = NDArray.fromList(
        Float64List.fromList([99.0, 99.0, 99.0, 99.0, 99.0]),
        [5],
        DType.float64,
      );
      final out = baseOut.slice([
        Slice(start: 1, stop: 5, step: 2),
      ]); // strided, shape [2]

      sin(a, out: out);

      final outList = baseOut.toList();
      expect(outList[0], 99.0);
      expect(outList[1], closeTo(0.0, 1e-7)); // sin(0)
      expect(outList[2], 99.0);
      expect(outList[3], closeTo(1.0, 1e-7)); // sin(pi/2)
      expect(outList[4], 99.0);
    });

    test("comparison (greater) with broadcasting", () {
      final a = NDArray.fromList([1.0, 2.0], [2], DType.float64); // shape [2]
      final b = NDArray.fromList(
        [0.0, 3.0, 4.0, 1.0],
        [2, 2],
        DType.float64,
      ); // shape [2, 2]

      final res = greater(a, b);
      expect(res.shape, [2, 2]);
      expect(res.toList(), [true, false, false, true]);
    });
  });

  group('Math Out Param', () {
    test('hypot with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([3.0, 5.0]), [
        2,
      ], DType.float64);
      final b = NDArray.fromList(Float64List.fromList([4.0, 12.0]), [
        2,
      ], DType.float64);
      final out = NDArray.create([2], DType.float64);

      final res = hypot(a, b, out: out);
      expect(identical(res, out), isTrue);
      final outList = out.toList();
      expect(outList[0], closeTo(5.0, 1e-10));
      expect(outList[1], closeTo(13.0, 1e-10));

      // Incompatible shape/dtype validation
      final badOut = NDArray.create([3], DType.float64);
      expect(() => hypot(a, b, out: badOut), throwsArgumentError);
    });

    test('power with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([2.0, 3.0]), [
        2,
      ], DType.float64);
      final b = NDArray.fromList(Float64List.fromList([3.0, 2.0]), [
        2,
      ], DType.float64);
      final out = NDArray.create([2], DType.float64);

      final res = power(a, b, out: out);
      expect(identical(res, out), isTrue);
      expect(out.toList(), [8.0, 9.0]);

      final badOut = NDArray.create([2], DType.float32);
      expect(
        () => power(a.asAnySpec, b.asAnySpec, out: badOut.asAnySpec),
        throwsArgumentError,
      );

      final diffDType = NDArray.fromList([2, 3], [2], DType.int64);
      expect(
        () => power(a.asAnySpec, diffDType.asAnySpec),
        throwsArgumentError,
      );
    });

    test('integer power contiguous and strided', () {
      NDArray.scope(() {
        // Contiguous
        final a64 = NDArray.fromList([2, 3, 4], [3], DType.int64);
        final b64 = NDArray.fromList([3, 2, 0], [3], DType.int64);
        final r64 = power(a64, b64);
        expect(r64.dtype, DType.int64);
        expect(r64.toList(), [8, 9, 1]);

        final a32 = NDArray.fromList([2, 3, 4], [3], DType.int32);
        final b32 = NDArray.fromList([3, 2, 0], [3], DType.int32);
        final r32 = power(a32, b32);
        expect(r32.dtype, DType.int32);
        expect(r32.toList(), [8, 9, 1]);

        final a16 = NDArray.fromList([2, 3, 4], [3], DType.int16);
        final b16 = NDArray.fromList([3, 2, 0], [3], DType.int16);
        final r16 = power(a16, b16);
        expect(r16.dtype, DType.int16);
        expect(r16.toList(), [8, 9, 1]);

        final u8 = NDArray.fromList([2, 3, 4], [3], DType.uint8);
        final bu8 = NDArray.fromList([3, 2, 0], [3], DType.uint8);
        final ru8 = power(u8, bu8);
        expect(ru8.dtype, DType.uint8);
        expect(ru8.toList(), [8, 9, 1]);

        // Strided
        final a64s = NDArray.fromList(
          [2, 99, 3, 99, 4],
          [5],
          DType.int64,
        ).slice([const Slice(start: 0, stop: 5, step: 2)]);
        final b64s = NDArray.fromList(
          [3, 99, 2, 99, 0],
          [5],
          DType.int64,
        ).slice([const Slice(start: 0, stop: 5, step: 2)]);
        final r64s = power(a64s, b64s);
        expect(r64s.dtype, DType.int64);
        expect(r64s.toList(), [8, 9, 1]);

        // Strided Broadcasting
        // a: [2, 3] (2,)
        // b: [[3], [2]] (2, 1)
        // res: (2, 2)
        // [[2^3, 3^3],
        //  [2^2, 3^2]] = [[8, 27], [4, 9]]
        final ab = NDArray.fromList([2, 3], [2], DType.int32);
        final bb = NDArray.fromList([3, 2], [2, 1], DType.int32);
        final rb = power(ab, bb);
        expect(rb.shape, [2, 2]);
        expect(rb.toList(), [8, 27, 4, 9]);

        // Edge cases for exponentiation
        // 0^0 -> 1
        // negative base, positive exponent
        final negativeBase = NDArray.fromList([-2], [1], DType.int32);
        final oddExp = NDArray.fromList([3], [1], DType.int32);
        final evenExp = NDArray.fromList([4], [1], DType.int32);
        expect(power(negativeBase, oddExp).toList()[0], -8);
        expect(power(negativeBase, evenExp).toList()[0], 16);

        final zeroBase = NDArray.fromList([0], [1], DType.int32);
        final positiveExp = NDArray.fromList([3], [1], DType.int32);
        final zeroExp = NDArray.fromList([0], [1], DType.int32);
        expect(power(zeroBase, positiveExp).toList()[0], 0);
        expect(power(zeroBase, zeroExp).toList()[0], 1);

        // Negative exponents should throw ArgumentError
        final negExp = NDArray.fromList([-3], [1], DType.int32);
        expect(() => power(a32, negExp), throwsArgumentError);

        final negExpStrided = NDArray.fromList([1, -1], [2], DType.int32);
        expect(() => power(a32, negExpStrided), throwsArgumentError);

        // Disposed state errors
        a32.dispose();
        expect(() => power(a32, b32), throwsStateError);
      });
    });

    test('negative with out parameter', () {
      final a = NDArray.fromList(Int32List.fromList([2, -3]), [2], DType.int32);
      final out = NDArray.create([2], DType.int32);

      final res = negative(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.toList(), [-2, 3]);

      final badOut = NDArray.create([3], DType.int32);
      expect(() => negative(a, out: badOut), throwsArgumentError);
    });

    test('floorDivide with out parameter', () {
      final a = NDArray.fromList(Int32List.fromList([10, 25]), [
        2,
      ], DType.int32);
      final b = NDArray.fromList(Int32List.fromList([3, 4]), [2], DType.int32);
      final out = NDArray.create([2], DType.int32);

      final res = floorDivide(a, b, out: out);
      expect(identical(res, out), isTrue);
      expect(out.toList(), [3, 6]);

      final badOut = NDArray.create([2], DType.float64);
      expect(
        () => floorDivide(a.asAnySpec, b.asAnySpec, out: badOut.asAnySpec),
        throwsArgumentError,
      );
    });

    test('remainder and mod with out parameter', () {
      final a = NDArray.fromList(Int32List.fromList([10, 25]), [
        2,
      ], DType.int32);
      final b = NDArray.fromList(Int32List.fromList([3, 4]), [2], DType.int32);
      final out = NDArray.create([2], DType.int32);

      final res = remainder(a, b, out: out);
      expect(identical(res, out), isTrue);
      expect(out.toList(), [1, 1]);

      final outMod = NDArray.create([2], DType.int32);
      final resMod = mod(a, b, out: outMod);
      expect(identical(resMod, outMod), isTrue);
      expect(outMod.toList(), [1, 1]);
    });

    test('abs with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([-1.5, 2.0]), [
        2,
      ], DType.float64);
      final out = NDArray.create([2], DType.float64);

      final res = abs(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.toList(), [1.5, 2.0]);
    });

    test('sign with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([-1.5, 0.0, 2.5]), [
        3,
      ], DType.float64);
      final out = NDArray.create([3], DType.float64);

      final res = sign(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.toList(), [-1.0, 0.0, 1.0]);
    });

    test('ceil with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([1.2, -1.7]), [
        2,
      ], DType.float64);
      final out = NDArray.create([2], DType.float64);

      final res = ceil(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.toList(), [2.0, -1.0]);
    });

    test('floor with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([1.2, -1.7]), [
        2,
      ], DType.float64);
      final out = NDArray.create([2], DType.float64);

      final res = floor(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.toList(), [1.0, -2.0]);
    });

    test('round with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([1.2, -1.7]), [
        2,
      ], DType.float64);
      final out = NDArray.create([2], DType.float64);

      final res = round(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.toList(), [1.0, -2.0]);
    });

    test('isnan with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, double.nan]), [
        2,
      ], DType.float64);
      final out = NDArray<Boolean>.create([2], DType.boolean);

      final res = isnan(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.toList(), [false, true]);
    });

    test('isinf with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, double.infinity]), [
        2,
      ], DType.float64);
      final out = NDArray<Boolean>.create([2], DType.boolean);

      final res = isinf(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.toList(), [false, true]);
    });

    test('isfinite with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, double.infinity]), [
        2,
      ], DType.float64);
      final out = NDArray<Boolean>.create([2], DType.boolean);

      final res = isfinite(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.toList(), [true, false]);
    });

    test('copysign with out parameter', () {
      final x1 = NDArray.fromList(Float64List.fromList([2.0, -3.0]), [
        2,
      ], DType.float64);
      final x2 = NDArray.fromList(Float64List.fromList([-1.0, 1.0]), [
        2,
      ], DType.float64);
      final out = NDArray.create([2], DType.float64);

      final res = copysign(x1, x2, out: out);
      expect(identical(res, out), isTrue);
      expect(out.toList(), [-2.0, 3.0]);
    });
    test(
      'add() contiguous float32 SIMD fast paths and remainders coverage',
      () {
        final a8 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [8],
          DType.float32,
        );
        final b8 = NDArray.fromList(
          [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
          [8],
          DType.float32,
        );

        final res8 = add(a8, b8);
        expect(res8.dtype, DType.float32);
        expect(res8.toList(), [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]);

        final a6 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [6],
          DType.float32,
        );
        final b6 = NDArray.fromList(
          [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
          [6],
          DType.float32,
        );

        final res6 = add(a6, b6);
        expect(res6.dtype, DType.float32);
        expect(res6.toList(), [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
      },
    );

    test(
      'subtract() contiguous float64 and float32 FFI fast paths coverage',
      () {
        final a64 = NDArray.fromList([20.0, 30.0], [2], DType.float64);
        final b64 = NDArray.fromList([5.0, 10.0], [2], DType.float64);

        final res64 = subtract(a64, b64);
        expect(res64.dtype, DType.float64);
        expect(res64.toList(), [15.0, 20.0]);

        final a32 = NDArray.fromList([20.0, 30.0], [2], DType.float32);
        final b32 = NDArray.fromList([5.0, 10.0], [2], DType.float32);

        final res32 = subtract(a32, b32);
        expect(res32.dtype, DType.float32);
        expect(res32.toList(), [15.0, 20.0]);
      },
    );

    test(
      'add() cross-type complex/int and int/complex additions coverage',
      () => NDArray.scope(() {
        final c = NDArray.fromList([Complex(1.0, 1.0)], [1], DType.complex128);
        final i = NDArray.fromList([2], [1], DType.int64);
        final d = NDArray.fromList([3.0], [1], DType.float64);

        // Mixed dtypes are rejected on add(); explicit astype works
        expect(() => add(c.asAnySpec, i.asAnySpec), throwsArgumentError);
        final res1 = add(c, i.astype(DType.complex128));
        expect(res1.dtype, DType.complex128);
        expect(res1.getCell([0]).real, 3.0);
        expect(res1.getCell([0]).imag, 1.0);

        expect(() => add(d.asAnySpec, c.asAnySpec), throwsArgumentError);
        final res2 = add(d.astype(DType.complex128), c);
        expect(res2.dtype, DType.complex128);
        expect(res2.getCell([0]).real, 4.0);
        expect(res2.getCell([0]).imag, 1.0);

        expect(() => add(i.asAnySpec, c.asAnySpec), throwsArgumentError);
        final res3 = add(i.astype(DType.complex128), c);
        expect(res3.dtype, DType.complex128);
        expect(res3.getCell([0]).real, 3.0);
      }),
    );

    test(
      'Cross-type arithmetic coverage for subtract, multiply, and divide',
      () {
        final c = NDArray.fromList(
          [Complex(10.0, 10.0)],
          [1],
          DType.complex128,
        );
        final i = NDArray.fromList([2], [1], DType.int64);
        final d = NDArray.fromList([4.0], [1], DType.float64);
        final iC = i.astype(DType.complex128);
        final dC = d.astype(DType.complex128);
        final iD = i.astype(DType.float64);

        // --- subtract() Gaps ---
        // 1. Complex - int
        expect(() => subtract(c.asAnySpec, i.asAnySpec), throwsArgumentError);
        final s1 = subtract(c, iC);
        expect(s1.dtype, DType.complex128);
        expect(s1.getCell([0]), Complex(8.0, 10.0));

        // 2. int - Complex
        expect(() => subtract(i.asAnySpec, c.asAnySpec), throwsArgumentError);
        final s2 = subtract(iC, c);
        expect(s2.dtype, DType.complex128);
        expect(s2.getCell([0]), Complex(-8.0, -10.0));

        // 3. double - int
        expect(() => subtract(d.asAnySpec, i.asAnySpec), throwsArgumentError);
        final s3 = subtract(d, iD);
        expect(s3.dtype, DType.float64);
        expect(s3.getCell([0]), 2.0);

        // 4. int - double
        expect(() => subtract(i.asAnySpec, d.asAnySpec), throwsArgumentError);
        final s4 = subtract(iD, d);
        expect(s4.dtype, DType.float64);
        expect(s4.getCell([0]), -2.0);

        // 5. int - int
        final s5 = subtract(i, i);
        expect(s5.dtype, DType.int64);
        expect(s5.getCell([0]), 0);

        // --- multiply() Gaps ---
        // 1. Complex * Complex
        final m1 = multiply(c, c);
        expect(m1.dtype, DType.complex128);
        expect(m1.getCell([0]), Complex(0.0, 200.0)); // (10+10i)^2 = 0 + 200i

        // 2. Complex * double
        expect(() => multiply(c.asAnySpec, d.asAnySpec), throwsArgumentError);
        final m2 = multiply(c, dC);
        expect(m2.dtype, DType.complex128);
        expect(m2.getCell([0]), Complex(40.0, 40.0));

        // 3. Complex * int
        expect(() => multiply(c.asAnySpec, i.asAnySpec), throwsArgumentError);
        final m3 = multiply(c, iC);
        expect(m3.dtype, DType.complex128);
        expect(m3.getCell([0]), Complex(20.0, 20.0));

        // 4. double * Complex
        expect(() => multiply(d.asAnySpec, c.asAnySpec), throwsArgumentError);
        final m4 = multiply(dC, c);
        expect(m4.dtype, DType.complex128);
        expect(m4.getCell([0]), Complex(40.0, 40.0));

        // 5. int * Complex
        expect(() => multiply(i.asAnySpec, c.asAnySpec), throwsArgumentError);
        final m5 = multiply(iC, c);
        expect(m5.dtype, DType.complex128);
        expect(m5.getCell([0]), Complex(20.0, 20.0));

        // 6. double * int
        expect(() => multiply(d.asAnySpec, i.asAnySpec), throwsArgumentError);
        final m6 = multiply(d, iD);
        expect(m6.dtype, DType.float64);
        expect(m6.getCell([0]), 8.0);

        // 7. int * double
        expect(() => multiply(i.asAnySpec, d.asAnySpec), throwsArgumentError);
        final m7 = multiply(iD, d);
        expect(m7.dtype, DType.float64);
        expect(m7.getCell([0]), 8.0);

        // 8. int * int
        final m8 = multiply(i, i);
        expect(m8.dtype, DType.int64);
        expect(m8.getCell([0]), 4);

        // --- divide() Gaps ---
        // 1. Complex / Complex
        final div1 = divide(c, c);
        expect(div1.dtype, DType.complex128);
        expect(div1.getCell([0]), Complex(1.0, 0.0));

        // 2. Complex / double
        expect(() => divide<AnySpec, DTypeTag>(c, d), throwsArgumentError);
        final div2 = divide(c, dC);
        expect(div2.dtype, DType.complex128);
        expect(div2.getCell([0]), Complex(2.5, 2.5));

        // 3. Complex / int
        expect(() => divide<AnySpec, DTypeTag>(c, i), throwsArgumentError);
        final div3 = divide(c, iC);
        expect(div3.dtype, DType.complex128);
        expect(div3.getCell([0]), Complex(5.0, 5.0));

        // 4. double / Complex
        expect(() => divide<AnySpec, DTypeTag>(d, c), throwsArgumentError);
        final div4 = divide(dC, c);
        expect(div4.dtype, DType.complex128);
        expect(
          div4.getCell([0]),
          Complex(0.2, -0.2),
        ); // 4 / (10+10i) = 0.2 - 0.2i

        // 5. int / Complex
        expect(() => divide<AnySpec, DTypeTag>(i, c), throwsArgumentError);
        final div5 = divide(iC, c);
        expect(
          div5.getCell([0]),
          Complex(0.1, -0.1),
        ); // 2 / (10+10i) = 0.1 - 0.1i
        iC.dispose();
        dC.dispose();
        iD.dispose();
      },
    );

    test(
      'Phase 2: Strided Binary Operators (add, subtract, multiply, divide)',
      () {
        NDArray.scope(() {
          // float64 strided
          final a = NDArray.fromList(
            [1.0, -9.9, 2.0, -9.9, 3.0],
            [5],
            DType.float64,
          ).slice([const Slice(start: 0, stop: 5, step: 2)]);
          final b = NDArray.fromList(
            [10.0, -9.9, 20.0, -9.9, 30.0],
            [5],
            DType.float64,
          ).slice([const Slice(start: 0, stop: 5, step: 2)]);

          expect(a.isContiguous, false);
          expect(b.isContiguous, false);

          final resAdd = add(a, b);
          expect(resAdd.dtype, DType.float64);
          expect(resAdd.toList(), [11.0, 22.0, 33.0]);

          final resSub = subtract(a, b);
          expect(resSub.toList(), [-9.0, -18.0, -27.0]);

          final resMul = multiply(a, b);
          expect(resMul.toList(), [10.0, 40.0, 90.0]);

          final resDiv = divide(b, a);
          expect(resDiv.toList(), [10.0, 10.0, 10.0]);

          // complex128 strided
          final z1 = NDArray.fromList(
            [
              Complex(1.0, 2.0),
              Complex(9, 9),
              Complex(3.0, 4.0),
              Complex(9, 9),
            ],
            [4],
            DType.complex128,
          ).slice([const Slice(start: 0, stop: 4, step: 2)]);

          final z2 = NDArray.fromList(
            [
              Complex(10.0, 20.0),
              Complex(9, 9),
              Complex(30.0, 40.0),
              Complex(9, 9),
            ],
            [4],
            DType.complex128,
          ).slice([const Slice(start: 0, stop: 4, step: 2)]);

          final zAdd = add(z1, z2);
          expect(zAdd.dtype, DType.complex128);
          expect(zAdd.getCell([0]), Complex(11.0, 22.0));
          expect(zAdd.getCell([1]), Complex(33.0, 44.0));

          final zSub = subtract(z2, z1);
          expect(zSub.getCell([0]), Complex(9.0, 18.0));
          expect(zSub.getCell([1]), Complex(27.0, 36.0));
        });
      },
    );

    test('Phase 2: Combinatorial DType Promotions', () {
      NDArray.scope(() {
        final i32 = NDArray.fromList([1, 2], [2], DType.int32);
        final f64 = NDArray.fromList([0.5, 1.5], [2], DType.float64);
        expect(() => add(i32.asAnySpec, f64.asAnySpec), throwsArgumentError);
        final res1 = addAs(i32, f64, DType.float64);
        expect(res1.dtype, DType.float64);
        expect(res1.toList(), [1.5, 3.5]);

        final i64 = NDArray.fromList([3, 4], [2], DType.int64);
        expect(
          () => multiply(i64.asAnySpec, i32.asAnySpec),
          throwsArgumentError,
        );
        final res2 = multiplyAs(i64, i32, DType.int64);
        expect(res2.dtype, DType.int64);
        expect(res2.toList(), [3, 8]);

        final u8 = NDArray.fromList([10, 20], [2], DType.uint8);
        final i16 = NDArray.fromList([100, 200], [2], DType.int16);
        expect(() => add(u8.asAnySpec, i16.asAnySpec), throwsArgumentError);
        final res3 = addAs(u8, i16, DType.int16);
        expect(res3.dtype, DType.int16);
        expect(res3.toList(), [110, 220]);

        // Same-dtype addAs/multiplyAs into wider output dtype works
        final u8b = NDArray.fromList([250, 20], [2], DType.uint8);
        final resWide = addAs(u8, u8b, DType.int16);
        expect(resWide.dtype, DType.int16);
        expect(resWide.toList(), [260, 40]);

        // Operators keep the receiver's dtype, so mixed dtypes are rejected.
        expect(() => (i32 as dynamic) + f64, throwsArgumentError);
        expect(() => (i32 as dynamic) * i64, throwsArgumentError);
      });
    });

    test(
      'Phase 2: logicalNot on non-boolean numeric arrays (contiguous & strided)',
      () {
        NDArray.scope(() {
          // --- int32 contiguous ---
          final i32 = NDArray.fromList([0, 1, -2, 0], [4], DType.int32);
          final resI32 = logicalNot(i32);
          expect(resI32.dtype, DType.boolean);
          expect(resI32.toList(), [true, false, false, true]);

          // --- int32 strided ---
          final i32Strided = NDArray.fromList(
            [0, 99, 1, 99, -2, 99, 0],
            [7],
            DType.int32,
          ).slice([const Slice(start: 0, stop: 7, step: 2)]);
          expect(i32Strided.isContiguous, false);
          final resI32Strided = logicalNot(i32Strided);
          expect(resI32Strided.toList(), [true, false, false, true]);

          // --- float64 contiguous ---
          final f64 = NDArray.fromList(
            [0.0, 0.5, -1.2, 0.0],
            [4],
            DType.float64,
          );
          final resF64 = logicalNot(f64);
          expect(resF64.dtype, DType.boolean);
          expect(resF64.toList(), [true, false, false, true]);

          // --- float64 strided ---
          final f64Strided = NDArray.fromList(
            [0.0, 9.9, 0.5, 9.9, -1.2, 9.9, 0.0],
            [7],
            DType.float64,
          ).slice([const Slice(start: 0, stop: 7, step: 2)]);
          final resF64Strided = logicalNot(f64Strided);
          expect(resF64Strided.toList(), [true, false, false, true]);
        });
      },
    );

    test(
      'Phase 2: Precondition and State Validations (Disposed StateError)',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);

          // 1. Disposed input array
          a.dispose();
          expect(() => add(a, b), throwsStateError);
          expect(() => subtract(b, a), throwsStateError);
          expect(() => sin(a), throwsStateError);

          // Re-create to test disposed out buffer
          final a2 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final outDisposed = NDArray<Float64>.create([2], DType.float64);
          outDisposed.dispose();
          expect(() => add(a2, b, out: outDisposed), throwsStateError);
          expect(() => sin(a2, out: outDisposed), throwsStateError);
        });
      },
    );

    test(
      'Phase 2: Precondition and State Validations (Mismatched Shape/DType ArgumentError)',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);

          // Mismatched shape for out buffer
          final outBadShape = NDArray<Float64>.create([3], DType.float64);
          expect(() => add(a, b, out: outBadShape), throwsArgumentError);
          expect(() => sin(a, out: outBadShape), throwsArgumentError);

          // Incompatible DType for out buffer
          final outBadDType = NDArray<Int32>.create([2], DType.int32);
          expect(
            () => add(a.asAnySpec, b.asAnySpec, out: outBadDType.asAnySpec),
            throwsArgumentError,
          );
          expect(() => sin(a, out: outBadDType), throwsArgumentError);
        });
      },
    );
  });
}
