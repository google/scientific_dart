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

import "package:ndarray/ndarray.dart";
import "package:test/test.dart";

void main() {
  group("NDArray.scalar Constructor Tests", () {
    test("Float64 scalar", () {
      NDArray.scope(() {
        final a = NDArray.scalar(Float64(3.14159), dtype: DType.float64);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.float64);
        expect(a.scalar, closeTo(3.14159, 1e-5));
      });
    });

    test("Float32 scalar", () {
      NDArray.scope(() {
        final a = NDArray.scalar(Float32(2.718), dtype: DType.float32);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.float32);
        expect(a.scalar, closeTo(2.718, 1e-3));
      });
    });

    test("Int64 scalar", () {
      NDArray.scope(() {
        final a = NDArray.scalar(
          Int64(9223372036854775807),
          dtype: DType.int64,
        );
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.int64);
        expect(a.scalar, 9223372036854775807);
      });
    });

    test("Int32 scalar", () {
      NDArray.scope(() {
        final a = NDArray.scalar(Int32(42), dtype: DType.int32);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.int32);
        expect(a.scalar, 42);
      });
    });

    test("Int16 scalar", () {
      NDArray.scope(() {
        final a = NDArray.scalar(Int16(32767), dtype: DType.int16);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.int16);
        expect(a.scalar, 32767);
      });
    });

    test("Uint8 scalar", () {
      NDArray.scope(() {
        final a = NDArray.scalar(Uint8(255), dtype: DType.uint8);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.uint8);
        expect(a.scalar, 255);
      });
    });

    test("Boolean scalar", () {
      NDArray.scope(() {
        final a = NDArray.scalar(true, dtype: DType.boolean);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.boolean);
        expect(a.scalar, true);

        final b = NDArray.scalar(false, dtype: DType.boolean);
        expect(b.scalar, false);
      });
    });

    test("Complex128 scalar", () {
      NDArray.scope(() {
        final val = Complex128(1.5, -2.5);
        final a = NDArray.scalar(val, dtype: DType.complex128);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.complex128);
        expect(a.scalar.real, 1.5);
        expect(a.scalar.imag, -2.5);
      });
    });

    test("Complex64 scalar", () {
      NDArray.scope(() {
        final val = Complex64(0.5, 1.0);
        final a = NDArray.scalar(val, dtype: DType.complex64);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.complex64);
        expect(a.scalar.real, closeTo(0.5, 1e-5));
        expect(a.scalar.imag, closeTo(1.0, 1e-5));
      });
    });

    test("Optional dtype inference tests", () {
      NDArray.scope(() {
        final aInt = NDArray.scalar(42);
        expect(aInt.dtype, DType.int64);
        expect(aInt.scalar, 42);

        final aDouble = NDArray.scalar(3.14);
        expect(aDouble.dtype, DType.float64);
        expect(aDouble.scalar, closeTo(3.14, 1e-5));

        final aBool = NDArray.scalar(true);
        expect(aBool.dtype, DType.boolean);
        expect(aBool.scalar, true);

        final aComplex = NDArray.scalar(Complex(1.0, 2.0));
        expect(aComplex.dtype, DType.complex128);
        expect(aComplex.scalar.real, 1.0);

        final aF64 = NDArray<Float64>.scalar(Float64(3.14159));
        expect(aF64.dtype, DType.float64);

        final aF32 = NDArray<Float32>.scalar(
          Float32(1.5),
          dtype: DType.float32,
        );
        expect(aF32.dtype, DType.float32);

        final aI64 = NDArray<Int64>.scalar(Int64(999));
        expect(aI64.dtype, DType.int64);

        final aI32 = NDArray<Int32>.scalar(Int32(100), dtype: DType.int32);
        expect(aI32.dtype, DType.int32);

        final aI16 = NDArray<Int16>.scalar(Int16(50), dtype: DType.int16);
        expect(aI16.dtype, DType.int16);

        final aU8 = NDArray<Uint8>.scalar(Uint8(200), dtype: DType.uint8);
        expect(aU8.dtype, DType.uint8);

        final aC128 = NDArray<Complex128>.scalar(Complex128(1.0, 2.0));
        expect(aC128.dtype, DType.complex128);

        final aC64 = NDArray<Complex64>.scalar(
          Complex64(0.5, 1.5),
          dtype: DType.complex64,
        );
        expect(aC64.dtype, DType.complex64);
      });
    });

    test("0-D view with offsetElements > 0", () {
      NDArray.scope(() {
        final arr = NDArray.fromList([10, 20, 30, 40], [4], DType.int64);
        final view0D = NDArray.view(
          arr,
          shape: <int>[],
          strides: <int>[],
          offsetElements: 2,
        );
        expect(view0D.rank, 0);
        expect(view0D.scalar, 30);
        expect(view0D.scalar, view0D.getCell([]));
      });
    });
  });
}
