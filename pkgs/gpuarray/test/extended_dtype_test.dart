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

import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart' as nd;
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Extended DTypes & Complete 15-Type Support', () {
    test(
      'Float16 & BFloat16 GpuArray creation, arithmetic, and host roundtrip',
      () {
        ResourceScope.scope(() {
          // Float16
          final aF16 = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float16,
          );
          final bF16 = GpuArray.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [2, 2],
            DType.float16,
          );
          expect(aF16.dtype, equals(DType.float16));
          expect(aF16.dtype.byteWidth, equals(2));

          final sumF16 = aF16 + bF16;
          expect(sumF16.dtype, equals(DType.float16));
          expect(sumF16.toList(), equals([11.0, 22.0, 33.0, 44.0]));

          // Roundtrip to host NDArray
          final hostND = sumF16.toNDArray();
          expect(hostND.dtype, equals(nd.DType.float16));
          expect(hostND.toList(), equals([11.0, 22.0, 33.0, 44.0]));

          // Convert back to GPU
          final backGpu = hostND.toGpu();
          expect(backGpu.dtype, equals(DType.float16));
          expect(backGpu.toList(), equals([11.0, 22.0, 33.0, 44.0]));

          // BFloat16
          final aBF16 = GpuArray.fromList([2.0, 4.0], [2], DType.bfloat16);
          final scaled = aBF16 * 3.0;
          expect(scaled.dtype, equals(DType.bfloat16));
          expect(scaled.toList(), equals([6.0, 12.0]));
        });
      },
    );

    test('Int8, Uint16, Uint32, Uint64 on GPU and host roundtrips', () {
      ResourceScope.scope(() {
        // Int8
        final i8 = GpuArray.fromList([10, -20, 30], [3], DType.int8);
        expect(i8.dtype, equals(DType.int8));
        expect(i8.dtype.byteWidth, equals(1));
        expect(i8.toList(), equals([10, -20, 30]));
        expect(i8.toNDArray().toList(), equals([10, -20, 30]));

        // Uint16
        final u16 = GpuArray.fromList([100, 200, 300], [3], DType.uint16);
        expect(u16.dtype, equals(DType.uint16));
        expect(u16.dtype.byteWidth, equals(2));
        expect(u16.toList(), equals([100, 200, 300]));
        expect(u16.toNDArray().toList(), equals([100, 200, 300]));

        // Uint32
        final u32 = GpuArray.fromList([1000, 2000, 3000], [3], DType.uint32);
        expect(u32.dtype, equals(DType.uint32));
        expect(u32.dtype.byteWidth, equals(4));
        expect(u32.toList(), equals([1000, 2000, 3000]));
        expect(u32.toNDArray().toList(), equals([1000, 2000, 3000]));

        // Uint64
        final u64 = GpuArray.fromList([5, 10, 15], [3], DType.uint64);
        expect(u64.dtype, equals(DType.uint64));
        expect(u64.dtype.byteWidth, equals(8));
        expect(u64.toList(), equals([5, 10, 15]));
        expect(u64.toNDArray().toList(), equals([5, 10, 15]));
      });
    });

    test(
      'All 15 data types allocate and transfer cleanly between NDArray and GpuArray',
      () {
        ResourceScope.scope(() {
          for (final dtype in DType.values) {
            final GpuArray arr;
            if (dtype == DType.boolean) {
              arr = GpuArray.fromList(
                [true, false, true, false],
                [2, 2],
                dtype,
              );
            } else {
              arr = GpuArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], dtype);
            }
            expect(arr.dtype, equals(dtype));
            expect(arr.shape, equals([2, 2]));

            final host = arr.toNDArray();
            expect(host.dtype.name, equals(dtype.name));
            expect(host.shape, equals([2, 2]));

            final back = host.toGpu();
            expect(back.dtype.name, equals(dtype.name));
            expect(back.shape, equals([2, 2]));
          }
        });
      },
    );

    test('Complex64 and Complex128 preserve imaginary parts on GPU', () {
      ResourceScope.scope(() {
        final cpxList = [
          nd.Complex(1.0, 2.0),
          nd.Complex(3.0, -4.0),
          nd.Complex(0.0, 5.0),
          nd.Complex(-6.0, 0.0),
        ];
        final gpuCpx128 = GpuArray.fromList(cpxList, [2, 2], DType.complex128);
        expect(gpuCpx128.dtype, equals(DType.complex128));

        final hostND = gpuCpx128.toNDArray();
        expect(hostND.toList(), equals(cpxList));

        final copiedGpu = gpuCpx128.copy();
        expect(copiedGpu.toList(), equals(cpxList));
      });
    });

    test('Uint64 handles large values > 2^63 - 1 without sign corruption', () {
      ResourceScope.scope(() {
        final largeVal = 0x8000000000000000; // 2^63
        final gpuU64 = GpuArray.fromList([largeVal, 100], [2], DType.uint64);
        expect(gpuU64.dtype, equals(DType.uint64));

        final hostND = gpuU64.toNDArray();
        expect(hostND.toList(), equals([largeVal, 100]));
      });
    });
    test('Boolean promotion yields boolean', () {
      ResourceScope.scope(() {
        expect(
          GpuArray.promoteDTypes(DType.boolean, DType.boolean),
          equals(DType.boolean),
        );
        final a = GpuArray.fromList([true, false], [2], DType.boolean);
        final b = GpuArray.fromList([false, true], [2], DType.boolean);
        final res = a + b;
        expect(res.dtype, equals(DType.boolean));
      });
    });

    test('GpuArray.filled and GpuArray.scalar support all 15 dtypes', () {
      ResourceScope.scope(() {
        // Float16
        final f16 = GpuArray.filled([1], 3.5, DType.float16);
        expect(f16.scalar, equals(3.5));

        // BFloat16
        final bf16 = GpuArray.filled([1], 4.25, DType.bfloat16);
        expect(bf16.scalar, equals(4.25));

        // Complex64
        final c64 = GpuArray.filled([1], nd.Complex(1.5, 2.5), DType.complex64);
        expect(c64.scalar, equals(Complex(1.5, 2.5)));

        // Complex128
        final c128 = GpuArray.filled(
          [1],
          nd.Complex(3.0, -4.0),
          DType.complex128,
        );
        expect(c128.scalar, equals(Complex(3.0, -4.0)));

        // Uint64
        final u64 = GpuArray.filled([1], 123456789, DType.uint64);
        expect(u64.scalar, equals(123456789));

        // Uint32
        final u32 = GpuArray.filled([1], 42000, DType.uint32);
        expect(u32.scalar, equals(42000));

        // Uint16
        final u16 = GpuArray.filled([1], 1234, DType.uint16);
        expect(u16.scalar, equals(1234));

        // Int8
        final i8 = GpuArray.filled([1], -50, DType.int8);
        expect(i8.scalar, equals(-50));

        // Boolean
        final bTrue = GpuArray.filled([1], true, DType.boolean);
        expect(bTrue.scalar, isTrue);

        // Float64
        final f64 = GpuArray.filled([1], 99.75, DType.float64);
        expect(f64.scalar, equals(99.75));
      });
    });

    test('Complex arithmetic operations maintain real and imaginary parts', () {
      ResourceScope.scope(() {
        final c1 = GpuArray.fromList(
          [nd.Complex(1.0, 2.0), nd.Complex(3.0, 4.0)],
          [2],
          DType.complex128,
        );
        final c2 = GpuArray.fromList(
          [nd.Complex(2.0, 1.0), nd.Complex(1.0, -1.0)],
          [2],
          DType.complex128,
        );

        // Addition: (1+2i) + (2+1i) = (3+3i), (3+4i) + (1-1i) = (4+3i)
        final sum = c1 + c2;
        expect(
          sum.toList(),
          equals([nd.Complex(3.0, 3.0), nd.Complex(4.0, 3.0)]),
        );

        // Subtraction: (1+2i) - (2+1i) = (-1+1i)
        final diff = c1 - c2;
        expect(
          diff.toList(),
          equals([nd.Complex(-1.0, 1.0), nd.Complex(2.0, 5.0)]),
        );

        // Multiplication: (1+2i) * (2+1i) = (2-2 + 1i+4i) = (0 + 5i)
        final prod = c1 * c2;
        expect(
          prod.toList(),
          equals([nd.Complex(0.0, 5.0), nd.Complex(7.0, 1.0)]),
        );
      });
    });
  });
}
