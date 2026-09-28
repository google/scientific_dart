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
import 'package:ndarray/ndarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  group(
    'Extended Data Types (Float16, BFloat16, Int8, Uint16, Uint32, Uint64)',
    () {
      test('Float16 bitwise encoding and decoding', () {
        expect(
          Float16Utils.decodeFloat16(Float16Utils.encodeFloat16(0.0)),
          equals(0.0),
        );
        expect(
          Float16Utils.decodeFloat16(Float16Utils.encodeFloat16(1.0)),
          equals(1.0),
        );
        expect(
          Float16Utils.decodeFloat16(Float16Utils.encodeFloat16(-1.0)),
          equals(-1.0),
        );
        expect(
          Float16Utils.decodeFloat16(Float16Utils.encodeFloat16(2.0)),
          equals(2.0),
        );
        expect(
          Float16Utils.decodeFloat16(Float16Utils.encodeFloat16(0.5)),
          equals(0.5),
        );
        expect(
          Float16Utils.decodeFloat16(Float16Utils.encodeFloat16(65504.0)),
          equals(65504.0),
        ); // max float16

        // Infinities
        expect(
          Float16Utils.decodeFloat16(
            Float16Utils.encodeFloat16(double.infinity),
          ),
          equals(double.infinity),
        );
        expect(
          Float16Utils.decodeFloat16(
            Float16Utils.encodeFloat16(double.negativeInfinity),
          ),
          equals(double.negativeInfinity),
        );

        // NaN
        expect(
          Float16Utils.decodeFloat16(
            Float16Utils.encodeFloat16(double.nan),
          ).isNaN,
          isTrue,
        );
      });

      test('BFloat16 bitwise encoding and decoding', () {
        expect(
          Float16Utils.decodeBFloat16(Float16Utils.encodeBFloat16(0.0)),
          equals(0.0),
        );
        expect(
          Float16Utils.decodeBFloat16(Float16Utils.encodeBFloat16(1.0)),
          equals(1.0),
        );
        expect(
          Float16Utils.decodeBFloat16(Float16Utils.encodeBFloat16(-1.0)),
          equals(-1.0),
        );
        expect(
          Float16Utils.decodeBFloat16(Float16Utils.encodeBFloat16(2.5)),
          equals(2.5),
        );
        expect(
          Float16Utils.decodeBFloat16(Float16Utils.encodeBFloat16(100.0)),
          equals(100.0),
        );
      });

      test('NDArray.create and fromList with Float16 and BFloat16', () {
        ResourceScope.scope(() {
          final f16Arr = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float16,
          );
          expect(f16Arr.dtype, equals(DType.float16));
          expect(f16Arr.dtype.byteWidth, equals(2));
          expect(f16Arr.dtype.isHalf, isTrue);
          expect(f16Arr.dtype.isFloating, isTrue);
          expect(f16Arr.toList(), equals([1.0, 2.0, 3.0, 4.0]));

          final bf16Arr = NDArray.fromList([10.0, 20.0], [2], DType.bfloat16);
          expect(bf16Arr.dtype, equals(DType.bfloat16));
          expect(bf16Arr.dtype.byteWidth, equals(2));
          expect(bf16Arr.dtype.isHalf, isTrue);
          expect(bf16Arr.toList(), equals([10.0, 20.0]));
        });
      });

      test('NDArray with Int8, Uint16, Uint32, Uint64', () {
        ResourceScope.scope(() {
          // Int8
          final i8 = NDArray.fromList([-128, 0, 127], [3], DType.int8);
          expect(i8.dtype, equals(DType.int8));
          expect(i8.dtype.byteWidth, equals(1));
          expect(i8.dtype.isSignedInteger, isTrue);
          expect(i8.toList(), equals([-128, 0, 127]));

          // Uint16
          final u16 = NDArray.fromList([0, 1000, 65535], [3], DType.uint16);
          expect(u16.dtype, equals(DType.uint16));
          expect(u16.dtype.byteWidth, equals(2));
          expect(u16.dtype.isUnsigned, isTrue);
          expect(u16.toList(), equals([0, 1000, 65535]));

          // Uint32
          final u32 = NDArray.fromList(
            [0, 70000, 4294967295],
            [3],
            DType.uint32,
          );
          expect(u32.dtype, equals(DType.uint32));
          expect(u32.dtype.byteWidth, equals(4));
          expect(u32.dtype.isUnsigned, isTrue);
          expect(u32.toList(), equals([0, 70000, 4294967295]));

          // Uint64
          final u64 = NDArray.fromList([1, 2, 3], [3], DType.uint64);
          expect(u64.dtype, equals(DType.uint64));
          expect(u64.dtype.byteWidth, equals(8));
          expect(u64.dtype.isUnsigned, isTrue);
          expect(u64.toList(), equals([1, 2, 3]));
        });
      });

      test('Slicing and views on extended types', () {
        ResourceScope.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float16,
          );
          final sliced = a.slice([Index(1), Slice.all()]);
          expect(sliced.shape, equals([3]));
          expect(sliced.dtype, equals(DType.float16));
          expect(sliced.toList(), equals([4.0, 5.0, 6.0]));

          final reshaped = a.reshape([3, 2]);
          expect(reshaped.shape, equals([3, 2]));
          expect(reshaped.dtype, equals(DType.float16));
          expect(reshaped.toList(), equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        });
      });

      test('NDArray.ones and NDArray.eye with extended types', () {
        ResourceScope.scope(() {
          final onesF16 = NDArray.ones([2, 2], DType.float16);
          expect(onesF16.dtype, equals(DType.float16));
          expect(onesF16.toList(), equals([1.0, 1.0, 1.0, 1.0]));

          final onesBF16 = NDArray.ones([3], DType.bfloat16);
          expect(onesBF16.dtype, equals(DType.bfloat16));
          expect(onesBF16.toList(), equals([1.0, 1.0, 1.0]));

          final eyeF16 = NDArray.eye(2, DType.float16);
          expect(eyeF16.dtype, equals(DType.float16));
          expect(eyeF16.toList(), equals([1.0, 0.0, 0.0, 1.0]));

          final eyeI8 = NDArray.eye(2, DType.int8);
          expect(eyeI8.dtype, equals(DType.int8));
          expect(eyeI8.toList(), equals([1, 0, 0, 1]));
        });
      });

      test('Arithmetic and ufuncs with extended types', () {
        ResourceScope.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float16);
          final b = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float16);

          final sum = add(a, b);
          expect(sum.dtype, equals(DType.float16));
          expect(sum.toList(), equals([5.0, 7.0, 9.0]));

          final diff = subtract(b, a);
          expect(diff.dtype, equals(DType.float16));
          expect(diff.toList(), equals([3.0, 3.0, 3.0]));

          final prod = multiply(a, b);
          expect(prod.dtype, equals(DType.float16));
          expect(prod.toList(), equals([4.0, 10.0, 18.0]));

          final quot = divide(b, a);
          expect(quot.toList(), equals([4.0, 2.5, 2.0]));
        });
      });

      test('BFloat16 NaN handling', () {
        final nanPattern = Float16Utils.encodeBFloat16(double.nan);
        expect(Float16Utils.decodeBFloat16(nanPattern).isNaN, isTrue);
      });
    },
  );
}
