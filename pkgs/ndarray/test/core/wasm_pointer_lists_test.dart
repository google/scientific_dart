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

import 'package:ffi/ffi.dart';
import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/wasm_pointer_lists.dart';
import 'package:test/test.dart';

void _expectUnresizableAndBoundsChecked<E>(List<E> list, E sampleValue) {
  expect(() => list.length = 0, throwsUnsupportedError);
  expect(() => list[-1], throwsRangeError);
  expect(() => list[list.length], throwsRangeError);
  expect(() => list[-1] = sampleValue, throwsRangeError);
  expect(() => list[list.length] = sampleValue, throwsRangeError);
}

void main() {
  group('wasm_pointer_lists helpers', () {
    test('maxWasm32AllocationBytes matches 32-bit signed max', () {
      expect(maxWasm32AllocationBytes, equals(0x7fffffff));
    });

    test('copyInt64PointerToList copies empty and non-empty buffers', () {
      final emptyPtr = calloc<ffi.Int64>(1);
      try {
        final empty = copyInt64PointerToList(emptyPtr, 0);
        expect(empty, isA<Int64List>());
        expect(empty, isEmpty);
      } finally {
        calloc.free(emptyPtr);
      }

      const values = [0, -1, 42, 0x123456789abcdef, -0x7fffffffffffffff];
      final ptr = calloc<ffi.Int64>(values.length);
      try {
        for (var i = 0; i < values.length; i++) {
          ptr[i] = values[i];
        }
        final copied = copyInt64PointerToList(ptr, values.length);
        expect(copied, equals(values));
        // Mutating the native buffer after copying does not affect the copy.
        ptr[0] = 999;
        expect(copied[0], equals(0));
      } finally {
        calloc.free(ptr);
      }
    });

    test('setPointerAt stores pointer addresses at arbitrary indices', () {
      final a = calloc<ffi.Int64>(1);
      final b = calloc<ffi.Int64>(1);
      final table = calloc<ffi.Pointer<ffi.Int64>>(3);
      try {
        a.value = 111;
        b.value = 222;
        setPointerAt(table, 0, a);
        setPointerAt(table, 1, ffi.Pointer<ffi.Int64>.fromAddress(0));
        setPointerAt(table, 2, b);

        final ptrSize = ffi.sizeOf<ffi.Pointer<ffi.Void>>();
        int readAddressAt(int index) => ptrSize == 4
            ? table.cast<ffi.Uint32>()[index]
            : table.cast<ffi.Uint64>()[index];

        expect(readAddressAt(0), equals(a.address));
        expect(readAddressAt(1), equals(0));
        expect(readAddressAt(2), equals(b.address));
        expect(ffi.Pointer<ffi.Int64>.fromAddress(readAddressAt(0)).value, 111);
        expect(ffi.Pointer<ffi.Int64>.fromAddress(readAddressAt(2)).value, 222);
      } finally {
        calloc.free(table);
        calloc.free(b);
        calloc.free(a);
      }
    });
  });

  group('Wasm*PointerList implementations', () {
    test('WasmFloat64PointerList reads, writes, and enforces bounds', () {
      final ptr = calloc<ffi.Double>(3);
      try {
        final list = WasmFloat64PointerList(ptr, 3);
        expect(list.length, equals(3));
        list[0] = 1.5;
        list[1] = -2.25;
        list[2] = double.infinity;
        expect(list, equals([1.5, -2.25, double.infinity]));
        expect(ptr[1], equals(-2.25));
        ptr[2] = 3.75;
        expect(list[2], equals(3.75));
        _expectUnresizableAndBoundsChecked(list, 0.0);
      } finally {
        calloc.free(ptr);
      }
    });

    test('WasmFloat32PointerList reads, writes, and enforces bounds', () {
      final ptr = calloc<ffi.Float>(3);
      try {
        final list = WasmFloat32PointerList(ptr, 3);
        expect(list.length, equals(3));
        list[0] = 1.25;
        list[1] = -4.5;
        list[2] = 8.0;
        expect(list, equals([1.25, -4.5, 8.0]));
        expect(ptr[0], equals(1.25));
        _expectUnresizableAndBoundsChecked(list, 0.0);
      } finally {
        calloc.free(ptr);
      }
    });

    test('WasmInt64PointerList reads, writes, and enforces bounds', () {
      final ptr = calloc<ffi.Int64>(3);
      try {
        final list = WasmInt64PointerList(ptr, 3);
        expect(list.length, equals(3));
        list[0] = -1000000000000;
        list[1] = 0;
        list[2] = 0x7fffffffffffffff;
        expect(list, equals([-1000000000000, 0, 0x7fffffffffffffff]));
        expect(ptr[2], equals(0x7fffffffffffffff));
        _expectUnresizableAndBoundsChecked(list, 0);
      } finally {
        calloc.free(ptr);
      }
    });

    test('WasmUint64PointerList reads, writes, and enforces bounds', () {
      final ptr = calloc<ffi.Uint64>(3);
      try {
        final list = WasmUint64PointerList(ptr, 3);
        expect(list.length, equals(3));
        list[0] = 0;
        list[1] = 42;
        list[2] = -1; // 0xffffffffffffffff bit pattern
        expect(list, equals([0, 42, -1]));
        expect(ptr[1], equals(42));
        _expectUnresizableAndBoundsChecked(list, 0);
      } finally {
        calloc.free(ptr);
      }
    });

    test('WasmInt32PointerList reads, writes, and enforces bounds', () {
      final ptr = calloc<ffi.Int32>(3);
      try {
        final list = WasmInt32PointerList(ptr, 3);
        expect(list.length, equals(3));
        list[0] = -2147483648;
        list[1] = 0;
        list[2] = 2147483647;
        expect(list, equals([-2147483648, 0, 2147483647]));
        expect(ptr[0], equals(-2147483648));
        _expectUnresizableAndBoundsChecked(list, 0);
      } finally {
        calloc.free(ptr);
      }
    });

    test('WasmUint32PointerList reads, writes, and enforces bounds', () {
      final ptr = calloc<ffi.Uint32>(3);
      try {
        final list = WasmUint32PointerList(ptr, 3);
        expect(list.length, equals(3));
        list[0] = 0;
        list[1] = 123456;
        list[2] = 0xffffffff;
        expect(list, equals([0, 123456, 0xffffffff]));
        expect(ptr[2], equals(0xffffffff));
        _expectUnresizableAndBoundsChecked(list, 0);
      } finally {
        calloc.free(ptr);
      }
    });

    test('WasmInt16PointerList reads, writes, and enforces bounds', () {
      final ptr = calloc<ffi.Int16>(3);
      try {
        final list = WasmInt16PointerList(ptr, 3);
        expect(list.length, equals(3));
        list[0] = -32768;
        list[1] = 0;
        list[2] = 32767;
        expect(list, equals([-32768, 0, 32767]));
        expect(ptr[0], equals(-32768));
        _expectUnresizableAndBoundsChecked(list, 0);
      } finally {
        calloc.free(ptr);
      }
    });

    test('WasmUint16PointerList reads, writes, and enforces bounds', () {
      final ptr = calloc<ffi.Uint16>(3);
      try {
        final list = WasmUint16PointerList(ptr, 3);
        expect(list.length, equals(3));
        list[0] = 0;
        list[1] = 1024;
        list[2] = 65535;
        expect(list, equals([0, 1024, 65535]));
        expect(ptr[2], equals(65535));
        _expectUnresizableAndBoundsChecked(list, 0);
      } finally {
        calloc.free(ptr);
      }
    });

    test('WasmInt8PointerList reads, writes, and enforces bounds', () {
      final ptr = calloc<ffi.Int8>(3);
      try {
        final list = WasmInt8PointerList(ptr, 3);
        expect(list.length, equals(3));
        list[0] = -128;
        list[1] = 0;
        list[2] = 127;
        expect(list, equals([-128, 0, 127]));
        expect(ptr[0], equals(-128));
        _expectUnresizableAndBoundsChecked(list, 0);
      } finally {
        calloc.free(ptr);
      }
    });

    test('WasmUint8PointerList reads, writes, and enforces bounds', () {
      final ptr = calloc<ffi.Uint8>(3);
      try {
        final list = WasmUint8PointerList(ptr, 3);
        expect(list.length, equals(3));
        list[0] = 0;
        list[1] = 128;
        list[2] = 255;
        expect(list, equals([0, 128, 255]));
        expect(ptr[2], equals(255));
        _expectUnresizableAndBoundsChecked(list, 0);
      } finally {
        calloc.free(ptr);
      }
    });
  });

  group('createWasmDataView across all 15 DTypes', () {
    test('creates live views for every DType', () {
      const count = 4;
      for (final dtype in DType.values) {
        final ptr = calloc<ffi.Uint8>(count * dtype.byteWidth);
        try {
          final view = createWasmDataView(ptr.cast<ffi.Void>(), dtype, count);
          expect(view.length, equals(count), reason: 'length for $dtype');
          switch (dtype) {
            case DType.float64:
              expect(view, isA<WasmFloat64PointerList>());
              view[0] = 1.5;
              view[3] = -2.5;
              expect(view[0], equals(1.5));
              expect(view[3], equals(-2.5));
            case DType.float32:
              expect(view, isA<WasmFloat32PointerList>());
              view[1] = 3.25;
              expect(view[1], equals(3.25));
            case DType.float16:
              expect(view, isA<Float16List>());
              view[0] = 1.5;
              view[2] = -0.5;
              expect(view[0], equals(1.5));
              expect(view[2], equals(-0.5));
            case DType.bfloat16:
              expect(view, isA<BFloat16List>());
              view[0] = 2.0;
              view[3] = -3.0;
              expect(view[0], equals(2.0));
              expect(view[3], equals(-3.0));
            case DType.int64:
              expect(view, isA<WasmInt64PointerList>());
              view[0] = -42;
              expect(view[0], equals(-42));
            case DType.uint64:
              expect(view, isA<WasmUint64PointerList>());
              view[1] = 100;
              expect(view[1], equals(100));
            case DType.int32:
              expect(view, isA<WasmInt32PointerList>());
              view[0] = -1234;
              expect(view[0], equals(-1234));
            case DType.uint32:
              expect(view, isA<WasmUint32PointerList>());
              view[2] = 5678;
              expect(view[2], equals(5678));
            case DType.int16:
              expect(view, isA<WasmInt16PointerList>());
              view[0] = -300;
              expect(view[0], equals(-300));
            case DType.uint16:
              expect(view, isA<WasmUint16PointerList>());
              view[1] = 60000;
              expect(view[1], equals(60000));
            case DType.int8:
              expect(view, isA<WasmInt8PointerList>());
              view[0] = -12;
              expect(view[0], equals(-12));
            case DType.uint8:
              expect(view, isA<WasmUint8PointerList>());
              view[3] = 200;
              expect(view[3], equals(200));
            case DType.complex128:
              view[0] = const Complex(1.5, -2.5);
              expect(view[0], equals(const Complex(1.5, -2.5)));
            case DType.complex64:
              view[1] = const Complex(0.5, 4.0);
              expect(view[1], equals(const Complex(0.5, 4.0)));
            case DType.boolean:
              view[0] = true;
              view[1] = false;
              view[2] = true;
              expect(view[0], isTrue);
              expect(view[1], isFalse);
              expect(view[2], isTrue);
          }
        } finally {
          calloc.free(ptr);
        }
      }
    });
  });
}
