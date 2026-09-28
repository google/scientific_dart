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

import 'package:ndarray/ndarray.dart';

void main() {
  print('=== ScratchArena Helper Examples ===\n');

  NDArray.scope(() {
    // Record the stack marker to clean up memory when finished.
    final marker = ScratchArena.marker;

    try {
      // 1. Copy Dart List<int> to ffi.Pointer<ffi.Int>
      final intList = [1, 2, 3, 4, 5];
      final cInts = ScratchArena.copyInts(intList);
      print('Copied List<int> to Pointer<Int>:');
      for (var i = 0; i < intList.length; i++) {
        print('  cInts[$i] = ${cInts[i]}');
      }

      // 2. Copy Dart List<int> to ffi.Pointer<ffi.Int32>
      final cInt32s = ScratchArena.copyInt32s(intList);
      print('\nCopied List<int> to Pointer<Int32>:');
      for (var i = 0; i < intList.length; i++) {
        print('  cInt32s[$i] = ${cInt32s[i]}');
      }

      // 3. Copy Dart List<int> to ffi.Pointer<ffi.Int64>
      final cInt64s = ScratchArena.copyInt64s(intList);
      print('\nCopied List<int> to Pointer<Int64>:');
      for (var i = 0; i < intList.length; i++) {
        print('  cInt64s[$i] = ${cInt64s[i]}');
      }

      // 4. Copy Dart List<double> to ffi.Pointer<ffi.Double>
      final doubleList = [1.5, 2.5, 3.5];
      final cDoubles = ScratchArena.copyDoubles(doubleList);
      print('\nCopied List<double> to Pointer<Double>:');
      for (var i = 0; i < doubleList.length; i++) {
        print('  cDoubles[$i] = ${cDoubles[i]}');
      }

      // 5. Copy Dart List<double> to ffi.Pointer<ffi.Float>
      final cFloats = ScratchArena.copyFloats(doubleList);
      print('\nCopied List<double> to Pointer<Float>:');
      for (var i = 0; i < doubleList.length; i++) {
        print('  cFloats[$i] = ${cFloats[i]}');
      }

      // 6. Copy Dart List<Complex> to ffi.Pointer<ffi.Double> (Complexes)
      final complexList = [Complex(1.0, 2.0), Complex(3.0, 4.0)];
      final cComplexes = ScratchArena.copyComplexes(complexList);
      print('\nCopied List<Complex> to Pointer<Double> (Complexes):');
      for (var i = 0; i < complexList.length; i++) {
        print(
          '  cComplexes[$i] = real: ${cComplexes[i * 2]}, imag: ${cComplexes[i * 2 + 1]}',
        );
      }

      // 7. Copy Dart List<Complex> to ffi.Pointer<ffi.Float> (Float Complexes)
      final cFloatComplexes = ScratchArena.copyFloatComplexes(complexList);
      print('\nCopied List<Complex> to Pointer<Float> (Float Complexes):');
      for (var i = 0; i < complexList.length; i++) {
        print(
          '  cFloatComplexes[$i] = real: ${cFloatComplexes[i * 2]}, imag: ${cFloatComplexes[i * 2 + 1]}',
        );
      }

      // 8. Copy Dart List<bool> to ffi.Pointer<ffi.Uint8> (Bools)
      final boolList = [true, false, true, true];
      final cBools = ScratchArena.copyBools(boolList);
      print('\nCopied List<bool> to Pointer<Uint8>:');
      for (var i = 0; i < boolList.length; i++) {
        print('  cBools[$i] = ${cBools[i]} (${cBools[i] != 0})');
      }
    } finally {
      // Reset the arena stack to free all temporary allocations made in this block.
      ScratchArena.reset(marker);
      print('\nScratchArena successfully reset.');
    }
  });
}
