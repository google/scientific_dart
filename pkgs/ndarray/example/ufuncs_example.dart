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

void main() {
  NDArray.scope(() {
    print(
      '=== NDArray Universal Functions (ufuncs) & Broadcasting Examples ===\n',
    );
    runMixedTypeArithmeticExample();
    runTrigAndRoundingExample();
    runComplexAbsoluteValueExample();
    runComparisonBroadcastingExample();
    runLogicalOperationsExample();
    runLogicalReductionsExample();
    runAngleConvertersExample();
    runEnumerateAndComplexComponentsExample();
  });
}

void runMixedTypeArithmeticExample() {
  NDArray.scope(() {
    print('--- Mixed Real & Complex Arithmetic Upcasting ---');
    // Create a 1D complex array
    final a = NDArray<AnySpec>.create([2], DType.complex128);
    a.setCell([0], Complex(1.0, 2.0));
    a.setCell([1], Complex(3.0, 4.0));

    // Create a 1D real array
    final b = NDArray.fromList(Float64List.fromList([10.0, 20.0]), [
      2,
    ], DType.float64);

    // Add them together. The real array automatically upcasts to interact with the complex array!
    final c = add(a, b);
    print('a: ${a.toList()}');
    print('b: ${b.toList()}');
    print(
      'a + b (Upcasted to Complex): ${c.toList()}',
    ); // [Complex(11, 2), Complex(23, 4)]
  });
}

void runTrigAndRoundingExample() {
  NDArray.scope(() {
    print('\n--- New Math Ufuncs (Trig, Hyperbolic, Rounding, Clip) ---');
    final a = NDArray.fromList(Float64List.fromList([0.0, 0.5, 1.0]), [
      3,
    ], DType.float64);

    // Tangent and Atan2 with broadcasting
    final t = tan(a);
    print('tan(a): ${t.toList()}');

    final y = NDArray.fromList(Float64List.fromList([1.0]), [1], DType.float64);
    final x = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
      2,
    ], DType.float64);
    final at = atan2(y, x); // Broadcasts y to match x
    print('atan2(1.0, [1.0, 2.0]) shape: ${at.shape}, data: ${at.toList()}');

    // Hyperbolic trig functions
    print('sinh(a): ${sinh(a).toList()}');
    print('cosh(a): ${cosh(a).toList()}');
    print('tanh(a): ${tanh(a).toList()}');

    // Rounding operations
    final b = NDArray.fromList(Float64List.fromList([-1.6, 1.2, 2.7]), [
      3,
    ], DType.float64);
    print('b: ${b.toList()}');
    print('ceil(b): ${ceil(b).toList()}');
    print('floor(b): ${floor(b).toList()}');
    print('round(b): ${round(b).toList()}');

    // Clipping array elements
    final cl = clip(b, min: -1.0, max: 2.0);
    print('clip(b, -1.0, 2.0): ${cl.toList()}'); // [-1.0, 1.2, 2.0]
  });
}

void runComplexAbsoluteValueExample() {
  NDArray.scope(() {
    print('\n--- Complex Absolute Value (Magnitude) ---');
    final a = NDArray<AnySpec>.create([2], DType.complex128);
    a.setCell([0], Complex(3.0, 4.0)); // magnitude = sqrt(3^2 + 4^2) = 5.0
    a.setCell([1], Complex(0.0, -12.0)); // magnitude = 12.0

    // abs() on complex produces a real Float64 array containing magnitudes
    final magnitudes = abs(a);
    print('Complex array: ${a.toList()}');
    print(
      'Magnitudes (Real array): ${magnitudes.toList()} with dtype ${magnitudes.dtype}',
    );
  });
}

void runComparisonBroadcastingExample() {
  NDArray.scope(() {
    print('\n--- Full NumPy-style Comparison Operator Broadcasting ---');
    // Matrix of shape [2, 3]
    final mat = NDArray.fromList(
      Float64List.fromList([1.0, 5.0, 2.0, 4.0, 2.0, 6.0]),
      [2, 3],
      DType.float64,
    );

    // Row vector of shape [1, 3]
    final rowVec = NDArray.fromList(Float64List.fromList([3.0, 2.0, 4.0]), [
      1,
      3,
    ], DType.float64);

    // Perform broadcasting comparison: mat > rowVec
    // rowVec will be stretched vertically to compare each row!
    final mask = mat > rowVec;
    print('mat shape: ${mat.shape}');
    print('rowVec shape: ${rowVec.shape}');
    print('mat > rowVec mask shape: ${mask.shape}');
    // [1.0>3.0, 5.0>2.0, 2.0>4.0] -> [false, true, false]
    // [4.0>3.0, 2.0>2.0, 6.0>4.0] -> [true, false, true]
    print('mat > rowVec mask data: ${mask.toList()}');
  });
}

void runLogicalOperationsExample() {
  NDArray.scope(() {
    print('\n--- Logical Operations (Combining Boolean Masks) ---');
    final a = NDArray.fromList(
      Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0]),
      [5],
      DType.float64,
    );

    // Create two boolean masks
    final maskGT2 = a > 2.0;
    final maskLT5 = a < 5.0;

    // Combine them element-wise using logicalAnd
    final combined = logicalAnd(maskGT2, maskLT5);
    print('a: ${a.toList()}');
    print('a > 2.0: ${maskGT2.toList()}');
    print('a < 5.0: ${maskLT5.toList()}');
    print('logicalAnd(maskGT2, maskLT5): ${combined.toList()}');
  });
}

void runLogicalReductionsExample() {
  NDArray.scope(() {
    print('\n--- Logical Reductions (all, any) ---');
    final a = NDArray.fromList([true, true, false], [3], DType.boolean);
    print('a: ${a.toList()}');
    print('all(a): ${all(a).scalar}'); // false
    print('any(a): ${any(a).scalar}'); // true

    final mat = NDArray.fromList(
      [true, true, false, true, false, false],
      [2, 3],
      DType.boolean,
    );
    print('2D matrix:\n$mat');
    print(
      'all(mat, axis: 0): ${all(mat, axis: 0).toList()}',
    ); // [true, false, false]
  });
}

void runAngleConvertersExample() {
  NDArray.scope(() {
    print('\n--- Angle Converters (deg2rad, rad2deg) ---');
    final deg = NDArray.fromList([180.0, 90.0, 45.0], [3], DType.float64);
    final rad = deg2rad(deg);
    print('Degrees: ${deg.toList()}');
    print('Radians: ${rad.toList()}'); // [pi, pi/2, pi/4]

    final back = rad2deg(rad);
    print('Radians: ${rad.toList()}');
    print('Back to Degrees: ${back.toList()}'); // [180, 90, 45]
  });
}

void runEnumerateAndComplexComponentsExample() {
  NDArray.scope(() {
    print('\n--- Multidimensional Enumerator (ndenumerate) ---');
    final a = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int32);
    for (final entry in ndenumerate(a)) {
      print('  Coordinate: ${entry.$1}, Value: ${entry.$2}');
    }

    print('\n--- Complex Components Extractors (real, imag) ---');
    final c = NDArray<AnySpec>.create([2], DType.complex128);
    c.setCell([0], Complex(3.0, 4.0));
    c.setCell([1], Complex(-1.0, 0.0));

    final r = real(c);
    final im = imag(c);

    print('Complex array: ${c.toList()}');
    print('Real component (Float64): ${r.toList()}'); // [3.0, -1.0]
    print('Imaginary component (Float64): ${im.toList()}'); // [4.0, 0.0]

    print('\n--- Zero-Copy view Demonstration ---');
    final realArr = NDArray.fromList([1.5, 2.5], [2], DType.float64);
    final realView = real(realArr); // Zero-copy view!
    print('Original Array: ${realArr.toList()}');
    print('Real View: ${realView.toList()}');

    // Mutating view updates parent automatically!
    realView.setCell([0], 99.9);
    print('Mutated Real View: ${realView.toList()}');
    print('Propagated Parent Array: ${realArr.toList()}');
  });
}
