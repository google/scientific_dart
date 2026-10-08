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

/// Zero-dependency static-type probe tests for `package:ndarray_ma` (R5).
///
/// Uses `Type staticTypeOf<T>(T Function() f) => T` so that exact static types
/// inferred by the Dart compiler are verified without adding `package:analyzer`
/// to `pkgs/ndarray_ma/pubspec.yaml`.
library;

import 'package:ndarray/ndarray.dart'
    show
        AnySpec,
        Boolean,
        Complex,
        Complex128,
        Complex64,
        DTypeTag,
        Float16,
        Float32,
        Float64,
        Int32,
        Int64,
        IntegerDType,
        Uint64,
        Uint8;
import 'package:ndarray_ma/ndarray_ma.dart';
import 'package:test/test.dart';

/// Captures the exact static return type `T` inferred by the Dart compiler for
/// a closure `f` without executing `f`.
Type staticTypeOf<T>(T Function() f) => T;

/// Returns the runtime representation of the static type `T`.
Type typeOf<T>() => T;

late final MaskedArray<Float32> _m32;
late final MaskedArray<Int64> _mi64;
late final MaskedArray<Int32> _mi32;
late final MaskedArray<Uint8> _mu8;
late final MaskedArray<Boolean> _mb;
late final MaskedArray<Float64> _m64;
late final MaskedArray<Float16> _m16;
late final MaskedArray<Complex64> _mc64;
late final MaskedArray<DTypeTag> _mdyn;
late final MaskedArray<AnySpec> _manySpec;

void main() {
  group('ndarray_ma static type probes (staticTypeOf)', () {
    test('MaskedArray<Float32>', () {
      expect(
        staticTypeOf(() => sum(_m32)),
        equals(typeOf<MaskedArray<Float32>>()),
      );
      expect(
        staticTypeOf(() => _m32.sum()),
        equals(typeOf<MaskedArray<Float32>>()),
      );
      expect(
        staticTypeOf(() => prod(_m32)),
        equals(typeOf<MaskedArray<Float32>>()),
      );
      expect(
        staticTypeOf(() => _m32.prod()),
        equals(typeOf<MaskedArray<Float32>>()),
      );
      expect(
        staticTypeOf(() => mean(_m32)),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => _m32.mean()),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => divide(_m32, _m32)),
        equals(typeOf<MaskedArray<Float32>>()),
      );
      expect(
        staticTypeOf(() => _m32 / _m32),
        equals(typeOf<MaskedArray<Float32>>()),
      );
      expect(
        staticTypeOf(() => min(_m32)),
        equals(typeOf<MaskedArray<Float32>>()),
      );
      expect(
        staticTypeOf(() => _m32 + _m32),
        equals(typeOf<MaskedArray<Float32>>()),
      );
      expect(staticTypeOf(() => _m32.scalar), equals(typeOf<double?>()));
      expect(staticTypeOf(() => _m32.getCell([0])), equals(typeOf<double?>()));
    });

    test('MaskedArray<Int32>', () {
      expect(
        staticTypeOf(() => sum(_mi32)),
        equals(typeOf<MaskedArray<Int64>>()),
      );
      expect(
        staticTypeOf(() => _mi32.sum()),
        equals(typeOf<MaskedArray<Int64>>()),
      );
      expect(
        staticTypeOf(() => prod(_mi32)),
        equals(typeOf<MaskedArray<Int64>>()),
      );
      expect(
        staticTypeOf(() => _mi32.prod()),
        equals(typeOf<MaskedArray<Int64>>()),
      );
      expect(
        staticTypeOf(() => mean(_mi32)),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => _mi32.mean()),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => divide(_mi32, _mi32)),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => _mi32 / _mi32),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => min(_mi32)),
        equals(typeOf<MaskedArray<Int32>>()),
      );
      expect(
        staticTypeOf(() => _mi32 + _mi32),
        equals(typeOf<MaskedArray<Int32>>()),
      );
      expect(staticTypeOf(() => _mi32.scalar), equals(typeOf<int?>()));
      expect(staticTypeOf(() => _mi32.getCell([0])), equals(typeOf<int?>()));
    });

    test('MaskedArray<Uint8>', () {
      expect(
        staticTypeOf(() => sum(_mu8)),
        equals(typeOf<MaskedArray<Uint64>>()),
      );
      expect(
        staticTypeOf(() => _mu8.sum()),
        equals(typeOf<MaskedArray<Uint64>>()),
      );
      expect(
        staticTypeOf(() => prod(_mu8)),
        equals(typeOf<MaskedArray<Uint64>>()),
      );
      expect(
        staticTypeOf(() => _mu8.prod()),
        equals(typeOf<MaskedArray<Uint64>>()),
      );
      expect(
        staticTypeOf(() => mean(_mu8)),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => _mu8.mean()),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => divide(_mu8, _mu8)),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => _mu8 / _mu8),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => min(_mu8)),
        equals(typeOf<MaskedArray<Uint8>>()),
      );
      expect(
        staticTypeOf(() => _mu8 + _mu8),
        equals(typeOf<MaskedArray<Uint8>>()),
      );
      expect(staticTypeOf(() => _mu8.scalar), equals(typeOf<int?>()));
      expect(staticTypeOf(() => _mu8.getCell([0])), equals(typeOf<int?>()));
    });

    test('MaskedArray<Boolean>', () {
      expect(
        staticTypeOf(() => sum(_mb)),
        equals(typeOf<MaskedArray<Int64>>()),
      );
      expect(
        staticTypeOf(() => _mb.sum()),
        equals(typeOf<MaskedArray<Int64>>()),
      );
      expect(
        staticTypeOf(() => prod(_mb)),
        equals(typeOf<MaskedArray<Int64>>()),
      );
      expect(
        staticTypeOf(() => _mb.prod()),
        equals(typeOf<MaskedArray<Int64>>()),
      );
      expect(
        staticTypeOf(() => mean(_mb)),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => _mb.mean()),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => divide(_mb, _mb)),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => _mb / _mb),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => min(_mb)),
        equals(typeOf<MaskedArray<Boolean>>()),
      );
      expect(
        staticTypeOf(() => _mb + _mb),
        equals(typeOf<MaskedArray<Boolean>>()),
      );
      expect(staticTypeOf(() => _mb.scalar), equals(typeOf<bool?>()));
      expect(staticTypeOf(() => _mb.getCell([0])), equals(typeOf<bool?>()));
    });

    test('MaskedArray<Float64>', () {
      expect(
        staticTypeOf(() => sum(_m64)),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => _m64.sum()),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => prod(_m64)),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => _m64.prod()),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => mean(_m64)),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => _m64.mean()),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => divide(_m64, _m64)),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => _m64 / _m64),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => min(_m64)),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => _m64 + _m64),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(staticTypeOf(() => _m64.scalar), equals(typeOf<double?>()));
      expect(staticTypeOf(() => _m64.getCell([0])), equals(typeOf<double?>()));
    });

    test('MaskedArray<Float16>', () {
      expect(
        staticTypeOf(() => sum(_m16)),
        equals(typeOf<MaskedArray<Float16>>()),
      );
      expect(
        staticTypeOf(() => _m16.sum()),
        equals(typeOf<MaskedArray<Float16>>()),
      );
      expect(
        staticTypeOf(() => prod(_m16)),
        equals(typeOf<MaskedArray<Float16>>()),
      );
      expect(
        staticTypeOf(() => _m16.prod()),
        equals(typeOf<MaskedArray<Float16>>()),
      );
      expect(
        staticTypeOf(() => mean(_m16)),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => _m16.mean()),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => divide(_m16, _m16)),
        equals(typeOf<MaskedArray<Float16>>()),
      );
      expect(
        staticTypeOf(() => _m16 / _m16),
        equals(typeOf<MaskedArray<Float16>>()),
      );
      expect(
        staticTypeOf(() => min(_m16)),
        equals(typeOf<MaskedArray<Float16>>()),
      );
      expect(
        staticTypeOf(() => _m16 + _m16),
        equals(typeOf<MaskedArray<Float16>>()),
      );
      expect(staticTypeOf(() => _m16.scalar), equals(typeOf<double?>()));
      expect(staticTypeOf(() => _m16.getCell([0])), equals(typeOf<double?>()));
    });

    test('MaskedArray<Complex64>', () {
      expect(
        staticTypeOf(() => sum(_mc64)),
        equals(typeOf<MaskedArray<Complex64>>()),
      );
      expect(
        staticTypeOf(() => _mc64.sum()),
        equals(typeOf<MaskedArray<Complex64>>()),
      );
      expect(
        staticTypeOf(() => prod(_mc64)),
        equals(typeOf<MaskedArray<Complex64>>()),
      );
      expect(
        staticTypeOf(() => _mc64.prod()),
        equals(typeOf<MaskedArray<Complex64>>()),
      );
      expect(
        staticTypeOf(() => mean(_mc64)),
        equals(typeOf<MaskedArray<Complex128>>()),
      );
      expect(
        staticTypeOf(() => _mc64.mean()),
        equals(typeOf<MaskedArray<Complex128>>()),
      );
      expect(
        staticTypeOf(() => divide(_mc64, _mc64)),
        equals(typeOf<MaskedArray<Complex64>>()),
      );
      expect(
        staticTypeOf(() => _mc64 / _mc64),
        equals(typeOf<MaskedArray<Complex64>>()),
      );
      expect(
        staticTypeOf(() => min(_mc64)),
        equals(typeOf<MaskedArray<Complex64>>()),
      );
      expect(
        staticTypeOf(() => _mc64 + _mc64),
        equals(typeOf<MaskedArray<Complex64>>()),
      );
      expect(staticTypeOf(() => _mc64.scalar), equals(typeOf<Complex?>()));
      expect(
        staticTypeOf(() => _mc64.getCell([0])),
        equals(typeOf<Complex?>()),
      );
    });

    test('MaskedArray<DTypeTag> fallback extensions', () {
      expect(
        staticTypeOf(() => _mdyn.sum()),
        equals(typeOf<MaskedArray<DTypeTag>>()),
      );
      expect(
        staticTypeOf(() => _mdyn.prod()),
        equals(typeOf<MaskedArray<DTypeTag>>()),
      );
      expect(
        staticTypeOf(() => _mdyn.mean()),
        equals(typeOf<MaskedArray<DTypeTag>>()),
      );
      expect(
        staticTypeOf(() => _mdyn / _mdyn),
        equals(typeOf<MaskedArray<DTypeTag>>()),
      );
      expect(
        staticTypeOf(() => min(_mdyn)),
        equals(typeOf<MaskedArray<DTypeTag>>()),
      );
      expect(
        staticTypeOf(() => _mdyn + _mdyn),
        equals(typeOf<MaskedArray<DTypeTag>>()),
      );
      expect(staticTypeOf(() => _mdyn.scalar), equals(typeOf<dynamic>()));
      expect(staticTypeOf(() => _mdyn.getCell([0])), equals(typeOf<dynamic>()));
    });

    test('MaskedArray<AnySpec>', () {
      expect(
        staticTypeOf(() => sum(_manySpec)),
        equals(typeOf<MaskedArray<DTypeTag>>()),
      );
      expect(
        staticTypeOf(() => _manySpec.sum()),
        equals(typeOf<MaskedArray<DTypeTag>>()),
      );
      expect(
        staticTypeOf(() => prod(_manySpec)),
        equals(typeOf<MaskedArray<DTypeTag>>()),
      );
      expect(
        staticTypeOf(() => _manySpec.prod()),
        equals(typeOf<MaskedArray<DTypeTag>>()),
      );
      expect(
        staticTypeOf(() => mean(_manySpec)),
        equals(typeOf<MaskedArray<DTypeTag>>()),
      );
      expect(
        staticTypeOf(() => _manySpec.mean()),
        equals(typeOf<MaskedArray<DTypeTag>>()),
      );
      expect(
        staticTypeOf(() => divide(_manySpec, _manySpec)),
        equals(typeOf<MaskedArray<DTypeTag>>()),
      );
      expect(
        staticTypeOf(() => _manySpec / _manySpec),
        equals(typeOf<MaskedArray<DTypeTag>>()),
      );
      expect(
        staticTypeOf(() => min(_manySpec)),
        equals(typeOf<MaskedArray<AnySpec>>()),
      );
      expect(
        staticTypeOf(() => _manySpec + _manySpec),
        equals(typeOf<MaskedArray<AnySpec>>()),
      );
      expect(staticTypeOf(() => _manySpec.scalar), equals(typeOf<dynamic>()));
      expect(
        staticTypeOf(() => _manySpec.getCell([0])),
        equals(typeOf<dynamic>()),
      );
    });

    test('MaskedArray integer LUB [mi64, mi32].first', () {
      expect(
        staticTypeOf(() => [_mi64, _mi32].first),
        equals(typeOf<MaskedArray<IntegerDType>>()),
      );
      expect(
        staticTypeOf(() => mean([_mi64, _mi32].first)),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => [_mi64, _mi32].first.mean()),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => divide([_mi64, _mi32].first, [_mi64, _mi32].first)),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => [_mi64, _mi32].first / [_mi64, _mi32].first),
        equals(typeOf<MaskedArray<Float64>>()),
      );
      expect(
        staticTypeOf(() => [_mi64, _mi32].first.scalar),
        equals(typeOf<int?>()),
      );
      expect(
        staticTypeOf(() => [_mi64, _mi32].first.getCell([0])),
        equals(typeOf<int?>()),
      );
    });
  });
}
