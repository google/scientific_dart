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

/// Compile-time projection table: 15 dtype tags × 8 projections = 120 cells.
///
/// Every cell asserts the *exact* dtype (or element type) that one concrete
/// [DTypeTag] projects to through one of the eight single-slot interfaces
/// ([RealOf], [ElementOf], [RealFloatOf], [ComplexOf], [InexactOf],
/// [AccumulatorOf], [DoublePrecisionOf], [DivideOf]). The assertion is made
/// by the type checker: a wrong row in `lib/src/ndarray.dart` makes
/// `dart analyze` (and the test compile) fail at the offending cell with
/// both `type_argument_not_matching_bounds` (on the `isSubtype` witness) and
/// `argument_type_not_assignable` (on the invariant `mustBe` witness), naming
/// both the row's actual projection and the expected one.
///
/// How a cell is exact rather than merely "assignable":
///
/// 1. `isSubtype<Tag, XOf<Expected>>()` checks the direct subtype bound
///    `Tag <: XOf<Expected>`, failing with `type_argument_not_matching_bounds`
///    whenever the row's entry disagrees with `Expected`.
/// 2. `inexactOf(DType.boolean)` infers its result type parameter `R` solely
///    from the tag's superinterface (`Boolean <: InexactOf<R>` solves to
///    `R = Float64`). The call sits in receiver position, which gives the
///    inference no context type, so nothing written in the cell can widen
///    `R`.
/// 3. `Projection<R>.mustBe` takes an `Invariant<R>` (`R Function(R)`), and
///    function types are invariant in that position. `exactly<Float64>` is
///    assignable only when `R` *is* `Float64`; a supertype (`DTypeTag`,
///    `Object`, `dynamic`) is rejected as surely as a wrong tag or `Never`.
///
/// Rules that keep the cells honest (checked by the last two tests):
///
/// * Never pass explicit type arguments `<Boolean, DTypeTag>` to a projection
///   function; that turns the bound into a one-sided subtype check that a
///   wider expectation satisfies vacuously.
/// * Always spell the expectation as `mustBe(exactly<X>)`; a bare `exactly`
///   or a lambda would be instantiated from the parameter type and assert
///   nothing.
///
/// The rows below are copied from the `DTypeSpec<…>` declarations in
/// `lib/src/ndarray.dart` (slot order: RealTag, Element, RealFloatTag,
/// ComplexTag, InexactTag, AccumulatorTag, DoublePrecisionTag, DivideTag).
/// They are the library's contract, not NumPy's: half floats promote to
/// `Float64` for transcendental math but keep their own tag for sums and
/// true division.
library;

import 'dart:io';

import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

/// A type position in which `T` is invariant.
///
/// `Invariant<A>` is assignable to `Invariant<B>` only if `A` and `B` are
/// mutual subtypes, i.e. the same type (top types such as `dynamic` and
/// `Object?` are equivalent to each other, which is irrelevant for the tags).
typedef Invariant<T> = T Function(T);

/// Produces an [Invariant] witness for `T` when instantiated as a tear-off:
/// `exactly<Float64>` has the static type `Float64 Function(Float64)`.
T exactly<T>(T value) => value;

/// Every `mustBe` call records which tag was checked through which interface
/// so that the bookkeeping test can prove all 120 cells executed exactly once.
final List<({String interface, Type tag})> _checked = [];

/// The projection of a dtype tag through one interface, carrying the inferred
/// result type `R`.
final class Projection<R> {
  Projection._(this.interface, this.tag);

  /// Name of the interface the tag was projected through, e.g. `'InexactOf'`.
  final String interface;

  /// The tag type that was projected (as a runtime [Type], for bookkeeping).
  final Type tag;

  /// Compiles only if [expected] carries exactly `R`.
  ///
  /// Call it as `mustBe(exactly<X>)`. The argument is otherwise unused; the
  /// assertion is the type check of the argument against `Invariant<R>`.
  void mustBe(Invariant<R> expected) {
    _checked.add((interface: interface, tag: tag));
  }
}

/// Projects `T` through [RealOf]; `R` is inferred from `T <: RealOf<R>`.
Projection<R> realOf<T extends RealOf<R>, R extends DTypeTag>(DType<T> dtype) =>
    Projection<R>._('RealOf', T);

/// Projects `T` through [ElementOf]; `E` is inferred from `T <: ElementOf<E>`.
Projection<E> elementOf<T extends ElementOf<E>, E>(DType<T> dtype) =>
    Projection<E>._('ElementOf', T);

/// Projects `T` through [RealFloatOf].
Projection<R> realFloatOf<T extends RealFloatOf<R>, R extends DTypeTag>(
  DType<T> dtype,
) => Projection<R>._('RealFloatOf', T);

/// Projects `T` through [ComplexOf].
Projection<R> complexOf<T extends ComplexOf<R>, R extends DTypeTag>(
  DType<T> dtype,
) => Projection<R>._('ComplexOf', T);

/// Projects `T` through [InexactOf].
Projection<R> inexactOf<T extends InexactOf<R>, R extends DTypeTag>(
  DType<T> dtype,
) => Projection<R>._('InexactOf', T);

/// Projects `T` through [AccumulatorOf].
Projection<R> accumulatorOf<T extends AccumulatorOf<R>, R extends DTypeTag>(
  DType<T> dtype,
) => Projection<R>._('AccumulatorOf', T);

/// Projects `T` through [DoublePrecisionOf].
Projection<R> doublePrecisionOf<
  T extends DoublePrecisionOf<R>,
  R extends DTypeTag
>(DType<T> dtype) => Projection<R>._('DoublePrecisionOf', T);

/// Projects `T` through [DivideOf].
Projection<R> divideOf<T extends DivideOf<R>, R extends DTypeTag>(
  DType<T> dtype,
) => Projection<R>._('DivideOf', T);

/// Compiles only if `Sub` is a subtype of `Super`.
void isSubtype<Sub extends Super, Super>() {}

/// The eight interfaces, in `DTypeSpec` slot order.
const List<String> _interfaces = [
  'RealOf',
  'ElementOf',
  'RealFloatOf',
  'ComplexOf',
  'InexactOf',
  'AccumulatorOf',
  'DoublePrecisionOf',
  'DivideOf',
];

/// The 15 concrete tags, in `DType` declaration order.
const List<Type> _tags = [
  Float64,
  Float32,
  Float16,
  BFloat16,
  Int64,
  Int32,
  Int16,
  Int8,
  Uint64,
  Uint32,
  Uint16,
  Uint8,
  Complex128,
  Complex64,
  Boolean,
];

/// Number of guard cells outside the 15 × 8 table (markers and `AnySpec`).
const int _guardCells = 6 + 5 + 8;

void main() {
  group('projection table (15 dtypes × 8 projections, compile-time)', () {
    // Slot order: RealTag, Element, RealFloatTag, ComplexTag, InexactTag,
    //             AccumulatorTag, DoublePrecisionTag, DivideTag.

    test('Float64', () {
      const t = DType.float64;
      // DTypeSpec<Float64, double, Float64, Complex128, Float64, Float64, Float64, Float64>
      isSubtype<Float64, RealOf<Float64>>();
      isSubtype<Float64, ElementOf<double>>();
      isSubtype<Float64, RealFloatOf<Float64>>();
      isSubtype<Float64, ComplexOf<Complex128>>();
      isSubtype<Float64, InexactOf<Float64>>();
      isSubtype<Float64, AccumulatorOf<Float64>>();
      isSubtype<Float64, DoublePrecisionOf<Float64>>();
      isSubtype<Float64, DivideOf<Float64>>();
      realOf(t).mustBe(exactly<Float64>);
      elementOf(t).mustBe(exactly<double>);
      realFloatOf(t).mustBe(exactly<Float64>);
      complexOf(t).mustBe(exactly<Complex128>);
      inexactOf(t).mustBe(exactly<Float64>);
      accumulatorOf(t).mustBe(exactly<Float64>);
      doublePrecisionOf(t).mustBe(exactly<Float64>);
      divideOf(t).mustBe(exactly<Float64>);
    });

    test('Float32', () {
      const t = DType.float32;
      // DTypeSpec<Float32, double, Float32, Complex64, Float32, Float32, Float64, Float32>
      isSubtype<Float32, RealOf<Float32>>();
      isSubtype<Float32, ElementOf<double>>();
      isSubtype<Float32, RealFloatOf<Float32>>();
      isSubtype<Float32, ComplexOf<Complex64>>();
      isSubtype<Float32, InexactOf<Float32>>();
      isSubtype<Float32, AccumulatorOf<Float32>>();
      isSubtype<Float32, DoublePrecisionOf<Float64>>();
      isSubtype<Float32, DivideOf<Float32>>();
      realOf(t).mustBe(exactly<Float32>);
      elementOf(t).mustBe(exactly<double>);
      realFloatOf(t).mustBe(exactly<Float32>);
      complexOf(t).mustBe(exactly<Complex64>);
      inexactOf(t).mustBe(exactly<Float32>);
      accumulatorOf(t).mustBe(exactly<Float32>);
      doublePrecisionOf(t).mustBe(exactly<Float64>);
      divideOf(t).mustBe(exactly<Float32>);
    });

    test('Float16', () {
      const t = DType.float16;
      // DTypeSpec<Float16, double, Float64, Complex128, Float64, Float16, Float64, Float16>
      isSubtype<Float16, RealOf<Float16>>();
      isSubtype<Float16, ElementOf<double>>();
      isSubtype<Float16, RealFloatOf<Float64>>();
      isSubtype<Float16, ComplexOf<Complex128>>();
      isSubtype<Float16, InexactOf<Float64>>();
      isSubtype<Float16, AccumulatorOf<Float16>>();
      isSubtype<Float16, DoublePrecisionOf<Float64>>();
      isSubtype<Float16, DivideOf<Float16>>();
      realOf(t).mustBe(exactly<Float16>);
      elementOf(t).mustBe(exactly<double>);
      realFloatOf(t).mustBe(exactly<Float64>);
      complexOf(t).mustBe(exactly<Complex128>);
      inexactOf(t).mustBe(exactly<Float64>);
      accumulatorOf(t).mustBe(exactly<Float16>);
      doublePrecisionOf(t).mustBe(exactly<Float64>);
      divideOf(t).mustBe(exactly<Float16>);
    });

    test('BFloat16', () {
      const t = DType.bfloat16;
      // DTypeSpec<BFloat16, double, Float64, Complex128, Float64, BFloat16, Float64, BFloat16>
      isSubtype<BFloat16, RealOf<BFloat16>>();
      isSubtype<BFloat16, ElementOf<double>>();
      isSubtype<BFloat16, RealFloatOf<Float64>>();
      isSubtype<BFloat16, ComplexOf<Complex128>>();
      isSubtype<BFloat16, InexactOf<Float64>>();
      isSubtype<BFloat16, AccumulatorOf<BFloat16>>();
      isSubtype<BFloat16, DoublePrecisionOf<Float64>>();
      isSubtype<BFloat16, DivideOf<BFloat16>>();
      realOf(t).mustBe(exactly<BFloat16>);
      elementOf(t).mustBe(exactly<double>);
      realFloatOf(t).mustBe(exactly<Float64>);
      complexOf(t).mustBe(exactly<Complex128>);
      inexactOf(t).mustBe(exactly<Float64>);
      accumulatorOf(t).mustBe(exactly<BFloat16>);
      doublePrecisionOf(t).mustBe(exactly<Float64>);
      divideOf(t).mustBe(exactly<BFloat16>);
    });

    test('Int64', () {
      const t = DType.int64;
      // DTypeSpec<Int64, int, Float64, Complex128, Float64, Int64, Float64, Float64>
      isSubtype<Int64, RealOf<Int64>>();
      isSubtype<Int64, ElementOf<int>>();
      isSubtype<Int64, RealFloatOf<Float64>>();
      isSubtype<Int64, ComplexOf<Complex128>>();
      isSubtype<Int64, InexactOf<Float64>>();
      isSubtype<Int64, AccumulatorOf<Int64>>();
      isSubtype<Int64, DoublePrecisionOf<Float64>>();
      isSubtype<Int64, DivideOf<Float64>>();
      realOf(t).mustBe(exactly<Int64>);
      elementOf(t).mustBe(exactly<int>);
      realFloatOf(t).mustBe(exactly<Float64>);
      complexOf(t).mustBe(exactly<Complex128>);
      inexactOf(t).mustBe(exactly<Float64>);
      accumulatorOf(t).mustBe(exactly<Int64>);
      doublePrecisionOf(t).mustBe(exactly<Float64>);
      divideOf(t).mustBe(exactly<Float64>);
    });

    test('Int32', () {
      const t = DType.int32;
      // DTypeSpec<Int32, int, Float64, Complex128, Float64, Int64, Float64, Float64>
      isSubtype<Int32, RealOf<Int32>>();
      isSubtype<Int32, ElementOf<int>>();
      isSubtype<Int32, RealFloatOf<Float64>>();
      isSubtype<Int32, ComplexOf<Complex128>>();
      isSubtype<Int32, InexactOf<Float64>>();
      isSubtype<Int32, AccumulatorOf<Int64>>();
      isSubtype<Int32, DoublePrecisionOf<Float64>>();
      isSubtype<Int32, DivideOf<Float64>>();
      realOf(t).mustBe(exactly<Int32>);
      elementOf(t).mustBe(exactly<int>);
      realFloatOf(t).mustBe(exactly<Float64>);
      complexOf(t).mustBe(exactly<Complex128>);
      inexactOf(t).mustBe(exactly<Float64>);
      accumulatorOf(t).mustBe(exactly<Int64>);
      doublePrecisionOf(t).mustBe(exactly<Float64>);
      divideOf(t).mustBe(exactly<Float64>);
    });

    test('Int16', () {
      const t = DType.int16;
      // DTypeSpec<Int16, int, Float64, Complex128, Float64, Int64, Float64, Float64>
      isSubtype<Int16, RealOf<Int16>>();
      isSubtype<Int16, ElementOf<int>>();
      isSubtype<Int16, RealFloatOf<Float64>>();
      isSubtype<Int16, ComplexOf<Complex128>>();
      isSubtype<Int16, InexactOf<Float64>>();
      isSubtype<Int16, AccumulatorOf<Int64>>();
      isSubtype<Int16, DoublePrecisionOf<Float64>>();
      isSubtype<Int16, DivideOf<Float64>>();
      realOf(t).mustBe(exactly<Int16>);
      elementOf(t).mustBe(exactly<int>);
      realFloatOf(t).mustBe(exactly<Float64>);
      complexOf(t).mustBe(exactly<Complex128>);
      inexactOf(t).mustBe(exactly<Float64>);
      accumulatorOf(t).mustBe(exactly<Int64>);
      doublePrecisionOf(t).mustBe(exactly<Float64>);
      divideOf(t).mustBe(exactly<Float64>);
    });

    test('Int8', () {
      const t = DType.int8;
      // DTypeSpec<Int8, int, Float64, Complex128, Float64, Int64, Float64, Float64>
      isSubtype<Int8, RealOf<Int8>>();
      isSubtype<Int8, ElementOf<int>>();
      isSubtype<Int8, RealFloatOf<Float64>>();
      isSubtype<Int8, ComplexOf<Complex128>>();
      isSubtype<Int8, InexactOf<Float64>>();
      isSubtype<Int8, AccumulatorOf<Int64>>();
      isSubtype<Int8, DoublePrecisionOf<Float64>>();
      isSubtype<Int8, DivideOf<Float64>>();
      realOf(t).mustBe(exactly<Int8>);
      elementOf(t).mustBe(exactly<int>);
      realFloatOf(t).mustBe(exactly<Float64>);
      complexOf(t).mustBe(exactly<Complex128>);
      inexactOf(t).mustBe(exactly<Float64>);
      accumulatorOf(t).mustBe(exactly<Int64>);
      doublePrecisionOf(t).mustBe(exactly<Float64>);
      divideOf(t).mustBe(exactly<Float64>);
    });

    test('Uint64', () {
      const t = DType.uint64;
      // DTypeSpec<Uint64, int, Float64, Complex128, Float64, Uint64, Float64, Float64>
      isSubtype<Uint64, RealOf<Uint64>>();
      isSubtype<Uint64, ElementOf<int>>();
      isSubtype<Uint64, RealFloatOf<Float64>>();
      isSubtype<Uint64, ComplexOf<Complex128>>();
      isSubtype<Uint64, InexactOf<Float64>>();
      isSubtype<Uint64, AccumulatorOf<Uint64>>();
      isSubtype<Uint64, DoublePrecisionOf<Float64>>();
      isSubtype<Uint64, DivideOf<Float64>>();
      realOf(t).mustBe(exactly<Uint64>);
      elementOf(t).mustBe(exactly<int>);
      realFloatOf(t).mustBe(exactly<Float64>);
      complexOf(t).mustBe(exactly<Complex128>);
      inexactOf(t).mustBe(exactly<Float64>);
      accumulatorOf(t).mustBe(exactly<Uint64>);
      doublePrecisionOf(t).mustBe(exactly<Float64>);
      divideOf(t).mustBe(exactly<Float64>);
    });

    test('Uint32', () {
      const t = DType.uint32;
      // DTypeSpec<Uint32, int, Float64, Complex128, Float64, Uint64, Float64, Float64>
      isSubtype<Uint32, RealOf<Uint32>>();
      isSubtype<Uint32, ElementOf<int>>();
      isSubtype<Uint32, RealFloatOf<Float64>>();
      isSubtype<Uint32, ComplexOf<Complex128>>();
      isSubtype<Uint32, InexactOf<Float64>>();
      isSubtype<Uint32, AccumulatorOf<Uint64>>();
      isSubtype<Uint32, DoublePrecisionOf<Float64>>();
      isSubtype<Uint32, DivideOf<Float64>>();
      realOf(t).mustBe(exactly<Uint32>);
      elementOf(t).mustBe(exactly<int>);
      realFloatOf(t).mustBe(exactly<Float64>);
      complexOf(t).mustBe(exactly<Complex128>);
      inexactOf(t).mustBe(exactly<Float64>);
      accumulatorOf(t).mustBe(exactly<Uint64>);
      doublePrecisionOf(t).mustBe(exactly<Float64>);
      divideOf(t).mustBe(exactly<Float64>);
    });

    test('Uint16', () {
      const t = DType.uint16;
      // DTypeSpec<Uint16, int, Float64, Complex128, Float64, Uint64, Float64, Float64>
      isSubtype<Uint16, RealOf<Uint16>>();
      isSubtype<Uint16, ElementOf<int>>();
      isSubtype<Uint16, RealFloatOf<Float64>>();
      isSubtype<Uint16, ComplexOf<Complex128>>();
      isSubtype<Uint16, InexactOf<Float64>>();
      isSubtype<Uint16, AccumulatorOf<Uint64>>();
      isSubtype<Uint16, DoublePrecisionOf<Float64>>();
      isSubtype<Uint16, DivideOf<Float64>>();
      realOf(t).mustBe(exactly<Uint16>);
      elementOf(t).mustBe(exactly<int>);
      realFloatOf(t).mustBe(exactly<Float64>);
      complexOf(t).mustBe(exactly<Complex128>);
      inexactOf(t).mustBe(exactly<Float64>);
      accumulatorOf(t).mustBe(exactly<Uint64>);
      doublePrecisionOf(t).mustBe(exactly<Float64>);
      divideOf(t).mustBe(exactly<Float64>);
    });

    test('Uint8', () {
      const t = DType.uint8;
      // DTypeSpec<Uint8, int, Float64, Complex128, Float64, Uint64, Float64, Float64>
      isSubtype<Uint8, RealOf<Uint8>>();
      isSubtype<Uint8, ElementOf<int>>();
      isSubtype<Uint8, RealFloatOf<Float64>>();
      isSubtype<Uint8, ComplexOf<Complex128>>();
      isSubtype<Uint8, InexactOf<Float64>>();
      isSubtype<Uint8, AccumulatorOf<Uint64>>();
      isSubtype<Uint8, DoublePrecisionOf<Float64>>();
      isSubtype<Uint8, DivideOf<Float64>>();
      realOf(t).mustBe(exactly<Uint8>);
      elementOf(t).mustBe(exactly<int>);
      realFloatOf(t).mustBe(exactly<Float64>);
      complexOf(t).mustBe(exactly<Complex128>);
      inexactOf(t).mustBe(exactly<Float64>);
      accumulatorOf(t).mustBe(exactly<Uint64>);
      doublePrecisionOf(t).mustBe(exactly<Float64>);
      divideOf(t).mustBe(exactly<Float64>);
    });

    test('Complex128', () {
      const t = DType.complex128;
      // DTypeSpec<Float64, Complex, Float64, Complex128, Complex128, Complex128, Complex128, Complex128>
      isSubtype<Complex128, RealOf<Float64>>();
      isSubtype<Complex128, ElementOf<Complex>>();
      isSubtype<Complex128, RealFloatOf<Float64>>();
      isSubtype<Complex128, ComplexOf<Complex128>>();
      isSubtype<Complex128, InexactOf<Complex128>>();
      isSubtype<Complex128, AccumulatorOf<Complex128>>();
      isSubtype<Complex128, DoublePrecisionOf<Complex128>>();
      isSubtype<Complex128, DivideOf<Complex128>>();
      realOf(t).mustBe(exactly<Float64>);
      elementOf(t).mustBe(exactly<Complex>);
      realFloatOf(t).mustBe(exactly<Float64>);
      complexOf(t).mustBe(exactly<Complex128>);
      inexactOf(t).mustBe(exactly<Complex128>);
      accumulatorOf(t).mustBe(exactly<Complex128>);
      doublePrecisionOf(t).mustBe(exactly<Complex128>);
      divideOf(t).mustBe(exactly<Complex128>);
    });

    test('Complex64', () {
      const t = DType.complex64;
      // DTypeSpec<Float32, Complex, Float32, Complex64, Complex64, Complex64, Complex128, Complex64>
      isSubtype<Complex64, RealOf<Float32>>();
      isSubtype<Complex64, ElementOf<Complex>>();
      isSubtype<Complex64, RealFloatOf<Float32>>();
      isSubtype<Complex64, ComplexOf<Complex64>>();
      isSubtype<Complex64, InexactOf<Complex64>>();
      isSubtype<Complex64, AccumulatorOf<Complex64>>();
      isSubtype<Complex64, DoublePrecisionOf<Complex128>>();
      isSubtype<Complex64, DivideOf<Complex64>>();
      realOf(t).mustBe(exactly<Float32>);
      elementOf(t).mustBe(exactly<Complex>);
      realFloatOf(t).mustBe(exactly<Float32>);
      complexOf(t).mustBe(exactly<Complex64>);
      inexactOf(t).mustBe(exactly<Complex64>);
      accumulatorOf(t).mustBe(exactly<Complex64>);
      doublePrecisionOf(t).mustBe(exactly<Complex128>);
      divideOf(t).mustBe(exactly<Complex64>);
    });

    test('Boolean', () {
      const t = DType.boolean;
      // DTypeSpec<Boolean, bool, Float64, Complex128, Float64, Int64, Float64, Float64>
      isSubtype<Boolean, RealOf<Boolean>>();
      isSubtype<Boolean, ElementOf<bool>>();
      isSubtype<Boolean, RealFloatOf<Float64>>();
      isSubtype<Boolean, ComplexOf<Complex128>>();
      isSubtype<Boolean, InexactOf<Float64>>();
      isSubtype<Boolean, AccumulatorOf<Int64>>();
      isSubtype<Boolean, DoublePrecisionOf<Float64>>();
      isSubtype<Boolean, DivideOf<Float64>>();
      realOf(t).mustBe(exactly<Boolean>);
      elementOf(t).mustBe(exactly<bool>);
      realFloatOf(t).mustBe(exactly<Float64>);
      complexOf(t).mustBe(exactly<Complex128>);
      inexactOf(t).mustBe(exactly<Float64>);
      accumulatorOf(t).mustBe(exactly<Int64>);
      doublePrecisionOf(t).mustBe(exactly<Float64>);
      divideOf(t).mustBe(exactly<Float64>);
    });
  });

  group('guards: capability markers (R2) and AnySpec', () {
    test('IntegerDType pins Element and the five constant projections', () {
      const DType<IntegerDType> t = DType.int32;
      elementOf(t).mustBe(exactly<int>);
      realFloatOf(t).mustBe(exactly<Float64>);
      complexOf(t).mustBe(exactly<Complex128>);
      inexactOf(t).mustBe(exactly<Float64>);
      doublePrecisionOf(t).mustBe(exactly<Float64>);
      divideOf(t).mustBe(exactly<Float64>);
    });

    test('BitwiseDType pins the five constant projections', () {
      const DType<BitwiseDType> t = DType.boolean;
      realFloatOf(t).mustBe(exactly<Float64>);
      complexOf(t).mustBe(exactly<Complex128>);
      inexactOf(t).mustBe(exactly<Float64>);
      doublePrecisionOf(t).mustBe(exactly<Float64>);
      divideOf(t).mustBe(exactly<Float64>);
    });

    test('AnySpec projects every slot to DTypeTag (Element: dynamic)', () {
      const DType<AnySpec> t = DType.float64;
      realOf(t).mustBe(exactly<DTypeTag>);
      elementOf(t).mustBe(exactly<dynamic>);
      realFloatOf(t).mustBe(exactly<DTypeTag>);
      complexOf(t).mustBe(exactly<DTypeTag>);
      inexactOf(t).mustBe(exactly<DTypeTag>);
      accumulatorOf(t).mustBe(exactly<DTypeTag>);
      doublePrecisionOf(t).mustBe(exactly<DTypeTag>);
      divideOf(t).mustBe(exactly<DTypeTag>);
    });

    test('the integer tags implement IntegerDType, Boolean BitwiseDType', () {
      isSubtype<Int64, IntegerDType>();
      isSubtype<Int32, IntegerDType>();
      isSubtype<Int16, IntegerDType>();
      isSubtype<Int8, IntegerDType>();
      isSubtype<Uint64, IntegerDType>();
      isSubtype<Uint32, IntegerDType>();
      isSubtype<Uint16, IntegerDType>();
      isSubtype<Uint8, IntegerDType>();
      isSubtype<Boolean, BitwiseDType>();
      isSubtype<IntegerDType, BitwiseDType>();
    });
  });

  group('bookkeeping', () {
    test('every one of the 120 table cells ran exactly once', () {
      final table = _checked.where((c) => _tags.contains(c.tag)).toList();
      expect(table, hasLength(120));
      for (final interface in _interfaces) {
        final tags = [
          for (final c in table)
            if (c.interface == interface) c.tag,
        ];
        expect(
          tags,
          unorderedEquals(_tags),
          reason: '$interface must be checked once for each of the 15 tags.',
        );
      }
      expect(_checked.length - table.length, _guardCells);
    });

    test('cells rely on inference and spell their expectation', () {
      final source = _thisFile().readAsStringSync();
      final code = source.replaceAll(RegExp(r'^\s*//.*$', multiLine: true), '');
      // Explicit type arguments on a projection function would replace
      // inference by a one-sided bound check.
      final explicitArguments = RegExp(
        r'\b[a-z]\w*Of<\s*\w+\s*,',
      ).allMatches(code).map((m) => m.group(0)).toList();
      expect(explicitArguments, isEmpty);
      // `.mustBe(exactly)` or `.mustBe((x) => x)` would be instantiated from
      // the parameter type and assert nothing.
      final vacuous = RegExp(
        r'\.mustBe\((?!exactly<)',
      ).allMatches(code).map((m) => m.group(0)).toList();
      expect(vacuous, isEmpty);
      final cells = RegExp(r'\.mustBe\(exactly<').allMatches(code).length;
      expect(cells, 120 + _guardCells);
      final subtypePairs = RegExp(
        r'isSubtype<(\w+),\s*(\w+)<([^>]+)>>',
      ).allMatches(code).map((m) => (m.group(1)!, m.group(2)!)).toList();
      expect(subtypePairs, hasLength(120));
      expect(subtypePairs.toSet(), hasLength(120));
    });
  });
}

/// This file, located from the package root (`dart test` runs with the
/// package directory as the working directory).
File _thisFile() {
  var dir = Directory.current;
  while (!File('${dir.path}/pubspec.yaml').existsSync()) {
    final parent = dir.parent;
    if (parent.path == dir.path) {
      fail('Could not locate pkgs/ndarray from ${Directory.current.path}.');
    }
    dir = parent;
  }
  return File('${dir.path}/test/meta/static_projection_table_test.dart');
}
