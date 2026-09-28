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

import 'package:meta/meta.dart';

import 'dart:math' as math;
import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'dart:collection';

import 'ndarray_bindings.dart';
import 'ndarray_extensions_bindings.dart';
import 'scratch_arena.dart';

import 'package:openblas/openblas.dart' show openblas_set_num_threads;
import 'package:resource_scope/resource_scope.dart';

import 'operations.dart' as ops;
import 'operations/helpers.dart' as helpers;

import 'cpu_check.dart';
import 'float16_utils.dart';
import 'sendable_ndarray.dart';

/// Root marker for all [NDArray] dtype tags.
sealed class DTypeTag {
  const DTypeTag();
}

/// Type-level specification of a concrete [DTypeTag].
///
/// Each of the 15 concrete tag classes (`Float64`, `Float32`, `Int32`, …)
/// extends [DTypeSpec] with its deterministic type-level counterparts so that
/// operations can infer concrete return types without explicit type arguments:
/// - [R]: the real/magnitude counterpart (`Float32` for `Complex64`;
///   `Float64` for `Complex128`; `Self` otherwise).
/// - [E]: the Dart element type (`double`, `int`, [Complex], `bool`).
/// - [F]: the real-float computation tag (`Float32` for `Float32`/`Complex64`;
///   `Float64` otherwise).
/// - [C]: the complex computation tag (`Complex64` for `Float32`/`Complex64`;
///   `Complex128` otherwise).
/// - [M]: the inexact/math-promoted tag (`Self` for `Float64`, `Float32`,
///   `Complex128`, `Complex64`; `Float64` for integers, booleans, and half
///   floats).
/// - [S]: the sum/product accumulation tag (`Int64` for `Boolean`; `Self`
///   otherwise).
/// - [CS]: the cumulative sum/product tag (`Int32` for `Boolean`; `Self`
///   otherwise).
sealed class DTypeSpec<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  CS extends DTypeTag
>
    extends DTypeTag {
  const DTypeSpec();
}

/// Wildcard [DTypeSpec] bound matching any [DTypeSpec] subtype.
typedef AnySpec =
    DTypeSpec<
      DTypeTag,
      dynamic,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag,
      DTypeTag
    >;

/// Tag for the `float64` dtype. Elements are `double`.
abstract final class Float64
    extends
        DTypeSpec<
          Float64,
          double,
          Float64,
          Complex128,
          Float64,
          Float64,
          Float64
        > {}

/// Tag for the `float32` dtype. Elements are `double`.
abstract final class Float32
    extends
        DTypeSpec<
          Float32,
          double,
          Float32,
          Complex64,
          Float32,
          Float32,
          Float32
        > {}

/// Tag for the `float16` dtype. Elements are `double`.
abstract final class Float16
    extends
        DTypeSpec<
          Float16,
          double,
          Float64,
          Complex128,
          Float64,
          Float16,
          Float16
        > {}

/// Tag for the `bfloat16` dtype. Elements are `double`.
abstract final class BFloat16
    extends
        DTypeSpec<
          BFloat16,
          double,
          Float64,
          Complex128,
          Float64,
          BFloat16,
          BFloat16
        > {}

/// Tag for the `int64` dtype. Elements are `int`.
abstract final class Int64
    extends DTypeSpec<Int64, int, Float64, Complex128, Float64, Int64, Int64> {}

/// Tag for the `int32` dtype. Elements are `int`.
abstract final class Int32
    extends DTypeSpec<Int32, int, Float64, Complex128, Float64, Int32, Int32> {}

/// Tag for the `int16` dtype. Elements are `int`.
abstract final class Int16
    extends DTypeSpec<Int16, int, Float64, Complex128, Float64, Int16, Int16> {}

/// Tag for the `int8` dtype. Elements are `int`.
abstract final class Int8
    extends DTypeSpec<Int8, int, Float64, Complex128, Float64, Int8, Int8> {}

/// Tag for the `uint64` dtype. Elements are `int`.
///
/// Dart `int` is signed 64-bit; bit patterns with the MSB set represent
/// negative values. Use [uint64Compare] for unsigned comparisons.
abstract final class Uint64
    extends
        DTypeSpec<Uint64, int, Float64, Complex128, Float64, Uint64, Uint64> {}

/// Tag for the `uint32` dtype. Elements are `int`.
abstract final class Uint32
    extends
        DTypeSpec<Uint32, int, Float64, Complex128, Float64, Uint32, Uint32> {}

/// Tag for the `uint16` dtype. Elements are `int`.
abstract final class Uint16
    extends
        DTypeSpec<Uint16, int, Float64, Complex128, Float64, Uint16, Uint16> {}

/// Tag for the `uint8` dtype. Elements are `int`.
abstract final class Uint8
    extends DTypeSpec<Uint8, int, Float64, Complex128, Float64, Uint8, Uint8> {}

/// Tag for the `complex64` dtype. Elements are [Complex].
abstract final class Complex64
    extends
        DTypeSpec<
          Float32,
          Complex,
          Float32,
          Complex64,
          Complex64,
          Complex64,
          Complex64
        > {}

/// Tag for the `complex128` dtype. Elements are [Complex].
abstract final class Complex128
    extends
        DTypeSpec<
          Float64,
          Complex,
          Float64,
          Complex128,
          Complex128,
          Complex128,
          Complex128
        > {}

/// Tag for the `boolean` dtype. Elements are `bool`.
abstract final class Boolean
    extends
        DTypeSpec<Boolean, bool, Float64, Complex128, Float64, Int64, Int32> {}

/// Supported data types for the elements of an [NDArray].

/// Compares two 64-bit integers as unsigned values.
///
/// In Dart, `int` is a signed 64-bit integer. Bit patterns with the MSB set
/// (values >= 2^63) represent negative integers in Dart. This function compares
/// two integers treating their bit patterns as unsigned 64-bit values.
int uint64Compare(int a, int b) {
  if (a == b) return 0;
  if (a < 0 && b >= 0) return 1;
  if (a >= 0 && b < 0) return -1;
  return a.compareTo(b);
}

int _computeCheckedTotalSize(List<int> shape) {
  ensureCpuSupported();
  var totalSize = 1;
  for (final dim in shape) {
    if (dim < 0) {
      throw ArgumentError('Shape dimensions cannot be negative: $shape');
    }
    if (dim > 2147483647) {
      throw UnsupportedError(
        'NDArray operations currently support arrays up to 2^31 - 1 elements. Got ${shape.length == 1 ? dim : shape}.',
      );
    }
    if (dim == 0) {
      totalSize = 0;
      continue;
    }
    if (totalSize != 0 && totalSize > 2147483647 ~/ dim) {
      throw UnsupportedError(
        'NDArray operations currently support arrays up to 2^31 - 1 elements. Got $shape.',
      );
    }
    totalSize *= dim;
  }
  if (totalSize > 2147483647 || totalSize < 0) {
    throw UnsupportedError(
      'NDArray operations currently support arrays up to 2^31 - 1 elements. Got $totalSize.',
    );
  }
  return totalSize;
}

@internal
int checkTotalSize(List<int> shape) => _computeCheckedTotalSize(shape);

/// The runtime description of an [NDArray]'s element type.
///
/// The type parameter [T] is the corresponding [DTypeTag], which ties a
/// `DType` value to the static type of the arrays it can describe: a
/// `DType<Float64>` can only be used to build an `NDArray<Float64>`.
enum DType<T extends DTypeTag> {
  float64<Float64>('float64', 8, '<f8'),
  float32<Float32>('float32', 4, '<f4'),
  float16<Float16>('float16', 2, '<f2'),
  bfloat16<BFloat16>('bfloat16', 2, '|V2'),
  int64<Int64>('int64', 8, '<i8'),
  int32<Int32>('int32', 4, '<i4'),
  int16<Int16>('int16', 2, '<i2'),
  int8<Int8>('int8', 1, '<i1'),

  /// Unsigned 64-bit integer ('<u8', 8 bytes).
  ///
  /// Note: Dart `int` is signed 64-bit. Bit patterns with MSB set (>= 2^63)
  /// represent negative ints in Dart. Use [uint64Compare] for unsigned comparisons.
  uint64<Uint64>('uint64', 8, '<u8'),
  uint32<Uint32>('uint32', 4, '<u4'),
  uint16<Uint16>('uint16', 2, '<u2'),
  uint8<Uint8>('uint8', 1, '|u1'),
  complex128<Complex128>('complex128', 16, '<c16'),
  complex64<Complex64>('complex64', 8, '<c8'),
  boolean<Boolean>('boolean', 1, '|b1');

  /// All 15 [DType] values typed as [DType<AnySpec>].
  static const List<DType<AnySpec>> specs = [
    float64,
    float32,
    float16,
    bfloat16,
    int64,
    int32,
    int16,
    int8,
    uint64,
    uint32,
    uint16,
    uint8,
    complex128,
    complex64,
    boolean,
  ];

  final String name;
  final int byteWidth;
  final String npyDescriptor;

  const DType(this.name, this.byteWidth, this.npyDescriptor);

  bool get isComplex => this == DType.complex64 || this == DType.complex128;
  bool get isFloating =>
      this == DType.float32 ||
      this == DType.float64 ||
      this == DType.float16 ||
      this == DType.bfloat16;
  bool get isHalf => this == DType.float16 || this == DType.bfloat16;
  bool get isInteger =>
      this == DType.int64 ||
      this == DType.int32 ||
      this == DType.int16 ||
      this == DType.int8 ||
      this == DType.uint64 ||
      this == DType.uint32 ||
      this == DType.uint16 ||
      this == DType.uint8;
  bool get isUnsigned =>
      this == DType.uint64 ||
      this == DType.uint32 ||
      this == DType.uint16 ||
      this == DType.uint8;
  bool get isSignedInteger =>
      this == DType.int64 ||
      this == DType.int32 ||
      this == DType.int16 ||
      this == DType.int8;

  NDArray<T> _createRaw(
    ffi.Pointer<ffi.Void> pointer,
    List<Object?> data,
    NDArray? parent, {
    required List<int> shape,
    required List<int> strides,
    int offsetElements = 0,
    ffi.Pointer<ffi.Void>? allocPointer,
    bool isExternallyOwned = false,
    ffi.Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>?
    customNativeFinalizer,
  }) =>
      (switch (this) {
            DType.float64 => _NDArrayFloat64(
              pointer,
              data,
              parent,
              shape: shape,
              strides: strides,
              offsetElements: offsetElements,
              allocPointer: allocPointer,
              isExternallyOwned: isExternallyOwned,
              customNativeFinalizer: customNativeFinalizer,
            ),
            DType.float32 => _NDArrayFloat32(
              pointer,
              data,
              parent,
              shape: shape,
              strides: strides,
              offsetElements: offsetElements,
              allocPointer: allocPointer,
              isExternallyOwned: isExternallyOwned,
              customNativeFinalizer: customNativeFinalizer,
            ),
            DType.float16 => _NDArrayFloat16(
              pointer,
              data,
              parent,
              shape: shape,
              strides: strides,
              offsetElements: offsetElements,
              allocPointer: allocPointer,
              isExternallyOwned: isExternallyOwned,
              customNativeFinalizer: customNativeFinalizer,
            ),
            DType.bfloat16 => _NDArrayBFloat16(
              pointer,
              data,
              parent,
              shape: shape,
              strides: strides,
              offsetElements: offsetElements,
              allocPointer: allocPointer,
              isExternallyOwned: isExternallyOwned,
              customNativeFinalizer: customNativeFinalizer,
            ),
            DType.int64 => _NDArrayInt64(
              pointer,
              data,
              parent,
              shape: shape,
              strides: strides,
              offsetElements: offsetElements,
              allocPointer: allocPointer,
              isExternallyOwned: isExternallyOwned,
              customNativeFinalizer: customNativeFinalizer,
            ),
            DType.int32 => _NDArrayInt32(
              pointer,
              data,
              parent,
              shape: shape,
              strides: strides,
              offsetElements: offsetElements,
              allocPointer: allocPointer,
              isExternallyOwned: isExternallyOwned,
              customNativeFinalizer: customNativeFinalizer,
            ),
            DType.int16 => _NDArrayInt16(
              pointer,
              data,
              parent,
              shape: shape,
              strides: strides,
              offsetElements: offsetElements,
              allocPointer: allocPointer,
              isExternallyOwned: isExternallyOwned,
              customNativeFinalizer: customNativeFinalizer,
            ),
            DType.int8 => _NDArrayInt8(
              pointer,
              data,
              parent,
              shape: shape,
              strides: strides,
              offsetElements: offsetElements,
              allocPointer: allocPointer,
              isExternallyOwned: isExternallyOwned,
              customNativeFinalizer: customNativeFinalizer,
            ),
            DType.uint64 => _NDArrayUint64(
              pointer,
              data,
              parent,
              shape: shape,
              strides: strides,
              offsetElements: offsetElements,
              allocPointer: allocPointer,
              isExternallyOwned: isExternallyOwned,
              customNativeFinalizer: customNativeFinalizer,
            ),
            DType.uint32 => _NDArrayUint32(
              pointer,
              data,
              parent,
              shape: shape,
              strides: strides,
              offsetElements: offsetElements,
              allocPointer: allocPointer,
              isExternallyOwned: isExternallyOwned,
              customNativeFinalizer: customNativeFinalizer,
            ),
            DType.uint16 => _NDArrayUint16(
              pointer,
              data,
              parent,
              shape: shape,
              strides: strides,
              offsetElements: offsetElements,
              allocPointer: allocPointer,
              isExternallyOwned: isExternallyOwned,
              customNativeFinalizer: customNativeFinalizer,
            ),
            DType.uint8 => _NDArrayUint8(
              pointer,
              data,
              parent,
              shape: shape,
              strides: strides,
              offsetElements: offsetElements,
              allocPointer: allocPointer,
              isExternallyOwned: isExternallyOwned,
              customNativeFinalizer: customNativeFinalizer,
            ),
            DType.complex128 => _NDArrayComplex128(
              pointer,
              data,
              parent,
              shape: shape,
              strides: strides,
              offsetElements: offsetElements,
              allocPointer: allocPointer,
              isExternallyOwned: isExternallyOwned,
              customNativeFinalizer: customNativeFinalizer,
            ),
            DType.complex64 => _NDArrayComplex64(
              pointer,
              data,
              parent,
              shape: shape,
              strides: strides,
              offsetElements: offsetElements,
              allocPointer: allocPointer,
              isExternallyOwned: isExternallyOwned,
              customNativeFinalizer: customNativeFinalizer,
            ),
            DType.boolean => _NDArrayBoolean(
              pointer,
              data,
              parent,
              shape: shape,
              strides: strides,
              offsetElements: offsetElements,
              allocPointer: allocPointer,
              isExternallyOwned: isExternallyOwned,
              customNativeFinalizer: customNativeFinalizer,
            ),
          })
          as NDArray<T>;
}

/// An n-dimensional array with memory allocated on the C heap.
///
/// **Memory Management Guidelines:**
/// - **Explicit Disposal & Scopes**: Always call [dispose] explicitly as soon as an array is no
///   longer needed, or wrap your computations in [NDArray.scope] for automated scope-level resource
///   management. While the garbage collector will eventually free C memory to prevent hard
///   leaks, it is blind to large native allocations, and garbage collection might not be
///   triggered early enough.
/// - **Views & Shared Memory**: Views (slices, reshapes, transposes, etc.) share the exact same
///   C memory as their parent; modifying a view mutates the parent and vice versa. Calling [dispose]
///   on a parent array immediately **invalidates all views** derived from it; accessing an
///   invalidated view causes crashes or undefined behavior.
///
/// **Arithmetic Error & Overflow Handling:**
/// Mathematical operations on [NDArray] are backed by native C ufuncs,
/// and adhere to the following rules:
/// - **Division by Zero**:
///   - **True Division (`/`)**: Performs floating-point division under IEEE 754 standards.
///     If one or both operands are integers, they are promoted to [DType.float64] (matching NumPy).
///     Division of non-zero values by zero results in `double.infinity` or `double.negativeInfinity`.
///     Division of zero by zero results in `double.nan`. No exceptions are thrown.
///   - **Floor Division (`~/` or `floorDivide`) & Remainder (`%` or `remainder`)**:
///     - **For Floating-Point Types**: Behaves identically to true division, returning `double.nan`
///       on division by zero without throwing exceptions.
///     - **For Integer Types**: It is an error if any element of the divisor is `0`.
///       This upfront safety check in Dart prevents native C integer division by zero, which is
///       undefined behavior in C and would crash the entire Dart VM process with a `SIGFPE` signal.
/// - **Overflow**:
///   - **Integer Overflow**: Integer operations (such as addition, subtraction, and multiplication)
///     performed on arrays of type `int32`, `int64`, `int16`, or `uint8` wrap around silently
///     using two's complement representation matching NumPy's wrapping behavior (unless operands
///     are upcasted, e.g., mixing `int32` and `int64` upcasts to `int64`).
///   - **Floating-Point Overflow**: Floating-point operations that exceed the representable bounds
///     of `float32` or `float64` overflow silently to `double.infinity` or `double.negativeInfinity`
///     per IEEE 754 rules (matching NumPy).
///
/// **Example Usage:**
/// ```dart
/// // Create a 2x3 array filled with ones
/// final a = NDArray<Float64>.ones([2, 3], DType.float64);
/// print(a.toList()); // [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
///
/// // Explicitly free memory when done
/// a.dispose();
/// ```
sealed class NDArray<T extends DTypeTag>
    implements ffi.Finalizable, ScopedResource {
  /// Pointer to the raw C memory allocated for this array (logical origin).
  final ffi.Pointer<ffi.Void> _pointer;

  /// Pointer to the physical start of the memory allocation (for root arrays).
  final ffi.Pointer<ffi.Void>? _allocPointer;

  /// A Dart list view of the raw C memory.
  ///
  /// **Restrictions:**
  /// - This list has a fixed length and cannot be resized.
  /// - When returned via [NDArrayElements.data], [NDArrayBaseElements.data], or
  ///   [dataRaw], this list retains a reference to the root owning [NDArray] so
  ///   the backing C memory is not garbage-collected while the list remains
  ///   reachable.
  /// - This list becomes invalid if the owning array is explicitly freed via
  ///   [dispose] (or when an enclosing [NDArray.scope] exits). Accessing it
  ///   after explicit disposal leads to undefined behavior or crashes.
  final List<Object?> _data;

  /// Retains the root owning [NDArray] on any escaping `.data` list view so the
  /// backing C heap allocation cannot be garbage-collected while the returned
  /// Dart list (or its backing typed-data buffer) remains reachable.
  static final Expando<NDArray<DTypeTag>> _dataOwnerExpando =
      Expando<NDArray<DTypeTag>>('NDArray.dataOwner');

  bool _dataEscaped = false;

  void _attachDataOwner() {
    final owner = _rootParent;
    final list = _data;
    _dataOwnerExpando[list] = owner;
    if (list is ComplexList) {
      _dataOwnerExpando[list.backingList] = owner;
    } else if (list is BoolList) {
      _dataOwnerExpando[list.backingList] = owner;
    }
  }

  @pragma('vm:prefer-inline')
  List<Object?> get _checkedData {
    if (isDisposed) {
      throw StateError('Cannot access a disposed NDArray.');
    }
    return _data;
  }

  /// A Dart list view of the raw C memory, with elements untyped.
  ///
  /// The tag [T] does not name the element type, so this getter cannot be
  /// typed. Use [NDArrayElements.data] for a `List<E>` view where `E` is the
  /// element type implied by the tag.
  @internal
  List<Object?> get dataRaw {
    if (isDisposed) {
      throw StateError('Cannot access a disposed NDArray.');
    }
    if (!_dataEscaped) {
      _attachDataOwner();
      _dataEscaped = true;
    }
    return _data;
  }

  /// The physical capacity in bytes of the native memory buffer backing this array or view.
  ///
  /// It is an error if this array has been disposed.
  ///
  /// **Performance considerations:**
  /// - Time complexity: $O(1)$.
  int get physicalByteCapacity {
    if (isDisposed) {
      throw StateError('Cannot access a disposed NDArray.');
    }
    return _data.length * dtype.byteWidth;
  }

  /// The dimensions of the n-dimensional array.
  final List<int> shape;

  /// The number of elements to skip in memory to move to the next position along each dimension.
  @internal
  final List<int> strides;

  /// The logical start of the array in the [data] list.
  final int offsetElements;

  /// The data type of the elements in the array.
  DType<T> get dtype;

  /// Returns true if the array is C-contiguous in memory.
  ///
  /// An array is C-contiguous if its elements are stored sequentially in memory
  /// such that the last dimension varies the fastest.
  ///
  /// **When does an array become non-contiguous?**
  /// - When creating views via slicing with a step greater than 1.
  /// - When transposing or permuting axes.
  /// - When reshaping a non-contiguous view.
  ///
  /// **Why is it relevant?**
  /// - **Performance**: Contiguous arrays allow for vectorized operations
  ///   (like BLAS calls). Non-contiguous arrays may fall back to element-by-element iteration.
  ///
  /// **How to make an array contiguous?**
  /// - Call [copy] to allocate a new contiguous array with the same elements.
  final bool isContiguous;

  /// Whether this array is writeable.
  ///
  /// Broadcast views created by `broadcast_to` (where any dimension `i` has
  /// `shape[i] > 1` and `strides[i] == 0`) alias multiple coordinates to the
  /// same memory address and are read-only (`isWriteable == false`).
  bool get isWriteable {
    for (var i = 0; i < shape.length; i++) {
      if (shape[i] > 1 && strides[i] == 0) return false;
    }
    return true;
  }

  /// The parent array if this is a view, to prevent it from being garbage collected.
  final NDArray? _parent;

  /// Whether the backing native memory is externally/user-allocated.
  final bool _isExternallyOwned;

  /// Optional user-provided native finalizer to deallocate external memory.
  final ffi.Pointer<
    ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>
  >?
  _customNativeFinalizer;

  /// The custom native finalizer instance for this array if it has one.
  final ffi.NativeFinalizer? _customFinalizerInstance;

  /// The total number of elements in the n-dimensional array.
  int get size => shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);

  /// The number of dimensions of the n-dimensional array.
  int get rank => shape.length;

  /// Whether this array is a 2D square matrix.
  ///
  /// An array is square if it has rank 2 (exactly 2 dimensions) and the
  /// size of both dimensions is equal (i.e. number of rows equals number of columns).
  ///
  /// Example:
  /// ```dart
  /// final a = NDArray.zeros([3, 3], DType.float64);
  /// print(a.isSquare); // true
  ///
  /// final b = NDArray.zeros([3, 4], DType.float64);
  /// print(b.isSquare); // false
  /// ```
  bool get isSquare => rank == 2 && shape[0] == shape[1];

  /// Whether this array has the same shape as [other].
  ///
  /// Comparing shapes is a $O(D)$ operation where $D$ is the rank (number of dimensions)
  /// of the array.
  ///
  /// Example:
  /// ```dart
  /// final a = NDArray.zeros([2, 3], DType.float64);
  /// final b = NDArray.ones([2, 3], DType.float64);
  /// final c = NDArray.zeros([3, 2], DType.float64);
  ///
  /// print(a.hasSameShape(b)); // true
  /// print(a.hasSameShape(c)); // false
  /// ```
  bool hasSameShape(NDArray<DTypeTag> other) => listEquals(shape, other.shape);

  static final _finalizer = ffi.NativeFinalizer(malloc.nativeFree);

  /// Cache of custom [ffi.NativeFinalizer] instances keyed by native function
  /// pointer address. `dart:ffi` requires the [ffi.NativeFinalizer] object
  /// itself to remain reachable for finalizer callbacks to run on GC.
  static final Map<int, ffi.NativeFinalizer> _customFinalizers = {};

  static ffi.NativeFinalizer _finalizerFor(
    ffi.Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>
    fn,
  ) => _customFinalizers[fn.address] ??= ffi.NativeFinalizer(fn);

  /// Whether to track all created memory-allocating (root) [NDArray]s.
  ///
  /// When enabled, all root [NDArray]s are tracked until they are disposed.
  /// This can be used to detect memory leaks in tests or during debugging.
  ///
  /// Note: enabling this will keep undisposed [NDArray]s in memory,
  /// preventing them from being garbage collected, which allows
  /// [checkNoLeaks] to report them.
  static const bool trackAllocations = ResourceScope.trackAllocations;

  /// Currently live [NDArray] allocations when [trackAllocations] is enabled.
  static List<NDArray<DTypeTag>> get trackedAllocations =>
      ResourceScope.trackedAllocations.whereType<NDArray<DTypeTag>>().toList();

  /// Verifies that no tracked [NDArray] allocations have leaked.
  static bool checkNoLeaks() => ResourceScope.checkNoLeaks();

  /// Clears the list of tracked [NDArray] allocations.
  static void clearTrackedAllocations() =>
      ResourceScope.clearTrackedAllocations();

  /// Executes [callback] in a lexical [ResourceScope] that disposes all
  /// non-detached [NDArray] instances allocated during [callback] upon exit.
  static R scope<R>(R Function() callback) => ResourceScope.scope(callback);

  /// Executes [callback] within an automatic resource management scope and
  /// promotes the returned owning [NDArray] to the caller's parent scope (or to
  /// manual ownership if there is no outer scope), while automatically disposing
  /// all intermediate arrays allocated inside [callback].
  ///
  /// It is an error if [callback] returns a view ([isView] is `true`), because
  /// `returning` calls [detachToParentScope] on the result. To return a slice or
  /// view of an inner temporary array, return `view.copy()` so that a compact
  /// owning array is promoted and the temporary parent buffer is freed on scope exit.
  static NDArray<T> returning<T extends DTypeTag>(
    NDArray<T> Function() callback,
  ) => ResourceScope.returning(callback);

  /// Executes [callback] outside any active [ResourceScope] so arrays created
  /// inside [callback] are not automatically disposed when the outer scope exits.
  static R unmanaged<R>(R Function() callback) =>
      ResourceScope.unmanaged(callback);

  static bool _checkContiguous(List<int> shape, List<int> strides) {
    if (strides.length != shape.length) return false;
    if (shape.contains(0)) return true;
    final cStrides = computeCStrides(shape);
    for (var i = 0; i < strides.length; i++) {
      if (shape[i] > 1 && strides[i] != cStrides[i]) return false;
    }
    return true;
  }

  /// Private factory that ensures the runtime generic type parameter matches [dtype].
  factory NDArray._(
    ffi.Pointer<ffi.Void> pointer,
    List<Object?> data,
    NDArray? parent, {
    required List<int> shape,
    required List<int> strides,
    required DType dtype,
    int offsetElements = 0,
    ffi.Pointer<ffi.Void>? allocPointer,
    bool isExternallyOwned = false,
    ffi.Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>?
    customNativeFinalizer,
  }) =>
      dtype._createRaw(
            pointer,
            data,
            parent,
            shape: shape,
            strides: strides,
            offsetElements: offsetElements,
            allocPointer: allocPointer,
            isExternallyOwned: isExternallyOwned,
            customNativeFinalizer: customNativeFinalizer,
          )
          as NDArray<T>;

  /// Private generative constructor for internal use.
  NDArray._raw(
    this._pointer,
    this._data,
    this._parent, {
    required List<int> shape,
    required List<int> strides,
    this.offsetElements = 0,
    ffi.Pointer<ffi.Void>? allocPointer,
    bool isExternallyOwned = false,
    ffi.Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>?
    customNativeFinalizer,
  }) : _allocPointer = allocPointer,
       _isExternallyOwned = isExternallyOwned,
       _customNativeFinalizer = customNativeFinalizer,
       _customFinalizerInstance =
           (_parent == null &&
               isExternallyOwned &&
               customNativeFinalizer != null)
           ? _finalizerFor(customNativeFinalizer)
           : null,
       shape = List<int>.unmodifiable(shape),
       strides = List<int>.unmodifiable(strides),
       isContiguous = _checkContiguous(shape, strides) {
    _computeCheckedTotalSize(shape);
    for (final stride in strides) {
      if (stride < -0x80000000 || stride > 0x7fffffff) {
        throw UnsupportedError(
          'Stride $stride exceeds 32-bit native int limit.',
        );
      }
    }
    _initializeOpenBLASOnce();
    if (_parent == null) {
      final ptrToFree = _allocPointer ?? _pointer;
      // NOTE: Do NOT pass `externalSize` to `NativeFinalizer.attach`.
      // `externalSize` is too blunt a tool: reporting large off-heap buffer
      // sizes can trigger severe GC thrashing rather than prompt reclamation.
      // Instead, users and internal operations should use `NDArray.scope` (or
      // manual `dispose()`) for deterministic native memory cleanup, with
      // `NativeFinalizer` acting strictly as a backstop safety net.
      if (!_isExternallyOwned) {
        _finalizer.attach(this, ptrToFree, detach: this);
      } else if (_customFinalizerInstance != null) {
        _customFinalizerInstance.attach(this, ptrToFree, detach: this);
      }
      ResourceScope.track(this);
    }
  }

  /// Recursively locates the root memory-allocating parent array.
  NDArray get _rootParent {
    var current = this as NDArray;
    while (current._parent != null) {
      current = current._parent;
    }
    return current;
  }

  /// Removes this owning array from its current automatic [ResourceScope] so it
  /// survives beyond the scope's lifetime and must be manually disposed via [dispose].
  ///
  /// It is an error if this array is a view ([isView] is `true`), because views
  /// do not own native memory ([dispose] on a view is a no-op):
  /// - Untracking `_rootParent` would remove the backing allocation from automatic
  ///   scope cleanup while leaving the caller with a view whose [dispose] cannot
  ///   free `_rootParent`.
  /// - Returning a hidden `copy()` would leave `_rootParent` behind in the current
  ///   scope (causing a use-after-free if the caller invoked `view.detachFromScope()`
  ///   as a statement or cascade `..detachFromScope()`).
  ///
  /// To escape a scope with a slice/view of an inner temporary array, materialize
  /// an owning copy explicitly (`view.copy().detachFromScope()`) or detach the
  /// owning parent array directly.
  @override
  NDArray<T> detachFromScope() {
    if (_parent != null) {
      throw StateError(
        'Cannot call detachFromScope() on a view (views do not own memory and '
        'cannot be disposed). Call view.copy().detachFromScope() or detach the '
        'owning parent array instead.',
      );
    }
    ResourceScope.untrack(this);
    return this;
  }

  /// Detaches this owning array from the current inner [ResourceScope] and promotes
  /// it to the parent outer scope (or to manual ownership if there is no outer scope).
  ///
  /// It is an error if this array is a view ([isView] is `true`), because views
  /// do not own native memory ([dispose] on a view is a no-op):
  /// - Promoting `_rootParent` from a top-level scope (`_parentScope == null`)
  ///   would leave `_rootParent` un-disposable via `view.dispose()` (leaking native
  ///   memory and failing `ResourceScope.checkNoLeaks()`), and in nested scopes
  ///   would silently retain the entire backing allocation of a small slice.
  /// - Returning a hidden `copy()` would leave `_rootParent` behind in the current
  ///   scope to be freed on scope exit, causing a use-after-free whenever
  ///   `detachToParentScope()` is called as a statement or cascade (`..detachToParentScope()`).
  ///
  /// To return a slice/view of an inner temporary array from a scope, materialize
  /// an owning copy explicitly (`view.copy().detachToParentScope()`). If the view's
  /// parent was already created in an outer scope, the view can be returned directly
  /// without calling `detachToParentScope()`.
  @override
  NDArray<T> detachToParentScope() {
    if (_parent != null) {
      throw StateError(
        'Cannot call detachToParentScope() on a view (views do not own memory). '
        'Call view.copy().detachToParentScope() to materialize an owning copy, '
        'or detach the owning parent array instead.',
      );
    }
    ResourceScope.promoteToParent(this);
    return this;
  }

  bool _isDisposed = false;

  @override
  bool get isDisposed => _isDisposed || (_parent != null && _parent.isDisposed);

  /// Returns true if this is a zero-copy view sharing memory with another array.
  bool get isView => _parent != null;

  /// Factory to create a new multi-dimensional array with backing unmanaged C heap memory.
  ///
  /// This allocates raw, stable memory directly on the unmanaged C heap using `malloc` or `calloc`.
  /// The resulting array is backed by standard Dart TypedLists mapping directly to the unmanaged pages.
  ///
  /// **Preconditions:**
  /// - All dimensions in [shape] must be strictly non-negative ($\ge 0$).
  /// - Total element count must not exceed $2^{31} - 1$ ($2,147,483,647$).
  ///
  /// - It is an error if any dimension in [shape] is negative.
  /// - It is an error if the total element count exceeds $2^{31} - 1$ ($2,147,483,647$).
  /// - It is an error if the provided [dtype] is unsupported.
  ///
  /// **Performance considerations:**
  /// - Algorithmic time complexity is $O(N)$ and space complexity is $O(N)$ where $N$ is the total
  ///   number of elements (product of all dimensions in [shape]).
  /// - Allocates memory directly from the OS heap (virtual memory page mappings).
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<Float64>.create([2, 2], DType.float64, zeroInit: true);
  /// print(a.toList()); // [0.0, 0.0, 0.0, 0.0]
  /// ```
  ///
  /// **Edge cases:**
  /// - Backing heap pages are not managed by isolate garbage collection. Call `dispose()` explicitly to prevent leaks.
  ///
  /// Refer to the [NumPy Array Creation Guidelines](https://numpy.org/doc/stable/reference/routines.array-creation.html)
  /// and [Dart FFI Memory Management](https://dart.dev/guides/libraries/c-interop) for additional details.
  factory NDArray.create(
    List<int> shape,
    DType<T> dtype, {
    bool zeroInit = false,
    @internal List<int>? strides,
  }) {
    final totalSize = _computeCheckedTotalSize(shape);
    final finalStrides = strides ?? computeCStrides(shape);
    final bool isEmpty = shape.contains(0);
    var minRelativeOffset = 0;
    var maxRelativeOffset = 0;
    if (!isEmpty && strides != null) {
      if (strides.length != shape.length) {
        throw ArgumentError(
          'Strides length (${strides.length}) must match shape length (${shape.length}).',
        );
      }
      for (var d = 0; d < shape.length; d++) {
        final stride = strides[d];
        final size = shape[d];
        if (stride > 0) {
          maxRelativeOffset += (size - 1) * stride;
        } else if (stride < 0) {
          minRelativeOffset += (size - 1) * stride;
        }
      }
    }
    final int allocSize = strides == null
        ? totalSize
        : (isEmpty ? 0 : (maxRelativeOffset - minRelativeOffset + 1));
    final int initialOffsetElements = (strides == null || isEmpty)
        ? 0
        : -minRelativeOffset;

    final allocator = zeroInit ? calloc : malloc;
    ffi.Pointer<ffi.Void> pointer;
    List<Object?> data;

    switch (dtype) {
      case DType.float64:
        final p = allocator<ffi.Double>(allocSize);
        pointer = p.cast();
        data = p.asTypedList(allocSize);
      case DType.float32:
        final p = allocator<ffi.Float>(allocSize);
        pointer = p.cast();
        data = p.asTypedList(allocSize);
      case DType.float16:
        final p = allocator<ffi.Uint16>(allocSize);
        pointer = p.cast();
        data = Float16List(p.asTypedList(allocSize));
      case DType.bfloat16:
        final p = allocator<ffi.Uint16>(allocSize);
        pointer = p.cast();
        data = BFloat16List(p.asTypedList(allocSize));
      case DType.int64:
        final p = allocator<ffi.Int64>(allocSize);
        pointer = p.cast();
        data = p.asTypedList(allocSize);
      case DType.int32:
        final p = allocator<ffi.Int32>(allocSize);
        pointer = p.cast();
        data = p.asTypedList(allocSize);
      case DType.int16:
        final p = allocator<ffi.Int16>(allocSize);
        pointer = p.cast();
        data = p.asTypedList(allocSize);
      case DType.int8:
        final p = allocator<ffi.Int8>(allocSize);
        pointer = p.cast();
        data = p.asTypedList(allocSize);
      case DType.uint64:
        final p = allocator<ffi.Uint64>(allocSize);
        pointer = p.cast();
        data = p.asTypedList(allocSize);
      case DType.uint32:
        final p = allocator<ffi.Uint32>(allocSize);
        pointer = p.cast();
        data = p.asTypedList(allocSize);
      case DType.uint16:
        final p = allocator<ffi.Uint16>(allocSize);
        pointer = p.cast();
        data = p.asTypedList(allocSize);
      case DType.uint8:
        final p = allocator<ffi.Uint8>(allocSize);
        pointer = p.cast();
        data = p.asTypedList(allocSize);
      case DType.complex128:
        final p = allocator<ffi.Double>(allocSize * 2);
        pointer = p.cast();
        final doubleList = p.asTypedList(allocSize * 2);
        data = ComplexList(doubleList);
      case DType.complex64:
        final p = allocator<ffi.Float>(allocSize * 2);
        pointer = p.cast();
        final floatList = p.asTypedList(allocSize * 2);
        data = ComplexList(floatList);
      case DType.boolean:
        final p = allocator<ffi.Uint8>(allocSize);
        pointer = p.cast();
        final uint8List = p.asTypedList(allocSize);
        data = BoolList(uint8List);
    }

    final logicalPointer = initialOffsetElements == 0
        ? pointer
        : _offsetPointer(pointer, initialOffsetElements, dtype);

    return NDArray._(
      logicalPointer,
      data,
      null,
      shape: shape,
      strides: finalStrides,
      dtype: dtype,
      offsetElements: initialOffsetElements,
      allocPointer: pointer,
    );
  }

  /// Factory to create a C-contiguous array from a Dart list (copies data).
  ///
  /// The [list] is flattened and copied into the newly allocated array memory.
  /// The total size of the [shape] must match the number of elements in [list].
  ///
  /// It is an error if the total size of [shape] does not match the length of [list].
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
  /// ```
  factory NDArray.fromList(
    List<dynamic> list,
    List<int> shape,
    DType<T> dtype,
  ) {
    final totalSize = _computeCheckedTotalSize(shape);
    if (totalSize != list.length) {
      throw ArgumentError(
        'Total size of shape $shape ($totalSize) must match list length (${list.length})',
      );
    }
    final arr = NDArray<T>.create(shape, dtype);
    final List<dynamic> eagerList = switch (dtype) {
      DType.float64 => Float64List.fromList(
        list.map((e) => (e as num).toDouble()).toList(),
      ),
      DType.float32 => Float32List.fromList(
        list.map((e) => (e as num).toDouble()).toList(),
      ),
      DType.float16 ||
      DType.bfloat16 => list.map((e) => (e as num).toDouble()).toList(),
      DType.int64 => Int64List.fromList(
        list.map((e) => (e as num).toInt()).toList(),
      ),
      DType.int32 => Int32List.fromList(
        list.map((e) => (e as num).toInt()).toList(),
      ),
      DType.int16 => Int16List.fromList(
        list.map((e) => (e as num).toInt()).toList(),
      ),
      DType.int8 => Int8List.fromList(
        list.map((e) => (e as num).toInt()).toList(),
      ),
      DType.uint64 => Uint64List.fromList(
        list.map((e) => (e as num).toInt()).toList(),
      ),
      DType.uint32 => Uint32List.fromList(
        list.map((e) => (e as num).toInt()).toList(),
      ),
      DType.uint16 => Uint16List.fromList(
        list.map((e) => (e as num).toInt()).toList(),
      ),
      DType.uint8 => Uint8List.fromList(
        list.map((e) => (e as num).toInt()).toList(),
      ),
      DType.boolean => List<bool>.from(list),
      DType.complex128 || DType.complex64 => List<Complex>.from(list),
    };
    for (var i = 0; i < eagerList.length; i++) {
      arr.setCellRaw(i, eagerList[i]);
    }
    return arr;
  }

  /// Factory to create a 0-dimensional scalar array containing a single [value].
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.scalar(42, dtype: DType.int32);
  /// print(a.shape); // []
  /// print(a.scalar); // 42
  /// ```
  factory NDArray.scalar(Object? value, {required DType<T> dtype}) {
    return NDArray.fromList([value], [], dtype);
  }

  /// Factory to create a new C-contiguous array filled with zeros.
  ///
  /// Backed directly by unmanaged C heap memory pages allocated via `calloc`.
  ///
  /// **Preconditions:**
  /// - All dimensions in [shape] must be strictly non-negative ($\ge 0$).
  /// - Total element count must not exceed $2^{31} - 1$ ($2,147,483,647$).
  ///
  /// - It is an error if any dimension in [shape] is negative.
  /// - It is an error if the total element count exceeds $2^{31} - 1$ ($2,147,483,647$).
  /// - It is an error if the provided [dtype] is unsupported.
  ///
  /// **Performance considerations:**
  /// - Algorithmic time complexity is $O(N)$ and space complexity is $O(N)$ where $N$ is the total
  ///   number of elements (product of all dimensions in [shape]).
  /// - Allocates memory from the C heap using `calloc`.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<Float64>.zeros([2, 2], DType.float64);
  /// print(a.toList()); // [0.0, 0.0, 0.0, 0.0]
  /// ```
  ///
  /// Refer to the [NumPy zeros reference](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html)
  /// and [Dart FFI calloc allocator](https://pub.dev/documentation/ffi/latest/ffi/calloc-constant.html) for additional details.
  factory NDArray.zeros(List<int> shape, DType<T> dtype) {
    return NDArray<T>.create(shape, dtype, zeroInit: true);
  }

  /// Factory to create an array filled with ones.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<Float64>.ones([2, 2], DType.float64);
  /// print(a.toList()); // [[1.0, 1.0], [1.0, 1.0]]
  /// ```
  factory NDArray.ones(List<int> shape, DType<T> dtype) {
    final arr = NDArray<T>.create(shape, dtype);
    if (dtype.isComplex) {
      arr.fill(Complex(1.0, 0.0));
    } else if (dtype == DType.boolean) {
      arr.fill(true);
    } else if (dtype.isFloating) {
      arr.fill(1.0);
    } else {
      arr.fill(1);
    }
    return arr;
  }

  /// Factory to create an array filled with a specified scalar [fillValue].
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<Float64>.full([2, 2], 0.5, dtype: DType.float64);
  /// print(a.toList()); // [0.5, 0.5, 0.5, 0.5]
  /// ```
  ///
  /// Refer to the [NumPy full reference](https://numpy.org/doc/stable/reference/generated/numpy.full.html) for additional details.
  factory NDArray.full(
    List<int> shape,
    Object? fillValue, {
    required DType<T> dtype,
  }) {
    final arr = NDArray<T>.create(shape, dtype);
    arr.fill(fillValue);
    return arr;
  }

  /// Factory to create an array with a range of values.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<Float64>.arange(0.0, 5.0, step: 1.0, dtype: DType.float64);
  /// print(a.toList()); // [0.0, 1.0, 2.0, 3.0, 4.0]
  /// ```
  factory NDArray.arange(
    double start,
    double stop, {
    double step = 1.0,
    required DType<T> dtype,
  }) {
    if (step == 0.0) {
      throw ArgumentError('Step size cannot be zero.');
    }
    if ((stop > start && step < 0.0) || (stop < start && step > 0.0)) {
      throw ArgumentError('Step size direction must match start/stop range.');
    }
    final length = ((stop - start) / step).ceil();
    final arr = NDArray<T>.create([length], dtype);
    for (var i = 0; i < length; i++) {
      final val = start + i * step;
      if (dtype.isComplex) {
        arr.setCellRaw(i, Complex(val, 0.0));
      } else if (dtype.isInteger) {
        arr.setCellRaw(i, val.toInt());
      } else if (dtype == DType.boolean) {
        arr.setCellRaw(i, (val != 0.0));
      } else {
        arr.setCellRaw(i, val);
      }
    }
    return arr;
  }

  /// Factory to create a 2D identity matrix.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<Float64>.eye(3, DType.float64);
  /// print(a.toList()); // [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  /// ```
  ///
  /// **Edge cases:**
  /// - This only creates 2D square matrices.
  factory NDArray.eye(int n, DType<T> dtype) {
    final arr = NDArray<T>.zeros([n, n], dtype);
    for (var i = 0; i < n; i++) {
      if (dtype.isFloating) {
        arr.setCellRaw(i * n + i, 1.0);
      } else if (dtype.isComplex) {
        arr.setCellRaw(i * n + i, Complex(1.0, 0.0));
      } else if (dtype == DType.boolean) {
        arr.setCellRaw(i * n + i, true);
      } else {
        arr.setCellRaw(i * n + i, 1);
      }
    }
    return arr;
  }

  /// Factory to create a view sharing the same memory.
  ///
  /// **Example:**
  /// ```dart
  /// final view = NDArray.view(parent, shape: [2], strides: [1], offsetElements: 1);
  /// ```
  ///
  /// **Restrictions:**
  /// - **Lifetime Dependency**: The view is only valid as long as the parent's memory is not freed. If you call `parent.dispose()`, this view becomes invalid.
  /// - **Shared Mutations**: Modifications to the view affect the parent and vice versa.
  /// - **No Ownership**: Calling `dispose()` on a view does nothing.
  factory NDArray.view(
    NDArray<DTypeTag> parent, {
    required List<int> shape,
    required List<int> strides,
    int offsetElements = 0,
  }) {
    if (parent.isDisposed) {
      throw StateError('Cannot create a view of a disposed NDArray.');
    }
    if (shape.length != strides.length) {
      throw ArgumentError(
        'shape length (${shape.length}) must match strides length (${strides.length}).',
      );
    }
    _computeCheckedTotalSize(shape);
    final root = parent._rootParent;
    final rootPhysicalStart = root._allocPointer ?? root._pointer;
    final bool isEmpty = shape.contains(0);
    final childLogicalPointer = isEmpty
        ? rootPhysicalStart
        : _offsetPointer(parent.pointer, offsetElements, parent.dtype);
    final cumulativeOffset =
        (childLogicalPointer.address - rootPhysicalStart.address) ~/
        parent.dtype.byteWidth;

    // Calculate min and max relative offsets
    var minRelativeOffset = 0;
    var maxRelativeOffset = 0;
    if (!isEmpty) {
      for (var d = 0; d < shape.length; d++) {
        final stride = strides[d];
        final size = shape[d];
        if (stride > 0) {
          maxRelativeOffset += (size - 1) * stride;
        } else if (stride < 0) {
          minRelativeOffset += (size - 1) * stride;
        }
      }
    }

    final minPhysicalOffset = cumulativeOffset + minRelativeOffset;
    final maxPhysicalOffset = cumulativeOffset + maxRelativeOffset;

    if (!isEmpty) {
      if (minPhysicalOffset < 0 || maxPhysicalOffset >= root._data.length) {
        throw RangeError(
          'View physical offsets [$minPhysicalOffset..$maxPhysicalOffset] exceed root buffer bounds [0..${root._data.length - 1}].',
        );
      }
    }

    final physicalPointer = _offsetPointer(
      rootPhysicalStart,
      minPhysicalOffset,
      parent.dtype,
    );
    final int viewSize = isEmpty
        ? 0
        : (maxPhysicalOffset - minPhysicalOffset + 1);
    final List<Object?> data;

    switch (parent.dtype) {
      case DType.float64:
        data = physicalPointer.cast<ffi.Double>().asTypedList(viewSize);
      case DType.float32:
        data = physicalPointer.cast<ffi.Float>().asTypedList(viewSize);
      case DType.float16:
        data = Float16List(
          physicalPointer.cast<ffi.Uint16>().asTypedList(viewSize),
        );
      case DType.bfloat16:
        data = BFloat16List(
          physicalPointer.cast<ffi.Uint16>().asTypedList(viewSize),
        );
      case DType.int64:
        data = physicalPointer.cast<ffi.Int64>().asTypedList(viewSize);
      case DType.int32:
        data = physicalPointer.cast<ffi.Int32>().asTypedList(viewSize);
      case DType.int16:
        data = physicalPointer.cast<ffi.Int16>().asTypedList(viewSize);
      case DType.int8:
        data = physicalPointer.cast<ffi.Int8>().asTypedList(viewSize);
      case DType.uint64:
        data = physicalPointer.cast<ffi.Uint64>().asTypedList(viewSize);
      case DType.uint32:
        data = physicalPointer.cast<ffi.Uint32>().asTypedList(viewSize);
      case DType.uint16:
        data = physicalPointer.cast<ffi.Uint16>().asTypedList(viewSize);
      case DType.uint8:
        data = physicalPointer.cast<ffi.Uint8>().asTypedList(viewSize);
      case DType.complex128:
        final p = _offsetPointer(
          rootPhysicalStart,
          minPhysicalOffset * 2,
          DType.float64,
        );
        final doubleList = p.cast<ffi.Double>().asTypedList(viewSize * 2);
        data = ComplexList(doubleList);
      case DType.complex64:
        final p = _offsetPointer(
          rootPhysicalStart,
          minPhysicalOffset * 2,
          DType.float32,
        );
        final floatList = p.cast<ffi.Float>().asTypedList(viewSize * 2);
        data = ComplexList(floatList);
      case DType.boolean:
        data = BoolList(
          physicalPointer.cast<ffi.Uint8>().asTypedList(viewSize),
        );
    }

    final viewOffsetElements = isEmpty ? 0 : -minRelativeOffset;

    return NDArray._(
      childLogicalPointer,
      data,
      parent,
      shape: shape,
      strides: strides,
      dtype: parent.dtype as DType<T>,
      offsetElements: viewOffsetElements,
    );
  }

  /// Factory to create a new [NDArray] view backed by a user-allocated external C memory pointer.
  ///
  /// The user must ensure that [pointer] points to a valid block of contiguous memory
  /// of at least `size * dtype.byteWidth` bytes, where `size` is the product of all dimensions in [shape].
  ///
  /// **Preconditions:**
  /// - [pointer] must not be null or point to an invalid memory location.
  /// - All dimensions in [shape] must be strictly non-negative ($\ge 0$).
  ///
  /// **Lifetime Management Options:**
  /// - **Externally Managed (Default):** If [nativeFinalizer] is omitted or `null`, the array does not
  ///   own the memory. Calling [dispose] will invalidate the array and any of its views, but will **not**
  ///   free the raw C memory pointer. The user is fully responsible for freeing the memory.
  /// - **Custom Finalization:** If [nativeFinalizer] is provided, it will be registered with a Dart
  ///   [NativeFinalizer] to automatically deallocate the backing C pointer when this array is garbage collected,
  ///   or when [dispose] is called.
  ///
  /// It is an error if any dimension in [shape] is negative.
  ///
  /// **Performance Considerations:**
  /// - This is an $O(1)$ operation that performs zero copies, constructing a direct list view over
  ///   the provided raw C memory address.
  ///
  /// **Example:**
  /// {@example /example/external_memory_example.dart}
  factory NDArray.fromPointer(
    ffi.Pointer<ffi.Void> pointer,
    List<int> shape,
    DType<T> dtype, {
    ffi.Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>?
    nativeFinalizer,
    List<int>? strides,
  }) {
    final totalSize = _computeCheckedTotalSize(shape);
    final finalStrides = strides ?? computeCStrides(shape);
    final bool isEmpty = shape.contains(0);
    var minRelativeOffset = 0;
    var maxRelativeOffset = 0;
    if (!isEmpty && strides != null) {
      if (strides.length != shape.length) {
        throw ArgumentError(
          'Strides length (${strides.length}) must match shape length (${shape.length}).',
        );
      }
      for (var d = 0; d < shape.length; d++) {
        final stride = strides[d];
        final size = shape[d];
        if (stride > 0) {
          maxRelativeOffset += (size - 1) * stride;
        } else if (stride < 0) {
          minRelativeOffset += (size - 1) * stride;
        }
      }
    }
    final int allocSize = strides == null
        ? totalSize
        : (isEmpty ? 0 : (maxRelativeOffset - minRelativeOffset + 1));
    final int initialOffsetElements = (strides == null || isEmpty)
        ? 0
        : -minRelativeOffset;

    List<Object?> data;
    switch (dtype) {
      case DType.float64:
        data = pointer.cast<ffi.Double>().asTypedList(allocSize);
      case DType.float32:
        data = pointer.cast<ffi.Float>().asTypedList(allocSize);
      case DType.float16:
        data = Float16List(pointer.cast<ffi.Uint16>().asTypedList(allocSize));
      case DType.bfloat16:
        data = BFloat16List(pointer.cast<ffi.Uint16>().asTypedList(allocSize));
      case DType.int64:
        data = pointer.cast<ffi.Int64>().asTypedList(allocSize);
      case DType.int32:
        data = pointer.cast<ffi.Int32>().asTypedList(allocSize);
      case DType.int16:
        data = pointer.cast<ffi.Int16>().asTypedList(allocSize);
      case DType.int8:
        data = pointer.cast<ffi.Int8>().asTypedList(allocSize);
      case DType.uint64:
        data = pointer.cast<ffi.Uint64>().asTypedList(allocSize);
      case DType.uint32:
        data = pointer.cast<ffi.Uint32>().asTypedList(allocSize);
      case DType.uint16:
        data = pointer.cast<ffi.Uint16>().asTypedList(allocSize);
      case DType.uint8:
        data = pointer.cast<ffi.Uint8>().asTypedList(allocSize);
      case DType.complex128:
        data = ComplexList(
          pointer.cast<ffi.Double>().asTypedList(allocSize * 2),
        );
      case DType.complex64:
        data = ComplexList(
          pointer.cast<ffi.Float>().asTypedList(allocSize * 2),
        );
      case DType.boolean:
        data = BoolList(pointer.cast<ffi.Uint8>().asTypedList(allocSize));
    }

    final logicalPointer = initialOffsetElements == 0
        ? pointer
        : _offsetPointer(pointer, initialOffsetElements, dtype);

    return NDArray._(
      logicalPointer,
      data,
      null,
      shape: shape,
      strides: finalStrides,
      dtype: dtype,
      offsetElements: initialOffsetElements,
      allocPointer: pointer,
      isExternallyOwned: true,
      customNativeFinalizer: nativeFinalizer,
    );
  }

  /// Helper to calculate default strides for a C-contiguous array (in elements).
  @internal
  static List<int> computeCStrides(List<int> shape) {
    if (shape.isEmpty) return [];
    final strides = List<int>.filled(shape.length, 1);
    for (var i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }

  static ffi.Pointer<ffi.Void> _offsetPointer(
    ffi.Pointer<ffi.Void> ptr,
    int offsetElements,
    DType dtype,
  ) {
    switch (dtype) {
      case DType.float64:
        return (ptr.cast<ffi.Double>() + offsetElements).cast();
      case DType.float32:
        return (ptr.cast<ffi.Float>() + offsetElements).cast();
      case DType.float16:
      case DType.bfloat16:
        return (ptr.cast<ffi.Uint16>() + offsetElements).cast();
      case DType.int64:
        return (ptr.cast<ffi.Int64>() + offsetElements).cast();
      case DType.int32:
        return (ptr.cast<ffi.Int32>() + offsetElements).cast();
      case DType.int16:
        return (ptr.cast<ffi.Int16>() + offsetElements).cast();
      case DType.int8:
        return (ptr.cast<ffi.Int8>() + offsetElements).cast();
      case DType.uint64:
        return (ptr.cast<ffi.Uint64>() + offsetElements).cast();
      case DType.uint32:
        return (ptr.cast<ffi.Uint32>() + offsetElements).cast();
      case DType.uint16:
        return (ptr.cast<ffi.Uint16>() + offsetElements).cast();
      case DType.uint8:
        return (ptr.cast<ffi.Uint8>() + offsetElements).cast();
      case DType.complex128:
        return (ptr.cast<ffi.Double>() + (offsetElements * 2)).cast();
      case DType.complex64:
        return (ptr.cast<ffi.Float>() + (offsetElements * 2)).cast();
      case DType.boolean:
        return (ptr.cast<ffi.Uint8>() + offsetElements).cast();
    }
  }

  /// Expose the raw pointer for FFI use.
  ffi.Pointer<ffi.Void> get pointer {
    if (isDisposed) {
      throw StateError(
        'Cannot access an array or view whose memory has been explicitly freed/disposed!',
      );
    }
    return _pointer;
  }

  /// Returns a new view of this array with a new shape.
  ///
  /// **Preconditions:**
  /// - The total size (product of dimensions) of the [newShape] must exactly match the current size.
  ///
  /// It is an error if the array has been disposed, or if the total size of [newShape] does not match the original size.
  ///
  /// **Performance considerations:**
  /// - If the array [isContiguous], this is a $O(1)$ operation, returning a zero-allocation view sharing backing memory.
  /// - If the array is a non-contiguous view, this flattens it first, performing a copy and allocating a new contiguous array ($O(N)$ complexity).
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
  /// final b = a.reshape([2, 2]);
  /// print(b.shape); // [2, 2]
  /// ```
  NDArray<T> reshape(List<int> newShape) {
    if (isDisposed) {
      throw StateError(
        'Cannot access an array or view whose memory has been explicitly freed/disposed!',
      );
    }
    final oldSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);

    int negOneIdx = -1;
    int knownProd = 1;
    for (var i = 0; i < newShape.length; i++) {
      final d = newShape[i];
      if (d == -1) {
        if (negOneIdx != -1) {
          throw ArgumentError(
            'Can only specify one unknown dimension (-1) in reshape, got $newShape',
          );
        }
        negOneIdx = i;
      } else if (d < 0) {
        throw ArgumentError('Negative dimension size: $d');
      } else {
        knownProd *= d;
      }
    }

    final List<int> resolvedShape;
    if (negOneIdx != -1) {
      if (knownProd == 0 || oldSize % knownProd != 0) {
        throw ArgumentError(
          'Cannot reshape array of size $oldSize into shape $newShape',
        );
      }
      resolvedShape = List<int>.from(newShape);
      resolvedShape[negOneIdx] = oldSize ~/ knownProd;
    } else {
      resolvedShape = newShape;
    }

    final newSize = _computeCheckedTotalSize(resolvedShape);
    if (oldSize != newSize) {
      throw ArgumentError(
        'Total size must not change during reshape (was $oldSize, new is $newSize)',
      );
    }

    if (listEquals(resolvedShape, shape)) {
      return NDArray._(
        _pointer,
        _data,
        _parent ?? this,
        shape: resolvedShape,
        strides: strides,
        offsetElements: offsetElements,
        allocPointer: _allocPointer,
        dtype: dtype,
      );
    }

    if (!isContiguous) {
      final result = NDArray<T>.create(resolvedShape, dtype);
      _copyStridedToContiguous(result);
      return result;
    }

    final newStrides = computeCStrides(resolvedShape);
    return NDArray._(
      _pointer,
      _data,
      _parent ?? this,
      shape: resolvedShape,
      strides: newStrides,
      offsetElements: offsetElements,
      allocPointer: _allocPointer,
      dtype: dtype,
    );
  }

  /// Returns a copy of the array collapsed into a one-dimensional tensor list.
  ///
  /// **Performance considerations:**
  /// - For C-contiguous layouts, copies memory directly.
  /// - For strided non-contiguous views, performs dynamic coordinate walk copy.
  /// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<Float64>.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
  /// final flat = a.flatten();
  /// print(flat.shape); // [4]
  /// print(flat.toList()); // [1.0, 2.0, 3.0, 4.0]
  /// ```
  NDArray<T> flatten() {
    if (isDisposed) {
      throw StateError('Cannot flatten a disposed NDArray.');
    }
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final result = NDArray<T>.create([totalSize], dtype);

    if (totalSize == 0) {
      return result;
    }

    try {
      if (isContiguous) {
        _copyContiguousNDArray(this, result, totalSize);
      } else {
        _copyStridedToContiguous(result);
      }
    } catch (_) {
      result.dispose();
      rethrow;
    }
    return result;
  }

  /// Returns a deep, C-contiguous copy of this array.
  ///
  /// The copy preserves the logical order and values of the elements defined by
  /// this array's shape and strides. However, the physical memory layout of the
  /// returned array is always contiguous (and its strides are reset to standard
  /// C-contiguous row-major strides).
  ///
  /// **Preconditions:**
  /// - This array must not be disposed.
  /// - If [out] is provided, it must not be disposed and must have the exact
  ///   same [shape] and [dtype] as this array.
  ///
  /// **Performance considerations:**
  /// - Algorithmic complexity is $O(N)$ time and $O(N)$ space, where $N$ is
  ///   the total number of elements.
  /// - Uses fast SIMD/C-level `memcpy` if both source and destination are C-contiguous.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
  /// final b = a.copy();
  /// b.setCell([0, 0], 99);
  /// print(a.getCell([0, 0])); // 1 (decoupled memory!)
  /// ```
  NDArray<T> copy({NDArray<T>? out}) {
    if (isDisposed) {
      throw StateError('Cannot copy a disposed array.');
    }

    final NDArray<T> result;
    if (out != null) {
      if (out.isDisposed) {
        throw StateError('Cannot copy to a disposed array.');
      }
      if (!out.isWriteable ||
          !listEquals(shape, out.shape) ||
          dtype != out.dtype) {
        throw ArgumentError(
          'Destination array must be writeable and have matching shape and dtype (expected shape $shape, dtype $dtype; got shape ${out.shape}, dtype ${out.dtype}).',
        );
      }
      result = out;
    } else {
      result = NDArray<T>.create(shape, dtype);
    }

    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    if (totalSize == 0) {
      return result;
    }

    if (out != null && helpers.sharesMemory(this, result)) {
      if (isContiguous &&
          result.isContiguous &&
          pointer == result.pointer &&
          offsetElements == result.offsetElements) {
        return result;
      }
      return NDArray.scope(() {
        final temp = copy();
        return temp.copy(out: result);
      });
    }

    if (isContiguous && result.isContiguous) {
      _copyContiguousNDArray(this, result, totalSize);
    } else if (result.isContiguous) {
      _copyStridedToContiguous(result);
    } else {
      _copyStrided(result);
    }

    return result;
  }

  /// Returns a copy of the array cast to the specified [targetDType].
  ///
  /// If [copy] is `false` and [targetDType] matches this array's [dtype],
  /// returns this array directly without copying. Otherwise, allocates
  /// and returns a new [NDArray] of type [R].
  ///
  /// **Preconditions:**
  /// - This array must not be disposed.
  ///
  /// **Performance considerations:**
  /// - If [copy] is `false` and dtypes match, returns this in $O(1)$ time and $O(1)$ memory.
  /// - Otherwise, complexity is $O(N)$ where $N$ is the total number of elements.
  ///
  /// **Throws:**
  /// - It is an error if the array is already disposed.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
  /// final b = a.astype(DType.float64);
  /// print(b.dtype); // DType.float64
  /// ```
  NDArray<R> astype<R extends DTypeTag>(
    DType<R> targetDType, {
    bool copy = true,
  }) {
    if (isDisposed) {
      throw StateError('Cannot cast a disposed array.');
    }
    if (!copy && dtype == targetDType) {
      return this as NDArray<R>;
    }
    if (dtype == targetDType) {
      return this.copy() as NDArray<R>;
    }
    return helpers.castNDArray<R>(this, targetDType);
  }

  /// Creates a [SendableNDArray] by copying this array's data into an isolate-transferable buffer.
  ///
  /// This operation copies all elements into a [TransferableTypedData], allowing
  /// the array to be safely passed across Dart Isolates (via `Isolate.run` or [SendPort]).
  /// The receiving isolate can then call [SendableNDArray.materialize] to reconstruct a fresh,
  /// scope-registered [NDArray] that owns its memory on the destination isolate.
  ///
  /// **Preconditions:**
  /// - This array must not be disposed.
  ///
  /// It is an error if this array has been disposed.
  ///
  /// **Performance considerations:**
  /// - Time complexity: $O(N)$ where $N$ is the total number of elements.
  /// - Space complexity: $O(N)$ to allocate the transferable buffer.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<Float64>.ones([100], DType.float64);
  /// final sendable = a.toSendable();
  /// final result = await Isolate.run(() {
  ///   final workerArray = sendable.materialize();
  ///   return workerArray.toSendable();
  /// });
  /// final finalArray = result.materialize();
  /// ```
  SendableNDArray<T> toSendable() => SendableNDArray<T>.fromCopy(this);

  /// Creates a zero-copy [SendableNDArray] borrowing the raw native memory address of this array.
  ///
  /// **Safety Contract:**
  /// - This array **must remain alive and undisposed** on the sending isolate for the
  ///   entire duration that the worker isolate accesses it. Typically, this is achieved
  ///   by keeping this array within an [NDArray.scope] on the main isolate and awaiting
  ///   the completion of `Isolate.run`.
  /// - The worker isolate reconstructs a view over this memory using
  ///   [SendableNDArray.materializeView] (backed by [NDArray.fromPointer] with no finalizer).
  /// - Concurrent unsynchronized writes to overlapping memory regions from multiple isolates
  ///   result in undefined behavior.
  ///
  /// **Preconditions:**
  /// - This array must not be disposed.
  ///
  /// It is an error if this array has been disposed.
  ///
  /// **Performance considerations:**
  /// - Time complexity: $O(1)$.
  /// - Space complexity: $O(1)$.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<Float64>.zeros([1000], DType.float64);
  /// final borrowed = a.toSendableBorrow();
  /// await Isolate.run(() {
  ///   final view = borrowed.materializeView();
  ///   view.fill(42.0 as Float64);
  /// });
  /// print(a[0]); // 42.0
  /// ```
  SendableNDArray<T> toSendableBorrow() =>
      SendableNDArray<T>.unsafeBorrow(this);

  /// Internal helper to copy contiguous array elements to another contiguous array,
  /// bypassing generic type constraints.
  @internal
  void copyToContiguous(NDArray dest) {
    if (isDisposed || dest.isDisposed) {
      throw StateError('Cannot copy to or from a disposed array.');
    }
    if (!listEquals(shape, dest.shape) || dtype != dest.dtype) {
      throw ArgumentError('Mismatched shape or dtype in copyToContiguous.');
    }
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    if (totalSize == 0) {
      return;
    }
    if (!isContiguous || !dest.isContiguous) {
      throw ArgumentError(
        'Both arrays must be contiguous in copyToContiguous.',
      );
    }
    _copyContiguousNDArray(this, dest, totalSize);
  }

  void _copyStridedToContiguous(NDArray<T> dest) {
    final marker = ScratchArena.marker;
    try {
      final cShape = ScratchArena.copyInts(shape);
      final cStridesSrc = ScratchArena.copyInts(strides);
      switch (dtype) {
        case DType.float64 || DType.int64 || DType.uint64:
          s_flatten_double(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case DType.float32 || DType.int32 || DType.uint32:
          s_flatten_float(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case DType.float16 || DType.bfloat16 || DType.int16 || DType.uint16:
          s_flatten_int16(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case DType.int8 || DType.uint8 || DType.boolean:
          s_flatten_uint8(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case DType.complex128:
          s_flatten_complex128(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case DType.complex64:
          s_flatten_complex64(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }

  void _copyStrided(NDArray<T> dest) {
    final marker = ScratchArena.marker;
    try {
      final rank = shape.length;
      final cShape = rank == 0 ? ffi.nullptr : ScratchArena.copyInts(shape);
      final cStridesSrc = rank == 0
          ? ffi.nullptr
          : ScratchArena.copyInts(strides);
      final cStridesDest = rank == 0
          ? ffi.nullptr
          : ScratchArena.copyInts(dest.strides);
      native_copy_strided(
        pointer,
        cStridesSrc,
        dest.pointer,
        cStridesDest,
        cShape,
        rank,
        dtype.byteWidth,
      );
    } finally {
      ScratchArena.reset(marker);
    }
  }

  /// Returns a flattened one-dimensional view or copy of this array.
  ///
  /// **Preconditions:**
  /// - The array must not be disposed.
  ///
  /// It is an error if the array has been disposed.
  ///
  /// **View vs. Copy Behavior:**
  /// - **Returns a VIEW** when the array is **C-contiguous** (`isContiguous` is `true`).
  ///   Shares the exact same backing memory and raw pointer (`_pointer`). Mutations made to the
  ///   returned raveled array will directly affect the original array (and vice versa).
  /// - **Returns a COPY** when the array is **non-contiguous / strided** (e.g. sliced views or
  ///   transposed matrices). Allocates a brand-new contiguous C heap array and duplicates elements.
  ///   Mutations made to the returned raveled array are completely decoupled and will **not** affect
  ///   the original array.
  ///
  /// **Performance considerations:**
  /// - If the array [isContiguous], this returns a zero-allocation, zero-copy 1D view sharing backing memory ($O(1)$ complexity).
  /// - Otherwise, falls back to returning a deep flattened copy ($O(N)$ complexity).
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
  /// final r = a.ravel();
  /// print(r.shape); // [4]
  /// ```
  NDArray<T> ravel() {
    if (isDisposed) throw StateError('Cannot access a disposed NDArray.');
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    if (isContiguous) {
      return NDArray._(
        _pointer,
        _data,
        _parent ?? this,
        shape: [totalSize],
        strides: [1],
        offsetElements: offsetElements,
        allocPointer: _allocPointer,
        dtype: dtype,
      );
    } else {
      return flatten();
    }
  }

  /// Fills the array with [value] in-place.
  ///
  /// **Performance considerations:**
  /// - For contiguous arrays, utilizes native C SIMD fill kernels (`v_fill_*`).
  /// - For strided or non-contiguous views, utilizes native C strided fill kernels (`s_fill_*`).
  /// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<Float64>.create([100], DType.float64);
  /// a.fill(42.0);
  /// ```
  void fillUntyped(Object? value) {
    if (isDisposed) {
      throw StateError('Cannot fill an array whose memory has been freed.');
    }
    if (!isWriteable) {
      throw ArgumentError(
        'Assignment destination is a read-only broadcast view.',
      );
    }
    final size = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    if (size == 0) return;

    if (isContiguous) {
      switch (dtype) {
        case DType.float64:
          v_fill_double(_pointer.cast(), (value as num).toDouble(), size);
        case DType.float32:
          v_fill_float(_pointer.cast(), (value as num).toDouble(), size);
        case DType.float16:
          v_fill_int16(
            _pointer.cast(),
            Float16Utils.encodeFloat16((value as num).toDouble()),
            size,
          );
        case DType.bfloat16:
          v_fill_int16(
            _pointer.cast(),
            Float16Utils.encodeBFloat16((value as num).toDouble()),
            size,
          );
        case DType.int64 || DType.uint64:
          v_fill_int64(_pointer.cast(), (value as num).toInt(), size);
        case DType.int32 || DType.uint32:
          v_fill_int32(_pointer.cast(), (value as num).toInt(), size);
        case DType.int16 || DType.uint16:
          v_fill_int16(_pointer.cast(), (value as num).toInt(), size);
        case DType.int8 || DType.uint8:
          v_fill_uint8(_pointer.cast(), (value as num).toInt() & 0xFF, size);
        case DType.complex128:
          final c = value is Complex
              ? value
              : Complex((value as num).toDouble(), 0.0);
          v_fill_complex128(_pointer.cast(), c.real, c.imag, size);
        case DType.complex64:
          final c = value is Complex
              ? value
              : Complex((value as num).toDouble(), 0.0);
          v_fill_complex64(_pointer.cast(), c.real, c.imag, size);
        case DType.boolean:
          v_fill_boolean(_pointer.cast(), (value as bool) ? 1 : 0, size);
      }
      return;
    }

    final marker = ScratchArena.marker;
    try {
      final rank = shape.length;
      final cShape = rank == 0 ? ffi.nullptr : ScratchArena.copyInts(shape);
      final cStrides = rank == 0 ? ffi.nullptr : ScratchArena.copyInts(strides);
      switch (dtype) {
        case DType.float64:
          s_fill_double(
            _pointer.cast(),
            cStrides,
            cShape,
            rank,
            (value as num).toDouble(),
          );
        case DType.float32:
          s_fill_float(
            _pointer.cast(),
            cStrides,
            cShape,
            rank,
            (value as num).toDouble(),
          );
        case DType.float16:
          s_fill_int16(
            _pointer.cast(),
            cStrides,
            cShape,
            rank,
            Float16Utils.encodeFloat16((value as num).toDouble()),
          );
        case DType.bfloat16:
          s_fill_int16(
            _pointer.cast(),
            cStrides,
            cShape,
            rank,
            Float16Utils.encodeBFloat16((value as num).toDouble()),
          );
        case DType.int64 || DType.uint64:
          s_fill_int64(
            _pointer.cast(),
            cStrides,
            cShape,
            rank,
            (value as num).toInt(),
          );
        case DType.int32 || DType.uint32:
          s_fill_int32(
            _pointer.cast(),
            cStrides,
            cShape,
            rank,
            (value as num).toInt(),
          );
        case DType.int16 || DType.uint16:
          s_fill_int16(
            _pointer.cast(),
            cStrides,
            cShape,
            rank,
            (value as num).toInt(),
          );
        case DType.int8 || DType.uint8:
          s_fill_uint8(
            _pointer.cast(),
            cStrides,
            cShape,
            rank,
            (value as num).toInt() & 0xFF,
          );
        case DType.complex128:
          final c = value is Complex
              ? value
              : Complex((value as num).toDouble(), 0.0);
          s_fill_complex128(
            _pointer.cast(),
            cStrides,
            cShape,
            rank,
            c.real,
            c.imag,
          );
        case DType.complex64:
          final c = value is Complex
              ? value
              : Complex((value as num).toDouble(), 0.0);
          s_fill_complex64(
            _pointer.cast(),
            cStrides,
            cShape,
            rank,
            c.real,
            c.imag,
          );
        case DType.boolean:
          s_fill_boolean(
            _pointer.cast(),
            cStrides,
            cShape,
            rank,
            (value as bool) ? 1 : 0,
          );
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }

  /// Transposes the dimensions of this array.
  ///
  /// By default, reverses the order of dimensions (equivalent to calling `transposed`).
  /// If [axes] is provided, permutes the dimensions according to the specified
  /// permutation list.
  ///
  /// **Axes Interpretation:**
  /// - The length of [axes] must equal the rank of the array.
  /// - The value `axes[i]` specifies the index of the dimension in the original array
  ///   that will map to the `i`-th dimension of the transposed array.
  /// - Negative indices in [axes] are resolved relative to the end of the dimensions,
  ///   where `-1` represents the last dimension, `-2` represents the second-to-last,
  ///   and so on.
  /// - For example, on a 3-dimensional array with shape `[A, B, C]`:
  ///   - `transpose()` (or `transpose(null)`) results in a shape of `[C, B, A]`.
  ///   - `transpose([1, 0, 2])` results in a shape of `[B, A, C]`.
  ///   - `transpose([-1, -2, -3])` is equivalent to `transpose([2, 1, 0])`, which
  ///     results in a shape of `[C, B, A]`.
  ///
  /// **Preconditions:**
  /// - If provided, the length of [axes] must exactly match the array rank.
  /// - Every axis value must be a valid dimension index (within `[-rank, rank - 1]`).
  /// - [axes] must contain unique, non-duplicate indices.
  ///
  /// It is an error if the array has been disposed, if [axes] length does not match the rank of the array,
  /// if any axis index is out of bounds, or if [axes] contains duplicate indices.
  ///
  /// **Performance considerations:**
  /// - This is a zero-allocation, copy-free view manipulation ($O(1)$ complexity). Strides are
  ///   re-arranged internally without copying any underlying elements.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], DType.float64);
  /// final b = a.transpose(); // b has shape [2, 3] view
  /// ```
  NDArray<T> transpose([List<int>? axes]) {
    if (isDisposed) {
      throw StateError(
        'Cannot access an array or view whose memory has been explicitly freed/disposed!',
      );
    }
    List<int> permutedAxes;
    if (axes == null) {
      permutedAxes = List.generate(shape.length, (i) => shape.length - 1 - i);
    } else {
      if (axes.length != shape.length) {
        throw ArgumentError('Axes must match the rank of the array');
      }
      final seen = <int>{};
      final normAxes = <int>[];
      for (var i = 0; i < axes.length; i++) {
        final axis = axes[i];
        if (axis < -shape.length || axis >= shape.length) {
          throw RangeError.range(axis, -shape.length, shape.length - 1, 'axis');
        }
        final normAxis = axis < 0 ? shape.length + axis : axis;
        if (seen.contains(normAxis)) {
          throw ArgumentError('Axes must be a permutation without duplicates');
        }
        seen.add(normAxis);
        normAxes.add(normAxis);
      }
      permutedAxes = normAxes;
    }

    final newShape = List<int>.filled(shape.length, 0);
    final newStrides = List<int>.filled(shape.length, 0);

    for (var i = 0; i < shape.length; i++) {
      newShape[i] = shape[permutedAxes[i]];
      newStrides[i] = strides[permutedAxes[i]];
    }

    return NDArray._(
      _pointer,
      _data,
      _parent ?? this,
      shape: newShape,
      strides: newStrides,
      offsetElements: offsetElements,
      allocPointer: _allocPointer,
      dtype: dtype,
    );
  }

  /// Returns a view of the array with dimensions reversed.
  ///
  /// Equivalent to calling `transpose()` with no arguments. Reverses the order
  /// of dimensions (e.g., a 3-dimensional array of shape `[A, B, C]` becomes a
  /// view with shape `[C, B, A]`).
  ///
  /// To permute the dimensions in a custom order, use [transpose].
  ///
  /// **Preconditions:**
  /// - The array must not be disposed.
  ///
  /// **Performance considerations:**
  /// - This is a zero-allocation, copy-free view manipulation ($O(1)$ complexity).
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
  /// final t = a.transposed; // shape [2, 2]
  /// ```
  NDArray<T> get transposed => transpose();

  /// Returns the single scalar value of a 0-dimensional array.
  ///
  /// **Preconditions:**
  /// - The array must be 0-dimensional (empty [shape]).
  ///
  /// It is an error if the array has dimensions.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.scalar(42, dtype: DType.int32);
  /// print(a.scalar); // 42
  /// ```
  Object? get scalarRaw {
    if (isDisposed) throw StateError('Cannot access a disposed NDArray.');
    if (shape.isNotEmpty) {
      throw StateError(
        'scalar can only be called on 0-dimensional arrays (has shape $shape)',
      );
    }
    return getCellFlat(0);
  }

  /// Fetches the single scalar element at the specified multi-dimensional [coords].
  ///
  /// **Polymorphic Equivalence:**
  /// Equivalent to calling `this[coords]` via a flat list parameter.
  ///
  /// **Preconditions:**
  /// - [coords] length must match the rank of the array.
  ///
  /// It is an error if coords.length does not match the array rank, or if any coordinate is out of bounds for its dimension.
  Object? getCellUntyped(List<int> coords) {
    if (isDisposed) throw StateError('Cannot access a disposed NDArray.');
    if (coords.length != shape.length) {
      throw ArgumentError(
        'Number of coordinates (${coords.length}) must match array rank (${shape.length})',
      );
    }
    var offset = 0;
    for (var i = 0; i < coords.length; i++) {
      var idx = coords[i];
      if (idx < 0) idx += shape[i];
      if (idx < 0 || idx >= shape[i]) {
        throw RangeError.range(
          coords[i],
          0,
          shape[i] - 1,
          'coordinate at dimension $i',
        );
      }
      offset += idx * strides[i];
    }
    return _checkedData[offsetElements + offset];
  }

  /// Sets the single scalar element at the specified multi-dimensional [coords] to [value].
  ///
  /// **Polymorphic Equivalence:**
  /// Equivalent to calling `this[coords] = value` via a flat list parameter.
  ///
  /// **Preconditions:**
  /// - [coords] length must match the rank of the array.
  ///
  /// It is an error if coords.length does not match the array rank, or if any coordinate is out of bounds for its dimension.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.zeros([2, 2], DType.int32);
  /// a.setCell([0, 1], 42);
  /// ```
  void setCellUntyped(List<int> coords, Object? value) {
    if (isDisposed) throw StateError('Cannot access a disposed NDArray.');
    if (!isWriteable) {
      throw ArgumentError(
        'Assignment destination is a read-only broadcast view.',
      );
    }
    if (coords.length != shape.length) {
      throw ArgumentError(
        'Number of coordinates (${coords.length}) must match array rank (${shape.length})',
      );
    }
    var offset = 0;
    for (var i = 0; i < coords.length; i++) {
      var idx = coords[i];
      if (idx < 0) idx += shape[i];
      if (idx < 0 || idx >= shape[i]) {
        throw RangeError.range(
          coords[i],
          0,
          shape[i] - 1,
          'coordinate at dimension $i',
        );
      }
      offset += idx * strides[i];
    }
    _checkedData[offsetElements + offset] = value;
  }

  /// Internal helper to read the element at a flat index [flatIndex].
  /// Internal helper to read at raw physical storage index [rawOffset].
  @internal
  Object? getCellRawUntyped(int rawOffset) => _checkedData[rawOffset];

  /// Internal helper to write at raw physical storage index [rawOffset].
  @internal
  void setCellRawUntyped(int rawOffset, Object? value) {
    if (!isWriteable) {
      throw ArgumentError(
        'Assignment destination is a read-only broadcast view.',
      );
    }
    _checkedData[rawOffset] = value;
  }

  @internal
  Object? getCellFlatUntyped(int flatIndex) {
    if (isContiguous) {
      return _checkedData[offsetElements + flatIndex];
    }
    var offset = offsetElements;
    var rem = flatIndex;
    for (var i = shape.length - 1; i >= 0; i--) {
      offset += (rem % shape[i]) * strides[i];
      rem ~/= shape[i];
    }
    return _checkedData[offset];
  }

  /// Internal helper to write [value] to the element at a flat index [flatIndex].
  @internal
  void setCellFlatUntyped(int flatIndex, Object? value) {
    if (!isWriteable) {
      throw ArgumentError(
        'Assignment destination is a read-only broadcast view.',
      );
    }
    if (isContiguous) {
      _checkedData[offsetElements + flatIndex] = value;
      return;
    }
    var offset = offsetElements;
    var rem = flatIndex;
    for (var i = shape.length - 1; i >= 0; i--) {
      offset += (rem % shape[i]) * strides[i];
      rem ~/= shape[i];
    }
    _checkedData[offset] = value;
  }

  /// Modifies elements where the provided boolean [mask] contains `true`,
  /// drawing sequential values from another [NDArray] [values].
  ///
  /// **Polymorphic Equivalence:**
  /// Equivalent to calling `this[mask] = values`.
  ///
  /// **Preconditions:**
  /// - [mask] must share identical dimensions ([shape]) with this array.
  ///
  /// It is an error if [mask] shape does not match this array's shape, or if [values] has fewer elements than the number of true targets in [mask].
  void setByMask(NDArray<Boolean> mask, NDArray values) {
    if (isDisposed || mask.isDisposed || values.isDisposed) {
      throw StateError('Cannot access a disposed NDArray.');
    }
    if (!isWriteable) {
      throw ArgumentError(
        'Assignment destination is a read-only broadcast view.',
      );
    }
    if (mask.shape.length != shape.length) {
      throw ArgumentError(
        'Mask shape length (${mask.shape.length}) must match array rank (${shape.length})',
      );
    }
    for (var i = 0; i < shape.length; i++) {
      if (mask.shape[i] != shape[i]) {
        throw ArgumentError(
          'Mask dimensions (${mask.shape}) must exactly match array shape ($shape)',
        );
      }
    }

    if (values.shape.isEmpty) {
      setByMaskScalar(mask, _coerceScalar(values.getCellFlat(0)));
      return;
    }

    if (identical(values._rootParent, _rootParent) ||
        identical(mask._rootParent, _rootParent)) {
      final tempVal = identical(values._rootParent, _rootParent)
          ? values.copy()
          : values;
      final tempMask = identical(mask._rootParent, _rootParent)
          ? mask.copy()
          : mask;
      try {
        setByMask(tempMask, tempVal);
      } finally {
        if (!identical(tempVal, values)) tempVal.dispose();
        if (!identical(tempMask, mask)) tempMask.dispose();
      }
      return;
    }

    var valueIndex = 0;
    final selfDType = dtype;

    void walk(int dim, int currentOffset, int maskOffset) {
      if (dim == shape.length) {
        if (mask.getCellRaw(maskOffset)) {
          if (valueIndex >= values.size) {
            throw ArgumentError(
              'Source values array contains fewer elements than the mask targets',
            );
          }
          _checkedData[currentOffset] = _coerceScalar(
            values.getCellFlat(valueIndex++),
            selfDType,
          );
        }
        return;
      }

      for (var i = 0; i < shape[dim]; i++) {
        walk(
          dim + 1,
          currentOffset + i * strides[dim],
          maskOffset + i * mask.strides[dim],
        );
      }
    }

    walk(0, offsetElements, mask.offsetElements);
  }

  /// Modifies elements where the provided boolean binary [mask] contains `true`,
  /// setting them all uniformly to the single scalar [value].
  ///
  /// **Preconditions:**
  /// - [mask] must share identical dimensions ([shape]) with this array.
  void setByMaskScalar(NDArray<Boolean> mask, Object? value) {
    if (isDisposed || mask.isDisposed) {
      throw StateError('Cannot access a disposed NDArray.');
    }
    if (!isWriteable) {
      throw ArgumentError(
        'Assignment destination is a read-only broadcast view.',
      );
    }
    if (mask.shape.length != shape.length) {
      throw ArgumentError(
        'Mask shape length (${mask.shape.length}) must match array rank (${shape.length})',
      );
    }
    for (var i = 0; i < shape.length; i++) {
      if (mask.shape[i] != shape[i]) {
        throw ArgumentError(
          'Mask dimensions (${mask.shape}) must exactly match array shape ($shape)',
        );
      }
    }

    void walk(int dim, int currentOffset, int maskOffset) {
      if (dim == shape.length) {
        if (mask.getCellRaw(maskOffset)) {
          _checkedData[currentOffset] = value;
        }
        return;
      }

      for (var i = 0; i < shape[dim]; i++) {
        walk(
          dim + 1,
          currentOffset + i * strides[dim],
          maskOffset + i * mask.strides[dim],
        );
      }
    }

    walk(0, offsetElements, mask.offsetElements);
  }

  /// Modifies entire sub-matrix rows or slices along the specified [axis] targeted by a 1D list of [indices], setting them all to a single [value].
  ///
  /// **Polymorphic Equivalence:**
  /// When [axis] is `0`, equivalent to calling `this[ [indices] ] = value` (advanced row stack scalar mutation).
  ///
  void setIndicesScalar(
    NDArray<DTypeTag> indices,
    Object? value, {
    int axis = 0,
  }) {
    if (isDisposed || indices.isDisposed) {
      throw StateError('Cannot access a disposed NDArray.');
    }
    if (!isWriteable) {
      throw ArgumentError(
        'Assignment destination is a read-only broadcast view.',
      );
    }
    if (axis < 0 || axis >= shape.length) {
      throw RangeError.range(axis, 0, shape.length - 1, 'axis');
    }

    final sliceShape = List<int>.from(shape)..removeAt(axis);
    final sliceStrides = List<int>.from(strides)..removeAt(axis);

    for (var idx = 0; idx < indices.size; idx++) {
      final rawIdx = indices.getCellFlat(idx) as int;
      var targetIdx = rawIdx;
      if (targetIdx < 0) targetIdx += shape[axis];
      if (targetIdx < 0 || targetIdx >= shape[axis]) {
        throw RangeError.range(
          rawIdx,
          0,
          shape[axis] - 1,
          'index entry at position $idx',
        );
      }

      void overwriteSlice(int dim, int currentOffset) {
        if (dim == sliceShape.length) {
          _checkedData[currentOffset] = value;
          return;
        }
        for (var i = 0; i < sliceShape[dim]; i++) {
          overwriteSlice(dim + 1, currentOffset + i * sliceStrides[dim]);
        }
      }

      overwriteSlice(0, offsetElements + targetIdx * strides[axis]);
    }
  }

  /// Modifies entire sub-matrix rows or slices along the specified [axis] targeted by a 1D list of [indices], overwriting them with sequential values from [values].
  ///
  /// **Polymorphic Equivalence:**
  /// When [axis] is `0`, equivalent to calling `this[ [indices] ] = values` (advanced row stack array assignment).
  ///
  void setIndices(NDArray<DTypeTag> indices, NDArray values, {int axis = 0}) {
    if (isDisposed || indices.isDisposed || values.isDisposed) {
      throw StateError('Cannot access a disposed NDArray.');
    }
    if (!isWriteable) {
      throw ArgumentError(
        'Assignment destination is a read-only broadcast view.',
      );
    }
    if (axis < 0 || axis >= shape.length) {
      throw RangeError.range(axis, 0, shape.length - 1, 'axis');
    }

    final selfDType = dtype;
    if (values.shape.isEmpty) {
      setIndicesScalar(
        indices,
        _coerceScalar(values.getCellFlat(0), selfDType),
        axis: axis,
      );
      return;
    }

    if (identical(values._rootParent, _rootParent) ||
        identical(indices._rootParent, _rootParent)) {
      final tempVal = identical(values._rootParent, _rootParent)
          ? values.copy()
          : values;
      final tempIdx = identical(indices._rootParent, _rootParent)
          ? indices.copy()
          : indices;
      try {
        setIndices(tempIdx, tempVal, axis: axis);
      } finally {
        if (!identical(tempVal, values)) tempVal.dispose();
        if (!identical(tempIdx, indices)) tempIdx.dispose();
      }
      return;
    }

    final sliceShape = List<int>.from(shape)..removeAt(axis);
    final sliceStrides = List<int>.from(strides)..removeAt(axis);

    var valOffset = 0;

    for (var idx = 0; idx < indices.size; idx++) {
      final rawIdx = indices.getCellFlat(idx) as int;
      var targetIdx = rawIdx;
      if (targetIdx < 0) targetIdx += shape[axis];
      if (targetIdx < 0 || targetIdx >= shape[axis]) {
        throw RangeError.range(
          rawIdx,
          0,
          shape[axis] - 1,
          'index entry at position $idx',
        );
      }

      void writeSlice(int dim, int currentOffset) {
        if (dim == sliceShape.length) {
          if (valOffset >= values.size) {
            throw ArgumentError(
              'Source values array contains fewer elements than required for the advanced index allocation',
            );
          }
          _checkedData[currentOffset] = _coerceScalar(
            values.getCellFlat(valOffset++),
            selfDType,
          );
          return;
        }
        for (var i = 0; i < sliceShape[dim]; i++) {
          writeSlice(dim + 1, currentOffset + i * sliceStrides[dim]);
        }
      }

      writeSlice(0, offsetElements + targetIdx * strides[axis]);
    }
  }

  /// Accesses elements of the array polymorphically based on the runtime type of [spec].
  /// Safely coercing scalar inputs to matching array element type [T].
  Object? _coerceScalar(dynamic value, [DType? cachedDType]) {
    if (value is NDArray && (value.shape.isEmpty || value.size == 1)) {
      value = value.getCellFlat(0);
    }
    switch (cachedDType ?? dtype) {
      case DType.float64:
      case DType.float32:
      case DType.float16:
      case DType.bfloat16:
        if (value is num) return value.toDouble();
      case DType.int64:
      case DType.int32:
      case DType.int16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
      case DType.uint8:
        if (value is num) return value.toInt();
      case DType.complex128:
      case DType.complex64:
        if (value is num) return Complex(value.toDouble(), 0.0);
      case DType.boolean:
        if (value is num) return (value != 0);
    }
    return value;
  }

  /// Normalizes heterogeneous selection items into standard [Selector] objects.
  Selector _toSelector(dynamic item, [List<NDArray>? tempAllocations]) {
    if (item is Selector) return item;
    if (item is int) return Index(item);
    if (item is List) {
      if (item.isEmpty) return Indices(const <int>[]);
      if (item.every((e) => e is int)) {
        return Indices(item.cast<int>());
      }
      if (item.every((e) => e is bool)) {
        final boolArr = NDArray<Boolean>.fromList(item.cast<bool>(), [
          item.length,
        ], DType.boolean);
        tempAllocations?.add(boolArr);
        return Mask(BooleanMask(boolArr));
      }
      throw ArgumentError(
        "Selector lists must contain homogeneous integer coordinates or booleans, found: ${item.runtimeType}",
      );
    }
    if (item is NDArray) {
      if (item.isDisposed) {
        throw StateError('Cannot access a disposed NDArray.');
      }
      if (item.dtype == DType.boolean) {
        return Mask(BooleanMask(item as NDArray<Boolean>));
      }
      if (item.dtype.isInteger) {
        final intList = <int>[];
        for (var i = 0; i < item.size; i++) {
          intList.add((item.getCellFlat(i) as num).toInt());
        }
        return Indices(intList);
      }
    }
    if (item is BooleanMask) return Mask(item);
    throw ArgumentError("Unsupported selector item type: ${item.runtimeType}");
  }

  /// Mutates multi-dimensional slices targeted by [selectors] with [value].
  ///
  /// **Preconditions:**
  /// - The array must not be disposed.
  /// - [selectors] length must not exceed the rank of the array.
  ///
  /// It is an error if [selectors] has more elements than the array rank or if [value] cannot be broadcast to the selected slice shape.
  void sliceAssign(List<Selector> selectors, Object? value) {
    if (isDisposed) {
      throw StateError(
        "Cannot access an array or view whose memory has been explicitly freed/disposed!",
      );
    }
    if (!isWriteable) {
      throw ArgumentError(
        'Assignment destination is a read-only broadcast view.',
      );
    }
    _sliceAssign(selectors, value);
  }

  /// Mutates multi-dimensional slices targeted by normalized [selectors].
  void _sliceAssign(List<Selector> selectors, Object? value) {
    if (selectors.length > shape.length) {
      throw ArgumentError(
        "Too many selectors for array rank (${shape.length})",
      );
    }

    NDArray? tempValueCopy;
    if (value is NDArray &&
        (identical(value._rootParent, _rootParent) ||
            helpers.sharesMemory(this, value))) {
      tempValueCopy = value.copy();
      value = tempValueCopy;
    }
    try {
      _sliceAssignImpl(selectors, value);
    } finally {
      tempValueCopy?.dispose();
    }
  }

  void _sliceAssignImpl(List<Selector> selectors, Object? value) {
    if (value is NDArray && (value.shape.isEmpty || value.size == 1)) {
      value = value.getCellFlat(0);
    }

    final processedSelectors = List<Selector>.from(selectors);
    for (var i = 0; i < processedSelectors.length; i++) {
      final sel = processedSelectors[i];
      if (sel is Mask) {
        final mask = sel.mask;
        if (mask.mask.shape.length != 1 || mask.mask.shape[0] != shape[i]) {
          throw ArgumentError(
            "Boolean mask shape must match the size of dimension $i",
          );
        }
        final size = shape[i];
        final List<int> indices;
        final maskMarker = ScratchArena.marker;
        try {
          final pIndices = ScratchArena.allocate<ffi.Int>(
            size * ffi.sizeOf<ffi.Int>(),
          );
          final count = unpack_mask_c(
            mask.mask.pointer.cast(),
            size,
            mask.mask.strides[0],
            pIndices,
          );
          indices = pIndices.cast<ffi.Int32>().asTypedList(count).toList();
        } finally {
          ScratchArena.reset(maskMarker);
        }
        processedSelectors[i] = Indices(indices);
      }
    }

    for (var i = 0; i < processedSelectors.length; i++) {
      final sel = processedSelectors[i];
      if (sel is Index) {
        final idx = sel.value < 0 ? shape[i] + sel.value : sel.value;
        if (idx < 0 || idx >= shape[i]) {
          throw RangeError.range(
            sel.value,
            -shape[i],
            shape[i] - 1,
            'index',
            'Index out of bounds for dimension $i with size ${shape[i]}',
          );
        }
      } else if (sel is Indices) {
        for (final rawIdx in sel.values) {
          final realIdx = rawIdx < 0 ? shape[i] + rawIdx : rawIdx;
          if (realIdx < 0 || realIdx >= shape[i]) {
            throw RangeError.range(
              rawIdx,
              -shape[i],
              shape[i] - 1,
              'indices',
              'Index out of bounds for dimension $i with size ${shape[i]}',
            );
          }
        }
      }
    }

    var isAdvanced = false;
    for (var i = 0; i < shape.length; i++) {
      final sel = i < processedSelectors.length
          ? processedSelectors[i]
          : Slice.all();
      if (sel is Indices) {
        isAdvanced = true;
        break;
      }
    }

    if (!isAdvanced) {
      final view = slice(processedSelectors);
      try {
        if (value is NDArray) {
          NDArray? castedVal;
          NDArray? broadcastedVal;
          try {
            final NDArray typedVal;
            if (value.dtype != dtype) {
              castedVal = helpers.castNDArray<T>(value, dtype);
              typedVal = castedVal;
            } else {
              typedVal = value;
            }
            final NDArray valArr;
            if (listEquals(typedVal.shape, view.shape)) {
              valArr = typedVal;
            } else {
              broadcastedVal = ops.broadcastTo(typedVal, view.shape);
              valArr = broadcastedVal;
            }
            if (helpers.sharesMemory(this, valArr)) {
              final temp = valArr.copy();
              try {
                temp.copy(out: view);
              } finally {
                temp.dispose();
              }
            } else {
              valArr.copy(out: view);
            }
          } finally {
            if (broadcastedVal != null && !identical(broadcastedVal, value)) {
              broadcastedVal.dispose();
            }
            if (castedVal != null && !identical(castedVal, value)) {
              castedVal.dispose();
            }
          }
        } else {
          view.fill(_coerceScalar(value));
        }
      } finally {
        view.dispose();
      }
    } else {
      final targetShape = <int>[];
      for (var i = 0; i < shape.length; i++) {
        final sel = i < processedSelectors.length
            ? processedSelectors[i]
            : Slice.all();
        if (sel is Slice) {
          final step = sel.step;
          final int realStart;
          final int realStop;
          final int dimSize;
          if (step > 0) {
            final startIdx = sel.start == null
                ? 0
                : (sel.start! < 0 ? shape[i] + sel.start! : sel.start!);
            final stopIdx = sel.stop == null
                ? shape[i]
                : (sel.stop! < 0 ? shape[i] + sel.stop! : sel.stop!);
            realStart = startIdx.clamp(0, shape[i]);
            realStop = stopIdx.clamp(0, shape[i]);
            dimSize = realStop > realStart
                ? ((realStop - realStart + step - 1) ~/ step)
                : 0;
          } else {
            final startIdx = sel.start == null
                ? shape[i] - 1
                : (sel.start! < 0 ? shape[i] + sel.start! : sel.start!);
            final stopIdx = sel.stop == null
                ? -1
                : (sel.stop! < 0 ? shape[i] + sel.stop! : sel.stop!);
            realStart = startIdx.clamp(-1, shape[i] - 1);
            realStop = stopIdx.clamp(-1, shape[i] - 1);
            dimSize = realStart > realStop
                ? ((realStart - realStop - step - 1) ~/ -step)
                : 0;
          }
          targetShape.add(dimSize);
        } else if (sel is Indices) {
          targetShape.add(sel.values.length);
        }
      }

      final NDArray? valArr;
      if (value is NDArray) {
        if (listEquals(value.shape, targetShape)) {
          valArr = value;
        } else {
          valArr = ops.broadcastTo(value, targetShape);
        }
      } else {
        valArr = null;
      }

      final currentCoords = List<int>.filled(shape.length, 0);
      final valIndices = List<int>.filled(targetShape.length, 0);

      void walk(int dim, int valDim) {
        if (dim == shape.length) {
          if (valArr != null) {
            setCell(currentCoords, _coerceScalar(valArr.getCell(valIndices)));
          } else {
            setCell(currentCoords, _coerceScalar(value));
          }
          return;
        }

        final sel = dim < processedSelectors.length
            ? processedSelectors[dim]
            : Slice.all();
        if (sel is Index) {
          final idx = sel.value < 0 ? shape[dim] + sel.value : sel.value;
          currentCoords[dim] = idx;
          walk(dim + 1, valDim);
        } else if (sel is Slice) {
          final step = sel.step;
          final int realStart;
          final int realStop;
          if (step > 0) {
            final startIdx = sel.start == null
                ? 0
                : (sel.start! < 0 ? shape[dim] + sel.start! : sel.start!);
            final stopIdx = sel.stop == null
                ? shape[dim]
                : (sel.stop! < 0 ? shape[dim] + sel.stop! : sel.stop!);
            realStart = startIdx.clamp(0, shape[dim]);
            realStop = stopIdx.clamp(0, shape[dim]);
          } else {
            final startIdx = sel.start == null
                ? shape[dim] - 1
                : (sel.start! < 0 ? shape[dim] + sel.start! : sel.start!);
            final stopIdx = sel.stop == null
                ? -1
                : (sel.stop! < 0 ? shape[dim] + sel.stop! : sel.stop!);
            realStart = startIdx.clamp(-1, shape[dim] - 1);
            realStop = stopIdx.clamp(-1, shape[dim] - 1);
          }
          var stepIdx = 0;
          for (
            var idx = realStart;
            step > 0 ? idx < realStop : idx > realStop;
            idx += step
          ) {
            currentCoords[dim] = idx;
            if (valDim < valIndices.length) valIndices[valDim] = stepIdx;
            walk(dim + 1, valDim + 1);
            stepIdx++;
          }
        } else if (sel is Indices) {
          for (var i = 0; i < sel.values.length; i++) {
            final idx = sel.values[i];
            final realIdx = idx < 0 ? shape[dim] + idx : idx;
            currentCoords[dim] = realIdx;
            if (valDim < valIndices.length) valIndices[valDim] = i;
            walk(dim + 1, valDim + 1);
          }
        }
      }

      walk(0, 0);
    }
  }

  /// Fetches elements polymorphically based on selection specification object [spec].
  ///
  /// Corresponds to NumPy multi-dimensional indexing and slicing syntax.
  ///
  /// **Behavior by Parameter Type:**
  /// - **[int]**: Extracts a view along the first axis with rank reduced by 1.
  /// - **[Slice] / [Index] / [Indices] / [Mask] / [BooleanMask]**: Single-axis selector along dimension 0.
  /// - **`List<int>`**: Fetches a single coordinate cell scalar matching array rank.
  /// - **`List<List<int>>`**: Fetches sub-matrix row slices targeting axis 0.
  /// - **`List<dynamic>`**: Multi-dimensional selection objects (e.g. mixed lists of slices, index lists, integers).
  /// - **`NDArray<Boolean>`**:
  ///   - Full mask (`spec.shape == shape`): Calls [applyMask].
  ///   - 1D mask along axis 0: Calls [slice].
  /// - **`NDArray` (integer)**: Performs [take] for fancy index selection.
  ///
  /// **Preconditions:**
  /// - The array must not be disposed.
  /// - [spec] must be a supported indexing / slicing object type or list of selector objects.
  ///
  /// - It is an error if this array has been explicitly disposed.
  /// - It is an error if [spec] is an unsupported type or contains dimension mismatch.
  dynamic operator [](Object? spec) {
    if (isDisposed) {
      throw StateError(
        "Cannot access an array or view whose memory has been explicitly freed/disposed!",
      );
    }
    if (spec is int) {
      return slice([Index(spec)]);
    } else if (spec is Slice ||
        spec is Index ||
        spec is Indices ||
        spec is Mask ||
        spec is BooleanMask) {
      final temps = <NDArray>[];
      try {
        return slice([_toSelector(spec, temps)]);
      } finally {
        for (final t in temps) {
          t.dispose();
        }
      }
    } else if (spec is List) {
      if (spec.isNotEmpty && spec.first is List) {
        final subList = spec.first as List;
        if (subList.every((e) => e is int)) {
          final intIndices = subList.cast<int>().toList();
          return take(intIndices);
        }
      } else if (spec.every((e) => e is int)) {
        if (shape.length == 1 && spec.length > 1) {
          return take(spec.cast<int>());
        }
        if (spec.length != shape.length) {
          throw ArgumentError(
            "Number of coordinate indices (${spec.length}) must match array rank (${shape.length})",
          );
        }
        return getCell(spec.cast<int>());
      }
      final temps = <NDArray>[];
      try {
        final selectors = spec
            .map((e) => _toSelector(e, temps))
            .cast<Selector>()
            .toList();
        return slice(selectors);
      } finally {
        for (final t in temps) {
          t.dispose();
        }
      }
    } else if (spec is NDArray && spec.dtype == DType.boolean) {
      if (spec.isDisposed) {
        throw StateError('Cannot access a disposed NDArray.');
      }
      final boolMask = spec as NDArray<Boolean>;
      if (listEquals(boolMask.shape, shape)) {
        return applyMask(boolMask);
      } else if (boolMask.shape.length == 1 && boolMask.shape[0] == shape[0]) {
        return slice([Mask(BooleanMask(boolMask))]);
      } else {
        throw ArgumentError(
          "Boolean mask shape must exactly match array shape",
        );
      }
    } else if (spec is NDArray && spec.dtype.isInteger) {
      if (spec.isDisposed) {
        throw StateError('Cannot access a disposed NDArray.');
      }
      final intList = <int>[];
      for (var i = 0; i < spec.size; i++) {
        intList.add((spec.getCellFlat(i) as num).toInt());
      }
      final taken = take(intList);
      if (spec.shape.length == 1) {
        return taken;
      }
      final newShape = <int>[...spec.shape, ...shape.sublist(1)];
      final result = NDArray<T>.create(newShape, dtype);
      final flatView = result.reshape(<int>[
        intList.length,
        ...shape.sublist(1),
      ]);
      try {
        taken.copy(out: flatView);
      } finally {
        flatView.dispose();
        taken.dispose();
      }
      return result;
    } else {
      throw ArgumentError(
        "Unsupported selector type for operator []: ${spec.runtimeType}",
      );
    }
  }

  /// Mutates elements polymorphically based on selection specification object [spec].
  ///
  /// Corresponds to NumPy multi-dimensional slice assignment.
  ///
  /// **Behavior by Parameter Type:**
  /// - **[int]**: Modifies row or slice along the first axis.
  /// - **[Slice] / [Index] / [Indices] / [Mask] / [BooleanMask]**: Mutates targeted sub-matrix elements along dimension 0.
  /// - **`List<int>`**: Modifies a single coordinate cell scalar matching array rank.
  /// - **`List<List<int>>`**: Modifies targeted row slices along axis 0.
  /// - **`List<dynamic>`**: Multi-dimensional selection objects (e.g. mixed lists of slices, index lists, integers).
  /// - **`NDArray<Boolean>`**:
  ///   - Full mask (`spec.shape == shape`): Calls [setByMask] or [setByMaskScalar].
  ///   - 1D mask along axis 0: Performs slice assignment along dimension 0.
  /// - **`NDArray` (integer)**: Modifies elements selected by fancy integer array indices.
  ///
  /// **Preconditions:**
  /// - The array must not be disposed.
  /// - [spec] must be a supported selection specification object.
  /// - [value] must match elements or broadcast to the selected shape.
  ///
  /// - It is an error if this array has been explicitly disposed.
  /// - It is an error if [spec] or [value] is unsupported or has dimension mismatch.
  void operator []=(Object? spec, Object? value) {
    if (isDisposed) {
      throw StateError(
        "Cannot access an array or view whose memory has been explicitly freed/disposed!",
      );
    }
    if (!isWriteable) {
      throw ArgumentError(
        'Assignment destination is a read-only broadcast view.',
      );
    }
    if (spec is int) {
      final indices = NDArray<Int32>.fromList([spec], [1], DType.int32);
      NDArray? broadcastedVal;
      try {
        if (value is NDArray) {
          final targetShape = <int>[1, ...shape.sublist(1)];
          final NDArray valArr;
          if (listEquals(value.shape, targetShape)) {
            valArr = value;
          } else {
            broadcastedVal = ops.broadcastTo(value, targetShape);
            valArr = broadcastedVal;
          }
          setIndices(indices, valArr);
        } else {
          setIndicesScalar(indices, _coerceScalar(value));
        }
      } finally {
        indices.dispose();
        if (broadcastedVal != null && !identical(broadcastedVal, value)) {
          broadcastedVal.dispose();
        }
      }
    } else if (spec is Slice ||
        spec is Index ||
        spec is Indices ||
        spec is Mask ||
        spec is BooleanMask) {
      final temps = <NDArray>[];
      try {
        _sliceAssign([_toSelector(spec, temps)], value);
      } finally {
        for (final t in temps) {
          t.dispose();
        }
      }
    } else if (spec is List) {
      if (spec.isNotEmpty && spec.first is List) {
        final subList = spec.first as List;
        if (subList.every((e) => e is int)) {
          final intIndices = subList.cast<int>().toList();
          final indices = NDArray<Int32>.fromList(intIndices, [
            intIndices.length,
          ], DType.int32);
          NDArray? broadcastedVal;
          try {
            if (value is NDArray) {
              final targetShape = <int>[intIndices.length, ...shape.sublist(1)];
              final NDArray valArr;
              if (listEquals(value.shape, targetShape)) {
                valArr = value;
              } else {
                broadcastedVal = ops.broadcastTo(value, targetShape);
                valArr = broadcastedVal;
              }
              setIndices(indices, valArr);
            } else {
              setIndicesScalar(indices, _coerceScalar(value));
            }
          } finally {
            indices.dispose();
            if (broadcastedVal != null && !identical(broadcastedVal, value)) {
              broadcastedVal.dispose();
            }
          }
          return;
        }
      }
      if (spec.every((e) => e is int)) {
        if ((shape.length == 1 && spec.length > 1) ||
            (value is NDArray && value.size > 1)) {
          final intIndices = spec.cast<int>();
          final indices = NDArray<Int32>.fromList(intIndices, [
            intIndices.length,
          ], DType.int32);
          NDArray? broadcastedVal;
          try {
            if (value is NDArray) {
              final targetShape = <int>[intIndices.length, ...shape.sublist(1)];
              final NDArray valArr;
              if (listEquals(value.shape, targetShape)) {
                valArr = value;
              } else {
                broadcastedVal = ops.broadcastTo(value, targetShape);
                valArr = broadcastedVal;
              }
              setIndices(indices, valArr);
            } else {
              setIndicesScalar(indices, _coerceScalar(value));
            }
          } finally {
            indices.dispose();
            if (broadcastedVal != null && !identical(broadcastedVal, value)) {
              broadcastedVal.dispose();
            }
          }
          return;
        }
        if (spec.length != shape.length) {
          throw ArgumentError(
            "Number of coordinate indices (${spec.length}) must match array rank (${shape.length})",
          );
        }
        final intCoords = spec.cast<int>();
        setCell(intCoords, _coerceScalar(value));
        return;
      } else {
        final temps = <NDArray>[];
        try {
          final selectors = spec
              .map((e) => _toSelector(e, temps))
              .cast<Selector>()
              .toList();
          _sliceAssign(selectors, value);
        } finally {
          for (final t in temps) {
            t.dispose();
          }
        }
      }
    } else if (spec is NDArray && spec.dtype == DType.boolean) {
      if (spec.isDisposed) {
        throw StateError('Cannot access a disposed NDArray.');
      }
      final boolMask = spec as NDArray<Boolean>;
      if (listEquals(boolMask.shape, shape)) {
        if (value is NDArray) {
          setByMask(boolMask, value);
        } else {
          setByMaskScalar(boolMask, _coerceScalar(value));
        }
      } else if (boolMask.shape.length == 1 && boolMask.shape[0] == shape[0]) {
        _sliceAssign([Mask(BooleanMask(boolMask))], value);
      } else {
        throw ArgumentError(
          "Boolean mask shape must exactly match array shape",
        );
      }
    } else if (spec is NDArray && spec.dtype.isInteger) {
      if (spec.isDisposed) {
        throw StateError('Cannot access a disposed NDArray.');
      }
      final intList = <int>[];
      for (var i = 0; i < spec.size; i++) {
        intList.add((spec.getCellFlat(i) as num).toInt());
      }
      final indices = NDArray<Int32>.fromList(intList, [
        intList.length,
      ], DType.int32);
      NDArray? broadcastedVal;
      try {
        if (value is NDArray) {
          final targetShape = <int>[...spec.shape, ...shape.sublist(1)];
          final NDArray valArr;
          if (listEquals(value.shape, targetShape)) {
            valArr = value;
          } else {
            broadcastedVal = ops.broadcastTo(value, targetShape);
            valArr = broadcastedVal;
          }
          setIndices(indices, valArr);
        } else {
          setIndicesScalar(indices, _coerceScalar(value));
        }
      } finally {
        indices.dispose();
        if (broadcastedVal != null && !identical(broadcastedVal, value)) {
          broadcastedVal.dispose();
        }
      }
    } else {
      throw ArgumentError(
        "Unsupported selector type for operator []=: ${spec.runtimeType}",
      );
    }
  }

  static bool _scalarIntFitsDType(int value, DType dtype) {
    switch (dtype) {
      case DType.int8:
        return value >= -128 && value <= 127;
      case DType.uint8:
        return value >= 0 && value <= 255;
      case DType.int16:
        return value >= -32768 && value <= 32767;
      case DType.uint16:
        return value >= 0 && value <= 65535;
      case DType.int32:
        return value >= -2147483648 && value <= 2147483647;
      case DType.uint32:
        return value >= 0 && value <= 4294967295;
      case DType.int64:
        return true;
      case DType.uint64:
        return value >= 0;
      default:
        return false;
    }
  }

  static bool _scalarDoubleFitsDType(double value, DType dtype) {
    if (value.isNaN || value.isInfinite) return true;
    final absVal = value.abs();
    switch (dtype) {
      case DType.float16:
        return absVal <= 65504.0;
      case DType.bfloat16:
        return absVal <= 3.38953139e38;
      case DType.float32:
        return absVal <= 3.4028234663852886e38;
      case DType.float64:
        return true;
      default:
        return false;
    }
  }

  NDArray _wrapScalarArena(
    dynamic value,
    List<int> targetShape, [
    DType? targetDType,
  ]) {
    final shape1 = List<int>.filled(targetShape.length, 1);
    final rawPtr = ScratchArena.allocate<ffi.Uint8>(16);

    DType chosenDType;
    if (targetDType != null) {
      if (targetDType.isFloating && value is num) {
        final dVal = value.toDouble();
        if (_scalarDoubleFitsDType(dVal, targetDType)) {
          chosenDType = targetDType;
        } else {
          chosenDType = DType.float64;
        }
      } else if (targetDType.isInteger &&
          value is int &&
          _scalarIntFitsDType(value, targetDType)) {
        chosenDType = targetDType;
      } else if (targetDType.isComplex && (value is Complex || value is num)) {
        chosenDType = targetDType;
      } else if (targetDType == DType.boolean && value is bool) {
        chosenDType = DType.boolean;
      } else if (value is Complex) {
        chosenDType = DType.complex128;
      } else if (value is int) {
        chosenDType = DType.int64;
      } else if (value is double) {
        chosenDType = DType.float64;
      } else if (value is bool) {
        chosenDType = DType.boolean;
      } else {
        throw ArgumentError('Unsupported scalar type: ${value.runtimeType}');
      }
    } else if (value is Complex) {
      chosenDType = DType.complex128;
    } else if (value is int) {
      chosenDType = DType.int64;
    } else if (value is double) {
      chosenDType = DType.float64;
    } else if (value is bool) {
      chosenDType = DType.boolean;
    } else {
      throw ArgumentError('Unsupported scalar type: ${value.runtimeType}');
    }

    switch (chosenDType) {
      case DType.float64:
        rawPtr.cast<ffi.Double>()[0] = (value as num).toDouble();
      case DType.float32:
        rawPtr.cast<ffi.Float>()[0] = (value as num).toDouble();
      case DType.float16:
        rawPtr.cast<ffi.Uint16>()[0] = Float16Utils.encodeFloat16(
          (value as num).toDouble(),
        );
      case DType.bfloat16:
        rawPtr.cast<ffi.Uint16>()[0] = Float16Utils.encodeBFloat16(
          (value as num).toDouble(),
        );
      case DType.int64:
        rawPtr.cast<ffi.Int64>()[0] = (value as num).toInt();
      case DType.int32:
        rawPtr.cast<ffi.Int32>()[0] = (value as num).toInt();
      case DType.int16:
        rawPtr.cast<ffi.Int16>()[0] = (value as num).toInt();
      case DType.int8:
        rawPtr.cast<ffi.Int8>()[0] = (value as num).toInt();
      case DType.uint64:
        rawPtr.cast<ffi.Uint64>()[0] = (value as num).toInt();
      case DType.uint32:
        rawPtr.cast<ffi.Uint32>()[0] = (value as num).toInt();
      case DType.uint16:
        rawPtr.cast<ffi.Uint16>()[0] = (value as num).toInt();
      case DType.uint8:
        rawPtr.cast<ffi.Uint8>()[0] = (value as num).toInt();
      case DType.complex128:
        final c = value is Complex
            ? value
            : Complex((value as num).toDouble(), 0.0);
        final dPtr = rawPtr.cast<ffi.Double>();
        dPtr[0] = c.real;
        dPtr[1] = c.imag;
      case DType.complex64:
        final c = value is Complex
            ? value
            : Complex((value as num).toDouble(), 0.0);
        final fPtr = rawPtr.cast<ffi.Float>();
        fPtr[0] = c.real;
        fPtr[1] = c.imag;
      case DType.boolean:
        rawPtr[0] = (value as bool) ? 1 : 0;
    }

    return ResourceScope.unmanaged(
      () => NDArray.fromPointer(rawPtr.cast(), shape1, chosenDType),
    );
  }

  R _withWrappedScalar<R>(Object? other, R Function(NDArray otherArr) fn) {
    if (other is NDArray) {
      return fn(other);
    }
    final marker = ScratchArena.marker;
    try {
      final otherArr = _wrapScalarArena(other, shape, dtype);
      return fn(otherArr);
    } finally {
      ScratchArena.reset(marker);
    }
  }

  /// Like [_withWrappedScalar], but for operators that return `NDArray<T>`.
  ///
  /// The result of these operators has the dtype of `this`, so an array
  /// operand of a different dtype or an out-of-range/incompatible scalar is
  /// rejected with [ArgumentError]: use the `*As` functions (e.g.
  /// `addAs(a, b, DType.float64)`) for mixed-dtype arithmetic.
  NDArray<T> _withSameDTypeOperand(
    Object? other,
    String operator,
    NDArray<DTypeTag> Function(NDArray<T> otherArr) fn,
  ) {
    if (other is NDArray && other.dtype != dtype) {
      throw ArgumentError.value(
        other,
        'other',
        'Must have the same dtype as the receiver ($dtype) for operator '
            '$operator, but has dtype ${other.dtype}. Use the *As functions '
            '(for example addAs(a, b, DType.float64)) for mixed-dtype '
            'arithmetic',
      );
    }
    return _withWrappedScalar(other, (otherArr) {
          if (otherArr.dtype != dtype) {
            throw ArgumentError.value(
              other,
              'other',
              'Scalar value cannot be represented in receiver dtype $dtype for '
                  'operator $operator (inferred ${otherArr.dtype}).',
            );
          }
          return fn(otherArr as NDArray<T>);
        })
        as NDArray<T>;
  }

  /// Element-wise addition with full broadcasting support.
  ///
  /// A scalar [other] is converted to this array's dtype. An array [other]
  /// must have the same dtype as this array; use `addAs` to combine dtypes.
  NDArray<T> operator +(Object? other) =>
      _withSameDTypeOperand(other, '+', (otherArr) => ops.add(this, otherArr));

  /// Element-wise subtraction with full broadcasting support.
  ///
  /// A scalar [other] is converted to this array's dtype. An array [other]
  /// must have the same dtype as this array; use `subtractAs` to combine
  /// dtypes.
  NDArray<T> operator -(Object? other) => _withSameDTypeOperand(
    other,
    '-',
    (otherArr) => ops.subtract(this, otherArr),
  );

  /// Element-wise multiplication with full broadcasting support.
  ///
  /// A scalar [other] is converted to this array's dtype. An array [other]
  /// must have the same dtype as this array; use `multiplyAs` to combine
  /// dtypes.
  NDArray<T> operator *(Object? other) => _withSameDTypeOperand(
    other,
    '*',
    (otherArr) => ops.multiply(this, otherArr),
  );

  /// Element-wise floor division with full broadcasting support.
  ///
  /// A scalar [other] is converted to this array's dtype. An array [other]
  /// must have the same dtype as this array.
  NDArray<T> operator ~/(Object? other) => _withSameDTypeOperand(
    other,
    '~/',
    (otherArr) => ops.floorDivide(this, otherArr),
  );

  /// Element-wise remainder with full broadcasting support.
  ///
  /// A scalar [other] is converted to this array's dtype. An array [other]
  /// must have the same dtype as this array.
  NDArray<T> operator %(Object? other) => _withSameDTypeOperand(
    other,
    '%',
    (otherArr) => ops.remainder(this, otherArr),
  );

  /// Numerical negative, element-wise.
  NDArray<T> operator -() {
    return ops.negative<T>(this);
  }

  /// Element-wise bitwise AND with full broadcasting support.
  NDArray<T> operator &(Object? other) => _withSameDTypeOperand(
    other,
    '&',
    (otherArr) => ops.bitwiseAnd<T>(this, otherArr),
  );

  /// Element-wise bitwise OR with full broadcasting support.
  NDArray<T> operator |(Object? other) => _withSameDTypeOperand(
    other,
    '|',
    (otherArr) => ops.bitwiseOr<T>(this, otherArr),
  );

  /// Element-wise bitwise XOR with full broadcasting support.
  NDArray<T> operator ^(Object? other) => _withSameDTypeOperand(
    other,
    '^',
    (otherArr) => ops.bitwiseXor<T>(this, otherArr),
  );

  /// Element-wise bitwise NOT.
  NDArray<T> operator ~() {
    return ops.invert<T>(this);
  }

  /// Element-wise left shift with full broadcasting support.
  NDArray<T> operator <<(Object? other) => _withSameDTypeOperand(
    other,
    '<<',
    (otherArr) => ops.leftShift<T>(this, otherArr),
  );

  /// Element-wise right shift with full broadcasting support.
  NDArray<T> operator >>(Object? other) => _withSameDTypeOperand(
    other,
    '>>',
    (otherArr) => ops.rightShift<T>(this, otherArr),
  );

  /// Element-wise greater than comparison (`this > other`) with full broadcasting support.
  ///
  /// Returns a boolean [NDArray] where each element is `true` if the corresponding
  /// element in this array is greater than the element in [other], and `false` otherwise.
  ///
  /// **Preconditions:**
  /// - The shape of [other] (or this array) must be broadcast-compatible with the other.
  /// - Both arrays must be numeric (non-complex).
  ///
  /// It is an error if either array has a complex data type ([DType.complex64] or [DType.complex128]), or if the shapes are not broadcast-compatible.
  ///
  /// **Performance:**
  /// - Uses native C++ SIMD vectorization.
  /// - Time Complexity: $O(N)$ where `N` is the broadcasted size of the arrays.
  /// - Space Complexity: $O(N)$ to allocate the resulting boolean array (unless an `out` parameter is used in the underlying ufunc).
  ///
  /// **Example:**
  /// {@example /example/comparison_operations_example.dart lang=dart}
  ///
  /// Reference: See NumPy's [greater](https://numpy.org/doc/stable/reference/generated/numpy.greater.html).
  NDArray<Boolean> operator >(Object? other) {
    return _withWrappedScalar(other, (otherArr) => ops.greater(this, otherArr));
  }

  /// Element-wise less than comparison (`this < other`) with full broadcasting support.
  ///
  /// Returns a boolean [NDArray] where each element is `true` if the corresponding
  /// element in this array is less than the element in [other], and `false` otherwise.
  ///
  /// **Preconditions:**
  /// - The shape of [other] (or this array) must be broadcast-compatible with the other.
  /// - Both arrays must be numeric (non-complex).
  ///
  /// It is an error if either array has a complex data type ([DType.complex64] or [DType.complex128]), or if the shapes are not broadcast-compatible.
  ///
  /// **Performance:**
  /// - Uses native C++ SIMD vectorization.
  /// - Time Complexity: $O(N)$ where `N` is the broadcasted size.
  ///
  /// **Example:**
  /// {@example /example/comparison_operations_example.dart lang=dart}
  ///
  /// Reference: See NumPy's [less](https://numpy.org/doc/stable/reference/generated/numpy.less.html).
  NDArray<Boolean> operator <(Object? other) {
    return _withWrappedScalar(other, (otherArr) => ops.less(this, otherArr));
  }

  /// Element-wise greater-or-equal comparison (`this >= other`) with full broadcasting support.
  ///
  /// Returns a boolean [NDArray] where each element is `true` if the corresponding
  /// element in this array is greater than or equal to the element in [other], and `false` otherwise.
  ///
  /// **Preconditions:**
  /// - The shape of [other] (or this array) must be broadcast-compatible with the other.
  /// - Both arrays must be numeric (non-complex).
  ///
  /// It is an error if either array has a complex data type ([DType.complex64] or [DType.complex128]), or if the shapes are not broadcast-compatible.
  ///
  /// **Performance:**
  /// - Uses native C++ SIMD vectorization.
  /// - Time Complexity: $O(N)$ where `N` is the broadcasted size.
  ///
  /// **Example:**
  /// {@example /example/comparison_operations_example.dart lang=dart}
  ///
  /// Reference: See NumPy's [greater_equal](https://numpy.org/doc/stable/reference/generated/numpy.greater_equal.html).
  NDArray<Boolean> operator >=(Object? other) {
    return _withWrappedScalar(
      other,
      (otherArr) => ops.greaterEqual(this, otherArr),
    );
  }

  /// Element-wise less-or-equal comparison (`this <= other`) with full broadcasting support.
  ///
  /// Returns a boolean [NDArray] where each element is `true` if the corresponding
  /// element in this array is less than or equal to the element in [other], and `false` otherwise.
  ///
  /// **Preconditions:**
  /// - The shape of [other] (or this array) must be broadcast-compatible with the other.
  /// - Both arrays must be numeric (non-complex).
  ///
  /// It is an error if either array has a complex data type ([DType.complex64] or [DType.complex128]), or if the shapes are not broadcast-compatible.
  ///
  /// **Performance:**
  /// - Uses native C++ SIMD vectorization.
  /// - Time Complexity: $O(N)$ where `N` is the broadcasted size.
  ///
  /// **Example:**
  /// {@example /example/comparison_operations_example.dart lang=dart}
  ///
  /// Reference: See NumPy's [less_equal](https://numpy.org/doc/stable/reference/generated/numpy.less_equal.html).
  NDArray<Boolean> operator <=(Object? other) {
    return _withWrappedScalar(
      other,
      (otherArr) => ops.lessEqual(this, otherArr),
    );
  }

  /// Element-wise equality comparison (`eq(other)`) with full broadcasting support.
  ///
  /// Returns a boolean [NDArray] where each element is `true` if the corresponding
  /// element in this array equals the element in [other], and `false` otherwise.
  ///
  /// Unlike the standard Dart operator `==` which defaults to object identity,
  /// the `==` operator on [NDArray] checks for structural equality of the
  /// arrays themselves (returning a single boolean). In contrast, [eq]
  /// performs element-wise value comparison and returns an [NDArray<Boolean>].
  ///
  /// **Preconditions:**
  /// - The shape of [other] (or this array) must be broadcast-compatible with the other.
  /// - Supports complex types (checks real and imaginary parts).
  ///
  /// It is an error if the shapes are not broadcast-compatible.
  ///
  /// **Performance:**
  /// - Uses native C++ SIMD vectorization.
  /// - Time Complexity: $O(N)$ where `N` is the broadcasted size.
  ///
  /// **Example:**
  /// {@example /example/comparison_operations_example.dart lang=dart}
  ///
  /// Reference: See NumPy's [equal](https://numpy.org/doc/stable/reference/generated/numpy.equal.html).
  NDArray<Boolean> eq(Object? other) {
    return _withWrappedScalar(other, (otherArr) => ops.equal(this, otherArr));
  }

  /// Element-wise inequality comparison (`ne(other)`) with full broadcasting support.
  NDArray<Boolean> ne(Object? other) {
    return _withWrappedScalar(
      other,
      (otherArr) => ops.notEqual(this, otherArr),
    );
  }

  /// Returns a view or copy of the array with elements sliced based on [selectors].
  ///
  /// [selectors] must contain instances of [Selector] subclasses: [Index], [Slice], [Indices], [Mask].
  /// [selectors] can contain integers (to select a single index and reduce rank)
  /// or [Slice] objects (to select a range and keep rank).
  ///
  /// **Example:**
  /// ```dart
  /// final view = arr.slice([Slice(1, 3), 2]);
  /// ```
  NDArray<T> slice(List<Selector> selectors) {
    if (selectors.length > shape.length) {
      throw ArgumentError('Too many selectors for array rank');
    }

    final newShape = <int>[];
    final newStrides = <int>[];
    var offsetElements = 0;
    var isAdvanced = false;

    final processedSelectors = List<Selector>.from(selectors);
    for (var i = 0; i < processedSelectors.length; i++) {
      final sel = processedSelectors[i];
      if (sel is Mask) {
        final mask = sel.mask;
        if (mask.mask.shape.length != 1 || mask.mask.shape[0] != shape[i]) {
          throw ArgumentError(
            'Boolean mask shape must match the size of dimension $i',
          );
        }
        final size = shape[i];
        final List<int> indices;
        final maskMarker = ScratchArena.marker;
        try {
          final pIndices = ScratchArena.allocate<ffi.Int>(
            size * ffi.sizeOf<ffi.Int>(),
          );
          final count = unpack_mask_c(
            mask.mask.pointer.cast(),
            size,
            mask.mask.strides[0],
            pIndices,
          );
          indices = pIndices.cast<ffi.Int32>().asTypedList(count).toList();
        } finally {
          ScratchArena.reset(maskMarker);
        }
        processedSelectors[i] = Indices(indices);
      }
    }

    for (var i = 0; i < shape.length; i++) {
      final selector = i < processedSelectors.length
          ? processedSelectors[i]
          : Slice.all();

      if (selector is Index) {
        final idx = selector.value < 0
            ? shape[i] + selector.value
            : selector.value;
        if (idx < 0 || idx >= shape[i]) {
          throw RangeError.index(
            idx,
            shape,
            'index out of range for dimension $i',
          );
        }
        offsetElements += idx * strides[i];
        // Rank reduction: don't add to newShape or newStrides
      } else if (selector is Slice) {
        final step = selector.step;
        if (step == 0) {
          throw ArgumentError('Slice step cannot be zero.');
        }
        final start = selector.start;
        final stop = selector.stop;
        final length = shape[i];

        final int realStart;
        final int realStop;
        final int dimSize;

        if (step > 0) {
          if (start == null) {
            realStart = 0;
          } else {
            final s = start < 0 ? length + start : start;
            realStart = s.clamp(0, length);
          }
          if (stop == null) {
            realStop = length;
          } else {
            final s = stop < 0 ? length + stop : stop;
            realStop = s.clamp(0, length);
          }
          dimSize = realStop > realStart
              ? ((realStop - realStart + step - 1) ~/ step)
              : 0;
        } else {
          if (start == null) {
            realStart = length - 1;
          } else {
            final s = start < 0 ? length + start : start;
            realStart = s.clamp(-1, length - 1);
          }
          if (stop == null) {
            realStop = -1;
          } else {
            final s = stop < 0 ? length + stop : stop;
            realStop = s.clamp(-1, length - 1);
          }
          dimSize = realStart > realStop
              ? ((realStart - realStop - step - 1) ~/ -step)
              : 0;
        }

        if (dimSize <= 0) {
          newShape.add(0);
          newStrides.add(0);
        } else {
          newShape.add(dimSize);
          newStrides.add(strides[i] * step);
          offsetElements += realStart * strides[i];
        }
      } else if (selector is Indices) {
        isAdvanced = true;
        newShape.add(selector.values.length);
        newStrides.add(0); // Dummy value for now
      }
    }

    if (newShape.contains(0)) {
      offsetElements = 0;
    }

    if (isAdvanced) {
      final result = NDArray<T>.create(newShape, dtype);
      final rank = shape.length;

      final sliceMarker = ScratchArena.marker;
      try {
        final pTypes = ScratchArena.allocate<ffi.Int>(
          rank * ffi.sizeOf<ffi.Int>(),
        );
        final pIndexVals = ScratchArena.allocate<ffi.Int>(
          rank * ffi.sizeOf<ffi.Int>(),
        );
        final pSliceStarts = ScratchArena.allocate<ffi.Int>(
          rank * ffi.sizeOf<ffi.Int>(),
        );
        final pSliceStops = ScratchArena.allocate<ffi.Int>(
          rank * ffi.sizeOf<ffi.Int>(),
        );
        final pSliceSteps = ScratchArena.allocate<ffi.Int>(
          rank * ffi.sizeOf<ffi.Int>(),
        );
        final pIndicesPtrs = ScratchArena.allocate<ffi.Pointer<ffi.Int>>(
          rank * ffi.sizeOf<ffi.Pointer<ffi.Int>>(),
        );
        final pIndicesLens = ScratchArena.allocate<ffi.Int>(
          rank * ffi.sizeOf<ffi.Int>(),
        );

        for (var i = 0; i < rank; i++) {
          final selector = i < processedSelectors.length
              ? processedSelectors[i]
              : Slice.all();

          if (selector is Index) {
            pTypes[i] = 0;
            final idx = selector.value < 0
                ? shape[i] + selector.value
                : selector.value;
            pIndexVals[i] = idx;
            pSliceStarts[i] = 0;
            pSliceStops[i] = 0;
            pSliceSteps[i] = 0;
            pIndicesPtrs[i] = ffi.Pointer.fromAddress(0);
            pIndicesLens[i] = 0;
          } else if (selector is Slice) {
            pTypes[i] = 1;
            pIndexVals[i] = 0;

            final step = selector.step;
            final int startIdx;
            if (selector.start == null) {
              startIdx = step > 0 ? 0 : shape[i] - 1;
            } else {
              final s = selector.start!;
              startIdx = s < 0 ? shape[i] + s : s;
            }

            final int stopIdx;
            if (selector.stop == null) {
              stopIdx = step > 0 ? shape[i] : -1;
            } else {
              final s = selector.stop!;
              stopIdx = s < 0 ? shape[i] + s : s;
            }

            if (shape[i] == 0) {
              pSliceStarts[i] = 0;
              pSliceStops[i] = 0;
            } else if (step > 0) {
              pSliceStarts[i] = startIdx.clamp(0, shape[i]);
              pSliceStops[i] = stopIdx.clamp(0, shape[i]);
            } else {
              pSliceStarts[i] = startIdx.clamp(-1, shape[i] - 1);
              pSliceStops[i] = stopIdx.clamp(-1, shape[i] - 1);
            }
            pSliceSteps[i] = step;
            pIndicesPtrs[i] = ffi.Pointer.fromAddress(0);
            pIndicesLens[i] = 0;
          } else if (selector is Indices) {
            pTypes[i] = 2;
            pIndexVals[i] = 0;
            pSliceStarts[i] = 0;
            pSliceStops[i] = 0;
            pSliceSteps[i] = 0;

            final values = selector.values;
            final pIndices = ScratchArena.allocate<ffi.Int>(
              values.length * ffi.sizeOf<ffi.Int>(),
            );
            for (var j = 0; j < values.length; j++) {
              final idx = values[j];
              final realIdx = idx < 0 ? shape[i] + idx : idx;
              if (realIdx < 0 || realIdx >= shape[i]) {
                throw RangeError.index(
                  realIdx,
                  shape,
                  'index out of range for dimension $i',
                );
              }
              pIndices[j] = realIdx;
            }
            pIndicesPtrs[i] = pIndices;
            pIndicesLens[i] = values.length;
          }
        }

        final pSrcStrides = ScratchArena.copyInts(strides);
        final pSrcShape = ScratchArena.copyInts(shape);

        copy_advanced_c(
          pointer.cast(),
          result.pointer.cast(),
          pSrcStrides,
          pSrcShape,
          rank,
          dtype.byteWidth,
          pTypes,
          pIndexVals,
          pSliceStarts,
          pSliceStops,
          pSliceSteps,
          pIndicesPtrs,
          pIndicesLens,
        );
      } catch (_) {
        result.dispose();
        rethrow;
      } finally {
        ScratchArena.reset(sliceMarker);
      }

      return result;
    }

    return NDArray.view(
      this,
      shape: newShape,
      strides: newStrides,
      offsetElements: offsetElements,
    );
  }

  /// Selects elements along an [axis] using a list of [indices].
  ///
  /// This method corresponds to NumPy's `take` function.
  ///
  /// It is an error if [axis] is out of bounds for the array rank.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
  /// final b = a.take([0, 1], axis: 1); // Select columns 0 and 1
  /// ```
  NDArray<T> take(List<int> indices, {int axis = 0}) {
    final normalizedAxis = axis < 0 ? shape.length + axis : axis;
    if (normalizedAxis < 0 || normalizedAxis >= shape.length) {
      throw RangeError.index(axis, shape, 'axis out of range');
    }
    final selectors = List<Selector>.filled(shape.length, Slice.all());
    selectors[normalizedAxis] = Indices(indices);
    return slice(selectors);
  }

  /// Selects elements matching a boolean [mask].
  ///
  /// The [mask] array must have elements with value true or false.
  /// Returns a 1D array containing the elements where the mask is true.
  ///
  /// It is an error if [mask] shape does not match the target shape.
  NDArray<T> applyMask(NDArray<Boolean> mask) {
    if (isDisposed || mask.isDisposed) {
      throw StateError('Cannot access a disposed NDArray.');
    }
    if (!listEquals(mask.shape, shape)) {
      if (mask.shape.length == 1 && mask.shape[0] == shape[0]) {
        return slice([Mask(BooleanMask(mask))]);
      }
      throw ArgumentError(
        'Boolean mask shape ${mask.shape} must match target shape $shape',
      );
    }
    if (size == 0) {
      return NDArray<T>.create([0], dtype);
    }
    final contigThis = isContiguous ? this : copy();
    final contigMask = mask.isContiguous ? mask : mask.copy();
    try {
      final count = native_count_mask(contigMask.pointer.cast(), size);
      final result = NDArray<T>.create([count], dtype);
      if (count > 0) {
        native_apply_mask(
          dtype.index,
          contigThis.pointer.cast(),
          contigMask.pointer.cast(),
          result.pointer.cast(),
          size,
        );
      }
      return result;
    } finally {
      if (!identical(contigThis, this)) contigThis.dispose();
      if (!identical(contigMask, mask)) contigMask.dispose();
    }
  }

  /// Returns a flat Dart list containing a copy of the elements in this array,
  /// traversed in the logical order defined by its shape and strides.
  ///
  /// Note for [DType.uint64]: Values >= 2^63 are represented as negative integers
  /// in Dart due to Dart's signed 64-bit integer representation.
  List<Object?> toListRaw() {
    if (isDisposed) {
      throw StateError(
        'Cannot access an array or view whose memory has been explicitly freed/disposed!',
      );
    }
    final result = <Object?>[];
    _fillListRecursive(this, List<int>.filled(shape.length, 0), 0, result);
    return result;
  }

  void _fillListRecursive(
    NDArray<T> arr,
    List<int> indices,
    int dim,
    List<Object?> result,
  ) {
    if (dim == arr.shape.length) {
      result.add(arr.getCell(indices));
      return;
    }
    for (var i = 0; i < arr.shape[dim]; i++) {
      indices[dim] = i;
      _fillListRecursive(arr, indices, dim + 1, result);
    }
  }

  /// Returns a new view of this array with a new dimension of size 1 inserted at [axis].
  ///
  /// This method corresponds to NumPy's `expand_dims` function. It does not copy the
  /// underlying memory; it returns a lightweight view of the same array with updated
  /// shape and strides.
  ///
  /// **Preconditions:**
  /// - [axis] must be within the range `[-rank - 1, rank]`, where `rank` is the rank
  ///   (number of dimensions) of this array.
  ///
  /// It is an error if [axis] is out of bounds.
  ///
  /// **Example:**
  /// {@example /example/shape_examples.dart lang=dart}
  NDArray<T> expandDims(int axis) {
    if (isDisposed) throw StateError('Cannot access a disposed NDArray.');
    final rank = shape.length;
    if (axis < -rank - 1 || axis > rank) {
      throw RangeError.range(
        axis,
        -rank - 1,
        rank,
        'axis',
        'Axis out of range for expandDims',
      );
    }

    final normAxis = axis < 0 ? rank + 1 + axis : axis;

    final newShape = List<int>.from(shape);
    final newStrides = List<int>.from(strides);

    newShape.insert(normAxis, 1);

    if (normAxis == rank) {
      newStrides.insert(normAxis, 1);
    } else {
      newStrides.insert(normAxis, strides[normAxis]);
    }

    return NDArray.view(this, shape: newShape, strides: newStrides);
  }

  /// Returns a new view of this array with single-dimensional entries removed from the shape.
  ///
  /// This method corresponds to NumPy's `squeeze` function. It returns a view sharing the
  /// same memory.
  ///
  /// Squeezes either all dimensions of size 1 (if [axis] is omitted/null), or only specific
  /// axes (if [axis] is an `int` or `List<int>`).
  ///
  /// **Preconditions:**
  /// - If an [axis] is specified, the target dimension(s) must have size equal to 1.
  /// - [axis] (or components of it) must be within `[-rank, rank - 1]`.
  ///
  /// It is an error if any specified axis is out of range, or if a specified axis has a size greater than 1.
  ///
  /// **Example:**
  /// {@example /example/shape_examples.dart lang=dart}
  NDArray<T> squeeze({dynamic axis}) {
    if (isDisposed) throw StateError('Cannot access a disposed NDArray.');
    final rank = shape.length;
    final axesToRemove = <int>{};

    if (axis == null) {
      for (var i = 0; i < rank; i++) {
        if (shape[i] == 1) {
          axesToRemove.add(i);
        }
      }
    } else if (axis is int) {
      if (axis < -rank || axis >= rank) {
        throw RangeError.range(axis, -rank, rank - 1, 'axis');
      }
      final normAxis = axis < 0 ? rank + axis : axis;
      if (shape[normAxis] != 1) {
        throw ArgumentError(
          'Cannot squeeze axis $axis: size is ${shape[normAxis]}, must be 1',
        );
      }
      axesToRemove.add(normAxis);
    } else if (axis is List<int>) {
      for (final ax in axis) {
        if (ax < -rank || ax >= rank) {
          throw RangeError.range(ax, -rank, rank - 1, 'axis');
        }
        final normAxis = ax < 0 ? rank + ax : ax;
        if (shape[normAxis] != 1) {
          throw ArgumentError(
            'Cannot squeeze axis $ax: size is ${shape[normAxis]}, must be 1',
          );
        }
        axesToRemove.add(normAxis);
      }
    } else {
      throw ArgumentError('axis must be null, int, or List<int>');
    }

    final newShape = <int>[];
    final newStrides = <int>[];

    for (var i = 0; i < rank; i++) {
      if (!axesToRemove.contains(i)) {
        newShape.add(shape[i]);
        newStrides.add(strides[i]);
      }
    }

    return NDArray.view(this, shape: newShape, strides: newStrides);
  }

  /// Returns a new view of this array with [axis1] and [axis2] interchanged.
  ///
  /// This method corresponds to NumPy's `swapaxes` function.
  ///
  /// **Preconditions:**
  /// - Both [axis1] and [axis2] must be within `[-rank, rank - 1]`.
  ///
  /// It is an error if either axis is out of bounds.
  ///
  /// **Example:**
  /// {@example /example/shape_examples.dart lang=dart}
  NDArray<T> swapaxes(int axis1, int axis2) {
    final rank = shape.length;
    if (axis1 < -rank || axis1 >= rank) {
      throw RangeError.range(axis1, -rank, rank - 1, 'axis1');
    }
    if (axis2 < -rank || axis2 >= rank) {
      throw RangeError.range(axis2, -rank, rank - 1, 'axis2');
    }

    final norm1 = axis1 < 0 ? rank + axis1 : axis1;
    final norm2 = axis2 < 0 ? rank + axis2 : axis2;

    if (norm1 == norm2) return this;

    final newShape = List<int>.from(shape);
    final newStrides = List<int>.from(strides);

    final tempShape = newShape[norm1];
    newShape[norm1] = newShape[norm2];
    newShape[norm2] = tempShape;

    final tempStride = newStrides[norm1];
    newStrides[norm1] = newStrides[norm2];
    newStrides[norm2] = tempStride;

    return NDArray.view(this, shape: newShape, strides: newStrides);
  }

  /// Returns a new view of this array with axes moved from [source] positions to [destination] positions.
  ///
  /// This method corresponds to NumPy's `moveaxis` function. Other axes remain in their original
  /// relative order.
  ///
  /// **Preconditions:**
  /// - [source] and [destination] can be `int` or `List<int>`. If lists, they must have the same length.
  /// - All axis indices must be within `[-rank, rank - 1]`.
  /// - No duplicate axes can be specified in [source] or [destination].
  ///
  /// It is an error if an axis index is out of range, or if inputs have mismatched lengths or contain duplicates.
  ///
  /// **Example:**
  /// {@example /example/shape_examples.dart lang=dart}
  NDArray<T> moveaxis(dynamic source, dynamic destination) {
    final rank = shape.length;

    List<int> srcList;
    List<int> destList;

    if (source is int && destination is int) {
      srcList = [source];
      destList = [destination];
    } else if (source is List<int> && destination is List<int>) {
      if (source.length != destination.length) {
        throw ArgumentError(
          'source and destination lists must have the same length',
        );
      }
      srcList = List<int>.from(source);
      destList = List<int>.from(destination);
    } else {
      throw ArgumentError(
        'source and destination must be both ints or both List<int>',
      );
    }

    final normSrc = <int>[];
    final normDest = <int>[];

    for (var i = 0; i < srcList.length; i++) {
      final s = srcList[i];
      final d = destList[i];

      if (s < -rank || s >= rank) {
        throw RangeError.range(s, -rank, rank - 1, 'source');
      }
      if (d < -rank || d >= rank) {
        throw RangeError.range(d, -rank, rank - 1, 'destination');
      }

      normSrc.add(s < 0 ? rank + s : s);
      normDest.add(d < 0 ? rank + d : d);
    }

    if (normSrc.toSet().length != normSrc.length) {
      throw ArgumentError('Duplicate axes in source are not allowed');
    }
    if (normDest.toSet().length != normDest.length) {
      throw ArgumentError('Duplicate axes in destination are not allowed');
    }

    final remaining = <int>[];
    for (var i = 0; i < rank; i++) {
      if (!normSrc.contains(i)) {
        remaining.add(i);
      }
    }

    final newOrder = List<int>.filled(rank, -1);

    for (var i = 0; i < normDest.length; i++) {
      newOrder[normDest[i]] = normSrc[i];
    }

    var remIdx = 0;
    for (var i = 0; i < rank; i++) {
      if (newOrder[i] == -1) {
        newOrder[i] = remaining[remIdx++];
      }
    }

    final newShape = List<int>.filled(rank, 0);
    final newStrides = List<int>.filled(rank, 0);

    for (var i = 0; i < rank; i++) {
      newShape[i] = shape[newOrder[i]];
      newStrides[i] = strides[newOrder[i]];
    }

    return NDArray.view(this, shape: newShape, strides: newStrides);
  }

  /// Manually free the allocated C memory.
  ///
  /// This method detaches the finalizer to prevent double-freeing.
  /// Calling this on a view does nothing, as the memory is owned by the parent.
  @override
  void dispose() {
    if (_parent != null) return; // Views don't own memory
    if (_isDisposed) return; // Guard against double-free!
    _isDisposed = true;

    ResourceScope.untrack(this);

    final ptrToFree = _allocPointer ?? _pointer;
    if (!_isExternallyOwned) {
      _finalizer.detach(this);
      malloc.free(ptrToFree);
    } else {
      if (_customFinalizerInstance != null) {
        _customFinalizerInstance.detach(this);
      }
      if (_customNativeFinalizer != null) {
        final freeFunc = _customNativeFinalizer
            .asFunction<void Function(ffi.Pointer<ffi.Void>)>();
        freeFunc(ptrToFree);
      }
    }
  }

  @override
  bool operator ==(Object other) => identical(this, other);

  @override
  int get hashCode => identityHashCode(this);

  /// Structural element-wise equality check across two [NDArray] instances.
  ///
  /// Returns `true` if [other] is an [NDArray] with the same [dtype], [shape],
  /// and equal element values across all coordinates (adhering to IEEE 754
  /// `+0.0 == -0.0` and `NaN != NaN` semantics for floating-point and complex arrays).
  bool equals(Object other) {
    if (identical(this, other)) return true;
    if (other is! NDArray) return false;
    if (dtype != other.dtype) return false;
    if (!listEquals(shape, other.shape)) return false;

    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    if (totalSize == 0) return true;

    // 1. High-speed direct C memcmp block byte check for C-contiguous integer/boolean arrays
    if (isContiguous &&
        other.isContiguous &&
        (dtype.isInteger || dtype == DType.boolean)) {
      final byteSize = totalSize * dtype.byteWidth;
      return custom_memcmp(pointer, other.pointer, byteSize) == 0;
    }

    // 2. Zero-allocation strided C comparison (also used for float/complex contiguous to preserve IEEE 754 +0.0 == -0.0 and NaN != NaN semantics)
    final marker = ScratchArena.marker;
    try {
      final cShape = shape.isEmpty ? ffi.nullptr : ScratchArena.copyInts(shape);
      final cStridesA = strides.isEmpty
          ? ffi.nullptr
          : ScratchArena.copyInts(strides);
      final cStridesB = other.strides.isEmpty
          ? ffi.nullptr
          : ScratchArena.copyInts(other.strides);
      return ndarray_equals(
            dtype.index,
            pointer,
            cStridesA,
            other.pointer,
            cStridesB,
            cShape,
            shape.length,
          ) ==
          1;
    } finally {
      ScratchArena.reset(marker);
    }
  }

  /// Computes an $O(N)$ structural hash code over the array's [dtype], [shape], and element contents.
  ///
  /// Consistent with [equals]: if `a.equals(b)` is `true`, then `a.contentHashCode == b.contentHashCode`.
  int get contentHashCode {
    final baseHash = Object.hash(dtype, Object.hashAll(shape));

    final int elementsHash;
    final marker = ScratchArena.marker;
    try {
      final cShape = ScratchArena.copyInts(shape);
      final cStrides = ScratchArena.copyInts(strides);
      switch (dtype) {
        case DType.float64:
          elementsHash = s_hash_double(
            pointer.cast(),
            cStrides,
            cShape,
            shape.length,
            isContiguous ? 1 : 0,
          );
        case DType.int64 || DType.uint64:
          elementsHash = s_hash_int64(
            pointer.cast(),
            cStrides,
            cShape,
            shape.length,
            isContiguous ? 1 : 0,
          );
        case DType.float32:
          elementsHash = s_hash_float(
            pointer.cast(),
            cStrides,
            cShape,
            shape.length,
            isContiguous ? 1 : 0,
          );
        case DType.int32 || DType.uint32:
          elementsHash = s_hash_int32(
            pointer.cast(),
            cStrides,
            cShape,
            shape.length,
            isContiguous ? 1 : 0,
          );
        case DType.float16 || DType.bfloat16:
          var h = 2166136261;
          final rawBuffer = pointer.cast<ffi.Uint16>();
          final isF16 = dtype == DType.float16;
          void hashBits(int bits) {
            if (bits == 0x8000) {
              bits = 0;
            } else if (isF16 && (bits & 0x7FFF) > 0x7C00) {
              bits = 0x7E00;
            } else if (!isF16 && (bits & 0x7FFF) > 0x7F80) {
              bits = 0x7FC0;
            }
            h = ((h ^ (bits & 0xFF)) * 16777619) & 0xFFFFFFFF;
            h = ((h ^ ((bits >> 8) & 0xFF)) * 16777619) & 0xFFFFFFFF;
          }
          final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
          if (isContiguous) {
            for (var i = 0; i < totalSize; i++) {
              hashBits(rawBuffer[i]);
            }
          } else if (shape.isEmpty) {
            hashBits(rawBuffer[0]);
          } else if (totalSize > 0) {
            final rank = shape.length;
            final coord = List<int>.filled(rank, 0);
            var offset = 0;
            for (var el = 0; el < totalSize; el++) {
              hashBits(rawBuffer[offset]);
              for (var d = rank - 1; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                  offset += strides[d];
                  break;
                }
                coord[d] = 0;
                offset -= (shape[d] - 1) * strides[d];
              }
            }
          }
          elementsHash = h;
        case DType.int16 || DType.uint16:
          elementsHash = s_hash_int16(
            pointer.cast(),
            cStrides,
            cShape,
            shape.length,
            isContiguous ? 1 : 0,
          );
        case DType.int8 || DType.uint8:
          elementsHash = s_hash_uint8(
            pointer.cast(),
            cStrides,
            cShape,
            shape.length,
            isContiguous ? 1 : 0,
          );
        case DType.complex128:
          elementsHash = s_hash_complex128(
            pointer.cast(),
            cStrides,
            cShape,
            shape.length,
            isContiguous ? 1 : 0,
          );
        case DType.complex64:
          elementsHash = s_hash_complex64(
            pointer.cast(),
            cStrides,
            cShape,
            shape.length,
            isContiguous ? 1 : 0,
          );
        case DType.boolean:
          elementsHash = s_hash_boolean(
            pointer.cast(),
            cStrides,
            cShape,
            shape.length,
            isContiguous ? 1 : 0,
          );
      }
    } finally {
      ScratchArena.reset(marker);
    }

    return Object.hash(baseHash, elementsHash);
  }

  @override
  dynamic noSuchMethod(Invocation invocation) {
    if (invocation.isMethod && invocation.positionalArguments.length == 1) {
      final arg = invocation.positionalArguments[0];
      if (invocation.memberName == #+) {
        return this + arg;
      } else if (invocation.memberName == #-) {
        return this - arg;
      } else if (invocation.memberName == #*) {
        return this * arg;
      } else if (invocation.memberName == #/) {
        return NDArrayBaseDivide(this) / arg;
      } else if (invocation.memberName == #~/) {
        return this ~/ arg;
      } else if (invocation.memberName == #%) {
        return this % arg;
      }
    }
    if (invocation.isGetter) {
      if (invocation.memberName == #scalar) return scalarRaw;
      if (invocation.memberName == #data) return dataRaw;
    } else if (invocation.isMethod) {
      final args = invocation.positionalArguments;
      switch (invocation.memberName) {
        case #toList:
          return toListRaw();
        case #getCell:
          return getCellUntyped((args[0] as List).cast<int>());
        case #setCell:
          setCellUntyped((args[0] as List).cast<int>(), args[1]);
          return null;
        case #getCellFlat:
          return getCellFlatUntyped(args[0] as int);
        case #setCellFlat:
          setCellFlatUntyped(args[0] as int, args[1]);
          return null;
        case #getCellRaw:
          return getCellRawUntyped(args[0] as int);
        case #setCellRaw:
          setCellRawUntyped(args[0] as int, args[1]);
          return null;
        case #fill:
          fillUntyped(args[0]);
          return null;
      }
    }
    return super.noSuchMethod(invocation);
  }

  @override
  String toString() => _ndarrayToString(this);
}

final class _NDArrayFloat64 extends NDArray<Float64> {
  _NDArrayFloat64(
    super.pointer,
    super.data,
    super.parent, {
    required super.shape,
    required super.strides,
    super.offsetElements,
    super.allocPointer,
    super.isExternallyOwned,
    super.customNativeFinalizer,
  }) : super._raw();

  @pragma('vm:prefer-inline')
  @override
  DType<Float64> get dtype => DType.float64;
}

final class _NDArrayFloat32 extends NDArray<Float32> {
  _NDArrayFloat32(
    super.pointer,
    super.data,
    super.parent, {
    required super.shape,
    required super.strides,
    super.offsetElements,
    super.allocPointer,
    super.isExternallyOwned,
    super.customNativeFinalizer,
  }) : super._raw();

  @pragma('vm:prefer-inline')
  @override
  DType<Float32> get dtype => DType.float32;
}

final class _NDArrayFloat16 extends NDArray<Float16> {
  _NDArrayFloat16(
    super.pointer,
    super.data,
    super.parent, {
    required super.shape,
    required super.strides,
    super.offsetElements,
    super.allocPointer,
    super.isExternallyOwned,
    super.customNativeFinalizer,
  }) : super._raw();

  @pragma('vm:prefer-inline')
  @override
  DType<Float16> get dtype => DType.float16;
}

final class _NDArrayBFloat16 extends NDArray<BFloat16> {
  _NDArrayBFloat16(
    super.pointer,
    super.data,
    super.parent, {
    required super.shape,
    required super.strides,
    super.offsetElements,
    super.allocPointer,
    super.isExternallyOwned,
    super.customNativeFinalizer,
  }) : super._raw();

  @pragma('vm:prefer-inline')
  @override
  DType<BFloat16> get dtype => DType.bfloat16;
}

final class _NDArrayInt64 extends NDArray<Int64> {
  _NDArrayInt64(
    super.pointer,
    super.data,
    super.parent, {
    required super.shape,
    required super.strides,
    super.offsetElements,
    super.allocPointer,
    super.isExternallyOwned,
    super.customNativeFinalizer,
  }) : super._raw();

  @pragma('vm:prefer-inline')
  @override
  DType<Int64> get dtype => DType.int64;
}

final class _NDArrayInt32 extends NDArray<Int32> {
  _NDArrayInt32(
    super.pointer,
    super.data,
    super.parent, {
    required super.shape,
    required super.strides,
    super.offsetElements,
    super.allocPointer,
    super.isExternallyOwned,
    super.customNativeFinalizer,
  }) : super._raw();

  @pragma('vm:prefer-inline')
  @override
  DType<Int32> get dtype => DType.int32;
}

final class _NDArrayInt16 extends NDArray<Int16> {
  _NDArrayInt16(
    super.pointer,
    super.data,
    super.parent, {
    required super.shape,
    required super.strides,
    super.offsetElements,
    super.allocPointer,
    super.isExternallyOwned,
    super.customNativeFinalizer,
  }) : super._raw();

  @pragma('vm:prefer-inline')
  @override
  DType<Int16> get dtype => DType.int16;
}

final class _NDArrayInt8 extends NDArray<Int8> {
  _NDArrayInt8(
    super.pointer,
    super.data,
    super.parent, {
    required super.shape,
    required super.strides,
    super.offsetElements,
    super.allocPointer,
    super.isExternallyOwned,
    super.customNativeFinalizer,
  }) : super._raw();

  @pragma('vm:prefer-inline')
  @override
  DType<Int8> get dtype => DType.int8;
}

final class _NDArrayUint64 extends NDArray<Uint64> {
  _NDArrayUint64(
    super.pointer,
    super.data,
    super.parent, {
    required super.shape,
    required super.strides,
    super.offsetElements,
    super.allocPointer,
    super.isExternallyOwned,
    super.customNativeFinalizer,
  }) : super._raw();

  @pragma('vm:prefer-inline')
  @override
  DType<Uint64> get dtype => DType.uint64;
}

final class _NDArrayUint32 extends NDArray<Uint32> {
  _NDArrayUint32(
    super.pointer,
    super.data,
    super.parent, {
    required super.shape,
    required super.strides,
    super.offsetElements,
    super.allocPointer,
    super.isExternallyOwned,
    super.customNativeFinalizer,
  }) : super._raw();

  @pragma('vm:prefer-inline')
  @override
  DType<Uint32> get dtype => DType.uint32;
}

final class _NDArrayUint16 extends NDArray<Uint16> {
  _NDArrayUint16(
    super.pointer,
    super.data,
    super.parent, {
    required super.shape,
    required super.strides,
    super.offsetElements,
    super.allocPointer,
    super.isExternallyOwned,
    super.customNativeFinalizer,
  }) : super._raw();

  @pragma('vm:prefer-inline')
  @override
  DType<Uint16> get dtype => DType.uint16;
}

final class _NDArrayUint8 extends NDArray<Uint8> {
  _NDArrayUint8(
    super.pointer,
    super.data,
    super.parent, {
    required super.shape,
    required super.strides,
    super.offsetElements,
    super.allocPointer,
    super.isExternallyOwned,
    super.customNativeFinalizer,
  }) : super._raw();

  @pragma('vm:prefer-inline')
  @override
  DType<Uint8> get dtype => DType.uint8;
}

final class _NDArrayComplex128 extends NDArray<Complex128> {
  _NDArrayComplex128(
    super.pointer,
    super.data,
    super.parent, {
    required super.shape,
    required super.strides,
    super.offsetElements,
    super.allocPointer,
    super.isExternallyOwned,
    super.customNativeFinalizer,
  }) : super._raw();

  @pragma('vm:prefer-inline')
  @override
  DType<Complex128> get dtype => DType.complex128;
}

final class _NDArrayComplex64 extends NDArray<Complex64> {
  _NDArrayComplex64(
    super.pointer,
    super.data,
    super.parent, {
    required super.shape,
    required super.strides,
    super.offsetElements,
    super.allocPointer,
    super.isExternallyOwned,
    super.customNativeFinalizer,
  }) : super._raw();

  @pragma('vm:prefer-inline')
  @override
  DType<Complex64> get dtype => DType.complex64;
}

final class _NDArrayBoolean extends NDArray<Boolean> {
  _NDArrayBoolean(
    super.pointer,
    super.data,
    super.parent, {
    required super.shape,
    required super.strides,
    super.offsetElements,
    super.allocPointer,
    super.isExternallyOwned,
    super.customNativeFinalizer,
  }) : super._raw();

  @pragma('vm:prefer-inline')
  @override
  DType<Boolean> get dtype => DType.boolean;
}

/// Arithmetic operators (`+`, `-`, `*`, `~/`, `%`) preserving the concrete
/// dtype tag [T] of the left operand.
extension NDArrayArithmetic<T extends DTypeTag> on NDArray<T> {}

/// True division operator (`/`) inferring the concrete math-promoted dtype [M]
/// (`Float64` for integer arrays, and preserving [T] for floating-point and
/// complex arrays).
extension NDArrayDivide<
  T extends DTypeSpec<
    DTypeTag,
    Object?,
    DTypeTag,
    DTypeTag,
    M,
    DTypeTag,
    DTypeTag
  >,
  M extends DTypeTag
>
    on NDArray<T> {
  /// Element-wise true division with full broadcasting support.
  NDArray<M> operator /(Object? other) =>
      _withWrappedScalar(other, (otherArr) => ops.divide(this, otherArr))
          as NDArray<M>;
}

String _ndarrayToString(NDArray arr) {
  if (arr.isDisposed) {
    return '<disposed NDArray<${arr.dtype.name}>>';
  }
  final content = _formatND(arr);
  if (arr.shape.isEmpty || arr.dtype != DType.float64) {
    return '$content, dtype=${arr.dtype.name}';
  }
  return content;
}

String _formatScalar(dynamic value, DType dtype) {
  if (value is double || dtype.isFloating) {
    final d = (value as num).toDouble();
    if (d.isNaN) return 'nan';
    if (d == double.infinity) return 'inf';
    if (d == double.negativeInfinity) return '-inf';
    if (d == 0.0 && 1 / d < 0) return '-0.';
    if (d.truncateToDouble() == d && !d.toString().contains('e')) {
      return '${d.toInt()}.';
    }
    return d.toString();
  } else if (dtype.isInteger) {
    if (value is int) return value.toString();
    return (value as num).toInt().toString();
  } else if (value is bool || dtype == DType.boolean) {
    return value == true ? 'true' : 'false';
  } else if (value is Complex || dtype.isComplex) {
    final c = value as Complex;
    final rStr = _formatScalar(c.real, DType.float64);
    final iStr = _formatScalar(c.imag.abs(), DType.float64);
    final sign = c.imag < 0 ? '-' : '+';
    return '$rStr $sign ${iStr}j';
  }
  return value.toString();
}

String _format1D(NDArray arr) {
  final len = arr.shape[0];
  if (len == 0) return '[]';
  final dtype = arr.dtype;
  final items = <String>[];
  if (len <= 6) {
    for (var i = 0; i < len; i++) {
      items.add(_formatScalar(arr.getCell([i]), dtype));
    }
  } else {
    for (var i = 0; i < 3; i++) {
      items.add(_formatScalar(arr.getCell([i]), dtype));
    }
    items.add('...');
    for (var i = len - 3; i < len; i++) {
      items.add(_formatScalar(arr.getCell([i]), dtype));
    }
  }
  return '[${items.join(", ")}]';
}

String _format2D(NDArray arr, {String indent = ' '}) {
  final numRows = arr.shape[0];
  final numCols = arr.shape[1];
  if (numRows == 0 || numCols == 0) {
    return '[], shape=[$numRows, $numCols]';
  }

  final dtype = arr.dtype;
  final rowIndices = numRows <= 6
      ? List.generate(numRows, (i) => i)
      : [0, 1, 2, -1, numRows - 3, numRows - 2, numRows - 1];

  final colIndices = numCols <= 6
      ? List.generate(numCols, (j) => j)
      : [0, 1, 2, -1, numCols - 3, numCols - 2, numCols - 1];

  final grid = <List<String>>[];
  final colWidths = List<int>.filled(colIndices.length, 0);

  for (final r in rowIndices) {
    if (r == -1) {
      grid.add(['...']);
      continue;
    }
    final rowStrs = <String>[];
    for (var cIdx = 0; cIdx < colIndices.length; cIdx++) {
      final c = colIndices[cIdx];
      final String str;
      if (c == -1) {
        str = '...';
      } else {
        str = _formatScalar(arr.getCell([r, c]), dtype);
      }
      rowStrs.add(str);
      if (str.length > colWidths[cIdx]) {
        colWidths[cIdx] = str.length;
      }
    }
    grid.add(rowStrs);
  }

  final sb = StringBuffer();
  for (var rIdx = 0; rIdx < rowIndices.length; rIdx++) {
    final r = rowIndices[rIdx];
    final isFirst = rIdx == 0;
    final isLast = rIdx == rowIndices.length - 1;

    if (r == -1) {
      sb.write('$indent...');
      if (!isLast) sb.write(',\n');
      continue;
    }

    final rowStrs = grid[rIdx];
    final paddedCells = <String>[];
    for (var cIdx = 0; cIdx < colIndices.length; cIdx++) {
      final str = rowStrs[cIdx];
      paddedCells.add(str.padLeft(colWidths[cIdx]));
    }

    final rowContent = '[${paddedCells.join(", ")}]';
    if (isFirst) {
      sb.write('[$rowContent');
    } else {
      sb.write('$indent$rowContent');
    }
    if (!isLast) {
      sb.write(',\n');
    } else {
      sb.write(']');
    }
  }
  return sb.toString();
}

String _formatND(NDArray arr, {String indent = ''}) {
  final rank = arr.shape.length;
  if (rank == 0) {
    return _formatScalar(arr.getCell([]), arr.dtype);
  }
  if (rank == 1) {
    return _format1D(arr);
  }
  if (rank == 2) {
    return _format2D(arr, indent: indent.isEmpty ? ' ' : '$indent ');
  }

  final dim0 = arr.shape[0];
  if (dim0 == 0) {
    return '[], shape=${arr.shape}';
  }

  final indices = dim0 <= 6
      ? List.generate(dim0, (i) => i)
      : [0, 1, 2, -1, dim0 - 3, dim0 - 2, dim0 - 1];

  final sb = StringBuffer();
  final separator = '\n' * (rank - 1);

  for (var iIdx = 0; iIdx < indices.length; iIdx++) {
    final idx = indices[iIdx];
    final isFirst = iIdx == 0;
    final isLast = iIdx == indices.length - 1;

    if (idx == -1) {
      sb.write('$indent ...,\n$separator');
      continue;
    }

    final subArray = arr[idx] as NDArray;
    final formattedSub = _formatND(subArray, indent: '$indent ');
    if (isFirst) {
      sb.write('[$formattedSub');
    } else {
      sb.write('$indent$formattedSub');
    }
    if (!isLast) {
      sb.write(',$separator');
    } else {
      sb.write(']');
    }
  }
  return sb.toString();
}

/// Structural elements equality check between two lists.
bool listEquals<E>(List<E>? a, List<E>? b) {
  if (identical(a, b)) return true;
  if (a == null || b == null) return false;
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

/// A wrapper class for boolean masks used in advanced indexing.
final class BooleanMask {
  /// The underlying boolean array.
  final NDArray<Boolean> mask;

  /// Creates a new boolean mask. Precondition: mask dtype must be `DType.boolean.`
  BooleanMask(this.mask) {
    if (mask.dtype != DType.boolean) {
      throw ArgumentError('Boolean mask must have DType.boolean');
    }
  }
}

/// Represents a complex number with double precision real and imaginary parts.
final class Complex {
  final double real;
  final double imag;

  Complex(this.real, this.imag);

  /// Adds [other] (a [Complex] or real [num]) to this complex number.
  Complex operator +(Object? other) {
    if (other is Complex) {
      return Complex(real + other.real, imag + other.imag);
    } else if (other is num) {
      return Complex(real + other.toDouble(), imag);
    } else {
      throw ArgumentError(
        'Unsupported operand type for +: ${other.runtimeType}',
      );
    }
  }

  /// Subtracts [other] (a [Complex] or real [num]) from this complex number.
  Complex operator -(Object? other) {
    if (other is Complex) {
      return Complex(real - other.real, imag - other.imag);
    } else if (other is num) {
      return Complex(real - other.toDouble(), imag);
    } else {
      throw ArgumentError(
        'Unsupported operand type for -: ${other.runtimeType}',
      );
    }
  }

  /// Negates both real and imaginary parts of this complex number.
  Complex operator -() => Complex(-real, -imag);

  /// Multiplies this complex number by [other] (a [Complex] or real [num]).
  Complex operator *(Object? other) {
    if (other is Complex) {
      return Complex(
        real * other.real - imag * other.imag,
        real * other.imag + imag * other.real,
      );
    } else if (other is num) {
      final val = other.toDouble();
      return Complex(real * val, imag * val);
    } else {
      throw ArgumentError(
        'Unsupported operand type for *: ${other.runtimeType}',
      );
    }
  }

  /// Divides this complex number by [other] (a [Complex] or real [num]).
  Complex operator /(Object? other) {
    if (other is Complex) {
      final div = other.real * other.real + other.imag * other.imag;
      if (div == 0.0) {
        return Complex(real / 0.0, imag / 0.0);
      }
      return Complex(
        (real * other.real + imag * other.imag) / div,
        (imag * other.real - real * other.imag) / div,
      );
    } else if (other is num) {
      final val = other.toDouble();
      return Complex(real / val, imag / val);
    } else {
      throw ArgumentError(
        'Unsupported operand type for /: ${other.runtimeType}',
      );
    }
  }

  /// Returns the absolute value (magnitude) of this complex number.
  double get abs => math.sqrt(real * real + imag * imag);

  /// Returns the argument (phase) of this complex number in radians.
  double get arg => math.atan2(imag, real);

  /// Returns the natural logarithm of this complex number.
  Complex log() => Complex(math.log(abs), arg);

  /// Returns this complex number raised to the power of [exponent].
  ///
  /// Supports [num] and [Complex] exponents.
  Complex pow(Object? exponent) {
    if (exponent is num) {
      if (exponent == 0) return Complex(1.0, 0.0);
      final r = abs;
      final theta = arg;
      final newR = math.pow(r, exponent);
      final newTheta = theta * exponent;
      return Complex(newR * math.cos(newTheta), newR * math.sin(newTheta));
    } else if (exponent is Complex) {
      if (exponent.real == 0.0 && exponent.imag == 0.0) {
        return Complex(1.0, 0.0);
      }
      // z^w = exp(w * log(z))
      final lz = log();
      final prod = exponent * lz;
      final r = math.exp(prod.real);
      return Complex(r * math.cos(prod.imag), r * math.sin(prod.imag));
    } else {
      throw ArgumentError('Unsupported exponent type: ${exponent.runtimeType}');
    }
  }

  @override
  String toString() => '$real + ${imag}i';

  @override
  bool operator ==(Object other) =>
      other is Complex && real == other.real && imag == other.imag;

  @override
  int get hashCode => Object.hash(real, imag);
}

/// A list view of complex numbers backed by a flat list of doubles.
final class ComplexList extends ListBase<Complex> {
  final List<double> _list;
  ComplexList(this._list);

  /// Returns the backing list of doubles.
  List<double> get backingList => _list;

  @override
  int get length => _list.length ~/ 2;

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot resize ComplexList');
  }

  @override
  Complex operator [](int index) {
    return Complex(_list[index * 2], _list[index * 2 + 1]);
  }

  @override
  void operator []=(int index, Complex value) {
    _list[index * 2] = value.real;
    _list[index * 2 + 1] = value.imag;
  }

  /// Returns the real part of the complex number at [index] without allocating a [Complex] object.
  double getReal(int index) => _list[index * 2];

  /// Returns the imaginary part of the complex number at [index] without allocating a [Complex] object.
  double getImag(int index) => _list[index * 2 + 1];

  /// Sets the real and imaginary parts of the complex number at [index] without allocating a [Complex] object.
  void setRealImag(int index, double real, double imag) {
    _list[index * 2] = real;
    _list[index * 2 + 1] = imag;
  }
}

/// A list view of boolean values backed by a flat list of uint8 bytes on the FFI heap.
final class BoolList extends ListBase<bool> {
  final Uint8List _list;
  BoolList(this._list);

  /// Returns the backing list of raw bytes.
  Uint8List get backingList => _list;

  @override
  int get length => _list.length;

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot resize BoolList');
  }

  @override
  bool operator [](int index) {
    return _list[index] != 0;
  }

  @override
  void operator []=(int index, bool value) {
    _list[index] = value ? 1 : 0;
  }
}

/// Base class for selectors used in slicing and advanced indexing on [NDArray].
///
/// Subclasses represent different indexing modes:
/// - [Index] to extract a single scalar index along a dimension and reduce rank.
/// - [Slice] to extract a continuous range of values along a dimension, keeping rank.
/// - [Indices] to extract specific coordinates along a dimension (advanced indexing).
/// - [Mask] to filter elements based on a boolean mask array.
///
/// Refer to the [Advanced Slicing & Indexing Guide](https://numpy.org/doc/stable/user/basics.indexing.html)
/// for standard concepts of array slicing.
///
/// {@example /example/indexing_example.dart lang=dart}
sealed class Selector {
  const Selector();
}

/// Selects a single index along a dimension of an [NDArray], reducing the rank of the resulting array by 1.
///
/// **Preconditions:**
/// - The [value] index must be within `[-dimSize, dimSize - 1]` where `dimSize` is the size of the targeted dimension.
///
/// - It is an error if [value] is out of bounds during slicing.
///
/// **Example:**
/// ```dart
/// // Select the element at index 1 along the first dimension
/// final rowView = arr.slice([Index(1)]);
/// ```
final class Index extends Selector {
  /// The coordinate index to select. Can be negative to index from the end.
  final int value;

  /// Creates a single index selector with the specified [value].
  Index(this.value);
}

/// Represents a continuous or strided slice of an [NDArray] dimension.
///
/// Similar to Python's `start:stop:step` slice notation. Keeps the rank of the dimension intact.
///
/// **Preconditions:**
/// - [step] must be strictly non-zero.
/// - [start] and [stop], if provided, represent inclusive start and exclusive stop bounds.
///
/// - It is an error if [step] is zero.
///
/// **Example:**
/// ```dart
/// // Select elements from index 1 to 5 with step size of 2
/// final sliceView = arr.slice([Slice(start: 1, stop: 5, step: 2)]);
/// ```
final class Slice extends Selector {
  /// The starting index of the slice (inclusive).
  /// If null, defaults to the beginning of the dimension.
  final int? start;

  /// The ending index of the slice (exclusive).
  /// If null, defaults to the end of the dimension.
  final int? stop;

  /// The step size for the slice. Defaults to 1.
  final int step;

  /// Creates a slice from [start] to [stop] with [step].
  ///
  /// Precondition: [step] must be non-zero.
  const Slice({this.start, this.stop, this.step = 1})
    : assert(step != 0, 'Step cannot be zero');

  /// Creates a slice representing all elements along a dimension.
  const Slice.all({int step = 1}) : this(start: null, stop: null, step: step);
}

/// Selects specific coordinate indices along an [NDArray] dimension (advanced indexing).
///
/// Useful for extracting irregular intervals or custom lists of indices.
///
/// **Preconditions:**
/// - Every index in [values] must be within `[-dimSize, dimSize - 1]` where `dimSize` is the size of the dimension.
///
/// **Example:**
/// ```dart
/// // Extract rows at index 0 and 2 from a 2D matrix
/// final subMatrix = arr.slice([Indices([0, 2])]);
/// ```
final class Indices extends Selector {
  /// The list of specific indices to select.
  final List<int> values;

  /// Creates an indices selector with the specified coordinate [values].
  Indices(this.values);
}

/// Selects elements of an [NDArray] matching a boolean mask array.
///
/// Triggers boolean indexing/masking.
///
/// **Preconditions:**
/// - The [mask] must share identical shape and dimensions with the targeted dimension array.
///
/// **Example:**
/// ```dart
/// // Filter elements matching a boolean condition
/// final maskCondition = arr > 0.5;
/// final positiveValues = arr.slice([Mask(BooleanMask(maskCondition))]);
/// ```
final class Mask extends Selector {
  /// The boolean mask wrapper.
  final BooleanMask mask;

  /// Creates a mask selector wrapping the specified boolean [mask].
  Mask(this.mask);
}

void _copyContiguousNDArray(NDArray src, NDArray dest, int size) {
  if (size <= 0) return;
  custom_memcpy(dest._pointer, src._pointer, size * src.dtype.byteWidth);
}

bool _openblasInitialized = false;

void _initializeOpenBLASOnce() {
  if (_openblasInitialized) return;
  _openblasInitialized = true;
  try {
    openblas_set_num_threads(1);
  } catch (_) {
    // Silently ignore library load/init errors in non-OpenBLAS environments
  }
}

/// Typed element access for an [NDArray].
///
/// The element type [E] is recovered from the array's dtype tag [T] through
/// its [DTypeTag] bound, so `NDArray<Float64>.scalar` has static type
/// `double` and `NDArray<Int32>.scalar` has static type `int`, without
/// [NDArray] needing a second type parameter.
///
/// In code that is generic over all dtypes (`T extends DTypeTag`), [E]
/// resolves to `Object?`, which is the correct answer for dtype-agnostic
/// operations.
extension NDArrayElements<
  T extends DTypeSpec<
    DTypeTag,
    E,
    DTypeTag,
    DTypeTag,
    DTypeTag,
    DTypeTag,
    DTypeTag
  >,
  E
>
    on NDArray<T> {
  /// A Dart list view of the raw C memory, typed as the element type.
  ///
  /// The returned list retains the root owning [NDArray], preventing the
  /// garbage collector from freeing the backing C memory while the list (or
  /// its backing buffer) remains reachable.
  ///
  /// **Restrictions:**
  /// - Fixed length; cannot be resized.
  /// - Becomes invalid if the owning array is explicitly freed via
  ///   [NDArray.dispose] (or when an enclosing [NDArray.scope] exits).
  ///   Accessing it after explicit disposal is undefined behavior.
  ///
  /// **Performance considerations:**
  /// - Time complexity: $O(1)$. No copy is made.
  List<E> get data => dataRaw as List<E>;

  /// The single value of a 0-dimensional array.
  ///
  /// It is an error if the array has any dimensions, or has been disposed.
  ///
  /// **Performance considerations:**
  /// - Time complexity: $O(1)$.
  E get scalar => scalarRaw as E;

  /// The elements of this array as a Dart list, in C (row-major) order.
  ///
  /// **Performance considerations:**
  /// - Time complexity: $O(n)$; allocates a new list.
  List<E> toList() => toListRaw().cast<E>();

  /// The element at the given multi-dimensional [coords].
  ///
  /// Negative coordinates index from the end of the corresponding axis.
  ///
  /// It is an error if [coords] has a different length than [NDArray.ndim],
  /// or if any coordinate is out of range.
  ///
  /// **Performance considerations:**
  /// - Time complexity: $O(\text{ndim})$.
  E getCell(List<int> coords) => getCellUntyped(coords) as E;

  /// Writes [value] at the given multi-dimensional [coords].
  ///
  /// See [getCell] for the coordinate rules.
  void setCell(List<int> coords, E value) => setCellUntyped(coords, value);

  /// The element at [rawOffset] elements into the backing buffer.
  ///
  /// This bypasses shape and stride arithmetic entirely; [rawOffset] is an
  /// index into [data], not a logical index. It is an error if the offset is
  /// out of the buffer's bounds.
  E getCellRaw(int rawOffset) => getCellRawUntyped(rawOffset) as E;

  /// Writes [value] at [rawOffset] elements into the backing buffer.
  ///
  /// See [getCellRaw].
  void setCellRaw(int rawOffset, E value) =>
      setCellRawUntyped(rawOffset, value);

  /// The element at logical flat index [flatIndex] in C (row-major) order.
  ///
  /// Unlike [getCellRaw] this respects the array's shape and strides, so it is
  /// correct for views and transposes.
  ///
  /// **Performance considerations:**
  /// - Time complexity: $O(\text{ndim})$ for a strided array, $O(1)$ when the
  ///   array is C-contiguous.
  E getCellFlat(int flatIndex) => getCellFlatUntyped(flatIndex) as E;

  /// Writes [value] at logical flat index [flatIndex] in C (row-major) order.
  ///
  /// See [getCellFlat].
  void setCellFlat(int flatIndex, E value) =>
      setCellFlatUntyped(flatIndex, value);

  /// Sets every element of this array to [value].
  ///
  /// **Performance considerations:**
  /// - Time complexity: $O(n)$.
  void fill(E value) => fillUntyped(value);
}

/// Fallback element access when the type argument is widened to [DTypeTag].
extension NDArrayBaseElements on NDArray<DTypeTag> {
  List<dynamic> get data => dataRaw;
  dynamic get scalar => scalarRaw;
  List<dynamic> toList() => toListRaw();
  dynamic getCell(List<int> coords) => getCellUntyped(coords);
  void setCell(List<int> coords, Object? value) =>
      setCellUntyped(coords, value);
  dynamic getCellRaw(int rawOffset) => getCellRawUntyped(rawOffset);
  void setCellRaw(int rawOffset, Object? value) =>
      setCellRawUntyped(rawOffset, value);
  dynamic getCellFlat(int flatIndex) => getCellFlatUntyped(flatIndex);
  void setCellFlat(int flatIndex, Object? value) =>
      setCellFlatUntyped(flatIndex, value);
  void fill(Object? value) => fillUntyped(value);
}

/// Fallback true division operator (`/`) when the receiver is typed as [DTypeTag].
extension NDArrayBaseDivide on NDArray<DTypeTag> {
  NDArray<DTypeTag> operator /(Object? other) =>
      _withWrappedScalar(other, (otherArr) => ops.divide(this, otherArr));
}
