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
import 'scratch_arena.dart';

import 'package:openblas/openblas.dart' show openblas_set_num_threads;
import 'package:resource_scope/resource_scope.dart';

import 'operations.dart' as ops;
import 'operations/helpers.dart' as helpers;

import 'float16_utils.dart';

/// Supported data types for the elements of an [NDArray].
extension type const Float64(double value) implements double {}

extension type const Float32(double value) implements double {}

extension type const Float16(double value) implements double {}

extension type const BFloat16(double value) implements double {}

extension type const Int64(int value) implements int {}

extension type const Int32(int value) implements int {}

extension type const Int16(int value) implements int {}

extension type const Int8(int value) implements int {}

extension type const Uint64(int value) implements int {}

extension type const Uint32(int value) implements int {}

extension type const Uint16(int value) implements int {}

extension type const Uint8(int value) implements int {}

extension type const Complex64._(Complex value) implements Complex {
  Complex64(double real, double imag) : this._(Complex(real, imag));
}

extension type const Complex128._(Complex value) implements Complex {
  Complex128(double real, double imag) : this._(Complex(real, imag));
}

/// Supported data types for the elements of an [NDArray].
enum DType<T> {
  float64<Float64>('float64', 8, '<f8'),
  float32<Float32>('float32', 4, '<f4'),
  float16<Float16>('float16', 2, '<f2'),
  bfloat16<BFloat16>('bfloat16', 2, '<b2'),
  int64<Int64>('int64', 8, '<i8'),
  int32<Int32>('int32', 4, '<i4'),
  int16<Int16>('int16', 2, '<i2'),
  int8<Int8>('int8', 1, '<i1'),
  uint64<Uint64>('uint64', 8, '<u8'),
  uint32<Uint32>('uint32', 4, '<u4'),
  uint16<Uint16>('uint16', 2, '<u2'),
  uint8<Uint8>('uint8', 1, '|u1'),
  complex128<Complex128>('complex128', 16, '<c16'),
  complex64<Complex64>('complex64', 8, '<c8'),
  boolean<bool>('boolean', 1, '|b1');

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
///   - **Floor Division (`~/` or `floor_divide`) & Remainder (`%` or `remainder`)**:
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
final class NDArray<T> implements ffi.Finalizable, ScopedResource {
  /// Pointer to the raw C memory allocated for this array.
  final ffi.Pointer<ffi.Void> _pointer;

  /// A Dart list view of the raw C memory.
  ///
  /// **Restrictions:**
  /// - This list has a fixed length and cannot be resized.
  /// - This list becomes invalid as soon as the underlying C memory is freed (via `dispose()` or garbage collection). Accessing it afterwards leads to undefined behavior or crashes.
  final List<T> _data;

  /// A Dart list view of the raw C memory.
  @internal
  List<T> get data {
    if (isDisposed) {
      throw StateError('Cannot access a disposed NDArray.');
    }
    return _data;
  }

  /// The dimensions of the n-dimensional array.
  final List<int> shape;

  /// The number of elements to skip in memory to move to the next position along each dimension.
  @internal
  final List<int> strides;

  /// The logical start of the array in the [data] list.
  final int offsetElements;

  /// The data type of the elements in the array.
  final DType<T> dtype;

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
  bool hasSameShape(NDArray<dynamic> other) => listEquals(shape, other.shape);

  static final _finalizer = ffi.NativeFinalizer(malloc.nativeFree);

  /// Whether to track all created memory-allocating (root) [NDArray]s.
  ///
  /// When enabled, all root [NDArray]s are tracked until they are disposed.
  /// This can be used to detect memory leaks in tests or during debugging.
  ///
  /// Note: enabling this will keep undisposed [NDArray]s in memory,
  /// preventing them from being garbage collected, which allows
  /// [checkNoLeaks] to report them.
  static const bool trackAllocations = ResourceScope.trackAllocations;

  static List<NDArray> get trackedAllocations =>
      ResourceScope.trackedAllocations.whereType<NDArray>().toList();

  static void checkNoLeaks() => ResourceScope.checkNoLeaks();

  static void clearTrackedAllocations() =>
      ResourceScope.clearTrackedAllocations();

  static R scope<R>(R Function() callback) => ResourceScope.scope(callback);

  static NDArray<T> returning<T extends Object>(
    NDArray<T> Function() callback,
  ) => ResourceScope.returning(callback);

  static R unmanaged<R>(R Function() callback) =>
      ResourceScope.unmanaged(callback);

  static bool _checkContiguous(List<int> shape, List<int> strides) {
    final cStrides = computeCStrides(shape);
    if (strides.length != cStrides.length) return false;
    for (var i = 0; i < strides.length; i++) {
      if (shape[i] > 1 && strides[i] != cStrides[i]) return false;
    }
    return true;
  }

  /// Private constructor for internal use and factories.
  NDArray._(
    this._pointer,
    this._data,
    this._parent, {
    required List<int> shape,
    required List<int> strides,
    required this.dtype,
    this.offsetElements = 0,
    bool isExternallyOwned = false,
    ffi.Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>?
    customNativeFinalizer,
  }) : _isExternallyOwned = isExternallyOwned,
       _customNativeFinalizer = customNativeFinalizer,
       _customFinalizerInstance =
           (_parent == null &&
               isExternallyOwned &&
               customNativeFinalizer != null)
           ? ffi.NativeFinalizer(customNativeFinalizer)
           : null,
       shape = List<int>.unmodifiable(shape),
       strides = List<int>.unmodifiable(strides),
       isContiguous = _checkContiguous(shape, strides) {
    _initializeOpenBLASOnce();
    assert(
      identical(T, dynamic) ||
          identical(T, Object) ||
          identical(T, num) ||
          identical(T, Float64) ||
          identical(T, Float32) ||
          identical(T, Float16) ||
          identical(T, BFloat16) ||
          identical(T, Int64) ||
          identical(T, Int32) ||
          identical(T, Int16) ||
          identical(T, Int8) ||
          identical(T, Uint64) ||
          identical(T, Uint32) ||
          identical(T, Uint16) ||
          identical(T, Uint8) ||
          identical(T, Complex128) ||
          identical(T, Complex64) ||
          identical(T, double) ||
          identical(T, int) ||
          identical(T, Complex) ||
          identical(T, bool),
      'NDArray cannot be created with type parameter $T. '
      'Supported types: Float64, Float32, Float16, BFloat16, Int64, Int32, Int16, Int8, Uint64, Uint32, Uint16, Uint8, Complex128, Complex64, double, int, Complex, bool.',
    );
    if (_parent == null) {
      if (!_isExternallyOwned) {
        _finalizer.attach(this, _pointer, detach: this);
      } else if (_customFinalizerInstance != null) {
        _customFinalizerInstance.attach(this, _pointer, detach: this);
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

  @override
  NDArray<T> detachFromScope() {
    ResourceScope.untrack(_rootParent);
    return this;
  }

  @override
  NDArray<T> detachToParentScope() {
    ResourceScope.promoteToParent(_rootParent);
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
  ///
  /// - It is an error if any dimension in [shape] is negative.
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
    if (shape.any((dim) => dim < 0)) {
      throw ArgumentError('Shape dimensions cannot be negative: $shape');
    }
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final finalStrides = strides ?? computeCStrides(shape);

    final allocator = zeroInit ? calloc : malloc;
    ffi.Pointer<ffi.Void> pointer;
    List<T> data;

    switch (dtype) {
      case DType.float64:
        final p = allocator<ffi.Double>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case DType.float32:
        final p = allocator<ffi.Float>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case DType.float16:
        final p = allocator<ffi.Uint16>(totalSize);
        pointer = p.cast();
        data = Float16List(p.asTypedList(totalSize)) as List<T>;
      case DType.bfloat16:
        final p = allocator<ffi.Uint16>(totalSize);
        pointer = p.cast();
        data = BFloat16List(p.asTypedList(totalSize)) as List<T>;
      case DType.int64:
        final p = allocator<ffi.Int64>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case DType.int32:
        final p = allocator<ffi.Int32>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case DType.int16:
        final p = allocator<ffi.Int16>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case DType.int8:
        final p = allocator<ffi.Int8>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case DType.uint64:
        final p = allocator<ffi.Uint64>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case DType.uint32:
        final p = allocator<ffi.Uint32>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case DType.uint16:
        final p = allocator<ffi.Uint16>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case DType.uint8:
        final p = allocator<ffi.Uint8>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case DType.complex128:
        final p = allocator<ffi.Double>(totalSize * 2);
        pointer = p.cast();
        final doubleList = p.asTypedList(totalSize * 2);
        data = ComplexList<Complex128>(doubleList) as List<T>;
      case DType.complex64:
        final p = allocator<ffi.Float>(totalSize * 2);
        pointer = p.cast();
        final floatList = p.asTypedList(totalSize * 2);
        data = ComplexList<Complex64>(floatList) as List<T>;
      case DType.boolean:
        final p = allocator<ffi.Uint8>(totalSize);
        pointer = p.cast();
        final uint8List = p.asTypedList(totalSize);
        data = BoolList(uint8List) as List<T>;
    }

    return NDArray._(
      pointer,
      data,
      null,
      shape: shape,
      strides: finalStrides,
      dtype: dtype,
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
  factory NDArray.fromList(List list, List<int> shape, DType<T> dtype) {
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    if (totalSize != list.length) {
      throw ArgumentError(
        'Total size of shape $shape ($totalSize) must match list length (${list.length})',
      );
    }
    final arr = NDArray<T>.create(shape, dtype);
    final List eagerList = switch (dtype) {
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
      arr.setCellRaw(i, eagerList[i] as T);
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
  factory NDArray.scalar(T value, {DType<T>? dtype}) {
    final resolvedDType = dtype ?? _resolveDType<T>(value);
    return NDArray.fromList([value], [], resolvedDType);
  }

  static DType<T> _resolveDType<T>(T value) {
    if (T == Float64) return DType.float64 as DType<T>;
    if (T == Float32) return DType.float32 as DType<T>;
    if (T == Float16) return DType.float16 as DType<T>;
    if (T == BFloat16) return DType.bfloat16 as DType<T>;
    if (T == Int64) return DType.int64 as DType<T>;
    if (T == Int32) return DType.int32 as DType<T>;
    if (T == Int16) return DType.int16 as DType<T>;
    if (T == Int8) return DType.int8 as DType<T>;
    if (T == Uint64) return DType.uint64 as DType<T>;
    if (T == Uint32) return DType.uint32 as DType<T>;
    if (T == Uint16) return DType.uint16 as DType<T>;
    if (T == Uint8) return DType.uint8 as DType<T>;
    if (T == Complex128) return DType.complex128 as DType<T>;
    if (T == Complex64) return DType.complex64 as DType<T>;
    if (T == bool || value is bool) return DType.boolean as DType<T>;
    if (T == int || value is int) return DType.int64 as DType<T>;
    if (T == double || value is double) return DType.float64 as DType<T>;
    if (T == Complex || value is Complex) return DType.complex128 as DType<T>;
    return helpers.defaultDType<T>();
  }

  /// Factory to create a new C-contiguous array filled with zeros.
  ///
  /// Backed directly by unmanaged C heap memory pages allocated via `calloc`.
  ///
  /// **Preconditions:**
  /// - All dimensions in [shape] must be strictly non-negative ($\ge 0$).
  ///
  /// - It is an error if any dimension in [shape] is negative.
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
      arr.fill(Complex(1.0, 0.0) as T);
    } else if (dtype == DType.boolean) {
      arr.fill(true as T);
    } else if (dtype.isFloating) {
      arr.fill(1.0 as T);
    } else {
      arr.fill(1 as T);
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
  factory NDArray.full(List<int> shape, T fillValue, {DType<T>? dtype}) {
    final resolvedDType = dtype ?? _resolveDType<T>(fillValue);
    final arr = NDArray<T>.create(shape, resolvedDType);
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
    DType<T>? dtype,
  }) {
    final DType<T> resolvedDType = dtype ?? (DType.float64 as DType<T>);
    if (step == 0.0) {
      throw ArgumentError('Step size cannot be zero.');
    }
    if ((stop > start && step < 0.0) || (stop < start && step > 0.0)) {
      throw ArgumentError('Step size direction must match start/stop range.');
    }
    final length = ((stop - start) / step).ceil();
    final arr = NDArray<T>.create([length], resolvedDType);
    for (var i = 0; i < length; i++) {
      final val = start + i * step;
      if (resolvedDType.isComplex) {
        arr.setCellRaw(i, Complex(val, 0.0) as T);
      } else if (resolvedDType.isInteger) {
        arr.setCellRaw(i, val.toInt() as T);
      } else if (resolvedDType == DType.boolean) {
        arr.setCellRaw(i, (val != 0.0) as T);
      } else {
        arr.setCellRaw(i, val as T);
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
        arr.setCellRaw(i * n + i, 1.0 as T);
      } else if (dtype.isComplex) {
        arr.setCellRaw(i * n + i, Complex(1.0, 0.0) as T);
      } else if (dtype == DType.boolean) {
        arr.setCellRaw(i * n + i, true as T);
      } else {
        arr.setCellRaw(i * n + i, 1 as T);
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
    NDArray<dynamic> parent, {
    required List<int> shape,
    required List<int> strides,
    int offsetElements = 0,
  }) {
    if (shape.any((dim) => dim < 0)) {
      throw ArgumentError('Shape dimensions cannot be negative: $shape');
    }
    final root = parent._rootParent;
    final childLogicalPointer = _offsetPointer(
      parent.pointer,
      offsetElements,
      parent.dtype,
    );
    final cumulativeOffset =
        (childLogicalPointer.address - root._pointer.address) ~/
        parent.dtype.byteWidth;

    final bool isEmpty = shape.contains(0);

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

    final physicalPointer = _offsetPointer(
      root._pointer,
      minPhysicalOffset,
      parent.dtype,
    );
    final int viewSize = isEmpty
        ? 0
        : (maxPhysicalOffset - minPhysicalOffset + 1);
    final List<T> data;

    switch (parent.dtype) {
      case DType.float64:
        data =
            physicalPointer.cast<ffi.Double>().asTypedList(viewSize) as List<T>;
      case DType.float32:
        data =
            physicalPointer.cast<ffi.Float>().asTypedList(viewSize) as List<T>;
      case DType.float16:
        data = Float16List(
          physicalPointer.cast<ffi.Uint16>().asTypedList(viewSize),
        ) as List<T>;
      case DType.bfloat16:
        data = BFloat16List(
          physicalPointer.cast<ffi.Uint16>().asTypedList(viewSize),
        ) as List<T>;
      case DType.int64:
        data =
            physicalPointer.cast<ffi.Int64>().asTypedList(viewSize) as List<T>;
      case DType.int32:
        data =
            physicalPointer.cast<ffi.Int32>().asTypedList(viewSize) as List<T>;
      case DType.int16:
        data =
            physicalPointer.cast<ffi.Int16>().asTypedList(viewSize) as List<T>;
      case DType.int8:
        data =
            physicalPointer.cast<ffi.Int8>().asTypedList(viewSize) as List<T>;
      case DType.uint64:
        data =
            physicalPointer.cast<ffi.Uint64>().asTypedList(viewSize) as List<T>;
      case DType.uint32:
        data =
            physicalPointer.cast<ffi.Uint32>().asTypedList(viewSize) as List<T>;
      case DType.uint16:
        data =
            physicalPointer.cast<ffi.Uint16>().asTypedList(viewSize) as List<T>;
      case DType.uint8:
        data =
            physicalPointer.cast<ffi.Uint8>().asTypedList(viewSize) as List<T>;
      case DType.complex128:
        final p = _offsetPointer(
          root._pointer,
          minPhysicalOffset * 2,
          DType.float64,
        );
        final doubleList = p.cast<ffi.Double>().asTypedList(viewSize * 2);
        data = ComplexList<Complex128>(doubleList) as List<T>;
      case DType.complex64:
        final p = _offsetPointer(
          root._pointer,
          minPhysicalOffset * 2,
          DType.float32,
        );
        final floatList = p.cast<ffi.Float>().asTypedList(viewSize * 2);
        data = ComplexList<Complex64>(floatList) as List<T>;
      case DType.boolean:
        data = BoolList(
          physicalPointer.cast<ffi.Uint8>().asTypedList(viewSize),
        ) as List<T>;
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
    if (shape.any((dim) => dim < 0)) {
      throw ArgumentError('Shape dimensions cannot be negative: $shape');
    }
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final finalStrides = strides ?? computeCStrides(shape);

    List<T> data;
    switch (dtype) {
      case DType.float64:
        data = pointer.cast<ffi.Double>().asTypedList(totalSize) as List<T>;
      case DType.float32:
        data = pointer.cast<ffi.Float>().asTypedList(totalSize) as List<T>;
      case DType.float16:
        data = Float16List(
          pointer.cast<ffi.Uint16>().asTypedList(totalSize),
        ) as List<T>;
      case DType.bfloat16:
        data = BFloat16List(
          pointer.cast<ffi.Uint16>().asTypedList(totalSize),
        ) as List<T>;
      case DType.int64:
        data = pointer.cast<ffi.Int64>().asTypedList(totalSize) as List<T>;
      case DType.int32:
        data = pointer.cast<ffi.Int32>().asTypedList(totalSize) as List<T>;
      case DType.int16:
        data = pointer.cast<ffi.Int16>().asTypedList(totalSize) as List<T>;
      case DType.int8:
        data = pointer.cast<ffi.Int8>().asTypedList(totalSize) as List<T>;
      case DType.uint64:
        data = pointer.cast<ffi.Uint64>().asTypedList(totalSize) as List<T>;
      case DType.uint32:
        data = pointer.cast<ffi.Uint32>().asTypedList(totalSize) as List<T>;
      case DType.uint16:
        data = pointer.cast<ffi.Uint16>().asTypedList(totalSize) as List<T>;
      case DType.uint8:
        data = pointer.cast<ffi.Uint8>().asTypedList(totalSize) as List<T>;
      case DType.complex128:
        data = ComplexList<Complex128>(
          pointer.cast<ffi.Double>().asTypedList(totalSize * 2),
        ) as List<T>;
      case DType.complex64:
        data = ComplexList<Complex64>(
          pointer.cast<ffi.Float>().asTypedList(totalSize * 2),
        ) as List<T>;
      case DType.boolean:
        data = BoolList(
          pointer.cast<ffi.Uint8>().asTypedList(totalSize),
        ) as List<T>;
    }

    return NDArray._(
      pointer,
      data,
      null,
      shape: shape,
      strides: finalStrides,
      dtype: dtype,
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
    final newSize = newShape.isEmpty ? 1 : newShape.reduce((a, b) => a * b);
    if (oldSize != newSize) {
      throw ArgumentError(
        'Total size must not change during reshape (was $oldSize, new is $newSize)',
      );
    }

    if (!isContiguous) {
      final result = NDArray<T>.create(newShape, dtype);
      _copyStridedToContiguous(result);
      return result;
    }

    final newStrides = computeCStrides(newShape);
    return NDArray._(
      _pointer,
      data,
      _parent ?? this,
      shape: newShape,
      strides: newStrides,
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
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final result = NDArray<T>.create([totalSize], dtype);

    if (isContiguous) {
      _copyContiguousNDArray(this, result, totalSize);
    } else {
      _copyStridedToContiguous(result);
    }
    return result;
  }

  /// Returns a deep, C-contiguous copy of this array.
  ///
  /// The copy preserves the logical order and values of the elements defined by
  /// this array's shape and strides. However, the physical memory layout of the
  /// returned array is always contiguous (and its strides are reset to standard
  /// C-contiguous strides).
  ///
  /// **Performance considerations:**
  /// - For C-contiguous layouts, copies elements directly using memmove/memcpy.
  /// - For strided non-contiguous views, uses native C intrinsics to copy into the contiguous destination array.
  ///
  /// It is an error if the array is already disposed.
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
      if (!listEquals(shape, out.shape) || dtype != out.dtype) {
        throw ArgumentError(
          'Destination array must have matching shape and dtype (expected shape $shape, dtype $dtype; got shape ${out.shape}, dtype ${out.dtype}).',
        );
      }
      result = out;
    } else {
      result = NDArray<T>.create(shape, dtype);
    }

    if (isContiguous && result.isContiguous) {
      final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
      _copyContiguousNDArray(this, result, totalSize);
    } else if (result.isContiguous) {
      _copyStridedToContiguous(result);
    } else {
      _copyStrided(result);
    }

    return result;
  }

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
    if (!isContiguous || !dest.isContiguous) {
      throw ArgumentError(
        'Both arrays must be contiguous in copyToContiguous.',
      );
    }
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
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
    helpers.unaryOp<dynamic, dynamic>(
      dest,
      this,
      shape,
      strides,
      dest.strides,
      0,
      offsetElements,
      dest.offsetElements,
      (x) => x,
    );
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
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    if (isContiguous) {
      return NDArray._(
        _pointer,
        data,
        _parent ?? this,
        shape: [totalSize],
        strides: [1],
        dtype: dtype,
      );
    } else {
      return flatten();
    }
  }

  /// Fills the array with [value] in-place.
  ///
  /// **Performance considerations:**
  /// - For contiguous same-type arrays, utilizes native C fill kernels.
  /// - For strided or non-contiguous views, falls back to sequential element walk mutations.
  /// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<Float64>.create([100], DType.float64);
  /// a.fill(42.0);
  /// ```
  void fill(T value) {
    if (isDisposed) {
      throw StateError('Cannot fill an array whose memory has been freed.');
    }
    final size = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);

    if (isContiguous) {
      switch (dtype) {
        case DType.float64:
          if (value is num) {
            v_fill_double(_pointer.cast(), value.toDouble(), size);
            return;
          }
        case DType.float32:
          if (value is num) {
            v_fill_float(_pointer.cast(), value.toDouble(), size);
            return;
          }
        case DType.int64:
          if (value is int) {
            v_fill_int64(_pointer.cast(), value, size);
            return;
          }
        case DType.int32:
          if (value is int) {
            v_fill_int32(_pointer.cast(), value, size);
            return;
          }
        default:
          break;
      }
    }

    // Fallback JIT loop for complex, boolean, or non-contiguous views
    final targetValue = value;

    void fillWalk(int dim, int currentOffset) {
      if (dim == shape.length) {
        data[currentOffset] = targetValue;
        return;
      }
      for (var i = 0; i < shape[dim]; i++) {
        fillWalk(dim + 1, currentOffset + i * strides[dim]);
      }
    }

    fillWalk(0, offsetElements);
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
        var axis = axes[i];
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
      data,
      _parent ?? this,
      shape: newShape,
      strides: newStrides,
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
  T get scalar {
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
  T getCell(List<int> coords) {
    if (coords.length != shape.length) {
      throw ArgumentError(
        'Number of coordinates (${coords.length}) must match array rank (${shape.length})',
      );
    }
    var offset = 0;
    for (var i = 0; i < coords.length; i++) {
      final idx = coords[i];
      if (idx < 0 || idx >= shape[i]) {
        throw RangeError.range(
          idx,
          0,
          shape[i] - 1,
          'coordinate at dimension $i',
        );
      }
      offset += idx * strides[i];
    }
    return data[offsetElements + offset];
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
  void setCell(List<int> coords, T value) {
    if (coords.length != shape.length) {
      throw ArgumentError(
        'Number of coordinates (${coords.length}) must match array rank (${shape.length})',
      );
    }
    var offset = 0;
    for (var i = 0; i < coords.length; i++) {
      final idx = coords[i];
      if (idx < 0 || idx >= shape[i]) {
        throw RangeError.range(
          idx,
          0,
          shape[i] - 1,
          'coordinate at dimension $i',
        );
      }
      offset += idx * strides[i];
    }
    data[offsetElements + offset] = value;
  }

  /// Internal helper to read the element at a flat index [flatIndex].
  /// Internal helper to read at raw physical storage index [rawOffset].
  @internal
  T getCellRaw(int rawOffset) => data[rawOffset];

  /// Internal helper to write at raw physical storage index [rawOffset].
  @internal
  void setCellRaw(int rawOffset, T value) {
    data[rawOffset] = value;
  }

  @internal
  T getCellFlat(int flatIndex) {
    if (isContiguous) {
      return data[offsetElements + flatIndex];
    }
    var offset = offsetElements;
    var rem = flatIndex;
    for (var i = shape.length - 1; i >= 0; i--) {
      offset += (rem % shape[i]) * strides[i];
      rem ~/= shape[i];
    }
    return data[offset];
  }

  /// Internal helper to write [value] to the element at a flat index [flatIndex].
  @internal
  void setCellFlat(int flatIndex, T value) {
    if (isContiguous) {
      data[offsetElements + flatIndex] = value;
      return;
    }
    var offset = offsetElements;
    var rem = flatIndex;
    for (var i = shape.length - 1; i >= 0; i--) {
      offset += (rem % shape[i]) * strides[i];
      rem ~/= shape[i];
    }
    data[offset] = value;
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
  void setByMask(NDArray<bool> mask, NDArray<T> values) {
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

    var valueIndex = 0;

    void walk(int dim, int currentOffset, int maskOffset) {
      if (dim == shape.length) {
        if (mask.getCellRaw(maskOffset)) {
          if (valueIndex >= values.size) {
            throw ArgumentError(
              'Source values array contains fewer elements than the mask targets',
            );
          }
          data[currentOffset] = values.getCellFlat(valueIndex++);
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
  void setByMaskScalar(NDArray<bool> mask, T value) {
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
          data[currentOffset] = value;
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
  void setIndicesScalar(NDArray<int> indices, T value, {int axis = 0}) {
    if (axis < 0 || axis >= shape.length) {
      throw RangeError.range(axis, 0, shape.length - 1, 'axis');
    }

    final sliceShape = List<int>.from(shape)..removeAt(axis);
    final sliceStrides = List<int>.from(strides)..removeAt(axis);

    for (var idx = 0; idx < indices.size; idx++) {
      final targetIdx = indices.getCellFlat(idx);
      if (targetIdx < 0 || targetIdx >= shape[axis]) {
        throw RangeError.range(
          targetIdx,
          0,
          shape[axis] - 1,
          'index entry at position $idx',
        );
      }

      void overwriteSlice(int dim, int currentOffset) {
        if (dim == sliceShape.length) {
          data[currentOffset] = value;
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
  void setIndices(NDArray<int> indices, NDArray values, {int axis = 0}) {
    if (axis < 0 || axis >= shape.length) {
      throw RangeError.range(axis, 0, shape.length - 1, 'axis');
    }

    final sliceShape = List<int>.from(shape)..removeAt(axis);
    final sliceStrides = List<int>.from(strides)..removeAt(axis);

    var valOffset = 0;

    for (var idx = 0; idx < indices.size; idx++) {
      final targetIdx = indices.getCellFlat(idx);
      if (targetIdx < 0 || targetIdx >= shape[axis]) {
        throw RangeError.range(
          targetIdx,
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
          data[currentOffset] = _coerceScalar(values.getCellFlat(valOffset++));
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
  T _coerceScalar(dynamic value) {
    if (value is NDArray && (value.shape.isEmpty || value.size == 1)) {
      value = value.getCellFlat(0);
    }
    if (value is T) return value;
    switch (dtype) {
      case DType.float64:
      case DType.float32:
      case DType.float16:
      case DType.bfloat16:
        if (value is num) return value.toDouble() as T;
      case DType.int64:
      case DType.int32:
      case DType.int16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
      case DType.uint8:
        if (value is num) return value.toInt() as T;
      case DType.complex128:
      case DType.complex64:
        if (value is num) return Complex(value.toDouble(), 0.0) as T;
      case DType.boolean:
        if (value is num) return (value != 0) as T;
    }
    return value as T;
  }

  /// Normalizes heterogeneous selection items into standard [Selector] objects.
  Selector _toSelector(dynamic item) {
    if (item is Selector) return item;
    if (item is int) return Index(item);
    if (item is List) {
      if (item.isEmpty) return Indices(const <int>[]);
      if (item.every((e) => e is int)) {
        return Indices(item.cast<int>());
      }
      if (item.every((e) => e is bool)) {
        final boolArr = NDArray<bool>.fromList(item.cast<bool>(), [
          item.length,
        ], DType.boolean);
        return Mask(BooleanMask(boolArr));
      }
      throw ArgumentError(
        "Selector lists must contain homogeneous integer coordinates or booleans, found: ${item.runtimeType}",
      );
    }
    if (item is NDArray) {
      if (item.dtype == DType.boolean) {
        return Mask(BooleanMask(item as NDArray<bool>));
      }
      if (item.dtype.isInteger) {
        final intList = item.toList().map((e) => e as int).toList();
        return Indices(intList);
      }
    }
    if (item is BooleanMask) return Mask(item);
    throw ArgumentError("Unsupported selector item type: ${item.runtimeType}");
  }

  /// Mutates multi-dimensional slices targeted by normalized [selectors].
  void _sliceAssign(List<Selector> selectors, dynamic value) {
    if (selectors.length > shape.length) {
      throw ArgumentError(
        "Too many selectors for array rank (${shape.length})",
      );
    }

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
        final maskMarker = ScratchArena.marker;
        final List<int> indices;
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
      if (value is NDArray) {
        final NDArray valArr;
        if (listEquals(value.shape, view.shape)) {
          valArr = value;
        } else {
          valArr = ops.broadcastTo(value, view.shape);
        }
        valArr.copy(out: view);
      } else {
        view.fill(_coerceScalar(value));
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
  /// - **`NDArray<bool>`**:
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
  dynamic operator [](dynamic spec) {
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
      return slice([_toSelector(spec)]);
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
      final selectors = spec
          .map((e) => _toSelector(e))
          .cast<Selector>()
          .toList();
      return slice(selectors);
    } else if (spec is NDArray && spec.dtype == DType.boolean) {
      final boolMask = spec as NDArray<bool>;
      var shapesMatch = boolMask.shape.length == shape.length;
      if (shapesMatch) {
        for (var i = 0; i < shape.length; i++) {
          if (boolMask.shape[i] != shape[i]) {
            shapesMatch = false;
            break;
          }
        }
      }
      if (shapesMatch) {
        return applyMask(boolMask);
      } else if (boolMask.shape.length == 1 && boolMask.shape[0] == shape[0]) {
        return slice([Mask(BooleanMask(boolMask))]);
      } else {
        throw ArgumentError(
          "Boolean mask shape must exactly match array shape",
        );
      }
    } else if (spec is NDArray && spec.dtype.isInteger) {
      var shapesMatch = spec.shape.length == shape.length;
      if (shapesMatch) {
        for (var i = 0; i < shape.length; i++) {
          if (spec.shape[i] != shape[i]) {
            shapesMatch = false;
            break;
          }
        }
      }
      if (shapesMatch) {
        throw ArgumentError(
          "Masking requires an NDArray of DType.boolean, not integers.",
        );
      } else {
        final intList = spec.toList().map((e) => e as int).toList();
        return take(intList);
      }
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
  /// - **`NDArray<bool>`**:
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
  void operator []=(dynamic spec, dynamic value) {
    if (isDisposed) {
      throw StateError(
        "Cannot access an array or view whose memory has been explicitly freed/disposed!",
      );
    }
    if (spec is int) {
      final indices = NDArray<int>.fromList([spec], [1], DType.int32);
      if (value is NDArray) {
        setIndices(indices, value);
      } else {
        setIndicesScalar(indices, _coerceScalar(value));
      }
    } else if (spec is Slice ||
        spec is Index ||
        spec is Indices ||
        spec is Mask ||
        spec is BooleanMask) {
      _sliceAssign([_toSelector(spec)], value);
    } else if (spec is List) {
      if (spec.isNotEmpty && spec.first is List) {
        final subList = spec.first as List;
        if (subList.every((e) => e is int)) {
          final intIndices = subList.cast<int>().toList();
          final indices = NDArray<int>.fromList(intIndices, [
            intIndices.length,
          ], DType.int32);
          if (value is NDArray) {
            setIndices(indices, value);
          } else {
            setIndicesScalar(indices, _coerceScalar(value));
          }
          return;
        }
      }
      if (spec.every((e) => e is int)) {
        if (shape.length == 1 && spec.length > 1) {
          final intIndices = spec.cast<int>();
          final indices = NDArray<int>.fromList(intIndices, [
            intIndices.length,
          ], DType.int32);
          if (value is NDArray) {
            setIndices(indices, value);
          } else {
            setIndicesScalar(indices, _coerceScalar(value));
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
        final selectors = spec
            .map((e) => _toSelector(e))
            .cast<Selector>()
            .toList();
        _sliceAssign(selectors, value);
      }
    } else if (spec is NDArray && spec.dtype == DType.boolean) {
      final boolMask = spec as NDArray<bool>;
      var shapesMatch = boolMask.shape.length == shape.length;
      if (shapesMatch) {
        for (var i = 0; i < shape.length; i++) {
          if (boolMask.shape[i] != shape[i]) {
            shapesMatch = false;
            break;
          }
        }
      }
      if (shapesMatch) {
        if (value is NDArray<T>) {
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
      var shapesMatch = spec.shape.length == shape.length;
      if (shapesMatch) {
        for (var i = 0; i < shape.length; i++) {
          if (spec.shape[i] != shape[i]) {
            shapesMatch = false;
            break;
          }
        }
      }
      if (shapesMatch) {
        throw ArgumentError(
          "Masking requires an NDArray of DType.boolean, not integers.",
        );
      } else {
        final intList = spec.toList().map((e) => e as int).toList();
        final indices = NDArray<int>.fromList(intList, [
          intList.length,
        ], DType.int32);
        if (value is NDArray) {
          setIndices(indices, value);
        } else {
          setIndicesScalar(indices, _coerceScalar(value));
        }
      }
    } else {
      throw ArgumentError(
        "Unsupported selector type for operator []: ${spec.runtimeType}",
      );
    }
  }

  NDArray _wrapScalar(dynamic value, List<int> targetShape) {
    if (value is Complex) {
      return NDArray<Complex128>.fromList(
        <Complex>[value],
        List.filled(targetShape.length, 1),
        DType.complex128,
      );
    } else if (value is int) {
      return NDArray<Int64>.fromList(
        <int>[value],
        List.filled(targetShape.length, 1),
        DType.int64,
      );
    } else if (value is double) {
      return NDArray<Float64>.fromList(
        <double>[value],
        List.filled(targetShape.length, 1),
        DType.float64,
      );
    } else if (value is bool) {
      return NDArray<bool>.fromList(
        <bool>[value],
        List.filled(targetShape.length, 1),
        DType.boolean,
      );
    } else {
      throw ArgumentError('Unsupported scalar type: ${value.runtimeType}');
    }
  }

  /// Element-wise addition with full broadcasting support.
  NDArray operator +(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.add(this, otherArr);
  }

  /// Element-wise subtraction with full broadcasting support.
  NDArray operator -(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.subtract(this, otherArr);
  }

  /// Element-wise multiplication with full broadcasting support.
  ///
  /// **Overflow behavior:**
  /// - **Integer arrays** (`int32`, `int64`, etc.) overflow silently wrapping around via standard two's complement.
  /// - **Floating-point arrays** (`float32`, `float64`) overflow silently to `double.infinity` or `double.negativeInfinity` per IEEE 754.
  NDArray operator *(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.multiply(this, otherArr);
  }

  /// Element-wise division with full broadcasting support.
  ///
  /// Always upcasts integer operands to [DType.float64] and performs floating-point division.
  ///
  /// **Division by Zero:**
  /// Division by zero is handled silently under IEEE 754 floating-point rules:
  /// - Dividing a non-zero value by zero results in `double.infinity` or `double.negativeInfinity`.
  /// - Dividing zero by zero results in `double.nan`.
  /// No exception is thrown.
  NDArray operator /(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.divide(this, otherArr);
  }

  /// Element-wise floor division with full broadcasting support.
  ///
  /// **Division by Zero:**
  /// - **Integer arrays**: It is an error if divisor contains any `0` elements.
  ///   This upfront safety check prevents a native C integer division by zero which would crash the entire Dart process.
  /// - **Floating-point arrays**: Returns `double.nan` silently without throwing exceptions.
  NDArray operator ~/(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.floor_divide(this as NDArray<num>, otherArr as NDArray<num>);
  }

  /// Element-wise remainder with full broadcasting support.
  ///
  /// **Division by Zero:**
  /// - **Integer divisor**: It is an error if divisor contains any `0` elements.
  ///   This upfront safety check prevents a native C integer division by zero which would crash the entire Dart process.
  /// - **Floating-point divisor**: Returns `double.nan` silently without throwing exceptions.
  NDArray operator %(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.remainder(this as NDArray<num>, otherArr as NDArray<num>);
  }

  /// Numerical negative, element-wise.
  NDArray operator -() {
    return ops.negative(this);
  }

  /// Element-wise bitwise AND with full broadcasting support.
  NDArray operator &(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.bitwise_and(this, otherArr);
  }

  /// Element-wise bitwise OR with full broadcasting support.
  NDArray operator |(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.bitwise_or(this, otherArr);
  }

  /// Element-wise bitwise XOR with full broadcasting support.
  NDArray operator ^(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.bitwise_xor(this, otherArr);
  }

  /// Element-wise bitwise NOT.
  NDArray operator ~() {
    return ops.invert(this);
  }

  /// Element-wise left shift with full broadcasting support.
  NDArray operator <<(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.left_shift(this, otherArr);
  }

  /// Element-wise right shift with full broadcasting support.
  NDArray operator >>(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.right_shift(this, otherArr);
  }

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
  NDArray<bool> operator >(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.greater(this, otherArr);
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
  NDArray<bool> operator <(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.less(this, otherArr);
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
  NDArray<bool> operator >=(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.greaterEqual(this, otherArr);
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
  NDArray<bool> operator <=(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.lessEqual(this, otherArr);
  }

  /// Element-wise equality comparison (`eq(other)`) with full broadcasting support.
  ///
  /// Returns a boolean [NDArray] where each element is `true` if the corresponding
  /// element in this array equals the element in [other], and `false` otherwise.
  ///
  /// Unlike the standard Dart operator `==` which defaults to object identity,
  /// the `==` operator on [NDArray] checks for structural equality of the
  /// arrays themselves (returning a single boolean). In contrast, [eq]
  /// performs element-wise value comparison and returns an [NDArray<bool>].
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
  NDArray<bool> eq(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.equal(this, otherArr);
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
        final maskMarker = ScratchArena.marker;
        final List<int> indices;
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
  NDArray<T> applyMask(NDArray<bool> mask) {
    if (shape.length == 1 &&
        mask.shape.length == 1 &&
        isContiguous &&
        mask.isContiguous) {
      if (mask.shape[0] != shape[0]) {
        throw ArgumentError(
          'Boolean mask shape ${mask.shape} must match target shape $shape',
        );
      }
      final size = shape[0];
      final count = native_count_mask(mask.pointer.cast(), size);
      final result = NDArray<T>.create([count], dtype);
      native_apply_mask(
        dtype.index,
        pointer.cast(),
        mask.pointer.cast(),
        result.pointer.cast(),
        size,
      );
      return result;
    }
    return slice([Mask(BooleanMask(mask))]);
  }

  /// Returns a flat Dart list containing a copy of the elements in this array,
  /// traversed in the logical order defined by its shape and strides.
  List<T> toList() {
    if (isDisposed) {
      throw StateError(
        'Cannot access an array or view whose memory has been explicitly freed/disposed!',
      );
    }
    final result = <T>[];
    _fillListRecursive(this, List<int>.filled(shape.length, 0), 0, result);
    return result;
  }

  void _fillListRecursive(
    NDArray<T> arr,
    List<int> indices,
    int dim,
    List<T> result,
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

    if (!_isExternallyOwned) {
      _finalizer.detach(this);
      malloc.free(_pointer);
    } else {
      if (_customFinalizerInstance != null) {
        _customFinalizerInstance.detach(this);
      }
      if (_customNativeFinalizer != null) {
        final freeFunc = _customNativeFinalizer
            .asFunction<void Function(ffi.Pointer<ffi.Void>)>();
        freeFunc(_pointer);
      }
    }
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! NDArray) return false;
    if (dtype != other.dtype) return false;
    if (!listEquals(shape, other.shape)) return false;

    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);

    // 1. High-speed direct C memcmp block byte check for C-contiguous same-layout arrays
    if (isContiguous && other.isContiguous) {
      final byteSize = totalSize * dtype.byteWidth;
      return custom_memcmp(pointer, other.pointer, byteSize) == 0;
    }

    // 2. Zero-allocation strided C comparison
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

  @override
  int get hashCode {
    var baseHash = Object.hash(dtype, Object.hashAll(shape));

    final int elementsHash;
    final marker = ScratchArena.marker;
    try {
      final cShape = ScratchArena.copyInts(shape);
      final cStrides = ScratchArena.copyInts(strides);
      switch (dtype) {
        case DType.float64 || DType.int64 || DType.uint64:
          elementsHash = s_hash_double(
            pointer.cast(),
            cStrides,
            cShape,
            shape.length,
            isContiguous ? 1 : 0,
          );
        case DType.float32 || DType.int32 || DType.uint32:
          elementsHash = s_hash_float(
            pointer.cast(),
            cStrides,
            cShape,
            shape.length,
            isContiguous ? 1 : 0,
          );
        case DType.float16 || DType.bfloat16 || DType.int16 || DType.uint16:
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
  final NDArray<bool> mask;

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

  Complex operator +(dynamic other) {
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

  Complex operator -(dynamic other) {
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

  Complex operator -() => Complex(-real, -imag);

  Complex operator *(dynamic other) {
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

  Complex operator /(dynamic other) {
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
  Complex pow(dynamic exponent) {
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
final class ComplexList<T extends Complex> extends ListBase<T> {
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
  T operator [](int index) {
    return Complex(_list[index * 2], _list[index * 2 + 1]) as T;
  }

  @override
  void operator []=(int index, T value) {
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
